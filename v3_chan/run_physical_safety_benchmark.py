from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from v3_chan.physical_safety_controllers import PHYSICAL_SAFETY_MODES
except ImportError:
    from physical_safety_controllers import PHYSICAL_SAFETY_MODES


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TASK_CHECKPOINT = (
    SCRIPT_DIR / "policies" / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
)
DEFAULT_MODES = ("none", "rmpflow", "cbf", "rmpflow_cbf")
STRICT_SINGLE_PICK_CONTRACT = "released_pick_and_place_with_no_human_collision_v1"
SUMMARY_SCHEMA = "physical_safety_paired_benchmark_summary_v2"
FEASIBILITY_THRESHOLD_SCHEMA = "physical_safety_feasibility_thresholds_v1"
FEASIBILITY_REPORT_SCHEMA = "physical_safety_feasibility_report_v1"
TASK_FAILURE_DIAGNOSTIC_SCHEMA = "task_cbf_failure_diagnostics_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate analytic physical-safety controllers on paired controlled "
            "encounters. The calibration/heldout role must be explicit, and the "
            "frozen task policy and scene seeds are identical for every controller."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("eval", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument(
        "--evaluation-split",
        choices=("calibration", "heldout"),
        required=True,
        help=(
            "Explicit evidence role. Only calibration evidence can unlock the "
            "physical-feasibility gate; heldout evidence is report-only."
        ),
    )
    parser.add_argument(
        "--feasibility-thresholds",
        default="",
        help=(
            "Pre-existing frozen JSON threshold contract. Omitting it writes a "
            "fail-closed NO_GO report."
        ),
    )
    parser.add_argument("--task-checkpoint", default=str(DEFAULT_TASK_CHECKPOINT))
    parser.add_argument("--controllers", default=",".join(DEFAULT_MODES))
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--experiment-tag", default="physical_safety_v1")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--task-reward-version",
        default="",
        help=(
            "Evaluation-only reward override forwarded to evaluate_rollout_policy.py. "
            "Use reward_v4_post_release_stability_hri_errp for the legacy BC checkpoint."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-log-every", type=int, default=25)
    parser.add_argument(
        "--encounter-timebase",
        choices=("recorded", "step"),
        default="recorded",
    )
    parser.add_argument("--encounter-playback-speed", type=float, default=1.0)
    parser.add_argument("--rmpflow-human-safety-margin-m", type=float, default=0.05)
    parser.add_argument("--cbf-safe-gap-m", type=float, default=0.05)
    parser.add_argument("--cbf-activation-gap-m", type=float, default=0.13)
    parser.add_argument("--cbf-gamma-per-s", type=float, default=8.0)
    parser.add_argument("--cbf-prediction-horizon-s", type=float, default=0.15)
    parser.add_argument("--cbf-max-prediction-buffer-m", type=float, default=0.08)
    parser.add_argument("--cbf-max-joint-speed-rad-s", type=float, default=2.0)
    parser.add_argument(
        "--strict-task-semantics",
        action="store_true",
        help=(
            "Use the opt-in event-driven task controller contract for every "
            "paired controller condition."
        ),
    )
    parser.add_argument(
        "--strict-place-xy-tolerance-m",
        type=float,
        default=0.04,
        help=(
            "Horizontal released-cube target tolerance forwarded to every "
            "strict paired evaluation condition."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    controllers = _parse_controllers(args.controllers)
    seeds = _parse_seeds(args.eval_seeds)
    manifest = Path(args.eval_manifest).expanduser().resolve()
    task_checkpoint = Path(args.task_checkpoint).expanduser().resolve()
    for path in (manifest, task_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_contract = _validate_manifest_split(manifest, args.evaluation_split)
    expected_episodes = int(manifest_contract["scenario_count"])
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if (
        not np.isfinite(args.encounter_playback_speed)
        or args.encounter_playback_speed <= 0.0
    ):
        raise ValueError("--encounter-playback-speed must be finite and positive")
    tag = args.experiment_tag.strip()
    if not tag or any(character in tag for character in "/\\"):
        raise ValueError("--experiment-tag must be one directory name")
    result_dir = SCRIPT_DIR / "eval_results" / "physical_safety" / tag

    if args.stage in ("eval", "all"):
        _evaluate(
            controllers=controllers,
            seeds=seeds,
            manifest=manifest,
            task_checkpoint=task_checkpoint,
            result_dir=result_dir,
            expected_episodes=expected_episodes,
            args=args,
        )
    if args.stage in ("summarize", "all"):
        _summarize(
            controllers,
            seeds,
            result_dir,
            evaluation_split=args.evaluation_split,
            expected_episodes=expected_episodes,
            thresholds_path=(
                Path(args.feasibility_thresholds).expanduser().resolve()
                if args.feasibility_thresholds
                else None
            ),
            manifest_contract=manifest_contract,
            strict_task_semantics=bool(args.strict_task_semantics),
            expected_max_steps=int(args.max_steps),
        )


def _evaluate(
    *,
    controllers: tuple[str, ...],
    seeds: tuple[int, ...],
    manifest: Path,
    task_checkpoint: Path,
    result_dir: Path,
    expected_episodes: int,
    args: argparse.Namespace,
) -> None:
    for controller in controllers:
        for seed in seeds:
            output_dir = result_dir / controller
            output_json = output_dir / f"seed_{seed}.json"
            output_csv = output_dir / f"seed_{seed}.csv"
            output_steps = output_dir / f"seed_{seed}_steps.csv"
            outputs = (output_json, output_csv, output_steps)
            if not args.force and all(path.exists() for path in outputs):
                _validate_result(
                    output_json,
                    expected_episodes,
                    expected_controller=controller,
                    require_episode_metrics=True,
                    expected_strict_task_semantics=bool(
                        args.strict_task_semantics
                    ),
                )
                print(
                    f"[PhysicalSafetyBenchmark] skip controller={controller} "
                    f"seed={seed}",
                    flush=True,
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.force:
                for path in outputs:
                    path.unlink(missing_ok=True)
            command = [
                str(PROJECT_DIR / "launch_isaac.sh"),
                str(SCRIPT_DIR / "evaluate_rollout_policy.py"),
                "--checkpoint",
                str(task_checkpoint),
                "--encounter-manifest",
                str(manifest),
                "--encounter-policy",
                "cycle",
                "--encounter-timebase",
                args.encounter_timebase,
                "--encounter-playback-speed",
                str(args.encounter_playback_speed),
                "--episodes",
                "0",
                "--max-steps",
                str(args.max_steps),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--mask-human-obs-for-policy",
                "--no-pseudo-errp",
                "--require-release-for-success",
                "--physical-safety-controller",
                controller,
                "--rmpflow-human-safety-margin-m",
                str(args.rmpflow_human_safety_margin_m),
                "--cbf-safe-gap-m",
                str(args.cbf_safe_gap_m),
                "--cbf-activation-gap-m",
                str(args.cbf_activation_gap_m),
                "--cbf-gamma-per-s",
                str(args.cbf_gamma_per_s),
                "--cbf-prediction-horizon-s",
                str(args.cbf_prediction_horizon_s),
                "--cbf-max-prediction-buffer-m",
                str(args.cbf_max_prediction_buffer_m),
                "--cbf-max-joint-speed-rad-s",
                str(args.cbf_max_joint_speed_rad_s),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-step-csv",
                str(output_steps),
                "--log-every",
                str(args.eval_log_every),
            ]
            if args.task_reward_version:
                command.extend(
                    ["--task-reward-version", str(args.task_reward_version)]
                )
            if args.strict_task_semantics:
                command.append("--strict-task-semantics")
                command.extend(
                    [
                        "--strict-place-xy-tolerance-m",
                        str(args.strict_place_xy_tolerance_m),
                    ]
                )
            print(
                f"[PhysicalSafetyBenchmark] eval controller={controller} "
                f"seed={seed} episodes={expected_episodes}",
                flush=True,
            )
            environment = os.environ.copy()
            environment["ISAAC_SKIP_VR_WAIT"] = "1"
            subprocess.run(
                command,
                cwd=PROJECT_DIR,
                env=environment,
                check=True,
            )
            for path in outputs:
                if not path.exists():
                    raise RuntimeError(f"Evaluation did not produce {path}")
            _validate_result(
                output_json,
                expected_episodes,
                expected_controller=controller,
                require_episode_metrics=True,
                expected_strict_task_semantics=bool(
                    args.strict_task_semantics
                ),
            )


def _summarize(
    controllers: tuple[str, ...],
    seeds: tuple[int, ...],
    result_dir: Path,
    *,
    evaluation_split: str,
    expected_episodes: int | None = None,
    thresholds_path: Path | None = None,
    manifest_contract: dict[str, Any] | None = None,
    strict_task_semantics: bool | None = None,
    expected_max_steps: int | None = None,
) -> dict[str, Any]:
    if evaluation_split not in {"calibration", "heldout"}:
        raise ValueError("evaluation_split must be calibration or heldout")
    episode_rows: list[dict[str, Any]] = []
    strict_config_snapshots: list[dict[str, Any]] = []
    paired_config_snapshots: list[tuple[Path, dict[str, Any]]] = []
    for controller in controllers:
        for seed in seeds:
            path = result_dir / controller / f"seed_{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            result_config = dict(payload.get("config", {}))
            paired_config_snapshots.append(
                (
                    path,
                    {
                        key: value
                        for key, value in result_config.items()
                        if key != "physical_safety_controller"
                    },
                )
            )
            if strict_task_semantics:
                strict_config = payload.get("config", {}).get(
                    "strict_task_semantics_config"
                )
                if not isinstance(strict_config, dict) or not strict_config:
                    raise RuntimeError(
                        f"Strict task semantics config is missing in {path}"
                    )
                strict_config_snapshots.append(dict(strict_config))
            episode_payloads = payload.get("episodes", ())
            required_count = (
                int(expected_episodes)
                if expected_episodes is not None
                else len(episode_payloads)
            )
            _validate_result(
                path,
                required_count,
                expected_controller=controller,
                require_episode_metrics=True,
                expected_strict_task_semantics=strict_task_semantics,
            )
            for episode in episode_payloads:
                steps = _required_integer(episode, "steps", source=path, minimum=1)
                physics_dt_s = _required_finite(
                    episode,
                    "physics_dt_s",
                    source=path,
                    minimum=0.0,
                    minimum_exclusive=True,
                )
                collision_steps = _required_integer(
                    episode,
                    "collision_steps",
                    source=path,
                    minimum=0,
                    maximum=steps,
                )
                near_steps = _required_integer(
                    episode,
                    "near_steps",
                    source=path,
                    minimum=0,
                    maximum=steps,
                )
                success = _required_boolean(episode, "success", source=path)
                terminated = _required_boolean(
                    episode, "terminated", source=path
                )
                truncated = _required_boolean(
                    episode, "truncated", source=path
                )
                collision = _required_boolean(episode, "collision", source=path)
                if collision != (collision_steps > 0):
                    raise RuntimeError(f"collision/collision_steps mismatch in {path}")
                episode_rows.append(
                    {
                        "controller": controller,
                        "eval_seed": seed,
                        "episode": _required_integer(
                            episode, "episode", source=path, minimum=0
                        ),
                        "encounter_id": str(episode.get("encounter_id", "")),
                        "scene_layout_id": str(episode.get("scene_layout_id", "")),
                        "target_severity": str(
                            episode.get("encounter_target_severity", "")
                        ),
                        "success": int(success),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "grasp": int(
                            _required_boolean(episode, "grasped_any", source=path)
                        ),
                        "controller_lift_phase_reached": int(
                            _required_boolean(
                                episode,
                                "controller_lift_phase_reached",
                                source=path,
                            )
                        ),
                        "controller_place_phase_reached": int(
                            _required_boolean(
                                episode,
                                "controller_place_phase_reached",
                                source=path,
                            )
                        ),
                        "controller_release_phase_reached": int(
                            _required_boolean(
                                episode,
                                "controller_release_phase_reached",
                                source=path,
                            )
                        ),
                        "place_tolerance_reached": int(
                            _required_boolean(
                                episode,
                                "cube_entered_target_tolerance",
                                source=path,
                            )
                        ),
                        "released_after_grasp": int(
                            _required_boolean(
                                episode, "released_after_grasp", source=path
                            )
                        ),
                        "task_terminal_reason": str(
                            episode.get("task_terminal_reason", "")
                        ),
                        "task_failure_diagnostic_schema": str(
                            episode.get("task_failure_diagnostic_schema", "")
                        ),
                        "strict_task_semantics_enabled": int(
                            bool(
                                episode.get(
                                    "strict_task_semantics_enabled", False
                                )
                            )
                        ),
                        "first_grasp_step": int(
                            episode.get("first_grasp_step", -1)
                        ),
                        "first_grasp_loss_step": int(
                            episode.get("first_grasp_loss_step", -1)
                        ),
                        "first_target_entry_step": int(
                            episode.get("first_target_entry_step", -1)
                        ),
                        "first_release_step": int(
                            episode.get("first_release_step", -1)
                        ),
                        "first_success_step": int(
                            episode.get("first_success_step", -1)
                        ),
                        "grasp_acquisition_count": int(
                            episode.get("grasp_acquisition_count", 0)
                        ),
                        "grasp_loss_count": int(
                            episode.get("grasp_loss_count", 0)
                        ),
                        "release_command_count": int(
                            episode.get("release_command_count", 0)
                        ),
                        "task_phase_paused_steps": int(
                            episode.get("task_phase_paused_steps", 0)
                        ),
                        "task_phase_reentry_count": int(
                            episode.get("task_phase_reentry_count", 0)
                        ),
                        "max_consecutive_intervention_steps": int(
                            episode.get(
                                "max_consecutive_physical_safety_intervention_steps",
                                0,
                            )
                        ),
                        "intervention_steps_before_grasp": int(
                            episode.get(
                                "physical_safety_intervention_steps_before_grasp",
                                0,
                            )
                        ),
                        "intervention_steps_during_transport": int(
                            episode.get(
                                "physical_safety_intervention_steps_during_transport",
                                0,
                            )
                        ),
                        "intervention_steps_during_place": int(
                            episode.get(
                                "physical_safety_intervention_steps_during_place",
                                0,
                            )
                        ),
                        "collision_episode": int(collision),
                        "human_collision_incidence": int(collision),
                        "collision_free_success": int(success and not collision),
                        "steps": steps,
                        "completion_time_s": _required_finite(
                            episode,
                            "completion_time_s",
                            source=path,
                            minimum=0.0,
                        ),
                        "total_reward": _required_finite(
                            episode, "total_reward", source=path
                        ),
                        "collision_steps": collision_steps,
                        "collision_duration_s": _required_finite(
                            episode,
                            "collision_duration_s",
                            source=path,
                            minimum=0.0,
                        ),
                        "collision_max_consecutive_duration_s": _required_finite(
                            episode,
                            "collision_max_consecutive_duration_s",
                            source=path,
                            minimum=0.0,
                        ),
                        "collision_event_count": _required_integer(
                            episode,
                            "collision_event_count",
                            source=path,
                            minimum=0,
                            maximum=collision_steps,
                        ),
                        "collision_rate": _required_finite(
                            episode,
                            "collision_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "near_steps": near_steps,
                        "near_human_duration_s": _required_finite(
                            episode,
                            "near_human_duration_s",
                            source=path,
                            minimum=0.0,
                        ),
                        "near_miss_rate": _required_finite(
                            episode,
                            "near_miss_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "near_rate": _required_finite(
                            episode,
                            "near_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "gate_activation_rate": _required_finite(
                            episode,
                            "gate_activation_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "min_surface_gap_m": _required_finite(
                            episode, "min_surface_gap", source=path
                        ),
                        "minimum_ttc_s": _required_finite(
                            episode,
                            "minimum_ttc_s",
                            source=path,
                            minimum=0.0,
                        ),
                        "minimum_ttc_valid": int(
                            _required_boolean(episode, "minimum_ttc_valid", source=path)
                        ),
                        "physical_active_rate": _required_finite(
                            episode,
                            "physical_safety_active_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "physical_intervention_rate": _required_finite(
                            episode,
                            "physical_safety_intervention_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "mean_intervention_norm_radps": _required_finite(
                            episode,
                            "mean_physical_safety_intervention_norm_radps",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_intervention_norm_radps": _required_finite(
                            episode,
                            "max_physical_safety_intervention_norm_radps",
                            source=path,
                            minimum=0.0,
                        ),
                        "feasible_rate": _required_finite(
                            episode,
                            "physical_safety_feasible_rate",
                            source=path,
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "mean_slack_mps": _required_finite(
                            episode,
                            "mean_physical_safety_slack_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_slack_mps": _required_finite(
                            episode,
                            "max_physical_safety_slack_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "mean_constraint_violation_before_mps": _required_finite(
                            episode,
                            "mean_physical_safety_constraint_violation_before_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_constraint_violation_before_mps": _required_finite(
                            episode,
                            "max_physical_safety_constraint_violation_before_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "mean_constraint_violation_after_mps": _required_finite(
                            episode,
                            "mean_physical_safety_constraint_violation_after_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_constraint_violation_after_mps": _required_finite(
                            episode,
                            "max_physical_safety_constraint_violation_after_mps",
                            source=path,
                            minimum=0.0,
                        ),
                        "mean_solve_time_ms": _required_finite(
                            episode,
                            "mean_physical_safety_solve_time_ms",
                            source=path,
                            minimum=0.0,
                        ),
                        "ee_path_length_m": _required_finite(
                            episode,
                            "ee_path_length_m",
                            source=path,
                            minimum=0.0,
                        ),
                        "rms_ee_acceleration_mps2": _required_finite(
                            episode,
                            "rms_ee_acceleration_mps2",
                            source=path,
                            minimum=0.0,
                        ),
                        "p95_ee_jerk_mps3": _required_finite(
                            episode,
                            "p95_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                        "rms_ee_jerk_mps3": _required_finite(
                            episode,
                            "rms_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_ee_jerk_mps3": _required_finite(
                            episode,
                            "max_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                        "integrated_squared_ee_jerk_m2ps5": _required_finite(
                            episode,
                            "integrated_squared_ee_jerk_m2ps5",
                            source=path,
                            minimum=0.0,
                        ),
                        "rms_gate_ee_acceleration_mps2": _required_finite(
                            episode,
                            "rms_gate_ee_acceleration_mps2",
                            source=path,
                            minimum=0.0,
                        ),
                        "p95_gate_ee_jerk_mps3": _required_finite(
                            episode,
                            "p95_gate_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                        "rms_gate_ee_jerk_mps3": _required_finite(
                            episode,
                            "rms_gate_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                        "max_gate_ee_jerk_mps3": _required_finite(
                            episode,
                            "max_gate_ee_jerk_mps3",
                            source=path,
                            minimum=0.0,
                        ),
                    }
                )
                materialized = episode_rows[-1]
                if not math.isclose(
                    materialized["completion_time_s"],
                    steps * physics_dt_s,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(f"completion duration drift in {path}")
                if not math.isclose(
                    materialized["collision_duration_s"],
                    collision_steps * physics_dt_s,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(f"collision duration drift in {path}")
                if not math.isclose(
                    materialized["near_human_duration_s"],
                    near_steps * physics_dt_s,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(f"near-human duration drift in {path}")
    if strict_config_snapshots and any(
        snapshot != strict_config_snapshots[0]
        for snapshot in strict_config_snapshots[1:]
    ):
        raise RuntimeError(
            "Strict task semantics config differs across paired controller runs"
        )
    if paired_config_snapshots:
        reference_path, reference_config = paired_config_snapshots[0]
        for candidate_path, candidate_config in paired_config_snapshots[1:]:
            if candidate_config != reference_config:
                raise RuntimeError(
                    "Non-controller evaluation config differs across paired runs: "
                    f"reference={reference_path} candidate={candidate_path}"
                )
        if expected_max_steps is not None and int(
            reference_config.get("max_steps", -1)
        ) != int(expected_max_steps):
            raise RuntimeError(
                "Evaluation max_steps differs from the requested paired contract: "
                f"artifact={reference_config.get('max_steps')!r} "
                f"requested={expected_max_steps!r}"
            )
        paired_config_sha256 = hashlib.sha256(
            json.dumps(
                reference_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    else:
        reference_config = {}
        paired_config_sha256 = ""
    _validate_pairing(episode_rows, controllers)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "episode_results.csv", episode_rows)

    metrics = (
        "success",
        "terminated",
        "truncated",
        "grasp",
        "controller_lift_phase_reached",
        "controller_place_phase_reached",
        "controller_release_phase_reached",
        "place_tolerance_reached",
        "released_after_grasp",
        "collision_episode",
        "human_collision_incidence",
        "collision_free_success",
        "task_phase_paused_steps",
        "task_phase_reentry_count",
        "max_consecutive_intervention_steps",
        "intervention_steps_before_grasp",
        "intervention_steps_during_transport",
        "intervention_steps_during_place",
        "steps",
        "completion_time_s",
        "total_reward",
        "collision_steps",
        "collision_duration_s",
        "collision_max_consecutive_duration_s",
        "collision_event_count",
        "collision_rate",
        "near_steps",
        "near_human_duration_s",
        "near_miss_rate",
        "near_rate",
        "gate_activation_rate",
        "min_surface_gap_m",
        "minimum_ttc_s",
        "minimum_ttc_valid",
        "physical_active_rate",
        "physical_intervention_rate",
        "mean_intervention_norm_radps",
        "max_intervention_norm_radps",
        "feasible_rate",
        "mean_slack_mps",
        "max_slack_mps",
        "mean_constraint_violation_before_mps",
        "max_constraint_violation_before_mps",
        "mean_constraint_violation_after_mps",
        "max_constraint_violation_after_mps",
        "mean_solve_time_ms",
        "ee_path_length_m",
        "rms_ee_acceleration_mps2",
        "p95_ee_jerk_mps3",
        "rms_ee_jerk_mps3",
        "max_ee_jerk_mps3",
        "integrated_squared_ee_jerk_m2ps5",
        "rms_gate_ee_acceleration_mps2",
        "p95_gate_ee_jerk_mps3",
        "rms_gate_ee_jerk_mps3",
        "max_gate_ee_jerk_mps3",
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "evaluation_contract": STRICT_SINGLE_PICK_CONTRACT,
        "evaluation_split": evaluation_split,
        "evidence_role": (
            "open_calibration" if evaluation_split == "calibration" else "heldout"
        ),
        "threshold_selection_permitted": evaluation_split == "calibration",
        "strict_task_semantics": strict_task_semantics,
        "strict_task_semantics_config": (
            strict_config_snapshots[0] if strict_config_snapshots else {}
        ),
        "paired_evaluation_config": reference_config,
        "paired_evaluation_config_sha256": paired_config_sha256,
        "paired_evaluation_controller_specific_keys": [
            "physical_safety_controller"
        ],
        "task_failure_diagnostic_schema": (
            TASK_FAILURE_DIAGNOSTIC_SCHEMA
            if strict_task_semantics
            else "legacy_or_unspecified"
        ),
        "manifest_contract": dict(manifest_contract or {}),
        "success_definition": "released cube inside the target tolerance",
        "collision_free_success_definition": (
            "strict released-pick success and zero human-collision steps"
        ),
        "task_stage_metric_definitions": {
            "grasp": "has_grasped_cube was observed",
            "controller_lift_phase_reached": "controller event >= 4 was observed",
            "controller_place_phase_reached": "controller event >= 5 was observed",
            "controller_release_phase_reached": "controller event >= 7 was observed",
            "place_tolerance_reached": "cube entered the configured target tolerance",
            "released_after_grasp": "a grasped-to-not-grasped transition was observed",
        },
        "collision_signal_scope": {
            "human": "measured",
            "static": "unavailable_not_inferred",
            "self": "unavailable_not_inferred",
        },
        "pairing_unit": "eval_seed + encounter_id + scene_layout_id",
        "pairing": {
            "exact": True,
            "duplicate_keys_rejected": True,
            "pair_count_per_controller": int(len(episode_rows) / len(controllers)),
        },
        "reference_controller": "none" if "none" in controllers else "",
        "controllers": {},
    }
    for controller in controllers:
        selected = [row for row in episode_rows if row["controller"] == controller]
        valid_ttc = [
            float(row["minimum_ttc_s"])
            for row in selected
            if bool(row["minimum_ttc_valid"])
        ]
        summary["controllers"][controller] = {
            "episodes": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in metrics
            },
            "minimum_ttc_valid_episode_count": len(valid_ttc),
            "minimum_ttc_s_valid_only_mean": (
                float(np.mean(valid_ttc)) if valid_ttc else None
            ),
        }
        diagnostic_rows = [
            row
            for row in selected
            if row.get("task_failure_diagnostic_schema")
            == TASK_FAILURE_DIAGNOSTIC_SCHEMA
        ]
        if len(diagnostic_rows) == len(selected):
            terminal_reason_counts: dict[str, int] = {}
            for row in diagnostic_rows:
                reason = str(row.get("task_terminal_reason", "") or "unspecified")
                terminal_reason_counts[reason] = (
                    terminal_reason_counts.get(reason, 0) + 1
                )
            valid_first_steps: dict[str, dict[str, float | int | None]] = {}
            for key in (
                "first_grasp_step",
                "first_target_entry_step",
                "first_release_step",
                "first_success_step",
            ):
                values = [
                    int(row[key])
                    for row in diagnostic_rows
                    if int(row[key]) >= 0
                ]
                valid_first_steps[key] = {
                    "valid_episode_count": len(values),
                    "mean_step_valid_only": (
                        float(np.mean(values)) if values else None
                    ),
                }
            summary["controllers"][controller]["task_failure_diagnostics"] = {
                "schema_version": TASK_FAILURE_DIAGNOSTIC_SCHEMA,
                "terminal_reason_counts": terminal_reason_counts,
                "first_event_timing": valid_first_steps,
                "mean_grasp_acquisition_count": float(
                    np.mean(
                        [row["grasp_acquisition_count"] for row in diagnostic_rows]
                    )
                ),
                "mean_grasp_loss_count": float(
                    np.mean([row["grasp_loss_count"] for row in diagnostic_rows])
                ),
                "mean_release_command_count": float(
                    np.mean(
                        [row["release_command_count"] for row in diagnostic_rows]
                    )
                ),
                "mean_task_phase_paused_steps": float(
                    np.mean(
                        [row["task_phase_paused_steps"] for row in diagnostic_rows]
                    )
                ),
                "mean_task_phase_reentry_count": float(
                    np.mean(
                        [row["task_phase_reentry_count"] for row in diagnostic_rows]
                    )
                ),
                "mean_max_consecutive_intervention_steps": float(
                    np.mean(
                        [
                            row["max_consecutive_intervention_steps"]
                            for row in diagnostic_rows
                        ]
                    )
                ),
                "mean_intervention_steps_before_grasp": float(
                    np.mean(
                        [
                            row["intervention_steps_before_grasp"]
                            for row in diagnostic_rows
                        ]
                    )
                ),
                "mean_intervention_steps_during_transport": float(
                    np.mean(
                        [
                            row["intervention_steps_during_transport"]
                            for row in diagnostic_rows
                        ]
                    )
                ),
                "mean_intervention_steps_during_place": float(
                    np.mean(
                        [
                            row["intervention_steps_during_place"]
                            for row in diagnostic_rows
                        ]
                    )
                ),
            }
    if "none" in controllers:
        reference = {
            _pairing_key(row): row
            for row in episode_rows
            if row["controller"] == "none"
        }
        for controller in controllers:
            if controller == "none":
                continue
            selected = [row for row in episode_rows if row["controller"] == controller]
            summary["controllers"][controller]["paired_delta_vs_none"] = {
                metric: float(
                    np.mean(
                        [
                            row[metric] - reference[_pairing_key(row)][metric]
                            for row in selected
                        ]
                    )
                )
                for metric in metrics
            }
            if all(
                row.get("task_failure_diagnostic_schema")
                == TASK_FAILURE_DIAGNOSTIC_SCHEMA
                and reference[_pairing_key(row)].get(
                    "task_failure_diagnostic_schema"
                )
                == TASK_FAILURE_DIAGNOSTIC_SCHEMA
                for row in selected
            ):
                regressions = [
                    row
                    for row in selected
                    if bool(reference[_pairing_key(row)]["success"])
                    and not bool(row["success"])
                ]
                regression_reasons: dict[str, int] = {}
                for row in regressions:
                    reason = str(
                        row.get("task_terminal_reason", "") or "unspecified"
                    )
                    regression_reasons[reason] = (
                        regression_reasons.get(reason, 0) + 1
                    )
                summary["controllers"][controller][
                    "paired_task_failure_diagnostics_vs_none"
                ] = {
                    "schema_version": TASK_FAILURE_DIAGNOSTIC_SCHEMA,
                    "none_success_candidate_failure_count": len(regressions),
                    "candidate_failure_reason_counts": regression_reasons,
                }
    summary["physical_feasibility"] = _load_and_decide_physical_feasibility(
        summary,
        thresholds_path=thresholds_path,
    )
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (result_dir / "physical_feasibility.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary["physical_feasibility"], handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[PhysicalSafetyBenchmark] saved {result_dir}", flush=True)
    return summary


def _required_boolean(row: dict[str, Any], key: str, *, source: Path) -> bool:
    if key not in row or not isinstance(row[key], bool):
        raise RuntimeError(f"{key} must be boolean in {source}")
    return bool(row[key])


def _required_integer(
    row: dict[str, Any],
    key: str,
    *,
    source: Path,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{key} must be an integer in {source}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{key} is below {minimum} in {source}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{key} is above {maximum} in {source}")
    return result


def _required_finite(
    row: dict[str, Any],
    key: str,
    *,
    source: Path,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if key not in row or isinstance(row[key], bool):
        raise RuntimeError(f"{key} must be numeric in {source}")
    try:
        result = float(row[key])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{key} must be numeric in {source}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{key} must be finite in {source}")
    if minimum is not None:
        invalid_minimum = result <= minimum if minimum_exclusive else result < minimum
        if invalid_minimum:
            relation = "greater than" if minimum_exclusive else "at least"
            raise RuntimeError(f"{key} must be {relation} {minimum} in {source}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{key} must be at most {maximum} in {source}")
    return result


def _fail_closed_feasibility_report(
    *reason_codes: str,
    status: str = "NOT_EVALUATED",
) -> dict[str, Any]:
    return {
        "schema_version": FEASIBILITY_REPORT_SCHEMA,
        "status": status,
        "decision": "NO_GO",
        "passed": False,
        "fail_closed": True,
        "unlocks_constrained_ppo": False,
        "reason_codes": list(reason_codes) or ["unspecified_gate_failure"],
        "checks": {},
    }


def decide_physical_feasibility(
    summary: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Apply a pre-frozen gate without estimating any threshold from results."""

    contract_errors: list[str] = []
    if thresholds.get("schema_version") != FEASIBILITY_THRESHOLD_SCHEMA:
        contract_errors.append("threshold_schema_mismatch")
    protocol_id = thresholds.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        contract_errors.append("threshold_protocol_id_missing")
    if thresholds.get("frozen_before_evaluation") is not True:
        contract_errors.append("thresholds_not_declared_frozen")
    evaluation_split = summary.get("evaluation_split")
    if evaluation_split != "calibration":
        contract_errors.append("heldout_cannot_unlock_calibration_gate")
    if thresholds.get("evaluation_split") != evaluation_split:
        contract_errors.append("threshold_evaluation_split_mismatch")
    pairing = summary.get("pairing")
    if not isinstance(pairing, dict) or pairing.get("exact") is not True:
        contract_errors.append("exact_pairing_not_verified")

    reference_name = thresholds.get("reference_controller")
    candidate_name = thresholds.get("candidate_controller")
    if not isinstance(reference_name, str) or not reference_name:
        contract_errors.append("reference_controller_missing")
    if not isinstance(candidate_name, str) or not candidate_name:
        contract_errors.append("candidate_controller_missing")
    if reference_name == candidate_name:
        contract_errors.append("reference_and_candidate_must_differ")
    controllers = summary.get("controllers")
    if not isinstance(controllers, dict):
        contract_errors.append("controller_summary_missing")
        controllers = {}
    if reference_name not in controllers:
        contract_errors.append("reference_controller_evidence_missing")
    if candidate_name not in controllers:
        contract_errors.append("candidate_controller_evidence_missing")

    threshold_specs = {
        "minimum_collision_episode_rate_reduction": (0.0, 1.0, True),
        "maximum_success_rate_drop": (0.0, 1.0, False),
        "minimum_candidate_success_rate": (0.0, 1.0, True),
    }
    threshold_values: dict[str, float] = {}
    for key, (minimum, maximum, minimum_exclusive) in threshold_specs.items():
        value = thresholds.get(key)
        if isinstance(value, bool):
            contract_errors.append(f"invalid_{key}")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            contract_errors.append(f"invalid_{key}")
            continue
        if (
            not math.isfinite(numeric)
            or numeric > maximum
            or (numeric <= minimum if minimum_exclusive else numeric < minimum)
        ):
            contract_errors.append(f"invalid_{key}")
            continue
        threshold_values[key] = numeric

    if contract_errors:
        report = _fail_closed_feasibility_report(
            *dict.fromkeys(contract_errors), status="INVALID_CONTRACT"
        )
        report["protocol_id"] = protocol_id if isinstance(protocol_id, str) else ""
        return report

    reference = controllers[str(reference_name)]
    candidate = controllers[str(candidate_name)]
    try:
        reference_collision = float(reference["collision_episode"])
        candidate_collision = float(candidate["collision_episode"])
        reference_success = float(reference["success"])
        candidate_success = float(candidate["success"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        report = _fail_closed_feasibility_report(
            "required_gate_metric_missing", status="INVALID_EVIDENCE"
        )
        report["protocol_id"] = str(protocol_id)
        return report
    gate_values = (
        reference_collision,
        candidate_collision,
        reference_success,
        candidate_success,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in gate_values):
        report = _fail_closed_feasibility_report(
            "required_gate_metric_nonfinite_or_out_of_range",
            status="INVALID_EVIDENCE",
        )
        report["protocol_id"] = str(protocol_id)
        return report

    collision_reduction = reference_collision - candidate_collision
    success_drop = reference_success - candidate_success
    checks = {
        "human_collision_episode_rate_reduction": {
            "actual": float(collision_reduction),
            "operator": ">=",
            "threshold": threshold_values["minimum_collision_episode_rate_reduction"],
            "passed": bool(
                collision_reduction
                >= threshold_values["minimum_collision_episode_rate_reduction"]
            ),
        },
        "task_success_rate_drop": {
            "actual": float(success_drop),
            "operator": "<=",
            "threshold": threshold_values["maximum_success_rate_drop"],
            "passed": bool(
                success_drop <= threshold_values["maximum_success_rate_drop"]
            ),
        },
        "candidate_task_success_rate": {
            "actual": float(candidate_success),
            "operator": ">=",
            "threshold": threshold_values["minimum_candidate_success_rate"],
            "passed": bool(
                candidate_success >= threshold_values["minimum_candidate_success_rate"]
            ),
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema_version": FEASIBILITY_REPORT_SCHEMA,
        "status": "EVALUATED",
        "decision": "GO" if passed else "NO_GO",
        "passed": bool(passed),
        "fail_closed": True,
        "unlocks_constrained_ppo": bool(passed),
        "protocol_id": str(protocol_id),
        "evaluation_split": str(evaluation_split),
        "reference_controller": str(reference_name),
        "candidate_controller": str(candidate_name),
        "thresholds": dict(threshold_values),
        "measured": {
            "reference_human_collision_episode_rate": reference_collision,
            "candidate_human_collision_episode_rate": candidate_collision,
            "reference_task_success_rate": reference_success,
            "candidate_task_success_rate": candidate_success,
        },
        "reason_codes": (
            [] if passed else [f"failed_{name}" for name in failed_checks]
        ),
        "checks": checks,
    }


def _load_and_decide_physical_feasibility(
    summary: dict[str, Any], *, thresholds_path: Path | None
) -> dict[str, Any]:
    if thresholds_path is None:
        return _fail_closed_feasibility_report("frozen_thresholds_not_supplied")
    resolved = thresholds_path.expanduser().resolve()
    artifact = {"path": str(resolved), "sha256": ""}
    try:
        payload_bytes = resolved.read_bytes()
        artifact["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
        thresholds = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        report = _fail_closed_feasibility_report(
            f"threshold_artifact_unreadable:{type(error).__name__}",
            status="INVALID_CONTRACT",
        )
        report["threshold_artifact"] = artifact
        return report
    if not isinstance(thresholds, dict):
        report = _fail_closed_feasibility_report(
            "threshold_artifact_not_object", status="INVALID_CONTRACT"
        )
        report["threshold_artifact"] = artifact
        return report
    report = decide_physical_feasibility(summary, thresholds)
    report["threshold_artifact"] = artifact
    return report


def _parse_controllers(value: str) -> tuple[str, ...]:
    controllers = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(controllers)) != len(controllers):
        raise ValueError("--controllers must not contain duplicates")
    unknown = sorted(set(controllers) - set(PHYSICAL_SAFETY_MODES))
    if not controllers or unknown:
        raise ValueError(
            f"Unknown physical safety controllers {unknown}; "
            f"supported={PHYSICAL_SAFETY_MODES}"
        )
    return controllers


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--eval-seeds must not contain duplicates")
    return seeds


def _manifest_episode_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios", ())
    if not scenarios:
        raise ValueError(f"Encounter manifest has no scenarios: {path}")
    return len(scenarios)


def _validate_manifest_split(path: Path, evaluation_split: str) -> dict[str, Any]:
    """Reject a locked manifest before Isaac starts when calibration is requested."""

    resolved = path.expanduser().resolve()
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read encounter manifest {resolved}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Encounter manifest must be an object: {resolved}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"Encounter manifest has no scenarios: {resolved}")
    split_metadata = payload.get("split_metadata")
    if not isinstance(split_metadata, dict):
        raise RuntimeError(f"Encounter manifest split metadata is missing: {resolved}")
    native_role = str(split_metadata.get("role", "")).strip().lower()
    permitted_roles = {
        "calibration": {"calibration", "development"},
        "heldout": {"eval", "evaluation", "heldout", "test", "locked_test"},
    }
    if evaluation_split not in permitted_roles:
        raise ValueError("evaluation_split must be calibration or heldout")
    if native_role not in permitted_roles[evaluation_split]:
        raise RuntimeError(
            "Manifest/evaluation split mismatch: "
            f"requested={evaluation_split!r} native_role={native_role!r} "
            f"path={resolved}"
        )
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "schema_version": str(payload.get("schema_version", "")),
        "native_role": native_role,
        "evaluation_split": evaluation_split,
        "scenario_count": len(scenarios),
        "locked_heldout_rejected_for_calibration": True,
    }


def _validate_result(
    path: Path,
    expected_episodes: int,
    *,
    expected_controller: str | None = None,
    require_episode_metrics: bool = False,
    expected_strict_task_semantics: bool | None = None,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    actual = len(payload.get("episodes", ()))
    if actual != expected_episodes:
        raise RuntimeError(
            f"Incomplete evaluation {path}: episodes={actual}, "
            f"expected={expected_episodes}"
        )
    config = dict(payload.get("config", {}))
    if expected_controller is not None:
        actual_controller = str(config.get("physical_safety_controller", ""))
        if actual_controller != expected_controller:
            raise RuntimeError(
                f"Controller mismatch in {path}: actual={actual_controller!r}, "
                f"expected={expected_controller!r}"
            )
    if expected_strict_task_semantics is not None:
        actual_strict = bool(config.get("strict_task_semantics", False))
        if actual_strict != bool(expected_strict_task_semantics):
            raise RuntimeError(
                "Strict task semantics mismatch in "
                f"{path}: actual={actual_strict!r}, "
                f"expected={bool(expected_strict_task_semantics)!r}"
            )
        if actual_strict and config.get("task_failure_diagnostic_schema") != (
            TASK_FAILURE_DIAGNOSTIC_SCHEMA
        ):
            raise RuntimeError(
                f"Strict task diagnostic schema is missing in {path}"
            )
    if not bool(config.get("mask_human_obs_for_policy", False)):
        raise RuntimeError(f"Task policy human masking is disabled in {path}")
    if bool(config.get("pseudo_errp_enabled", True)):
        raise RuntimeError(f"Pseudo-ErrP must be disabled in {path}")
    if not bool(config.get("require_release_for_success", False)):
        raise RuntimeError(f"Strict released-pick success is disabled in {path}")
    if str(config.get("encounter_timebase", "")) != "recorded":
        raise RuntimeError(f"Encounter timebase is not recorded in {path}")
    invalid_restoration = [
        str(row.get("encounter_id", row.get("episode", "unknown")))
        for row in payload.get("episodes", ())
        if row.get("restoration_mode") != "exact_pose"
        or not bool(row.get("source_configuration_available", False))
        or bool(row.get("pose_mismatch", False))
    ]
    if invalid_restoration:
        raise RuntimeError(
            f"Source configuration restoration failed in {path}: "
            f"count={len(invalid_restoration)} first={invalid_restoration[:3]}"
        )
    if not require_episode_metrics:
        return
    local_pairing_keys: list[tuple[str, str]] = []
    for index, row in enumerate(payload.get("episodes", ())):
        if not isinstance(row, dict):
            raise RuntimeError(f"Episode {index} is not an object in {path}")
        if expected_strict_task_semantics:
            if (
                row.get("task_failure_diagnostic_schema")
                != TASK_FAILURE_DIAGNOSTIC_SCHEMA
                or row.get("strict_task_semantics_enabled") is not True
            ):
                raise RuntimeError(
                    f"Episode {index} strict task diagnostics are missing in {path}"
                )
        encounter_id = str(row.get("encounter_id", ""))
        layout_id = str(row.get("scene_layout_id", ""))
        if not encounter_id or not layout_id:
            raise RuntimeError(f"Episode {index} has an empty pairing key in {path}")
        local_pairing_keys.append((encounter_id, layout_id))
        steps = _required_integer(row, "steps", source=path, minimum=1)
        collision_steps = _required_integer(
            row,
            "collision_steps",
            source=path,
            minimum=0,
            maximum=steps,
        )
        near_steps = _required_integer(
            row, "near_steps", source=path, minimum=0, maximum=steps
        )
        success = _required_boolean(row, "success", source=path)
        terminated = _required_boolean(row, "terminated", source=path)
        truncated = _required_boolean(row, "truncated", source=path)
        if terminated and truncated:
            raise RuntimeError(
                f"Episode {index} cannot be both terminated and truncated in {path}"
            )
        if success and (not terminated or truncated):
            raise RuntimeError(
                f"Episode {index} success/termination provenance drift in {path}"
            )
        if expected_strict_task_semantics:
            terminal_reason = str(row.get("task_terminal_reason", ""))
            if terminal_reason == "max_episode_steps":
                valid_terminal = truncated and not terminated and not success
            elif terminal_reason == "success":
                valid_terminal = success and terminated and not truncated
            else:
                valid_terminal = terminated and not success and not truncated
            if not valid_terminal:
                raise RuntimeError(
                    f"Episode {index} strict terminal semantics drift in {path}: "
                    f"reason={terminal_reason!r}"
                )
        collision = _required_boolean(row, "collision", source=path)
        if collision != (collision_steps > 0):
            raise RuntimeError(f"Episode {index} collision provenance drift in {path}")
        for boolean_key in (
            "grasped_any",
            "released_after_grasp",
            "cube_entered_target_tolerance",
            "controller_lift_phase_reached",
            "controller_place_phase_reached",
            "controller_release_phase_reached",
            "minimum_ttc_valid",
        ):
            _required_boolean(row, boolean_key, source=path)
        for numeric_key, minimum, maximum in (
            ("physics_dt_s", 0.0, None),
            ("total_reward", None, None),
            ("collision_rate", 0.0, 1.0),
            ("near_rate", 0.0, 1.0),
            ("near_miss_rate", 0.0, 1.0),
            ("gate_activation_rate", 0.0, 1.0),
            ("min_surface_gap", None, None),
            ("minimum_ttc_s", 0.0, None),
            ("completion_time_s", 0.0, None),
            ("collision_duration_s", 0.0, None),
            ("collision_max_consecutive_duration_s", 0.0, None),
            ("near_human_duration_s", 0.0, None),
            ("physical_safety_active_rate", 0.0, 1.0),
            ("physical_safety_intervention_rate", 0.0, 1.0),
            ("physical_safety_feasible_rate", 0.0, 1.0),
            ("mean_physical_safety_intervention_norm_radps", 0.0, None),
            ("max_physical_safety_intervention_norm_radps", 0.0, None),
            ("mean_physical_safety_slack_mps", 0.0, None),
            ("max_physical_safety_slack_mps", 0.0, None),
            (
                "mean_physical_safety_constraint_violation_before_mps",
                0.0,
                None,
            ),
            (
                "max_physical_safety_constraint_violation_before_mps",
                0.0,
                None,
            ),
            (
                "mean_physical_safety_constraint_violation_after_mps",
                0.0,
                None,
            ),
            (
                "max_physical_safety_constraint_violation_after_mps",
                0.0,
                None,
            ),
            ("mean_physical_safety_solve_time_ms", 0.0, None),
            ("ee_path_length_m", 0.0, None),
            ("rms_ee_acceleration_mps2", 0.0, None),
            ("p95_ee_jerk_mps3", 0.0, None),
            ("rms_ee_jerk_mps3", 0.0, None),
            ("max_ee_jerk_mps3", 0.0, None),
            ("integrated_squared_ee_jerk_m2ps5", 0.0, None),
            ("rms_gate_ee_acceleration_mps2", 0.0, None),
            ("p95_gate_ee_jerk_mps3", 0.0, None),
            ("rms_gate_ee_jerk_mps3", 0.0, None),
            ("max_gate_ee_jerk_mps3", 0.0, None),
        ):
            _required_finite(
                row,
                numeric_key,
                source=path,
                minimum=minimum,
                maximum=maximum,
                minimum_exclusive=numeric_key == "physics_dt_s",
            )
        actual_episode_controller = str(row.get("physical_safety_controller", ""))
        if (
            expected_controller is not None
            and actual_episode_controller != expected_controller
        ):
            raise RuntimeError(
                f"Episode controller mismatch in {path}: "
                f"actual={actual_episode_controller!r}, "
                f"expected={expected_controller!r}"
            )
        expected_collision_rate = collision_steps / steps
        if not math.isclose(
            float(row["collision_rate"]),
            expected_collision_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Episode collision rate drift in {path}")
        physics_dt_s = float(row["physics_dt_s"])
        duration_checks = {
            "completion_time_s": steps * physics_dt_s,
            "collision_duration_s": collision_steps * physics_dt_s,
            "near_human_duration_s": near_steps * physics_dt_s,
        }
        for key, expected_value in duration_checks.items():
            if not math.isclose(
                float(row[key]), expected_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise RuntimeError(f"Episode {key} drift in {path}")
    if len(set(local_pairing_keys)) != len(local_pairing_keys):
        raise RuntimeError(f"Duplicate encounter/layout pairing keys in {path}")


def _pairing_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["eval_seed"]),
        str(row["encounter_id"]),
        str(row["scene_layout_id"]),
    )


def _validate_pairing(rows: list[dict[str, Any]], controllers: tuple[str, ...]) -> None:
    if not rows:
        raise RuntimeError("Paired evaluation contains no episode rows")
    unexpected = sorted(
        set(str(row.get("controller", "")) for row in rows) - set(controllers)
    )
    if unexpected:
        raise RuntimeError(f"Unexpected controller rows: {unexpected}")
    expected: set[tuple[int, str, str]] | None = None
    for controller in controllers:
        selected = [row for row in rows if row["controller"] == controller]
        if not selected:
            raise RuntimeError(f"Controller {controller} has no paired evidence")
        key_rows = [_pairing_key(row) for row in selected]
        if any(not key[1] or not key[2] for key in key_rows):
            raise RuntimeError(f"Controller {controller} has an empty pairing key")
        if len(set(key_rows)) != len(key_rows):
            raise RuntimeError(
                f"Controller {controller} contains duplicate pairing keys"
            )
        keys = set(key_rows)
        if expected is None:
            expected = keys
        elif keys != expected:
            raise RuntimeError(
                f"Paired evaluation mismatch for controller={controller}: "
                f"expected={len(expected)} actual={len(keys)}"
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
