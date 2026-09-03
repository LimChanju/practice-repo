from __future__ import annotations

"""Fail-closed A/B/C CBF response-family experiment orchestrator.

The development split is the only split on which B/C parameters may be
selected.  Held-out evaluation is deliberately inaccessible until a frozen
selection artifact binds the selected B and C condition IDs to the SHA-256 of
the completed development run manifest.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

SCHEMA_VERSION = "cbf_response_family_experiment_v1"
CONDITION_MANIFEST_SCHEMA = "cbf_response_family_condition_manifest_v1"
RUN_MANIFEST_SCHEMA = "cbf_response_family_run_manifest_v1"
SELECTION_SCHEMA = "cbf_response_family_frozen_selection_v1"
STRICT_TASK_SCHEMA = "physical_event_driven_pick_place_v5"
RECOVERY_SCHEMA = "state_aware_pick_place_recovery_bridge_v3"

CHECKPOINT = SCRIPT_DIR / "policies" / "bc_pick_place_v2_release_settle.pt"
MANIFEST_DIR = SCRIPT_DIR / "trajectories" / "manifests" / "p01_v8_22sessions_cv4"
DEV_MANIFEST = MANIFEST_DIR / "fold_01_arch2x2_development_bc_cbf_clean19_v1.json"
HELDOUT_MANIFEST = MANIFEST_DIR / "fold_01_eval_bc_cbf_clean17_v1.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "eval_results" / "cbf_response_family_v1"

# These are the qualified, immutable data/model inputs recorded in
# state_aware_recovery_bridge_validation_20260903.md and its source manifest.
FROZEN_INPUT_SHA256 = {
    str(CHECKPOINT.relative_to(PROJECT_DIR)): (
        "d0fe9f9dc48a6049b77c8e1eab6de0905197207bb57ff858e0d7d332172b5230"
    ),
    str(DEV_MANIFEST.relative_to(PROJECT_DIR)): (
        "0f8ee23a262a28b92a56a7b1244f3fdb7c0869ac0d41829133651345d532f8cd"
    ),
    str(HELDOUT_MANIFEST.relative_to(PROJECT_DIR)): (
        "85658d0269efd8592593d55022be286ebcb59a93078f33be9166187c4cc34909"
    ),
}

SEED = 11
MAX_STEPS = 1200
TASK_REWARD_VERSION = "reward_v4_post_release_stability_hri_errp"
STRICT_PLACE_XY_TOLERANCE_M = 0.05

SAFETY_CONSTRAINT_CONFIG = {
    "cbf_safe_gap_m": 0.05,
    "cbf_activation_gap_m": 0.13,
    "cbf_gamma_per_s": 8.0,
    "cbf_prediction_horizon_s": 0.15,
    "cbf_max_prediction_buffer_m": 0.08,
    "cbf_max_joint_speed_rad_s": 2.0,
}

OBJECTIVE_CONFIG_KEYS = (
    "cbf_objective_mode",
    "cbf_task_space_weight",
    "cbf_task_yaw_length_scale_m_per_rad",
    "cbf_joint_regularization_epsilon",
    "cbf_correction_smoothness_weight",
)

RUNTIME_SOURCE_PATHS = (
    "v3_chan/evaluate_rollout_policy.py",
    "v3_chan/physical_safety_controllers.py",
    "v3_chan/rl/pick_place_env.py",
    "v3_chan/rl/strict_task_semantics.py",
    "v3_chan/rl/state_aware_recovery.py",
    "v3_chan/run_cbf_response_family_experiment.py",
)

REQUIRED_EPISODE_METRICS = (
    "logged_surface_gap_below_configured_margin_steps",
    "logged_surface_gap_below_configured_margin_episode",
    "configured_safe_gap_m",
    "static_collision_steps",
    "static_collision_episode",
    "static_geometry_valid_steps",
    "self_collision_steps",
    "self_collision_episode",
    "self_geometry_valid_steps",
)

REQUIRED_STEP_COLUMNS = (
    "logged_surface_gap_below_configured_margin",
    "configured_safe_gap_m",
    "static_collision",
    "static_geometry_valid",
    "static_surface_gap_m",
    "self_collision",
    "self_geometry_valid",
    "self_surface_gap_m",
)


class ExperimentContractError(RuntimeError):
    """An artifact or configuration violated the frozen experiment contract."""


@dataclass(frozen=True)
class Condition:
    condition_id: str
    family: str
    controller: str
    objective_mode: str
    task_space_weight: float = 1.0
    task_yaw_length_scale_m_per_rad: float = 0.10
    joint_regularization_epsilon: float = 0.05
    correction_smoothness_weight: float = 1.0

    def objective_config(self) -> dict[str, Any]:
        return {
            "cbf_objective_mode": self.objective_mode,
            "cbf_task_space_weight": self.task_space_weight,
            "cbf_task_yaw_length_scale_m_per_rad": (
                self.task_yaw_length_scale_m_per_rad
            ),
            "cbf_joint_regularization_epsilon": (self.joint_regularization_epsilon),
            "cbf_correction_smoothness_weight": (self.correction_smoothness_weight),
        }


def development_conditions() -> tuple[Condition, ...]:
    return (
        Condition(
            condition_id="BC_ONLY",
            family="BC_ONLY",
            controller="none",
            objective_mode="joint_nominal",
        ),
        Condition(
            condition_id="A_joint_nominal",
            family="A",
            controller="cbf",
            objective_mode="joint_nominal",
        ),
        Condition(
            condition_id="B_task_consistent_eps_0p01",
            family="B",
            controller="cbf",
            objective_mode="task_consistent",
            joint_regularization_epsilon=0.01,
        ),
        Condition(
            condition_id="B_task_consistent_eps_0p05",
            family="B",
            controller="cbf",
            objective_mode="task_consistent",
            joint_regularization_epsilon=0.05,
        ),
        Condition(
            condition_id="B_task_consistent_eps_0p20",
            family="B",
            controller="cbf",
            objective_mode="task_consistent",
            joint_regularization_epsilon=0.20,
        ),
        Condition(
            condition_id="C_smooth_intervention_lambda_0p25",
            family="C",
            controller="cbf",
            objective_mode="smooth_intervention",
            correction_smoothness_weight=0.25,
        ),
        Condition(
            condition_id="C_smooth_intervention_lambda_1p00",
            family="C",
            controller="cbf",
            objective_mode="smooth_intervention",
            correction_smoothness_weight=1.0,
        ),
        Condition(
            condition_id="C_smooth_intervention_lambda_4p00",
            family="C",
            controller="cbf",
            objective_mode="smooth_intervention",
            correction_smoothness_weight=4.0,
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen BC + CBF + Recovery A/B/C experiment. Held-out "
            "execution requires a SHA-bound development selection artifact."
        )
    )
    parser.add_argument("--stage", choices=("dev", "heldout", "all"), required=True)
    parser.add_argument(
        "--selection-json",
        default="",
        help=(
            "Frozen B/C selection. Required for heldout/all; generated template "
            "is written after a successful development run."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--eval-log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_log_every < 1:
        parser.error("--eval-log-every must be positive")
    if args.stage in {"heldout", "all"} and not args.selection_json:
        parser.error("--selection-json is required for heldout/all")
    if args.stage == "all" and args.force:
        parser.error(
            "--force cannot be used with --stage all: rerunning development "
            "would invalidate its already-frozen selection"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    _validate_frozen_inputs()

    if args.stage == "dev":
        _run_split(
            split="dev",
            encounter_manifest=DEV_MANIFEST,
            conditions=development_conditions(),
            output_root=output_root,
            device=args.device,
            eval_log_every=args.eval_log_every,
            force=bool(args.force),
            allow_launch=True,
            selection_reference=None,
        )
        return

    selection_path = Path(args.selection_json).expanduser().resolve()
    selection = _load_frozen_selection(selection_path, output_root=output_root)
    selected_conditions = _selected_heldout_conditions(selection)

    if args.stage == "all":
        # _load_frozen_selection validates every referenced development output.
        # Do not rewrite even a manifest timestamp after its SHA has been frozen.
        print("[CBFResponseFamily] frozen development artifacts validated")

    _run_split(
        split="heldout",
        encounter_manifest=HELDOUT_MANIFEST,
        conditions=selected_conditions,
        output_root=output_root,
        device=args.device,
        eval_log_every=args.eval_log_every,
        force=bool(args.force),
        allow_launch=True,
        selection_reference=_file_reference(selection_path),
    )


def _run_split(
    *,
    split: str,
    encounter_manifest: Path,
    conditions: Sequence[Condition],
    output_root: Path,
    device: str,
    eval_log_every: int,
    force: bool,
    allow_launch: bool,
    selection_reference: Mapping[str, str] | None,
) -> dict[str, Any]:
    if split not in {"dev", "heldout"}:
        raise ValueError(f"unknown split: {split}")
    _validate_frozen_inputs()
    runtime_sources_at_start = _runtime_source_references()
    manifest_payload = _load_json(encounter_manifest)
    expected_role = "development" if split == "dev" else "eval"
    actual_role = str(manifest_payload.get("split_metadata", {}).get("role", ""))
    if actual_role != expected_role:
        raise ExperimentContractError(
            f"{split} manifest role mismatch: expected {expected_role}, got {actual_role}"
        )
    expected_episodes = int(manifest_payload.get("scenario_count", 0))
    expected_count = 19 if split == "dev" else 17
    if expected_episodes != expected_count:
        raise ExperimentContractError(
            f"{split} scenario count mismatch: {expected_episodes} != {expected_count}"
        )

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    artifact_rows: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        condition_dir = split_dir / condition.condition_id
        output_json = condition_dir / "result.json"
        output_csv = condition_dir / "episodes.csv"
        output_steps = condition_dir / "steps.csv"
        output_log = condition_dir / "run.log"
        condition_manifest = condition_dir / "condition_manifest.json"
        outputs = (output_json, output_csv, output_steps)
        existing = tuple(path.exists() for path in outputs)
        if any(existing) and not all(existing) and not force:
            raise ExperimentContractError(
                f"partial cached outputs for {condition.condition_id}; use --force"
            )

        command = _build_command(
            condition=condition,
            encounter_manifest=encounter_manifest,
            output_json=output_json,
            output_csv=output_csv,
            output_steps=output_steps,
            device=device,
            eval_log_every=eval_log_every,
        )
        plan = _condition_manifest_payload(
            split=split,
            condition=condition,
            encounter_manifest=encounter_manifest,
            command=command,
            outputs=outputs,
            output_log=output_log,
            status="planned",
        )
        condition_dir.mkdir(parents=True, exist_ok=True)

        should_launch = force or not all(existing)
        if should_launch:
            if not allow_launch:
                raise ExperimentContractError(
                    f"frozen development artifact missing for {condition.condition_id}"
                )
            for path in (*outputs, output_log, condition_manifest):
                path.unlink(missing_ok=True)
            _atomic_write_json(condition_manifest, plan)
            _launch_condition(command, output_log)
            if _runtime_source_references() != runtime_sources_at_start:
                raise ExperimentContractError(
                    "runtime source changed while a condition was executing"
                )
        else:
            print(
                f"[CBFResponseFamily] validate cached {split}/{condition.condition_id}"
            )

        payload = _validate_result(
            output_json=output_json,
            output_csv=output_csv,
            output_steps=output_steps,
            encounter_manifest=encounter_manifest,
            expected_episodes=expected_episodes,
            condition=condition,
        )
        payloads[condition.condition_id] = payload
        completed = _condition_manifest_payload(
            split=split,
            condition=condition,
            encounter_manifest=encounter_manifest,
            command=command,
            outputs=outputs,
            output_log=output_log,
            status="completed",
        )
        _atomic_write_json(condition_manifest, completed)
        artifact_rows[condition.condition_id] = {
            "family": condition.family,
            "condition": asdict(condition),
            "condition_manifest": _file_reference(condition_manifest),
            "result": _file_reference(output_json),
            "episodes_csv": _file_reference(output_csv),
            "steps_csv": _file_reference(output_steps),
            "run_log": (_file_reference(output_log) if output_log.exists() else None),
        }

    cross_contract = _validate_cross_condition_contracts(payloads, conditions)
    _validate_frozen_inputs()
    if _runtime_source_references() != runtime_sources_at_start:
        raise ExperimentContractError("runtime source changed during the split run")
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "experiment_schema": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "split": split,
        "evidence_role": (
            "development_selection" if split == "dev" else "heldout_report_only"
        ),
        "heldout_opened": split == "heldout",
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "checkpoint": _file_reference(CHECKPOINT),
        "encounter_manifest": _file_reference(encounter_manifest),
        "runtime_sources": runtime_sources_at_start,
        "fixed_safety_constraints": dict(SAFETY_CONSTRAINT_CONFIG),
        "strict_task_schema": STRICT_TASK_SCHEMA,
        "state_aware_recovery_schema": RECOVERY_SCHEMA,
        "selection_reference": dict(selection_reference or {}),
        "conditions": artifact_rows,
        "cross_condition_validation": cross_contract,
    }
    run_manifest_path = split_dir / "run_manifest.json"
    _atomic_write_json(run_manifest_path, run_manifest)
    if split == "dev":
        _write_selection_template(split_dir, run_manifest_path)
    print(f"[CBFResponseFamily] completed split={split} manifest={run_manifest_path}")
    return run_manifest


def _build_command(
    *,
    condition: Condition,
    encounter_manifest: Path,
    output_json: Path,
    output_csv: Path,
    output_steps: Path,
    device: str,
    eval_log_every: int,
) -> list[str]:
    objective = condition.objective_config()
    return [
        str(PROJECT_DIR / "launch_isaac.sh"),
        str(SCRIPT_DIR / "evaluate_rollout_policy.py"),
        "--checkpoint",
        str(CHECKPOINT.resolve()),
        "--encounter-manifest",
        str(encounter_manifest.resolve()),
        "--encounter-policy",
        "cycle",
        "--encounter-anchor-mode",
        "world",
        "--encounter-timebase",
        "recorded",
        "--encounter-playback-speed",
        "1.0",
        "--episodes",
        "0",
        "--max-steps",
        str(MAX_STEPS),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--action-scale",
        "1.0",
        "--task-reward-version",
        TASK_REWARD_VERSION,
        "--residual-gate-mode",
        "checkpoint",
        "--mask-human-obs-for-policy",
        "--fixed-orientation",
        "--gripper-mode",
        "policy",
        "--no-pseudo-errp",
        "--release-gate-dist",
        "-1",
        "--require-release-for-success",
        "--physical-safety-controller",
        condition.controller,
        "--rmpflow-human-safety-margin-m",
        "0.05",
        "--cbf-safe-gap-m",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_safe_gap_m"]),
        "--cbf-activation-gap-m",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_activation_gap_m"]),
        "--cbf-gamma-per-s",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_gamma_per_s"]),
        "--cbf-prediction-horizon-s",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_prediction_horizon_s"]),
        "--cbf-max-prediction-buffer-m",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_max_prediction_buffer_m"]),
        "--cbf-max-joint-speed-rad-s",
        str(SAFETY_CONSTRAINT_CONFIG["cbf_max_joint_speed_rad_s"]),
        "--cbf-objective-mode",
        str(objective["cbf_objective_mode"]),
        "--cbf-task-space-weight",
        str(objective["cbf_task_space_weight"]),
        "--cbf-task-yaw-length-scale-m-per-rad",
        str(objective["cbf_task_yaw_length_scale_m_per_rad"]),
        "--cbf-joint-regularization-epsilon",
        str(objective["cbf_joint_regularization_epsilon"]),
        "--cbf-correction-smoothness-weight",
        str(objective["cbf_correction_smoothness_weight"]),
        "--extended-safety-logging",
        "--strict-task-semantics",
        "--strict-place-xy-tolerance-m",
        str(STRICT_PLACE_XY_TOLERANCE_M),
        "--state-aware-recovery",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--output-step-csv",
        str(output_steps),
        "--log-every",
        str(eval_log_every),
    ]


def _launch_condition(command: Sequence[str], output_log: Path) -> None:
    output_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["ISAAC_SKIP_VR_WAIT"] = "1"
    environment["ISAAC_SKIP_XR_RUNTIME_SEARCH"] = "1"
    print(
        "[CBFResponseFamily] launch "
        f"controller={command[command.index('--physical-safety-controller') + 1]} "
        f"objective={command[command.index('--cbf-objective-mode') + 1]} "
        f"log={output_log}",
        flush=True,
    )
    with output_log.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(
                list(command),
                cwd=PROJECT_DIR,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ExperimentContractError(
                f"Isaac evaluation failed with exit {exc.returncode}; see {output_log}"
            ) from exc


def _validate_result(
    *,
    output_json: Path,
    output_csv: Path,
    output_steps: Path,
    encounter_manifest: Path,
    expected_episodes: int,
    condition: Condition,
) -> dict[str, Any]:
    for path in (output_json, output_csv, output_steps):
        if not path.is_file() or path.stat().st_size == 0:
            raise ExperimentContractError(f"missing or empty result artifact: {path}")
    payload = _load_json(output_json)
    config = _mapping(payload.get("config"), context=f"{output_json} config")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise ExperimentContractError(
            f"{condition.condition_id}: expected {expected_episodes} episodes"
        )

    expected_base: dict[str, Any] = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "episodes": expected_episodes,
        "max_steps": MAX_STEPS,
        "seed": SEED,
        "mask_human_obs_for_policy": True,
        "fixed_orientation": True,
        "gripper_mode": "policy",
        "pseudo_errp_enabled": False,
        "encounter_manifest": str(encounter_manifest.resolve()),
        "encounter_policy": "cycle",
        "encounter_anchor_mode": "world",
        "encounter_timebase": "recorded",
        "encounter_playback_speed": 1.0,
        "require_release_for_success": True,
        "strict_task_semantics": True,
        "state_aware_recovery": True,
        "physical_safety_controller": condition.controller,
        "extended_safety_logging": True,
        "effective_task_reward_version": TASK_REWARD_VERSION,
    }
    for key, expected in expected_base.items():
        _require_equal(config, key, expected, condition.condition_id)
    for key, expected in SAFETY_CONSTRAINT_CONFIG.items():
        _require_equal(config, key, expected, condition.condition_id)
    for key, expected in condition.objective_config().items():
        _require_equal(config, key, expected, condition.condition_id)

    strict = _mapping(
        config.get("strict_task_semantics_config"),
        context=f"{condition.condition_id} strict config",
    )
    recovery = _mapping(
        config.get("state_aware_recovery_config"),
        context=f"{condition.condition_id} recovery config",
    )
    if strict.get("schema_version") != STRICT_TASK_SCHEMA or not strict.get("enabled"):
        raise ExperimentContractError(
            f"{condition.condition_id}: strict task schema is not frozen v5"
        )
    if float(strict.get("place_xy_tolerance_m", -1.0)) != STRICT_PLACE_XY_TOLERANCE_M:
        raise ExperimentContractError(
            f"{condition.condition_id}: strict place tolerance drift"
        )
    if strict.get("state_aware_recovery") is not True:
        raise ExperimentContractError(
            f"{condition.condition_id}: strict semantics did not bind recovery"
        )
    if recovery.get("schema_version") != RECOVERY_SCHEMA or not recovery.get("enabled"):
        raise ExperimentContractError(
            f"{condition.condition_id}: recovery schema is not frozen v3"
        )

    for index, episode in enumerate(episodes):
        row = _mapping(episode, context=f"episode {index}")
        if row.get("physical_safety_controller") != condition.controller:
            raise ExperimentContractError(
                f"{condition.condition_id}: episode {index} controller drift"
            )
        for key in REQUIRED_EPISODE_METRICS:
            if key not in row:
                raise ExperimentContractError(
                    f"{condition.condition_id}: episode {index} lacks {key}"
                )
        if int(row["static_geometry_valid_steps"]) <= 0:
            raise ExperimentContractError(
                f"{condition.condition_id}: static safety metric unavailable in episode {index}"
            )
        if int(row["self_geometry_valid_steps"]) <= 0:
            raise ExperimentContractError(
                f"{condition.condition_id}: self safety metric unavailable in episode {index}"
            )

    with output_steps.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
    missing_columns = [name for name in REQUIRED_STEP_COLUMNS if name not in header]
    if missing_columns:
        raise ExperimentContractError(
            f"{condition.condition_id}: step CSV lacks {missing_columns}"
        )
    return payload


def _validate_cross_condition_contracts(
    payloads: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[Condition],
) -> dict[str, Any]:
    if set(payloads) != {condition.condition_id for condition in conditions}:
        raise ExperimentContractError(
            "cross-condition validation received wrong conditions"
        )
    pair_fingerprints: dict[str, str] = {}
    recovery_fingerprints: dict[str, str] = {}
    safety_fingerprints: dict[str, str] = {}
    objective_fingerprints: dict[str, str] = {}
    cbf_ids: list[str] = []
    for condition in conditions:
        payload = payloads[condition.condition_id]
        config = _mapping(payload.get("config"), context="result config")
        pair_fingerprints[condition.condition_id] = _fingerprint(
            _pairing_contract(payload)
        )
        recovery_fingerprints[condition.condition_id] = _fingerprint(
            _recovery_contract(config)
        )
        objective = {key: config.get(key) for key in OBJECTIVE_CONFIG_KEYS}
        if objective != condition.objective_config():
            raise ExperimentContractError(
                f"{condition.condition_id}: objective result contract mismatch"
            )
        objective_fingerprints[condition.condition_id] = _fingerprint(objective)
        if condition.controller == "cbf":
            cbf_ids.append(condition.condition_id)
            safety_fingerprints[condition.condition_id] = _fingerprint(
                _safety_constraint_contract(config)
            )

    if len(set(pair_fingerprints.values())) != 1:
        raise ExperimentContractError(
            f"paired initial/source states differ: {pair_fingerprints}"
        )
    if len(set(recovery_fingerprints.values())) != 1:
        raise ExperimentContractError(
            f"Recovery contract differs across conditions: {recovery_fingerprints}"
        )
    if not cbf_ids or len(set(safety_fingerprints.values())) != 1:
        raise ExperimentContractError(
            f"A/B/C safety constraints differ: {safety_fingerprints}"
        )
    return {
        "status": "PASS",
        "pairing_scope": (
            "deterministic_reset_source_and_initial_scene_contract; "
            "not claimed as branch-state identity"
        ),
        "pairing_fingerprint": next(iter(pair_fingerprints.values())),
        "condition_pairing_fingerprints": pair_fingerprints,
        "recovery_fingerprint": next(iter(recovery_fingerprints.values())),
        "condition_recovery_fingerprints": recovery_fingerprints,
        "safety_constraint_fingerprint": next(iter(safety_fingerprints.values())),
        "condition_safety_constraint_fingerprints": safety_fingerprints,
        "condition_objective_fingerprints": objective_fingerprints,
    }


def _pairing_contract(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ExperimentContractError("episodes are missing for pairing")
    keys = (
        "episode",
        "seed",
        "encounter_id",
        "source_layout_id",
        "scene_layout_id",
        "source_layout_seed",
        "collection_seed",
        "screening_seed",
        "active_cube",
        "source_cube_index",
        "initial_cube_positions",
        "initial_active_cube_position",
        "place_target_position",
        "restoration_mode",
        "cube_pose_restored",
        "target_pose_restored",
        "robot_initial_state_restored",
        "pose_mismatch",
    )
    contract: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        row = _mapping(value, context=f"pairing episode {index}")
        missing = [key for key in keys if key not in row]
        if missing:
            raise ExperimentContractError(
                f"pairing episode {index} lacks initial-state fields {missing}"
            )
        if row["restoration_mode"] != "exact_pose" or bool(row["pose_mismatch"]):
            raise ExperimentContractError(
                f"pairing episode {index} was not exact-pose restored"
            )
        contract.append({key: row[key] for key in keys})
    return contract


def _safety_constraint_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "physical_safety_controller": config.get("physical_safety_controller"),
        **{key: config.get(key) for key in SAFETY_CONSTRAINT_CONFIG},
        "safety_geometry_source": config.get("safety_geometry_source"),
        "safety_geometry_metadata": config.get("safety_geometry_metadata"),
    }
    expected = {"physical_safety_controller": "cbf", **SAFETY_CONSTRAINT_CONFIG}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ExperimentContractError(f"safety constraint drift: {key}")
    return result


def _recovery_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strict_task_semantics": config.get("strict_task_semantics"),
        "strict_task_semantics_config": config.get("strict_task_semantics_config"),
        "state_aware_recovery": config.get("state_aware_recovery"),
        "state_aware_recovery_config": config.get("state_aware_recovery_config"),
        "task_failure_diagnostic_schema": config.get("task_failure_diagnostic_schema"),
    }


def _condition_manifest_payload(
    *,
    split: str,
    condition: Condition,
    encounter_manifest: Path,
    command: Sequence[str],
    outputs: Sequence[Path],
    output_log: Path,
    status: str,
) -> dict[str, Any]:
    output_refs: dict[str, Any] = {}
    for label, path in zip(("result", "episodes_csv", "steps_csv"), outputs):
        output_refs[label] = (
            _file_reference(path) if path.exists() else {"path": str(path)}
        )
    output_refs["run_log"] = (
        _file_reference(output_log)
        if output_log.exists()
        else {"path": str(output_log)}
    )
    return {
        "schema_version": CONDITION_MANIFEST_SCHEMA,
        "experiment_schema": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "status": status,
        "split": split,
        "heldout_data": split == "heldout",
        "condition": asdict(condition),
        "fixed_safety_constraints": dict(SAFETY_CONSTRAINT_CONFIG),
        "objective_config": condition.objective_config(),
        "checkpoint": _file_reference(CHECKPOINT),
        "encounter_manifest": _file_reference(encounter_manifest),
        "runtime_sources": _runtime_source_references(),
        "command": list(command),
        "outputs": output_refs,
    }


def _write_selection_template(split_dir: Path, run_manifest_path: Path) -> None:
    template = {
        "schema_version": SELECTION_SCHEMA,
        "development_run_manifest": _file_reference(run_manifest_path),
        "selected_conditions": {"B": None, "C": None},
        "eligible_conditions": {
            "B": [
                condition.condition_id
                for condition in development_conditions()
                if condition.family == "B"
            ],
            "C": [
                condition.condition_id
                for condition in development_conditions()
                if condition.family == "C"
            ],
        },
        "selection_rule": (
            "Choose exactly one B and one C using development evidence only, "
            "then replace null IDs and save this as a frozen selection JSON."
        ),
        "heldout_unlocked": False,
    }
    _atomic_write_json(split_dir / "frozen_selection_template.json", template)


def _load_frozen_selection(path: Path, *, output_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentContractError(f"selection JSON does not exist: {path}")
    selection = _load_json(path)
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ExperimentContractError("unknown frozen selection schema")
    selected = _mapping(
        selection.get("selected_conditions"), context="selected_conditions"
    )
    if set(selected) != {"B", "C"}:
        raise ExperimentContractError("selection must contain exactly B and C")
    by_id = {
        condition.condition_id: condition for condition in development_conditions()
    }
    for family in ("B", "C"):
        condition_id = selected.get(family)
        if not isinstance(condition_id, str) or condition_id not in by_id:
            raise ExperimentContractError(f"invalid selected {family} condition")
        if by_id[condition_id].family != family:
            raise ExperimentContractError(
                f"selected {family} condition belongs to {by_id[condition_id].family}"
            )

    reference = _mapping(
        selection.get("development_run_manifest"),
        context="development_run_manifest reference",
    )
    expected_path = (output_root / "dev" / "run_manifest.json").resolve()
    declared_path = _resolve_declared_path(reference.get("path"))
    if declared_path != expected_path:
        raise ExperimentContractError(
            f"selection binds the wrong development manifest: {declared_path}"
        )
    if not expected_path.is_file():
        raise ExperimentContractError("development run manifest is missing")
    declared_sha = str(reference.get("sha256", ""))
    actual_sha = _sha256(expected_path)
    if declared_sha != actual_sha:
        raise ExperimentContractError("development run manifest SHA-256 mismatch")
    dev = _load_json(expected_path)
    if dev.get("schema_version") != RUN_MANIFEST_SCHEMA or dev.get("split") != "dev":
        raise ExperimentContractError(
            "selection source is not a valid dev run manifest"
        )
    if dev.get("heldout_opened") is not False:
        raise ExperimentContractError("development run manifest claims heldout access")
    rows = _mapping(dev.get("conditions"), context="development conditions")
    for family in ("B", "C"):
        condition_id = str(selected[family])
        if condition_id not in rows:
            raise ExperimentContractError(
                f"selected condition was not completed in development: {condition_id}"
            )
        row = _mapping(rows[condition_id], context=condition_id)
        if row.get("family") != family:
            raise ExperimentContractError(
                f"development family drift for {condition_id}"
            )
    _validate_runtime_sources_against_dev(dev)
    _validate_frozen_dev_artifacts(dev, output_root=output_root)
    return selection


def _selected_heldout_conditions(selection: Mapping[str, Any]) -> tuple[Condition, ...]:
    selected = _mapping(
        selection.get("selected_conditions"), context="selected_conditions"
    )
    by_id = {
        condition.condition_id: condition for condition in development_conditions()
    }
    return (
        by_id["BC_ONLY"],
        by_id["A_joint_nominal"],
        by_id[str(selected["B"])],
        by_id[str(selected["C"])],
    )


def _validate_runtime_sources_against_dev(dev: Mapping[str, Any]) -> None:
    frozen = _mapping(dev.get("runtime_sources"), context="dev runtime_sources")
    current = _runtime_source_references()
    if frozen != current:
        raise ExperimentContractError(
            "runtime source drift after development selection; heldout remains locked"
        )


def _validate_frozen_dev_artifacts(
    dev: Mapping[str, Any], *, output_root: Path
) -> None:
    rows = _mapping(dev.get("conditions"), context="development conditions")
    conditions = development_conditions()
    if set(rows) != {condition.condition_id for condition in conditions}:
        raise ExperimentContractError(
            "development run does not contain the complete sweep"
        )
    payloads: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        condition_dir = output_root / "dev" / condition.condition_id
        expected_paths = {
            "condition_manifest": condition_dir / "condition_manifest.json",
            "result": condition_dir / "result.json",
            "episodes_csv": condition_dir / "episodes.csv",
            "steps_csv": condition_dir / "steps.csv",
        }
        row = _mapping(rows[condition.condition_id], context=condition.condition_id)
        for label, expected_path in expected_paths.items():
            reference = _mapping(
                row.get(label), context=f"{condition.condition_id} {label} reference"
            )
            _validate_reference(reference, expected_path=expected_path)
        payloads[condition.condition_id] = _validate_result(
            output_json=expected_paths["result"],
            output_csv=expected_paths["episodes_csv"],
            output_steps=expected_paths["steps_csv"],
            encounter_manifest=DEV_MANIFEST,
            expected_episodes=19,
            condition=condition,
        )
    actual_cross = _validate_cross_condition_contracts(payloads, conditions)
    frozen_cross = _mapping(
        dev.get("cross_condition_validation"),
        context="development cross-condition validation",
    )
    for key in (
        "status",
        "pairing_fingerprint",
        "recovery_fingerprint",
        "safety_constraint_fingerprint",
        "condition_objective_fingerprints",
    ):
        if frozen_cross.get(key) != actual_cross.get(key):
            raise ExperimentContractError(
                f"development cross-condition evidence drift: {key}"
            )


def _validate_reference(
    reference: Mapping[str, Any], *, expected_path: Path | None = None
) -> Path:
    resolved = _resolve_declared_path(reference.get("path"))
    if expected_path is not None and resolved != expected_path.resolve():
        raise ExperimentContractError(
            f"artifact reference path mismatch: {resolved} != {expected_path.resolve()}"
        )
    if not resolved.is_file():
        raise ExperimentContractError(f"referenced artifact is missing: {resolved}")
    declared_sha = str(reference.get("sha256", ""))
    if not declared_sha or _sha256(resolved) != declared_sha:
        raise ExperimentContractError(f"artifact SHA-256 mismatch: {resolved}")
    return resolved


def _validate_frozen_inputs() -> None:
    for relative, expected_sha in FROZEN_INPUT_SHA256.items():
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise ExperimentContractError(f"frozen input is missing: {path}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ExperimentContractError(
                f"frozen input drift for {relative}: {actual_sha} != {expected_sha}"
            )


def _runtime_source_references() -> dict[str, dict[str, str]]:
    return {
        relative: _file_reference(PROJECT_DIR / relative)
        for relative in RUNTIME_SOURCE_PATHS
    }


def _require_equal(
    config: Mapping[str, Any], key: str, expected: Any, condition_id: str
) -> None:
    if key not in config or config[key] != expected:
        raise ExperimentContractError(
            f"{condition_id}: config {key}={config.get(key)!r}, expected {expected!r}"
        )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentContractError(f"{context} must be an object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentContractError(f"JSON root must be an object: {path}")
    return value


def _fingerprint(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def _file_reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": _display_path(resolved), "sha256": _sha256(resolved)}


def _resolve_declared_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError("artifact reference path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
