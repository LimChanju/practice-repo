#!/usr/bin/env python3
"""Collect randomized A/C safety-response feedback in a single-pick VR task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (PROJECT_ROOT, SCRIPT_DIR, SCRIPT_DIR / "rl"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from v3_chan.collection_provenance import source_tree_sha256  # noqa: E402
from v3_chan.ac_feedback.action_trace import ActionTraceRecorder  # noqa: E402
from v3_chan.ac_feedback.config import (  # noqa: E402
    DEFAULT_CONFIG,
    RUNTIME_CONFIG,
    actual_cbf_config,
    canonical_sha256,
    file_sha256,
    load_config,
)
from v3_chan.ac_feedback.online_protocol import (  # noqa: E402
    CrossingPathMonitor,
    EncounterDetector,
    EncounterDetectorConfig,
    resolve_actual_crossing_direction,
)
from v3_chan.ac_feedback.study import (  # noqa: E402
    CONDITIONS,
    ResponsePhase,
    ResponsePhaseConfig,
    SafetyResponsePhaseDetector,
    TrialLifecycle,
    TrialSpec,
    TrialState,
    build_participant_schedule,
    validate_decision_context,
)
from v3_chan.ac_feedback.online_recorder import OnlineExplicitFeedbackRecorder  # noqa: E402
from v3_chan.ac_feedback.online_rows import build_online_transition_row  # noqa: E402
from v3_chan.ac_feedback.online_scenario import (  # noqa: E402
    NOMINAL_BC_SWEEP_SEMANTICS,
    CrossingCorridor,
    CrossingCueVisuals,
    NominalBCCorridorSweep,
    ProximalArmProtocolMonitor,
    calibrate_surface_gap_corridor,
)
from v3_chan.ac_feedback.online_schema import (  # noqa: E402
    EncounterRecordV1,
    Q1_ID,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
)
from v3_chan.ac_feedback.policy import FrozenBCPolicy  # noqa: E402
from v3_chan.ac_feedback.rows import validate_runtime_step  # noqa: E402
from v3_chan.ac_feedback.runtime import (  # noqa: E402
    apply_controlled_hold,
    measured_joint_state,
    validate_environment_contract,
)
from v3_chan.ac_feedback.schema import (  # noqa: E402
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from v3_chan.ac_feedback.tracking import TrackingDropout, TrackingWatchdog  # noqa: E402
from v3_chan.ac_feedback.tracking_guard import PreApplyTrackingGuard  # noqa: E402
from v3_chan.ac_feedback.video import (  # noqa: E402
    SynchronizedSpectatorVideoRecorder,
)
from v3_chan.runtime_contracts.ac_selective_smoothing_v1.validate_runtime_contract import (  # noqa: E402
    validate_runtime_contract,
)


class CollectionAbort(RuntimeError):
    pass


def _resolve_verified_code_provenance(
    project_root: Path, explicit_commit: str
) -> dict[str, Any]:
    """Resolve Git provenance without accepting a claimed, mismatched SHA."""

    tree_hash = source_tree_sha256(project_root)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    try:
        head = git("rev-parse", "HEAD").lower()
        branch = git("branch", "--show-current")
        dirty_output = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        if explicit_commit:
            raise RuntimeError(
                "cannot verify --code-commit-sha outside a readable Git checkout"
            )
        return {
            "code_commit_sha": "",
            "code_version": f"source-sha256:{tree_hash}",
            "code_version_source": "source_tree_sha256",
            "code_commit_verification": "source_tree_only",
            "git_branch": "",
            "repository_dirty": 1,
            "source_tree_sha256": tree_hash,
        }
    claimed = str(explicit_commit).strip().lower()
    if claimed and claimed != head:
        raise RuntimeError(
            f"claimed code commit {claimed} does not match Git HEAD {head}"
        )
    dirty = bool(dirty_output)
    return {
        "code_commit_sha": head,
        "code_version": head,
        "code_version_source": "verified_git_head",
        "code_commit_verification": (
            "verified_git_head_clean" if not dirty else "git_head_dirty"
        ),
        "git_branch": branch,
        "repository_dirty": int(dirty),
        "source_tree_sha256": tree_hash,
    }


@dataclass
class LoggedStep:
    obs: np.ndarray
    info: dict[str, Any]
    tracking: Any
    terminated: bool
    truncated: bool
    source_step: int
    result_step: int
    simulation_time_s: float
    monotonic_ns: int
    unix_ns: int
    minimum_gap_m: float
    left_gap_m: float
    right_gap_m: float
    ttc_s: float
    closing_speed_m_s: float
    dynamic_measurement_valid: bool
    cbf_active: bool
    intervention_norm_rad_s: float
    crossing_update: Any = None
    encounter_update: Any = None
    questionnaire_result: Any = None
    realtime_markers: tuple[Any, ...] = ()
    proximal_gap_m: float = 10.0
    proximal_collider_path: str = ""
    left_proximal_gap_m: float = 10.0
    right_proximal_gap_m: float = 10.0
    left_proximal_collider_path: str = ""
    right_proximal_collider_path: str = ""
    response_phase: str = ResponsePhase.PRE_RESPONSE.value
    decision_context: dict[str, Any] | None = None
    response_update: Any = None
    recovery_active: bool = False
    smooth_tail_active: bool = False
    correction_arm_rad_s: np.ndarray | None = None
    measured_arm_velocity_rad_s: np.ndarray | None = None
    ee_position_world_m: np.ndarray | None = None
    ee_linear_velocity_world_m_s: np.ndarray | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("minimal_pilot", "pilot_with_anchors"),
        default=None,
        help="Override only the two frozen schedule modes declared by the config.",
    )
    parser.add_argument("--practice-trials", type=int, default=4)
    parser.add_argument(
        "--practice-only",
        action="store_true",
        help="Run only practice trials; intended for equipment smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the frozen schedule without starting Isaac Sim.",
    )
    parser.add_argument(
        "--force-condition",
        choices=tuple(CONDITIONS),
        default=None,
        help="Practice-only hardware check; production assignment cannot be forced.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Practice-only limit used by one-trial smoke tests.",
    )
    # Kept as an internal false-valued compatibility field only.  The public
    # collector deliberately exposes no way to skip the post-query objective
    # task outcome required by this protocol.
    parser.set_defaults(reset_after_query=False)
    parser.add_argument("--tracking-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-pose-age-ms", type=float, default=100.0)
    parser.add_argument("--tracking-stable-frames", type=int, default=6)
    parser.add_argument(
        "--code-commit-sha",
        default=os.environ.get("HRI_CODE_COMMIT_SHA", ""),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.participant_id).strip() or not str(args.session_id).strip():
        raise ValueError("participant/session IDs must be non-empty pseudonyms")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.practice_trials < 0:
        raise ValueError("--practice-trials must be non-negative")
    if args.practice_trials != 4:
        raise ValueError(
            "the frozen protocol always schedules four practice trials; use "
            "--practice-only --max-trials N for a shorter equipment smoke"
        )
    if args.force_condition and not args.practice_only:
        raise ValueError("--force-condition is permitted only with --practice-only")
    if args.max_trials is not None and (not args.practice_only or args.max_trials < 1):
        raise ValueError("--max-trials is a positive practice-only option")
    if args.reset_after_query:
        raise ValueError(
            "--reset-after-query is incompatible with this protocol: every "
            "trial must continue to an observed task success or failure after "
            "the mandatory in-HMD response"
        )
    if args.tracking_stable_frames < 1:
        raise ValueError("--tracking-stable-frames must be positive")
    if not math.isfinite(args.tracking_timeout_s) or args.tracking_timeout_s <= 0:
        raise ValueError("--tracking-timeout-s must be finite and positive")
    if not math.isfinite(args.max_pose_age_ms) or args.max_pose_age_ms <= 0:
        raise ValueError("--max-pose-age-ms must be finite and positive")
    commit = str(args.code_commit_sha).strip().lower()
    if commit and (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError(
            "--code-commit-sha, when supplied, must be a 40-character lowercase Git SHA"
        )


def _assert_haptics_disabled(config: Mapping[str, Any]) -> None:
    if bool(config["feedback"]["haptics_enabled"]):
        raise ValueError("this A/C study requires haptics_enabled=false")
    truthy = {"1", "true", "yes", "on", "enabled"}
    enabled_variables = [
        name
        for name in (
            "BHAPTICS_ENABLED",
            "HRI_HAPTICS_ENABLED",
            "HRI_HAPTIC_FEEDBACK",
        )
        if os.environ.get(name, "").strip().lower() in truthy
    ]
    if enabled_variables:
        raise RuntimeError(
            "haptic output is forbidden; disable: "
            + ", ".join(enabled_variables)
        )
    haptic_condition = os.environ.get("HRI_HAPTIC_CONDITION", "").strip().lower()
    if haptic_condition not in {"", "0", "off", "none", "disabled"}:
        raise RuntimeError(
            "haptic output is forbidden; set HRI_HAPTIC_CONDITION=off"
        )


def _load_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = load_config(args.config)
    selected_mode = args.mode or str(frozen["study"]["mode"])
    if selected_mode not in frozen["study"]["available_modes"]:
        raise ValueError("--mode is not declared by study.available_modes")
    protocol_mode = dict(frozen)
    protocol_mode["study"] = dict(frozen["study"])
    protocol_mode["study"]["selected_mode"] = selected_mode
    feedback = dict(frozen["feedback"])
    feedback["opposite_controller_buttons"] = {
        "left": dict(feedback["realtime_markers"]["crossing_left"]),
        "right": dict(feedback["realtime_markers"]["crossing_right"]),
    }
    feedback["navigation"] = dict(feedback["questionnaire_navigation"])
    feedback["stable_release_frames"] = int(feedback["debounce_release_frames"])
    protocol = protocol_mode
    protocol["cbf"] = dict(frozen["shared_cbf"])
    protocol["feedback"] = feedback
    response = frozen["response_episode"]
    protocol["encounter"] = {
        "onset_confirmation_frames": int(response["onset_confirmation_frames"]),
        "activation_gap_m": float(frozen["shared_cbf"]["activation_gap_m"]),
        "ttc_onset_s": 0.75,
        "intervention_onset_rad_s": float(response["onset_intervention_rad_s"]),
        "clear_gap_m": float(frozen["crossing"]["neutral_clear_gap_m"]),
        "ttc_clear_s": 1.0,
        "intervention_clear_rad_s": float(response["stable_intervention_rad_s"]),
        "clear_duration_s": float(response["stable_duration_s"]),
        "merge_reentry_s": 1.0,
        "timeout_s": float(response["maximum_duration_s"]),
        "window_pre_s": 1.0,
        "window_post_s": 2.0,
    }
    scenario = {
        "schema_version": frozen["schema_version"],
        "surface_gap_reference": "time_aligned_bc_only_nominal_protected_surface_gap",
        "actual_gap_reference": "actual_cbf_protected_surface_to_tracked_hand_surface",
        "severity": {
            name: {
                "minimum_gap_range_m": list(values["planned_minimum_gap_range_m"]),
                "target_gap_m": float(values["target_gap_m"]),
            }
            for name, values in frozen["severity"].items()
        },
        "crossing": dict(frozen["crossing"]),
        "nominal_bc_reference": dict(frozen["nominal_bc_reference"]),
        "layout_precheck": dict(frozen["layout_precheck"]),
        "protected_crossing": {
            "requested_distal_links": list(frozen["crossing"]["protected_distal_links"]),
            "proximal_link_tokens": list(frozen["crossing"]["proximal_link_tokens"]),
            "proximal_crossing_action": str(frozen["crossing"]["off_protocol_action"]),
        },
    }
    return protocol, scenario


def _build_schedule(
    args: argparse.Namespace,
) -> tuple[tuple[TrialSpec, ...], tuple[TrialPlan, ...]]:
    schedule = build_participant_schedule(
        str(args.participant_id),
        session_id=str(args.session_id),
        seed=int(args.seed),
        mode=str(args.mode or load_config(args.config)["study"]["mode"]),
        practice_trials=int(args.practice_trials),
    )
    specs = tuple(schedule.practice_trials if args.practice_only else schedule.trials)
    if args.force_condition:
        forced = CONDITIONS[str(args.force_condition)]
        specs = tuple(
            replace(
                spec,
                condition_id=forced.condition_id,
                objective_mode=forced.objective_mode,
                lambda_s=forced.lambda_s,
            )
            for spec in specs
        )
    if args.max_trials is not None:
        specs = specs[: int(args.max_trials)]
    plans = tuple(_trial_plan(spec, index=index) for index, spec in enumerate(specs))
    return specs, plans


def _trial_plan(spec: TrialSpec, *, index: int) -> TrialPlan:
    return TrialPlan(
        trial_index=index,
        trial_id=spec.trial_id,
        query_id=spec.query_id,
        encounter_id=spec.encounter_id,
        task_phase=spec.task_phase,
        severity=spec.severity,
        direction=spec.crossing_direction,
        speed=spec.crossing_speed,
        crossing_hand=spec.crossing_hand,
        condition_id=spec.condition_id,
        objective_mode=spec.objective_mode,
        lambda_s=spec.lambda_s,
        schedule_seed=spec.schedule_seed,
        block_id=spec.block_id,
        counterbalancing_group=spec.counterbalancing_group,
        practice=spec.practice,
        anchor_repeat=spec.anchor_repeat,
        pilot=spec.pilot,
        analysis_exclude=spec.analysis_exclude,
        source_trial_id=spec.anchor_context_id,
    )


def _default_output(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(args.session_id)
    )
    return (
        PROJECT_ROOT
        / "v3_chan"
        / "ac_selective_smoothing_feedback_data"
        / f"{session}_{timestamp}.hdf5"
    )


def _simulation_config() -> dict[str, Any]:
    isaac_root = Path(
        os.environ.get("ISAACSIM_ROOT", str(Path.home() / "isaac-sim-4.5.0"))
    )
    xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
    name = (
        "isaacsim.exp.base.xr.openxr.kit"
        if xr_mode == "openxr"
        else "isaacsim.exp.base.xr.vr.kit"
    )
    experience = isaac_root / "apps" / name
    if not experience.is_file():
        raise FileNotFoundError(f"Live VR experience not found: {experience}")
    return {
        "headless": False,
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "experience": str(experience),
    }


def _build_environment(provider: Any, *, seed: int):
    from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv, PickPlaceEnvConfig

    return IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            cube_count=1,
            max_episode_steps=12_000,
            success_dist=0.06,
            action_scale=1.0,
            action_version="action_v1_controller_target_delta",
            fixed_orientation=True,
            gripper_mode="policy",
            require_release_for_success=True,
            strict_task_semantics=True,
            strict_place_xy_tolerance_m=0.05,
            state_aware_recovery=True,
            reward_version="reward_v4_post_release_stability_hri_errp",
            observation_mode="flat",
            seed=int(seed),
            render=True,
            pseudo_errp_enabled=False,
            pseudo_errp_sources=(),
            visualize_human_replay=True,
            synthetic_human_enabled=False,
            physical_safety_controller="cbf",
            cbf_safe_gap_m=0.05,
            cbf_activation_gap_m=0.13,
            cbf_gamma_per_s=8.0,
            cbf_prediction_horizon_s=0.15,
            cbf_max_prediction_buffer_m=0.08,
            cbf_max_joint_speed_rad_s=2.0,
            cbf_objective_mode="joint_nominal",
            cbf_task_space_weight=1.0,
            cbf_task_yaw_length_scale_m_per_rad=0.10,
            cbf_joint_regularization_epsilon=0.05,
            cbf_correction_smoothness_weight=0.0,
            extended_backup_safety_geometry=False,
        ),
        human_state_fn=provider,
    )


def _apply_trial_condition(env: Any, spec: TrialSpec) -> dict[str, Any]:
    """Select A or C only at RESET while preserving every shared CBF field."""

    from dataclasses import asdict as dataclass_dict
    from dataclasses import replace as dataclass_replace

    cbf = getattr(env, "_cbf_filter", None)
    if cbf is None or getattr(cbf, "config", None) is None:
        raise CollectionAbort("A/C selection requires the fixed CBF filter")
    before = dataclass_dict(cbf.config)
    after_config = dataclass_replace(
        cbf.config,
        objective_mode=str(spec.objective_mode),
        correction_smoothness_weight=float(spec.lambda_s),
    ).validated()
    after = dataclass_dict(after_config)
    allowed = {"objective_mode", "correction_smoothness_weight"}
    drift = {
        name: (before.get(name), after.get(name))
        for name in sorted(set(before).union(after))
        if before.get(name) != after.get(name) and name not in allowed
    }
    if drift:
        raise CollectionAbort(f"A/C selection changed shared CBF fields: {drift}")
    cbf.config = after_config
    env.config.cbf_objective_mode = str(spec.objective_mode)
    env.config.cbf_correction_smoothness_weight = float(spec.lambda_s)
    cbf.reset()
    if getattr(cbf, "_previous_correction", None) is not None or getattr(cbf, "_pending_correction", None) is not None:
        raise CollectionAbort("CBF correction memory did not clear at trial RESET")
    return actual_cbf_config(cbf, condition_id=spec.condition_id)


def _capture_scene_and_restoration(
    env: Any, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    from v3_chan.scene_randomization import scene_layout_id

    cube_poses = [cube.get_world_pose() for cube in env.cubes]
    target_position, target_orientation = env.place_target.get_world_pose()
    positions, velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    names = tuple(str(cube.name) for cube in env.cubes)
    cube_positions = np.asarray([pose[0] for pose in cube_poses], dtype=np.float64)
    cube_orientations = np.asarray([pose[1] for pose in cube_poses], dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_orientation = np.asarray(target_orientation, dtype=np.float64)
    scene = {
        "cube_names": names,
        "cube_positions_world_m": cube_positions,
        "cube_orientations_wxyz": cube_orientations,
        "place_target_position_world_m": target_position,
        "place_target_orientation_wxyz": target_orientation,
        "robot_initial_joint_positions_rad": positions,
        "robot_initial_joint_velocities_rad_s": velocities,
    }
    source_configuration = {
        "cube_names": names,
        "cube_positions_world": cube_positions,
        "cube_orientations_wxyz": cube_orientations,
        "place_target_position_world": target_position,
        "place_target_orientation_wxyz": target_orientation,
        "robot_initial_joint_positions": positions,
        "robot_initial_joint_velocities": velocities,
    }
    restoration = {
        "source_configuration_available": True,
        "restoration_mode": "exact_pose",
        "restoration_reason": "passed_frozen_bc_only_precheck",
        "source_cube_index": 0,
        "source_cube_name": names[0],
        "collection_seed": int(seed),
        "layout_seed": int(seed),
        "source_configuration": source_configuration,
    }
    layout_id = scene_layout_id(
        cube_positions, cube_orientations, target_position, target_orientation
    )
    return scene, restoration, layout_id


def _strict_state(info: Mapping[str, Any]) -> dict[str, Any]:
    value = info.get("strict_task_semantics", {})
    value = dict(value) if isinstance(value, Mapping) else {}
    state = value.get("state", {})
    return dict(state) if isinstance(state, Mapping) else {}


def _wait_for_tracking_and_clear(
    *,
    env: Any,
    provider: Any,
    simulation_app: Any,
    joint_count: int,
    timeout_s: float,
    stable_frames: int,
    clear_gap_m: float,
    proximal_monitor: ProximalArmProtocolMonitor | None,
    stop_requested: Callable[[], None],
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Refresh a paused initial observation until tracking and both gaps are clear."""

    env.world.pause()
    apply_controlled_hold(env.robot, joint_count=joint_count)
    started = time.monotonic()
    stable = 0
    last_reason = "no observation"
    while stable < stable_frames:
        stop_requested()
        obs, info = env.refresh_observation()
        snapshot = provider.observation_snapshot
        try:
            if snapshot is None:
                raise RuntimeError("missing observation tracking snapshot")
            provider.watchdog.require_current(snapshot)
            proximal_clear = True
            if proximal_monitor is not None:
                proximal_clear = all(
                    proximal_monitor.closest_surface_gap_m(
                        getattr(snapshot, hand).position_world
                    )[0]
                    > float(clear_gap_m)
                    for hand in ("left", "right")
                )
            gaps_clear = (
                bool(info.get("geometry_valid", False))
                and float(info.get("left_end_effector_surface_gap_m", -1.0))
                > float(clear_gap_m)
                and float(info.get("right_end_effector_surface_gap_m", -1.0))
                > float(clear_gap_m)
                and proximal_clear
            )
            if not gaps_clear:
                raise RuntimeError(
                    "move both hands to staging "
                    f"(>{float(clear_gap_m):.3f} m protected-surface gap)"
                )
        except (TrackingDropout, RuntimeError) as error:
            stable = 0
            last_reason = str(error)
        else:
            stable += 1
            last_reason = ""
        apply_controlled_hold(env.robot, joint_count=joint_count)
        simulation_app.update()
        if time.monotonic() - started > timeout_s:
            raise CollectionAbort("tracking/clear staging timeout: " + last_reason)
        time.sleep(0.01)
    env.world.play()
    assert snapshot is not None
    return np.asarray(obs, dtype=np.float32), dict(info), snapshot


def _feedback_clocks(env: Any, *, monotonic_ns: int, unix_ns: int):
    from v3_chan.ac_feedback.xr_feedback import FeedbackClocks

    return FeedbackClocks(
        sim_time_s=max(0.0, float(getattr(env.world, "current_time", 0.0))),
        monotonic_ns=int(monotonic_ns),
        unix_ns=int(unix_ns),
        control_step=max(0, int(getattr(env, "step_count", 0))),
    )


def _wait_for_controller_release(
    *,
    source: Any,
    feedback: Any,
    provider: Any,
    simulation_app: Any,
    env: Any,
    timeout_s: float,
    stop_requested: Callable[[], None],
) -> None:
    started = time.monotonic()
    stable = 0
    feedback.set_crossing_hand(None)
    while stable < 3:
        stop_requested()
        # Refresh VRAvatar's per-frame device cache before reading buttons.
        # The button adapter then reuses those wrappers instead of querying
        # XRCore a second time during the same frame.
        provider.sample()
        snapshot = source.read()
        known_release = bool(
            snapshot.left_connected is True
            and snapshot.right_connected is True
            and snapshot.left_x is False
            and snapshot.left_y is False
            and snapshot.right_a is False
            and snapshot.right_b is False
        )
        stable = stable + 1 if known_release else 0
        feedback.update(
            snapshot,
            _feedback_clocks(
                env, monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns()
            ),
        )
        simulation_app.update()
        if time.monotonic() - started > timeout_s:
            detail = getattr(source, "last_error", "") or "buttons unavailable/held"
            raise CollectionAbort("VR controller preflight failed: " + detail)
        time.sleep(0.01)


def _wait_for_clear_control_observation(
    *,
    env: Any,
    provider: Any,
    simulation_app: Any,
    joint_count: int,
    timeout_s: float,
    clear_confirmation_frames: int,
    stop_requested: Callable[[], None],
    context: str,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Legacy collector helper retained for API/schema compatibility."""

    apply_controlled_hold(env.robot, joint_count=joint_count)
    env.world.pause()
    required_clear = max(1, int(clear_confirmation_frames))
    clear_frames = 0
    started = time.monotonic()
    last_reason = "no control observation"
    while clear_frames < required_clear:
        stop_requested()
        obs, info = env.refresh_observation()
        snapshot = provider.observation_snapshot
        tracking_error = ""
        if snapshot is None:
            tracking_error = "missing tracking snapshot"
        else:
            try:
                provider.watchdog.require_current(snapshot)
            except TrackingDropout as error:
                tracking_error = str(error)
        geometry_valid = bool(info.get("geometry_valid", False))
        left_gap = float(info.get("left_end_effector_surface_gap_m", math.nan))
        right_gap = float(info.get("right_end_effector_surface_gap_m", math.nan))
        clear = bool(
            not tracking_error
            and geometry_valid
            and math.isfinite(left_gap)
            and math.isfinite(right_gap)
            and left_gap > 0.13
            and right_gap > 0.13
        )
        if clear:
            clear_frames += 1
            last_reason = ""
        else:
            clear_frames = 0
            last_reason = (
                f"tracking={tracking_error or 'valid'}, "
                f"geometry_valid={geometry_valid}, left_gap={left_gap:.6g}, "
                f"right_gap={right_gap:.6g}"
            )
        if clear_frames >= required_clear:
            assert snapshot is not None
            env.world.play()
            return np.asarray(obs, dtype=np.float32), dict(info), snapshot
        apply_controlled_hold(env.robot, joint_count=joint_count)
        is_playing = getattr(env.world, "is_playing", None)
        if callable(is_playing) and bool(is_playing()):
            raise CollectionAbort(
                f"{context} readiness attempted while physics was playing"
            )
        simulation_app.update()
        if time.monotonic() - started >= timeout_s:
            raise CollectionAbort(
                f"{context} did not reach a stable clear CBF band: "
                + last_reason
            )
        time.sleep(0.01)
    raise AssertionError("unreachable clear-control-observation state")


def _commit_validated_with_signal_barrier(
    *,
    recorder: Any,
    abort_reason_fn: Callable[[], str],
    commit_state: dict[str, bool],
) -> Path:
    """Make the validated rename an explicit SIGINT/SIGTERM boundary."""

    required_signal_api = (
        "pthread_sigmask",
        "sigpending",
        "SIG_BLOCK",
        "SIG_SETMASK",
    )
    missing = [name for name in required_signal_api if not hasattr(signal, name)]
    if missing:
        raise RuntimeError(
            "atomic collection commit requires POSIX signal masking; missing "
            + ", ".join(missing)
        )
    watched = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    final_path: Path | None = None

    def pending_reason() -> str:
        pending = set(signal.sigpending()).intersection(watched)
        if not pending:
            return ""
        return f"signal_{min(int(signum) for signum in pending)}"

    def restore_partial(reason: str) -> None:
        commit_state["value"] = False
        assert final_path is not None
        os.replace(final_path, recorder.partial_path)
        raise CollectionAbort(reason)

    try:
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            raise CollectionAbort(reason)
        final_path = recorder.commit_validated()
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            restore_partial(reason)
        commit_state["value"] = True
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            restore_partial(reason)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    # Unmasking may synchronously deliver a signal to request_stop().  Check
    # once more after restoration so a just-committed artifact is moved back
    # to its recoverable .partial name rather than being published.
    reason = str(abort_reason_fn() or "").strip()
    if reason:
        restore_partial(reason)
    assert final_path is not None
    return final_path


def _nominal_sweep_offsets(config: Mapping[str, Any]) -> tuple[float, ...]:
    bounds = tuple(
        float(value) for value in config["candidate_outward_offset_range_m"]
    )
    if len(bounds) != 2:
        raise ValueError("candidate_outward_offset_range_m must have two values")
    low, high = bounds
    step = float(config["candidate_outward_offset_step_m"])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("nominal sweep offset range is invalid")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("nominal sweep offset step must be positive")
    count_float = (high - low) / step
    count = int(round(count_float))
    if not math.isclose(count_float, count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("nominal sweep range must be divisible by its step")
    return tuple(float(low + index * step) for index in range(count + 1))


def _require_clear_nominal_calibration_state(
    *,
    info: Mapping[str, Any],
    provider: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    clear_gap_m: float,
) -> Any:
    snapshot = provider.observation_snapshot
    if snapshot is None:
        raise CollectionAbort("nominal corridor sweep has no tracking snapshot")
    provider.watchdog.require_current(snapshot)
    distal = (
        float(info.get("left_end_effector_surface_gap_m", math.nan)),
        float(info.get("right_end_effector_surface_gap_m", math.nan)),
    )
    proximal = tuple(
        proximal_monitor.closest_surface_gap_m(
            getattr(snapshot, hand).position_world
        )[0]
        for hand in ("left", "right")
    )
    if (
        not bool(info.get("geometry_valid", False))
        or not all(math.isfinite(value) for value in (*distal, *proximal))
        or min(*distal, *proximal) < float(clear_gap_m)
    ):
        raise CollectionAbort(
            "hands entered protected/proximal space during BC-only nominal "
            "corridor calibration"
        )
    return snapshot


def _nominal_trajectory_sample(
    *,
    env: Any,
    info: Mapping[str, Any],
    phase_start_simulation_time_s: float,
    arm_joint_indices: Sequence[int],
) -> dict[str, Any]:
    positions, velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    indices = np.asarray(arm_joint_indices, dtype=int)
    ee_position, ee_orientation = _ee_pose(env)
    simulation_time_s = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    return {
        "relative_time_s": simulation_time_s
        - float(phase_start_simulation_time_s),
        "simulation_time_s": simulation_time_s,
        "control_step": int(info.get("step", 0)),
        "controller_event": int(info.get("controller_event", -1)),
        "ee_position_world_m": ee_position.tolist(),
        "ee_orientation_wxyz": ee_orientation.tolist(),
        "arm_joint_positions_rad": np.asarray(positions, dtype=float)[
            indices
        ].tolist(),
        "arm_joint_velocities_rad_s": np.asarray(velocities, dtype=float)[
            indices
        ].tolist(),
    }


def _calibrate_nominal_corridor_bank(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    restoration: Mapping[str, Any],
    seed: int,
    simulation_app: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    arm_joint_indices: Sequence[int],
    stop_requested: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build severity corridors from replayed, time-aligned BC-only motion.

    Each phase is replayed from the accepted source layout.  At its trigger the
    task-event clock is held only through the cue, then advances normally from
    START through crossing/recovery while direct frozen BC drives RMPFlow with
    CBF disabled.  The sweep queries hypothetical hand paths against the moving
    protected PhysX geometry; actual tracked-hand gaps never define severity.
    """

    reference_cfg = dict(scenario["nominal_bc_reference"])
    crossing_cfg = dict(scenario["crossing"])
    layout_cfg = dict(scenario["layout_precheck"])
    offsets = _nominal_sweep_offsets(reference_cfg)
    sample_interval_s = float(reference_cfg["sweep_sample_interval_s"])
    maximum_sample_interval_s = float(
        reference_cfg["maximum_sweep_sample_interval_s"]
    )
    minimum_samples = int(reference_cfg["minimum_trajectory_samples"])
    if not (
        0.0 < sample_interval_s <= maximum_sample_interval_s
        and minimum_samples >= 2
    ):
        raise ValueError("invalid nominal trajectory sampling contract")
    trigger_events = {
        str(phase): int(event)
        for phase, event in protocol["task"]["crossing_trigger_event"].items()
    }
    saved_references = restoration.get("nominal_bc_phase_references", {})
    if not isinstance(saved_references, Mapping):
        raise RuntimeError("nominal phase references are unavailable")

    trajectories: dict[str, Any] = {}
    bank: dict[str, Any] = {}
    base_severity = dict(scenario["severity"]["shallow"])
    maximum_steps = int(layout_cfg["maximum_steps"])
    clear_gap_m = float(layout_cfg["hands_clear_gap_m"])

    for phase, trigger_event in trigger_events.items():
        stop_requested()
        obs, info = env.reset(
            seed=int(seed), source_restoration=dict(restoration)
        )
        obs, info, _tracking = _wait_for_tracking_and_clear(
            env=env,
            provider=provider,
            simulation_app=simulation_app,
            joint_count=len(env.robot.dof_names),
            timeout_s=60.0,
            stable_frames=int(layout_cfg["hands_clear_confirmation_frames"]),
            clear_gap_m=clear_gap_m,
            proximal_monitor=proximal_monitor,
            stop_requested=stop_requested,
        )
        reached = False
        for _ in range(maximum_steps):
            _require_clear_nominal_calibration_state(
                info=info,
                provider=provider,
                proximal_monitor=proximal_monitor,
                clear_gap_m=clear_gap_m,
            )
            if int(info.get("controller_event", -1)) == trigger_event:
                reached = True
                break
            action = policy.predict(np.asarray(obs, dtype=np.float32)).action
            obs, _reward, terminated, truncated, info = env.step(
                action, advance_task_phase=True
            )
            obs = np.asarray(obs, dtype=np.float32)
            if (terminated or truncated) and not bool(info.get("success", False)):
                break
        if not reached:
            raise RuntimeError(
                f"BC-only nominal calibration did not reach {phase} event "
                f"{trigger_event}"
            )

        trigger_reference = saved_references.get(phase, {})
        if not isinstance(trigger_reference, Mapping):
            raise RuntimeError(f"missing accepted trigger reference for {phase}")
        trigger_sample = _nominal_trajectory_sample(
            env=env,
            info=info,
            phase_start_simulation_time_s=float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            ),
            arm_joint_indices=arm_joint_indices,
        )
        reference_ee = np.asarray(
            trigger_reference.get("ee_position_world_m", ()), dtype=float
        ).reshape(-1)
        reference_joints = np.asarray(
            trigger_reference.get("arm_joint_positions_rad", ()), dtype=float
        ).reshape(-1)
        trigger_ee = np.asarray(
            trigger_sample["ee_position_world_m"], dtype=float
        )
        trigger_joints = np.asarray(
            trigger_sample["arm_joint_positions_rad"], dtype=float
        )
        if reference_ee.shape != (3,) or reference_joints.shape != (
            len(arm_joint_indices),
        ):
            raise RuntimeError(f"invalid accepted trigger reference for {phase}")
        if (
            float(np.linalg.norm(trigger_ee - reference_ee))
            > float(reference_cfg["maximum_ee_position_error_m"])
            or float(np.max(np.abs(trigger_joints - reference_joints)))
            > float(reference_cfg["maximum_arm_joint_error_rad"])
        ):
            raise RuntimeError(
                f"BC-only nominal calibration trigger drifted for {phase}"
            )

        phase_start = float(trigger_sample["simulation_time_s"])
        base_corridor = calibrate_surface_gap_corridor(
            env.safety_geometry,
            hand="left",
            end_effector_position_world_m=trigger_ee,
            target_gap_m=float(base_severity["target_gap_m"]),
            allowed_gap_range_m=base_severity["minimum_gap_range_m"],
            corridor_length_m=float(crossing_cfg["corridor_length_m"]),
        )
        sweeps: dict[tuple[str, str], NominalBCCorridorSweep] = {}
        for direction in ("left_to_right", "right_to_left"):
            hand = "left" if direction == "left_to_right" else "right"
            for speed_name in ("slow", "fast"):
                sweeps[(direction, speed_name)] = NominalBCCorridorSweep(
                    phase=phase,
                    base_corridor=base_corridor,
                    crossing_direction=direction,
                    speed_m_s=float(
                        crossing_cfg["speed_target_m_s"][speed_name]
                    ),
                    cue_lead_s=float(crossing_cfg["cue_lead_s"]),
                    corridor_length_m=float(crossing_cfg["corridor_length_m"]),
                    hand=hand,
                    candidate_normal_offsets_m=offsets,
                    phase_start_simulation_time_s=phase_start,
                    maximum_sample_interval_s=maximum_sample_interval_s,
                )
        maximum_end = max(
            sweep.crossing_end_simulation_time_s for sweep in sweeps.values()
        )
        crossing_start_time = min(
            sweep.crossing_start_simulation_time_s for sweep in sweeps.values()
        )
        samples: list[dict[str, Any]] = []
        last_sample_time = -math.inf
        calibration_steps = 0
        while True:
            stop_requested()
            _require_clear_nominal_calibration_state(
                info=info,
                provider=provider,
                proximal_monitor=proximal_monitor,
                clear_gap_m=clear_gap_m,
            )
            now = float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            )
            should_sample = bool(
                not samples
                or now - last_sample_time >= sample_interval_s - 1e-12
                or now >= maximum_end
            )
            if should_sample:
                samples.append(
                    _nominal_trajectory_sample(
                        env=env,
                        info=info,
                        phase_start_simulation_time_s=phase_start,
                        arm_joint_indices=arm_joint_indices,
                    )
                )
                for sweep in sweeps.values():
                    sweep.observe(
                        simulation_time_s=now,
                        safety_geometry=env.safety_geometry,
                    )
                last_sample_time = now
            if now >= maximum_end:
                break
            if calibration_steps >= maximum_steps:
                raise RuntimeError(
                    f"BC-only nominal calibration horizon exhausted for {phase}"
                )
            action = policy.predict(np.asarray(obs, dtype=np.float32)).action
            obs, _reward, _terminated, _truncated, info = env.step(
                action,
                advance_task_phase=bool(now >= crossing_start_time),
            )
            obs = np.asarray(obs, dtype=np.float32)
            calibration_steps += 1
            if (
                now < crossing_start_time
                and int(info.get("controller_event", -1)) != trigger_event
            ):
                raise RuntimeError(
                    f"cue-latched BC-only nominal calibration left {phase} event"
                )
            strict = _strict_state(info)
            if str(strict.get("failure_reason", "")):
                raise RuntimeError(
                    f"BC-only nominal calibration failed for {phase}: "
                    + str(strict["failure_reason"])
                )
        if len(samples) < minimum_samples:
            raise RuntimeError(
                f"BC-only nominal trajectory for {phase} has only "
                f"{len(samples)} samples"
            )
        trajectory_payload: dict[str, Any] = {
            "semantics": NOMINAL_BC_SWEEP_SEMANTICS,
            "phase": phase,
            "controller": "bc_only",
            "cbf_enabled": False,
            "cue_phase_clock_latched": True,
            "task_progression_during_crossing": True,
            "trigger_event": trigger_event,
            "sample_interval_target_s": sample_interval_s,
            "maximum_sample_interval_s": maximum_sample_interval_s,
            "samples": samples,
        }
        trajectory_payload["sha256"] = canonical_sha256(
            {key: value for key, value in trajectory_payload.items() if key != "sha256"}
        )
        trajectories[phase] = trajectory_payload

        phase_bank: dict[str, Any] = {}
        for (direction, speed_name), sweep in sweeps.items():
            direction_bank = phase_bank.setdefault(direction, {})
            speed_bank: dict[str, Any] = {}
            for severity_name in ("shallow", "threat"):
                severity_cfg = dict(scenario["severity"][severity_name])
                audit = sweep.select_severity(
                    severity=severity_name,
                    target_gap_m=float(severity_cfg["target_gap_m"]),
                    allowed_gap_range_m=severity_cfg["minimum_gap_range_m"],
                    minimum_pre_cue_start_gap_m=float(
                        reference_cfg["minimum_pre_cue_start_gap_m"]
                    ),
                ).as_dict()
                audit["trajectory_sha256"] = trajectory_payload["sha256"]
                audit["trajectory_sample_count"] = len(samples)
                audit["trigger_reference_control_step"] = int(
                    trigger_reference.get("control_step", -1)
                )
                audit["trigger_reference_simulation_time_s"] = float(
                    trigger_reference.get("simulation_time_s", -1.0)
                )
                audit["sweep_audit_sha256"] = canonical_sha256(
                    {"nominal_bc_sweep": audit}
                )
                speed_bank[severity_name] = audit
            direction_bank[speed_name] = speed_bank
        bank[phase] = phase_bank
        print(
            f"[OnlineFeedback][NominalSweep] phase={phase} "
            f"samples={len(samples)} conditions=8 "
            f"trajectory_sha256={trajectory_payload['sha256']}",
            flush=True,
        )
    return trajectories, bank


def _run_bc_only_precheck(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    recorder: OnlineExplicitFeedbackRecorder,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    simulation_app: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    arm_joint_indices: Sequence[int],
    stop_requested: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    cfg = dict(scenario["layout_precheck"])
    original_cbf = env._cbf_filter
    original_mode = env._physical_safety_mode
    env._cbf_filter = None
    env._physical_safety_mode = "none"
    try:
        for candidate_index in range(int(cfg["maximum_candidate_attempts"])):
            candidate_seed = int(seed + 10_000 + candidate_index)
            started_ns = time.monotonic_ns()
            obs, info = env.reset(seed=candidate_seed)
            obs, info, _tracking = _wait_for_tracking_and_clear(
                env=env,
                provider=provider,
                simulation_app=simulation_app,
                joint_count=len(env.robot.dof_names),
                timeout_s=60.0,
                stable_frames=int(cfg["hands_clear_confirmation_frames"]),
                clear_gap_m=float(cfg["hands_clear_gap_m"]),
                proximal_monitor=proximal_monitor,
                stop_requested=stop_requested,
            )
            scene, restoration, layout_id = _capture_scene_and_restoration(
                env, seed=candidate_seed
            )
            nominal_task_start_sim = float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            )
            nominal_previous_ee, _ = _ee_pose(env)
            nominal_ee_path_m = 0.0
            contaminated = False
            nominal_phase_references: dict[str, dict[str, Any]] = {}
            trigger_events = {
                str(phase): int(event)
                for phase, event in protocol["task"][
                    "crossing_trigger_event"
                ].items()
            }
            for _ in range(int(cfg["maximum_steps"])):
                stop_requested()
                snapshot = provider.observation_snapshot
                if snapshot is None:
                    raise CollectionAbort("BC precheck has no tracking snapshot")
                provider.watchdog.require_current(snapshot)
                event = int(info.get("controller_event", -1))
                for phase, trigger_event in trigger_events.items():
                    if (
                        event == trigger_event
                        and phase not in nominal_phase_references
                    ):
                        ee_position, _ee_orientation = _ee_pose(env)
                        joint_positions, _joint_velocities = measured_joint_state(
                            env.robot, joint_count=len(env.robot.dof_names)
                        )
                        nominal_phase_references[phase] = {
                            "phase": phase,
                            "controller_event": event,
                            "controller": "bc_only",
                            "ee_position_world_m": ee_position.tolist(),
                            "arm_joint_positions_rad": np.asarray(
                                joint_positions, dtype=float
                            )[np.asarray(arm_joint_indices, dtype=int)].tolist(),
                            "control_step": int(info.get("step", 0)),
                            "simulation_time_s": float(
                                info.get(
                                    "sim_time",
                                    getattr(env.world, "current_time", 0.0),
                                )
                            ),
                        }
                gaps = (
                    float(info.get("left_end_effector_surface_gap_m", -1.0)),
                    float(info.get("right_end_effector_surface_gap_m", -1.0)),
                )
                proximal_gaps = tuple(
                    proximal_monitor.closest_surface_gap_m(
                        getattr(snapshot, hand).position_world
                    )[0]
                    for hand in ("left", "right")
                )
                if (
                    not bool(info.get("geometry_valid", False))
                    or min(gaps) < float(cfg["hands_clear_gap_m"])
                    or min(proximal_gaps) < float(cfg["hands_clear_gap_m"])
                ):
                    contaminated = True
                    break
                action = policy.predict(np.asarray(obs, dtype=np.float32)).action
                obs, _reward, terminated, truncated, info = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                nominal_current_ee, _ = _ee_pose(env)
                nominal_ee_path_m += float(
                    np.linalg.norm(nominal_current_ee - nominal_previous_ee)
                )
                nominal_previous_ee = nominal_current_ee
                if terminated or truncated:
                    terminal_snapshot = provider.observation_snapshot
                    if terminal_snapshot is None:
                        raise CollectionAbort(
                            "BC precheck terminal step has no tracking snapshot"
                        )
                    provider.watchdog.require_current(terminal_snapshot)
                    terminal_distal_gaps = (
                        float(
                            info.get(
                                "left_end_effector_surface_gap_m", -1.0
                            )
                        ),
                        float(
                            info.get(
                                "right_end_effector_surface_gap_m", -1.0
                            )
                        ),
                    )
                    terminal_proximal_gaps = tuple(
                        proximal_monitor.closest_surface_gap_m(
                            getattr(terminal_snapshot, hand).position_world
                        )[0]
                        for hand in ("left", "right")
                    )
                    if (
                        not bool(info.get("geometry_valid", False))
                        or min(terminal_distal_gaps)
                        < float(cfg["hands_clear_gap_m"])
                        or min(terminal_proximal_gaps)
                        < float(cfg["hands_clear_gap_m"])
                    ):
                        contaminated = True
                    break
            state = _strict_state(info)
            strict_success = bool(
                info.get("success", False) and state.get("success_latched", False)
            )
            release_observed = bool(
                strict_success and int(info.get("controller_event", -1)) >= 7
            )
            references_complete = set(nominal_phase_references) == set(
                trigger_events
            )
            restoration = dict(restoration)
            restoration["nominal_bc_task_ee_path_m"] = float(
                nominal_ee_path_m
            )
            restoration["nominal_bc_task_completion_time_s"] = max(
                0.0,
                float(
                    info.get(
                        "sim_time", getattr(env.world, "current_time", 0.0)
                    )
                )
                - nominal_task_start_sim,
            )
            restoration["nominal_bc_phase_references"] = (
                nominal_phase_references
            )
            if contaminated:
                failure = "precheck_human_not_clear"
            elif not references_complete:
                failure = "precheck_nominal_phase_reference_incomplete:" + ",".join(
                    sorted(set(trigger_events) - set(nominal_phase_references))
                )
            elif not strict_success:
                terminal = str(info.get("task_terminal_reason", ""))
                failure = str(
                    state.get("failure_reason", "")
                    or ("" if terminal == "success" else terminal)
                    or "precheck_horizon"
                )
            else:
                failure = ""
            accepted = bool(
                strict_success
                and release_observed
                and references_complete
                and not contaminated
            )
            nominal_trajectories: dict[str, Any] = {}
            nominal_corridor_bank: dict[str, Any] = {}
            if accepted:
                try:
                    nominal_trajectories, nominal_corridor_bank = (
                        _calibrate_nominal_corridor_bank(
                            env=env,
                            policy=policy,
                            provider=provider,
                            protocol=protocol,
                            scenario=scenario,
                            restoration=restoration,
                            seed=candidate_seed,
                            simulation_app=simulation_app,
                            proximal_monitor=proximal_monitor,
                            arm_joint_indices=arm_joint_indices,
                            stop_requested=stop_requested,
                        )
                    )
                except (CollectionAbort, TrackingDropout):
                    raise
                except RuntimeError as error:
                    accepted = False
                    failure = (
                        "precheck_nominal_corridor_sweep_failed:"
                        f"{type(error).__name__}:{error}"
                    )
                else:
                    restoration["nominal_bc_phase_trajectories"] = (
                        nominal_trajectories
                    )
                    restoration["nominal_corridor_bank"] = (
                        nominal_corridor_bank
                    )
            corridor_bank_complete = bool(
                accepted
                and set(nominal_trajectories) == set(trigger_events)
                and set(nominal_corridor_bank) == set(trigger_events)
            )
            diagnostic_group = (
                "precheck_invalid_human_intrusion"
                if contaminated
                else (
                    cfg["accepted_group"]
                    if accepted
                    else (
                        "nominal_sweep_infeasible_diagnostic"
                        if strict_success and references_complete
                        else cfg["failed_group"]
                    )
                )
            )
            recorder.append_layout_precheck(
                {
                    "candidate_index": candidate_index,
                    "layout_id": layout_id,
                    "seed": candidate_seed,
                    "controller": "bc_only",
                    "policy_sha256": POLICY_SHA256,
                    "cbf_enabled": False,
                    "human_absent_or_clear": not contaminated,
                    "strict_success": strict_success,
                    "release_observed": release_observed,
                    "nominal_phase_references_complete": references_complete,
                    "nominal_phase_references": nominal_phase_references,
                    "nominal_corridor_bank_complete": corridor_bank_complete,
                    "nominal_phase_trajectories": nominal_trajectories,
                    "nominal_corridor_bank": nominal_corridor_bank,
                    "steps": int(info.get("step", 0)),
                    "diagnostic_group": diagnostic_group,
                    "failure_reason": failure,
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": time.monotonic_ns(),
                    "source_restoration": _jsonable(restoration),
                }
            )
            print(
                f"[OnlineFeedback][BCPrecheck] candidate={candidate_index} "
                f"layout={layout_id} success={strict_success} clear={not contaminated} "
                f"steps={int(info.get('step', 0))} reason={failure or 'passed'}",
                flush=True,
            )
            env.world.pause()
            apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
            if accepted:
                return scene, restoration, layout_id, candidate_index
            if contaminated:
                provider.wait_until_ready(
                    simulation_app,
                    timeout_s=60.0,
                    stable_frames=6,
                    stop_requested=stop_requested,
                )
        raise CollectionAbort(
            "no BC-feasible single-cube layout passed frozen BC-only precheck"
        )
    finally:
        env._cbf_filter = original_cbf
        env._physical_safety_mode = original_mode
        if original_cbf is not None:
            original_cbf.reset()


def _experimental_phase(event: int, protocol: Mapping[str, Any]) -> str:
    for phase, events in protocol["task"]["phase_event_map"].items():
        if int(event) in [int(value) for value in events]:
            return str(phase)
    return "place_release" if int(event) >= 8 else "reach_approach"


def _object_state(env: Any) -> dict[str, np.ndarray]:
    cube_position, cube_orientation = env.active_cube.get_world_pose()
    goal_position, goal_orientation = env.place_target.get_world_pose()
    cube_velocity = np.zeros(3, dtype=np.float64)
    velocity_getter = getattr(env.active_cube, "get_linear_velocity", None)
    if callable(velocity_getter):
        try:
            candidate = np.asarray(
                velocity_getter(), dtype=np.float64
            ).reshape(-1)
            if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                cube_velocity = candidate.copy()
        except Exception:
            pass
    return {
        "cube_position_world_m": np.asarray(cube_position, dtype=np.float64)[:3],
        "cube_orientation_wxyz": np.asarray(cube_orientation, dtype=np.float64)[:4],
        "cube_linear_velocity_world_m_s": cube_velocity,
        "goal_position_world_m": np.asarray(goal_position, dtype=np.float64)[:3],
        "goal_orientation_wxyz": np.asarray(goal_orientation, dtype=np.float64)[:4],
    }


def _ee_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = env.robot.end_effector.get_world_pose()
    return (
        np.asarray(position, dtype=np.float64)[:3],
        np.asarray(orientation, dtype=np.float64)[:4],
    )


def _ee_linear_velocity(env: Any) -> np.ndarray:
    getter = getattr(env.robot.end_effector, "get_linear_velocity", None)
    if callable(getter):
        try:
            value = np.asarray(getter(), dtype=np.float64).reshape(-1)
            if value.shape == (3,) and np.all(np.isfinite(value)):
                return value.copy()
        except Exception:
            pass
    return np.zeros(3, dtype=np.float64)


def _dynamic_values(info: Mapping[str, Any]) -> tuple[float, float, bool]:
    dynamic = info.get("dynamic_safety", {})
    dynamic = dict(dynamic) if isinstance(dynamic, Mapping) else {}
    return (
        float(dynamic.get("min_ttc_s", 10.0)),
        max(0.0, float(dynamic.get("max_closing_speed_mps", 0.0))),
        bool(dynamic.get("dynamic_measurement_valid", False)),
    )


def _recovery_is_active(info: Mapping[str, Any]) -> bool:
    payload = info.get("state_aware_recovery", {})
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    state = payload.get("state", {})
    state = dict(state) if isinstance(state, Mapping) else {}
    return bool(
        payload.get("control_authority", False)
        or state.get("active", False)
    )


def _decision_context(
    *,
    plan: TrialPlan,
    policy_output: Any,
    action_trace: Any,
    tracking: Any,
    info_before: Mapping[str, Any],
    object_state_before: Mapping[str, np.ndarray],
    positions_before: np.ndarray,
    velocities_before: np.ndarray,
    ee_position: np.ndarray,
    ee_orientation: np.ndarray,
    ee_velocity: np.ndarray,
    minimum_gap_m: float,
    ttc_s: float,
    closing_speed_m_s: float,
    arm_joint_indices: Sequence[int],
) -> dict[str, Any]:
    """Capture only information available at the first response candidate."""

    dynamic = info_before.get("dynamic_safety", {})
    dynamic = dict(dynamic) if isinstance(dynamic, Mapping) else {}
    strict = info_before.get("strict_task_semantics", {})
    strict = dict(strict) if isinstance(strict, Mapping) else {}
    strict_state = strict.get("state", {})
    strict_state = dict(strict_state) if isinstance(strict_state, Mapping) else {}
    hand_payload: dict[str, Any] = {}
    for hand in ("left", "right"):
        pose = getattr(tracking, hand)
        hand_payload[hand] = {
            "position_world_m": np.asarray(
                pose.position_world, dtype=np.float64
            )[:3],
            "velocity_world_m_s": np.asarray(
                dynamic.get(f"{hand}_hand_vel_filtered_mps", (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )[:3],
            "pose_valid": bool(pose.pose_valid),
        }
    indices = np.asarray(tuple(arm_joint_indices), dtype=np.int64)
    context = {
        "task_phase": plan.task_phase,
        "grasp_state": {
            "has_grasped": bool(info_before.get("has_grasped_cube", False)),
            "strict_event": int(strict_state.get("event", info_before.get("controller_event", -1))),
        },
        "attachment_state": {
            "attached": bool(
                info_before.get("cube_attached", info_before.get("has_grasped_cube", False))
            ),
            "gripper_command": str(info_before.get("gripper_command", "") or ""),
        },
        "robot_joint_state": {
            "positions_rad": np.asarray(positions_before, dtype=np.float64),
            "velocities_rad_s": np.asarray(velocities_before, dtype=np.float64),
        },
        "ee_pose_velocity": {
            "position_world_m": np.asarray(ee_position, dtype=np.float64),
            "orientation_wxyz": np.asarray(ee_orientation, dtype=np.float64),
            "linear_velocity_world_m_s": np.asarray(ee_velocity, dtype=np.float64),
        },
        "cube_pose": {
            "position_world_m": object_state_before[
                "cube_position_world_m"
            ],
            "orientation_wxyz": object_state_before[
                "cube_orientation_wxyz"
            ],
        },
        "goal_pose": {
            "position_world_m": object_state_before[
                "goal_position_world_m"
            ],
            "orientation_wxyz": object_state_before[
                "goal_orientation_wxyz"
            ],
        },
        "hand_pose_velocity": hand_payload,
        "surface_gap_m": float(minimum_gap_m),
        "ttc_s": float(ttc_s),
        "closing_speed_m_s": float(closing_speed_m_s),
        "active_candidate": {
            "crossing_hand": plan.crossing_hand,
            "robot_link": str(
                info_before.get(f"closest_link_{plan.crossing_hand}", "") or ""
            ),
            "collider_path": str(
                info_before.get(
                    f"closest_collider_{plan.crossing_hand}", ""
                )
                or ""
            ),
            "controller_event": int(info_before.get("controller_event", -1)),
        },
        "nominal_rmpflow_joint_command": {
            "joint_velocities_rad_s": action_trace.nominal_rmpflow.joint_velocities[
                indices
            ],
            "valid_mask": action_trace.nominal_rmpflow.joint_velocities_mask[
                indices
            ],
        },
        "bc_raw_action": np.asarray(policy_output.action, dtype=np.float32),
        "condition_id": plan.condition_id,
        "lambda_s": float(plan.lambda_s),
    }
    validate_decision_context(context)
    return _jsonable(context)


def _read_required_controller_inputs(
    *,
    source: Any,
    input_watchdog: Any,
    robot: Any,
    joint_count: int,
    sample_phase: str,
) -> Any:
    """Read a controller sample or enter a verified hold before aborting."""

    try:
        snapshot = source.read()
        return input_watchdog.require_available(snapshot)
    except Exception as error:
        apply_controlled_hold(robot, joint_count=joint_count)
        detail = str(error).strip() or type(error).__name__
        source_detail = str(getattr(source, "last_error", "") or "").strip()
        if source_detail and source_detail not in detail:
            detail = f"{detail}; {source_detail}"
        raise CollectionAbort(
            f"controller_input_unavailable_{sample_phase}: {detail}"
        ) from error


def _perform_logged_step(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    watchdog: TrackingWatchdog,
    trace: ActionTraceRecorder,
    recorder: OnlineExplicitFeedbackRecorder,
    feedback: Any,
    button_source: Any,
    controller_input_watchdog: Any,
    emergency_abort_monitor: Any,
    video: SynchronizedSpectatorVideoRecorder,
    plan: TrialPlan,
    participant_id: str,
    session_id: str,
    obs: np.ndarray,
    info: Mapping[str, Any],
    tracking: Any,
    arm_joint_indices: Sequence[int],
    control_mode: str,
    trial_state: str,
    advance_task_phase: bool,
    crossing_monitor: CrossingPathMonitor | None,
    detector: EncounterDetector | None,
    response_detector: SafetyResponsePhaseDetector | None,
    proximal_monitor: ProximalArmProtocolMonitor | None,
    protocol: Mapping[str, Any],
    stop_requested: Callable[[], None],
) -> LoggedStep:
    from v3_chan.rl.actions import zero_action

    stop_requested()
    preflight = provider.sample()
    preflight_ns = time.monotonic_ns()
    watchdog.require_current(preflight, now_monotonic_ns=preflight_ns)
    watchdog.require_current(tracking, now_monotonic_ns=preflight_ns)
    joint_count = len(env.robot.dof_names)
    pre_step_buttons = _read_required_controller_inputs(
        source=button_source,
        input_watchdog=controller_input_watchdog,
        robot=env.robot,
        joint_count=joint_count,
        sample_phase="pre_step",
    )
    # This pre-actuation sample is health/e-stop evidence only.  Rising-edge
    # processing remains owned by the single post-step sample below so a
    # marker or questionnaire action cannot be emitted twice.
    if emergency_abort_monitor.update(
        pre_step_buttons, monotonic_ns=time.monotonic_ns()
    ):
        apply_controlled_hold(env.robot, joint_count=joint_count)
        raise CollectionAbort("participant_emergency_abort_chord")
    positions_before, velocities_before = measured_joint_state(
        env.robot, joint_count=joint_count
    )
    ee_before, ee_orientation_before = _ee_pose(env)
    ee_velocity_before = _ee_linear_velocity(env)
    # Freeze every physical field used by the decision context before the
    # condition-dependent CBF solve and physics step.  Reading cube/goal state
    # later would mix a pre-solve robot/hand snapshot with post-A/C outcomes.
    object_state_before = _object_state(env)
    policy_output = policy.predict(np.asarray(obs, dtype=np.float32))
    controller_action = (
        policy_output.action if control_mode == "nominal_bc" else zero_action()
    )
    source_step = int(info.get("step", env.step_count))
    info_before = dict(info)
    trace.start_step(step_id=(plan.trial_index, source_step))
    try:
        obs_next, _reward, terminated, truncated, info_after = env.step(
            controller_action,
            advance_task_phase=bool(advance_task_phase),
        )
        action_trace = trace.finish_step()
    except Exception:
        trace.abort_step()
        apply_controlled_hold(env.robot, joint_count=joint_count)
        raise
    post_step_ns = time.monotonic_ns()
    wall_ns = time.time_ns()
    tracking_next = provider.observation_snapshot
    if tracking_next is None:
        raise CollectionAbort("control step produced no tracking provenance")
    watchdog.require_current(tracking_next, now_monotonic_ns=post_step_ns)
    positions_after, velocities_after = measured_joint_state(
        env.robot, joint_count=joint_count
    )
    validate_runtime_step(
        info_before=info_before,
        info_after=info_after,
        trace=action_trace,
        tracking=tracking,
        arm_joint_indices=arm_joint_indices,
    )
    ee_after, ee_orientation = _ee_pose(env)
    ee_velocity = (ee_after - ee_before) / max(float(env.physics_dt_s), 1e-9)
    simulation_time_s = float(
        info_after.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    cbf = info_after.get("physical_safety", {})
    cbf = dict(cbf) if isinstance(cbf, Mapping) else {}
    if str(cbf.get("objective_mode", "")) != plan.objective_mode:
        raise CollectionAbort("runtime CBF objective_mode changed within trial")
    runtime_weight = float(
        getattr(env._cbf_filter.config, "correction_smoothness_weight", math.nan)
    )
    if not math.isclose(
        runtime_weight, float(plan.lambda_s), rel_tol=0.0, abs_tol=1e-12
    ):
        raise CollectionAbort("runtime CBF lambda_s changed within trial")
    intervention_norm = float(cbf.get("intervention_norm_radps", 0.0))
    cbf_active = bool(cbf.get("active", False))
    smooth_tail_active = str(cbf.get("status", "")) == "smooth_intervention_tail"
    recovery_active = _recovery_is_active(info_after)
    minimum_gap = float(
        info_before.get("min_hand_end_effector_surface_gap", 10.0)
    )
    left_gap = float(info_before.get("left_end_effector_surface_gap_m", 10.0))
    right_gap = float(info_before.get("right_end_effector_surface_gap_m", 10.0))
    ttc_s, closing_speed, dynamic_measurement_valid = _dynamic_values(
        info_before
    )

    crossing_update = None
    proximal_by_hand: dict[str, tuple[float, str]] = {
        "left": (10.0, ""),
        "right": (10.0, ""),
    }
    crossing_payload = {
        "active": False,
        "state": "not_observed",
        "started_now": False,
        "intended_completed_now": False,
        "reverse_completed_now": False,
        "progress_fraction": 0.0,
        "perpendicular_deviation_m": 0.0,
        "intended_hand": plan.crossing_hand,
        "observed_hand": "",
        "runtime_feature_cutoff_available": detector is not None,
    }
    if crossing_monitor is not None:
        pose = getattr(tracking_next, plan.crossing_hand)
        valid_index = 1 if plan.crossing_hand == "left" else 2
        valid = bool(tracking_next.valid_mask[valid_index])
        crossing_update = crossing_monitor.observe(
            position_world=pose.position_world if valid else None,
            simulation_time_s=simulation_time_s,
            valid=valid,
        )
        crossing_payload.update(
            {
                "active": crossing_update.state
                in {"crossing", "completed", "extra_traversal"},
                "state": crossing_update.state,
                "started_now": crossing_update.crossing_started,
                "intended_completed_now": (
                    crossing_update.intended_crossing_completed
                ),
                "reverse_completed_now": (
                    crossing_update.reverse_crossing_completed
                ),
                "progress_fraction": 0.0
                if crossing_update.progress is None
                else crossing_update.progress,
                "perpendicular_deviation_m": 0.0
                if crossing_update.path_deviation_m is None
                else crossing_update.path_deviation_m,
                "observed_hand": plan.crossing_hand if valid else "",
            }
        )
    if proximal_monitor is not None:
        for hand, valid_index in (("left", 1), ("right", 2)):
            if bool(tracking_next.valid_mask[valid_index]):
                pose = getattr(tracking_next, hand)
                proximal_by_hand[hand] = proximal_monitor.closest_surface_gap_m(
                    pose.position_world
                )
    proximal_hand = min(
        proximal_by_hand, key=lambda hand: proximal_by_hand[hand][0]
    )
    proximal_gap, proximal_path = proximal_by_hand[proximal_hand]

    encounter_update = None
    if detector is not None:
        encounter_update = detector.observe(
            step=source_step,
            simulation_time_s=simulation_time_s,
            surface_gap_m=minimum_gap,
            ttc_s=ttc_s,
            closing=closing_speed > 0.0,
            cbf_delta_norm_radps=intervention_norm,
            cbf_active=cbf_active,
            dynamic_measurement_valid=dynamic_measurement_valid,
        )
    encounter_id = (
        ""
        if encounter_update is None or not encounter_update.row_encounter_id
        else str(plan.encounter_id)
    )
    encounter_state = (
        "not_observed" if encounter_update is None else encounter_update.state
    )

    response_update = None
    response_phase = ResponsePhase.PRE_RESPONSE.value
    decision_context = None
    if response_detector is not None:
        response_update = response_detector.update(
            control_step=source_step,
            simulation_time_s=simulation_time_s,
            cbf_constraint_active=cbf_active,
            intervention_norm_rad_s=intervention_norm,
            smooth_tail_active=smooth_tail_active,
            recovery_active=recovery_active,
            bc_resumed=bool(
                control_mode == "nominal_bc"
                and not (cbf_active or smooth_tail_active or recovery_active)
            ),
        )
        response_phase = response_update.phase.value
        if response_update.onset_candidate_now:
            decision_context = _decision_context(
                plan=plan,
                policy_output=policy_output,
                action_trace=action_trace,
                tracking=tracking,
                info_before=info_before,
                object_state_before=object_state_before,
                positions_before=positions_before,
                velocities_before=velocities_before,
                ee_position=ee_before,
                ee_orientation=ee_orientation_before,
                ee_velocity=ee_velocity_before,
                minimum_gap_m=minimum_gap,
                ttc_s=ttc_s,
                closing_speed_m_s=closing_speed,
                arm_joint_indices=arm_joint_indices,
            )

    button_snapshot = _read_required_controller_inputs(
        source=button_source,
        input_watchdog=controller_input_watchdog,
        robot=env.robot,
        joint_count=joint_count,
        sample_phase="post_step",
    )
    if emergency_abort_monitor.update(
        button_snapshot, monotonic_ns=post_step_ns
    ):
        apply_controlled_hold(env.robot, joint_count=joint_count)
        raise CollectionAbort("participant_emergency_abort_chord")
    feedback_update = feedback.update(
        button_snapshot,
        _feedback_clocks(
            env, monotonic_ns=post_step_ns, unix_ns=wall_ns
        ),
    )
    for marker in feedback_update.realtime_markers:
        recorder.append_marker(
            RealtimeMarkerRecord(
                marker_id=marker.marker_id,
                trial_id=plan.trial_id,
                participant_id=str(participant_id),
                session_id=str(session_id),
                encounter_id=str(getattr(plan, "encounter_id", "")),
                condition_id=plan.condition_id,
                lambda_s=float(plan.lambda_s),
                marker_type=marker.marker_type,
                simulation_time_s=marker.sim_time_s,
                monotonic_ns=marker.monotonic_ns,
                unix_ns=marker.unix_ns,
                control_step=marker.control_step,
                response_phase=response_phase,
                time_since_response_onset_s=(
                    -1.0
                    if response_detector is None
                    or response_detector.onset_confirmed_step < 0
                    else max(
                        0.0,
                        marker.sim_time_s
                        - response_detector.onset_simulation_time_s,
                    )
                ),
                time_since_recovery_onset_s=(
                    -1.0
                    if response_detector is None
                    or response_detector.recovery_onset_step < 0
                    else max(
                        0.0,
                        marker.sim_time_s
                        - response_detector.recovery_onset_simulation_time_s,
                    )
                ),
                controller_hand=marker.controller_hand,
                button=marker.button,
                button_identifier=(
                    f"{marker.controller_hand}.{marker.button}"
                ),
            )
        )

    experimental_phase = _experimental_phase(
        int(info_before.get("controller_event", -1)), protocol
    )
    row = build_online_transition_row(
        source_step=source_step,
        observation_monotonic_ns=int(tracking.sample_monotonic_ns),
        preflight_monotonic_ns=preflight_ns,
        wall_time_unix_ns=wall_ns,
        obs_raw=np.asarray(obs, dtype=np.float32),
        obs_next=np.asarray(obs_next, dtype=np.float32),
        policy_output=policy_output,
        trace=action_trace,
        tracking=tracking,
        tracking_next=tracking_next,
        measured_positions_before=positions_before,
        measured_velocities_before=velocities_before,
        measured_positions_after=positions_after,
        measured_velocities_after=velocities_after,
        info_before=info_before,
        info_after=info_after,
        terminated=bool(terminated),
        truncated=bool(truncated),
        post_step_monotonic_ns=post_step_ns,
        arm_joint_indices=arm_joint_indices,
        encounter_id=encounter_id,
        encounter_state=encounter_state,
        controller_task_action=controller_action,
        controller_task_action_reason=(
            "frozen_bc_policy"
            if control_mode == "nominal_bc"
            else "current_ee_protocol_hold"
        ),
        control_mode=control_mode,
        trial_state=trial_state,
        trial_id=plan.trial_id,
        query_id=plan.query_id,
        experimental_phase=experimental_phase,
        phase_advance_enabled=bool(advance_task_phase),
        object_state=_object_state(env),
        ee_state={
            "position_world_m": ee_after,
            "orientation_wxyz": ee_orientation,
            "linear_velocity_world_m_s": ee_velocity,
        },
        crossing_state=crossing_payload,
        proximal_state={
            "left_gap_m": proximal_by_hand["left"][0],
            "right_gap_m": proximal_by_hand["right"][0],
            "left_collider_path": proximal_by_hand["left"][1],
            "right_collider_path": proximal_by_hand["right"][1],
        },
        condition_id=plan.condition_id,
        objective_mode=plan.objective_mode,
        lambda_s=float(plan.lambda_s),
        response_phase=response_phase,
        decision_context_available=decision_context is not None,
        response_onset_simulation_time_s=(
            None
            if response_detector is None
            or response_detector.onset_confirmed_step < 0
            else response_detector.onset_simulation_time_s
        ),
        recovery_onset_simulation_time_s=(
            None
            if response_detector is None
            or response_detector.recovery_onset_step < 0
            else response_detector.recovery_onset_simulation_time_s
        ),
        recovery_onset_now=bool(
            response_update is not None
            and response_update.recovery_onset_now
        ),
        recovery_end_now=bool(
            response_update is not None
            and response_update.recovery_end_now
        ),
    )
    recorder.append_transition(row)
    video.capture(
        simulation_time_s,
        source_step,
        plan.trial_id,
    )
    return LoggedStep(
        obs=np.asarray(obs_next, dtype=np.float32),
        info=dict(info_after),
        tracking=tracking_next,
        terminated=bool(terminated),
        truncated=bool(truncated),
        source_step=source_step,
        result_step=int(info_after.get("step", source_step + 1)),
        simulation_time_s=simulation_time_s,
        monotonic_ns=post_step_ns,
        unix_ns=wall_ns,
        minimum_gap_m=minimum_gap,
        left_gap_m=left_gap,
        right_gap_m=right_gap,
        ttc_s=ttc_s,
        closing_speed_m_s=closing_speed,
        dynamic_measurement_valid=dynamic_measurement_valid,
        cbf_active=cbf_active,
        intervention_norm_rad_s=intervention_norm,
        crossing_update=crossing_update,
        encounter_update=encounter_update,
        questionnaire_result=feedback_update.questionnaire_result,
        realtime_markers=feedback_update.realtime_markers,
        proximal_gap_m=proximal_gap,
        proximal_collider_path=proximal_path,
        left_proximal_gap_m=proximal_by_hand["left"][0],
        right_proximal_gap_m=proximal_by_hand["right"][0],
        left_proximal_collider_path=proximal_by_hand["left"][1],
        right_proximal_collider_path=proximal_by_hand["right"][1],
        response_phase=response_phase,
        decision_context=decision_context,
        response_update=response_update,
        recovery_active=recovery_active,
        smooth_tail_active=smooth_tail_active,
        correction_arm_rad_s=(
            action_trace.cbf_filtered.joint_velocities[
                np.asarray(tuple(arm_joint_indices), dtype=np.int64)
            ]
            - action_trace.nominal_rmpflow.joint_velocities[
                np.asarray(tuple(arm_joint_indices), dtype=np.int64)
            ]
        ),
        measured_arm_velocity_rad_s=np.asarray(velocities_after, dtype=np.float64)[
            np.asarray(tuple(arm_joint_indices), dtype=np.int64)
        ],
        ee_position_world_m=ee_after,
        ee_linear_velocity_world_m_s=ee_velocity,
    )


def _oriented_corridor(
    corridor: CrossingCorridor, direction: str
) -> CrossingCorridor:
    # Avatar/world convention: participant left is +Y and right is -Y.
    # Calibration's canonical tangent is +Y, hence its base start(-Y)->end(+Y)
    # is right-to-left.
    if direction == "right_to_left":
        return corridor
    if direction == "left_to_right":
        return replace(
            corridor,
            start_world_m=corridor.end_world_m.copy(),
            end_world_m=corridor.start_world_m.copy(),
            tangent_world=-corridor.tangent_world.copy(),
        )
    raise ValueError(f"unsupported crossing direction: {direction}")


def _corridor_from_mapping(value: Mapping[str, Any]) -> CrossingCorridor:
    semantics = str(value.get("geometry_query_semantics", ""))
    if semantics != "physx_protected_surface_to_hand_sphere":
        raise CollectionAbort("nominal corridor has unknown geometry semantics")
    corridor = CrossingCorridor(
        start_world_m=np.asarray(value.get("start_world_m", ()), dtype=float),
        end_world_m=np.asarray(value.get("end_world_m", ()), dtype=float),
        center_world_m=np.asarray(value.get("center_world_m", ()), dtype=float),
        tangent_world=np.asarray(value.get("tangent_world", ()), dtype=float),
        outward_normal_world=np.asarray(
            value.get("outward_normal_world", ()), dtype=float
        ),
        planned_minimum_surface_gap_m=float(
            value.get("planned_minimum_surface_gap_m", math.nan)
        ),
        closest_link=str(value.get("closest_link", "")),
        closest_collider_path=str(value.get("closest_collider_path", "")),
        calibration_iterations=int(value.get("calibration_iterations", 0)),
        geometry_query_semantics=semantics,
    )
    vectors = (
        corridor.start_world_m,
        corridor.end_world_m,
        corridor.center_world_m,
        corridor.tangent_world,
        corridor.outward_normal_world,
    )
    if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors):
        raise CollectionAbort("nominal corridor contains an invalid 3-vector")
    if (
        not math.isfinite(corridor.planned_minimum_surface_gap_m)
        or not corridor.closest_link
        or not corridor.closest_collider_path
        or corridor.calibration_iterations < 1
    ):
        raise CollectionAbort("nominal corridor provenance is incomplete")
    return corridor


def _nearest_nominal_trajectory_sample(
    trajectory: Mapping[str, Any], relative_time_s: float
) -> tuple[Mapping[str, Any], float]:
    samples = trajectory.get("samples", ())
    if not isinstance(samples, list) or not samples:
        raise CollectionAbort("nominal BC trajectory samples are unavailable")
    usable = [sample for sample in samples if isinstance(sample, Mapping)]
    if len(usable) != len(samples):
        raise CollectionAbort("nominal BC trajectory contains a malformed sample")
    expected = min(
        usable,
        key=lambda sample: abs(
            float(sample.get("relative_time_s", math.inf))
            - float(relative_time_s)
        ),
    )
    error = abs(
        float(expected.get("relative_time_s", math.inf))
        - float(relative_time_s)
    )
    if not math.isfinite(error):
        raise CollectionAbort("nominal BC trajectory time is invalid")
    return expected, error


def _nominal_state_errors(
    *,
    env: Any,
    expected: Mapping[str, Any],
    arm_joint_indices: Sequence[int],
) -> tuple[float, float]:
    ee_position, _orientation = _ee_pose(env)
    positions, _velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    expected_ee = np.asarray(
        expected.get("ee_position_world_m", ()), dtype=float
    ).reshape(-1)
    expected_arm = np.asarray(
        expected.get("arm_joint_positions_rad", ()), dtype=float
    ).reshape(-1)
    actual_arm = np.asarray(positions, dtype=float)[
        np.asarray(arm_joint_indices, dtype=int)
    ]
    if expected_ee.shape != (3,) or expected_arm.shape != actual_arm.shape:
        raise CollectionAbort("nominal BC trajectory state has invalid dimensions")
    if not np.all(np.isfinite(expected_ee)) or not np.all(np.isfinite(expected_arm)):
        raise CollectionAbort("nominal BC trajectory state is non-finite")
    return (
        float(np.linalg.norm(ee_position - expected_ee)),
        float(np.max(np.abs(actual_arm - expected_arm))),
    )


def _convert_encounter(
    record: Any,
    clocks_by_step: Mapping[int, tuple[float, int]],
    *,
    encounter_id: str,
) -> EncounterRecordV1:
    observed = sorted(
        (step, sim, mono) for step, (sim, mono) in clocks_by_step.items()
    )
    if not observed:
        raise RuntimeError("cannot serialize encounter without recorded clocks")

    def mono_for(step: int) -> int:
        exact = [item for item in observed if item[0] == step]
        return int(
            (exact[0] if exact else min(observed, key=lambda item: abs(item[0] - step)))[2]
        )

    window_steps = [
        step
        for step, sim, _ in observed
        if record.window_start_simulation_time_s
        <= sim
        <= record.window_end_simulation_time_s
    ]
    window_start = (
        min(window_steps) if window_steps else max(0, int(record.risk_onset_step))
    )
    window_end = (
        max(window_steps) + 1
        if window_steps
        else int(record.end_step_exclusive)
    )
    return EncounterRecordV1(
        encounter_id=str(encounter_id),
        trial_id=record.trial_id,
        onset_step=int(record.risk_onset_step),
        onset_confirmed_step=int(record.onset_confirmed_step),
        offset_step_exclusive=int(record.end_step_exclusive),
        onset_simulation_time_s=float(record.risk_onset_simulation_time_s),
        offset_simulation_time_s=float(record.offset_simulation_time_s),
        onset_monotonic_ns=mono_for(int(record.risk_onset_step)),
        offset_monotonic_ns=mono_for(int(record.offset_step)),
        window_start_step=int(window_start),
        window_end_step_exclusive=int(window_end),
        minimum_surface_gap_m=float(record.minimum_surface_gap_m),
        maximum_intervention_norm_rad_s=float(
            record.maximum_cbf_delta_norm_radps
        ),
        risk_onset_step=int(record.risk_onset_step),
        cbf_intervention_start_step=(
            -1
            if record.cbf_intervention_start_step is None
            else int(record.cbf_intervention_start_step)
        ),
        cbf_intervention_confirmed_step=(
            -1
            if record.cbf_intervention_confirmed_step is None
            else int(record.cbf_intervention_confirmed_step)
        ),
        encounter_timeout=bool(record.timeout),
        merged_reentry_count=int(record.reentry_merge_count),
    )


def _query_record(
    plan: TrialPlan, result: Any, encounter_id: str
) -> QueryRecord:
    completed = bool(result.study_complete)
    status = (
        "completed"
        if completed
        else ("timeout" if result.completion_status == "timed_out" else "no_response")
    )
    return QueryRecord(
        query_id=plan.query_id,
        trial_id=plan.trial_id,
        encounter_id=str(encounter_id or ""),
        issued_simulation_time_s=float(result.started_sim_time_s),
        issued_monotonic_ns=int(result.started_monotonic_ns),
        issued_unix_ns=int(result.started_unix_ns),
        issued_control_step=int(result.started_control_step),
        completed_simulation_time_s=float(result.completed_sim_time_s),
        completed_monotonic_ns=int(result.completed_monotonic_ns),
        completed_unix_ns=int(result.completed_unix_ns),
        completed_control_step=int(result.completed_control_step),
        response_status=status,
        response_disposition=str(result.response_disposition),
        q1_response=str(result.q1_response or "") if completed else "",
        q2_perceived_danger=int(result.q2 or 0) if completed else 0,
        q3_abruptness=int(result.q3 or 0) if completed else 0,
        q4_excessive_duration=int(result.q4 or 0) if completed else 0,
        q5_task_disruption=int(result.q5 or 0) if completed else 0,
        q6_confidence=int(result.q6 or 0) if completed else 0,
        modification_reasons=(
            tuple(result.rejection_reason_ids or ()) if completed else ()
        ),
        response_latency_ms=(
            int(result.completed_monotonic_ns) - int(result.started_monotonic_ns)
        )
        / 1e6,
        input_device="vr_controller",
        first_input_simulation_time_s=float(result.first_input_sim_time_s),
        first_input_monotonic_ns=int(result.first_input_monotonic_ns),
        first_input_unix_ns=int(result.first_input_unix_ns),
        first_input_control_step=int(result.first_input_control_step),
        back_correction_count=int(result.back_correction_count),
        accidental_input_count=int(result.accidental_input_count),
    )


def _set_instruction(feedback: Any, text: str) -> None:
    setter = getattr(feedback, "set_instruction", None)
    if callable(setter):
        setter(text)


def _run_trial(
    *,
    spec: TrialSpec,
    plan: TrialPlan,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    watchdog: TrackingWatchdog,
    trace: ActionTraceRecorder,
    recorder: OnlineExplicitFeedbackRecorder,
    feedback: Any,
    button_source: Any,
    controller_input_watchdog: Any,
    emergency_abort_monitor: Any,
    video: SynchronizedSpectatorVideoRecorder,
    cue_visuals: CrossingCueVisuals,
    proximal_monitor: ProximalArmProtocolMonitor,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    restoration: Mapping[str, Any],
    initial_scene: Mapping[str, Any],
    layout_id: str,
    layout_precheck_index: int,
    arm_joint_indices: Sequence[int],
    simulation_app: Any,
    args: argparse.Namespace,
    stop_requested: Callable[[], None],
) -> None:
    lifecycle = TrialLifecycle(spec.condition_id)
    selected_cbf_config = _apply_trial_condition(env, spec)
    lifecycle.advance(TrialState.LOAD_BC_FEASIBLE_SCENARIO)
    trial_seed = int(args.seed + plan.trial_index)
    obs, info = env.reset(
        seed=trial_seed, source_restoration=dict(restoration)
    )
    actual_after_reset = actual_cbf_config(
        env._cbf_filter, condition_id=spec.condition_id
    )
    if actual_after_reset != selected_cbf_config:
        raise CollectionAbort("trial RESET changed the assigned A/C configuration")
    # A world reset can restore the cue prim poses without resetting this
    # wrapper's Python state.  Synchronize both sides explicitly.
    cue_visuals.hide()
    obs, info, tracking = _wait_for_tracking_and_clear(
        env=env,
        provider=provider,
        simulation_app=simulation_app,
        joint_count=len(env.robot.dof_names),
        timeout_s=args.tracking_timeout_s,
        stable_frames=args.tracking_stable_frames,
        clear_gap_m=float(
            scenario["layout_precheck"]["hands_clear_gap_m"]
        ),
        proximal_monitor=proximal_monitor,
        stop_requested=stop_requested,
    )
    start_sim = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    start_mono, start_unix = time.monotonic_ns(), time.time_ns()
    lifecycle.advance(TrialState.START_FROZEN_BC)
    lifecycle.advance(TrialState.WAIT_FOR_TARGET_TASK_PHASE)
    recorder.start_trial(
        plan,
        seed=trial_seed,
        layout_id=layout_id,
        layout_precheck_index=layout_precheck_index,
        planned_gap_min_m=spec.planned_gap_min_m,
        planned_gap_max_m=spec.planned_gap_max_m,
        planned_target_gap_m=spec.planned_gap_target_m,
        start_simulation_time_s=start_sim,
        start_monotonic_ns=start_mono,
        start_unix_ns=start_unix,
        initial_scene=initial_scene,
        condition_config={
            "condition_id": plan.condition_id,
            "objective_mode": plan.objective_mode,
            "lambda_s": float(plan.lambda_s),
        },
        actual_cbf_config=actual_after_reset,
    )
    dismiss = getattr(feedback, "dismiss_questionnaire", None)
    if callable(dismiss):
        dismiss()
    feedback.set_crossing_hand(None)
    trigger_event = int(
        protocol["task"]["crossing_trigger_event"][plan.task_phase]
    )
    crossing_cfg = dict(scenario["crossing"])
    staging_gap_m = float(scenario["layout_precheck"]["hands_clear_gap_m"])
    encounter_cfg = dict(protocol["encounter"])
    detector = EncounterDetector(
        EncounterDetectorConfig(
            activation_gap_m=float(encounter_cfg["activation_gap_m"]),
            onset_ttc_s=float(encounter_cfg["ttc_onset_s"]),
            intervention_norm_radps=float(
                encounter_cfg["intervention_onset_rad_s"]
            ),
            onset_confirmation_frames=int(
                encounter_cfg["onset_confirmation_frames"]
            ),
            intervention_confirmation_frames=3,
            clear_gap_m=float(encounter_cfg["clear_gap_m"]),
            clear_ttc_s=float(encounter_cfg["ttc_clear_s"]),
            clear_norm_radps=float(
                encounter_cfg["intervention_clear_rad_s"]
            ),
            clear_duration_s=float(encounter_cfg["clear_duration_s"]),
            reentry_merge_s=float(encounter_cfg["merge_reentry_s"]),
            timeout_s=float(encounter_cfg["timeout_s"]),
            pre_window_s=float(encounter_cfg["window_pre_s"]),
            post_window_s=float(encounter_cfg["window_post_s"]),
        ),
        trial_id=plan.trial_id,
    )
    response_cfg = protocol["response_episode"]
    response_detector = SafetyResponsePhaseDetector(
        ResponsePhaseConfig(
            onset_intervention_rad_s=float(
                response_cfg["onset_intervention_rad_s"]
            ),
            onset_confirmation_frames=int(
                response_cfg["onset_confirmation_frames"]
            ),
            stable_intervention_rad_s=float(
                response_cfg["stable_intervention_rad_s"]
            ),
            stable_duration_s=float(response_cfg["stable_duration_s"]),
        )
    )
    crossing_monitor: CrossingPathMonitor | None = None
    corridor: CrossingCorridor | None = None
    nominal_reference_audit: dict[str, Any] | None = None
    nominal_sweep_audit: dict[str, Any] | None = None
    nominal_trajectory: Mapping[str, Any] | None = None
    start_recheck_audit: dict[str, Any] | None = None
    nominal_timeline_audit: dict[str, Any] = {
        "semantics": (
            "pre_cbf_actual_vs_bc_only_cue_latched_then_task_advancing_trajectory_v1"
        ),
        "compared_sample_count": 0,
        "maximum_time_alignment_error_s": 0.0,
        "maximum_ee_position_error_m": 0.0,
        "maximum_arm_joint_error_rad": 0.0,
        "comparison_closed_at_cbf_step": -1,
        "match": True,
    }
    core_encounter = None
    state = "wait_phase"
    wait_phase_row_recorded = False
    trigger_event_latched = False
    cue_started_sim = math.inf
    crossing_completed_sim = math.inf
    crossing_started_sim = math.inf
    recovery_started_sim = math.inf
    query_started = False
    query_result = None
    pending_decision_context: dict[str, Any] | None = None
    frozen_decision_context: dict[str, Any] | None = None
    response_phase_durations = {
        phase.value: 0.0 for phase in ResponsePhase
    }
    response_phase_history: list[dict[str, Any]] = [
        {
            "phase": ResponsePhase.PRE_RESPONSE.value,
            "control_step": int(info.get("step", 0)),
            "simulation_time_s": start_sim,
        }
    ]
    previous_response_phase = ResponsePhase.PRE_RESPONSE.value
    # Buffer the still-unconfirmed onset run.  Once its third consecutive
    # qualifying frame confirms the episode, these first two frames belong to
    # the response window as well; accumulating only after confirmation would
    # systematically understate both outcome metrics.
    response_metric_samples: list[
        tuple[int, float, np.ndarray, bool, str]
    ] = []
    correction_total_variation_rad_s = 0.0
    integrated_intervention_rad = 0.0
    response_window_closed_step = -1
    response_episode_timed_out = False
    response_terminal_update: Any = None
    neutral_clear_frames = 0
    previous_arm_velocity: np.ndarray | None = None
    previous_arm_acceleration: np.ndarray | None = None
    previous_ee_velocity: np.ndarray | None = None
    previous_ee_acceleration: np.ndarray | None = None
    jerk_windows = (
        "PRE_RESPONSE",
        "CBF_ACTIVE",
        "SMOOTH_TAIL",
        "RECOVERY_ACTIVE",
        "BC_RESUMED_FIRST_0_5_S",
        "WHOLE_TASK",
    )
    joint_jerk_by_window: dict[str, list[float]] = {
        name: [] for name in jerk_windows
    }
    ee_jerk_by_window: dict[str, list[float]] = {
        name: [] for name in jerk_windows
    }
    bc_resumed_started_sim = math.inf
    actual_task_ee_path_m = 0.0
    previous_ee_position, _ = _ee_pose(env)
    tracked_step_count = 0
    valid_tracking_step_count = 0
    task_active_elapsed_s = 0.0
    task_completion_active_elapsed_s: float | None = None
    task_success_observed = bool(info.get("success", False))
    task_terminal_observed = False
    task_failure_reason_observed = ""
    task_terminal_kind_observed = ""
    clocks_by_step: dict[int, tuple[float, int]] = {}
    off_protocol_reasons: set[str] = set()
    minimum_gap = math.inf
    minimum_any_hand_gap = math.inf
    minimum_proximal_gap = math.inf
    closest_proximal_collider = ""
    cbf_any_active = False
    cbf_max_norm = 0.0
    intervention_duration = 0.0
    intervention_count = 0
    intervention_prior = False
    marker_counts = {
        "realtime_safety_concern": 0,
        "realtime_behavior_anomaly": 0,
    }
    _set_instruction(
        feedback,
        f"Trial {plan.trial_index + 1}: 로봇 작업을 관찰하세요. 손은 staging 위치에 유지하세요.",
    )

    while True:
        stop_requested()
        event = int(info.get("controller_event", -1))
        now_sim = float(
            info.get("sim_time", getattr(env.world, "current_time", 0.0))
        )
        if (
            state == "wait_phase"
            and wait_phase_row_recorded
            and (trigger_event_latched or event == trigger_event)
        ):
            lifecycle.advance(TrialState.SHOW_HAND_CROSSING_CUE)
            ee_position, _ = _ee_pose(env)
            references = restoration.get("nominal_bc_phase_references", {})
            trajectories = restoration.get(
                "nominal_bc_phase_trajectories", {}
            )
            corridor_bank = restoration.get("nominal_corridor_bank", {})
            reference = (
                references.get(plan.task_phase, {})
                if isinstance(references, Mapping)
                else {}
            )
            try:
                nominal_trajectory = trajectories[plan.task_phase]
                nominal_sweep_audit = dict(
                    corridor_bank[plan.task_phase][plan.direction][plan.speed][
                        plan.severity
                    ]
                )
                if not isinstance(nominal_trajectory, Mapping):
                    raise TypeError("trajectory is not a mapping")
                trajectory_hash = str(nominal_trajectory["sha256"])
                trajectory_body = {
                    key: value
                    for key, value in nominal_trajectory.items()
                    if key != "sha256"
                }
                if canonical_sha256(trajectory_body) != trajectory_hash:
                    raise ValueError("trajectory SHA-256 mismatch")
                sweep_hash = str(nominal_sweep_audit["sweep_audit_sha256"])
                sweep_body = {
                    key: value
                    for key, value in nominal_sweep_audit.items()
                    if key != "sweep_audit_sha256"
                }
                if canonical_sha256(
                    {"nominal_bc_sweep": sweep_body}
                ) != sweep_hash:
                    raise ValueError("sweep audit SHA-256 mismatch")
                if str(nominal_sweep_audit["trajectory_sha256"]) != trajectory_hash:
                    raise ValueError("sweep/trajectory relation mismatch")
                corridor = _corridor_from_mapping(
                    nominal_sweep_audit["corridor"]
                )
                reference_ee = np.asarray(
                    reference["ee_position_world_m"], dtype=float
                ).reshape(3)
                reference_joints = np.asarray(
                    reference["arm_joint_positions_rad"], dtype=float
                ).reshape(len(arm_joint_indices))
                current_positions, _current_velocities = measured_joint_state(
                    env.robot, joint_count=len(env.robot.dof_names)
                )
                current_arm = np.asarray(current_positions, dtype=float)[
                    np.asarray(arm_joint_indices, dtype=int)
                ]
                ee_error = float(np.linalg.norm(ee_position - reference_ee))
                joint_error = float(
                    np.max(np.abs(current_arm - reference_joints))
                )
            except (KeyError, TypeError, ValueError) as error:
                raise CollectionAbort(
                    "accepted layout lacks a valid time-aligned BC-only "
                    "nominal corridor/reference"
                ) from error
            reference_cfg = scenario["nominal_bc_reference"]
            expected_range = [
                float(spec.planned_gap_min_m),
                float(spec.planned_gap_max_m),
            ]
            sweep_exact = bool(
                nominal_sweep_audit.get("semantics")
                == NOMINAL_BC_SWEEP_SEMANTICS
                and nominal_sweep_audit.get("phase") == plan.task_phase
                and nominal_sweep_audit.get("severity") == plan.severity
                and nominal_sweep_audit.get("hand") == plan.crossing_hand
                and nominal_sweep_audit.get("crossing_direction")
                == plan.direction
                and math.isclose(
                    float(nominal_sweep_audit.get("speed_m_s", math.nan)),
                    float(crossing_cfg["speed_target_m_s"][plan.speed]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and nominal_sweep_audit.get("allowed_gap_range_m")
                == expected_range
                and math.isclose(
                    float(
                        nominal_sweep_audit.get("target_gap_m", math.nan)
                    ),
                    float(spec.planned_gap_target_m),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and bool(nominal_sweep_audit.get("geometry_valid", False))
                and float(spec.planned_gap_min_m)
                <= corridor.planned_minimum_surface_gap_m
                <= float(spec.planned_gap_max_m)
            )
            if not sweep_exact:
                raise CollectionAbort(
                    "nominal BC sweep does not match the scheduled condition"
                )
            reference_match = bool(
                str(reference.get("phase", "")) == plan.task_phase
                and str(reference.get("controller", "")) == "bc_only"
                and int(reference.get("controller_event", -1)) == trigger_event
                and ee_error
                <= float(reference_cfg["maximum_ee_position_error_m"])
                and joint_error
                <= float(reference_cfg["maximum_arm_joint_error_rad"])
            )
            nominal_reference_audit = {
                "semantics": str(reference_cfg["trigger_state_semantics"]),
                "phase": plan.task_phase,
                "controller": str(reference.get("controller", "")),
                "controller_event": int(
                    reference.get("controller_event", -1)
                ),
                "precheck_control_step": int(reference.get("control_step", -1)),
                "precheck_simulation_time_s": float(
                    reference.get("simulation_time_s", -1.0)
                ),
                "ee_position_error_m": ee_error,
                "maximum_arm_joint_error_rad": joint_error,
                "match": reference_match,
            }
            if not reference_match:
                off_protocol_reasons.add("nominal_bc_phase_reference_mismatch")
            crossing_monitor = CrossingPathMonitor(
                start_position_world=corridor.start_world_m,
                end_position_world=corridor.end_world_m,
                path_deviation_threshold_m=float(
                    crossing_cfg["maximum_path_deviation_m"]
                ),
                expected_speed_mps=float(
                    crossing_cfg["speed_target_m_s"][plan.speed]
                ),
                relative_speed_tolerance=float(
                    crossing_cfg["speed_tolerance_fraction"]
                ),
                start_radius_m=float(crossing_cfg["start_radius_m"]),
                end_radius_m=float(crossing_cfg["end_radius_m"]),
                minimum_progress_fraction=float(
                    crossing_cfg["minimum_progress_fraction"]
                ),
                maximum_reverse_progress_m=float(
                    crossing_cfg["maximum_reverse_progress_m"]
                ),
            )
            cue_started_sim = now_sim
            feedback.set_crossing_hand(plan.crossing_hand)
            cue_visuals.show(
                corridor,
                simulation_time_s=now_sim + float(crossing_cfg["cue_lead_s"]),
                speed_m_s=float(crossing_cfg["speed_target_m_s"][plan.speed]),
            )
            _set_instruction(
                feedback,
                f"{plan.task_phase} | {plan.crossing_hand}손 | {plan.direction} | {plan.speed}\n"
                "초록 시작점으로 이동해 START까지 대기하세요. "
                "START 후 노란 guide 속도로 빨강 끝점까지 한 번만 crossing 하세요.",
            )
            state = "pre_crossing"

        # The controlled crossing is an intervention into the *running* frozen
        # task policy.  Latch only during the pre-crossing cue so the assigned
        # phase cannot disappear before START.  From START through response
        # and recovery, advance the normal task controller under frozen BC.
        # The explicit PAUSE transition and questionnaire both use a
        # current-EE hold while physics and CBF continue.
        control_mode = (
            "protocol_hold" if state in {"pause", "query"} else "nominal_bc"
        )
        advance = state in {
            "wait_phase",
            "crossing",
            "response",
            "recovery",
            "return_neutral",
            "resume_task",
        }
        monitor_for_step = (
            crossing_monitor
            if state in {"pre_crossing", "crossing", "response", "recovery"}
            else None
        )
        # The cue-lead rows are retained as pre-window provenance, but risk
        # onset is armed only at START.  Otherwise moving into the start gate
        # could become the encounter instead of the controlled traversal.
        detector_for_step = (
            detector
            if state
            in {
                "crossing",
                "response",
                "recovery",
                "return_neutral",
                "pause",
                "query",
            }
            and core_encounter is None
            and (response_window_closed_step < 0 or detector.active)
            else None
        )
        detector_had_evidence_before_step = bool(
            detector.active or detector.state == "onset_candidate"
        )
        state_during_step = state
        step = _perform_logged_step(
            env=env,
            policy=policy,
            provider=provider,
            watchdog=watchdog,
            trace=trace,
            recorder=recorder,
            feedback=feedback,
            button_source=button_source,
            controller_input_watchdog=controller_input_watchdog,
            emergency_abort_monitor=emergency_abort_monitor,
            video=video,
            plan=plan,
            participant_id=str(args.participant_id),
            session_id=str(args.session_id),
            obs=obs,
            info=info,
            tracking=tracking,
            arm_joint_indices=arm_joint_indices,
            control_mode=control_mode,
            trial_state=lifecycle.state.value,
            advance_task_phase=advance,
            crossing_monitor=monitor_for_step,
            detector=detector_for_step,
            response_detector=(
                response_detector
                if state in {
                    "crossing",
                    "response",
                    "recovery",
                    "return_neutral",
                    "pause",
                    "query",
                    "resume_task",
                }
                else None
            ),
            proximal_monitor=proximal_monitor,
            protocol=protocol,
            stop_requested=stop_requested,
        )
        obs, info, tracking = step.obs, step.info, step.tracking
        if state_during_step not in {"pause", "query"}:
            task_active_elapsed_s += float(env.physics_dt_s)
        step_success_now = bool(info.get("success", False))
        step_terminal_reason = str(info.get("task_terminal_reason", "")).strip()
        step_strict_failure = str(
            _strict_state(info).get("failure_reason", "")
        ).strip()
        step_terminal_now = bool(
            step.terminated
            or step.truncated
            or step_strict_failure
            or (step_terminal_reason and step_terminal_reason != "success")
        )
        task_success_observed = bool(task_success_observed or step_success_now)
        if step_terminal_now and not step_success_now:
            task_terminal_observed = True
            if not task_failure_reason_observed:
                task_failure_reason_observed = str(
                    step_strict_failure
                    or (
                        step_terminal_reason
                        if step_terminal_reason != "success"
                        else ""
                    )
                )
            if not task_terminal_kind_observed:
                task_terminal_kind_observed = (
                    "truncated" if step.truncated else "terminated"
                )
        if (
            (step_success_now or step_terminal_now)
            and task_completion_active_elapsed_s is None
        ):
            task_completion_active_elapsed_s = task_active_elapsed_s
        clocks_by_step[step.source_step] = (
            step.simulation_time_s,
            step.monotonic_ns,
        )
        tracked_step_count += 1
        if np.array_equal(
            np.asarray(step.tracking.valid_mask, dtype=np.int8),
            np.ones(3, dtype=np.int8),
        ):
            valid_tracking_step_count += 1
        current_ee_position = np.asarray(
            step.ee_position_world_m, dtype=np.float64
        )
        if state_during_step not in {"pause", "query"}:
            actual_task_ee_path_m += float(
                np.linalg.norm(current_ee_position - previous_ee_position)
            )
        previous_ee_position = current_ee_position.copy()

        dt_s = max(float(env.physics_dt_s), 1e-9)
        phase_for_jerk = str(step.response_phase)
        if phase_for_jerk in {
            ResponsePhase.BC_RESUMED.value,
            ResponsePhase.STABLE_TASK_RESUMPTION.value,
        }:
            if not math.isfinite(bc_resumed_started_sim):
                bc_resumed_started_sim = step.simulation_time_s
            phase_window = (
                "BC_RESUMED_FIRST_0_5_S"
                if step.simulation_time_s - bc_resumed_started_sim <= 0.5
                else ""
            )
        elif phase_for_jerk in {
            "PRE_RESPONSE", "CBF_ACTIVE", "SMOOTH_TAIL", "RECOVERY_ACTIVE"
        }:
            phase_window = phase_for_jerk
        else:
            phase_window = ""
        if (
            response_window_closed_step >= 0
            and step.source_step > response_window_closed_step
        ):
            phase_window = ""
        measured_velocity = np.asarray(
            step.measured_arm_velocity_rad_s, dtype=np.float64
        )
        ee_velocity_sample = np.asarray(
            step.ee_linear_velocity_world_m_s, dtype=np.float64
        )
        joint_acceleration = None
        ee_acceleration = None
        if previous_arm_velocity is not None:
            joint_acceleration = (
                measured_velocity - previous_arm_velocity
            ) / dt_s
        if previous_ee_velocity is not None:
            ee_acceleration = (
                ee_velocity_sample - previous_ee_velocity
            ) / dt_s
        if joint_acceleration is not None and previous_arm_acceleration is not None:
            joint_jerk_norm = float(
                np.linalg.norm(
                    (joint_acceleration - previous_arm_acceleration) / dt_s
                )
            )
            joint_jerk_by_window["WHOLE_TASK"].append(joint_jerk_norm)
            if phase_window:
                joint_jerk_by_window[phase_window].append(joint_jerk_norm)
        if ee_acceleration is not None and previous_ee_acceleration is not None:
            ee_jerk_norm = float(
                np.linalg.norm(
                    (ee_acceleration - previous_ee_acceleration) / dt_s
                )
            )
            ee_jerk_by_window["WHOLE_TASK"].append(ee_jerk_norm)
            if phase_window:
                ee_jerk_by_window[phase_window].append(ee_jerk_norm)
        if joint_acceleration is not None:
            previous_arm_acceleration = joint_acceleration
        if ee_acceleration is not None:
            previous_ee_acceleration = ee_acceleration
        previous_arm_velocity = measured_velocity.copy()
        previous_ee_velocity = ee_velocity_sample.copy()

        if step.response_update is not None:
            phase = str(step.response_phase)
            # The terminal label remains latched for neutral return, UI, and
            # post-query task rows.  Those rows are not additional response
            # duration; close the scientific window at the first stable row.
            in_response_window = bool(
                response_window_closed_step < 0
                or step.source_step <= response_window_closed_step
            )
            if in_response_window:
                response_phase_durations[phase] += float(env.physics_dt_s)
                if phase != previous_response_phase:
                    response_phase_history.append(
                        {
                            "phase": phase,
                            "control_step": step.source_step,
                            "simulation_time_s": step.simulation_time_s,
                        }
                    )
                    previous_response_phase = phase
            if in_response_window and step.decision_context is not None:
                pending_decision_context = dict(step.decision_context)
            if (
                in_response_window
                and step.response_update.onset_confirmed_now
                and pending_decision_context is not None
            ):
                frozen_decision_context = dict(pending_decision_context)
            if (
                in_response_window
                and response_detector.onset_confirmed_step < 0
            ):
                if step.response_update.onset_candidate_step < 0:
                    # A broken streak is not part of the eventual confirmed
                    # safety-response episode.
                    response_metric_samples.clear()
                else:
                    if step.response_update.onset_candidate_now:
                        response_metric_samples.clear()
                    response_metric_samples.append(
                        (
                            int(step.source_step),
                            float(step.intervention_norm_rad_s),
                            np.asarray(
                                step.correction_arm_rad_s, dtype=np.float64
                            ).copy(),
                            bool(step.recovery_active),
                            str(step.response_phase),
                        )
                    )
            elif in_response_window and (
                response_window_closed_step < 0
                or step.source_step <= response_window_closed_step
            ):
                # Include the confirmation frame and the terminal stable row.
                response_metric_samples.append(
                    (
                        int(step.source_step),
                        float(step.intervention_norm_rad_s),
                        np.asarray(
                            step.correction_arm_rad_s, dtype=np.float64
                        ).copy(),
                        bool(step.recovery_active),
                        str(step.response_phase),
                    )
                )
        elif response_detector.onset_confirmed_step < 0:
            response_phase_durations[
                ResponsePhase.PRE_RESPONSE.value
            ] += float(env.physics_dt_s)
        cue_visuals.update(step.simulation_time_s)
        if (
            nominal_trajectory is not None
            and state_during_step in {"pre_crossing", "crossing"}
        ):
            if step.cbf_active or step.intervention_norm_rad_s > 0.01:
                if nominal_timeline_audit["comparison_closed_at_cbf_step"] < 0:
                    nominal_timeline_audit[
                        "comparison_closed_at_cbf_step"
                    ] = step.source_step
            elif nominal_timeline_audit["comparison_closed_at_cbf_step"] < 0:
                relative_time_s = step.simulation_time_s - cue_started_sim
                samples = nominal_trajectory.get("samples", ())
                final_relative_time_s = (
                    float(samples[-1].get("relative_time_s", -math.inf))
                    if isinstance(samples, list)
                    and samples
                    and isinstance(samples[-1], Mapping)
                    else -math.inf
                )
                if relative_time_s <= final_relative_time_s + 1e-12:
                    expected_state, time_error = (
                        _nearest_nominal_trajectory_sample(
                            nominal_trajectory, relative_time_s
                        )
                    )
                    ee_state_error, joint_state_error = _nominal_state_errors(
                        env=env,
                        expected=expected_state,
                        arm_joint_indices=arm_joint_indices,
                    )
                    nominal_timeline_audit["compared_sample_count"] += 1
                    nominal_timeline_audit[
                        "maximum_time_alignment_error_s"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_time_alignment_error_s"
                        ],
                        time_error,
                    )
                    nominal_timeline_audit[
                        "maximum_ee_position_error_m"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_ee_position_error_m"
                        ],
                        ee_state_error,
                    )
                    nominal_timeline_audit[
                        "maximum_arm_joint_error_rad"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_arm_joint_error_rad"
                        ],
                        joint_state_error,
                    )
                    reference_cfg = scenario["nominal_bc_reference"]
                    row_match = bool(
                        time_error
                        <= float(
                            reference_cfg[
                                "maximum_sweep_sample_interval_s"
                            ]
                        )
                        and ee_state_error
                        <= float(reference_cfg["maximum_ee_position_error_m"])
                        and joint_state_error
                        <= float(reference_cfg["maximum_arm_joint_error_rad"])
                    )
                    nominal_timeline_audit["match"] = bool(
                        nominal_timeline_audit["match"] and row_match
                    )
                    if not row_match:
                        off_protocol_reasons.add(
                            "nominal_timeline_drift_before_cbf"
                        )
        for marker in step.realtime_markers:
            marker_counts[marker.marker_type] += 1
        if state == "wait_phase" and (
            step.minimum_gap_m <= staging_gap_m
            or step.proximal_gap_m <= staging_gap_m
            or step.cbf_active
            or step.intervention_norm_rad_s > 0.01
        ):
            off_protocol_reasons.add("pre_cue_human_intrusion")
        if state in {"pre_crossing", "crossing", "response", "recovery"}:
            intended_gap = (
                step.left_gap_m
                if plan.crossing_hand == "left"
                else step.right_gap_m
            )
            minimum_gap = min(minimum_gap, intended_gap)
            minimum_any_hand_gap = min(
                minimum_any_hand_gap, step.minimum_gap_m
            )
            if step.proximal_gap_m < minimum_proximal_gap:
                minimum_proximal_gap = step.proximal_gap_m
                closest_proximal_collider = step.proximal_collider_path
            cbf_any_active = cbf_any_active or step.cbf_active
            cbf_max_norm = max(cbf_max_norm, step.intervention_norm_rad_s)
            intervening = step.intervention_norm_rad_s > 0.01
            intervention_duration += (
                float(env.physics_dt_s) if intervening else 0.0
            )
            if intervening and not intervention_prior:
                intervention_count += 1
            intervention_prior = intervening
            opposite_gap = (
                step.right_gap_m
                if plan.crossing_hand == "left"
                else step.left_gap_m
            )
            if opposite_gap <= float(protocol["cbf"]["activation_gap_m"]):
                off_protocol_reasons.add("non_crossing_hand_intrusion")
            opposite_proximal_gap = (
                step.right_proximal_gap_m
                if plan.crossing_hand == "left"
                else step.left_proximal_gap_m
            )
            if opposite_proximal_gap <= float(
                protocol["cbf"]["activation_gap_m"]
            ):
                off_protocol_reasons.add(
                    "non_crossing_hand_proximal_intrusion"
                )
            if step.proximal_gap_m <= 0.13:
                off_protocol_reasons.add("proximal_unprotected_crossing")
                if step.proximal_collider_path:
                    off_protocol_reasons.add(
                        "proximal_collider:" + step.proximal_collider_path
                    )
        if (
            step.encounter_update is not None
            and step.encounter_update.closed is not None
        ):
            core_encounter = step.encounter_update.closed
        risk_now = bool(
            step.minimum_gap_m <= float(encounter_cfg["activation_gap_m"])
            or (
                step.dynamic_measurement_valid
                and
                step.closing_speed_m_s > 0.0
                and 0.0 <= step.ttc_s <= float(encounter_cfg["ttc_onset_s"])
            )
            or step.intervention_norm_rad_s
            >= float(encounter_cfg["intervention_onset_rad_s"])
        )
        if (
            core_encounter is not None
            and risk_now
            and state in {"crossing", "response", "recovery"}
            and (
                step.encounter_update is None
                or step.encounter_update.row_encounter_id
                != core_encounter.encounter_id
            )
        ):
            # A >1 s post-offset risk episode would be a second encounter in a
            # one-crossing trial.  Preserve its raw per-step evidence, exclude
            # the trial, and keep the primary query relation unambiguous.
            off_protocol_reasons.add("multiple_encounter_risk_after_close")
        elif (
            state == "recovery"
            and core_encounter is None
            and not detector_had_evidence_before_step
            and risk_now
        ):
            off_protocol_reasons.add("risk_started_during_recovery")

        if state == "wait_phase":
            if step.terminated or step.truncated:
                raise CollectionAbort("task_ended_before_crossing_phase")
            # Preserve at least one auditable WAIT row even when the assigned
            # phase is event 0 immediately after reset.  Latching prevents a
            # one-step phase transition from making that trigger disappear.
            wait_phase_row_recorded = True
            trigger_event_latched = bool(
                trigger_event_latched
                or event == trigger_event
                or int(info.get("controller_event", -1)) == trigger_event
            )
        elif state == "pre_crossing":
            early_risk = bool(
                step.minimum_gap_m <= float(encounter_cfg["activation_gap_m"])
                or (
                    step.dynamic_measurement_valid
                    and
                    step.closing_speed_m_s > 0.0
                    and 0.0 <= step.ttc_s <= float(encounter_cfg["ttc_onset_s"])
                )
                or step.intervention_norm_rad_s
                >= float(encounter_cfg["intervention_onset_rad_s"])
            )
            if early_risk or step.cbf_active:
                off_protocol_reasons.add("early_encounter_before_crossing")
            if (
                step.crossing_update is not None
                and step.crossing_update.intended_crossing_completed
            ):
                off_protocol_reasons.add("crossing_completed_before_start_cue")
            if (
                step.simulation_time_s - cue_started_sim
                >= float(crossing_cfg["cue_lead_s"])
            ):
                assert crossing_monitor is not None
                assert corridor is not None
                assert nominal_sweep_audit is not None
                assert nominal_trajectory is not None
                reference_cfg = scenario["nominal_bc_reference"]
                coverage = nominal_sweep_audit.get("coverage", {})
                if not isinstance(coverage, Mapping):
                    raise CollectionAbort(
                        "nominal sweep START coverage audit is malformed"
                    )
                expected_start_relative_s = float(
                    nominal_sweep_audit.get(
                        "crossing_start_query_simulation_time_s", math.nan
                    )
                ) - float(
                    coverage.get("phase_start_simulation_time_s", math.nan)
                )
                actual_start_relative_s = (
                    step.simulation_time_s - cue_started_sim
                )
                expected_start_state, state_time_error = (
                    _nearest_nominal_trajectory_sample(
                        nominal_trajectory, actual_start_relative_s
                    )
                )
                start_ee_error, start_joint_error = _nominal_state_errors(
                    env=env,
                    expected=expected_start_state,
                    arm_joint_indices=arm_joint_indices,
                )
                start_result = env.safety_geometry.evaluate_hand(
                    plan.crossing_hand, corridor.start_world_m
                )
                actual_start_gap = float(
                    getattr(start_result, "surface_gap_m", math.nan)
                )
                expected_start_gap = float(
                    nominal_sweep_audit.get(
                        "crossing_start_surface_gap_m", math.nan
                    )
                )
                start_gap_error = abs(actual_start_gap - expected_start_gap)
                scheduled_time_error = abs(
                    actual_start_relative_s - expected_start_relative_s
                )
                start_geometry_valid = bool(
                    getattr(start_result, "geometry_valid", False)
                    and math.isfinite(actual_start_gap)
                    and str(getattr(start_result, "closest_link", ""))
                    and str(
                        getattr(start_result, "closest_collider_path", "")
                    )
                )
                start_match = bool(
                    start_geometry_valid
                    and math.isfinite(expected_start_relative_s)
                    and scheduled_time_error
                    <= float(reference_cfg["maximum_sweep_sample_interval_s"])
                    and state_time_error
                    <= float(reference_cfg["maximum_sweep_sample_interval_s"])
                    and start_ee_error
                    <= float(reference_cfg["maximum_ee_position_error_m"])
                    and start_joint_error
                    <= float(reference_cfg["maximum_arm_joint_error_rad"])
                    and start_gap_error
                    <= float(reference_cfg["maximum_start_gap_error_m"])
                )
                start_recheck_audit = {
                    "semantics": "actual_start_vs_time_aligned_nominal_sweep_v1",
                    "actual_relative_time_s": actual_start_relative_s,
                    "expected_query_relative_time_s": expected_start_relative_s,
                    "scheduled_time_error_s": scheduled_time_error,
                    "nearest_state_time_error_s": state_time_error,
                    "actual_start_surface_gap_m": actual_start_gap,
                    "expected_start_surface_gap_m": expected_start_gap,
                    "start_surface_gap_error_m": start_gap_error,
                    "ee_position_error_m": start_ee_error,
                    "maximum_arm_joint_error_rad": start_joint_error,
                    "actual_closest_link": str(
                        getattr(start_result, "closest_link", "")
                    ),
                    "actual_closest_collider_path": str(
                        getattr(start_result, "closest_collider_path", "")
                    ),
                    "geometry_valid": start_geometry_valid,
                    "match": start_match,
                }
                if not start_match:
                    off_protocol_reasons.add(
                        "nominal_start_recheck_failed"
                    )
                hand_pose = getattr(step.tracking, plan.crossing_hand)
                # Pre-START samples are useful audit rows, but dwell/jitter at
                # the green marker must not bias traversal speed, reverse
                # progress, or the one-crossing count.  Re-observe the current
                # pose after reset so START readiness uses the exact same
                # radial+axial gate predicate as the trial measurement.
                crossing_monitor.reset()
                start_update = crossing_monitor.observe(
                    position_world=hand_pose.position_world,
                    simulation_time_s=step.simulation_time_s,
                    valid=True,
                )
                if not start_update.crossing_started:
                    off_protocol_reasons.add(
                        "hand_not_at_start_gate_on_start_cue"
                    )
                state = "crossing"
                crossing_started_sim = step.simulation_time_s
                lifecycle.advance(TrialState.EXECUTE_SINGLE_CROSSING)
                lifecycle.record_crossing()
                lifecycle.advance(TrialState.RUN_ASSIGNED_RESPONSE_A_OR_C)
                lifecycle.advance(TrialState.TRACK_SAFETY_RESPONSE_EPISODE)
                _set_instruction(
                    feedback,
                    "START — guide를 따라 지정 손을 한 번 crossing 하세요.",
                )
        elif state == "crossing":
            if (
                step.crossing_update is not None
                and step.crossing_update.intended_crossing_completed
            ):
                crossing_completed_sim = step.simulation_time_s
                state = "response"
                _set_instruction(
                    feedback,
                    "손은 끝점 밖으로 이동하고 로봇 반응을 계속 관찰하세요.",
                )
            elif (
                step.simulation_time_s - crossing_started_sim
                > float(crossing_cfg["crossing_timeout_s"])
            ):
                off_protocol_reasons.add("crossing_timeout")
                crossing_completed_sim = step.simulation_time_s
                state = "response"
        elif state in {"response", "recovery"}:
            if step.recovery_active and state != "recovery":
                state = "recovery"
                recovery_started_sim = step.simulation_time_s
                _set_instruction(
                    feedback,
                    "Recovery 관찰 중 — 로봇 반응과 작업 재개를 계속 보세요.",
                )
            response_complete = bool(
                step.response_phase
                == ResponsePhase.STABLE_TASK_RESUMPTION.value
            )
            response_timed_out_now = bool(
                step.simulation_time_s - crossing_completed_sim
                >= float(response_cfg["maximum_duration_s"])
            )
            if response_complete or response_timed_out_now:
                if response_timed_out_now and not response_complete:
                    off_protocol_reasons.add("stable_task_resumption_timeout")
                    if response_detector.onset_confirmed_step < 0:
                        off_protocol_reasons.add("response_episode_not_observed")
                response_window_closed_step = int(step.source_step)
                response_episode_timed_out = bool(
                    response_timed_out_now and not response_complete
                )
                response_terminal_update = step.response_update
                if core_encounter is None:
                    closed_at_response_boundary = detector.flush(
                        step=int(step.source_step),
                        simulation_time_s=float(step.simulation_time_s),
                        terminal_status=(
                            "response_window_timeout"
                            if response_episode_timed_out
                            else "stable_task_resumption"
                        ),
                    )
                    if closed_at_response_boundary is not None:
                        core_encounter = closed_at_response_boundary
                lifecycle.advance(TrialState.DETECT_STABLE_TASK_RESUMPTION)
                lifecycle.advance(TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE)
                cue_visuals.hide()
                feedback.set_crossing_hand(None)
                _set_instruction(
                    feedback,
                    "양손을 로봇에서 15 cm 이상 떨어진 neutral 위치로 되돌려 주세요.",
                )
                state = "return_neutral"
        elif state == "return_neutral":
            tracking_valid = bool(
                np.array_equal(
                    np.asarray(step.tracking.valid_mask, dtype=np.int8),
                    np.ones(3, dtype=np.int8),
                )
            )
            clear = bool(
                tracking_valid
                and step.left_gap_m
                >= float(crossing_cfg["neutral_clear_gap_m"])
                and step.right_gap_m
                >= float(crossing_cfg["neutral_clear_gap_m"])
                and step.left_proximal_gap_m
                >= float(crossing_cfg["neutral_clear_gap_m"])
                and step.right_proximal_gap_m
                >= float(crossing_cfg["neutral_clear_gap_m"])
            )
            neutral_clear_frames = neutral_clear_frames + 1 if clear else 0
            if neutral_clear_frames >= int(
                crossing_cfg["neutral_confirmation_frames"]
            ):
                lifecycle.advance(TrialState.PAUSE_NOMINAL_TASK_PROGRESSION)
                _set_instruction(
                    feedback,
                    "작업 진행을 안전하게 멈추고 설문을 준비합니다.",
                )
                state = "pause"
        elif state == "pause":
            # The step just recorded proves phase_advance=0 and safe-hold
            # behavior for the explicit PAUSE FSM state before UI issuance.
            lifecycle.advance(TrialState.SHOW_MANDATORY_FEEDBACK_UI)
            lifecycle.record_query()
            feedback.start_questionnaire(
                _feedback_clocks(
                    env,
                    monotonic_ns=time.monotonic_ns(),
                    unix_ns=time.time_ns(),
                ),
                context_id=plan.trial_id,
                timeout_s=float(protocol["feedback"]["query_timeout_s"]),
            )
            query_started = True
            state = "query"
        elif state == "query":
            if step.questionnaire_result is not None:
                query_result = step.questionnaire_result
                break

        if (
            state not in {"query", "resume_task"}
            and (step.terminated or step.truncated)
            and not bool(info.get("success", False))
            and not query_started
            and state != "recovery"
        ):
            off_protocol_reasons.add("task_terminal_before_query")
        if (
            int(info.get("step", 0)) >= int(env.config.max_episode_steps) - 2
            and state != "query"
        ):
            raise CollectionAbort(
                "trial exhausted control horizon before mandatory query"
            )

    assert query_result is not None
    lifecycle.advance(TrialState.SAVE_FEEDBACK)
    query = _query_record(
        plan,
        query_result,
        str(plan.encounter_id),
    )
    for answer in query_result.answers:
        answer_payload = answer.as_dict()
        answer_payload["question_id"] = {
            "q1_modification_needed": Q1_ID,
            "q2": "q2_perceived_danger",
            "q3": "q3_abruptness",
            "q4": "q4_excessive_duration",
            "q5": "q5_task_disruption",
            "q6": "q6_confidence",
            "rejection_reasons": "modification_reasons",
        }.get(str(answer.question_id), str(answer.question_id))
        recorder.append_query_answer(
            query_id=plan.query_id,
            trial_id=plan.trial_id,
            answer=answer_payload,
        )
    recorder.append_query(query)
    lifecycle.record_feedback_saved()
    lifecycle.advance(TrialState.RESUME_TASK_TO_COMPLETION_OR_TERMINAL)
    feedback.set_crossing_hand(None)
    _set_instruction(feedback, "응답이 저장되었습니다. 다음 trial을 준비합니다.")

    if (
        not args.reset_after_query
        and bool(protocol["task"]["complete_task_after_query"])
    ):
        while not (
            bool(info.get("success", False))
            or str(info.get("task_terminal_reason", ""))
            or int(info.get("step", 0)) >= int(env.config.max_episode_steps)
        ):
            step = _perform_logged_step(
                env=env,
                policy=policy,
                provider=provider,
                watchdog=watchdog,
                trace=trace,
                recorder=recorder,
                feedback=feedback,
                button_source=button_source,
                controller_input_watchdog=controller_input_watchdog,
                emergency_abort_monitor=emergency_abort_monitor,
                video=video,
                plan=plan,
                participant_id=str(args.participant_id),
                session_id=str(args.session_id),
                obs=obs,
                info=info,
                tracking=tracking,
                arm_joint_indices=arm_joint_indices,
                control_mode="nominal_bc",
                trial_state=lifecycle.state.value,
                advance_task_phase=True,
                crossing_monitor=None,
                detector=(detector if core_encounter is None else None),
                response_detector=response_detector,
                proximal_monitor=None,
                protocol=protocol,
                stop_requested=stop_requested,
            )
            obs, info, tracking = step.obs, step.info, step.tracking
            task_active_elapsed_s += float(env.physics_dt_s)
            step_success_now = bool(info.get("success", False))
            step_terminal_reason = str(
                info.get("task_terminal_reason", "")
            ).strip()
            step_strict_failure = str(
                _strict_state(info).get("failure_reason", "")
            ).strip()
            step_terminal_now = bool(
                step.terminated
                or step.truncated
                or step_strict_failure
                or (step_terminal_reason and step_terminal_reason != "success")
            )
            task_success_observed = bool(
                task_success_observed or step_success_now
            )
            if step_terminal_now and not step_success_now:
                task_terminal_observed = True
                if not task_failure_reason_observed:
                    task_failure_reason_observed = str(
                        step_strict_failure
                        or (
                            step_terminal_reason
                            if step_terminal_reason != "success"
                            else ""
                        )
                    )
                if not task_terminal_kind_observed:
                    task_terminal_kind_observed = (
                        "truncated" if step.truncated else "terminated"
                    )
            if (
                (step_success_now or step_terminal_now)
                and task_completion_active_elapsed_s is None
            ):
                task_completion_active_elapsed_s = task_active_elapsed_s
            clocks_by_step[step.source_step] = (
                step.simulation_time_s,
                step.monotonic_ns,
            )
            if (
                step.encounter_update is not None
                and step.encounter_update.closed is not None
            ):
                core_encounter = step.encounter_update.closed
            tracked_step_count += 1
            if np.array_equal(
                np.asarray(step.tracking.valid_mask, dtype=np.int8),
                np.ones(3, dtype=np.int8),
            ):
                valid_tracking_step_count += 1
            current_ee_position = np.asarray(
                step.ee_position_world_m, dtype=np.float64
            )
            actual_task_ee_path_m += float(
                np.linalg.norm(current_ee_position - previous_ee_position)
            )
            previous_ee_position = current_ee_position.copy()
            dt_s = max(float(env.physics_dt_s), 1e-9)
            measured_velocity = np.asarray(
                step.measured_arm_velocity_rad_s, dtype=np.float64
            )
            ee_velocity_sample = np.asarray(
                step.ee_linear_velocity_world_m_s, dtype=np.float64
            )
            joint_acceleration = (
                None
                if previous_arm_velocity is None
                else (measured_velocity - previous_arm_velocity) / dt_s
            )
            ee_acceleration = (
                None
                if previous_ee_velocity is None
                else (ee_velocity_sample - previous_ee_velocity) / dt_s
            )
            if (
                joint_acceleration is not None
                and previous_arm_acceleration is not None
            ):
                joint_jerk_by_window["WHOLE_TASK"].append(
                    float(
                        np.linalg.norm(
                            (joint_acceleration - previous_arm_acceleration)
                            / dt_s
                        )
                    )
                )
            if (
                ee_acceleration is not None
                and previous_ee_acceleration is not None
            ):
                ee_jerk_by_window["WHOLE_TASK"].append(
                    float(
                        np.linalg.norm(
                            (ee_acceleration - previous_ee_acceleration) / dt_s
                        )
                    )
                )
            if joint_acceleration is not None:
                previous_arm_acceleration = joint_acceleration
            if ee_acceleration is not None:
                previous_ee_acceleration = ee_acceleration
            previous_arm_velocity = measured_velocity.copy()
            previous_ee_velocity = ee_velocity_sample.copy()
            if step.terminated or step.truncated:
                break

    if core_encounter is None:
        core_encounter = detector.flush(
            step=max(0, int(info.get("step", 1)) - 1),
            simulation_time_s=float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            ),
            terminal_status="trial_end",
        )
    encounter_v1 = None
    if core_encounter is not None:
        encounter_v1 = _convert_encounter(
            core_encounter,
            clocks_by_step,
            encounter_id=str(plan.encounter_id),
        )
        recorder.append_encounter(encounter_v1)
    encounter_id = str(plan.encounter_id)
    if encounter_id != query.encounter_id:
        raise CollectionAbort(
            "query/encounter relation changed after query completion"
        )
    if response_terminal_update is None:
        raise CollectionAbort("response observation window was never closed")
    response_onset_step = int(response_terminal_update.onset_candidate_step)
    response_onset_confirmed_step = int(
        response_terminal_update.onset_confirmed_step
    )
    response_onset_sim = float(
        response_terminal_update.onset_simulation_time_s
    )
    recovery_onset_step = int(response_terminal_update.recovery_onset_step)
    recovery_end_step_exclusive = int(
        response_terminal_update.recovery_end_step_exclusive
    )
    recovery_onset_sim = float(
        response_terminal_update.recovery_onset_simulation_time_s
    )
    recovery_end_sim = float(
        response_terminal_update.recovery_end_simulation_time_s
    )
    if response_onset_confirmed_step < 0:
        # A one/two-frame raw candidate remains visible in step rows but is
        # not promoted into an episode-level onset.
        response_onset_step = -1
        response_onset_sim = -1.0
        recovery_onset_step = -1
        recovery_end_step_exclusive = -1
        recovery_onset_sim = -1.0
        recovery_end_sim = -1.0
    if response_episode_timed_out:
        stable_task_resumption_step = -1
        response_end_step_exclusive = -1
        response_end_sim = -1.0
    else:
        stable_task_resumption_step = int(
            response_terminal_update.stable_resumption_step
        )
        response_end_step_exclusive = (
            stable_task_resumption_step + 1
            if stable_task_resumption_step >= 0
            else -1
        )
        response_end_sim = float(
            response_terminal_update.stable_resumption_simulation_time_s
        )

    metrics = (
        crossing_monitor.finalize(
            additional_off_protocol_reasons=off_protocol_reasons
        )
        if crossing_monitor is not None
        else None
    )
    if metrics is None:
        crossing_count = 0
        rms_dev = max_dev = actual_speed = 0.0
        off_protocol_reasons.add("crossing_not_initialized")
        actual_crossing_direction = resolve_actual_crossing_direction(
            plan.direction, 0
        )
    else:
        crossing_count = metrics.intended_crossing_count
        rms_dev = _finite_or_zero(metrics.rms_path_deviation_m)
        max_dev = _finite_or_zero(metrics.maximum_path_deviation_m)
        actual_speed = _finite_or_zero(metrics.mean_speed_mps)
        off_protocol_reasons.update(metrics.off_protocol_reasons)
        actual_crossing_direction = resolve_actual_crossing_direction(
            plan.direction, metrics.first_completed_traversal_sign
        )
    if actual_crossing_direction == "not_observed":
        off_protocol_reasons.add("actual_crossing_direction_not_observed")
    if core_encounter is not None and core_encounter.timeout:
        off_protocol_reasons.add("encounter_timeout")
    if query.response_status != "completed":
        off_protocol_reasons.add("mandatory_query_incomplete")
    if response_onset_confirmed_step < 0:
        off_protocol_reasons.add("response_onset_not_confirmed")
    if stable_task_resumption_step < 0:
        off_protocol_reasons.add("stable_task_resumption_not_confirmed")
    if (
        response_onset_confirmed_step >= 0
        and stable_task_resumption_step >= 0
    ):
        expected_metric_start = response_onset_step
        expected_metric_end = stable_task_resumption_step
        if (
            not response_metric_samples
            or response_metric_samples[0][0] != expected_metric_start
            or response_metric_samples[-1][0] != expected_metric_end
        ):
            off_protocol_reasons.add("response_metric_window_incomplete")
        else:
            integrated_intervention_rad = float(
                sum(sample[1] for sample in response_metric_samples)
                * float(env.physics_dt_s)
            )
            corrections = np.stack(
                [sample[2] for sample in response_metric_samples], axis=0
            )
            correction_total_variation_rad_s = float(
                np.sum(
                    np.linalg.norm(np.diff(corrections, axis=0), axis=1)
                )
            ) if len(corrections) > 1 else 0.0
    recovery_duration_s = (
        0.0
        if response_onset_confirmed_step < 0
        else float(
            sum(int(sample[3]) for sample in response_metric_samples)
            * float(env.physics_dt_s)
        )
    )
    # PRE_RESPONSE remains a whole pre-window audit channel.  Every outcome
    # phase duration, however, is bounded to the eventual confirmed onset
    # candidate through the stable/timeout boundary; failed one/two-frame CBF
    # candidates must not inflate the measured response.
    for response_phase in ResponsePhase:
        if response_phase is not ResponsePhase.PRE_RESPONSE:
            response_phase_durations[response_phase.value] = (
                0.0
                if response_onset_confirmed_step < 0
                else float(
                    sum(
                        int(sample[4] == response_phase.value)
                        for sample in response_metric_samples
                    )
                    * float(env.physics_dt_s)
                )
            )
    if frozen_decision_context is None:
        off_protocol_reasons.add("decision_context_unavailable")
    if nominal_sweep_audit is None or nominal_reference_audit is None:
        off_protocol_reasons.add("nominal_calibration_audit_unavailable")
    if start_recheck_audit is None or not bool(
        start_recheck_audit.get("match", False)
    ):
        off_protocol_reasons.add("nominal_start_recheck_failed")
    if (
        int(nominal_timeline_audit["compared_sample_count"]) < 1
        or not bool(nominal_timeline_audit["match"])
    ):
        off_protocol_reasons.add("nominal_timeline_audit_failed")
    protocol_valid = bool(
        crossing_count == 1
        and not off_protocol_reasons
        and query.response_status == "completed"
    )
    strict = _strict_state(info)
    task_success = bool(task_success_observed or info.get("success", False))
    terminal_reason = _normalized_task_failure_reason(
        info,
        strict_state=strict,
        task_success=task_success,
        reset_after_query=bool(args.reset_after_query),
        max_episode_steps=int(env.config.max_episode_steps),
    )
    if not task_success:
        terminal_reason = str(
            task_failure_reason_observed
            or terminal_reason
            or (
                "environment_truncated_without_reason"
                if task_terminal_kind_observed == "truncated"
                else "environment_terminated_without_reason"
                if task_terminal_observed
                else ""
            )
        )
    end_sim = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    def jerk_metrics(values: Sequence[float], *, kind: str) -> dict[str, Any]:
        unit = "m_s3" if kind == "ee" else "rad_s3"
        return {
            "sample_count": len(values),
            f"rms_norm_{unit}": (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else 0.0
            ),
            f"peak_norm_{unit}": max(values, default=0.0),
        }

    windowed_jerk = {
        window: {
            "ee": jerk_metrics(ee_jerk_by_window[window], kind="ee"),
            "joint_space": jerk_metrics(
                joint_jerk_by_window[window], kind="joint_space"
            ),
        }
        for window in jerk_windows
    }
    total_response_duration = (
        0.0
        if response_onset_sim < 0.0 or response_end_sim < response_onset_sim
        else response_end_sim - response_onset_sim
    )
    nominal_task_path_m = float(
        restoration.get("nominal_bc_task_ee_path_m", 0.0)
    )
    nominal_completion_time_s = float(
        restoration.get("nominal_bc_task_completion_time_s", 0.0)
    )
    measured_completion_time_s = (
        task_active_elapsed_s
        if task_completion_active_elapsed_s is None
        else task_completion_active_elapsed_s
    )
    lifecycle.advance(TrialState.SAVE_TRIAL)
    recorder.end_trial(
        {
            "condition_id": plan.condition_id,
            "objective_mode": plan.objective_mode,
            "lambda_s": float(plan.lambda_s),
            "actual_cbf_config_json": json.dumps(
                actual_cbf_config(
                    env._cbf_filter, condition_id=plan.condition_id
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "trial_fsm_history_json": json.dumps(
                [
                    *lifecycle.history,
                    TrialState.VALIDATE_TRIAL.value,
                    TrialState.NEXT_TRIAL_OR_END_SESSION.value,
                ],
                separators=(",", ":"),
            ),
            "response_phase_history_json": json.dumps(
                response_phase_history, sort_keys=True, separators=(",", ":")
            ),
            "response_phase_durations_json": json.dumps(
                response_phase_durations, sort_keys=True, separators=(",", ":")
            ),
            "decision_context_json": json.dumps(
                frozen_decision_context or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "response_onset_step": response_onset_step,
            "response_end_step_exclusive": response_end_step_exclusive,
            "response_onset_simulation_time_s": response_onset_sim,
            "response_end_simulation_time_s": response_end_sim,
            "cbf_active_duration_s": float(
                response_phase_durations[ResponsePhase.CBF_ACTIVE.value]
            ),
            "response_onset_confirmed_step": response_onset_confirmed_step,
            "recovery_onset_step": recovery_onset_step,
            "recovery_end_step_exclusive": recovery_end_step_exclusive,
            "recovery_onset_simulation_time_s": recovery_onset_sim,
            "recovery_end_simulation_time_s": recovery_end_sim,
            "stable_task_resumption_step": stable_task_resumption_step,
            "response_duration_s": total_response_duration,
            "total_safety_response_duration_s": total_response_duration,
            "smooth_tail_duration_s": float(
                response_phase_durations[ResponsePhase.SMOOTH_TAIL.value]
            ),
            "recovery_duration_s": recovery_duration_s,
            "bc_resumed_duration_s": min(
                0.5,
                float(
                    response_phase_durations[ResponsePhase.BC_RESUMED.value]
                ),
            ),
            "integrated_intervention_rad": float(
                integrated_intervention_rad
            ),
            "correction_total_variation_rad_s": float(
                correction_total_variation_rad_s
            ),
            "windowed_jerk_json": json.dumps(
                windowed_jerk, sort_keys=True, separators=(",", ":")
            ),
            "protocol_valid": protocol_valid,
            "actual_crossing_count": crossing_count,
            "actual_crossing_direction": actual_crossing_direction,
            "actual_path_deviation_rms_m": rms_dev,
            "actual_path_deviation_max_m": max_dev,
            "actual_crossing_speed_m_s": actual_speed,
            "tracking_validity_rate": (
                0.0
                if tracked_step_count == 0
                else valid_tracking_step_count / tracked_step_count
            ),
            "delta_path_m": float(
                actual_task_ee_path_m - nominal_task_path_m
            ),
            "delta_completion_time_s": float(
                measured_completion_time_s - nominal_completion_time_s
            ),
            "actual_minimum_gap_m": (
                10.0 if not math.isfinite(minimum_gap) else minimum_gap
            ),
            "actual_minimum_any_hand_gap_m": (
                10.0
                if not math.isfinite(minimum_any_hand_gap)
                else minimum_any_hand_gap
            ),
            "actual_minimum_proximal_gap_m": (
                10.0
                if not math.isfinite(minimum_proximal_gap)
                else minimum_proximal_gap
            ),
            "closest_proximal_collider": closest_proximal_collider,
            "off_protocol": bool(off_protocol_reasons),
            "off_protocol_reasons": sorted(off_protocol_reasons),
            "cbf_activated": cbf_any_active,
            "cbf_total_intervention_duration_s": intervention_duration,
            "cbf_max_intervention_norm_rad_s": cbf_max_norm,
            "cbf_intervention_count": intervention_count,
            "task_success": task_success,
            "task_failure_reason": terminal_reason,
            "object_drop": (
                "drop" in terminal_reason.lower()
                or "grasp_lost" in terminal_reason.lower()
            ),
            "grasp_outcome": (
                "grasped" if bool(strict.get("grasp_observed", False)) else "not_observed"
            ),
            "release_outcome": (
                "released_and_settled"
                if bool(strict.get("success_latched", False))
                else ("reset_after_query" if args.reset_after_query else "not_settled")
            ),
            "task_completion_observed": (
                task_success_observed or task_terminal_observed
            ),
            "completion_time_s": (
                -1.0
                if task_completion_active_elapsed_s is None
                else max(0.0, task_completion_active_elapsed_s)
            ),
            "completion_time_semantics": (
                "active_task_simulation_time_excluding_query_hold_v1"
            ),
            "trial_simulation_duration_s": max(0.0, end_sim - start_sim),
            "trial_wall_duration_s": max(
                0.0, (time.monotonic_ns() - start_mono) / 1e9
            ),
            "recovery_time_s": (
                0.0
                if not math.isfinite(recovery_started_sim)
                else max(
                    0.0,
                    query.issued_simulation_time_s - recovery_started_sim,
                )
            ),
            "risk_onset_step": (
                -1 if encounter_v1 is None else encounter_v1.risk_onset_step
            ),
            "cbf_intervention_start_step": (
                -1
                if encounter_v1 is None
                else encounter_v1.cbf_intervention_start_step
            ),
            "cbf_intervention_confirmed_step": (
                -1
                if encounter_v1 is None
                else encounter_v1.cbf_intervention_confirmed_step
            ),
            "realtime_safety_marker_count": marker_counts[
                "realtime_safety_concern"
            ],
            "realtime_behavior_anomaly_marker_count": marker_counts[
                "realtime_behavior_anomaly"
            ],
            "planned_corridor_json": (
                "{}"
                if corridor is None
                else json.dumps(
                    {
                        **corridor.as_dict(),
                        "nominal_bc_reference": nominal_reference_audit,
                        "nominal_bc_sweep": nominal_sweep_audit,
                        "start_recheck": start_recheck_audit,
                        "pre_cbf_timeline_audit": nominal_timeline_audit,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "end_simulation_time_s": end_sim,
            "end_monotonic_ns": time.monotonic_ns(),
            "end_unix_ns": time.time_ns(),
        }
    )
    lifecycle.advance(TrialState.VALIDATE_TRIAL)
    lifecycle.advance(TrialState.NEXT_TRIAL_OR_END_SESSION)
    lifecycle.validate_complete()
    cue_visuals.hide()
    print(
        f"[OnlineFeedback] trial={plan.trial_index:03d} "
        f"phase={plan.task_phase} severity={plan.severity} "
        f"crossings={crossing_count} encounter={encounter_id or 'null'} "
        f"q1={query.q1_response or 'missing'} "
        f"off_protocol={bool(off_protocol_reasons)} task_success={task_success}",
        flush=True,
    )


def _run_collection(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    scenario: dict[str, Any],
    specs: Sequence[TrialSpec],
    plans: Sequence[TrialPlan],
) -> Path:
    global simulation_app
    output = (args.output or _default_output(args)).expanduser().resolve()
    runtime_report = validate_runtime_contract(
        PROJECT_ROOT, check_checkpoint_metadata=True
    )
    if not runtime_report["valid"]:
        raise RuntimeError(
            "frozen runtime contract failed: "
            + "; ".join(runtime_report["failures"])
        )
    if file_sha256(RUNTIME_CONFIG) != RUNTIME_CONTRACT_SHA256:
        raise RuntimeError("runtime contract manifest SHA-256 is not frozen")
    if not bool(runtime_report.get("runtime_handoff_ready", False)):
        raise RuntimeError("frozen BC+CBF A/C runtime handoff is not ready")
    if (
        not args.practice_only
        and not bool(runtime_report.get("production_collection_ready", False))
    ):
        raise RuntimeError(
            "study collection is fail-closed: the frozen runtime contract "
            "still has production_collection_ready=false. Complete the "
            "in-HMD practice smoke and explicitly qualify/version the runtime "
            "contract before pilot or production collection."
        )
    live_hmd_qualified = os.environ.get(
        "HRI_LIVE_HMD_QUALIFIED", "0"
    ).strip() == "1"
    source_lineage_reconciled = os.environ.get(
        "HRI_AC_SOURCE_LINEAGE_RECONCILED", "0"
    ).strip() == "1"
    if not args.practice_only and not live_hmd_qualified:
        raise RuntimeError(
            "pilot/production collection requires an operator-qualified live "
            "HMD/controller smoke: set HRI_LIVE_HMD_QUALIFIED=1 only after "
            "completing the manual checklist"
        )
    if not args.practice_only and not source_lineage_reconciled:
        raise RuntimeError(
            "pilot/production collection requires reconciliation of the "
            "held-out source hashes with the runtime handoff; set "
            "HRI_AC_SOURCE_LINEAGE_RECONCILED=1 only after the reviewed "
            "qualification evidence is versioned"
        )
    code_provenance = _resolve_verified_code_provenance(
        PROJECT_ROOT, str(args.code_commit_sha)
    )
    if (
        not args.practice_only
        and code_provenance["code_commit_verification"]
        != "verified_git_head_clean"
    ):
        raise RuntimeError(
            "pilot/production collection requires a clean Git checkout with "
            "a verifiable HEAD; source-tree-only provenance is practice-only"
        )
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        _simulation_config(),
        experience=os.environ.get("ISAAC_SIM_EXPERIENCE", ""),
    )
    env = trace = tracking_guard = recorder = feedback = video = None
    stop = {"reason": ""}
    collection_committed = {"value": False}

    def request_stop(signum, _frame) -> None:
        if not stop["reason"]:
            stop["reason"] = f"signal_{int(signum)}"

    def current_abort_reason() -> str:
        if stop["reason"]:
            return str(stop["reason"])
        running = getattr(simulation_app, "is_running", None)
        if callable(running) and not running():
            stop["reason"] = "simulation_app_stopped"
        return str(stop["reason"])

    def raise_if_stopped() -> None:
        reason = current_abort_reason()
        if reason:
            raise CollectionAbort(reason)

    previous_signals = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        from v3_chan.ac_feedback.live_vr import LiveVRTrackingProvider
        from v3_chan.ac_feedback.xr_feedback import (
            ControllerInputWatchdog,
            EmergencyAbortMonitor,
            IsaacXRButtonSource,
            RealtimeButtonMapping,
            XRFeedbackController,
        )

        watchdog = TrackingWatchdog(max_pose_age_ms=args.max_pose_age_ms)
        provider = LiveVRTrackingProvider(watchdog)
        policy = FrozenBCPolicy(
            PROJECT_ROOT / POLICY_RELATIVE_PATH, device=args.device
        )
        env = _build_environment(provider, seed=args.seed)
        apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
        env.world.pause()
        provider.setup(env.world, simulation_app)
        env.world.reset()
        env.world.reset()
        apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
        env.world.pause()
        provider.wait_until_ready(
            simulation_app,
            timeout_s=args.tracking_timeout_s,
            stable_frames=args.tracking_stable_frames,
            stop_requested=raise_if_stopped,
        )
        button_source = IsaacXRButtonSource(avatar=provider.avatar)
        controller_discovery = button_source.discover()
        disconnected = [
            hand
            for hand in ("left", "right")
            if not bool(controller_discovery.get(hand, {}).get("connected", False))
        ]
        if disconnected:
            raise CollectionAbort(
                "VR controller discovery failed for: " + ", ".join(disconnected)
            )
        joint_names, arm_joint_indices = validate_environment_contract(
            env,
            expected_max_episode_steps=12_000,
        )
        if len(env.cubes) != 1:
            raise RuntimeError(
                f"single-pick contract requires one cube, got {len(env.cubes)}"
            )
        cbf_config = actual_cbf_config(
            env._cbf_filter, condition_id="A_reactive"
        )
        geometry_metadata = env.safety_geometry.metadata()
        protected_links = list(env.safety_geometry.available_links)
        protected_colliders = list(env.safety_geometry.collider_paths)
        if not protected_links or not protected_colliders:
            raise RuntimeError("runtime-discovered distal CBF geometry is empty")
        print(
            f"[OnlineFeedback] CBF protected links: {protected_links}", flush=True
        )
        print(
            f"[OnlineFeedback] CBF protected colliders: {protected_colliders}",
            flush=True,
        )
        proximal_tokens = tuple(
            str(value)
            for value in scenario["protected_crossing"]["proximal_link_tokens"]
        )
        proximal_monitor = ProximalArmProtocolMonitor(
            robot_prim_path=env.safety_geometry.robot_prim_path,
            proximal_link_tokens=proximal_tokens,
            hand_radius_m=env.safety_geometry.thresholds.hand_radius_m,
        )
        if not proximal_monitor.collider_paths:
            raise RuntimeError(
                "runtime-discovered proximal off-protocol geometry is empty"
            )
        missing_proximal_tokens = [
            token
            for token in proximal_tokens
            if not any(
                f"/{token}" in path
                for path in proximal_monitor.collider_paths
            )
        ]
        if missing_proximal_tokens:
            raise RuntimeError(
                "proximal off-protocol geometry is incomplete: "
                + ", ".join(missing_proximal_tokens)
            )
        overlap = sorted(
            set(proximal_monitor.collider_paths).intersection(protected_colliders)
        )
        if overlap:
            raise RuntimeError(
                "distal CBF and proximal off-protocol collider scopes overlap: "
                + ", ".join(overlap)
            )
        frozen_config = load_config(args.config)
        video = SynchronizedSpectatorVideoRecorder.from_config(
            output, frozen_config
        )
        video_ready = video.setup()
        if video.enabled and not video_ready:
            raise CollectionAbort(
                "spectator video was enabled but setup failed: "
                + str(video.metadata.get("setup_error", "unknown error"))
            )
        print(
            "[OnlineFeedback] proximal off-protocol colliders: "
            f"{list(proximal_monitor.collider_paths)}",
            flush=True,
        )
        recovery_config = env._state_aware_recovery_config.as_dict()
        physics_config = {
            "physics_dt_s": float(env.physics_dt_s),
            "max_episode_steps": int(env.config.max_episode_steps),
            "action_scale": float(env.config.action_scale),
            "action_version": str(env.config.action_version),
            "fixed_orientation": bool(env.config.fixed_orientation),
            "gripper_mode": str(env.config.gripper_mode),
            "cube_count": int(env.config.cube_count),
        }
        metadata = {
            "participant_id": str(args.participant_id),
            "session_id": str(args.session_id),
            "participant_split_key": str(args.participant_id),
            "session_seed": int(args.seed),
            **code_provenance,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "collector_config_path": str(Path(args.config).expanduser().resolve()),
            "collector_config_file_sha256": file_sha256(args.config),
            "collector_config_canonical_sha256": canonical_sha256(
                load_config(args.config)
            ),
            "live_hmd_qualified": int(live_hmd_qualified),
            "qualification_source_lineage_reconciled": int(
                source_lineage_reconciled
            ),
            "controller_input_discovery_json": json.dumps(
                controller_discovery,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "haptics_enabled": 0,
            "condition_assignment_source": (
                "participant_session_seed_preassigned_before_feedback"
            ),
            "spectator_video_metadata_json": json.dumps(
                video.metadata, sort_keys=True, separators=(",", ":")
            ),
            "protocol_config_sha256": canonical_sha256(protocol),
            "scenario_config_sha256": canonical_sha256(scenario),
            "cbf_config_sha256": canonical_sha256(cbf_config),
            "cbf_config_json": json.dumps(
                cbf_config, sort_keys=True, separators=(",", ":")
            ),
            "policy_metadata_json": json.dumps(
                policy.metadata, sort_keys=True, separators=(",", ":")
            ),
            "schedule_sha256": canonical_sha256(
                {
                    "trials": [plan.as_dict() for plan in plans],
                    "source": "preassigned_before_feedback",
                }
            ),
            "trial_order_json": json.dumps(
                [plan.trial_id for plan in plans], separators=(",", ":")
            ),
            "state_aware_recovery_config_json": json.dumps(
                recovery_config,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "state_aware_recovery_config_sha256": canonical_sha256(
                recovery_config
            ),
            "physics_config_json": json.dumps(
                physics_config, sort_keys=True, separators=(",", ":")
            ),
            "physics_config_sha256": canonical_sha256(physics_config),
            "cbf_protected_links_json": json.dumps(
                protected_links, separators=(",", ":")
            ),
            "cbf_protected_colliders_json": json.dumps(
                protected_colliders, separators=(",", ":")
            ),
            "cbf_missing_links_json": json.dumps(
                list(env.safety_geometry.missing_links), separators=(",", ":")
            ),
            "safety_geometry_metadata_json": json.dumps(
                geometry_metadata,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "proximal_protocol_monitor": (
                "usd_world_aabb_conservative_read_only_v1"
            ),
            "proximal_collider_paths_json": json.dumps(
                proximal_monitor.collider_paths, separators=(",", ":")
            ),
            "proximal_link_tokens_json": json.dumps(
                proximal_tokens, separators=(",", ":")
            ),
            "proximal_missing_tokens_json": "[]",
            "pilot": 1,
            "study_mode": str(protocol["study"]["selected_mode"]),
            "practice_only": int(args.practice_only),
            "smoke_test": int(args.practice_only and len(plans) == 1),
            "physics_dt_s": float(env.physics_dt_s),
            "task_max_episode_steps": int(env.config.max_episode_steps),
            "max_pose_age_ms": float(args.max_pose_age_ms),
            "tracking_stable_frames": int(args.tracking_stable_frames),
            "downstream_action_stage": (
                "gripper_merge_then_articulation_drive_state"
            ),
            "filtered_submitted_velocity_tolerance_rad_s": 1e-6,
            "applied_filtered_validation": (
                "full_vector_units_masks_and_stage_provenance_not_bitwise_equality"
            ),
            "human_feedback_runtime_effect": (
                "logging_only_never_policy_or_cbf_input"
            ),
            "runtime_handoff_ready": int(
                bool(runtime_report.get("runtime_handoff_ready", False))
            ),
            "runtime_production_collection_ready": int(
                bool(
                    runtime_report.get(
                        "production_collection_ready", False
                    )
                )
            ),
        }
        recorder = OnlineExplicitFeedbackRecorder(
            output,
            metadata=metadata,
            joint_names=joint_names,
            arm_joint_indices=arm_joint_indices,
            protocol_config=protocol,
            scenario_config=scenario,
            frozen_config=frozen_config,
            flush_every_steps=1,
        )
        recorder.write_schedule(plans)
        feedback_mapping = RealtimeButtonMapping.from_mapping(
            protocol["feedback"]["opposite_controller_buttons"]
        )
        feedback = XRFeedbackController(
            realtime_mapping=feedback_mapping,
            questionnaire_mapping=protocol["feedback"]["navigation"],
            questionnaire_timeout_s=float(
                protocol["feedback"]["query_timeout_s"]
            ),
            release_frames=int(protocol["feedback"]["stable_release_frames"]),
        )
        if not feedback.attach_xr_ui():
            raise CollectionAbort("in-HMD XRSceneView feedback UI is unavailable")
        navigation = protocol["feedback"]["navigation"]
        controller_input_watchdog = ControllerInputWatchdog()
        emergency_abort_monitor = EmergencyAbortMonitor(
            buttons=navigation["emergency_abort_chord"],
            hold_s=float(navigation["emergency_abort_hold_s"]),
        )
        _wait_for_controller_release(
            source=button_source,
            feedback=feedback,
            provider=provider,
            simulation_app=simulation_app,
            env=env,
            timeout_s=args.tracking_timeout_s,
            stop_requested=raise_if_stopped,
        )
        _set_instruction(
            feedback,
            "BC-only layout precheck 중입니다. 양손을 staging 위치에 유지하세요.",
        )
        initial_scene, restoration, layout_id, precheck_index = (
            _run_bc_only_precheck(
                env=env,
                policy=policy,
                provider=provider,
                recorder=recorder,
                protocol=protocol,
                scenario=scenario,
                seed=args.seed,
                simulation_app=simulation_app,
                proximal_monitor=proximal_monitor,
                arm_joint_indices=arm_joint_indices,
                stop_requested=raise_if_stopped,
            )
        )
        trace = ActionTraceRecorder(
            robot=env.robot, controller=env.controller, cbf=env._cbf_filter
        ).install()
        tracking_guard = PreApplyTrackingGuard(
            robot=env.robot,
            provider=provider,
            watchdog=watchdog,
            trace=trace,
            abort_reason_fn=lambda: stop["reason"],
        ).install()
        cue_visuals = CrossingCueVisuals(
            env.world,
            corridor_radius_m=float(scenario["crossing"]["corridor_radius_m"]),
            start_radius_m=float(scenario["crossing"]["start_radius_m"]),
            end_radius_m=float(scenario["crossing"]["end_radius_m"]),
        )
        for spec, plan in zip(specs, plans):
            _run_trial(
                spec=spec,
                plan=plan,
                env=env,
                policy=policy,
                provider=provider,
                watchdog=watchdog,
                trace=trace,
                recorder=recorder,
                feedback=feedback,
                button_source=button_source,
                controller_input_watchdog=controller_input_watchdog,
                emergency_abort_monitor=emergency_abort_monitor,
                video=video,
                cue_visuals=cue_visuals,
                proximal_monitor=proximal_monitor,
                protocol=protocol,
                scenario=scenario,
                restoration=restoration,
                initial_scene=initial_scene,
                layout_id=layout_id,
                layout_precheck_index=precheck_index,
                arm_joint_indices=arm_joint_indices,
                simulation_app=simulation_app,
                args=args,
                stop_requested=raise_if_stopped,
            )
        raise_if_stopped()
        apply_controlled_hold(env.robot, joint_count=len(joint_names))
        env.world.pause()
        # Snapshot after the last capture but before close changes the
        # recorder's transient ``available`` flag.  This is the same metadata
        # written to the JSON video sidecar during close.
        final_video_metadata = video.metadata
        if video.enabled and (
            int(final_video_metadata.get("frame_count", 0)) < 1
            or bool(final_video_metadata.get("capture_error", ""))
        ):
            raise CollectionAbort(
                "enabled spectator video produced no valid synchronized "
                "frames or reported a capture error"
            )
        video.close()
        recorder.update_spectator_video_metadata(final_video_metadata)
        partial = recorder.seal()
        from v3_chan.ac_feedback.validator import validate_collection

        report = validate_collection(
            partial,
            config_path=args.config,
            participant_id=str(args.participant_id),
            session_id=str(args.session_id),
            seed=int(args.seed),
            mode=str(protocol["study"]["selected_mode"]),
            allow_partial_path=True,
            raise_on_error=False,
        )
        if not report.valid:
            detail = "; ".join(
                f"[{issue.code}] {issue.path}: {issue.message}"
                for issue in report.issues[:16]
            )
            if not detail:
                detail = "artifact failed structural validation"
            raise CollectionAbort(
                "sealed collection failed validation: " + detail
            )
        if not bool(args.practice_only) and not report.study_eligible:
            # Study eligibility is an analysis/recollection decision, not a
            # file-integrity decision.  A complete off-protocol trial must be
            # preserved as a validated final artifact (with its trial-level
            # exclusion reason) instead of being mislabeled as a crash-only
            # ``.partial`` file.
            print(
                "[OnlineFeedback] WARNING: collection is structurally valid "
                "but study_eligible=false; preserve it as raw data and "
                "schedule replacement trials for excluded evaluated rows",
                flush=True,
            )
        final = _commit_validated_with_signal_barrier(
            recorder=recorder,
            abort_reason_fn=current_abort_reason,
            commit_state=collection_committed,
        )
        print(f"[OnlineFeedback] validated collection: {final}", flush=True)
        return final
    except BaseException as error:
        if trace is not None and trace.step_active:
            trace.abort_step()
        if env is not None:
            try:
                apply_controlled_hold(
                    env.robot, joint_count=len(env.robot.dof_names)
                )
                env.world.pause()
            except Exception:
                pass
        if recorder is not None and not getattr(recorder, "_closed", True):
            partial = recorder.abort(f"{type(error).__name__}:{error}")
            print(
                f"[OnlineFeedback] partial collection preserved: {partial}",
                flush=True,
            )
        raise
    finally:
        if feedback is not None:
            feedback.close()
        if video is not None:
            try:
                video.close()
            except Exception:
                pass
        if tracking_guard is not None:
            try:
                tracking_guard.uninstall()
            except Exception:
                pass
        if trace is not None:
            try:
                trace.uninstall()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _normalized_task_failure_reason(
    info: Mapping[str, Any],
    *,
    strict_state: Mapping[str, Any],
    task_success: bool,
    reset_after_query: bool,
    max_episode_steps: int,
) -> str:
    """Return an actual failure reason; success is never encoded as failure."""

    if bool(task_success):
        return ""
    terminal = str(info.get("task_terminal_reason", ""))
    if terminal == "success":
        terminal = ""
    return str(
        strict_state.get("failure_reason", "")
        or terminal
        or (
            "max_episode_steps"
            if int(info.get("step", 0)) >= int(max_episode_steps)
            else ("reset_after_query" if reset_after_query else "")
        )
    )


simulation_app = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        protocol, scenario = _load_configs(args)
        _assert_haptics_disabled(protocol)
        specs, plans = _build_schedule(args)
        if args.dry_run:
            payload = {
                "schema_version": protocol["schema_version"],
                "participant_id": args.participant_id,
                "session_id": args.session_id,
                "protocol_config_sha256": canonical_sha256(protocol),
                "scenario_config_sha256": canonical_sha256(scenario),
                "trial_count": len(plans),
                "practice_count": sum(plan.practice for plan in plans),
                "evaluated_count": sum(not plan.practice for plan in plans),
                "anchor_count": sum(plan.anchor_repeat for plan in plans),
                "pilot": True,
                "study_mode": protocol["study"]["selected_mode"],
                "haptics_enabled": False,
                "trials": [plan.as_dict() for plan in plans],
            }
            print(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2
                )
            )
            return 0
        _run_collection(args, protocol, scenario, specs, plans)
        return 0
    except KeyboardInterrupt:
        print(
            "[OnlineFeedback] interrupted; no final study file was produced",
            flush=True,
        )
        return 130
    except Exception as error:
        print(
            f"[OnlineFeedback] ABORTED: {type(error).__name__}: {error}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
