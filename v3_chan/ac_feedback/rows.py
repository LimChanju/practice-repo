"""Build and audit one transition row without importing Isaac Sim."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .action_trace import ActionStepTrace, CanonicalJointAction
from .policy import PolicyOutput
from .schema import ACTION_TRACE_SCHEMA_VERSION, ProtocolViolation
from .tracking import ENTITY_NAMES, TrackingSnapshot


def validate_runtime_step(
    *,
    info_before: Mapping[str, Any],
    info_after: Mapping[str, Any],
    trace: ActionStepTrace,
    tracking: TrackingSnapshot,
    arm_joint_indices: Sequence[int],
    max_joint_speed_rad_s: float = 2.0,
) -> None:
    """Fail closed on an incomplete or failed BC+CBF action pipeline."""

    if trace.schema_version != ACTION_TRACE_SCHEMA_VERSION:
        raise ProtocolViolation(
            f"action trace schema mismatch: {trace.schema_version!r}"
        )
    if not trace.pipeline_complete:
        raise ProtocolViolation("action pipeline is incomplete")
    if not np.array_equal(tracking.valid_mask, np.ones(3, dtype=np.int8)):
        raise ProtocolViolation("required live tracking is invalid for control row")
    if not _exact_bool(info_before.get("geometry_valid")):
        raise ProtocolViolation("human-robot safety geometry is invalid")
    left_gap = float(info_before.get("left_end_effector_surface_gap_m", math.nan))
    right_gap = float(info_before.get("right_end_effector_surface_gap_m", math.nan))
    minimum_gap = float(
        info_before.get("min_hand_end_effector_surface_gap", math.nan)
    )
    if not all(math.isfinite(value) for value in (left_gap, right_gap, minimum_gap)):
        raise ProtocolViolation("human-robot surface gaps are non-finite")
    if not math.isclose(
        minimum_gap,
        min(left_gap, right_gap),
        rel_tol=0.0,
        # The flattened observation stores the minimum as float32 while the
        # per-hand geometry diagnostics remain float64.
        abs_tol=1e-6,
    ):
        raise ProtocolViolation("minimum human-robot surface gap is inconsistent")
    if str(info_after.get("physical_safety_controller", "")) != "cbf":
        raise ProtocolViolation("physical safety controller is not pure CBF")
    cbf = _mapping(info_after.get("physical_safety"))
    if not cbf:
        raise ProtocolViolation("CBF diagnostics are missing")
    if not _exact_bool(cbf.get("intervention_available")):
        raise ProtocolViolation("CBF intervention was unavailable")
    if _exact_bool(cbf.get("fallback_applied")):
        raise ProtocolViolation("CBF applied a fallback stop")
    if not _exact_bool(cbf.get("feasible")):
        raise ProtocolViolation("CBF solution was infeasible")
    if not _exact_bool(cbf.get("solver_converged")):
        raise ProtocolViolation("CBF solver did not converge")
    failures = tuple(str(value) for value in cbf.get("failure_reasons", ()))
    if failures:
        raise ProtocolViolation("CBF reported failures: " + ", ".join(failures))
    if _exact_bool(cbf.get("relaxed_solution_applied")):
        raise ProtocolViolation("CBF applied a relaxed solution")
    if _exact_bool(cbf.get("intentional_human_absence")):
        raise ProtocolViolation("intentional human absence is forbidden in live collection")
    constraint_count = int(cbf.get("constraint_count", -1))
    active = _exact_bool(cbf.get("active"))
    if active != (constraint_count > 0):
        raise ProtocolViolation("CBF active flag and constraint count disagree")
    if constraint_count < 0 or constraint_count > 2:
        raise ProtocolViolation("CBF constraint count is outside [0, 2]")
    if int(cbf.get("tracked_hand_count", -1)) != 2:
        raise ProtocolViolation("CBF did not receive both tracked hands")
    if int(cbf.get("valid_hand_count", -1)) != 2:
        raise ProtocolViolation("CBF did not validate both tracked hands")
    if not active:
        status = str(cbf.get("status", ""))
        if status == "smooth_intervention_tail":
            if (
                str(cbf.get("objective_mode", "")) != "smooth_intervention"
                or not math.isclose(
                    float(cbf.get("correction_smoothness_weight", math.nan)),
                    4.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ProtocolViolation(
                    "smooth tail is only valid for frozen C_smooth/lambda_s=4"
                )
            indices = np.asarray(tuple(arm_joint_indices), dtype=np.int64)
            correction = np.asarray(
                cbf.get("correction_radps", ()), dtype=np.float64
            ).reshape(-1)
            if (
                correction.shape != (indices.size,)
                or not np.all(np.isfinite(correction))
                or float(np.linalg.norm(correction)) <= 0.0
            ):
                raise ProtocolViolation(
                    "smooth tail lacks a finite non-zero correction vector"
                )
        else:
            _require_same_stage(
                trace.nominal_rmpflow,
                trace.cbf_filtered,
                label="inactive CBF passthrough",
            )
    else:
        indices = np.asarray(tuple(arm_joint_indices), dtype=np.int64)
        velocity = trace.cbf_filtered.joint_velocities[indices]
        mask = trace.cbf_filtered.joint_velocities_mask[indices]
        if not np.all(mask):
            raise ProtocolViolation("active CBF filtered velocity is incomplete")
        if np.any(np.abs(velocity) > float(max_joint_speed_rad_s) + 1e-6):
            raise ProtocolViolation("active CBF velocity exceeds the configured cap")


def build_transition_row(
    *,
    source_step: int,
    observation_monotonic_ns: int,
    preflight_monotonic_ns: int,
    wall_time_unix_ns: int,
    obs_raw: np.ndarray,
    obs_next: np.ndarray,
    policy_output: PolicyOutput,
    trace: ActionStepTrace,
    tracking: TrackingSnapshot,
    tracking_next: TrackingSnapshot,
    measured_positions_before: np.ndarray,
    measured_velocities_before: np.ndarray,
    measured_positions_after: np.ndarray,
    measured_velocities_after: np.ndarray,
    info_before: Mapping[str, Any],
    info_after: Mapping[str, Any],
    terminated: bool,
    truncated: bool,
    post_step_monotonic_ns: int,
    arm_joint_indices: Sequence[int],
    encounter_id: str,
    encounter_state: str,
) -> dict[str, Any]:
    joint_count = len(trace.joint_names)
    arm_indices = np.asarray(tuple(arm_joint_indices), dtype=np.int64)
    if arm_indices.ndim != 1 or arm_indices.size <= 0:
        raise ProtocolViolation("arm joint indices are invalid")
    strict_payload = info_after.get("strict_task_semantics", {})
    strict_state = (
        strict_payload.get("state", {})
        if isinstance(strict_payload, Mapping)
        else {}
    )
    terminal_reason = str(
        (
            strict_state.get("failure_reason", "")
            if isinstance(strict_state, Mapping)
            else ""
        )
        or info_after.get("task_terminal_reason", "")
        or ""
    )
    if terminal_reason == "success":
        terminal_reason = ""
    row: dict[str, Any] = {
        "control/source_step": int(source_step),
        "control/result_step": int(info_after.get("step", source_step + 1)),
        "control/sim_time": float(info_after.get("sim_time", 0.0)),
        "control/observation_monotonic_ns": int(observation_monotonic_ns),
        "control/preflight_monotonic_ns": int(preflight_monotonic_ns),
        "control/wall_time_unix_ns": int(wall_time_unix_ns),
        "control/policy_inference_started_monotonic_ns": int(
            policy_output.inference_started_monotonic_ns
        ),
        "control/policy_output_monotonic_ns": int(
            policy_output.inference_completed_monotonic_ns
        ),
        "control/rmpflow_output_monotonic_ns": int(
            trace.rmpflow_output_monotonic_ns
        ),
        "control/cbf_output_monotonic_ns": int(trace.cbf_output_monotonic_ns),
        "control/action_command_monotonic_ns": int(
            trace.action_command_monotonic_ns
        ),
        "control/next_observation_monotonic_ns": int(
            tracking_next.sample_monotonic_ns
        ),
        "control/post_step_monotonic_ns": int(post_step_monotonic_ns),
        "observations/obs_raw": _finite_vector(
            obs_raw, 84, "obs_raw"
        ).astype(np.float32),
        "observations/obs_policy": _finite_vector(
            policy_output.policy_input, 84, "obs_policy"
        ).astype(np.float32),
        "observations/obs_next": _finite_vector(obs_next, 84, "obs_next").astype(
            np.float32
        ),
        "safety/left_surface_gap_m": float(
            info_before.get("left_end_effector_surface_gap_m", 10.0)
        ),
        "safety/right_surface_gap_m": float(
            info_before.get("right_end_effector_surface_gap_m", 10.0)
        ),
        "safety/min_surface_gap_m": float(
            info_before.get("min_hand_end_effector_surface_gap", 10.0)
        ),
        "safety/geometry_valid": int(bool(info_before.get("geometry_valid", False))),
        "safety/collision": int(bool(info_before.get("human_robot_collision", False))),
        "safety/near": int(bool(info_before.get("near_human", False))),
        "safety/near_miss": int(bool(info_before.get("near_miss", False))),
        "actions/bc_task_action": _finite_vector(
            policy_output.action, 5, "bc_task_action"
        ).astype(np.float32),
        "actions/applied_arm_joint_velocities_rad_s": (
            trace.post_applied.joint_velocities[arm_indices].copy()
        ),
        "actions/applied_arm_joint_velocities_mask": (
            trace.post_applied.joint_velocities_mask[arm_indices].astype(np.int8)
        ),
        "actions/arm_action_submitted": int(trace.arm_action_submitted),
        "actions/action_pipeline_complete": int(trace.pipeline_complete),
        "actions/gripper_command": str(info_after.get("gripper_command", "") or ""),
        "robot/measured_joint_positions_before_rad": _finite_vector(
            measured_positions_before, joint_count, "measured_positions_before"
        ),
        "robot/measured_joint_velocities_before_rad_s": _finite_vector(
            measured_velocities_before, joint_count, "measured_velocities_before"
        ),
        "robot/measured_joint_positions_after_rad": _finite_vector(
            measured_positions_after, joint_count, "measured_positions_after"
        ),
        "robot/measured_joint_velocities_after_rad_s": _finite_vector(
            measured_velocities_after, joint_count, "measured_velocities_after"
        ),
        "task/controller_event_before": int(info_before.get("controller_event", -1)),
        "task/controller_event_after": int(info_after.get("controller_event", -1)),
        "task/controller_t_before": float(info_before.get("controller_t", 0.0)),
        "task/controller_t_after": float(info_after.get("controller_t", 0.0)),
        "task/success": int(bool(info_after.get("success", False))),
        "task/terminated": int(bool(terminated)),
        "task/truncated": int(bool(truncated)),
        "task/terminal_reason": terminal_reason,
        "encounter/row_encounter_id": str(encounter_id),
        "encounter/state": str(encounter_state),
    }
    _add_tracking_snapshot(row, "human", tracking)
    _add_tracking_snapshot(row, "human_next", tracking_next)
    for name, stage in (
        ("nominal_rmpflow", trace.nominal_rmpflow),
        ("cbf_filtered", trace.cbf_filtered),
        ("submitted", trace.submitted),
        ("pre_applied", trace.pre_applied),
        ("applied", trace.post_applied),
    ):
        _add_action_stage(row, name, stage)
    cbf = _mapping(info_after.get("physical_safety"))
    aliases = {
        "max_constraint_violation_before_mps": "max_constraint_violation_before",
        "max_constraint_violation_after_mps": "max_constraint_violation_after",
        "slack_mps": "slack_radps",
        "buffered_current_gap_m": "min_predicted_gap_m",
    }
    for name in (
        "intervention_norm_radps",
        "nominal_velocity_norm_radps",
        "filtered_velocity_norm_radps",
        "max_constraint_violation_before_mps",
        "max_constraint_violation_after_mps",
        "slack_mps",
        "buffered_current_gap_m",
        "solve_time_ms",
    ):
        value = cbf.get(name, cbf.get(aliases.get(name, ""), 0.0))
        row[f"cbf/{name}"] = float(value)
    for name in (
        "active",
        "intervention_available",
        "constraint_count",
        "tracked_hand_count",
        "valid_hand_count",
        "feasible",
        "fallback_applied",
        "solver_converged",
        "infeasibility_proven",
        "relaxed_solution_available",
        "relaxed_solution_applied",
        "intentional_human_absence",
    ):
        row[f"cbf/{name}"] = int(cbf.get(name, 0))
    row["cbf/failure_reasons_json"] = json.dumps(
        list(cbf.get("failure_reasons", ())), separators=(",", ":")
    )
    for name in ("status", "solver_backend", "projection_status"):
        row[f"cbf/{name}"] = str(cbf.get(name, ""))
    _validate_row_timestamps(row)
    return row


def _add_action_stage(
    row: dict[str, Any], name: str, stage: CanonicalJointAction
) -> None:
    for channel, unit in (
        ("joint_positions", "rad"),
        ("joint_velocities", "rad_s"),
        ("joint_efforts", "nm"),
    ):
        base = f"actions/{name}_{channel}_{unit}"
        row[base] = getattr(stage, channel).copy()
        row[base + "_mask"] = getattr(stage, f"{channel}_mask").astype(np.int8)


def _add_tracking_snapshot(
    row: dict[str, Any], namespace: str, tracking: TrackingSnapshot
) -> None:
    row[f"{namespace}/valid_mask"] = tracking.valid_mask.astype(np.int8)
    for entity in ENTITY_NAMES:
        pose = getattr(tracking, entity)
        row.update(
            {
                f"{namespace}/{entity}_position_world_m": pose.position_world.copy(),
                f"{namespace}/{entity}_pose_valid": int(pose.pose_valid),
                f"{namespace}/{entity}_position_tracked": int(
                    pose.position_tracked
                ),
                f"{namespace}/{entity}_tracking_status_known": int(
                    pose.tracking_status_known
                ),
                f"{namespace}/{entity}_source_name": pose.source_name,
                f"{namespace}/{entity}_source_path": pose.source_path,
                f"{namespace}/{entity}_acquisition_monotonic_ns": int(
                    pose.acquisition_monotonic_ns
                ),
                f"{namespace}/{entity}_pose_age_ms": float(pose.pose_age_ms),
                f"{namespace}/{entity}_source_switched": int(
                    pose.source_switched
                ),
            }
        )


def _require_same_stage(
    first: CanonicalJointAction,
    second: CanonicalJointAction,
    *,
    label: str,
) -> None:
    for channel in ("joint_positions", "joint_velocities", "joint_efforts"):
        first_mask = getattr(first, f"{channel}_mask")
        second_mask = getattr(second, f"{channel}_mask")
        if not np.array_equal(first_mask, second_mask):
            raise ProtocolViolation(f"{label} changed {channel} mask")
        if not np.array_equal(getattr(first, channel), getattr(second, channel)):
            raise ProtocolViolation(f"{label} changed {channel} values")


def _validate_row_timestamps(row: Mapping[str, Any]) -> None:
    ordered = (
        int(row["control/observation_monotonic_ns"]),
        int(row["control/preflight_monotonic_ns"]),
        int(row["control/policy_inference_started_monotonic_ns"]),
        int(row["control/policy_output_monotonic_ns"]),
        int(row["control/rmpflow_output_monotonic_ns"]),
        int(row["control/cbf_output_monotonic_ns"]),
        int(row["control/action_command_monotonic_ns"]),
        int(row["control/next_observation_monotonic_ns"]),
        int(row["control/post_step_monotonic_ns"]),
    )
    if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
        raise ProtocolViolation("transition timestamps are out of order")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _exact_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ProtocolViolation(f"expected an exact boolean, got {value!r}")


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (int(size),) or not np.all(np.isfinite(array)):
        raise ProtocolViolation(f"{label} must be a finite ({int(size)},) vector")
    return array.copy()
