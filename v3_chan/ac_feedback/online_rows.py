"""Per-control-step rows for the A/C selective-smoothing feedback study.

The existing action tracer remains authoritative for the BC -> RMPFlow -> CBF
-> articulation boundary.  This module only extends that evidence with the
single-crossing protocol, raw XR orientation/velocity, object state, and clear
runtime-vs-outcome provenance.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from .recorder import DatasetSpec, _row_specs
from .rows import build_transition_row
from .online_schema import CONDITION_CONTRACT, RESPONSE_PHASES, TRIAL_STATES
from .schema import POLICY_SHA256


def online_row_specs(joint_count: int, arm_count: int) -> dict[str, DatasetSpec]:
    specs = dict(_row_specs(joint_count, arm_count))
    additions: dict[str, DatasetSpec] = {
        "control/unix_time_ns": DatasetSpec(np.int64),
        "control/control_mode": DatasetSpec("str"),
        "control/trial_state": DatasetSpec("str"),
        "control/trial_id": DatasetSpec("str"),
        "control/query_id": DatasetSpec("str"),
        "controller/condition_id": DatasetSpec("str"),
        "controller/objective_mode": DatasetSpec("str"),
        "controller/lambda_s": DatasetSpec(np.float64),
        "policy/bc_raw_action_5d": DatasetSpec(np.float32, (5,)),
        "policy/bc_checkpoint_sha256": DatasetSpec("str"),
        "human/left_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human/right_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human/left_velocity_raw_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/right_velocity_raw_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/left_velocity_filtered_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/right_velocity_filtered_world_m_s": DatasetSpec(np.float64, (3,)),
        "human/left_velocity_valid": DatasetSpec(np.int8),
        "human/right_velocity_valid": DatasetSpec(np.int8),
        "human_next/left_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "human_next/right_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "robot/ee_position_world_m": DatasetSpec(np.float64, (3,)),
        "robot/ee_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "robot/ee_linear_velocity_world_m_s": DatasetSpec(np.float64, (3,)),
        "object/cube_position_world_m": DatasetSpec(np.float64, (3,)),
        "object/cube_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "object/cube_linear_velocity_world_m_s": DatasetSpec(np.float64, (3,)),
        "object/goal_position_world_m": DatasetSpec(np.float64, (3,)),
        "object/goal_orientation_wxyz": DatasetSpec(np.float64, (4,)),
        "task/experimental_phase": DatasetSpec("str"),
        "task/phase_advance_enabled": DatasetSpec(np.int8),
        "task/has_grasped": DatasetSpec(np.int8),
        "task/release_observed": DatasetSpec(np.int8),
        "task/object_drop": DatasetSpec(np.int8),
        "task/gripper_closed": DatasetSpec(np.int8),
        "task/attachment_state": DatasetSpec("str"),
        "task/attachment_state_semantics": DatasetSpec("str"),
        "task/release_state": DatasetSpec("str"),
        "actions/controller_task_action": DatasetSpec(np.float32, (5,)),
        "actions/controller_task_action_reason": DatasetSpec("str"),
        "cbf/active_constraint_identity_available": DatasetSpec(np.int8),
        "cbf/active_constraint_summary_json": DatasetSpec("str"),
        "cbf/constraint_evidence_json": DatasetSpec("str"),
        "cbf/min_barrier_value_m": DatasetSpec(np.float64),
        "cbf/max_active_prediction_buffer_m": DatasetSpec(np.float64),
        "cbf/max_prediction_buffer_m": DatasetSpec(np.float64),
        "cbf/fail_closed_on_invalid_active_hand": DatasetSpec(np.int8),
        "cbf/stop_on_infeasible": DatasetSpec(np.int8),
        "cbf/fail_closed_status": DatasetSpec("str"),
        "cbf/active_hand": DatasetSpec("str"),
        "cbf/active_robot_link": DatasetSpec("str"),
        "cbf/active_robot_collider": DatasetSpec("str"),
        "cbf/identity_semantics": DatasetSpec("str"),
        "cbf/current_correction_rad_s": DatasetSpec(np.float64, (arm_count,)),
        "cbf/previous_correction_rad_s": DatasetSpec(np.float64, (arm_count,)),
        "cbf/correction_delta_rad_s": DatasetSpec(np.float64, (arm_count,)),
        "cbf/correction_memory_valid": DatasetSpec(np.int8),
        "response/phase": DatasetSpec("str"),
        "response/smooth_tail_active": DatasetSpec(np.int8),
        "response/time_since_response_onset_s": DatasetSpec(np.float64),
        "response/time_since_recovery_onset_s": DatasetSpec(np.float64),
        "recovery/active": DatasetSpec(np.int8),
        "recovery/onset_now": DatasetSpec(np.int8),
        "recovery/end_now": DatasetSpec(np.int8),
        "recovery/state": DatasetSpec("str"),
        "recovery/control_authority": DatasetSpec(np.int8),
        "recovery/bc_resumed": DatasetSpec(np.int8),
        "recovery/stable_task_resumption": DatasetSpec(np.int8),
        "safety/min_ttc_s": DatasetSpec(np.float64),
        "safety/ttc_valid": DatasetSpec(np.int8),
        "safety/dynamic_measurement_valid": DatasetSpec(np.int8),
        "safety/max_closing_speed_m_s": DatasetSpec(np.float64),
        "safety/closing": DatasetSpec(np.int8),
        "safety/left_proximal_surface_gap_m": DatasetSpec(np.float64),
        "safety/right_proximal_surface_gap_m": DatasetSpec(np.float64),
        "safety/left_proximal_collider_path": DatasetSpec("str"),
        "safety/right_proximal_collider_path": DatasetSpec("str"),
        "safety/static_collision": DatasetSpec(np.int8),
        "safety/self_collision": DatasetSpec(np.int8),
        "safety/static_contact_semantic_class": DatasetSpec("str"),
        "crossing/active": DatasetSpec(np.int8),
        "crossing/state": DatasetSpec("str"),
        "crossing/started_now": DatasetSpec(np.int8),
        "crossing/intended_completed_now": DatasetSpec(np.int8),
        "crossing/reverse_completed_now": DatasetSpec(np.int8),
        "crossing/progress_fraction": DatasetSpec(np.float64),
        "crossing/perpendicular_deviation_m": DatasetSpec(np.float64),
        "crossing/intended_hand": DatasetSpec("str"),
        "crossing/non_crossing_hand": DatasetSpec("str"),
        "crossing/observed_hand": DatasetSpec("str"),
        "provenance/runtime_feature_cutoff_available": DatasetSpec(np.int8),
        "provenance/decision_context_available": DatasetSpec(np.int8),
    }
    overlap = sorted(set(specs).intersection(additions))
    if overlap:
        raise RuntimeError(f"online row spec unexpectedly overlaps base paths: {overlap}")
    specs.update(additions)
    return specs


def build_online_transition_row(
    *,
    controller_task_action: Sequence[float],
    controller_task_action_reason: str,
    control_mode: str,
    trial_state: str,
    trial_id: str,
    query_id: str,
    condition_id: str,
    objective_mode: str,
    lambda_s: float,
    response_phase: str,
    decision_context_available: bool,
    experimental_phase: str,
    phase_advance_enabled: bool,
    object_state: Mapping[str, Any],
    ee_state: Mapping[str, Any],
    crossing_state: Mapping[str, Any],
    proximal_state: Mapping[str, Any] | None = None,
    response_onset_simulation_time_s: float | None = None,
    recovery_onset_simulation_time_s: float | None = None,
    recovery_onset_now: bool = False,
    recovery_end_now: bool = False,
    **base_kwargs: Any,
) -> dict[str, Any]:
    """Build one strict row while retaining the proven existing base checks."""

    row = build_transition_row(**base_kwargs)
    info_before = dict(base_kwargs["info_before"])
    info_after = dict(base_kwargs["info_after"])
    tracking = base_kwargs["tracking"]
    tracking_next = base_kwargs["tracking_next"]
    dynamic = _mapping(info_before.get("dynamic_safety"))
    proximal = _mapping(proximal_state)
    cbf = _mapping(info_after.get("physical_safety"))
    cbf_active = bool(cbf.get("active", False))
    expected_condition = CONDITION_CONTRACT.get(str(condition_id))
    if expected_condition is None or (
        str(objective_mode), float(lambda_s)
    ) != expected_condition:
        raise ValueError("condition_id/objective_mode/lambda_s violates A/C contract")
    actual_objective = str(cbf.get("objective_mode", ""))
    actual_lambda = _finite_or_default(
        cbf.get("correction_smoothness_weight"), -1.0
    )
    if actual_objective != str(objective_mode) or actual_lambda != float(lambda_s):
        raise ValueError("logged CBF objective/lambda differs from assigned condition")
    phase = str(getattr(response_phase, "value", response_phase))
    if phase not in RESPONSE_PHASES:
        raise ValueError(f"invalid safety-response phase: {phase!r}")
    mode = str(control_mode)
    if mode not in {"nominal_bc", "protocol_hold"}:
        raise ValueError(f"invalid online control mode: {mode!r}")
    lifecycle_state = str(getattr(trial_state, "value", trial_state))
    if lifecycle_state not in TRIAL_STATES:
        raise ValueError(f"invalid trial FSM state: {lifecycle_state!r}")
    arm_count = len(tuple(base_kwargs["arm_joint_indices"]))
    current_correction = _vec(
        cbf.get("correction_radps"), arm_count, "current CBF correction"
    )
    previous_correction = _vec(
        cbf.get("previous_correction_radps"),
        arm_count,
        "previous CBF correction",
    )
    correction_delta = current_correction - previous_correction
    smooth_tail_active = str(cbf.get("status", "")) == "smooth_intervention_tail"
    recovery = _mapping(info_after.get("state_aware_recovery"))
    recovery_state = _mapping(recovery.get("state"))
    recovery_control_authority = bool(recovery.get("control_authority", False))
    # On the accepted handoff step the bridge state has already become
    # inactive, while the pre-step Recovery command still owned this applied
    # control transition.  Preserve that authority as part of Recovery.
    recovery_active = bool(
        recovery_state.get("active", False) or recovery_control_authority
    )
    bc_resumed = mode == "nominal_bc" and phase in {
        "BC_RESUMED", "STABLE_TASK_RESUMPTION",
    }
    stable_task_resumption = phase == "STABLE_TASK_RESUMPTION"
    constraint_evidence = _constraint_evidence(cbf)
    constraint_count = int(cbf.get("constraint_count", 0))
    if len(constraint_evidence) != constraint_count:
        raise ValueError(
            "CBF constraint_evidence count differs from constraint_count"
        )
    barrier_values = [item["barrier_value_m"] for item in constraint_evidence]
    prediction_buffers = [
        item["prediction_buffer_m"] for item in constraint_evidence
    ]
    failure_reasons = tuple(str(value) for value in cbf.get("failure_reasons", ()))
    fail_closed_status = (
        "controlled_stop_applied"
        if bool(cbf.get("fallback_applied", False))
        else (
            "safe_solution_no_fallback"
            if bool(cbf.get("feasible", False))
            and bool(cbf.get("solver_converged", False))
            and not failure_reasons
            else "invalid_or_unresolved"
        )
    )
    strict_state = _mapping(
        _mapping(info_after.get("strict_task_semantics")).get("state")
    )
    terminal_failure = str(
        strict_state.get("failure_reason", "")
        or info_after.get("task_terminal_reason", "")
    ).lower()
    has_grasped = bool(info_after.get("has_grasped_cube", False))
    event_after = int(info_after.get("controller_event", -1))
    sim_time_s = _finite_or_default(info_after.get("sim_time"), 0.0)
    time_since_response = _elapsed_or_sentinel(
        sim_time_s, response_onset_simulation_time_s
    )
    time_since_recovery = _elapsed_or_sentinel(
        sim_time_s, recovery_onset_simulation_time_s
    )

    active_constraints = [
        {
            "hand": item["hand"],
            "robot_link": item["closest_link"],
            "collider_path": item["closest_collider_path"],
            "surface_gap_m": item["raw_surface_gap_m"],
        }
        for item in constraint_evidence
    ]
    identity_available = bool(
        cbf_active
        and len(active_constraints) == int(cbf.get("constraint_count", -1))
        and all(
            item["hand"] in {"left", "right"}
            and item["robot_link"]
            and item["collider_path"]
            for item in active_constraints
        )
    )
    identity_semantics = (
        "exact_reconstruction_from_frozen_cbf_filter_input_v1"
        if identity_available
        else "inactive_or_identity_unavailable"
    )
    active_hands = ",".join(item["hand"] for item in active_constraints)
    active_links = ",".join(item["robot_link"] for item in active_constraints)
    active_colliders = ",".join(
        item["collider_path"] for item in active_constraints
    )
    summary = {
        "constraint_count": int(cbf.get("constraint_count", 0)),
        "exact_identity_available": identity_available,
        "constraints": active_constraints,
        "derivation": (
            "same safety_result and observation consumed by "
            "DistalLinkVelocityCBF._constraints; successful runtime row "
            "proves constructed count equals diagnostics"
        ),
    }

    min_ttc = _finite_or_default(dynamic.get("min_ttc_s"), 10.0)
    ttc_valid = bool(dynamic.get("ttc_valid", False))
    closing_speed = max(
        0.0, _finite_or_default(dynamic.get("max_closing_speed_mps"), 0.0)
    )
    intended_hand = str(crossing_state.get("intended_hand", ""))
    if intended_hand not in {"left", "right"}:
        raise ValueError("crossing intended hand must be exactly left or right")
    non_crossing_hand = "right" if intended_hand == "left" else "left"
    row.update(
        {
            "control/unix_time_ns": int(base_kwargs["wall_time_unix_ns"]),
            "control/control_mode": mode,
            "control/trial_state": lifecycle_state,
            "control/trial_id": str(trial_id),
            "control/query_id": str(query_id),
            "controller/condition_id": str(condition_id),
            "controller/objective_mode": actual_objective,
            "controller/lambda_s": actual_lambda,
            "policy/bc_raw_action_5d": _vec(
                row["actions/bc_task_action"], 5, "BC raw action"
            ).astype(np.float32),
            "policy/bc_checkpoint_sha256": POLICY_SHA256,
            "human/left_orientation_wxyz": _vec(
                tracking.left.orientation_wxyz, 4, "left orientation"
            ),
            "human/right_orientation_wxyz": _vec(
                tracking.right.orientation_wxyz, 4, "right orientation"
            ),
            "human_next/left_orientation_wxyz": _vec(
                tracking_next.left.orientation_wxyz, 4, "next left orientation"
            ),
            "human_next/right_orientation_wxyz": _vec(
                tracking_next.right.orientation_wxyz, 4, "next right orientation"
            ),
            "human/left_velocity_raw_world_m_s": _vec_or_zero(
                dynamic.get("left_hand_vel_raw_mps"), 3
            ),
            "human/right_velocity_raw_world_m_s": _vec_or_zero(
                dynamic.get("right_hand_vel_raw_mps"), 3
            ),
            "human/left_velocity_filtered_world_m_s": _vec_or_zero(
                dynamic.get("left_hand_vel_filtered_mps"), 3
            ),
            "human/right_velocity_filtered_world_m_s": _vec_or_zero(
                dynamic.get("right_hand_vel_filtered_mps"), 3
            ),
            "human/left_velocity_valid": int(
                bool(dynamic.get("left_hand_velocity_valid", False))
            ),
            "human/right_velocity_valid": int(
                bool(dynamic.get("right_hand_velocity_valid", False))
            ),
            "robot/ee_position_world_m": _vec(
                ee_state.get("position_world_m"), 3, "EE position"
            ),
            "robot/ee_orientation_wxyz": _vec(
                ee_state.get("orientation_wxyz"), 4, "EE orientation"
            ),
            "robot/ee_linear_velocity_world_m_s": _vec(
                ee_state.get("linear_velocity_world_m_s"), 3, "EE velocity"
            ),
            "object/cube_position_world_m": _vec(
                object_state.get("cube_position_world_m"), 3, "cube position"
            ),
            "object/cube_orientation_wxyz": _vec(
                object_state.get("cube_orientation_wxyz"), 4, "cube orientation"
            ),
            "object/cube_linear_velocity_world_m_s": _vec(
                object_state.get("cube_linear_velocity_world_m_s"),
                3,
                "cube linear velocity",
            ),
            "object/goal_position_world_m": _vec(
                object_state.get("goal_position_world_m"), 3, "goal position"
            ),
            "object/goal_orientation_wxyz": _vec(
                object_state.get("goal_orientation_wxyz"), 4, "goal orientation"
            ),
            "task/experimental_phase": str(experimental_phase),
            "task/phase_advance_enabled": int(bool(phase_advance_enabled)),
            "task/has_grasped": int(has_grasped),
            "task/release_observed": int(
                bool(strict_state.get("success_latched", False))
                or (event_after >= 7 and not has_grasped)
            ),
            "task/object_drop": int(
                "drop" in terminal_failure
                or "grasp_lost" in terminal_failure
            ),
            "task/gripper_closed": int(bool(info_after.get("gripper_closed", False))),
            "task/attachment_state": (
                "physically_grasped" if has_grasped else "not_physically_grasped"
            ),
            "task/attachment_state_semantics": "has_grasped_cube_runtime_proxy_v1",
            "task/release_state": (
                "released_and_settled"
                if bool(strict_state.get("success_latched", False))
                else ("release_observed" if event_after >= 7 and not has_grasped else "not_released")
            ),
            "actions/controller_task_action": _vec(
                controller_task_action, 5, "controller task action"
            ).astype(np.float32),
            "actions/controller_task_action_reason": str(controller_task_action_reason),
            "cbf/active_constraint_identity_available": int(identity_available),
            "cbf/active_constraint_summary_json": json.dumps(
                summary, sort_keys=True, separators=(",", ":")
            ),
            "cbf/constraint_evidence_json": json.dumps(
                constraint_evidence, sort_keys=True, separators=(",", ":")
            ),
            "cbf/min_barrier_value_m": min(barrier_values, default=10.0),
            "cbf/max_active_prediction_buffer_m": max(
                prediction_buffers, default=0.0
            ),
            "cbf/max_prediction_buffer_m": 0.08,
            "cbf/fail_closed_on_invalid_active_hand": 1,
            "cbf/stop_on_infeasible": 1,
            "cbf/fail_closed_status": fail_closed_status,
            "cbf/active_hand": active_hands if cbf_active else "",
            "cbf/active_robot_link": active_links if cbf_active else "",
            "cbf/active_robot_collider": active_colliders if cbf_active else "",
            "cbf/identity_semantics": identity_semantics,
            "cbf/current_correction_rad_s": current_correction,
            "cbf/previous_correction_rad_s": previous_correction,
            "cbf/correction_delta_rad_s": correction_delta,
            "cbf/correction_memory_valid": 1,
            "response/phase": phase,
            "response/smooth_tail_active": int(smooth_tail_active),
            "response/time_since_response_onset_s": time_since_response,
            "response/time_since_recovery_onset_s": time_since_recovery,
            "recovery/active": int(recovery_active),
            "recovery/onset_now": int(bool(recovery_onset_now)),
            "recovery/end_now": int(bool(recovery_end_now)),
            "recovery/state": str(
                recovery_state.get("stage", recovery_state.get("mode", "inactive"))
            ),
            "recovery/control_authority": int(recovery_control_authority),
            "recovery/bc_resumed": int(bc_resumed),
            "recovery/stable_task_resumption": int(stable_task_resumption),
            "safety/min_ttc_s": min_ttc,
            "safety/ttc_valid": int(ttc_valid),
            "safety/dynamic_measurement_valid": int(
                bool(dynamic.get("dynamic_measurement_valid", False))
            ),
            "safety/max_closing_speed_m_s": closing_speed,
            "safety/closing": int(closing_speed > 0.0),
            "safety/left_proximal_surface_gap_m": _finite_or_default(
                proximal.get("left_gap_m"), 10.0
            ),
            "safety/right_proximal_surface_gap_m": _finite_or_default(
                proximal.get("right_gap_m"), 10.0
            ),
            "safety/left_proximal_collider_path": str(
                proximal.get("left_collider_path", "") or ""
            ),
            "safety/right_proximal_collider_path": str(
                proximal.get("right_collider_path", "") or ""
            ),
            "safety/static_collision": int(bool(info_after.get("static_collision", False))),
            "safety/self_collision": int(bool(info_after.get("self_collision", False))),
            "safety/static_contact_semantic_class": _static_contact_semantic_class(
                info_after
            ),
            "crossing/active": int(bool(crossing_state.get("active", False))),
            "crossing/state": str(
                crossing_state.get("state", "not_observed")
            ),
            "crossing/started_now": int(
                bool(crossing_state.get("started_now", False))
            ),
            "crossing/intended_completed_now": int(
                bool(crossing_state.get("intended_completed_now", False))
            ),
            "crossing/reverse_completed_now": int(
                bool(crossing_state.get("reverse_completed_now", False))
            ),
            "crossing/progress_fraction": _finite_or_default(
                crossing_state.get("progress_fraction"), 0.0
            ),
            "crossing/perpendicular_deviation_m": _finite_or_default(
                crossing_state.get("perpendicular_deviation_m"), 0.0
            ),
            "crossing/intended_hand": intended_hand,
            "crossing/non_crossing_hand": non_crossing_hand,
            "crossing/observed_hand": str(crossing_state.get("observed_hand", "")),
            "provenance/runtime_feature_cutoff_available": int(
                bool(crossing_state.get("runtime_feature_cutoff_available", False))
            ),
            "provenance/decision_context_available": int(
                bool(decision_context_available)
            ),
        }
    )
    return row


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _vec(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite ({size},) vector")
    return array.copy()


def _vec_or_zero(value: Any, size: int) -> np.ndarray:
    try:
        return _vec(value, size, "optional vector")
    except (TypeError, ValueError):
        return np.zeros(size, dtype=np.float64)


def _finite_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _elapsed_or_sentinel(now_s: float, onset_s: float | None) -> float:
    if onset_s is None:
        return -1.0
    onset = _finite_or_default(onset_s, -1.0)
    if onset < 0.0 or onset > now_s + 1e-9:
        return -1.0
    return max(0.0, now_s - onset)


def _static_contact_semantic_class(info: Mapping[str, Any]) -> str:
    if not bool(info.get("static_collision", False)):
        return "none"
    robot_path = str(info.get("static_closest_robot_collider", "") or "")
    environment_path = str(
        info.get("static_closest_environment_collider", "") or ""
    )
    robot_class = (
        "gripper_finger"
        if "panda_leftfinger" in robot_path or "panda_rightfinger" in robot_path
        else robot_path
    )
    environment_class = (
        "table"
        if environment_path == "/World/table"
        or environment_path.startswith("/World/table/")
        else environment_path
    )
    if not robot_class or not environment_class:
        return "identity_unavailable"
    return f"{robot_class}||{environment_class}"


def _constraint_evidence(cbf: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = cbf.get("constraint_evidence", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("CBF constraint_evidence must be a sequence")
    required = {
        "hand",
        "closest_link",
        "closest_collider_path",
        "raw_surface_gap_m",
        "prediction_buffer_m",
        "effective_safe_gap_m",
        "barrier_value_m",
        "buffered_current_gap_m",
        "closing_speed_mps",
        "lower_bound_mps",
        "nominal_residual_mps",
        "filtered_residual_mps",
    }
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValueError(f"CBF constraint_evidence[{index}] is incomplete")
        item = dict(value)
        for name in required - {"hand", "closest_link", "closest_collider_path"}:
            numeric = float(item[name])
            if not np.isfinite(numeric):
                raise ValueError(
                    f"CBF constraint_evidence[{index}].{name} is non-finite"
                )
            item[name] = numeric
        for name in ("hand", "closest_link", "closest_collider_path"):
            item[name] = str(item[name])
        if item["hand"] not in {"left", "right"}:
            raise ValueError(
                f"CBF constraint_evidence[{index}].hand must be left or right"
            )
        if not item["closest_link"] or not item["closest_collider_path"]:
            raise ValueError(
                f"CBF constraint_evidence[{index}] lacks link/collider identity"
            )
        prediction_buffer = item["prediction_buffer_m"]
        if not 0.0 <= prediction_buffer <= 0.08 + 1e-12:
            raise ValueError(
                f"CBF constraint_evidence[{index}] prediction buffer is out of range"
            )
        if not np.isclose(
            item["effective_safe_gap_m"],
            0.05 + prediction_buffer,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                f"CBF constraint_evidence[{index}] effective safe gap is inconsistent"
            )
        if item["filtered_residual_mps"] < -1e-7:
            raise ValueError(
                f"CBF constraint_evidence[{index}] filtered residual is unsafe"
            )
        result.append(item)
    hands = [item["hand"] for item in result]
    if len(set(hands)) != len(hands):
        raise ValueError("CBF constraint_evidence contains a duplicate hand")
    return result
