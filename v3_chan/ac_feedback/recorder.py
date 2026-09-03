"""Shared HDF5 dataset utilities for the A/C online recorder."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    dtype: Any
    tail_shape: tuple[int, ...] = ()


def _row_specs(joint_count: int, arm_count: int) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {
        "control/source_step": DatasetSpec(np.int64),
        "control/result_step": DatasetSpec(np.int64),
        "control/sim_time": DatasetSpec(np.float64),
        "control/observation_monotonic_ns": DatasetSpec(np.int64),
        "control/preflight_monotonic_ns": DatasetSpec(np.int64),
        "control/wall_time_unix_ns": DatasetSpec(np.int64),
        "control/policy_inference_started_monotonic_ns": DatasetSpec(np.int64),
        "control/policy_output_monotonic_ns": DatasetSpec(np.int64),
        "control/rmpflow_output_monotonic_ns": DatasetSpec(np.int64),
        "control/cbf_output_monotonic_ns": DatasetSpec(np.int64),
        "control/action_command_monotonic_ns": DatasetSpec(np.int64),
        "control/next_observation_monotonic_ns": DatasetSpec(np.int64),
        "control/post_step_monotonic_ns": DatasetSpec(np.int64),
        "observations/obs_raw": DatasetSpec(np.float32, (84,)),
        "observations/obs_policy": DatasetSpec(np.float32, (84,)),
        "observations/obs_next": DatasetSpec(np.float32, (84,)),
        "human/valid_mask": DatasetSpec(np.int8, (3,)),
        "human_next/valid_mask": DatasetSpec(np.int8, (3,)),
        "safety/left_surface_gap_m": DatasetSpec(np.float64),
        "safety/right_surface_gap_m": DatasetSpec(np.float64),
        "safety/min_surface_gap_m": DatasetSpec(np.float64),
        "safety/geometry_valid": DatasetSpec(np.int8),
        "safety/collision": DatasetSpec(np.int8),
        "safety/near": DatasetSpec(np.int8),
        "safety/near_miss": DatasetSpec(np.int8),
        "actions/bc_task_action": DatasetSpec(np.float32, (5,)),
        "actions/applied_arm_joint_velocities_rad_s": DatasetSpec(
            np.float64, (arm_count,)
        ),
        "actions/applied_arm_joint_velocities_mask": DatasetSpec(
            np.int8, (arm_count,)
        ),
        "actions/arm_action_submitted": DatasetSpec(np.int8),
        "actions/action_pipeline_complete": DatasetSpec(np.int8),
        "actions/gripper_command": DatasetSpec("str"),
        "robot/measured_joint_positions_before_rad": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_velocities_before_rad_s": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_positions_after_rad": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "robot/measured_joint_velocities_after_rad_s": DatasetSpec(
            np.float64, (joint_count,)
        ),
        "task/controller_event_before": DatasetSpec(np.int32),
        "task/controller_event_after": DatasetSpec(np.int32),
        "task/controller_t_before": DatasetSpec(np.float64),
        "task/controller_t_after": DatasetSpec(np.float64),
        "task/success": DatasetSpec(np.int8),
        "task/terminated": DatasetSpec(np.int8),
        "task/truncated": DatasetSpec(np.int8),
        "task/terminal_reason": DatasetSpec("str"),
        "encounter/row_encounter_id": DatasetSpec("str"),
        "encounter/state": DatasetSpec("str"),
    }
    for namespace in ("human", "human_next"):
        for entity in ("head", "left", "right"):
            specs.update(
                {
                    f"{namespace}/{entity}_position_world_m": DatasetSpec(
                        np.float64, (3,)
                    ),
                    f"{namespace}/{entity}_pose_valid": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_position_tracked": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_tracking_status_known": DatasetSpec(np.int8),
                    f"{namespace}/{entity}_source_name": DatasetSpec("str"),
                    f"{namespace}/{entity}_source_path": DatasetSpec("str"),
                    f"{namespace}/{entity}_acquisition_monotonic_ns": DatasetSpec(
                        np.int64
                    ),
                    f"{namespace}/{entity}_pose_age_ms": DatasetSpec(np.float64),
                    f"{namespace}/{entity}_source_switched": DatasetSpec(np.int8),
                }
            )
    for stage in (
        "nominal_rmpflow",
        "cbf_filtered",
        "submitted",
        "pre_applied",
        "applied",
    ):
        for channel, unit in (
            ("joint_positions", "rad"),
            ("joint_velocities", "rad_s"),
            ("joint_efforts", "nm"),
        ):
            base = f"actions/{stage}_{channel}_{unit}"
            specs[base] = DatasetSpec(np.float64, (joint_count,))
            specs[base + "_mask"] = DatasetSpec(np.int8, (joint_count,))
    cbf_float = (
        "intervention_norm_radps",
        "nominal_velocity_norm_radps",
        "filtered_velocity_norm_radps",
        "max_constraint_violation_before_mps",
        "max_constraint_violation_after_mps",
        "slack_mps",
        "buffered_current_gap_m",
        "solve_time_ms",
    )
    cbf_int = (
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
    )
    cbf_text = (
        "failure_reasons_json",
        "status",
        "solver_backend",
        "projection_status",
    )
    specs.update({f"cbf/{name}": DatasetSpec(np.float64) for name in cbf_float})
    specs.update({f"cbf/{name}": DatasetSpec(np.int32) for name in cbf_int})
    specs.update({f"cbf/{name}": DatasetSpec("str") for name in cbf_text})
    return specs


def _create_extendable_dataset(group, path: str, spec: DatasetSpec, *, h5py):
    parent = group
    parts = path.split("/")
    for part in parts[:-1]:
        parent = parent.require_group(part)
    dtype = h5py.string_dtype(encoding="utf-8") if spec.dtype == "str" else spec.dtype
    return parent.create_dataset(
        parts[-1],
        shape=(0, *spec.tail_shape),
        maxshape=(None, *spec.tail_shape),
        chunks=(max(1, min(256, 1)), *spec.tail_shape),
        dtype=dtype,
    )


def _coerce_value(value: Any, spec: DatasetSpec, *, path: str) -> Any:
    if spec.dtype == "str":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)
    array = np.asarray(value)
    if array.shape != spec.tail_shape:
        if spec.tail_shape == () and array.size == 1:
            array = array.reshape(())
        else:
            raise RuntimeError(
                f"{path} has shape {array.shape}, expected {spec.tail_shape}"
            )
    if np.issubdtype(np.dtype(spec.dtype), np.integer):
        if np.issubdtype(np.dtype(spec.dtype), np.signedinteger) and path.endswith(
            "position_tracked"
        ):
            allowed = (-1, 0, 1)
        elif _is_binary_row_path(path):
            allowed = (0, 1)
        else:
            allowed = None
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
            raise RuntimeError(f"{path} must contain finite integer values")
        if allowed is not None and not np.all(np.isin(numeric, allowed)):
            raise RuntimeError(f"{path} contains a value outside {allowed}")
    elif not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
        raise RuntimeError(f"{path} contains non-finite values")
    return np.asarray(array, dtype=spec.dtype)


def _is_binary_row_path(path: str) -> bool:
    return bool(
        path.endswith("_mask")
        or path.endswith("_pose_valid")
        or path.endswith("_tracking_status_known")
        or path.endswith("_source_switched")
        or path
        in {
            "safety/geometry_valid",
            "safety/collision",
            "safety/near",
            "safety/near_miss",
            "actions/arm_action_submitted",
            "actions/action_pipeline_complete",
            "task/success",
            "task/terminated",
            "task/truncated",
            "cbf/active",
            "cbf/intervention_available",
            "cbf/feasible",
            "cbf/fallback_applied",
            "cbf/solver_converged",
            "cbf/infeasibility_proven",
            "cbf/relaxed_solution_available",
            "cbf/relaxed_solution_applied",
            "cbf/intentional_human_absence",
        }
    )


def _append_table_row(group, values: Mapping[str, Any]) -> None:
    expected = set(group.keys())
    if set(values) != expected:
        raise RuntimeError(
            f"event row columns mismatch: missing={sorted(expected - set(values))}, "
            f"extra={sorted(set(values) - expected)}"
        )
    lengths = {dataset.shape[0] for dataset in group.values()}
    if len(lengths) != 1:
        raise RuntimeError("event table columns are not aligned")
    index = lengths.pop()
    converted: dict[str, Any] = {}
    for name, value in values.items():
        dataset = group[name]
        tail = dataset.shape[1:]
        if dataset.dtype.kind in "SUO":
            converted[name] = str(value)
        else:
            array = np.asarray(value)
            if array.shape != tail:
                if tail == () and array.size == 1:
                    array = array.reshape(())
                else:
                    raise RuntimeError(
                        f"event column {name} has shape {array.shape}, expected {tail}"
                    )
            if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
                raise RuntimeError(f"event column {name} is non-finite")
            converted[name] = array
    for name, value in converted.items():
        dataset = group[name]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value


def _write_static_value(group, name: str, value: Any, *, h5py) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        return
    array = np.asarray(value)
    if array.dtype.kind in "SUO":
        group.create_dataset(
            name,
            data=np.asarray(array, dtype=h5py.string_dtype("utf-8")),
        )
        return
    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError(f"initial_scene/{name} contains non-finite values")
    group.create_dataset(name, data=array)


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool, np.generic)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metadata attributes must be finite")
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
