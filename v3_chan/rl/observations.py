from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    from v3_chan.end_effector_safety_geometry import (
        MISSING_SURFACE_GAP_M,
        SafetyThresholds,
        classify_surface_gap,
    )
except ImportError:
    from end_effector_safety_geometry import (
        MISSING_SURFACE_GAP_M,
        SafetyThresholds,
        classify_surface_gap,
    )


OBSERVATION_VERSION = "obs_v3_builtin_panda_collision_surface"
TASK_PHASES = ("approach_cube", "grasp_cube", "move_to_target", "release_cube")
CONTROLLER_EVENT_COUNT = 10
MISSING_DISTANCE_M = MISSING_SURFACE_GAP_M
_SAFETY_THRESHOLDS = SafetyThresholds.from_env()
DEFAULT_NEAR_HUMAN_THRESHOLD_M = _SAFETY_THRESHOLDS.near_gap_m
DEFAULT_NEAR_MISS_THRESHOLD_M = _SAFETY_THRESHOLDS.near_miss_gap_m
DEFAULT_COLLISION_SURFACE_GAP_M = _SAFETY_THRESHOLDS.collision_gap_m
DEFAULT_HAND_PROXY_RADIUS_M = _SAFETY_THRESHOLDS.hand_radius_m
DEFAULT_GRIPPER_PROXY_RADIUS_M = float(
    os.environ.get("HRI_GRIPPER_PROXY_RADIUS_M", "0.025")
)
DEFAULT_DISTANCE_GATE_FULL_GAP_M = _SAFETY_THRESHOLDS.gate_full_gap_m
DEFAULT_DISTANCE_GATE_START_GAP_M = _SAFETY_THRESHOLDS.gate_start_gap_m


@dataclass(frozen=True)
class ObservationField:
    name: str
    shape: tuple[int, ...]
    description: str

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))


OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField("robot_joint_pos", (7,), "Panda arm joint positions."),
    ObservationField("robot_joint_vel", (7,), "Panda arm joint velocities."),
    ObservationField("gripper_width", (1,), "Sum of the two Franka finger joint positions."),
    ObservationField("ee_pos", (3,), "End-effector world position."),
    ObservationField("ee_quat", (4,), "End-effector world orientation quaternion."),
    ObservationField("cube_pos", (3,), "Current pick cube world position."),
    ObservationField("cube_quat", (4,), "Current pick cube world orientation quaternion."),
    ObservationField("cube_lin_vel", (3,), "Current pick cube linear velocity."),
    ObservationField("cube_ang_vel", (3,), "Current pick cube angular velocity."),
    ObservationField("place_target_pos", (3,), "Desired placement target world position."),
    ObservationField("ee_to_cube", (3,), "Vector from end-effector to current pick cube."),
    ObservationField("cube_to_place_target", (3,), "Vector from current pick cube to placement target."),
    ObservationField("ee_to_place_target", (3,), "Vector from end-effector to placement target."),
    ObservationField("human_head_pos", (3,), "Human head or HMD world position."),
    ObservationField("human_left_hand_pos", (3,), "Human left hand world position."),
    ObservationField("human_right_hand_pos", (3,), "Human right hand world position."),
    ObservationField("ee_to_left_hand", (3,), "Vector from end-effector to left hand."),
    ObservationField("ee_to_right_hand", (3,), "Vector from end-effector to right hand."),
    ObservationField(
        "min_hand_gripper_dist",
        (1,),
        "Canonical signed hand-to-gripper surface gap in meters; overlap is negative.",
    ),
    ObservationField("human_robot_collision", (1,), "Binary flag for human/robot collision."),
    ObservationField("near_human", (1,), "Binary flag for unsafe proximity to human hand."),
    ObservationField("collision_green", (1,), "Binary flag for collision with protected green cube."),
    ObservationField("pick_miss_recent", (1,), "Binary flag for recent pick miss event."),
    ObservationField("drop_throw_recent", (1,), "Binary flag for recent drop or throw event."),
    ObservationField("has_grasped_cube", (1,), "Binary flag for current grasp estimate."),
    ObservationField("task_phase", (4,), "One-hot task phase."),
    ObservationField("controller_event", (CONTROLLER_EVENT_COUNT,), "One-hot PickPlaceController event."),
    ObservationField("controller_t", (1,), "PickPlaceController event progress in [0, 1]."),
)

# These fields are recorded for HRI/safety learning but intentionally excluded
# from obs_policy so existing 84-D robot-only checkpoints remain compatible.
AUXILIARY_OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField(
        "min_hand_gripper_center_dist",
        (1,),
        "Minimum hand-center to gripper-center distance in meters.",
    ),
    ObservationField(
        "min_hand_gripper_surface_gap",
        (1,),
        "Signed hand-to-gripper surface gap; overlap is negative.",
    ),
    ObservationField(
        "near_miss",
        (1,),
        "Binary flag for positive surface gap inside the near-miss band.",
    ),
    ObservationField(
        "left_hand_end_effector_surface_gap",
        (1,),
        "Signed left-hand sphere gap to the nearest built-in distal Panda collider.",
    ),
    ObservationField(
        "right_hand_end_effector_surface_gap",
        (1,),
        "Signed right-hand sphere gap to the nearest built-in distal Panda collider.",
    ),
    ObservationField(
        "min_hand_end_effector_surface_gap",
        (1,),
        "Minimum signed hand gap to a built-in distal Panda collider.",
    ),
    ObservationField("left_hand_contact", (1,), "Left hand/Panda distal contact flag."),
    ObservationField("right_hand_contact", (1,), "Right hand/Panda distal contact flag."),
    ObservationField(
        "distance_gate",
        (1,),
        "Safety residual gate: 0 above 0.13 m and 1 at or below 0.05 m.",
    ),
    ObservationField(
        "geometry_valid",
        (1,),
        "At least one tracked hand was evaluated against valid Panda colliders.",
    ),
)

# Runtime-only dynamic features used by the safety residual. They are kept out
# of RECORDED_OBSERVATION_FIELDS so the established 83-D collection schema and
# existing HDF5 files remain unchanged.
DYNAMIC_HRI_OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField(
        "left_hand_vel_filtered_mps",
        (3,),
        "Filtered left-hand world velocity in meters per second.",
    ),
    ObservationField(
        "right_hand_vel_filtered_mps",
        (3,),
        "Filtered right-hand world velocity in meters per second.",
    ),
    ObservationField(
        "left_closest_robot_velocity_world_mps",
        (3,),
        "Velocity of the closest left-hand-facing robot surface point.",
    ),
    ObservationField(
        "right_closest_robot_velocity_world_mps",
        (3,),
        "Velocity of the closest right-hand-facing robot surface point.",
    ),
    ObservationField(
        "left_relative_velocity_world_mps",
        (3,),
        "Left hand velocity relative to its closest robot surface point.",
    ),
    ObservationField(
        "right_relative_velocity_world_mps",
        (3,),
        "Right hand velocity relative to its closest robot surface point.",
    ),
    ObservationField(
        "left_closing_speed_mps",
        (1,),
        "Non-negative left-hand surface-gap closing speed.",
    ),
    ObservationField(
        "right_closing_speed_mps",
        (1,),
        "Non-negative right-hand surface-gap closing speed.",
    ),
    ObservationField("left_ttc_s", (1,), "Left-hand time to contact in seconds."),
    ObservationField("right_ttc_s", (1,), "Right-hand time to contact in seconds."),
    ObservationField(
        "left_dynamic_measurement_valid",
        (1,),
        "Left dynamic velocity measurement validity flag.",
    ),
    ObservationField(
        "right_dynamic_measurement_valid",
        (1,),
        "Right dynamic velocity measurement validity flag.",
    ),
    ObservationField("left_ttc_valid", (1,), "Left TTC validity flag."),
    ObservationField("right_ttc_valid", (1,), "Right TTC validity flag."),
)
RECORDED_OBSERVATION_FIELDS = OBSERVATION_FIELDS + AUXILIARY_OBSERVATION_FIELDS

OBSERVATION_DIM = sum(field.dim for field in OBSERVATION_FIELDS)
_FIELD_MAP = {field.name: field for field in RECORDED_OBSERVATION_FIELDS}

# Safety residual input. This remains separate from the frozen 84-D task
# observation and uses the canonical v4 PhysX safety fields.
HRI_OBS_FIELD_NAMES = (
    "robot_joint_pos",
    "robot_joint_vel",
    "gripper_width",
    "ee_pos",
    "ee_quat",
    "cube_pos",
    "cube_quat",
    "place_target_pos",
    "ee_to_cube",
    "cube_to_place_target",
    "ee_to_place_target",
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "ee_to_left_hand",
    "ee_to_right_hand",
    "min_hand_gripper_dist",
    "min_hand_gripper_surface_gap",
    "left_hand_end_effector_surface_gap",
    "right_hand_end_effector_surface_gap",
    "min_hand_end_effector_surface_gap",
    "human_robot_collision",
    "near_human",
    "near_miss",
    "left_hand_contact",
    "right_hand_contact",
    "distance_gate",
    "geometry_valid",
    "has_grasped_cube",
    "task_phase",
    "controller_event",
)
HRI_OBS_DIM = int(sum(_FIELD_MAP[name].dim for name in HRI_OBS_FIELD_NAMES))
HRI_OBSERVATION_VERSION = "hri_policy_obs_v1_83d_surface_gap"

DYNAMIC_HRI_OBS_FIELD_NAMES = HRI_OBS_FIELD_NAMES + tuple(
    field.name for field in DYNAMIC_HRI_OBSERVATION_FIELDS
)
DYNAMIC_HRI_OBS_DIM = int(
    HRI_OBS_DIM + sum(field.dim for field in DYNAMIC_HRI_OBSERVATION_FIELDS)
)
DYNAMIC_HRI_OBSERVATION_VERSION = (
    "hri_policy_obs_v2_109d_surface_gap_dynamics"
)

# The stateful finite-difference action limiter makes the executed backup
# command depend on two previous decisions.  Keep that controller state in a
# Backup-only observation contract instead of changing the shared 109-D HRI
# observation used by the older residual policies.  The Backup policy keeps
# robot-surface and relative velocity, but omits the algebraically redundant
# hand velocity because v_hand = v_relative + v_robot_surface.
BACKUP_LIMITER_ACTION_DIM = 4
BACKUP_LIMITER_HISTORY_DIM = BACKUP_LIMITER_ACTION_DIM * 2
BACKUP_OMITTED_DYNAMIC_FIELD_NAMES = (
    "left_hand_vel_filtered_mps",
    "right_hand_vel_filtered_mps",
)
BACKUP_DYNAMIC_HRI_OBS_FIELD_NAMES = tuple(
    name
    for name in DYNAMIC_HRI_OBS_FIELD_NAMES
    if name not in BACKUP_OMITTED_DYNAMIC_FIELD_NAMES
)
BACKUP_DYNAMIC_HRI_OBS_DIM = int(
    sum(_FIELD_MAP[name].dim for name in HRI_OBS_FIELD_NAMES)
    + sum(
        field.dim
        for field in DYNAMIC_HRI_OBSERVATION_FIELDS
        if field.name not in BACKUP_OMITTED_DYNAMIC_FIELD_NAMES
    )
)

# Preserve the previous 117-D contract so existing checkpoints and generated
# risk datasets remain reproducible.
LEGACY_BACKUP_OBS_DIM = DYNAMIC_HRI_OBS_DIM + BACKUP_LIMITER_HISTORY_DIM
LEGACY_BACKUP_OBSERVATION_VERSION = (
    "backup_policy_obs_v3_117d_surface_gap_dynamics_limiter_history"
)
BACKUP_OBS_DIM = BACKUP_DYNAMIC_HRI_OBS_DIM + BACKUP_LIMITER_HISTORY_DIM
BACKUP_OBSERVATION_VERSION = (
    "backup_policy_obs_v4_111d_surface_relative_dynamics_limiter_history"
)
_DYNAMIC_FIELD_MAP = {
    field.name: field for field in DYNAMIC_HRI_OBSERVATION_FIELDS
}
_POLICY_FIELD_MAP = {**_FIELD_MAP, **_DYNAMIC_FIELD_MAP}


def observation_slices() -> dict[str, slice]:
    slices = {}
    cursor = 0
    for field in OBSERVATION_FIELDS:
        slices[field.name] = slice(cursor, cursor + field.dim)
        cursor += field.dim
    return slices


def empty_observation(dtype=np.float32) -> dict[str, np.ndarray]:
    obs = {}
    for field in RECORDED_OBSERVATION_FIELDS:
        obs[field.name] = np.zeros(field.shape, dtype=dtype)
    obs["min_hand_gripper_dist"][:] = MISSING_DISTANCE_M
    obs["min_hand_gripper_center_dist"][:] = MISSING_DISTANCE_M
    obs["min_hand_gripper_surface_gap"][:] = MISSING_DISTANCE_M
    obs["left_hand_end_effector_surface_gap"][:] = MISSING_DISTANCE_M
    obs["right_hand_end_effector_surface_gap"][:] = MISSING_DISTANCE_M
    obs["min_hand_end_effector_surface_gap"][:] = MISSING_DISTANCE_M
    for field in DYNAMIC_HRI_OBSERVATION_FIELDS:
        obs[field.name] = np.zeros(field.shape, dtype=dtype)
    obs["left_ttc_s"][:] = 10.0
    obs["right_ttc_s"][:] = 10.0
    return obs


def flatten_observation(obs: Mapping[str, np.ndarray], dtype=np.float32) -> np.ndarray:
    validate_observation(obs)
    return np.concatenate(
        [np.asarray(obs[field.name], dtype=dtype).reshape(-1) for field in OBSERVATION_FIELDS]
    ).astype(dtype, copy=False)


def flatten_hri_observation(
    obs: Mapping[str, np.ndarray],
    dtype=np.float32,
    *,
    field_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    validate_observation(obs)
    validate_auxiliary_observation(obs)
    names = HRI_OBS_FIELD_NAMES if field_names is None else tuple(field_names)
    _validate_policy_fields(obs, names)
    return np.concatenate(
        [np.asarray(obs[name], dtype=dtype).reshape(-1) for name in names]
    ).astype(dtype, copy=False)


def flatten_dynamic_hri_observation(
    obs: Mapping[str, np.ndarray], dtype=np.float32
) -> np.ndarray:
    return flatten_hri_observation(
        obs,
        dtype=dtype,
        field_names=DYNAMIC_HRI_OBS_FIELD_NAMES,
    )


def flatten_backup_dynamic_observation(
    obs: Mapping[str, np.ndarray], dtype=np.float32
) -> np.ndarray:
    """Build the compact Backup-only dynamic state without hand velocity."""

    result = flatten_hri_observation(
        obs,
        dtype=dtype,
        field_names=BACKUP_DYNAMIC_HRI_OBS_FIELD_NAMES,
    )
    if result.shape != (BACKUP_DYNAMIC_HRI_OBS_DIM,):
        raise RuntimeError(
            "Compact Backup dynamic observation shape mismatch: "
            f"{result.shape} != {(BACKUP_DYNAMIC_HRI_OBS_DIM,)}"
        )
    return result


def flatten_backup_observation(
    obs: Mapping[str, np.ndarray],
    *,
    previous_executed_action: np.ndarray,
    previous_action_delta: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """Build the Markov Backup observation used with the 4-D limiter.

    The tail is ``[a_{t-1}, a_{t-1} - a_{t-2}]`` in normalized Backup-action
    coordinates.  It is deliberately not added to the recorded HDF5 schema.
    """

    previous = np.asarray(previous_executed_action, dtype=dtype).reshape(-1)
    delta = np.asarray(previous_action_delta, dtype=dtype).reshape(-1)
    expected = int(BACKUP_LIMITER_ACTION_DIM)
    if previous.size != expected or delta.size != expected:
        raise ValueError(
            "Backup limiter history requires exactly "
            f"{expected} values per vector; got {previous.size} and {delta.size}"
        )
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(delta)):
        raise ValueError("Backup limiter history must contain only finite values")
    result = np.concatenate(
        (flatten_backup_dynamic_observation(obs, dtype=dtype), previous, delta),
        axis=0,
    ).astype(dtype, copy=False)
    if result.shape != (BACKUP_OBS_DIM,):
        raise RuntimeError(
            f"Backup observation shape mismatch: {result.shape} != {(BACKUP_OBS_DIM,)}"
        )
    return result


def flatten_legacy_backup_observation(
    obs: Mapping[str, np.ndarray],
    *,
    previous_executed_action: np.ndarray,
    previous_action_delta: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """Build the former 117-D observation for checkpoint compatibility."""

    previous = np.asarray(previous_executed_action, dtype=dtype).reshape(-1)
    delta = np.asarray(previous_action_delta, dtype=dtype).reshape(-1)
    expected = int(BACKUP_LIMITER_ACTION_DIM)
    if previous.size != expected or delta.size != expected:
        raise ValueError(
            "Backup limiter history requires exactly "
            f"{expected} values per vector; got {previous.size} and {delta.size}"
        )
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(delta)):
        raise ValueError("Backup limiter history must contain only finite values")
    result = np.concatenate(
        (flatten_dynamic_hri_observation(obs, dtype=dtype), previous, delta),
        axis=0,
    ).astype(dtype, copy=False)
    if result.shape != (LEGACY_BACKUP_OBS_DIM,):
        raise RuntimeError(
            "Legacy Backup observation shape mismatch: "
            f"{result.shape} != {(LEGACY_BACKUP_OBS_DIM,)}"
        )
    return result


def flatten_backup_policy_observation(
    obs: Mapping[str, np.ndarray],
    *,
    observation_dim: int,
    previous_executed_action: np.ndarray | None = None,
    previous_action_delta: np.ndarray | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Flatten a supported shared or Backup-only observation contract."""

    requested_dim = int(observation_dim)
    if requested_dim == DYNAMIC_HRI_OBS_DIM:
        return flatten_dynamic_hri_observation(obs, dtype=dtype)
    supported_limiter_dims = (BACKUP_OBS_DIM, LEGACY_BACKUP_OBS_DIM)
    if requested_dim not in supported_limiter_dims:
        raise ValueError(
            "Unsupported Backup observation dimension: "
            f"{requested_dim}; expected {DYNAMIC_HRI_OBS_DIM}, "
            f"{BACKUP_OBS_DIM}, or {LEGACY_BACKUP_OBS_DIM}"
        )
    if previous_executed_action is None or previous_action_delta is None:
        raise ValueError(
            "A limiter-aware Backup observation requires finite limiter history"
        )
    if requested_dim == LEGACY_BACKUP_OBS_DIM:
        return flatten_legacy_backup_observation(
            obs,
            previous_executed_action=previous_executed_action,
            previous_action_delta=previous_action_delta,
            dtype=dtype,
        )
    return flatten_backup_observation(
        obs,
        previous_executed_action=previous_executed_action,
        previous_action_delta=previous_action_delta,
        dtype=dtype,
    )


def restore_dynamic_hri_observations_from_backup(
    observations: np.ndarray,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Restore the shared 109-D HRI state from a Backup-policy batch.

    The compact 111-D Backup contract stores 103 physical/HRI features plus
    eight limiter-history values. It omits filtered hand velocity because it
    is exactly recoverable as ``relative velocity + robot-surface velocity``.
    Legacy 117-D rows already contain the full 109-D state before their
    limiter-history tail.
    """

    values = np.asarray(observations, dtype=dtype)
    vector_input = values.ndim == 1
    if vector_input:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Backup observations must be a vector or matrix")
    if values.shape[1] == DYNAMIC_HRI_OBS_DIM:
        restored = values.copy()
    elif values.shape[1] == LEGACY_BACKUP_OBS_DIM:
        restored = values[:, :DYNAMIC_HRI_OBS_DIM].copy()
    elif values.shape[1] == BACKUP_OBS_DIM:
        compact = values[:, :BACKUP_DYNAMIC_HRI_OBS_DIM]
        source_slices = _policy_observation_slices(
            BACKUP_DYNAMIC_HRI_OBS_FIELD_NAMES
        )
        target_slices = _policy_observation_slices(
            DYNAMIC_HRI_OBS_FIELD_NAMES
        )
        restored = np.empty(
            (values.shape[0], DYNAMIC_HRI_OBS_DIM),
            dtype=dtype,
        )
        for field_name in DYNAMIC_HRI_OBS_FIELD_NAMES:
            target = target_slices[field_name]
            source = source_slices.get(field_name)
            if source is not None:
                restored[:, target] = compact[:, source]
                continue
            side = field_name.removesuffix("_hand_vel_filtered_mps")
            if side not in ("left", "right"):
                raise RuntimeError(
                    f"Cannot restore omitted Backup field: {field_name}"
                )
            robot = source_slices[
                f"{side}_closest_robot_velocity_world_mps"
            ]
            relative = source_slices[f"{side}_relative_velocity_world_mps"]
            restored[:, target] = compact[:, robot] + compact[:, relative]
    else:
        raise ValueError(
            "Unsupported Backup normalization observation width: "
            f"{values.shape[1]}; expected {DYNAMIC_HRI_OBS_DIM}, "
            f"{BACKUP_OBS_DIM}, or {LEGACY_BACKUP_OBS_DIM}"
        )
    if not np.all(np.isfinite(restored)):
        raise ValueError("Restored dynamic HRI observations must be finite")
    return restored.reshape(-1) if vector_input else restored


def _policy_observation_slices(
    field_names: tuple[str, ...] | list[str],
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    cursor = 0
    for field_name in field_names:
        field = _POLICY_FIELD_MAP[field_name]
        result[field_name] = slice(cursor, cursor + field.dim)
        cursor += field.dim
    return result


def apply_dynamic_hri_observation(
    obs: dict[str, np.ndarray],
    dynamic_payload: Mapping[str, object] | None,
) -> dict[str, np.ndarray]:
    payload = dynamic_payload or {}
    for field in DYNAMIC_HRI_OBSERVATION_FIELDS:
        if field.name not in payload:
            continue
        value = np.asarray(payload[field.name], dtype=np.float32)
        if value.size != field.dim:
            raise ValueError(
                f"Dynamic observation field '{field.name}' has {value.size} values, "
                f"expected {field.dim}"
            )
        obs[field.name] = value.reshape(field.shape)
    return obs


def validate_observation(obs: Mapping[str, np.ndarray]) -> None:
    missing = [field.name for field in OBSERVATION_FIELDS if field.name not in obs]
    if missing:
        raise ValueError(f"Observation is missing fields: {missing}")
    for field in OBSERVATION_FIELDS:
        value = np.asarray(obs[field.name])
        if value.shape != field.shape:
            raise ValueError(
                f"Observation field '{field.name}' has shape {value.shape}, "
                f"expected {field.shape}"
            )


def validate_auxiliary_observation(obs: Mapping[str, np.ndarray]) -> None:
    missing = [field.name for field in AUXILIARY_OBSERVATION_FIELDS if field.name not in obs]
    if missing:
        raise ValueError(f"Observation is missing auxiliary fields: {missing}")
    for field in AUXILIARY_OBSERVATION_FIELDS:
        value = np.asarray(obs[field.name])
        if value.shape != field.shape:
            raise ValueError(
                f"Auxiliary observation field '{field.name}' has shape {value.shape}, "
                f"expected {field.shape}"
            )


def _validate_policy_fields(
    obs: Mapping[str, np.ndarray], field_names: tuple[str, ...]
) -> None:
    unknown = [name for name in field_names if name not in _POLICY_FIELD_MAP]
    if unknown:
        raise ValueError(f"Unknown safety observation fields: {unknown}")
    missing = [name for name in field_names if name not in obs]
    if missing:
        raise ValueError(f"Observation is missing safety fields: {missing}")
    for name in field_names:
        value = np.asarray(obs[name])
        expected = _POLICY_FIELD_MAP[name].shape
        if value.shape != expected:
            raise ValueError(
                f"Safety observation field '{name}' has shape {value.shape}, "
                f"expected {expected}"
            )


def build_observation(
    *,
    robot,
    cube,
    place_target,
    human_head_pos: np.ndarray | None = None,
    human_left_hand_pos: np.ndarray | None = None,
    human_right_hand_pos: np.ndarray | None = None,
    gripper_center_pos: np.ndarray | None = None,
    human_robot_collision: bool | None = None,
    near_human: bool | None = None,
    near_miss: bool | None = None,
    collision_green: bool = False,
    pick_miss_recent: bool = False,
    drop_throw_recent: bool = False,
    has_grasped_cube: bool = False,
    task_phase: str | int = "approach_cube",
    controller_event: int | None = None,
    controller_t: float = 0.0,
    near_human_threshold_m: float = DEFAULT_NEAR_HUMAN_THRESHOLD_M,
    min_hand_gripper_dist_override: float | None = None,
    min_hand_gripper_surface_gap_override: float | None = None,
    left_hand_surface_gap_override: float | None = None,
    right_hand_surface_gap_override: float | None = None,
    left_hand_contact: bool | None = None,
    right_hand_contact: bool | None = None,
    distance_gate_override: float | None = None,
    geometry_valid_override: bool | None = None,
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
    collision_surface_gap_m: float = DEFAULT_COLLISION_SURFACE_GAP_M,
    hand_proxy_radius_m: float = DEFAULT_HAND_PROXY_RADIUS_M,
    gripper_proxy_radius_m: float = DEFAULT_GRIPPER_PROXY_RADIUS_M,
) -> dict[str, np.ndarray]:
    """Build the state observation from Isaac runtime objects.

    The function is intentionally free of Isaac imports so it can be imported by
    trajectory tooling and unit tests outside SimulationApp. Runtime objects only
    need to expose the Isaac-style methods used below.
    """

    obs = empty_observation()
    joint_pos = _safe_array_call(robot, "get_joint_positions", 9)
    joint_vel = _safe_array_call(robot, "get_joint_velocities", 9)
    obs["robot_joint_pos"] = _fixed_array(joint_pos[:7], 7)
    obs["robot_joint_vel"] = _fixed_array(joint_vel[:7], 7)

    gripper_joints = _safe_gripper_joint_positions(robot)
    obs["gripper_width"] = np.array([float(np.sum(gripper_joints))], dtype=np.float32)

    ee_pos, ee_quat = _safe_world_pose(getattr(robot, "end_effector", None))
    cube_pos, cube_quat = _safe_world_pose(cube)
    place_target_pos, _ = _safe_world_pose(place_target)

    obs["ee_pos"] = ee_pos.astype(np.float32)
    obs["ee_quat"] = ee_quat.astype(np.float32)
    obs["cube_pos"] = cube_pos.astype(np.float32)
    obs["cube_quat"] = cube_quat.astype(np.float32)
    obs["cube_lin_vel"] = _safe_array_call(cube, "get_linear_velocity", 3).astype(np.float32)
    obs["cube_ang_vel"] = _safe_array_call(cube, "get_angular_velocity", 3).astype(np.float32)
    obs["place_target_pos"] = place_target_pos.astype(np.float32)

    obs["ee_to_cube"] = (cube_pos - ee_pos).astype(np.float32)
    obs["cube_to_place_target"] = (place_target_pos - cube_pos).astype(np.float32)
    obs["ee_to_place_target"] = (place_target_pos - ee_pos).astype(np.float32)

    head_pos = _optional_vec3(human_head_pos)
    left_pos = _optional_vec3(human_left_hand_pos)
    right_pos = _optional_vec3(human_right_hand_pos)
    obs["human_head_pos"] = head_pos.astype(np.float32)
    obs["human_left_hand_pos"] = left_pos.astype(np.float32)
    obs["human_right_hand_pos"] = right_pos.astype(np.float32)

    obs["ee_to_left_hand"] = (
        left_pos - ee_pos if human_left_hand_pos is not None else np.zeros(3)
    ).astype(np.float32)
    obs["ee_to_right_hand"] = (
        right_pos - ee_pos if human_right_hand_pos is not None else np.zeros(3)
    ).astype(np.float32)

    gripper_pos = _optional_vec3(gripper_center_pos) if gripper_center_pos is not None else ee_pos
    hand_distances = []
    if human_left_hand_pos is not None:
        hand_distances.append(float(np.linalg.norm(left_pos - gripper_pos)))
    if human_right_hand_pos is not None:
        hand_distances.append(float(np.linalg.norm(right_pos - gripper_pos)))
    center_dist = (
        float(min_hand_gripper_dist_override)
        if min_hand_gripper_dist_override is not None
        else min(hand_distances) if hand_distances else MISSING_DISTANCE_M
    )
    left_surface_gap = (
        float(left_hand_surface_gap_override)
        if left_hand_surface_gap_override is not None
        else MISSING_DISTANCE_M
    )
    right_surface_gap = (
        float(right_hand_surface_gap_override)
        if right_hand_surface_gap_override is not None
        else MISSING_DISTANCE_M
    )
    per_hand_surface_gaps = [
        gap
        for gap in (left_surface_gap, right_surface_gap)
        if np.isfinite(gap) and gap < MISSING_DISTANCE_M
    ]
    if per_hand_surface_gaps:
        surface_gap = min(per_hand_surface_gaps)
    elif min_hand_gripper_surface_gap_override is not None:
        surface_gap = float(min_hand_gripper_surface_gap_override)
    elif center_dist >= MISSING_DISTANCE_M:
        surface_gap = MISSING_DISTANCE_M
    else:
        surface_gap = center_dist - max(0.0, float(hand_proxy_radius_m)) - max(
            0.0, float(gripper_proxy_radius_m)
        )

    # min_hand_gripper_dist remains in the fixed 84-D policy schema, but from
    # observation v2 onward its canonical meaning is signed surface gap.
    obs["min_hand_gripper_dist"] = np.array([surface_gap], dtype=np.float32)
    obs["min_hand_gripper_center_dist"] = np.array([center_dist], dtype=np.float32)
    obs["min_hand_gripper_surface_gap"] = np.array([surface_gap], dtype=np.float32)
    obs["left_hand_end_effector_surface_gap"] = np.array(
        [left_surface_gap], dtype=np.float32
    )
    obs["right_hand_end_effector_surface_gap"] = np.array(
        [right_surface_gap], dtype=np.float32
    )
    obs["min_hand_end_effector_surface_gap"] = np.array(
        [surface_gap], dtype=np.float32
    )

    geometry_valid = (
        bool(geometry_valid_override)
        if geometry_valid_override is not None
        else bool(np.isfinite(surface_gap) and surface_gap < MISSING_DISTANCE_M)
    )
    thresholds = SafetyThresholds(
        hand_radius_m=max(0.001, float(hand_proxy_radius_m)),
        collision_gap_m=float(collision_surface_gap_m),
        near_miss_gap_m=float(near_miss_threshold_m),
        near_gap_m=float(near_human_threshold_m),
        gate_full_gap_m=DEFAULT_DISTANCE_GATE_FULL_GAP_M,
        gate_start_gap_m=DEFAULT_DISTANCE_GATE_START_GAP_M,
        max_query_gap_m=max(
            _SAFETY_THRESHOLDS.max_query_gap_m,
            DEFAULT_DISTANCE_GATE_START_GAP_M,
        ),
    ).validated()
    left_classification = classify_surface_gap(
        left_surface_gap,
        thresholds,
        contact=bool(left_hand_contact),
        geometry_valid=geometry_valid and left_surface_gap < MISSING_DISTANCE_M,
    )
    right_classification = classify_surface_gap(
        right_surface_gap,
        thresholds,
        contact=bool(right_hand_contact),
        geometry_valid=geometry_valid and right_surface_gap < MISSING_DISTANCE_M,
    )
    aggregate_classification = classify_surface_gap(
        surface_gap,
        thresholds,
        contact=bool(left_hand_contact) or bool(right_hand_contact),
        geometry_valid=geometry_valid,
    )

    if near_human is None:
        near_human = aggregate_classification.near
    if human_robot_collision is None:
        human_robot_collision = aggregate_classification.collision
    if near_miss is None:
        near_miss = aggregate_classification.near_miss
    gate = (
        float(np.clip(distance_gate_override, 0.0, 1.0))
        if distance_gate_override is not None
        else aggregate_classification.distance_gate
    )
    obs["human_robot_collision"] = _flag(bool(human_robot_collision))
    obs["near_human"] = _flag(near_human)
    obs["near_miss"] = _flag(near_miss)
    obs["left_hand_contact"] = _flag(
        bool(left_hand_contact)
        if left_hand_contact is not None
        else left_classification.collision
    )
    obs["right_hand_contact"] = _flag(
        bool(right_hand_contact)
        if right_hand_contact is not None
        else right_classification.collision
    )
    obs["distance_gate"] = np.array([gate], dtype=np.float32)
    obs["geometry_valid"] = _flag(geometry_valid)
    obs["collision_green"] = _flag(collision_green)
    obs["pick_miss_recent"] = _flag(pick_miss_recent)
    obs["drop_throw_recent"] = _flag(drop_throw_recent)
    obs["has_grasped_cube"] = _flag(has_grasped_cube)
    obs["task_phase"] = task_phase_onehot(task_phase)
    obs["controller_event"] = controller_event_onehot(controller_event)
    obs["controller_t"] = np.array([np.clip(float(controller_t), 0.0, 1.0)], dtype=np.float32)
    validate_observation(obs)
    validate_auxiliary_observation(obs)
    return obs


def task_phase_onehot(phase: str | int) -> np.ndarray:
    onehot = np.zeros(len(TASK_PHASES), dtype=np.float32)
    if isinstance(phase, int):
        if 0 <= phase < len(TASK_PHASES):
            onehot[phase] = 1.0
        return onehot
    try:
        onehot[TASK_PHASES.index(str(phase).strip().lower())] = 1.0
    except ValueError:
        pass
    return onehot


def controller_event_onehot(event: int | None) -> np.ndarray:
    onehot = np.zeros(CONTROLLER_EVENT_COUNT, dtype=np.float32)
    if event is None:
        return onehot
    event_idx = int(event)
    if 0 <= event_idx < CONTROLLER_EVENT_COUNT:
        onehot[event_idx] = 1.0
    return onehot


def _flag(value: bool) -> np.ndarray:
    return np.array([1.0 if value else 0.0], dtype=np.float32)


def _optional_vec3(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    return _fixed_array(value, 3).astype(float)


def _fixed_array(value, size: int, fill: float = 0.0) -> np.ndarray:
    result = np.full(size, fill, dtype=float)
    if value is None:
        return result
    arr = np.asarray(value, dtype=float).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        result[:n] = arr[:n]
    return result


def _safe_array_call(obj, method_name: str, size: int) -> np.ndarray:
    if obj is None or not hasattr(obj, method_name):
        return np.zeros(size, dtype=float)
    try:
        return _fixed_array(getattr(obj, method_name)(), size)
    except Exception:
        return np.zeros(size, dtype=float)


def _safe_gripper_joint_positions(robot) -> np.ndarray:
    try:
        return _fixed_array(robot.gripper.get_joint_positions(), 2)
    except Exception:
        joint_pos = _safe_array_call(robot, "get_joint_positions", 9)
        return _fixed_array(joint_pos[7:9], 2)


def _safe_world_pose(obj) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, (list, tuple, np.ndarray)):
        return _fixed_array(obj, 3), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if obj is None or not hasattr(obj, "get_world_pose"):
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    try:
        pos, quat = obj.get_world_pose()
        quat_arr = _fixed_array(quat, 4, fill=0.0)
        if not np.any(quat_arr):
            quat_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return _fixed_array(pos, 3), quat_arr
    except Exception:
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
