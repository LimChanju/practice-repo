from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .actions import (
    CONTROLLER_TARGET_ACTION_VERSION,
    MAX_EE_DELTA_M,
    MAX_YAW_DELTA_RAD,
    clip_action,
    controller_target_from_action,
)
from .observations import (
    apply_dynamic_hri_observation,
    build_observation,
    controller_event_onehot,
    flatten_observation,
    task_phase_onehot,
)
from .pick_place_phase import (
    advance_pick_place_event,
    event_gripper_command,
    task_phase_from_event,
)
from .pseudo_errp import (
    DEFAULT_PSEUDO_ERRP_SOURCES,
    PseudoErrPResult,
    extract_pseudo_errp_aux_flags,
    pseudo_errp_from_observation,
)
from .strict_task_semantics import (
    STRICT_TASK_SEMANTICS_SCHEMA,
    StrictTaskPhaseDecision,
    StrictTaskSemanticsConfig,
    StrictTaskSemanticsController,
    TaskPhysicalEvidence,
)
from .state_aware_recovery import (
    STATE_AWARE_RECOVERY_SCHEMA,
    StateAwareRecoveryBridge,
    StateAwareRecoveryConfig,
    StateAwareRecoveryDecision,
    StateAwareRecoveryEvidence,
)
from .rewards import (
    DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS,
    DEFAULT_MINIMAL_REWARD_WEIGHTS,
    DEFAULT_REWARD_WEIGHTS,
    REWARD_VERSION,
    SUPPORTED_REWARD_VERSIONS,
    IsaacFrankaDenseRewardWeights,
    MinimalEventPotentialRewardWeights,
    RewardWeights,
    compute_reward,
    is_success,
)

try:
    from v3_chan.dynamic_safety import DynamicSafetyEstimator
except ImportError:
    from dynamic_safety import DynamicSafetyEstimator

try:
    from v3_chan.robot_environment_safety import PandaEnvironmentSafetyRuntime
except ImportError:
    from robot_environment_safety import PandaEnvironmentSafetyRuntime

try:
    from v3_chan.scene_randomization import (
        resolve_active_cube_index,
        restore_cube_poses,
        restored_pose_errors,
    )
except ImportError:
    from scene_randomization import (
        resolve_active_cube_index,
        restore_cube_poses,
        restored_pose_errors,
    )

try:
    from v3_chan.physical_safety_controllers import (
        CBFConfig,
        PHYSICAL_SAFETY_MODES,
        DistalLinkVelocityCBF,
        PhysicalSafetyDiagnostics,
        mode_uses_cbf,
        mode_uses_curobo,
        mode_uses_rmpflow_obstacles,
    )
except ImportError:
    from physical_safety_controllers import (
        CBFConfig,
        PHYSICAL_SAFETY_MODES,
        DistalLinkVelocityCBF,
        PhysicalSafetyDiagnostics,
        mode_uses_cbf,
        mode_uses_curobo,
        mode_uses_rmpflow_obstacles,
    )


EXACT_POSE_ROBOT_RESET_CONTRACT = (
    "exact_pose_full_dof_q_qd_applied_targets_from_restored_measured_state_"
    "zero_tolerance_v1"
)
PICK_PLACE_BRANCH_STATE_SCHEMA = "pick_place_branch_state_v2"


GripperMode = Literal["event", "rule", "policy"]
ObservationMode = Literal["flat", "dict"]


@dataclass(frozen=True)
class PickPlaceBranchState:
    schema_version: str
    robot_joint_positions: np.ndarray
    robot_joint_velocities: np.ndarray
    robot_applied_action_state: dict[str, np.ndarray | None] | None
    cube_states: tuple[dict[str, np.ndarray], ...]
    place_target_position: np.ndarray
    place_target_orientation: np.ndarray
    place_pos: np.ndarray
    active_cube_index: int
    step_count: int
    phase_event: int
    phase_t: float
    phase_hold_steps: int
    gripper_closed: bool
    yaw: float
    rng_state: dict[str, Any]
    last_obs: dict[str, np.ndarray]
    pseudo_errp_aux_flags: dict[str, float]
    human_replay_aux_state: dict[str, Any]
    last_safety_result: Any
    dynamic_safety_state: dict[str, Any]
    last_dynamic_safety_sample: Any
    source_restoration_diagnostics: dict[str, Any]
    synthetic_human_state: dict[str, Any]
    human_replay_state: dict[str, Any] | None
    safety_geometry_state: dict[str, Any]
    last_physical_safety_diagnostics: Any
    strict_task_semantics_enabled: bool
    strict_task_semantics_config: dict[str, Any]
    strict_task_semantics_state: dict[str, Any]
    last_strict_task_decision: StrictTaskPhaseDecision | None
    state_aware_recovery_enabled: bool
    state_aware_recovery_config: dict[str, Any]
    state_aware_recovery_state: dict[str, Any]
    last_recovery_decision: StateAwareRecoveryDecision | None
    pending_recovery_handoff: StateAwareRecoveryDecision | None
    recovery_control_applied: bool
    recovery_handoff_accepted: bool
    last_recovery_alignment_token: tuple[int, int, str] | None
    last_gripper_command: str | None
    world_time: float


@dataclass
class PickPlaceEnvConfig:
    """Runtime knobs for the Isaac pick-and-place RL environment wrapper."""

    cube_count: int = 6
    max_episode_steps: int = 1200
    success_dist: float = 0.06
    action_scale: float = 1.0
    action_version: str = CONTROLLER_TARGET_ACTION_VERSION
    fixed_orientation: bool = True
    gripper_mode: GripperMode = "event"
    close_dist: float = 0.08
    release_dist: float = 0.07
    phase_gate_close_dist: float = 0.075
    phase_gate_max_hold: int = 320
    early_close_on_grasp_gate: bool = False
    fast_forward_grasp_gate: bool = False
    release_gate_dist: float | None = None
    release_gate_max_hold: int = 240
    require_release_for_success: bool = False
    synchronize_advanced_phase_observation: bool = False
    strict_task_semantics: bool = False
    strict_place_xy_tolerance_m: float = 0.04
    state_aware_recovery: bool = False
    observation_mode: ObservationMode = "flat"
    seed: int = 11
    render: bool = False
    reward_weights: RewardWeights = field(
        default_factory=lambda: DEFAULT_REWARD_WEIGHTS
    )
    reward_version: str = REWARD_VERSION
    minimal_reward_weights: MinimalEventPotentialRewardWeights = field(
        default_factory=lambda: DEFAULT_MINIMAL_REWARD_WEIGHTS
    )
    isaac_franka_dense_reward_weights: IsaacFrankaDenseRewardWeights = field(
        default_factory=lambda: DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS
    )
    pseudo_errp_enabled: bool = True
    pseudo_errp_sources: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_PSEUDO_ERRP_SOURCES
    )
    visualize_human_replay: bool = False
    human_replay_visual_z_offset: float = 0.0
    synthetic_human_enabled: bool = False
    synthetic_human_episode_prob: float = 0.35
    synthetic_human_start_min_step: int = 120
    synthetic_human_start_max_step: int = 520
    synthetic_human_duration_steps: int = 90
    synthetic_human_near_dist: float = 0.12
    synthetic_human_collision_dist: float = 0.035
    physical_safety_controller: str = "none"
    rmpflow_human_safety_margin_m: float = 0.05
    visualize_physical_safety: bool = False
    cbf_safe_gap_m: float = 0.05
    cbf_activation_gap_m: float = 0.13
    cbf_gamma_per_s: float = 8.0
    cbf_prediction_horizon_s: float = 0.15
    cbf_max_prediction_buffer_m: float = 0.08
    cbf_max_joint_speed_rad_s: float = 2.0
    cbf_objective_mode: str = "joint_nominal"
    cbf_task_space_weight: float = 1.0
    cbf_task_yaw_length_scale_m_per_rad: float = 0.10
    cbf_joint_regularization_epsilon: float = 0.05
    cbf_correction_smoothness_weight: float = 1.0
    cbf_progress_retention_rho: float = 0.70
    cbf_progress_penalty_weight: float = 50.0
    cbf_progress_nominal_threshold_mps: float = 0.01
    extended_backup_safety_geometry: bool = False


class IsaacPickPlaceEnv:
    """A light Gymnasium-style wrapper around the current Isaac pick-and-place scene.

    This class assumes `SimulationApp` has already been created by the caller.
    It owns the Isaac World, Panda robot, cubes, target marker, RMPFlow controller,
    observation construction, reward computation, and episode phase clock.
    """

    metadata = {
        "observation_modes": ("flat", "dict"),
        "action_version": CONTROLLER_TARGET_ACTION_VERSION,
    }

    def __init__(
        self,
        config: PickPlaceEnvConfig | None = None,
        *,
        human_state_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or PickPlaceEnvConfig()
        if self.config.physical_safety_controller not in PHYSICAL_SAFETY_MODES:
            raise ValueError(
                "Unknown physical_safety_controller "
                f"{self.config.physical_safety_controller!r}; "
                f"expected one of {PHYSICAL_SAFETY_MODES}"
            )
        if self.config.reward_version not in SUPPORTED_REWARD_VERSIONS:
            raise ValueError(
                f"Unsupported reward version: {self.config.reward_version}"
            )
        if self.config.rmpflow_human_safety_margin_m < 0.0:
            raise ValueError("rmpflow_human_safety_margin_m must be non-negative")
        if not isinstance(self.config.strict_task_semantics, bool):
            raise ValueError("strict_task_semantics must be boolean")
        if not isinstance(self.config.state_aware_recovery, bool):
            raise ValueError("state_aware_recovery must be boolean")
        if self.config.state_aware_recovery and not self.config.strict_task_semantics:
            raise ValueError(
                "state-aware recovery requires strict task semantics"
            )
        if self.config.strict_place_xy_tolerance_m <= 0.0:
            raise ValueError("strict_place_xy_tolerance_m must be positive")
        self.human_state_fn = human_state_fn
        self.rng = np.random.default_rng(self.config.seed)

        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.franka.controllers import RMPFlowController

        from panda_robot import add_panda
        from end_effector_safety_runtime import PandaEndEffectorSafetyRuntime
        from scene_setup import create_world, setup_scene

        self._euler_angles_to_quat = euler_angles_to_quat
        self.world = create_world()
        (
            self.cubes,
            self.place_target,
            self.table_top_z,
            self.cube_size,
            self.table_xy,
            self.table_size,
            self.stack_base_xy,
        ) = setup_scene(
            self.world,
            cube_count=self.config.cube_count,
            rng=self.rng,
        )
        self.pick_targets = self.cubes[: min(3, len(self.cubes))]
        self.cube_half = self.cube_size / 2.0
        self.cube_center_z = self.table_top_z + self.cube_half
        self.place_pos = np.array(
            [self.stack_base_xy[0], self.stack_base_xy[1], self.cube_center_z]
        )
        self.place_target.set_world_pose(position=self.place_pos)

        self.robot = add_panda(self.world, base_z=self.table_top_z)
        self.world.reset()
        self.world.play()
        self.controller = RMPFlowController(
            name="rl_env_rmpflow_controller", robot_articulation=self.robot
        )
        self.safety_geometry = PandaEndEffectorSafetyRuntime(
            robot_prim_path="/World/Franka"
        )
        self.environment_safety = (
            PandaEnvironmentSafetyRuntime(robot_prim_path="/World/Franka")
            if self.config.extended_backup_safety_geometry
            else None
        )
        self.dynamic_safety = DynamicSafetyEstimator()
        self.physics_dt_s = float(self.world.get_physics_dt())
        self._physical_safety_mode = str(self.config.physical_safety_controller)
        self._cbf_filter = (
            DistalLinkVelocityCBF(
                CBFConfig(
                    safe_gap_m=self.config.cbf_safe_gap_m,
                    activation_gap_m=self.config.cbf_activation_gap_m,
                    gamma_per_s=self.config.cbf_gamma_per_s,
                    prediction_horizon_s=self.config.cbf_prediction_horizon_s,
                    max_prediction_buffer_m=self.config.cbf_max_prediction_buffer_m,
                    max_joint_speed_rad_s=self.config.cbf_max_joint_speed_rad_s,
                    objective_mode=self.config.cbf_objective_mode,
                    task_space_weight=self.config.cbf_task_space_weight,
                    task_yaw_length_scale_m_per_rad=(
                        self.config.cbf_task_yaw_length_scale_m_per_rad
                    ),
                    joint_regularization_epsilon=(
                        self.config.cbf_joint_regularization_epsilon
                    ),
                    correction_smoothness_weight=(
                        self.config.cbf_correction_smoothness_weight
                    ),
                    progress_retention_rho=(
                        self.config.cbf_progress_retention_rho
                    ),
                    progress_penalty_weight=(
                        self.config.cbf_progress_penalty_weight
                    ),
                    progress_nominal_threshold_mps=(
                        self.config.cbf_progress_nominal_threshold_mps
                    ),
                )
            )
            if mode_uses_cbf(self._physical_safety_mode)
            else None
        )
        self._curobo_controller = None
        self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
            controller=self._physical_safety_mode,
            objective_mode=self.config.cbf_objective_mode,
        )
        self._strict_task_semantics_config = StrictTaskSemanticsConfig(
            enabled=bool(self.config.strict_task_semantics),
            place_xy_tolerance_m=float(
                self.config.strict_place_xy_tolerance_m
            ),
            state_aware_recovery=bool(self.config.state_aware_recovery),
        ).validated()
        self._strict_task_controller = StrictTaskSemanticsController(
            self._strict_task_semantics_config
        )
        self._last_strict_task_decision: StrictTaskPhaseDecision | None = None
        self._last_gripper_command: str | None = None
        self._state_aware_recovery_config = StateAwareRecoveryConfig(
            enabled=bool(self.config.state_aware_recovery)
        ).validated()
        self._state_aware_recovery = StateAwareRecoveryBridge(
            self._state_aware_recovery_config
        )
        self._last_recovery_decision: StateAwareRecoveryDecision | None = None
        self._pending_recovery_handoff: StateAwareRecoveryDecision | None = None
        self._recovery_control_applied = False
        self._recovery_handoff_accepted = False
        self._last_recovery_alignment_token: tuple[int, int, str] | None = None
        self._rmpflow_human_obstacles: dict[str, Any] = {}
        self._rmpflow_obstacles_registered = False
        self._rmpflow_valid_hand_count = 0

        self.episode_index = 0
        self.current_episode_index = 0
        self.active_cube = self.pick_targets[0]
        self.step_count = 0
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0
        self._last_obs: dict[str, np.ndarray] | None = None
        self._pseudo_errp_aux_flags: dict[str, float] = {}
        self._human_replay_aux_state: dict[str, Any] = {}
        self._last_safety_result = None
        self._last_environment_safety_result = None
        self._last_dynamic_safety_sample = None
        self._last_command_provenance: dict[str, Any] = {}
        self._source_restoration_diagnostics = _empty_source_restoration_diagnostics()
        self._synthetic_human_active = False
        self._synthetic_human_start_step = 0
        self._synthetic_human_duration_steps = 0
        self._synthetic_human_side = 1.0
        self._synthetic_human_height_offset = 0.0
        self._human_visual_prims: dict[str, Any] = {}
        if self.config.visualize_human_replay:
            self._setup_human_visuals()
        if mode_uses_rmpflow_obstacles(self._physical_safety_mode):
            self._setup_rmpflow_human_obstacles()
        if mode_uses_curobo(self._physical_safety_mode):
            self._setup_curobo_controller()

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (5,)

    @property
    def observation_shape(self) -> tuple[int, ...] | None:
        if self.config.observation_mode == "dict":
            return None
        from .observations import OBSERVATION_DIM

        return (OBSERVATION_DIM,)

    def reset(
        self,
        *,
        seed: int | None = None,
        active_cube_index: int | None = None,
        source_restoration: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        else:
            np.random.seed(int(self.rng.integers(0, 2**31 - 1)))

        from scene_setup import randomize_cubes

        restoration = _normalize_source_restoration(source_restoration, seed)
        restoration_mode = str(restoration["restoration_mode"])
        source = dict(restoration.get("source_configuration", {}))
        if source_restoration is not None and not bool(
            restoration.get("source_configuration_available", False)
        ):
            raise ValueError(
                "source_configuration_unavailable: "
                f"{restoration.get('restoration_reason', 'unknown')}"
            )

        if restoration_mode == "exact_pose":
            restore_cube_poses(
                self.cubes,
                source["cube_names"],
                source["cube_positions_world"],
                source["cube_orientations_wxyz"],
                set_default_state=True,
            )
            _set_robot_default_state(
                self.robot,
                source.get("robot_initial_joint_positions"),
                source.get("robot_initial_joint_velocities"),
            )
        else:
            scene_rng = self.rng
            if restoration_mode == "collection_seed":
                scene_rng = np.random.default_rng(int(restoration["layout_seed"]))
            randomize_cubes(
                self.cubes,
                self.table_xy,
                self.table_size,
                self.cube_center_z,
                self.cube_size,
                forbidden_xy=self.stack_base_xy,
                rng=scene_rng,
            )
        self.world.reset()
        self.world.play()
        self.controller.reset()
        self._rmpflow_obstacles_registered = False
        self._register_rmpflow_human_obstacles()
        if self._cbf_filter is not None:
            self._cbf_filter.reset()
        if self._curobo_controller is not None:
            self._curobo_controller.reset()
        self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
            controller=self._physical_safety_mode,
            objective_mode=self.config.cbf_objective_mode,
        )
        self.safety_geometry.reset_link_origin_pose_cache()
        if restoration_mode == "exact_pose":
            restore_cube_poses(
                self.cubes,
                source["cube_names"],
                source["cube_positions_world"],
                source["cube_orientations_wxyz"],
                set_default_state=False,
            )
            self.place_target.set_world_pose(
                position=np.asarray(source["place_target_position_world"], dtype=float),
                orientation=np.asarray(
                    source["place_target_orientation_wxyz"], dtype=float
                ),
            )
            robot_restored = _set_robot_joint_state(
                self.robot,
                source.get("robot_initial_joint_positions"),
                source.get("robot_initial_joint_velocities"),
            )
            if not robot_restored:
                raise ValueError("source_robot_state_restore_failed")
            restoration["robot_reset_exactness"] = (
                _canonicalize_exact_robot_reset(
                    self.robot,
                    source.get("robot_initial_joint_positions"),
                    source.get("robot_initial_joint_velocities"),
                )
            )
        else:
            self.place_target.set_world_pose(position=self.place_pos)
            robot_restored = False

        self.current_episode_index = self.episode_index
        if source_restoration is not None:
            active_cube_index = restoration.get("source_cube_index")
        elif active_cube_index is None:
            active_cube_index = self.current_episode_index % len(self.pick_targets)
        screening_cube_index = resolve_active_cube_index(
            active_cube_index,
            episode_index=self.current_episode_index,
            cube_count=len(self.pick_targets),
        )
        self.active_cube = self.pick_targets[screening_cube_index]
        restoration["screening_cube_index"] = int(screening_cube_index)
        restoration["screening_cube_name"] = str(getattr(self.active_cube, "name", ""))
        if restoration_mode == "exact_pose":
            verification = _verify_exact_restoration(
                self.cubes,
                self.place_target,
                source,
                robot_restored=robot_restored,
            )
            source_cube_name = restoration.get("source_cube_name")
            if (
                source_cube_name
                and str(source_cube_name) != restoration["screening_cube_name"]
            ):
                verification["pose_mismatch"] = True
                reasons = [
                    item
                    for item in str(verification["pose_mismatch_reason"]).split(",")
                    if item
                ]
                reasons.append("active_cube_identity_mismatch")
                verification["pose_mismatch_reason"] = ",".join(reasons)
            restoration.update(verification)
            if bool(restoration["pose_mismatch"]):
                raise ValueError(
                    "source_configuration_pose_mismatch: "
                    f"{restoration['pose_mismatch_reason']}"
                )
        self._source_restoration_diagnostics = restoration
        self.step_count = 0
        self.phase_event = 0
        self.phase_t = 0.0
        self.phase_hold_steps = 0
        self.gripper_closed = False
        self.yaw = 0.0
        self._reset_synthetic_human()
        self.dynamic_safety.reset()
        self._last_dynamic_safety_sample = None
        self._last_environment_safety_result = None
        self._last_command_provenance = {}

        obs = self._build_obs()
        self._reset_strict_task_semantics(obs)
        if self._strict_task_semantics_enabled():
            self._write_phase_observation(obs)
        self._last_obs = obs
        errp_result = self._pseudo_errp_result(obs, override_feedback=0.0)
        info = self._info(obs, reward_components={}, errp_result=errp_result)
        self.episode_index += 1
        return self._format_obs(obs), info

    def step(
        self,
        action: np.ndarray,
        *,
        errp_feedback: float | None = None,
        advance_task_phase: bool = True,
        reset_task_phase_after_step: bool = False,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step()")
        if bool(advance_task_phase) and bool(reset_task_phase_after_step):
            raise ValueError(
                "reset_task_phase_after_step requires advance_task_phase=False"
            )

        action = _finite_action(action)
        pre_joint_positions = _safe_robot_joint_vector(
            self.robot, "get_joint_positions"
        )
        pre_joint_velocities = _safe_robot_joint_vector(
            self.robot, "get_joint_velocities"
        )
        self._recovery_handoff_accepted = False
        recovery_decision = self._prepare_state_aware_recovery_control(
            self._last_obs
        )
        self._last_recovery_decision = recovery_decision
        self._pending_recovery_handoff = (
            recovery_decision if recovery_decision.handoff else None
        )
        self._recovery_control_applied = bool(
            recovery_decision.target_position_m is not None
            or recovery_decision.timed_out
        )
        if recovery_decision.target_position_m is not None:
            target_pos = np.asarray(
                recovery_decision.target_position_m, dtype=float
            ).reshape(3)
            target_quat = self._fixed_target_orientation(self.yaw)
            gripper_command = self._recovery_gripper_command(
                recovery_decision.desired_gripper_closed
            )
        elif recovery_decision.timed_out:
            target_pos = np.asarray(self._last_obs["ee_pos"], dtype=float).reshape(3)
            target_quat = self._fixed_target_orientation(self.yaw)
            gripper_command = None
        else:
            target_pos, target_quat, self.yaw = self._target_from_action(action)
            gripper_command = self._gripper_command(action, self._last_obs)
        self._last_gripper_command = gripper_command

        if self._curobo_controller is not None:
            arm_action, controller_diagnostics = self._curobo_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat,
                observation=self._last_obs,
            )
            self._last_physical_safety_diagnostics = controller_diagnostics
        else:
            arm_action = self.controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat,
            )
            self._last_physical_safety_diagnostics = PhysicalSafetyDiagnostics(
                controller=self._physical_safety_mode,
                objective_mode=self.config.cbf_objective_mode,
                active=bool(
                    mode_uses_rmpflow_obstacles(self._physical_safety_mode)
                    and self._rmpflow_valid_hand_count > 0
                    and self._last_safety_result is not None
                    and self._last_safety_result.distance_gate > 0.0
                ),
                valid_hand_count=int(self._rmpflow_valid_hand_count),
                status=(
                    "dynamic_obstacles_registered"
                    if mode_uses_rmpflow_obstacles(self._physical_safety_mode)
                    else "inactive"
                ),
            )
        if (
            self._cbf_filter is not None
            and self._last_safety_result is not None
            and self._last_dynamic_safety_sample is not None
        ):
            arm_action, self._last_physical_safety_diagnostics = (
                self._cbf_filter.filter_action(
                    robot=self.robot,
                    arm_action=arm_action,
                    safety_result=self._last_safety_result,
                    dynamic_sample=self._last_dynamic_safety_sample,
                    safety_geometry=self.safety_geometry,
                    observation=self._last_obs,
                    physics_dt_s=self.physics_dt_s,
                    task_progress_context=self._cbf_task_progress_context(
                        target_pos=target_pos,
                        recovery_decision=recovery_decision,
                    ),
                    human_valid_mask=self._human_replay_aux_state.get(
                        "human_valid_mask"
                    ),
                    intentional_human_absence=(
                        _intentional_human_absence_from_aux(
                            self._human_replay_aux_state
                        )
                    ),
                )
            )
        gripper_command = self._guard_strict_release_for_current_cbf(
            gripper_command
        )
        self._last_gripper_command = gripper_command
        control_action = self._merge_gripper_action(arm_action, gripper_command)
        arm_command_committed = bool(gripper_command is None)
        applied_action_payload = _articulation_action_payload(control_action)
        self.robot.apply_action(control_action)
        if self._cbf_filter is not None:
            self._cbf_filter.notify_action_committed(arm_command_committed)
        if self._strict_task_semantics_enabled():
            self._strict_task_controller.notify_gripper_command_applied(
                event=int(self.phase_event),
                command=gripper_command,
                grasp_candidate_before_command=bool(
                    float(
                        np.asarray(
                            self._last_obs["has_grasped_cube"], dtype=float
                        ).reshape(-1)[0]
                    )
                    > 0.5
                ),
            )
        self.world.step(render=self.config.render)
        self.step_count += 1

        self._last_command_provenance = {
            "schema_version": "physical_command_provenance_v1",
            "arm_command_committed": arm_command_committed,
            "gripper_command": gripper_command,
            "pre_joint_positions_rad": pre_joint_positions,
            "pre_joint_velocities_radps": pre_joint_velocities,
            "applied_action": applied_action_payload,
            "post_joint_positions_rad": _safe_robot_joint_vector(
                self.robot, "get_joint_positions"
            ),
            "post_joint_velocities_radps": _safe_robot_joint_vector(
                self.robot, "get_joint_velocities"
            ),
            "physics_dt_s": float(self.physics_dt_s),
        }

        next_obs = self._build_obs()
        phase_control = self._control_task_phase(
            next_obs,
            advance=bool(advance_task_phase),
            reset_for_reentry=bool(reset_task_phase_after_step),
        )
        success = self._is_success(next_obs)
        strict_failure = bool(
            self._strict_task_semantics_enabled()
            and self._strict_task_controller.state.failure_reason
        )
        terminated, truncated = _task_episode_flags(
            success=success,
            strict_failure=strict_failure,
            horizon_reached=(
                self.step_count >= self.config.max_episode_steps
            ),
        )
        errp_result = self._pseudo_errp_result(
            next_obs, override_feedback=errp_feedback
        )
        reward_result = compute_reward(
            self._last_obs,
            next_obs,
            action,
            errp_feedback=errp_result.feedback,
            success=success,
            success_dist=self.config.success_dist,
            weights=self.config.reward_weights,
            reward_version=self.config.reward_version,
            minimal_weights=self.config.minimal_reward_weights,
            isaac_franka_dense_weights=(
                self.config.isaac_franka_dense_reward_weights
            ),
        )
        self._last_obs = next_obs

        info = self._info(
            next_obs,
            reward_components=reward_result.components,
            errp_result=errp_result,
        )
        info["task_phase_control"] = phase_control
        info["task_phase_advanced"] = bool(phase_control["advanced"])
        info["task_phase_progressed"] = bool(phase_control["phase_changed"])
        info["task_phase_paused"] = bool(phase_control["paused"])
        info["task_phase_reentry"] = bool(phase_control["reentry"])
        return (
            self._format_obs(next_obs),
            reward_result.total,
            terminated,
            truncated,
            info,
        )

    def capture_branch_state(self) -> PickPlaceBranchState:
        """Capture the state needed for finite-horizon risk-label branches.

        Branching deliberately excludes active RMPflow/CBF/cuRobo shields: labels
        must describe the candidate task action and the learned backup policy,
        rather than a third controller's intervention.
        """

        if self._last_obs is None:
            raise RuntimeError("reset() must be called before capture_branch_state()")
        if self._physical_safety_mode != "none":
            raise RuntimeError(
                "Risk-label branching requires physical_safety_controller='none'"
            )
        replay_state = None
        if self.human_state_fn is not None:
            capture_replay = getattr(self.human_state_fn, "capture_state", None)
            if not callable(capture_replay):
                raise RuntimeError(
                    "human_state_fn must implement capture_state() for branching"
                )
            replay_state = capture_replay()

        active_cube_index = next(
            (
                index
                for index, cube in enumerate(self.pick_targets)
                if cube is self.active_cube
            ),
            -1,
        )
        if active_cube_index < 0:
            raise RuntimeError("active_cube is not present in pick_targets")
        target_position, target_orientation = self.place_target.get_world_pose()
        return PickPlaceBranchState(
            schema_version=PICK_PLACE_BRANCH_STATE_SCHEMA,
            robot_joint_positions=_required_runtime_vector(
                self.robot.get_joint_positions(), "robot_joint_positions"
            ),
            robot_joint_velocities=_required_runtime_vector(
                self.robot.get_joint_velocities(), "robot_joint_velocities"
            ),
            robot_applied_action_state=_capture_robot_applied_action_state(self.robot),
            cube_states=tuple(_capture_body_state(cube) for cube in self.cubes),
            place_target_position=_required_runtime_vector(
                target_position, "place_target_position", min_size=3
            )[:3],
            place_target_orientation=_required_runtime_vector(
                target_orientation, "place_target_orientation", min_size=4
            )[:4],
            place_pos=np.asarray(self.place_pos, dtype=float).copy(),
            active_cube_index=int(active_cube_index),
            step_count=int(self.step_count),
            phase_event=int(self.phase_event),
            phase_t=float(self.phase_t),
            phase_hold_steps=int(self.phase_hold_steps),
            gripper_closed=bool(self.gripper_closed),
            yaw=float(self.yaw),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
            last_obs=_copy_observation(self._last_obs),
            pseudo_errp_aux_flags=copy.deepcopy(self._pseudo_errp_aux_flags),
            human_replay_aux_state=copy.deepcopy(self._human_replay_aux_state),
            last_safety_result=copy.deepcopy(self._last_safety_result),
            dynamic_safety_state=copy.deepcopy(self.dynamic_safety.__dict__),
            last_dynamic_safety_sample=copy.deepcopy(self._last_dynamic_safety_sample),
            source_restoration_diagnostics=copy.deepcopy(
                self._source_restoration_diagnostics
            ),
            synthetic_human_state={
                "active": bool(self._synthetic_human_active),
                "start_step": int(self._synthetic_human_start_step),
                "duration_steps": int(self._synthetic_human_duration_steps),
                "side": float(self._synthetic_human_side),
                "height_offset": float(self._synthetic_human_height_offset),
            },
            human_replay_state=copy.deepcopy(replay_state),
            safety_geometry_state=_capture_safety_geometry_state(self.safety_geometry),
            last_physical_safety_diagnostics=copy.deepcopy(
                self._last_physical_safety_diagnostics
            ),
            strict_task_semantics_enabled=self._strict_task_semantics_enabled(),
            strict_task_semantics_config=copy.deepcopy(
                self._strict_task_semantics_config.as_dict()
            ),
            strict_task_semantics_state=copy.deepcopy(
                self._strict_task_controller.state.as_dict()
            ),
            last_strict_task_decision=copy.deepcopy(
                self._last_strict_task_decision
            ),
            state_aware_recovery_enabled=self._state_aware_recovery_enabled(),
            state_aware_recovery_config=copy.deepcopy(
                self._state_aware_recovery_config.as_dict()
            ),
            state_aware_recovery_state=copy.deepcopy(
                self._state_aware_recovery.state.as_dict()
            ),
            last_recovery_decision=copy.deepcopy(
                self._last_recovery_decision
            ),
            pending_recovery_handoff=copy.deepcopy(
                self._pending_recovery_handoff
            ),
            recovery_control_applied=bool(self._recovery_control_applied),
            recovery_handoff_accepted=bool(self._recovery_handoff_accepted),
            last_recovery_alignment_token=copy.deepcopy(
                self._last_recovery_alignment_token
            ),
            last_gripper_command=self._last_gripper_command,
            world_time=float(getattr(self.world, "current_time", 0.0)),
        )

    def refresh_observation(
        self,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        """Rebuild observation/geometry after an explicit state restoration."""

        if self._last_obs is None:
            raise RuntimeError("reset() must be called before refresh_observation()")
        self.dynamic_safety.reset()
        self._last_dynamic_safety_sample = None
        self.safety_geometry.reset_link_origin_pose_cache()
        obs = self._build_obs()
        self._write_phase_observation(obs)
        self._last_obs = obs
        errp_result = self._pseudo_errp_result(obs, override_feedback=0.0)
        info = self._info(obs, reward_components={}, errp_result=errp_result)
        return self._format_obs(obs), info

    def restore_branch_state(self, state: PickPlaceBranchState) -> dict[str, float]:
        if state.schema_version != PICK_PLACE_BRANCH_STATE_SCHEMA:
            raise ValueError(f"Unsupported branch state: {state.schema_version}")
        if self._physical_safety_mode != "none":
            raise RuntimeError(
                "Risk-label branching requires physical_safety_controller='none'"
            )
        if len(state.cube_states) != len(self.cubes):
            raise ValueError("Branch state cube count does not match the environment")
        if bool(state.strict_task_semantics_enabled) != bool(
            self._strict_task_semantics_enabled()
        ):
            raise ValueError(
                "Branch state strict-task mode does not match the environment"
            )
        if state.strict_task_semantics_config != (
            self._strict_task_semantics_config.as_dict()
        ):
            raise ValueError(
                "Branch state strict-task config does not match the environment"
            )
        if bool(state.state_aware_recovery_enabled) != bool(
            self._state_aware_recovery_enabled()
        ):
            raise ValueError(
                "Branch state recovery mode does not match the environment"
            )
        if (
            bool(state.state_aware_recovery_enabled)
            and not bool(state.strict_task_semantics_enabled)
        ):
            raise ValueError(
                "Branch state recovery requires strict-task semantics"
            )
        if state.state_aware_recovery_config != (
            self._state_aware_recovery_config.as_dict()
        ):
            raise ValueError(
                "Branch state recovery config does not match the environment"
            )

        self.robot.set_joint_positions(state.robot_joint_positions.copy())
        if hasattr(self.robot, "set_joint_velocities"):
            self.robot.set_joint_velocities(state.robot_joint_velocities.copy())
        for cube, cube_state in zip(self.cubes, state.cube_states):
            _restore_body_state(cube, cube_state)
        self.place_target.set_world_pose(
            position=state.place_target_position.copy(),
            orientation=state.place_target_orientation.copy(),
        )
        self.place_pos = state.place_pos.copy()
        self.active_cube = self.pick_targets[state.active_cube_index]

        if state.human_replay_state is not None:
            restore_replay = getattr(self.human_state_fn, "restore_state", None)
            if not callable(restore_replay):
                raise RuntimeError(
                    "human_state_fn must implement restore_state() for branching"
                )
            restore_replay(copy.deepcopy(state.human_replay_state))

        self.step_count = int(state.step_count)
        self.phase_event = int(state.phase_event)
        self.phase_t = float(state.phase_t)
        self.phase_hold_steps = int(state.phase_hold_steps)
        self.gripper_closed = bool(state.gripper_closed)
        self.yaw = float(state.yaw)
        self.rng.bit_generator.state = copy.deepcopy(state.rng_state)
        self._last_obs = _copy_observation(state.last_obs)
        self._pseudo_errp_aux_flags = copy.deepcopy(state.pseudo_errp_aux_flags)
        self._human_replay_aux_state = copy.deepcopy(state.human_replay_aux_state)
        self._last_safety_result = copy.deepcopy(state.last_safety_result)
        self.dynamic_safety.__dict__.clear()
        self.dynamic_safety.__dict__.update(copy.deepcopy(state.dynamic_safety_state))
        self._last_dynamic_safety_sample = copy.deepcopy(
            state.last_dynamic_safety_sample
        )
        self._source_restoration_diagnostics = copy.deepcopy(
            state.source_restoration_diagnostics
        )
        self._synthetic_human_active = bool(state.synthetic_human_state["active"])
        self._synthetic_human_start_step = int(
            state.synthetic_human_state["start_step"]
        )
        self._synthetic_human_duration_steps = int(
            state.synthetic_human_state["duration_steps"]
        )
        self._synthetic_human_side = float(state.synthetic_human_state["side"])
        self._synthetic_human_height_offset = float(
            state.synthetic_human_state["height_offset"]
        )
        self._last_physical_safety_diagnostics = copy.deepcopy(
            state.last_physical_safety_diagnostics
        )
        self._strict_task_controller.restore_state(
            copy.deepcopy(state.strict_task_semantics_state)
        )
        self._last_strict_task_decision = copy.deepcopy(
            state.last_strict_task_decision
        )
        self._state_aware_recovery.restore_state(
            copy.deepcopy(state.state_aware_recovery_state)
        )
        self._last_recovery_decision = copy.deepcopy(
            state.last_recovery_decision
        )
        self._pending_recovery_handoff = copy.deepcopy(
            state.pending_recovery_handoff
        )
        self._recovery_control_applied = bool(state.recovery_control_applied)
        self._recovery_handoff_accepted = bool(
            state.recovery_handoff_accepted
        )
        self._last_recovery_alignment_token = copy.deepcopy(
            state.last_recovery_alignment_token
        )
        self._last_gripper_command = state.last_gripper_command
        self.controller.reset()
        _restore_robot_applied_action_state(
            self.robot,
            state.robot_applied_action_state,
        )
        self.safety_geometry.reset_link_origin_pose_cache()
        _restore_safety_geometry_state(
            self.safety_geometry, state.safety_geometry_state
        )

        current_world_time = float(getattr(self.world, "current_time", 0.0))
        return {
            "captured_world_time": float(state.world_time),
            "restored_world_time": current_world_time,
            "world_time_advance_s": max(
                0.0, current_world_time - float(state.world_time)
            ),
        }

    def close(self) -> None:
        self.world.stop()

    def _build_obs(self) -> dict[str, np.ndarray]:
        finger_positions = _gripper_finger_world_positions(self.robot)
        gripper_center = (
            None
            if finger_positions is None
            else (finger_positions[0] + finger_positions[1]) * 0.5
        )
        ee_pos = None
        try:
            ee_pos, _ = self.robot.end_effector.get_world_pose()
            ee_pos = np.asarray(ee_pos, dtype=float).reshape(-1)[:3]
        except Exception:
            ee_pos = None
        if gripper_center is None:
            gripper_center = ee_pos
        has_grasped = _has_grasped_cube(self.robot, self.active_cube, gripper_center)
        task_phase = task_phase_from_event(self.phase_event)
        replay_context_setter = getattr(
            self.human_state_fn,
            "set_runtime_context",
            None,
        )
        if callable(replay_context_setter):
            replay_context_setter(
                step=self.step_count,
                task_phase=task_phase,
                controller_event=(
                    -1 if self.phase_event is None else int(self.phase_event)
                ),
                controller_t=int(self.phase_t),
                ee_pos=ee_pos,
                playback_time_s=float(self.step_count) * self.physics_dt_s,
            )
        human_state = dict(
            self.human_state_fn() if self.human_state_fn is not None else {}
        )
        synthetic_state = self._synthetic_human_state(gripper_center)
        human_state = {**synthetic_state, **human_state}
        self._update_rmpflow_human_obstacles(human_state)
        if self._curobo_controller is not None:
            self._curobo_controller.update_human_obstacles(human_state)
        human_state, self._pseudo_errp_aux_flags = extract_pseudo_errp_aux_flags(
            human_state
        )
        human_state, self._human_replay_aux_state = _split_observation_human_state(
            human_state
        )
        safety_result = self.safety_geometry.evaluate(
            human_state.get("human_left_hand_pos"),
            human_state.get("human_right_hand_pos"),
        )
        self._last_safety_result = safety_result
        self._last_environment_safety_result = (
            None
            if self.environment_safety is None
            else self.environment_safety.evaluate()
        )
        dynamic_sample = self._update_dynamic_safety(
            safety_result,
            human_state.get("human_left_hand_pos"),
            human_state.get("human_right_hand_pos"),
        )
        self._last_dynamic_safety_sample = dynamic_sample
        # Recorded labels never drive a rollout. Recompute them against the
        # current robot pose and its composed PhysX collision shapes.
        human_state.update(
            {
                "human_robot_collision": safety_result.collision,
                "near_human": safety_result.near,
                "near_miss": safety_result.near_miss,
                "min_hand_gripper_surface_gap_override": safety_result.min_surface_gap_m,
                "left_hand_surface_gap_override": safety_result.left.surface_gap_m,
                "right_hand_surface_gap_override": safety_result.right.surface_gap_m,
                "left_hand_contact": safety_result.left.contact,
                "right_hand_contact": safety_result.right.contact,
                "distance_gate_override": safety_result.distance_gate,
                "geometry_valid_override": safety_result.geometry_valid,
            }
        )
        obs = build_observation(
            robot=self.robot,
            cube=self.active_cube,
            place_target=self.place_pos,
            gripper_center_pos=gripper_center,
            has_grasped_cube=has_grasped,
            task_phase=task_phase,
            controller_event=self.phase_event,
            controller_t=self.phase_t,
            **human_state,
        )
        if finger_positions is not None:
            cube_position = np.asarray(obs["cube_pos"], dtype=float).reshape(-1)[:3]
            obs["_reward_isaac_franka_left_finger_to_cube"] = (
                cube_position - finger_positions[0]
            ).astype(np.float32)
            obs["_reward_isaac_franka_right_finger_to_cube"] = (
                cube_position - finger_positions[1]
            ).astype(np.float32)
        apply_dynamic_hri_observation(
            obs,
            {
                **dynamic_sample.human_payload(),
                **dynamic_sample.safety_payload(),
            },
        )
        if self.phase_event is None:
            obs["task_phase"] = task_phase_onehot("approach_cube")
        self._update_human_visuals(obs)
        return obs

    def _update_dynamic_safety(
        self,
        safety_result,
        left_hand_pos,
        right_hand_pos,
    ):
        left_origin, left_orientation, _ = self.safety_geometry.closest_link_world_pose(
            safety_result.left
        )
        right_origin, right_orientation, _ = (
            self.safety_geometry.closest_link_world_pose(safety_result.right)
        )
        _, left_angular_velocity, _ = self.safety_geometry.closest_link_world_velocity(
            safety_result.left
        )
        _, right_angular_velocity, _ = self.safety_geometry.closest_link_world_velocity(
            safety_result.right
        )
        left_surface_point, _ = (
            self.safety_geometry.closest_surface_point_world_position(
                safety_result.left,
                left_hand_pos,
            )
        )
        right_surface_point, _ = (
            self.safety_geometry.closest_surface_point_world_position(
                safety_result.right,
                right_hand_pos,
            )
        )
        return self.dynamic_safety.update(
            sim_time_s=float(self.step_count) * self.physics_dt_s,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_tracking_valid=_valid_runtime_position(left_hand_pos),
            right_tracking_valid=_valid_runtime_position(right_hand_pos),
            left_surface_gap_m=safety_result.left.surface_gap_m,
            right_surface_gap_m=safety_result.right.surface_gap_m,
            left_geometry_valid=safety_result.left.geometry_valid,
            right_geometry_valid=safety_result.right.geometry_valid,
            left_closest_collider_id=safety_result.left.closest_collider_id,
            right_closest_collider_id=safety_result.right.closest_collider_id,
            left_closest_robot_origin_pos=left_origin,
            right_closest_robot_origin_pos=right_origin,
            left_closest_robot_orientation_wxyz=left_orientation,
            right_closest_robot_orientation_wxyz=right_orientation,
            left_closest_surface_point_world_pos=left_surface_point,
            right_closest_surface_point_world_pos=right_surface_point,
            left_closest_robot_angular_velocity_world_radps=left_angular_velocity,
            right_closest_robot_angular_velocity_world_radps=right_angular_velocity,
        )

    def _setup_human_visuals(self) -> None:
        from omni.isaac.core.objects import VisualSphere

        specs = (
            (
                "head",
                "/World/HumanReplay/head",
                "human_replay_head",
                0.045,
                np.array([0.8, 0.8, 0.8]),
            ),
            (
                "left",
                "/World/HumanReplay/left_hand",
                "human_replay_left_hand",
                0.035,
                np.array([0.45, 0.65, 1.0]),
            ),
            (
                "right",
                "/World/HumanReplay/right_hand",
                "human_replay_right_hand",
                0.035,
                np.array([1.0, 0.55, 0.25]),
            ),
        )
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        for key, prim_path, name, radius, color in specs:
            self._human_visual_prims[key] = self.world.scene.add(
                VisualSphere(
                    prim_path=prim_path,
                    name=name,
                    position=parked,
                    radius=radius,
                    color=color,
                )
            )

    def _setup_rmpflow_human_obstacles(self) -> None:
        from omni.isaac.core.objects import VisualSphere

        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        radius = float(
            self.safety_geometry.thresholds.hand_radius_m
            + self.config.rmpflow_human_safety_margin_m
        )
        visible = bool(self.config.visualize_physical_safety)
        for hand, color in (
            ("left", np.array([0.1, 0.9, 0.9])),
            ("right", np.array([0.95, 0.25, 0.25])),
        ):
            self._rmpflow_human_obstacles[hand] = self.world.scene.add(
                VisualSphere(
                    prim_path=f"/World/PhysicalSafety/rmpflow_{hand}_hand",
                    name=f"rmpflow_{hand}_hand_obstacle",
                    position=parked,
                    radius=radius,
                    color=color,
                    visible=visible,
                )
            )
        self._register_rmpflow_human_obstacles()

    def _register_rmpflow_human_obstacles(self) -> None:
        if self._rmpflow_obstacles_registered:
            return
        if not self._rmpflow_human_obstacles:
            return
        for obstacle in self._rmpflow_human_obstacles.values():
            self.controller.add_obstacle(obstacle, static=False)
        self._rmpflow_obstacles_registered = True

    def _update_rmpflow_human_obstacles(self, human_state: dict[str, Any]) -> None:
        if not self._rmpflow_human_obstacles:
            self._rmpflow_valid_hand_count = 0
            return
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        valid_count = 0
        for hand, obstacle in self._rmpflow_human_obstacles.items():
            value = human_state.get(f"human_{hand}_hand_pos")
            if _valid_runtime_position(value):
                position = np.asarray(value, dtype=float).reshape(-1)[:3]
                valid_count += 1
            else:
                position = parked
            obstacle.set_world_pose(position=position)
        self._rmpflow_valid_hand_count = valid_count

    def _setup_curobo_controller(self) -> None:
        try:
            from v3_chan.curobo_mpc_controller import CuRoboMpcArmController
        except ImportError:
            from curobo_mpc_controller import CuRoboMpcArmController

        self._curobo_controller = CuRoboMpcArmController(
            robot=self.robot,
            physics_dt_s=self.physics_dt_s,
            hand_radius_m=self.safety_geometry.thresholds.hand_radius_m,
            safety_margin_m=self.config.rmpflow_human_safety_margin_m,
            table_center_world_m=np.array(
                [
                    self.table_xy[0],
                    self.table_xy[1],
                    self.table_top_z - (self.table_size[2] / 2.0),
                ],
                dtype=float,
            ),
            table_size_m=np.asarray(self.table_size, dtype=float),
        )

    def _update_human_visuals(self, obs: dict[str, np.ndarray]) -> None:
        if not self._human_visual_prims:
            return
        fields = {
            "head": "human_head_pos",
            "left": "human_left_hand_pos",
            "right": "human_right_hand_pos",
        }
        parked = np.array([0.0, 0.0, -10.0], dtype=float)
        for key, field_name in fields.items():
            prim = self._human_visual_prims.get(key)
            if prim is None:
                continue
            pos = np.asarray(obs.get(field_name, parked), dtype=float).reshape(-1)
            if (
                pos.size < 3
                or not np.all(np.isfinite(pos[:3]))
                or np.linalg.norm(pos[:3]) < 1e-6
            ):
                pos = parked
            else:
                pos = pos[:3].copy()
                pos[2] += float(self.config.human_replay_visual_z_offset)
            prim.set_world_pose(position=pos[:3])

    def _reset_synthetic_human(self) -> None:
        cfg = self.config
        self._synthetic_human_active = bool(cfg.synthetic_human_enabled) and float(
            self.rng.random()
        ) < float(np.clip(cfg.synthetic_human_episode_prob, 0.0, 1.0))
        start_min = max(0, int(cfg.synthetic_human_start_min_step))
        start_max = max(start_min, int(cfg.synthetic_human_start_max_step))
        if start_max > start_min:
            self._synthetic_human_start_step = int(
                self.rng.integers(start_min, start_max + 1)
            )
        else:
            self._synthetic_human_start_step = start_min
        self._synthetic_human_duration_steps = max(
            1, int(cfg.synthetic_human_duration_steps)
        )
        self._synthetic_human_side = -1.0 if float(self.rng.random()) < 0.5 else 1.0
        self._synthetic_human_height_offset = float(self.rng.uniform(-0.025, 0.055))

    def _synthetic_human_state(self, gripper_center: np.ndarray) -> dict[str, Any]:
        if not self._synthetic_human_active:
            return {}
        if gripper_center is None:
            return {}
        gripper_center = np.asarray(gripper_center, dtype=float).reshape(-1)
        if gripper_center.size < 3 or not np.all(np.isfinite(gripper_center[:3])):
            return {}
        gripper_center = gripper_center[:3]
        local_step = self.step_count - self._synthetic_human_start_step
        if local_step < 0 or local_step > self._synthetic_human_duration_steps:
            return {}

        progress = float(local_step / max(1, self._synthetic_human_duration_steps))
        cfg = self.config
        near_dist = max(float(cfg.synthetic_human_near_dist), 1e-3)
        collision_dist = max(float(cfg.synthetic_human_collision_dist), 1e-3)
        min_dist = max(collision_dist * 0.5, 0.015)

        # Sweep the hand across the gripper. The midpoint is closest, so some
        # episodes produce only proximity feedback while others produce collision
        # feedback depending on the randomized height offset.
        lateral = self._synthetic_human_side * np.interp(
            progress, [0.0, 1.0], [near_dist * 1.8, -near_dist * 1.8]
        )
        closest = min_dist + abs(self._synthetic_human_height_offset) * 0.35
        vertical = self._synthetic_human_height_offset
        forward = closest * np.sin(np.pi * progress)
        right_hand = gripper_center + np.array(
            [lateral, forward, vertical], dtype=float
        )
        left_hand = right_hand + np.array(
            [0.22 * self._synthetic_human_side, -0.18, 0.02], dtype=float
        )
        head = right_hand + np.array([0.0, -0.55, 0.55], dtype=float)

        dist = float(np.linalg.norm(right_hand - gripper_center))
        return {
            "human_head_pos": head,
            "human_left_hand_pos": left_hand,
            "human_right_hand_pos": right_hand,
            "min_hand_gripper_dist_override": dist,
        }

    def _format_obs(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.config.observation_mode == "dict":
            return obs
        return flatten_observation(obs)

    def _target_from_action(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        ee_pos = np.asarray(self._last_obs["ee_pos"], dtype=float)
        if self.config.action_version == CONTROLLER_TARGET_ACTION_VERSION:
            target_pos = controller_target_from_action(
                ee_pos,
                action,
                action_scale=self.config.action_scale,
            )
        else:
            target_pos = (
                ee_pos
                + np.asarray(action[:3], dtype=float)
                * MAX_EE_DELTA_M
                * self.config.action_scale
            )
        current_yaw = self.yaw if np.isfinite(self.yaw) else 0.0
        next_yaw = float(
            current_yaw
            + float(action[3]) * MAX_YAW_DELTA_RAD * self.config.action_scale
        )
        if not np.isfinite(next_yaw):
            next_yaw = 0.0
        target_pos = np.array(
            [
                np.clip(target_pos[0], 0.20, 0.75),
                np.clip(target_pos[1], -0.35, 0.35),
                np.clip(
                    target_pos[2], self.table_top_z + 0.035, self.table_top_z + 0.50
                ),
            ],
            dtype=float,
        )
        target_quat = None
        if self.config.fixed_orientation:
            target_quat = _safe_quat(
                self._euler_angles_to_quat(np.array([0.0, np.pi, next_yaw]))
            )
        return target_pos, target_quat, next_yaw

    def _fixed_target_orientation(self, yaw: float) -> np.ndarray | None:
        if not self.config.fixed_orientation:
            return None
        return _safe_quat(
            self._euler_angles_to_quat(
                np.array([0.0, np.pi, float(yaw)], dtype=float)
            )
        )

    def _state_aware_recovery_enabled(self) -> bool:
        return bool(
            getattr(getattr(self, "config", None), "state_aware_recovery", False)
        )

    def _state_aware_recovery_evidence(
        self, obs: Mapping[str, np.ndarray]
    ) -> StateAwareRecoveryEvidence:
        intervention = max(
            0.0,
            float(
                self._last_physical_safety_diagnostics.intervention_norm_radps
            ),
        )
        return StateAwareRecoveryEvidence(
            ee_position_m=np.asarray(obs["ee_pos"], dtype=float),
            cube_position_m=np.asarray(obs["cube_pos"], dtype=float),
            place_target_position_m=np.asarray(
                obs["place_target_pos"], dtype=float
            ),
            cube_speed_mps=float(
                np.linalg.norm(np.asarray(obs["cube_lin_vel"], dtype=float))
            ),
            joint_speed_radps=float(
                np.linalg.norm(np.asarray(obs["robot_joint_vel"], dtype=float))
            ),
            has_grasped_cube=bool(
                float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]) > 0.5
            ),
            cbf_clear=bool(
                not self._strict_task_controller.state.cbf_hold_latched
                and intervention
                <= float(
                    self._strict_task_semantics_config.cbf_pause_exit_norm_radps
                )
            ),
        )

    def _prepare_state_aware_recovery_control(
        self, obs: Mapping[str, np.ndarray]
    ) -> StateAwareRecoveryDecision:
        if not self._state_aware_recovery_enabled():
            return self._state_aware_recovery.update(
                self._state_aware_recovery_evidence(obs)
            )
        if self._strict_task_controller.state.success_latched:
            self._state_aware_recovery.cancel_for_success()
        return self._state_aware_recovery.update(
            self._state_aware_recovery_evidence(obs)
        )

    def _recovery_gripper_command(
        self, desired_closed: bool | None
    ) -> str | None:
        if desired_closed is None:
            return None
        desired = bool(desired_closed)
        if desired == bool(self.gripper_closed):
            return None
        self.gripper_closed = desired
        return "close" if desired else "open"

    def _reset_strict_task_semantics(
        self, obs: Mapping[str, np.ndarray]
    ) -> None:
        cube_pos = np.asarray(obs["cube_pos"], dtype=float).reshape(-1)
        if cube_pos.size < 3 or not np.all(np.isfinite(cube_pos[:3])):
            raise RuntimeError(
                "Strict task semantics requires a finite cube position"
            )
        self._strict_task_controller.reset(
            initial_cube_z_m=float(cube_pos[2])
        )
        self._last_strict_task_decision = None
        self._last_gripper_command = None
        self._state_aware_recovery.reset()
        self._last_recovery_decision = None
        self._pending_recovery_handoff = None
        self._recovery_control_applied = False
        self._recovery_handoff_accepted = False
        self._last_recovery_alignment_token = None

    def _strict_task_semantics_enabled(self) -> bool:
        return bool(
            getattr(getattr(self, "config", None), "strict_task_semantics", False)
        )

    def _strict_task_evidence(
        self, obs: Mapping[str, np.ndarray]
    ) -> TaskPhysicalEvidence:
        cube_to_target = np.asarray(
            obs["cube_to_place_target"], dtype=float
        ).reshape(-1)
        cube_pos = np.asarray(obs["cube_pos"], dtype=float).reshape(-1)
        cube_velocity = np.asarray(
            obs["cube_lin_vel"], dtype=float
        ).reshape(-1)
        if (
            cube_to_target.size < 3
            or cube_pos.size < 3
            or cube_velocity.size < 3
        ):
            raise RuntimeError(
                "Strict task semantics requires three-dimensional cube state"
            )
        return TaskPhysicalEvidence(
            ee_cube_distance_m=float(
                np.linalg.norm(np.asarray(obs["ee_to_cube"], dtype=float))
            ),
            cube_target_xy_error_m=float(
                np.linalg.norm(cube_to_target[:2])
            ),
            cube_target_z_error_m=float(abs(cube_to_target[2])),
            cube_speed_mps=float(np.linalg.norm(cube_velocity[:3])),
            cube_z_m=max(0.0, float(cube_pos[2])),
            grasp_candidate=bool(
                float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0])
                > 0.5
            ),
            cbf_intervention_norm_radps=max(
                0.0,
                float(
                    self._last_physical_safety_diagnostics.intervention_norm_radps
                ),
            ),
        )

    def _cbf_task_progress_context(
        self,
        *,
        target_pos: np.ndarray,
        recovery_decision: StateAwareRecoveryDecision,
    ) -> dict[str, Any]:
        """Expose only frozen task/recovery semantics needed by B v2."""

        obs = self._last_obs
        if obs is None:
            raise RuntimeError("task-progress context requires a current observation")
        evidence = self._strict_task_evidence(obs)
        strict_state = self._strict_task_controller.state
        place_spatially_ready = bool(
            evidence.cube_target_xy_error_m
            <= float(self._strict_task_semantics_config.place_xy_tolerance_m)
            and evidence.cube_target_z_error_m
            <= float(self._strict_task_semantics_config.place_z_tolerance_m)
            and strict_state.maximum_cube_lift_m
            >= float(self._strict_task_semantics_config.minimum_cube_lift_m)
            and strict_state.grasp_observed
            and evidence.grasp_candidate
        )
        return {
            "schema_version": "phase_progress_runtime_context_v1",
            "controller_event": int(self.phase_event),
            "controller_t": float(self.phase_t),
            "target_position_world_m": np.asarray(
                target_pos, dtype=float
            ).reshape(3),
            "recovery_control_active": bool(self._recovery_control_applied),
            "recovery_mode": str(recovery_decision.mode),
            "recovery_stage": str(recovery_decision.stage),
            "grasp_candidate": bool(evidence.grasp_candidate),
            "strict_grasp_observed": bool(strict_state.grasp_observed),
            "place_spatially_ready": place_spatially_ready,
        }

    def _gripper_command(
        self, action: np.ndarray, obs: dict[str, np.ndarray]
    ) -> str | None:
        if self.config.gripper_mode == "event":
            if self._strict_task_semantics_enabled():
                if self._strict_task_controller.consume_pending_retry_open():
                    self.gripper_closed = False
                    return "open"
                if self.phase_event == 3:
                    self.gripper_closed = True
                    return "close"
                if (
                    self.phase_event == 7
                    and self._strict_task_controller.release_command_allowed()
                ):
                    self.gripper_closed = False
                    return "open"
                return None
            if (
                self.config.early_close_on_grasp_gate
                and self.phase_event in (1, 2)
                and not self.gripper_closed
                and _rule_gripper_should_close(
                    obs,
                    self.gripper_closed,
                    close_dist=self.config.phase_gate_close_dist,
                    release_dist=self.config.release_dist,
                )
            ):
                if self.config.fast_forward_grasp_gate:
                    self.phase_event = 3
                    self.phase_t = 0.0
                    self.phase_hold_steps = 0
                self.gripper_closed = True
                return "close"
            self.gripper_closed = event_gripper_command(
                self.phase_event, self.gripper_closed
            )
            if self.phase_event == 3:
                return "close"
            if self.phase_event == 7:
                return "open"
            return None

        previous_closed = self.gripper_closed
        if self.config.gripper_mode == "rule":
            self.gripper_closed = _rule_gripper_should_close(
                obs,
                self.gripper_closed,
                close_dist=self.config.close_dist,
                release_dist=self.config.release_dist,
            )
        elif self.config.gripper_mode == "policy":
            requested_closed = _policy_gripper_should_close(
                action, self.gripper_closed
            )
            # A one-step policy spike must not release a physically grasped
            # object before the strict controller has confirmed placement.
            # The policy still owns the eventual event-7 open command.
            if (
                self._strict_task_semantics_enabled()
                and previous_closed
                and not requested_closed
                and float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]) > 0.5
                and (
                    int(self.phase_event) != 7
                    or not self._strict_task_controller.release_command_allowed()
                )
            ):
                requested_closed = True
            self.gripper_closed = requested_closed
        else:
            raise ValueError(f"Unknown gripper_mode: {self.config.gripper_mode}")

        if self.gripper_closed and not previous_closed:
            return "close"
        if not self.gripper_closed and previous_closed:
            return "open"
        return None

    def _merge_gripper_action(self, arm_action, gripper_command: str | None):
        if gripper_command is None:
            return arm_action
        return self.robot.gripper.forward(action=gripper_command)

    def _guard_strict_release_for_current_cbf(
        self, gripper_command: str | None
    ) -> str | None:
        """Prevent an event-7 release on the step that first activates CBF.

        The strict phase controller observes the current physical-safety
        diagnostic after physics advances.  Without this apply-time guard, an
        ``open`` command selected from the previous observation can therefore
        escape one step before the CBF hold is latched.
        """

        if (
            not self._strict_task_semantics_enabled()
            or int(self.phase_event) != 7
            or gripper_command != "open"
        ):
            return gripper_command
        intervention = max(
            0.0,
            float(
                self._last_physical_safety_diagnostics.intervention_norm_radps
            ),
        )
        if intervention < float(
            self._strict_task_semantics_config.cbf_pause_enter_norm_radps
        ):
            return gripper_command
        has_grasped = bool(
            self._last_obs is not None
            and float(
                np.asarray(self._last_obs["has_grasped_cube"]).reshape(-1)[0]
            )
            > 0.5
        )
        if not has_grasped:
            return gripper_command
        self.gripper_closed = True
        return None

    def _advance_phase(self, obs: dict[str, np.ndarray]) -> None:
        self._advance_phase_with_events(obs)

    def _advance_phase_with_events(
        self,
        obs: dict[str, np.ndarray],
        events_dt: tuple[float, ...] | None = None,
        *,
        freeze_for_recovery: bool = False,
    ) -> None:
        if freeze_for_recovery:
            next_event = int(self.phase_event)
            next_t = float(self.phase_t)
            terminal_event = 10 if events_dt is None else len(events_dt)
        elif events_dt is None:
            next_event, next_t = advance_pick_place_event(
                self.phase_event, self.phase_t
            )
            terminal_event = 10
        else:
            next_event, next_t = advance_pick_place_event(
                self.phase_event, self.phase_t, events_dt
            )
            terminal_event = len(events_dt)
        if self._strict_task_semantics_enabled():
            decision = self._strict_task_controller.update(
                event=int(self.phase_event),
                progress=float(self.phase_t),
                proposed_event=int(next_event),
                proposed_progress=float(next_t),
                terminal_event=int(terminal_event),
                evidence=self._strict_task_evidence(obs),
            )
            previous_event = int(self.phase_event)
            self.phase_event = int(decision.event)
            self.phase_t = float(decision.progress)
            if decision.held:
                self.phase_hold_steps += 1
            elif int(self.phase_event) != previous_event:
                self.phase_hold_steps = 0
            self._last_strict_task_decision = decision
            return
        ee_cube_dist = float(np.linalg.norm(obs["ee_to_cube"]))
        cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
        hold_lowering_for_grasp = (
            self.config.gripper_mode == "event"
            and self.phase_event in (1, 2)
            and next_event != self.phase_event
            and ee_cube_dist > self.config.phase_gate_close_dist
            and float(obs["has_grasped_cube"][0]) <= 0.5
            and self.phase_hold_steps < self.config.phase_gate_max_hold
        )
        hold_release_for_target = (
            self.config.gripper_mode == "event"
            and self.config.release_gate_dist is not None
            and self.phase_event == 6
            and next_event != self.phase_event
            and cube_target_dist > float(self.config.release_gate_dist)
            and self.phase_hold_steps < self.config.release_gate_max_hold
        )
        if hold_lowering_for_grasp or hold_release_for_target:
            self.phase_hold_steps += 1
            return
        if next_event != self.phase_event:
            self.phase_hold_steps = 0
        self.phase_event, self.phase_t = next_event, next_t

    def _control_task_phase(
        self,
        obs: dict[str, np.ndarray],
        *,
        advance: bool,
        reset_for_reentry: bool,
    ) -> dict[str, Any]:
        event_before = int(self.phase_event)
        t_before = float(self.phase_t)
        reason = "paused"
        recovery_control = False
        handoff_reason = ""
        if reset_for_reentry:
            reason = self._synchronize_task_phase_for_reentry(obs)
            self._write_phase_observation(obs)
        elif advance:
            recovery_control = bool(
                self._state_aware_recovery_enabled()
                and self._recovery_control_applied
            )
            if recovery_control:
                self._align_phase_for_active_recovery(obs)
                self._advance_phase_with_events(
                    obs,
                    freeze_for_recovery=True,
                )
            else:
                self._advance_phase(obs)
            strict_decision = getattr(self, "_last_strict_task_decision", None)
            recovery_requested = self._request_state_aware_recovery(
                obs, strict_decision
            )
            handoff_reason = self._finish_state_aware_recovery_step(obs)
            if bool(
                self._strict_task_semantics_enabled()
                or
                getattr(
                    self.config,
                    "synchronize_advanced_phase_observation",
                    False,
                )
            ):
                self._write_phase_observation(obs)
            if handoff_reason:
                reason = handoff_reason
            elif recovery_requested:
                reason = str(strict_decision.reason)
            elif recovery_control:
                reason = (
                    str(strict_decision.reason)
                    if strict_decision is not None
                    and strict_decision.reason == "cbf_intervention_pause"
                    else f"state_aware_recovery_{self._state_aware_recovery.state.stage}"
                )
            else:
                reason = (
                    self._last_strict_task_decision.reason
                    if self._strict_task_semantics_enabled()
                    and getattr(self, "_last_strict_task_decision", None) is not None
                    else "advanced"
                )
        strict_decision = getattr(self, "_last_strict_task_decision", None)
        internally_held = bool(
            self._strict_task_semantics_enabled()
            and advance
            and strict_decision is not None
            and strict_decision.held
        )
        internal_reentry = bool(
            self._strict_task_semantics_enabled()
            and advance
            and strict_decision is not None
            and strict_decision.reentry
        )
        recovery_paused = bool(
            advance
            and self._state_aware_recovery_enabled()
            and (
                recovery_control
                or self._state_aware_recovery.state.active
            )
        )
        return {
            "advanced": bool(advance),
            "paused": bool(not advance or internally_held or recovery_paused),
            "reentry": bool(
                reset_for_reentry
                or internal_reentry
                or bool(self._recovery_handoff_accepted if advance else False)
            ),
            "reason": reason,
            "event_before": event_before,
            "event_after": int(self.phase_event),
            "controller_t_before": t_before,
            "controller_t_after": float(self.phase_t),
            "phase_changed": bool(
                int(self.phase_event) != event_before
                or not np.isclose(float(self.phase_t), t_before)
            ),
        }

    def _align_phase_for_active_recovery(
        self, obs: Mapping[str, np.ndarray]
    ) -> None:
        decision = self._last_recovery_decision
        if decision is None or decision.mode not in ("place", "regrasp"):
            return
        target_event = 6 if decision.mode == "place" else 0
        alignment_token = (
            int(self._state_aware_recovery.state.activation_count),
            int(self._state_aware_recovery.state.replan_count),
            str(decision.mode),
        )
        if alignment_token != self._last_recovery_alignment_token:
            self._strict_task_controller.begin_recovery(
                mode=decision.mode,
                evidence=self._strict_task_evidence(obs),
            )
            self._last_recovery_alignment_token = alignment_token
        if int(self.phase_event) == target_event:
            return
        self.phase_event = target_event
        self.phase_t = 0.0
        self.phase_hold_steps = 0

    def _request_state_aware_recovery(
        self,
        obs: Mapping[str, np.ndarray],
        decision: StrictTaskPhaseDecision | None,
    ) -> bool:
        if (
            not self._state_aware_recovery_enabled()
            or decision is None
            or decision.recovery_request not in ("place", "regrasp")
        ):
            return False
        self._state_aware_recovery.request(
            decision.recovery_request,
            self._state_aware_recovery_evidence(obs),
            source_event=int(self.phase_event),
            source_progress=float(self.phase_t),
        )
        return True

    def _finish_state_aware_recovery_step(
        self, obs: Mapping[str, np.ndarray]
    ) -> str:
        if not self._state_aware_recovery_enabled():
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return ""
        recovery_decision = self._last_recovery_decision
        if recovery_decision is not None and recovery_decision.timed_out:
            self._strict_task_controller.latch_failure(
                "state_aware_recovery_timeout"
            )
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return "state_aware_recovery_timeout"

        pending = self._pending_recovery_handoff
        if pending is None:
            self._recovery_control_applied = False
            return ""
        mode = "place" if pending.handoff_event == 6 else "regrasp"
        evidence = self._state_aware_recovery_evidence(obs)
        intervention = max(
            0.0,
            float(
                self._last_physical_safety_diagnostics.intervention_norm_radps
            ),
        )
        renewed_cbf = bool(
            self._strict_task_controller.state.cbf_hold_latched
            or intervention
            >= float(
                self._strict_task_semantics_config.cbf_pause_enter_norm_radps
            )
        )
        live_state = self._state_aware_recovery.state
        stale_handoff = bool(
            not live_state.active
            or int(live_state.plan_id) != int(pending.plan_id)
            or str(live_state.mode) != mode
        )
        if stale_handoff:
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return "state_aware_recovery_stale_handoff_discarded"

        if renewed_cbf:
            self._state_aware_recovery.request(
                mode,
                evidence,
                source_event=int(self.phase_event),
                source_progress=float(self.phase_t),
            )
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return "state_aware_recovery_handoff_deferred_by_cbf"

        mode_consistent = bool(
            (mode == "place" and evidence.has_grasped_cube)
            or (mode == "regrasp" and not evidence.has_grasped_cube)
        )
        if not mode_consistent:
            self._state_aware_recovery.defer_handoff(
                "state_aware_recovery_handoff_mode_mismatch"
            )
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return "state_aware_recovery_handoff_mode_mismatch"

        post_anchor_ready, _ = self._state_aware_recovery.anchor_is_ready(
            evidence,
            np.asarray(pending.target_position_m, dtype=float).reshape(3),
        )
        if not post_anchor_ready:
            self._state_aware_recovery.defer_handoff(
                "state_aware_recovery_handoff_postcheck_failed"
            )
            self._pending_recovery_handoff = None
            self._recovery_control_applied = False
            return "state_aware_recovery_handoff_postcheck_failed"

        event_before = int(self.phase_event)
        progress_before = float(self.phase_t)
        handoff = self._strict_task_controller.recovery_handoff(
            event=int(pending.handoff_event),
            progress=float(pending.handoff_progress),
            mode=mode,
            evidence=self._strict_task_evidence(obs),
            event_before=event_before,
            progress_before=progress_before,
        )
        self.phase_event = int(handoff.event)
        self.phase_t = float(handoff.progress)
        self.phase_hold_steps = 0
        self._last_strict_task_decision = handoff
        self._state_aware_recovery.accept_handoff()
        self._recovery_handoff_accepted = True
        self._pending_recovery_handoff = None
        self._recovery_control_applied = False
        return str(handoff.reason)

    def _synchronize_task_phase_for_reentry(
        self,
        obs: dict[str, np.ndarray],
    ) -> str:
        has_grasped = bool(
            float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]) > 0.5
        )
        cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
        reason = "resume_paused_event"
        if self._state_aware_recovery_enabled():
            reentry_class = self._strict_task_controller.classify_external_reentry(
                event=int(self.phase_event),
                evidence=self._strict_task_evidence(obs),
            )
            if reentry_class == "released":
                return reason
            if reentry_class == "wait":
                self.phase_hold_steps += 1
                return "hold_for_external_reentry_grasp_loss_confirmation"
            recovery_mode = str(reentry_class)
            if recovery_mode not in ("place", "regrasp"):
                raise RuntimeError(
                    f"Unsupported external reentry class: {recovery_mode!r}"
                )
            self._state_aware_recovery.request(
                recovery_mode,
                self._state_aware_recovery_evidence(obs),
                source_event=int(self.phase_event),
                source_progress=float(self.phase_t),
            )
            return f"state_aware_{recovery_mode}_reentry_requested"
        if (
            self.phase_event >= 4
            and not has_grasped
            and cube_target_dist > float(self.config.success_dist)
        ):
            self.phase_event = 0
            self.phase_t = 0.0
            self.phase_hold_steps = 0
            reason = "rewind_missing_grasp"
        elif self.phase_event >= 7 and has_grasped:
            self.phase_event = 5
            self.phase_t = 0.0
            self.phase_hold_steps = 0
            reason = "reposition_still_grasped"
        elif (
            self.phase_event == 6
            and has_grasped
            and cube_target_dist
            > max(float(self.config.success_dist), float(self.config.release_dist))
        ):
            self.phase_event = 5
            self.phase_t = 0.0
            self.phase_hold_steps = 0
            reason = "reposition_before_release"
        return reason

    def _write_phase_observation(self, obs: dict[str, np.ndarray]) -> None:
        obs["task_phase"] = task_phase_onehot(task_phase_from_event(self.phase_event))
        obs["controller_event"] = controller_event_onehot(self.phase_event)
        obs["controller_t"] = np.array(
            [np.clip(float(self.phase_t), 0.0, 1.0)], dtype=np.float32
        )

    def _pseudo_errp_result(
        self,
        obs: dict[str, np.ndarray],
        *,
        override_feedback: float | None = None,
    ) -> PseudoErrPResult:
        return pseudo_errp_from_observation(
            obs,
            aux_flags=self._pseudo_errp_aux_flags,
            enabled=self.config.pseudo_errp_enabled,
            sources=self.config.pseudo_errp_sources,
            override_feedback=override_feedback,
        )

    def _is_success(self, obs: dict[str, np.ndarray]) -> bool:
        if self._strict_task_semantics_enabled():
            return bool(self._strict_task_controller.state.success_latched)
        if not is_success(obs, threshold_m=self.config.success_dist):
            return False
        if not self.config.require_release_for_success:
            return True
        has_grasped = bool(
            float(np.asarray(obs["has_grasped_cube"]).reshape(-1)[0]) > 0.5
        )
        return self.phase_event >= 7 and not has_grasped

    def _info(
        self,
        obs: dict[str, np.ndarray],
        *,
        reward_components: dict[str, float],
        errp_result: PseudoErrPResult,
    ) -> dict[str, Any]:
        physical_safety = self._last_physical_safety_diagnostics.as_dict()
        dynamic_safety = (
            {}
            if self._last_dynamic_safety_sample is None
            else {
                **self._last_dynamic_safety_sample.human_payload(),
                **self._last_dynamic_safety_sample.safety_payload(),
            }
        )
        environment_safety = self._last_environment_safety_result
        static_result = (
            None if environment_safety is None else environment_safety.static
        )
        self_result = (
            None if environment_safety is None else environment_safety.self_collision
        )
        strict_state = self._strict_task_controller.state.as_dict()
        strict_decision = (
            None
            if self._last_strict_task_decision is None
            else {
                "event": int(self._last_strict_task_decision.event),
                "progress": float(self._last_strict_task_decision.progress),
                "reason": str(self._last_strict_task_decision.reason),
                "phase_changed": bool(
                    self._last_strict_task_decision.phase_changed
                ),
                "held": bool(self._last_strict_task_decision.held),
                "retry_started": bool(
                    self._last_strict_task_decision.retry_started
                ),
                "reentry": bool(self._last_strict_task_decision.reentry),
                "success_latched": bool(
                    self._last_strict_task_decision.success_latched
                ),
                "failure_reason": str(
                    self._last_strict_task_decision.failure_reason
                ),
                "recovery_request": str(
                    self._last_strict_task_decision.recovery_request
                ),
            }
        )
        recovery_decision = self._last_recovery_decision
        recovery_payload = {
            "schema_version": STATE_AWARE_RECOVERY_SCHEMA,
            "enabled": self._state_aware_recovery_enabled(),
            "config": self._state_aware_recovery_config.as_dict(),
            "state": self._state_aware_recovery.state.as_dict(),
            "control_authority": bool(
                recovery_decision is not None
                and (
                    recovery_decision.target_position_m is not None
                    or recovery_decision.timed_out
                )
            ),
            "handoff_attempted": bool(
                recovery_decision is not None and recovery_decision.handoff
            ),
            "handoff_accepted": bool(self._recovery_handoff_accepted),
            "last_decision": (
                None
                if recovery_decision is None
                else {
                    "active": bool(recovery_decision.active),
                    "plan_id": int(recovery_decision.plan_id),
                    "mode": str(recovery_decision.mode),
                    "stage": str(recovery_decision.stage),
                    "target_position_m": (
                        None
                        if recovery_decision.target_position_m is None
                        else np.asarray(
                            recovery_decision.target_position_m, dtype=float
                        ).reshape(3).tolist()
                    ),
                    "desired_gripper_closed": (
                        None
                        if recovery_decision.desired_gripper_closed is None
                        else bool(recovery_decision.desired_gripper_closed)
                    ),
                    "anchor_ready": bool(recovery_decision.anchor_ready),
                    "ready_streak": int(recovery_decision.ready_streak),
                    "anchor_error_m": (
                        None
                        if recovery_decision.anchor_error_m is None
                        else float(recovery_decision.anchor_error_m)
                    ),
                    "handoff": bool(recovery_decision.handoff),
                    "handoff_event": recovery_decision.handoff_event,
                    "handoff_progress": recovery_decision.handoff_progress,
                    "stage_changed": bool(recovery_decision.stage_changed),
                    "timed_out": bool(recovery_decision.timed_out),
                    "reason": str(recovery_decision.reason),
                }
            ),
        }
        task_terminal_reason = ""
        if bool(strict_state["success_latched"]):
            task_terminal_reason = "success"
        elif str(strict_state["failure_reason"]):
            task_terminal_reason = str(strict_state["failure_reason"])
        elif self.step_count >= int(self.config.max_episode_steps):
            task_terminal_reason = "max_episode_steps"
        return {
            "reward_version": str(self.config.reward_version),
            "episode_index": self.current_episode_index,
            "step": self.step_count,
            "sim_time": float(getattr(self.world, "current_time", 0.0)),
            "active_cube": getattr(self.active_cube, "name", ""),
            "controller_event": int(self.phase_event),
            "controller_t": float(self.phase_t),
            "phase_hold_steps": int(self.phase_hold_steps),
            "gripper_closed": bool(self.gripper_closed),
            "gripper_command": self._last_gripper_command,
            "success": self._is_success(obs),
            "cube_target_dist": float(np.linalg.norm(obs["cube_to_place_target"])),
            "ee_cube_dist": float(np.linalg.norm(obs["ee_to_cube"])),
            "has_grasped_cube": bool(float(obs["has_grasped_cube"][0]) > 0.5),
            "grasp_candidate": bool(float(obs["has_grasped_cube"][0]) > 0.5),
            "task_terminal_reason": task_terminal_reason,
            "strict_task_semantics": {
                "schema_version": STRICT_TASK_SEMANTICS_SCHEMA,
                "enabled": self._strict_task_semantics_enabled(),
                "config": self._strict_task_semantics_config.as_dict(),
                "state": strict_state,
                "last_decision": strict_decision,
            },
            "state_aware_recovery": recovery_payload,
            "errp_feedback": float(errp_result.feedback),
            "errp_uncertainty": float(errp_result.uncertainty),
            "errp_label": int(errp_result.label),
            "errp_source_code": int(errp_result.source_code),
            "errp_source_names": tuple(errp_result.source_names),
            "pseudo_errp_flags": dict(errp_result.flags),
            "pseudo_errp_source_scores": dict(errp_result.source_scores),
            "source_restoration": dict(self._source_restoration_diagnostics),
            "human_replay_aux_state": dict(self._human_replay_aux_state),
            # Action provenance comes from the pre-action CBF diagnostics.
            # The aux state already belongs to the next policy observation,
            # because _build_obs() runs before _info() on environment steps.
            "intentional_human_absence": _intentional_human_absence_from_aux(
                {
                    "intentional_human_absence": getattr(
                        self._last_physical_safety_diagnostics,
                        "intentional_human_absence",
                        False,
                    )
                }
            ),
            "next_observation_intentional_human_absence": (
                _intentional_human_absence_from_aux(
                    self._human_replay_aux_state
                )
            ),
            "dynamic_safety": dynamic_safety,
            "human_robot_collision": bool(float(obs["human_robot_collision"][0]) > 0.5),
            "near_human": bool(float(obs["near_human"][0]) > 0.5),
            "near_miss": bool(float(obs["near_miss"][0]) > 0.5),
            "min_hand_end_effector_surface_gap": float(
                obs["min_hand_end_effector_surface_gap"][0]
            ),
            "distance_gate": float(obs["distance_gate"][0]),
            "geometry_valid": bool(float(obs["geometry_valid"][0]) > 0.5),
            "static_surface_gap_m": (
                10.0 if static_result is None else float(static_result.surface_gap_m)
            ),
            "static_collision": (
                False if static_result is None else bool(static_result.collision)
            ),
            "static_gap_valid": (
                False if static_result is None else bool(static_result.geometry_valid)
            ),
            "static_collision_valid": (
                False if static_result is None else bool(static_result.collision_valid)
            ),
            "static_geometry_valid": (
                False
                if static_result is None
                else bool(
                    static_result.geometry_valid and static_result.collision_valid
                )
            ),
            "self_surface_gap_m": (
                10.0 if self_result is None else float(self_result.surface_gap_m)
            ),
            "self_collision": (
                False if self_result is None else bool(self_result.collision)
            ),
            "self_gap_valid": (
                False if self_result is None else bool(self_result.geometry_valid)
            ),
            "self_collision_valid": (
                False if self_result is None else bool(self_result.collision_valid)
            ),
            "self_geometry_valid": (
                False
                if self_result is None
                else bool(self_result.geometry_valid and self_result.collision_valid)
            ),
            "combined_safety_collision": bool(
                bool(float(obs["human_robot_collision"][0]) > 0.5)
                or (static_result is not None and static_result.collision)
                or (self_result is not None and self_result.collision)
            ),
            "static_closest_robot_collider": (
                "" if static_result is None else static_result.first_path
            ),
            "static_closest_environment_collider": (
                "" if static_result is None else static_result.second_path
            ),
            "self_closest_first_collider": (
                "" if self_result is None else self_result.first_path
            ),
            "self_closest_second_collider": (
                "" if self_result is None else self_result.second_path
            ),
            "environment_safety_query_time_ms": (
                0.0
                if environment_safety is None
                else float(environment_safety.query_time_ms)
            ),
            "left_end_effector_surface_gap_m": (
                self._last_safety_result.left.surface_gap_m
                if self._last_safety_result is not None
                else 10.0
            ),
            "right_end_effector_surface_gap_m": (
                self._last_safety_result.right.surface_gap_m
                if self._last_safety_result is not None
                else 10.0
            ),
            "contact_left": (
                self._last_safety_result.left.contact
                if self._last_safety_result is not None
                else False
            ),
            "contact_right": (
                self._last_safety_result.right.contact
                if self._last_safety_result is not None
                else False
            ),
            "penetration_left_m": (
                self._last_safety_result.left.penetration_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "penetration_right_m": (
                self._last_safety_result.right.penetration_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "distance_gate_left": (
                self._last_safety_result.left.distance_gate
                if self._last_safety_result is not None
                else 0.0
            ),
            "distance_gate_right": (
                self._last_safety_result.right.distance_gate
                if self._last_safety_result is not None
                else 0.0
            ),
            "closest_link_left": (
                self._last_safety_result.left.closest_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_link_right": (
                self._last_safety_result.right.closest_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider_left": (
                self._last_safety_result.left.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider_right": (
                self._last_safety_result.right.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "closest_human_hand": (
                self._last_safety_result.closest_human_hand
                if self._last_safety_result is not None
                else ""
            ),
            "closest_robot_link": (
                self._last_safety_result.closest_robot_link
                if self._last_safety_result is not None
                else ""
            ),
            "closest_collider": (
                self._last_safety_result.closest_collider_path
                if self._last_safety_result is not None
                else ""
            ),
            "contact_active": (
                self._last_safety_result.contact
                if self._last_safety_result is not None
                else False
            ),
            "penetration_depth_m": (
                self._last_safety_result.penetration_depth_m
                if self._last_safety_result is not None
                else 0.0
            ),
            "physical_safety_controller": self._physical_safety_mode,
            "physical_safety": physical_safety,
            "physical_command_provenance": dict(
                self._last_command_provenance
            ),
            "physical_safety_active": bool(physical_safety["active"]),
            "physical_safety_intervention_available": bool(
                physical_safety["intervention_available"]
            ),
            "physical_safety_constraint_count": int(
                physical_safety["constraint_count"]
            ),
            "physical_safety_intervention_norm_radps": float(
                physical_safety["intervention_norm_radps"]
            ),
            "physical_safety_slack_radps": float(physical_safety["slack_radps"]),
            "physical_safety_slack_mps": float(physical_safety["slack_mps"]),
            "physical_safety_max_constraint_violation_before_mps": float(
                physical_safety["max_constraint_violation_before_mps"]
            ),
            "physical_safety_max_constraint_violation_after_mps": float(
                physical_safety["max_constraint_violation_after_mps"]
            ),
            "physical_safety_feasible": bool(physical_safety["feasible"]),
            "physical_safety_fallback_applied": bool(
                physical_safety["fallback_applied"]
            ),
            "physical_safety_failure_reasons": tuple(
                physical_safety["failure_reasons"]
            ),
            "physical_safety_status": str(physical_safety["status"]),
            "physical_safety_solve_time_ms": float(physical_safety["solve_time_ms"]),
            "rmpflow_valid_hand_obstacles": int(self._rmpflow_valid_hand_count),
            "safety_query_time_ms": (
                self._last_safety_result.left.query_time_ms
                + self._last_safety_result.right.query_time_ms
                if self._last_safety_result is not None
                else 0.0
            ),
            "reward_components": dict(reward_components),
            "obs_dict": obs,
        }


def _empty_source_restoration_diagnostics() -> dict[str, Any]:
    return {
        "source_configuration_available": True,
        "restoration_mode": "not_requested",
        "restoration_reason": "",
        "source_cube_index": None,
        "screening_cube_index": None,
        "source_cube_name": None,
        "screening_cube_name": None,
        "collection_seed": None,
        "layout_seed": None,
        "screening_seed": None,
        "cube_pose_restored": False,
        "target_pose_restored": False,
        "robot_initial_state_restored": False,
        "pose_mismatch": False,
        "pose_mismatch_reason": "",
        "max_cube_position_error_m": None,
        "max_cube_orientation_error_rad": None,
        "target_position_error_m": None,
        "target_orientation_error_rad": None,
        "missing_fields": [],
        "source_configuration": {},
    }


def _normalize_source_restoration(
    source_restoration: dict[str, Any] | None,
    screening_seed: int | None,
) -> dict[str, Any]:
    if source_restoration is None:
        result = _empty_source_restoration_diagnostics()
    else:
        result = {
            **_empty_source_restoration_diagnostics(),
            **dict(source_restoration),
        }
        source = result.get("source_configuration")
        result["source_configuration"] = (
            dict(source) if isinstance(source, dict) else {}
        )
    if result.get("screening_seed") is None and screening_seed is not None:
        result["screening_seed"] = int(screening_seed)
    return result


def _finite_joint_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size <= 0 or not np.all(np.isfinite(result)):
        return None
    return result


def _required_runtime_vector(
    value: Any,
    name: str,
    *,
    min_size: int = 1,
) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size < int(min_size) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Cannot capture finite {name}")
    return result.copy()


def _optional_applied_action_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _required_runtime_vector(value, "robot_applied_action")


def _capture_robot_applied_action_state(
    robot,
) -> dict[str, np.ndarray | None] | None:
    """Capture articulation drive targets in addition to measured q/qd.

    Isaac's ``set_joint_positions`` and ``set_joint_velocities`` restore the
    measured articulation state, but not the drive targets left by the last
    partial ``ArticulationAction``.  That distinction matters here because a
    gripper-only action intentionally leaves the arm targets unchanged, and an
    arm-only action leaves the finger targets unchanged.
    """

    getter = getattr(robot, "get_applied_action", None)
    if not callable(getter):
        return None
    action = getter()
    if action is None:
        return None
    return {
        "joint_positions": _optional_applied_action_vector(
            getattr(action, "joint_positions", None)
        ),
        "joint_velocities": _optional_applied_action_vector(
            getattr(action, "joint_velocities", None)
        ),
        "joint_efforts": _optional_applied_action_vector(
            getattr(action, "joint_efforts", None)
        ),
    }


def _restore_robot_applied_action_state(
    robot,
    state: dict[str, np.ndarray | None] | None,
) -> None:
    if state is None:
        return
    current_getter = getattr(robot, "get_applied_action", None)
    current_action = current_getter() if callable(current_getter) else None
    action_type = type(current_action) if current_action is not None else None
    if action_type is None:
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction

        action_type = ArticulationAction
    kwargs = {
        name: (
            None
            if state.get(name) is None
            else np.asarray(state[name], dtype=float).copy()
        )
        for name in ("joint_positions", "joint_velocities", "joint_efforts")
    }
    robot.apply_action(action_type(**kwargs))


def _optional_body_velocity(body, getter_name: str) -> np.ndarray:
    getter = getattr(body, getter_name, None)
    if not callable(getter):
        return np.zeros(3, dtype=float)
    try:
        return _required_runtime_vector(getter(), getter_name, min_size=3)[:3]
    except Exception:
        return np.zeros(3, dtype=float)


def _capture_body_state(body) -> dict[str, np.ndarray]:
    position, orientation = body.get_world_pose()
    return {
        "position": _required_runtime_vector(position, "body_position", min_size=3)[:3],
        "orientation": _required_runtime_vector(
            orientation, "body_orientation", min_size=4
        )[:4],
        "linear_velocity": _optional_body_velocity(body, "get_linear_velocity"),
        "angular_velocity": _optional_body_velocity(body, "get_angular_velocity"),
    }


def _restore_body_state(body, state: dict[str, np.ndarray]) -> None:
    body.set_world_pose(
        position=np.asarray(state["position"], dtype=float).copy(),
        orientation=np.asarray(state["orientation"], dtype=float).copy(),
    )
    if hasattr(body, "set_linear_velocity"):
        body.set_linear_velocity(
            np.asarray(state["linear_velocity"], dtype=float).copy()
        )
    if hasattr(body, "set_angular_velocity"):
        body.set_angular_velocity(
            np.asarray(state["angular_velocity"], dtype=float).copy()
        )


def _copy_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in obs.items()}


def _capture_safety_geometry_state(safety_geometry) -> dict[str, Any]:
    return {
        "previous_closest_collider": copy.deepcopy(
            getattr(safety_geometry, "_previous_closest_collider", {})
        ),
        "last_debug_state": copy.deepcopy(
            getattr(safety_geometry, "_last_debug_state", {})
        ),
    }


def _restore_safety_geometry_state(
    safety_geometry,
    state: dict[str, Any],
) -> None:
    if hasattr(safety_geometry, "_previous_closest_collider"):
        safety_geometry._previous_closest_collider = copy.deepcopy(
            state.get("previous_closest_collider", {})
        )
    if hasattr(safety_geometry, "_last_debug_state"):
        safety_geometry._last_debug_state = copy.deepcopy(
            state.get("last_debug_state", {})
        )


def _set_robot_default_state(robot, positions: Any, velocities: Any) -> bool:
    positions_array, velocities_array = _runtime_robot_joint_state(
        robot,
        positions,
        velocities,
    )
    if positions_array is None or not hasattr(robot, "set_joints_default_state"):
        return False
    try:
        robot.set_joints_default_state(
            positions=positions_array,
            velocities=velocities_array,
        )
    except TypeError:
        robot.set_joints_default_state(positions_array, velocities_array)
    return True


def _set_robot_joint_state(robot, positions: Any, velocities: Any) -> bool:
    positions_array, velocities_array = _runtime_robot_joint_state(
        robot,
        positions,
        velocities,
    )
    if positions_array is None or not hasattr(robot, "set_joint_positions"):
        return False
    robot.set_joint_positions(positions_array)
    if velocities_array is not None and hasattr(robot, "set_joint_velocities"):
        robot.set_joint_velocities(velocities_array)
    return True


def _canonicalize_exact_robot_reset(
    robot,
    positions: Any,
    velocities: Any,
) -> dict[str, Any]:
    """Canonicalize and verify measured state plus full q/qd drive targets.

    Isaac's measured articulation state and its most recently applied drive
    target are separate state.  In particular, an arm-only action leaves
    finger targets untouched and a gripper-only action leaves arm targets
    untouched.  Replaying a new exact-pose episode after an arbitrary terminal
    history therefore requires replacing *all* prior targets, not merely
    calling ``set_joint_positions``/``set_joint_velocities``.

    This helper deliberately performs exact (zero-tolerance) comparisons.  It
    runs after the exact-pose q/qd restoration and before the reset observation
    is built, so a reset cannot silently continue with history-dependent arm or
    finger targets.
    """

    source_positions = _finite_joint_vector(positions)
    source_velocities = _finite_joint_vector(velocities)
    if source_positions is None:
        raise RuntimeError("Cannot canonicalize an unavailable exact robot state")

    # Use the finite state read back *after* `_set_robot_joint_state` as the
    # canonical drive target.  Isaac articulation buffers may quantize the raw
    # JSON/HDF5 float representation; using that raw representation as the
    # equality target could reject an otherwise exact restored buffer solely
    # because of dtype conversion.  Source restoration remains independently
    # checked by the existing exact-pose restoration gate.
    expected_positions = _required_runtime_vector(
        robot.get_joint_positions(), "restored_measured_joint_positions"
    )
    expected_velocities = _required_runtime_vector(
        robot.get_joint_velocities(), "restored_measured_joint_velocities"
    )
    if expected_positions.shape != expected_velocities.shape:
        raise RuntimeError("Exact robot q/qd dimensions differ")
    if source_positions.size > expected_positions.size:
        raise RuntimeError("Source robot q exceeds runtime articulation dimensions")
    if source_velocities is not None and source_velocities.size > expected_velocities.size:
        raise RuntimeError("Source robot qd exceeds runtime articulation dimensions")

    _restore_robot_applied_action_state(
        robot,
        {
            "joint_positions": expected_positions,
            "joint_velocities": expected_velocities,
            # This environment uses position/velocity drives.  Supplying an
            # effort vector could change controller mode/semantics, so effort
            # is intentionally unset and excluded from the q/qd contract.
            "joint_efforts": None,
        },
    )

    measured_positions = _required_runtime_vector(
        robot.get_joint_positions(), "reset_measured_joint_positions"
    )
    measured_velocities = _required_runtime_vector(
        robot.get_joint_velocities(), "reset_measured_joint_velocities"
    )
    applied = _capture_robot_applied_action_state(robot)
    if applied is None:
        raise RuntimeError("Exact robot reset has no applied-action state")
    applied_positions = applied.get("joint_positions")
    applied_velocities = applied.get("joint_velocities")
    comparisons = {
        "measured_joint_positions": (
            measured_positions,
            expected_positions,
        ),
        "measured_joint_velocities": (
            measured_velocities,
            expected_velocities,
        ),
        "applied_joint_position_targets": (
            applied_positions,
            expected_positions,
        ),
        "applied_joint_velocity_targets": (
            applied_velocities,
            expected_velocities,
        ),
    }
    diagnostics: dict[str, Any] = {
        "contract": EXACT_POSE_ROBOT_RESET_CONTRACT,
        "joint_count": int(expected_positions.size),
        "source_joint_count": int(source_positions.size),
        "canonical_full_q_qd_targets_applied": True,
        "joint_effort_target_contract": "unset_not_effort_controlled",
    }
    mismatches: list[str] = []
    for name, (actual, expected) in comparisons.items():
        exact = bool(
            actual is not None
            and np.asarray(actual).shape == np.asarray(expected).shape
            and np.array_equal(np.asarray(actual), np.asarray(expected))
        )
        diagnostics[f"{name}_exact"] = exact
        if not exact:
            mismatches.append(name)
    diagnostics["passed"] = not mismatches
    diagnostics["mismatched_fields"] = mismatches
    if mismatches:
        raise RuntimeError(
            "Exact robot reset q/qd or applied target mismatch: "
            + ",".join(mismatches)
        )
    return diagnostics


def _runtime_robot_joint_state(
    robot,
    positions: Any,
    velocities: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    source_positions = _finite_joint_vector(positions)
    if source_positions is None:
        return None, None
    try:
        runtime_positions = np.asarray(
            robot.get_joint_positions(), dtype=float
        ).reshape(-1)
    except Exception:
        runtime_positions = np.empty(0, dtype=float)
    if runtime_positions.size > 0:
        if source_positions.size > runtime_positions.size:
            return None, None
        positions_array = runtime_positions.copy()
        positions_array[: source_positions.size] = source_positions
    else:
        positions_array = source_positions

    source_velocities = _finite_joint_vector(velocities)
    velocities_array = np.zeros_like(positions_array)
    if source_velocities is not None:
        count = min(source_velocities.size, velocities_array.size)
        velocities_array[:count] = source_velocities[:count]
    return positions_array, velocities_array


def _verify_exact_restoration(
    cubes,
    place_target,
    source: dict[str, Any],
    *,
    robot_restored: bool,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-4,
) -> dict[str, Any]:
    cube_position_error, cube_orientation_error = restored_pose_errors(
        cubes,
        source["cube_names"],
        source["cube_positions_world"],
        source["cube_orientations_wxyz"],
    )
    target_position, target_orientation = place_target.get_world_pose()
    target_position_error = float(
        np.linalg.norm(
            np.asarray(target_position, dtype=float)
            - np.asarray(source["place_target_position_world"], dtype=float)
        )
    )
    target_orientation_error = _quaternion_angle_error_rad(
        target_orientation,
        source["place_target_orientation_wxyz"],
    )
    cube_restored = bool(
        cube_position_error <= position_tolerance_m
        and cube_orientation_error <= orientation_tolerance_rad
    )
    target_restored = bool(
        target_position_error <= position_tolerance_m
        and target_orientation_error <= orientation_tolerance_rad
    )
    mismatch_reasons = []
    if not cube_restored:
        mismatch_reasons.append("cube_pose_mismatch")
    if not target_restored:
        mismatch_reasons.append("target_pose_mismatch")
    return {
        "cube_pose_restored": cube_restored,
        "target_pose_restored": target_restored,
        "robot_initial_state_restored": bool(robot_restored),
        "pose_mismatch": bool(mismatch_reasons),
        "pose_mismatch_reason": ",".join(mismatch_reasons),
        "max_cube_position_error_m": float(cube_position_error),
        "max_cube_orientation_error_rad": float(cube_orientation_error),
        "target_position_error_m": float(target_position_error),
        "target_orientation_error_rad": float(target_orientation_error),
    }


def _quaternion_angle_error_rad(actual: Any, expected: Any) -> float:
    first = np.asarray(actual, dtype=float).reshape(-1)[:4]
    second = np.asarray(expected, dtype=float).reshape(-1)[:4]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("inf")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _rule_gripper_should_close(
    obs: dict[str, np.ndarray],
    was_closed: bool,
    *,
    close_dist: float,
    release_dist: float,
) -> bool:
    ee_cube_dist = float(np.linalg.norm(obs["ee_to_cube"]))
    cube_target_dist = float(np.linalg.norm(obs["cube_to_place_target"]))
    has_grasped = bool(obs["has_grasped_cube"][0] > 0.5)
    if was_closed and cube_target_dist <= release_dist:
        return False
    if was_closed or has_grasped:
        return True
    return ee_cube_dist <= close_dist


def _finite_action(action: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
    )
    return clip_action(arr)


def _task_episode_flags(
    *, success: bool, strict_failure: bool, horizon_reached: bool
) -> tuple[bool, bool]:
    """Return Gymnasium termination flags without conflating task and time."""

    terminated = bool(success or strict_failure)
    truncated = bool(not terminated and horizon_reached)
    return terminated, truncated


def _valid_runtime_position(value) -> bool:
    if value is None:
        return False
    position = np.asarray(value, dtype=float).reshape(-1)
    return bool(
        position.size >= 3
        and np.all(np.isfinite(position[:3]))
        and np.linalg.norm(position[:3]) > 1e-6
    )


_OBSERVATION_HUMAN_STATE_KEYS = {
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "human_robot_collision",
    "near_human",
    "collision_green",
    "pick_miss_recent",
    "drop_throw_recent",
    "min_hand_gripper_dist_override",
    "min_hand_gripper_surface_gap_override",
    "left_hand_surface_gap_override",
    "right_hand_surface_gap_override",
    "left_hand_contact",
    "right_hand_contact",
    "near_miss",
    "distance_gate_override",
    "geometry_valid_override",
}


def _split_observation_human_state(
    payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep replay metadata out of build_observation's fixed keyword surface."""

    obs_payload: dict[str, Any] = {}
    aux_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _OBSERVATION_HUMAN_STATE_KEYS:
            obs_payload[key] = value
        else:
            aux_payload[key] = value
    return obs_payload, aux_payload


def _intentional_human_absence_from_aux(payload: Mapping[str, Any]) -> bool:
    """Read only the trusted boolean replay marker; reject truthy coercions."""

    value = payload.get("intentional_human_absence", False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("intentional_human_absence must be an exact boolean")
    return bool(value)


def _safe_robot_joint_vector(robot, method_name: str) -> list[float]:
    method = getattr(robot, method_name, None)
    if not callable(method):
        return []
    try:
        values = np.asarray(method(), dtype=float).reshape(-1)
    except Exception:
        return []
    if not np.all(np.isfinite(values)):
        return []
    return [float(value) for value in values]


def _articulation_action_payload(action) -> dict[str, Any]:
    """Serialize the exact action object passed to ``robot.apply_action``."""

    payload: dict[str, Any] = {}
    for source_name, output_name in (
        ("joint_indices", "joint_indices"),
        ("joint_positions", "joint_positions_rad"),
        ("joint_velocities", "joint_velocities_radps"),
        ("joint_efforts", "joint_efforts"),
    ):
        value = getattr(action, source_name, None)
        if value is None:
            payload[output_name] = []
            continue
        try:
            array = np.asarray(value).reshape(-1)
            if source_name == "joint_indices":
                payload[output_name] = [int(item) for item in array]
            else:
                numeric = np.asarray(array, dtype=float)
                payload[output_name] = (
                    [float(item) for item in numeric]
                    if np.all(np.isfinite(numeric))
                    else []
                )
        except Exception:
            payload[output_name] = []
    return payload


def _safe_quat(quat: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(quat, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
    )
    if arr.size < 4:
        result = np.zeros(4, dtype=float)
        result[: arr.size] = arr
        arr = result
    else:
        arr = arr[:4]
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / norm


def _policy_gripper_should_close(action: np.ndarray, was_closed: bool) -> bool:
    gripper_cmd = float(action[4])
    if gripper_cmd < -0.2:
        return True
    if gripper_cmd > 0.2:
        return False
    return was_closed


def _gripper_finger_world_positions(
    robot,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        left_pos, _ = robot.gripper._left_finger.get_world_pose()
        right_pos, _ = robot.gripper._right_finger.get_world_pose()
        left = np.asarray(left_pos, dtype=float).reshape(-1)
        right = np.asarray(right_pos, dtype=float).reshape(-1)
        if (
            left.size < 3
            or right.size < 3
            or not np.all(np.isfinite(left[:3]))
            or not np.all(np.isfinite(right[:3]))
        ):
            return None
        return left[:3].copy(), right[:3].copy()
    except Exception:
        return None


def _gripper_center_from_fingers(robot) -> np.ndarray | None:
    positions = _gripper_finger_world_positions(robot)
    if positions is None:
        return None
    return (positions[0] + positions[1]) * 0.5


def _has_grasped_cube(robot, cube, gripper_center: np.ndarray | None) -> bool:
    try:
        width = float(np.sum(robot.gripper.get_joint_positions()))
    except Exception:
        width = 0.1
    cube_pos, _ = cube.get_world_pose()
    center = gripper_center
    if center is None:
        center, _ = robot.end_effector.get_world_pose()
    dist = float(
        np.linalg.norm(
            np.asarray(cube_pos, dtype=float) - np.asarray(center, dtype=float)
        )
    )
    return width < 0.065 and dist < 0.11
