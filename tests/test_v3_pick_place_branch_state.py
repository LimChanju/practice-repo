from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv
from v3_chan.rl.state_aware_recovery import (
    StateAwareRecoveryBridge,
    StateAwareRecoveryConfig,
    StateAwareRecoveryDecision,
)
from v3_chan.rl.strict_task_semantics import (
    StrictTaskPhaseDecision,
    StrictTaskSemanticsConfig,
    StrictTaskSemanticsController,
)


class _Body:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.linear_velocity = np.array([0.1, 0.2, 0.3])
        self.angular_velocity = np.array([0.4, 0.5, 0.6])

    def get_world_pose(self):
        return self.position.copy(), self.orientation.copy()

    def set_world_pose(self, position, orientation=None):
        self.position = np.asarray(position, dtype=float).copy()
        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=float).copy()

    def get_linear_velocity(self):
        return self.linear_velocity.copy()

    def get_angular_velocity(self):
        return self.angular_velocity.copy()

    def set_linear_velocity(self, value):
        self.linear_velocity = np.asarray(value, dtype=float).copy()

    def set_angular_velocity(self, value):
        self.angular_velocity = np.asarray(value, dtype=float).copy()


class _Robot:
    def __init__(self):
        self.positions = np.arange(9, dtype=float)
        self.velocities = np.arange(9, dtype=float) * 0.1
        self.applied_action = _Action(
            joint_positions=np.arange(9, dtype=float) + 10.0,
            joint_velocities=np.arange(9, dtype=float) + 20.0,
            joint_efforts=np.arange(9, dtype=float) + 30.0,
        )

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_positions(self, value):
        self.positions = np.asarray(value, dtype=float).copy()

    def set_joint_velocities(self, value):
        self.velocities = np.asarray(value, dtype=float).copy()

    def get_applied_action(self):
        return self.applied_action

    def apply_action(self, action):
        self.applied_action = action


class _Action:
    def __init__(
        self,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
    ):
        self.joint_positions = _copy_optional(joint_positions)
        self.joint_velocities = _copy_optional(joint_velocities)
        self.joint_efforts = _copy_optional(joint_efforts)


def _copy_optional(value):
    return None if value is None else np.asarray(value, dtype=float).copy()


class _Resettable:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Replay:
    def __init__(self):
        self.cursor = 7

    def capture_state(self):
        return {"cursor": self.cursor}

    def restore_state(self, state):
        self.cursor = int(state["cursor"])


class _Dynamic:
    def __init__(self):
        self.value = np.array([1.0, 2.0])


class _World:
    current_time = 1.25


def _fake_env(*, strict: bool = True, recovery: bool = True):
    if recovery and not strict:
        raise ValueError("recovery test fixture requires strict semantics")
    env = IsaacPickPlaceEnv.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(
        strict_task_semantics=bool(strict),
        state_aware_recovery=bool(recovery),
    )
    env._physical_safety_mode = "none"
    env.robot = _Robot()
    env.cubes = [_Body([0.0, 0.0, 0.0]), _Body([1.0, 0.0, 0.0])]
    env.pick_targets = env.cubes
    env.active_cube = env.cubes[1]
    env.place_target = _Body([2.0, 3.0, 4.0])
    env.place_pos = np.array([2.0, 3.0, 4.0])
    env.step_count = 12
    env.phase_event = 6 if recovery else 4
    env.phase_t = 0.8 if recovery else 1.5
    env.phase_hold_steps = 3
    env.gripper_closed = True
    env.yaw = 0.4
    env.rng = np.random.default_rng(9)
    env._last_obs = {"ee_pos": np.array([0.3, 0.4, 0.5])}
    env._pseudo_errp_aux_flags = {"a": 1.0}
    env._human_replay_aux_state = {"encounter_active": 1.0}
    env._last_safety_result = {"gap": 0.03}
    env.dynamic_safety = _Dynamic()
    env._last_dynamic_safety_sample = {"ttc": 0.5}
    env._source_restoration_diagnostics = {"restoration_mode": "exact_pose"}
    env._synthetic_human_active = False
    env._synthetic_human_start_step = 2
    env._synthetic_human_duration_steps = 5
    env._synthetic_human_side = -1.0
    env._synthetic_human_height_offset = 0.02
    env.human_state_fn = _Replay()
    env.safety_geometry = _Resettable()
    env.safety_geometry._previous_closest_collider = {"left": "a"}
    env.safety_geometry._last_debug_state = {"left": (1,)}
    env.safety_geometry.reset_link_origin_pose_cache = env.safety_geometry.reset
    env.controller = _Resettable()
    env._last_physical_safety_diagnostics = {"controller": "none"}
    env._strict_task_semantics_config = StrictTaskSemanticsConfig(
        enabled=bool(strict),
        state_aware_recovery=bool(recovery),
    )
    env._strict_task_controller = StrictTaskSemanticsController(
        env._strict_task_semantics_config
    )
    env._strict_task_controller.reset(initial_cube_z_m=0.30)
    if strict:
        env._strict_task_controller.state.grasp_observed = True
        env._strict_task_controller.state.maximum_cube_lift_m = 0.08
        env._strict_task_controller.state.cbf_clear_streak = 3
        env._strict_task_controller.state.recovery_request_count = 2
        env._strict_task_controller.state.place_recovery_started = bool(recovery)
        env._strict_task_controller.state.place_recovery_in_progress = bool(recovery)
    env._last_strict_task_decision = (
        StrictTaskPhaseDecision(
            event=int(env.phase_event),
            progress=float(env.phase_t),
            reason="fixture_strict_decision",
            phase_changed=False,
            held=bool(recovery),
            retry_started=False,
            reentry=False,
            success_latched=False,
            failure_reason="",
            recovery_request="none",
        )
        if strict
        else None
    )
    env._state_aware_recovery_config = StateAwareRecoveryConfig(
        enabled=bool(recovery)
    )
    env._state_aware_recovery = StateAwareRecoveryBridge(
        env._state_aware_recovery_config
    )
    if recovery:
        env._state_aware_recovery.state.active = True
        env._state_aware_recovery.state.mode = "place"
        env._state_aware_recovery.state.stage = "place_descend"
        env._state_aware_recovery.state.plan_id = 7
        env._state_aware_recovery.state.ready_streak = 5
        env._state_aware_recovery.state.active_steps = 11
        env._state_aware_recovery.state.activation_count = 2
        env._state_aware_recovery.state.last_target_position_m = (
            0.60,
            0.00,
            0.315,
        )
        env._state_aware_recovery.state.last_anchor_error_m = 0.001
        recovery_decision = StateAwareRecoveryDecision(
            active=True,
            plan_id=7,
            mode="place",
            stage="place_descend",
            target_position_m=np.array([0.60, 0.00, 0.315]),
            desired_gripper_closed=True,
            anchor_ready=True,
            ready_streak=5,
            anchor_error_m=0.001,
            handoff=True,
            handoff_event=6,
            handoff_progress=1.0,
            stage_changed=False,
            timed_out=False,
            reason="place_endpoint_handoff",
        )
    else:
        recovery_decision = None
    env._last_recovery_decision = recovery_decision
    env._pending_recovery_handoff = recovery_decision
    env._recovery_control_applied = bool(recovery)
    env._recovery_handoff_accepted = False
    env._last_recovery_alignment_token = (
        (2, 0, "place") if recovery else None
    )
    env._last_gripper_command = "close"
    env.world = _World()
    return env


def test_branch_state_restores_robot_scene_task_dynamic_and_replay_state():
    env = _fake_env()
    state = env.capture_branch_state()
    expected_rng = copy.deepcopy(state.rng_state)
    expected_applied_action = copy.deepcopy(state.robot_applied_action_state)

    env.robot.positions[:] = -1.0
    env.robot.velocities[:] = -2.0
    env.robot.applied_action = _Action(
        joint_positions=np.full(9, -10.0),
        joint_velocities=np.full(9, -20.0),
        joint_efforts=np.full(9, -30.0),
    )
    env.cubes[0].position[:] = 8.0
    env.cubes[0].linear_velocity[:] = 9.0
    env.place_target.position[:] = 7.0
    env.place_pos[:] = 6.0
    env.active_cube = env.cubes[0]
    env.step_count = 99
    env.phase_event = 7
    env._last_obs["ee_pos"][:] = -3.0
    env.dynamic_safety.value[:] = 10.0
    env.human_state_fn.cursor = 42
    env.rng.random()
    env.world.current_time = 1.75
    env._strict_task_controller.state.grasp_observed = False
    env._strict_task_controller.state.maximum_cube_lift_m = 0.0
    env._strict_task_controller.state.cbf_clear_streak = 0
    env._strict_task_controller.state.recovery_request_count = 0
    env._strict_task_controller.state.place_recovery_started = False
    env._strict_task_controller.state.place_recovery_in_progress = False
    env._strict_task_controller.state.release_command_applied = True
    env._last_strict_task_decision = None
    env._state_aware_recovery.reset()
    env._last_recovery_decision = None
    env._pending_recovery_handoff = None
    env._recovery_control_applied = False
    env._recovery_handoff_accepted = True
    env._last_recovery_alignment_token = None
    env._last_gripper_command = None

    diagnostics = env.restore_branch_state(state)

    np.testing.assert_allclose(env.robot.positions, np.arange(9))
    np.testing.assert_allclose(
        env.robot.applied_action.joint_positions,
        expected_applied_action["joint_positions"],
    )
    np.testing.assert_allclose(
        env.robot.applied_action.joint_velocities,
        expected_applied_action["joint_velocities"],
    )
    np.testing.assert_allclose(
        env.robot.applied_action.joint_efforts,
        expected_applied_action["joint_efforts"],
    )
    np.testing.assert_allclose(env.cubes[0].position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(env.cubes[0].linear_velocity, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(env.place_target.position, [2.0, 3.0, 4.0])
    assert env.active_cube is env.cubes[1]
    assert env.step_count == 12
    assert env.phase_event == 6
    np.testing.assert_allclose(env._last_obs["ee_pos"], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(env.dynamic_safety.value, [1.0, 2.0])
    assert env.human_state_fn.cursor == 7
    assert env.rng.bit_generator.state == expected_rng
    assert env.controller.reset_count == 1
    assert env._strict_task_controller.state.grasp_observed is True
    assert env._strict_task_controller.state.maximum_cube_lift_m == 0.08
    assert env._strict_task_controller.state.cbf_clear_streak == 3
    assert env._strict_task_controller.state.recovery_request_count == 2
    assert env._strict_task_controller.state.place_recovery_started is True
    assert env._strict_task_controller.state.place_recovery_in_progress is True
    assert env._strict_task_controller.state.release_command_applied is False
    assert env._last_strict_task_decision is not None
    assert env._last_strict_task_decision.reason == "fixture_strict_decision"
    assert env._state_aware_recovery.state.active is True
    assert env._state_aware_recovery.state.mode == "place"
    assert env._state_aware_recovery.state.stage == "place_descend"
    assert env._state_aware_recovery.state.plan_id == 7
    assert env._state_aware_recovery.state.ready_streak == 5
    assert env._state_aware_recovery.state.active_steps == 11
    assert env._state_aware_recovery.state.activation_count == 2
    assert env._state_aware_recovery.state.last_anchor_error_m == 0.001
    assert env._last_recovery_decision is not None
    assert env._last_recovery_decision.handoff is True
    assert env._last_recovery_decision.handoff_event == 6
    assert env._last_recovery_decision.handoff_progress == 1.0
    np.testing.assert_allclose(
        env._last_recovery_decision.target_position_m,
        [0.60, 0.00, 0.315],
    )
    assert env._pending_recovery_handoff is not None
    assert env._pending_recovery_handoff.plan_id == 7
    np.testing.assert_allclose(
        env._pending_recovery_handoff.target_position_m,
        [0.60, 0.00, 0.315],
    )
    assert env._recovery_control_applied is True
    assert env._recovery_handoff_accepted is False
    assert env._last_recovery_alignment_token == (2, 0, "place")
    assert env._last_gripper_command == "close"
    assert diagnostics["world_time_advance_s"] == 0.5


def test_branch_state_restores_release_latch_without_recovery():
    env = _fake_env(strict=True, recovery=False)
    env._strict_task_controller.state.release_command_applied = True
    state = env.capture_branch_state()

    env._strict_task_controller.state.release_command_applied = False
    env.restore_branch_state(state)

    assert env._strict_task_controller.state.release_command_applied is True
    assert env._state_aware_recovery.state.active is False


def test_branch_state_rejects_mode_and_config_mismatch_before_robot_mutation():
    state = _fake_env().capture_branch_state()

    mode_mismatch = _fake_env()
    mode_mismatch.robot.positions[:] = -11.0
    mode_mismatch.config.state_aware_recovery = False
    with pytest.raises(ValueError, match="recovery mode"):
        mode_mismatch.restore_branch_state(state)
    np.testing.assert_allclose(mode_mismatch.robot.positions, -11.0)

    config_mismatch = _fake_env()
    config_mismatch.robot.positions[:] = -12.0
    config_mismatch._strict_task_semantics_config = StrictTaskSemanticsConfig(
        enabled=True,
        state_aware_recovery=True,
        place_xy_tolerance_m=0.123,
    )
    with pytest.raises(ValueError, match="strict-task config"):
        config_mismatch.restore_branch_state(state)
    np.testing.assert_allclose(config_mismatch.robot.positions, -12.0)


def test_disabled_branch_state_round_trip_and_recovery_decisions_are_deep_copied():
    disabled = _fake_env(strict=False, recovery=False)
    disabled_state = disabled.capture_branch_state()
    disabled._strict_task_controller.state.initial_cube_z_m = 9.0
    disabled._last_gripper_command = None

    disabled.restore_branch_state(disabled_state)

    assert disabled._strict_task_controller.state.initial_cube_z_m == 0.30
    assert disabled._state_aware_recovery.state.active is False
    assert disabled._last_strict_task_decision is None
    assert disabled._last_recovery_decision is None
    assert disabled._pending_recovery_handoff is None
    assert disabled._last_gripper_command == "close"

    enabled = _fake_env()
    enabled_state = enabled.capture_branch_state()
    assert enabled._last_recovery_decision is not None
    assert enabled_state.last_recovery_decision is not None
    enabled._last_recovery_decision.target_position_m[:] = -1.0
    np.testing.assert_allclose(
        enabled_state.last_recovery_decision.target_position_m,
        [0.60, 0.00, 0.315],
    )

    enabled.restore_branch_state(enabled_state)
    assert enabled._pending_recovery_handoff is not None
    enabled._pending_recovery_handoff.target_position_m[:] = -2.0
    np.testing.assert_allclose(
        enabled_state.pending_recovery_handoff.target_position_m,
        [0.60, 0.00, 0.315],
    )


def test_branch_state_rejects_active_third_party_safety_controller():
    env = _fake_env()
    env._physical_safety_mode = "rmpflow"
    try:
        env.capture_branch_state()
    except RuntimeError as exc:
        assert "physical_safety_controller='none'" in str(exc)
    else:
        raise AssertionError("Expected capture_branch_state() to reject rmpflow")
