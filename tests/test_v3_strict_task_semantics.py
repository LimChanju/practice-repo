from __future__ import annotations

from types import SimpleNamespace

import pytest

from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv, _task_episode_flags
from v3_chan.rl.strict_task_semantics import (
    StrictTaskSemanticsConfig,
    StrictTaskSemanticsController,
    TaskPhysicalEvidence,
)


def _evidence(
    *,
    grasp: bool = False,
    ee_cube: float = 0.04,
    xy: float = 0.20,
    z_error: float = 0.10,
    speed: float = 0.0,
    cube_z: float = 0.30,
    intervention: float = 0.0,
) -> TaskPhysicalEvidence:
    return TaskPhysicalEvidence(
        ee_cube_distance_m=ee_cube,
        cube_target_xy_error_m=xy,
        cube_target_z_error_m=z_error,
        cube_speed_mps=speed,
        cube_z_m=cube_z,
        grasp_candidate=grasp,
        cbf_intervention_norm_radps=intervention,
    )


def _apply_authorized_release_open(
    controller: StrictTaskSemanticsController,
) -> None:
    """Record the guarded event-7 OPEN that precedes strict settling."""

    assert controller.release_command_allowed() is True
    applied = controller.notify_gripper_command_applied(
        event=7,
        command="open",
        grasp_candidate_before_command=True,
    )
    assert applied is True
    assert controller.state.release_command_applied is True
    assert controller.release_command_allowed() is False


def test_unconfirmed_grasp_never_advances_to_lift_and_then_fails() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            grasp_confirmation_steps=2,
            grasp_confirmation_timeout_steps=2,
            maximum_grasp_retries=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)

    first = controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=False),
    )
    assert first.event == 3
    assert first.held is True
    assert first.reason == "hold_for_grasp_confirmation"

    retry = controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=False),
    )
    assert retry.event == 0
    assert retry.retry_started is True
    assert controller.consume_pending_retry_open() is True

    controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=False),
    )
    failure = controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=False),
    )
    assert failure.event == 10
    assert failure.failure_reason == "grasp_confirmation_timeout"
    assert controller.state.success_latched is False


def test_grasp_needs_stable_confirmation_before_lift() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(enabled=True, grasp_confirmation_steps=2)
    )
    controller.reset(initial_cube_z_m=0.30)

    held = controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=True),
    )
    advanced = controller.update(
        event=3,
        progress=0.9,
        proposed_event=4,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(grasp=True),
    )
    assert held.event == 3
    assert held.held is True
    assert advanced.event == 4
    assert advanced.reason == "physical_transition_permitted"


def test_release_requires_stable_place_geometry_speed_lift_and_grasp() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            grasp_confirmation_steps=1,
            place_confirmation_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)

    controller.update(
        event=4,
        progress=0.0,
        proposed_event=4,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(grasp=True, cube_z=0.36),
    )
    not_ready = controller.update(
        event=6,
        progress=0.99,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.34,
            xy=0.03,
            z_error=0.02,
            speed=0.08,
        ),
    )
    assert not_ready.event == 6
    assert controller.release_command_allowed() is False

    ready_once = controller.update(
        event=6,
        progress=0.99,
        proposed_event=6,
        proposed_progress=0.995,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.34,
            xy=0.03,
            z_error=0.02,
            speed=0.02,
        ),
    )
    released = controller.update(
        event=6,
        progress=0.99,
        proposed_event=6,
        proposed_progress=0.995,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.34,
            xy=0.03,
            z_error=0.02,
            speed=0.02,
        ),
    )
    assert ready_once.event == 6
    assert released.event == 7
    assert released.reason == "event_driven_release_after_place_confirmation"
    assert controller.release_command_allowed() is True


def test_stale_place_readiness_cannot_release_without_reconfirmation() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            place_confirmation_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    ready = _evidence(
        grasp=True,
        cube_z=0.34,
        xy=0.03,
        z_error=0.02,
        speed=0.02,
    )

    ready_before_stale = controller.update(
        event=6,
        progress=0.5,
        proposed_event=6,
        proposed_progress=0.6,
        terminal_event=10,
        evidence=ready,
    )
    assert ready_before_stale.event == 6
    assert controller.state.place_ready_latched is False

    stale = controller.update(
        event=6,
        progress=0.99,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.34,
            xy=0.20,
            z_error=0.02,
            speed=0.50,
        ),
    )

    assert stale.event == 6
    assert stale.held is True
    assert controller.state.place_ready_latched is False
    assert controller.state.place_ready_streak == 0
    assert controller.release_command_allowed() is False

    ready_once = controller.update(
        event=6,
        progress=0.99,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=ready,
    )
    reconfirmed = controller.update(
        event=6,
        progress=0.99,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=ready,
    )

    assert ready_once.event == 6
    assert ready_once.held is True
    assert reconfirmed.event == 7
    assert reconfirmed.reason == "event_driven_release_after_place_confirmation"
    assert controller.release_command_allowed() is True


def test_release_settling_is_confirmed_before_success_latches() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            place_confirmation_steps=1,
            release_settle_confirmation_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    _apply_authorized_release_open(controller)
    settled = _evidence(
        grasp=False,
        cube_z=0.32,
        xy=0.02,
        z_error=0.01,
        speed=0.01,
    )

    first = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=settled,
    )
    second = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=settled,
    )
    assert first.success_latched is False
    assert first.event == 7
    assert second.success_latched is True
    assert second.event == 10
    assert controller.state.success_latched is True


def test_accidental_drop_at_target_without_open_cannot_latch_success() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            release_settle_confirmation_steps=1,
            release_settle_timeout_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    dropped_at_target = _evidence(
        grasp=False,
        cube_z=0.32,
        xy=0.02,
        z_error=0.01,
        speed=0.01,
    )

    held = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=dropped_at_target,
    )
    failed = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=dropped_at_target,
    )

    assert held.event == 7
    assert held.held is True
    assert held.reason == "hold_for_release_command_application"
    assert held.success_latched is False
    assert controller.state.release_settle_streak == 0
    assert controller.state.release_command_applied is False
    assert failed.event == 10
    assert failed.failure_reason == "release_not_commanded"
    assert failed.success_latched is False


def test_release_open_notification_rejects_invalid_provenance() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(enabled=True)
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.place_ready_latched = True

    assert controller.notify_gripper_command_applied(
        event=6,
        command="open",
        grasp_candidate_before_command=True,
    ) is False
    assert controller.notify_gripper_command_applied(
        event=7,
        command="close",
        grasp_candidate_before_command=True,
    ) is False
    assert controller.notify_gripper_command_applied(
        event=7,
        command="open",
        grasp_candidate_before_command=False,
    ) is False
    assert controller.state.last_transition_reason == (
        "release_open_after_grasp_loss_rejected"
    )
    assert controller.state.release_command_applied is False

    with pytest.raises(ValueError, match="grasp candidate before release"):
        controller.notify_gripper_command_applied(
            event=7,
            command="open",
            grasp_candidate_before_command=1,  # type: ignore[arg-type]
        )

    controller.state.place_ready_latched = False
    with pytest.raises(RuntimeError, match="without a valid release gate"):
        controller.notify_gripper_command_applied(
            event=7,
            command="open",
            grasp_candidate_before_command=True,
        )
    assert controller.state.release_command_applied is False


def test_release_latch_resets_and_restores_with_controller_state() -> None:
    config = StrictTaskSemanticsConfig(enabled=True, state_aware_recovery=True)
    controller = StrictTaskSemanticsController(config)
    controller.reset(initial_cube_z_m=0.30)
    controller.state.place_ready_latched = True
    _apply_authorized_release_open(controller)
    captured = controller.state.as_dict()

    controller.reset(initial_cube_z_m=0.31)
    assert controller.state.release_command_applied is False

    restored = StrictTaskSemanticsController(config)
    restored.restore_state(captured)
    assert restored.state.release_command_applied is True
    assert restored.release_command_allowed() is False

    restored.begin_recovery(
        mode="place",
        evidence=_evidence(grasp=True, cube_z=0.36),
    )
    assert restored.state.release_command_applied is False

    restored.state.place_ready_latched = True
    _apply_authorized_release_open(restored)
    restored.recovery_handoff(
        event=6,
        progress=0.0,
        mode="place",
        evidence=_evidence(grasp=True, cube_z=0.36),
        event_before=6,
        progress_before=0.5,
    )
    assert restored.state.release_command_applied is False

    legacy_state = dict(captured)
    legacy_state.pop("release_command_applied")
    legacy = StrictTaskSemanticsController(config)
    legacy.restore_state(legacy_state)
    assert legacy.state.release_command_applied is False


def test_external_reentry_requires_confirmed_grasp_or_actual_release() -> None:
    config = StrictTaskSemanticsConfig(
        enabled=True,
        grasp_lost_confirmation_steps=2,
    )

    attached = StrictTaskSemanticsController(config)
    attached.reset(initial_cube_z_m=0.30)
    assert attached.classify_external_reentry(
        event=6,
        evidence=_evidence(grasp=True, cube_z=0.36),
    ) == "place"

    released = StrictTaskSemanticsController(config)
    released.reset(initial_cube_z_m=0.30)
    released.state.place_ready_latched = True
    _apply_authorized_release_open(released)
    assert released.classify_external_reentry(
        event=7,
        evidence=_evidence(
            grasp=False,
            cube_z=0.32,
            xy=0.02,
            z_error=0.01,
            speed=0.01,
        ),
    ) == "released"

    missing = StrictTaskSemanticsController(config)
    missing.reset(initial_cube_z_m=0.30)
    missing.state.grasp_observed = True
    missing_evidence = _evidence(grasp=False, cube_z=0.31)
    assert missing.classify_external_reentry(
        event=7,
        evidence=missing_evidence,
    ) == "wait"
    assert missing.classify_external_reentry(
        event=7,
        evidence=missing_evidence,
    ) == "regrasp"
    assert missing.state.release_command_applied is False


def test_cbf_pause_uses_hysteresis_then_reenters() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)

    paused = controller.update(
        event=1,
        progress=0.4,
        proposed_event=1,
        proposed_progress=0.5,
        terminal_event=10,
        evidence=_evidence(intervention=0.06),
    )
    clearing = controller.update(
        event=1,
        progress=0.4,
        proposed_event=1,
        proposed_progress=0.5,
        terminal_event=10,
        evidence=_evidence(intervention=0.0),
    )
    resumed = controller.update(
        event=1,
        progress=0.4,
        proposed_event=1,
        proposed_progress=0.5,
        terminal_event=10,
        evidence=_evidence(intervention=0.0),
    )
    assert paused.held is True
    assert clearing.held is True
    assert resumed.reentry is True
    assert resumed.reason == "cbf_reentry_resume_current_phase"
    assert resumed.event == 1
    assert controller.state.cbf_hold_latched is False


def test_current_step_cbf_blocks_release_before_action_is_applied() -> None:
    env = object.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(strict_task_semantics=True)
    env.phase_event = 7
    env.gripper_closed = False
    env._strict_task_semantics_config = StrictTaskSemanticsConfig(
        enabled=True,
        cbf_pause_enter_norm_radps=0.05,
    )
    env._last_physical_safety_diagnostics = SimpleNamespace(
        intervention_norm_radps=0.05
    )
    env._last_obs = {"has_grasped_cube": [1.0]}
    env._strict_task_controller = StrictTaskSemanticsController(
        env._strict_task_semantics_config
    )
    env._strict_task_controller.reset(initial_cube_z_m=0.30)
    env._strict_task_controller.state.place_ready_latched = True

    command = env._guard_strict_release_for_current_cbf("open")
    notified = env._strict_task_controller.notify_gripper_command_applied(
        event=env.phase_event,
        command=command,
        grasp_candidate_before_command=True,
    )

    assert command is None
    assert notified is False
    assert env._strict_task_controller.state.release_command_applied is False
    assert env.gripper_closed is True


def test_strict_policy_gripper_blocks_open_while_object_is_grasped_before_release() -> None:
    env = object.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(strict_task_semantics=True, gripper_mode="policy")
    env.phase_event = 3
    env.gripper_closed = True
    env._strict_task_controller = SimpleNamespace(
        release_command_allowed=lambda: False
    )

    command = env._gripper_command(
        [0.0, 0.0, 0.0, 0.0, 1.0],
        {"has_grasped_cube": [1.0]},
    )

    assert command is None
    assert env.gripper_closed is True


def test_strict_policy_gripper_keeps_policy_owned_open_at_ready_release() -> None:
    env = object.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(strict_task_semantics=True, gripper_mode="policy")
    env.phase_event = 7
    env.gripper_closed = True
    env._strict_task_controller = SimpleNamespace(
        release_command_allowed=lambda: True
    )

    command = env._gripper_command(
        [0.0, 0.0, 0.0, 0.0, 1.0],
        {"has_grasped_cube": [1.0]},
    )

    assert command == "open"
    assert env.gripper_closed is False


def test_current_step_release_guard_preserves_safe_and_legacy_open() -> None:
    env = object.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(strict_task_semantics=True)
    env.phase_event = 7
    env.gripper_closed = False
    env._strict_task_semantics_config = StrictTaskSemanticsConfig(
        enabled=True,
        cbf_pause_enter_norm_radps=0.05,
    )
    env._last_physical_safety_diagnostics = SimpleNamespace(
        intervention_norm_radps=0.049
    )
    env._last_obs = {"has_grasped_cube": [1.0]}
    assert env._guard_strict_release_for_current_cbf("open") == "open"
    assert env.gripper_closed is False

    env.config = SimpleNamespace(strict_task_semantics=False)
    env._last_physical_safety_diagnostics = SimpleNamespace(
        intervention_norm_radps=1.0
    )
    assert env._guard_strict_release_for_current_cbf("open") == "open"


def test_current_step_release_guard_does_not_reclose_released_cube() -> None:
    env = object.__new__(IsaacPickPlaceEnv)
    env.config = SimpleNamespace(strict_task_semantics=True)
    env.phase_event = 7
    env.gripper_closed = False
    env._strict_task_semantics_config = StrictTaskSemanticsConfig(
        enabled=True,
        cbf_pause_enter_norm_radps=0.05,
    )
    env._last_physical_safety_diagnostics = SimpleNamespace(
        intervention_norm_radps=0.50
    )
    env._last_obs = {"has_grasped_cube": [0.0]}

    assert env._guard_strict_release_for_current_cbf("open") == "open"
    assert env.gripper_closed is False


def test_task_episode_flags_separate_success_failure_and_horizon() -> None:
    assert _task_episode_flags(
        success=True, strict_failure=False, horizon_reached=False
    ) == (True, False)
    assert _task_episode_flags(
        success=False, strict_failure=True, horizon_reached=False
    ) == (True, False)
    assert _task_episode_flags(
        success=False, strict_failure=False, horizon_reached=True
    ) == (False, True)
    assert _task_episode_flags(
        success=False, strict_failure=True, horizon_reached=True
    ) == (True, False)


def test_released_cube_settling_is_not_blocked_by_cbf_intervention() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            maximum_grasp_retries=0,
            release_settle_confirmation_steps=1,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    _apply_authorized_release_open(controller)
    controller.state.cbf_hold_latched = True

    def released_evidence(intervention: float) -> TaskPhysicalEvidence:
        return _evidence(
            grasp=False,
            cube_z=0.32,
            xy=0.02,
            z_error=0.01,
            speed=0.01,
            intervention=intervention,
        )

    success = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=released_evidence(0.06),
    )

    assert success.event == 10
    assert success.success_latched is True


def test_cbf_reentry_reconfirms_still_ready_release_from_zero() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            place_confirmation_steps=2,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    controller.state.place_ready_streak = 2
    ready_paused = _evidence(
        grasp=True,
        cube_z=0.36,
        xy=0.02,
        z_error=0.01,
        speed=0.01,
        intervention=0.06,
    )

    paused = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=ready_paused,
    )
    reentered = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.02,
            z_error=0.01,
            speed=0.01,
            intervention=0.0,
        ),
    )

    assert paused.held is True
    assert reentered.event == 6
    assert reentered.reason == "cbf_reentry_reconfirm_release"
    assert controller.state.place_ready_streak == 0
    assert controller.release_command_allowed() is False

    first_confirmation = controller.update(
        event=6,
        progress=0.0,
        proposed_event=6,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.02,
            z_error=0.01,
            speed=0.01,
        ),
    )

    assert first_confirmation.event == 6
    assert controller.state.place_ready_latched is False
    assert controller.release_command_allowed() is False

    second_confirmation = controller.update(
        event=6,
        progress=0.0,
        proposed_event=6,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.02,
            z_error=0.01,
            speed=0.01,
        ),
    )

    assert controller.state.place_ready_latched is True
    assert second_confirmation.event == 7
    assert second_confirmation.reason == "event_driven_release_after_place_confirmation"


def test_cbf_reentry_revalidates_guarded_release() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            place_confirmation_steps=2,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=2,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    controller.state.place_ready_streak = 2

    unsafe = _evidence(
        grasp=True,
        cube_z=0.36,
        xy=0.20,
        z_error=0.10,
        speed=0.20,
        intervention=0.06,
    )
    paused = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=unsafe,
    )
    clearing = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.20,
            z_error=0.10,
            speed=0.20,
            intervention=0.0,
        ),
    )
    reentered = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.20,
            z_error=0.10,
            speed=0.20,
            intervention=0.0,
        ),
    )

    assert paused.held is True
    assert clearing.held is True
    assert reentered.event == 5
    assert reentered.reentry is True
    assert reentered.reason == "cbf_reentry_reposition_before_release"
    assert controller.state.place_ready_latched is False
    assert controller.state.place_ready_streak == 0
    assert controller.release_command_allowed() is False


def test_state_aware_cbf_reentry_preserves_attached_near_goal_progress() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            place_confirmation_steps=2,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    near_goal = dict(
        grasp=True,
        cube_z=0.36,
        xy=0.007,
        z_error=0.01,
        speed=0.01,
    )

    paused = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**near_goal, intervention=0.06),
    )
    reentered = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**near_goal, intervention=0.0),
    )

    assert paused.event == 6
    assert paused.progress == 0.91
    assert paused.held is True
    assert reentered.event == 6
    assert reentered.progress == 0.91
    assert reentered.reentry is True
    assert reentered.held is False
    assert reentered.recovery_request == "none"
    assert reentered.reason == "cbf_reentry_preserve_place_progress"

    first_confirmation = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**near_goal),
    )
    released = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**near_goal),
    )

    assert first_confirmation.event == 6
    assert controller.release_command_allowed() is True
    assert released.event == 7
    assert released.reason == "event_driven_release_after_place_confirmation"


def test_state_aware_cbf_reentry_requests_place_bridge_without_event5_rewind() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    far_from_goal = dict(
        grasp=True,
        cube_z=0.36,
        xy=0.20,
        z_error=0.10,
        speed=0.01,
    )

    controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**far_from_goal, intervention=0.06),
    )
    reentered = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(**far_from_goal, intervention=0.0),
    )

    assert reentered.event == 6
    assert reentered.progress == 0.91
    assert reentered.held is True
    assert reentered.reentry is True
    assert reentered.recovery_request == "place"
    assert reentered.reason == "cbf_reentry_request_place_recovery"
    assert controller.state.recovery_request_count == 1


def test_state_aware_cbf_reentry_requests_current_state_regrasp_after_loss() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            grasp_lost_confirmation_steps=1,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True

    controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=False,
            cube_z=0.31,
            xy=0.20,
            z_error=0.10,
            intervention=0.06,
        ),
    )
    reentered = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=False,
            cube_z=0.31,
            xy=0.20,
            z_error=0.10,
            intervention=0.0,
        ),
    )

    assert reentered.event == 6
    assert reentered.progress == 0.91
    assert reentered.held is True
    assert reentered.reentry is True
    assert reentered.retry_started is False
    assert reentered.recovery_request == "regrasp"
    assert reentered.reason == "cbf_reentry_request_regrasp_recovery"
    assert controller.state.grasp_retry_count == 0
    assert controller.consume_pending_retry_open() is False


def test_state_aware_regrasp_waits_for_confirmed_post_cbf_loss() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            grasp_lost_confirmation_steps=3,
            cbf_pause_enter_norm_radps=0.05,
            cbf_pause_exit_norm_radps=0.01,
            cbf_pause_exit_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    missing = _evidence(
        grasp=False,
        cube_z=0.31,
        xy=0.20,
        z_error=0.10,
    )

    controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=False,
            cube_z=0.31,
            xy=0.20,
            z_error=0.10,
            intervention=0.06,
        ),
    )
    unconfirmed = controller.update(
        event=6,
        progress=0.91,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=missing,
    )
    assert unconfirmed.recovery_request == "none"
    assert unconfirmed.reason == "hold_for_post_cbf_grasp_loss_confirmation"

    confirmed = None
    for _ in range(2):
        confirmed = controller.update(
            event=6,
            progress=0.91,
            proposed_event=7,
            proposed_progress=0.0,
            terminal_event=10,
            evidence=missing,
        )
    assert confirmed is not None
    assert confirmed.recovery_request == "regrasp"
    assert confirmed.reason == "request_regrasp_recovery_after_grasp_lost_before_release"


def test_state_aware_place_timeout_requests_bridge_instead_of_event5_replay() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            place_confirmation_timeout_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.cbf_intervention_observed = True

    decision = controller.update(
        event=6,
        progress=0.99,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.20,
            z_error=0.10,
            speed=0.01,
        ),
    )

    assert decision.event == 6
    assert decision.progress == 0.99
    assert decision.held is True
    assert decision.recovery_request == "place"
    assert decision.reason == "request_place_recovery_after_readiness_timeout"


def test_state_aware_completed_event6_requests_place_recovery_immediately() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            place_confirmation_timeout_steps=240,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.cbf_intervention_observed = True

    decision = controller.update(
        event=6,
        progress=1.0,
        proposed_event=7,
        proposed_progress=0.0,
        terminal_event=10,
        evidence=_evidence(
            grasp=True,
            cube_z=0.36,
            xy=0.20,
            z_error=0.10,
            speed=0.01,
        ),
    )

    assert decision.event == 6
    assert decision.progress == 1.0
    assert decision.held is True
    assert decision.reentry is True
    assert decision.recovery_request == "place"
    assert controller.state.recovery_request_count == 1
    assert controller.state.place_wait_steps == 0
    assert controller.state.failure_reason == ""


def test_state_aware_recovery_handoff_clears_stale_release_gate() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(enabled=True, state_aware_recovery=True)
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_streak = 6
    controller.state.place_ready_latched = True
    controller.state.place_wait_steps = 12
    controller.state.cbf_hold_latched = True

    controller.begin_recovery(
        mode="place",
        evidence=_evidence(grasp=True, cube_z=0.36),
    )
    handoff = controller.recovery_handoff(
        event=6,
        progress=0.0,
        mode="place",
        evidence=_evidence(grasp=True, cube_z=0.36),
        event_before=6,
        progress_before=0.91,
    )

    assert handoff.event == 6
    assert handoff.progress == 0.0
    assert handoff.reentry is True
    assert handoff.reason == "state_aware_place_handoff"
    assert controller.state.cbf_hold_latched is False
    assert controller.state.place_ready_streak == 0
    assert controller.state.place_ready_latched is False
    assert controller.state.place_wait_steps == 0
    assert controller.release_command_allowed() is False


def test_state_aware_success_latch_remains_absorbing_during_later_cbf_signal() -> None:
    controller = StrictTaskSemanticsController(
        StrictTaskSemanticsConfig(
            enabled=True,
            state_aware_recovery=True,
            release_settle_confirmation_steps=1,
        )
    )
    controller.reset(initial_cube_z_m=0.30)
    controller.state.maximum_cube_lift_m = 0.06
    controller.state.grasp_observed = True
    controller.state.place_ready_latched = True
    _apply_authorized_release_open(controller)
    settled = dict(
        grasp=False,
        cube_z=0.32,
        xy=0.02,
        z_error=0.01,
        speed=0.01,
    )

    success = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(**settled),
    )
    after_cbf = controller.update(
        event=7,
        progress=0.0,
        proposed_event=7,
        proposed_progress=0.1,
        terminal_event=10,
        evidence=_evidence(**settled, intervention=0.50),
    )

    assert success.event == 10
    assert success.success_latched is True
    assert after_cbf.event == 10
    assert after_cbf.success_latched is True
    assert after_cbf.reason == "terminal_success_latched"
    assert after_cbf.recovery_request == "none"
    assert controller.state.success_latched is True
