from __future__ import annotations

from types import SimpleNamespace

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

    command = env._guard_strict_release_for_current_cbf("open")

    assert command is None
    assert env.gripper_closed is True


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
