from __future__ import annotations

import numpy as np
import pytest

from v3_chan.rl.state_aware_recovery import (
    StateAwareRecoveryBridge,
    StateAwareRecoveryConfig,
    StateAwareRecoveryEvidence,
)


def _config(**overrides: object) -> StateAwareRecoveryConfig:
    values: dict[str, object] = {
        "enabled": True,
        "controller_target_y_offset_m": 0.0,
        "prepose_height_m": 0.20,
        "observed_ee_frame_offset_x_m": 0.0,
        "observed_ee_frame_offset_y_m": 0.0,
        "observed_ee_frame_offset_z_m": 0.0,
        "minimum_transport_clearance_m": 0.10,
        "anchor_position_tolerance_m": 0.01,
        "maximum_cube_speed_mps": 0.05,
        "maximum_joint_speed_radps": 0.25,
        "anchor_confirmation_steps": 2,
        "maximum_recovery_steps": 20,
    }
    values.update(overrides)
    return StateAwareRecoveryConfig(**values)


def _evidence(
    *,
    ee: tuple[float, float, float] = (0.30, 0.0, 0.50),
    cube: tuple[float, float, float] = (0.40, 0.10, 0.45),
    place: tuple[float, float, float] = (0.60, 0.0, 0.30),
    cube_speed: float = 0.0,
    joint_speed: float = 0.0,
    grasped: bool = True,
    cbf_clear: bool = True,
) -> StateAwareRecoveryEvidence:
    return StateAwareRecoveryEvidence(
        ee_position_m=ee,
        cube_position_m=cube,
        place_target_position_m=place,
        cube_speed_mps=cube_speed,
        joint_speed_radps=joint_speed,
        has_grasped_cube=grasped,
        cbf_clear=cbf_clear,
    )


def test_v3_rejects_preplacement_handoff_progress() -> None:
    with pytest.raises(ValueError, match="placed endpoint"):
        StateAwareRecoveryBridge(_config(place_handoff_progress=0.2))


def test_attached_far_descends_after_transport_and_hands_off_after_placement() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=0.91,
    )

    seeking = bridge.update(_evidence())

    assert seeking.active is True
    assert seeking.mode == "place"
    assert seeking.stage == "place_transport"
    assert np.allclose(seeking.target_position_m, [0.60, 0.0, 0.50])
    assert seeking.desired_gripper_closed is True
    assert seeking.handoff is False
    assert bridge.state.source_event == 6
    assert bridge.state.source_progress == 0.91

    transport_ready = _evidence(
        ee=(0.60, 0.0, 0.50),
        cube=(0.60, 0.0, 0.50),
    )
    confirming = bridge.update(transport_ready)
    descend = bridge.update(transport_ready)

    assert confirming.active is True
    assert confirming.anchor_ready is True
    assert confirming.ready_streak == 1
    assert confirming.handoff is False
    assert descend.active is True
    assert descend.stage == "place_descend"
    assert descend.stage_changed is True
    assert descend.handoff is False
    assert np.allclose(descend.target_position_m, [0.60, 0.0, 0.315])

    placed = _evidence(
        ee=(0.60, 0.0, 0.315),
        cube=(0.60, 0.0, 0.315),
    )
    placement_confirming = bridge.update(placed)
    handoff = bridge.update(placed)

    assert placement_confirming.anchor_ready is True
    assert placement_confirming.ready_streak == 1
    assert placement_confirming.handoff is False
    # The bridge retains authority until the environment validates the
    # post-action state and explicitly accepts the handoff.
    assert handoff.active is True
    assert handoff.stage == "place_descend"
    assert handoff.handoff is True
    assert handoff.handoff_event == 6
    assert handoff.handoff_progress == 1.0
    assert bridge.state.completion_count == 0
    bridge.accept_handoff()
    assert bridge.state.active is False
    assert bridge.state.completion_count == 1
    assert handoff.reason == "place_endpoint_handoff"
    assert bridge.state.completion_count == 1


def test_low_attached_cube_is_lifted_before_lateral_place_transport() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    low_cube = _evidence(cube=(0.40, 0.10, 0.34))
    bridge.request(
        "place",
        low_cube,
        source_event=5,
        source_progress=0.72,
    )

    lift = bridge.update(low_cube)
    assert lift.stage == "place_lift"
    assert np.allclose(lift.target_position_m, [0.40, 0.10, 0.50])

    lift_ready = _evidence(
        ee=(0.40, 0.10, 0.50),
        cube=(0.40, 0.10, 0.40),
    )
    bridge.update(lift_ready)
    transport = bridge.update(lift_ready)

    assert transport.active is True
    assert transport.stage_changed is True
    assert transport.stage == "place_transport"
    assert transport.handoff is False
    assert np.allclose(transport.target_position_m, [0.60, 0.0, 0.50])


def test_lost_grasp_regrasp_target_tracks_current_cube_pose() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    initial = _evidence(
        cube=(0.40, 0.10, 0.30),
        grasped=False,
    )
    bridge.request(
        "regrasp",
        initial,
        source_event=6,
        source_progress=0.91,
    )

    moved = _evidence(
        cube=(0.52, -0.12, 0.31),
        grasped=False,
    )
    decision = bridge.update(moved)

    assert decision.active is True
    assert decision.mode == "regrasp"
    assert decision.stage == "regrasp_prepose"
    assert decision.desired_gripper_closed is False
    assert np.allclose(decision.target_position_m, [0.52, -0.12, 0.51])
    assert not np.allclose(decision.target_position_m, [0.40, 0.10, 0.50])

    ready = _evidence(
        ee=(0.52, -0.12, 0.51),
        cube=(0.52, -0.12, 0.31),
        grasped=False,
    )
    bridge.update(ready)
    handoff = bridge.update(ready)

    assert handoff.handoff is True
    assert handoff.handoff_event == 1
    assert handoff.handoff_progress == 0.0
    assert handoff.desired_gripper_closed is False
    assert handoff.reason == "regrasp_prepose_handoff"


def test_repeated_cbf_during_recovery_resets_readiness_without_reactivation() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=0.91,
    )
    ready = _evidence(ee=(0.60, 0.0, 0.50))

    first_clear = bridge.update(ready)
    renewed_cbf = bridge.update(
        _evidence(ee=(0.60, 0.0, 0.50), cbf_clear=False)
    )
    bridge.request(
        "place",
        _evidence(ee=(0.60, 0.0, 0.50), cbf_clear=False),
        source_event=6,
        source_progress=0.91,
    )

    assert first_clear.ready_streak == 1
    assert renewed_cbf.active is True
    assert renewed_cbf.handoff is False
    assert renewed_cbf.anchor_ready is False
    assert renewed_cbf.ready_streak == 0
    assert bridge.state.active is True
    assert bridge.state.mode == "place"
    assert bridge.state.activation_count == 1
    assert bridge.state.replan_count == 1

    reconfirming = bridge.update(ready)
    descend = bridge.update(ready)

    assert reconfirming.handoff is False
    assert reconfirming.ready_streak == 1
    assert descend.handoff is False
    assert descend.stage == "place_descend"

    placed = _evidence(
        ee=(0.60, 0.0, 0.315),
        cube=(0.60, 0.0, 0.315),
    )
    bridge.update(placed)
    handoff = bridge.update(placed)
    assert handoff.handoff is True
    assert handoff.handoff_event == 6
    assert handoff.handoff_progress == 1.0


def test_anchor_handoff_requires_cbf_clear_and_low_cube_and_joint_speed() -> None:
    bridge = StateAwareRecoveryBridge(_config(anchor_confirmation_steps=2))
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=0.91,
    )

    transport_ready = _evidence(
        ee=(0.60, 0.0, 0.50),
        cube=(0.60, 0.0, 0.50),
    )
    bridge.update(transport_ready)
    descended = bridge.update(transport_ready)
    assert descended.stage == "place_descend"
    assert descended.handoff is False

    blocked_cases = (
        _evidence(
            ee=(0.60, 0.0, 0.315),
            cube=(0.60, 0.0, 0.315),
            cbf_clear=False,
        ),
        _evidence(
            ee=(0.60, 0.0, 0.315),
            cube=(0.60, 0.0, 0.315),
            cube_speed=0.051,
        ),
        _evidence(
            ee=(0.60, 0.0, 0.315),
            cube=(0.60, 0.0, 0.315),
            joint_speed=0.251,
        ),
        _evidence(
            ee=(0.621, 0.0, 0.315),
            cube=(0.621, 0.0, 0.315),
        ),
        _evidence(
            ee=(0.60, 0.0, 0.326),
            cube=(0.60, 0.0, 0.326),
        ),
    )
    for blocked in blocked_cases:
        decision = bridge.update(blocked)
        assert decision.active is True
        assert decision.anchor_ready is False
        assert decision.ready_streak == 0
        assert decision.handoff is False

    ready = _evidence(
        ee=(0.60, 0.0, 0.315),
        cube=(0.60, 0.0, 0.315),
    )
    bridge.update(ready)
    dropout = bridge.update(
        _evidence(
            ee=(0.60, 0.0, 0.315),
            cube=(0.60, 0.0, 0.315),
            joint_speed=0.251,
        )
    )
    assert dropout.ready_streak == 0
    assert dropout.handoff is False

    bridge.update(ready)
    handoff = bridge.update(ready)
    assert handoff.handoff is True
    assert handoff.handoff_event == 6
    assert handoff.handoff_progress == 1.0


def test_place_descend_dynamically_compensates_current_cube_to_ee_offset() -> None:
    bridge = StateAwareRecoveryBridge(
        _config(
            observed_ee_frame_offset_x_m=0.01,
            observed_ee_frame_offset_y_m=0.04,
            observed_ee_frame_offset_z_m=0.042,
        )
    )
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=1.0,
    )
    transport_ready = _evidence(
        ee=(0.61, 0.04, 0.542),
        cube=(0.60, 0.0, 0.50),
    )

    bridge.update(transport_ready)
    descend = bridge.update(transport_ready)

    desired_cube = np.array([0.60, 0.0, 0.315])
    offset = np.array([0.01, 0.04, 0.042])
    expected_initial_target = (
        np.asarray(transport_ready.ee_position_m)
        + desired_cube
        - np.asarray(transport_ready.cube_position_m)
        - offset
    )
    assert descend.stage == "place_descend"
    assert descend.handoff is False
    assert np.allclose(descend.target_position_m, expected_initial_target)

    displaced = _evidence(
        ee=(0.59, 0.075, 0.390),
        cube=(0.57, 0.025, 0.34),
    )
    replanned = bridge.update(displaced)
    expected_replanned_target = (
        np.asarray(displaced.ee_position_m)
        + desired_cube
        - np.asarray(displaced.cube_position_m)
        - offset
    )

    assert replanned.stage == "place_descend"
    assert replanned.handoff is False
    assert np.allclose(replanned.target_position_m, expected_replanned_target)
    assert not np.allclose(
        replanned.target_position_m,
        descend.target_position_m,
    )


def test_single_frame_grasp_state_change_cannot_replan_confirmed_place_mode() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=0.91,
    )

    transient_loss = bridge.update(
        _evidence(cube=(0.48, -0.08, 0.31), grasped=False)
    )

    assert transient_loss.active is True
    assert transient_loss.mode == "place"
    assert transient_loss.desired_gripper_closed is True
    assert bridge.state.activation_count == 1
    assert bridge.state.replan_count == 0

    old_plan_id = bridge.state.plan_id
    bridge.request(
        "regrasp",
        _evidence(cube=(0.48, -0.08, 0.31), grasped=False),
        source_event=6,
        source_progress=0.91,
    )
    replanned = bridge.update(
        _evidence(cube=(0.48, -0.08, 0.31), grasped=False)
    )
    assert replanned.mode == "regrasp"
    assert replanned.stage == "regrasp_prepose"
    assert replanned.desired_gripper_closed is False
    assert bridge.state.plan_id > old_plan_id
    assert bridge.state.replan_count == 1


def test_success_cancellation_is_absorbing_for_bridge_until_new_request() -> None:
    bridge = StateAwareRecoveryBridge(_config())
    bridge.request(
        "place",
        _evidence(),
        source_event=6,
        source_progress=0.91,
    )

    bridge.cancel_for_success()
    inactive = bridge.update(_evidence(cbf_clear=False))

    assert inactive.active is False
    assert inactive.target_position_m is None
    assert inactive.handoff is False
    assert bridge.state.active is False
    assert bridge.state.mode == "inactive"
    assert bridge.state.stage == "inactive"
    assert bridge.state.last_reason == "terminal_success_cancelled_recovery"
    assert bridge.state.completion_count == 0
