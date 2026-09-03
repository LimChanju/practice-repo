from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pytest

from v3_chan.rl.pick_place_env import (
    EXACT_POSE_ROBOT_RESET_CONTRACT,
    _canonicalize_exact_robot_reset,
)
from v3_chan.prepare_task_rlihf_exact_prefix import (
    PrefixPreparationError,
    summarize_runtime_robot_resets,
    validate_runtime_robot_reset_exactness,
)


@dataclass
class _Action:
    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    joint_efforts: np.ndarray | None = None


class _FakeRobot:
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        applied_action: _Action,
        *,
        corrupt_applied_finger_target: bool = False,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.velocities = np.asarray(velocities, dtype=np.float64).copy()
        self.applied_action = applied_action
        self.corrupt_applied_finger_target = corrupt_applied_finger_target

    def get_joint_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.velocities.copy()

    def get_applied_action(self) -> _Action:
        return self.applied_action

    def apply_action(self, action: _Action) -> None:
        positions = (
            None
            if action.joint_positions is None
            else np.asarray(action.joint_positions, dtype=np.float64).copy()
        )
        if self.corrupt_applied_finger_target and positions is not None:
            positions[-1] = -123.0
        self.applied_action = _Action(
            joint_positions=positions,
            joint_velocities=(
                None
                if action.joint_velocities is None
                else np.asarray(action.joint_velocities, dtype=np.float64).copy()
            ),
            joint_efforts=(
                None
                if action.joint_efforts is None
                else np.asarray(action.joint_efforts, dtype=np.float64).copy()
            ),
        )


def _source_state() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [0.012, -0.57, 0.0, -2.81, 0.0, 3.037, 0.741, 0.04, 0.04],
        dtype=np.float64,
    )
    velocities = np.array(
        [-1e-5, -1e-3, 3e-6, 4e-4, 2e-4, 2e-5, -3e-4, 0.0, 0.0],
        dtype=np.float64,
    )
    return positions, velocities


def test_exact_reset_overwrites_partial_arm_and_closed_finger_targets() -> None:
    positions, velocities = _source_state()
    robot = _FakeRobot(
        positions,
        velocities,
        _Action(
            joint_positions=np.array([9.0] * 7, dtype=np.float64),
            joint_velocities=None,
            joint_efforts=None,
        ),
    )

    diagnostics = _canonicalize_exact_robot_reset(robot, positions, velocities)

    assert diagnostics == {
        "contract": EXACT_POSE_ROBOT_RESET_CONTRACT,
        "joint_count": 9,
        "source_joint_count": 9,
        "canonical_full_q_qd_targets_applied": True,
        "joint_effort_target_contract": "unset_not_effort_controlled",
        "measured_joint_positions_exact": True,
        "measured_joint_velocities_exact": True,
        "applied_joint_position_targets_exact": True,
        "applied_joint_velocity_targets_exact": True,
        "passed": True,
        "mismatched_fields": [],
    }
    np.testing.assert_array_equal(robot.applied_action.joint_positions, positions)
    np.testing.assert_array_equal(robot.applied_action.joint_velocities, velocities)
    assert robot.applied_action.joint_positions.shape == (9,)
    assert robot.applied_action.joint_positions[-2:].tolist() == [0.04, 0.04]


def test_exact_reset_is_history_independent_for_full_applied_q_qd() -> None:
    positions, velocities = _source_state()
    partial_arm_history = _FakeRobot(
        positions,
        velocities,
        _Action(joint_positions=np.full(7, 5.0), joint_velocities=None),
    )
    closed_finger_history = _FakeRobot(
        positions,
        velocities,
        _Action(
            joint_positions=np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]
            ),
            joint_velocities=np.full(9, 0.25),
        ),
    )

    first = _canonicalize_exact_robot_reset(
        partial_arm_history, positions, velocities
    )
    second = _canonicalize_exact_robot_reset(
        closed_finger_history, positions, velocities
    )

    assert first == second
    np.testing.assert_array_equal(
        partial_arm_history.applied_action.joint_positions,
        closed_finger_history.applied_action.joint_positions,
    )
    np.testing.assert_array_equal(
        partial_arm_history.applied_action.joint_velocities,
        closed_finger_history.applied_action.joint_velocities,
    )


def test_exact_reset_fails_closed_when_runtime_rejects_full_finger_target() -> None:
    positions, velocities = _source_state()
    robot = _FakeRobot(
        positions,
        velocities,
        _Action(joint_positions=np.zeros(9), joint_velocities=np.zeros(9)),
        corrupt_applied_finger_target=True,
    )

    with pytest.raises(
        RuntimeError,
        match="applied_joint_position_targets",
    ):
        _canonicalize_exact_robot_reset(robot, positions, velocities)


def test_exact_pose_reset_canonicalizes_targets_before_reset_observation() -> None:
    from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv

    source = inspect.getsource(IsaacPickPlaceEnv.reset)
    measured_restore = source.index("robot_restored = _set_robot_joint_state(")
    target_reset = source.index("_canonicalize_exact_robot_reset(")
    observation = source.index("obs = self._build_obs()")
    assert measured_restore < target_reset < observation


def test_prefix_preparer_requires_and_summarizes_reset_readback_contract() -> None:
    positions, velocities = _source_state()
    robot = _FakeRobot(
        positions,
        velocities,
        _Action(joint_positions=np.zeros(9), joint_velocities=np.zeros(9)),
    )
    reset = _canonicalize_exact_robot_reset(robot, positions, velocities)
    info = {
        "source_restoration": {
            "restoration_mode": "exact_pose",
            "robot_reset_exactness": reset,
        }
    }
    validated = validate_runtime_robot_reset_exactness(info)
    summary = summarize_runtime_robot_resets(
        (validated, validated, validated), minimum_count=3
    )
    assert summary["passed"] is True
    assert summary["validated_reset_count"] == 3
    assert summary["all_measured_joint_positions_exact"] is True
    assert summary["all_applied_joint_position_targets_exact"] is True
    assert summary["all_applied_joint_velocity_targets_exact"] is True

    drifted = {**reset, "applied_joint_position_targets_exact": False}
    with pytest.raises(PrefixPreparationError, match="contract drift"):
        validate_runtime_robot_reset_exactness(
            {"source_restoration": {"robot_reset_exactness": drifted}}
        )
