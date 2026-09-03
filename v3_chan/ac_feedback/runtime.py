"""Fail-closed checks and controlled hold helpers for the Isaac runtime."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .schema import ProtocolViolation


EXPECTED_PANDA_JOINT_NAMES = tuple(
    [f"panda_joint{index}" for index in range(1, 8)]
    + ["panda_finger_joint1", "panda_finger_joint2"]
)


def validate_environment_contract(
    env,
    *,
    expected_max_episode_steps: int = 4500,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    config = env.config
    exact = {
        "max_episode_steps": int(expected_max_episode_steps),
        "action_version": "action_v1_controller_target_delta",
        "fixed_orientation": True,
        # The frozen handoff policy emits the fifth (gripper) action.  Requiring
        # event mode here would reject the exact runtime contract before the
        # first trial and would silently change the policy semantics if worked
        # around elsewhere.
        "gripper_mode": "policy",
        "require_release_for_success": True,
        "strict_task_semantics": True,
        "strict_place_xy_tolerance_m": 0.05,
        "reward_version": "reward_v4_post_release_stability_hri_errp",
        "pseudo_errp_enabled": False,
        "synthetic_human_enabled": False,
        "physical_safety_controller": "cbf",
        "cbf_safe_gap_m": 0.05,
        "cbf_activation_gap_m": 0.13,
        "cbf_gamma_per_s": 8.0,
        "cbf_prediction_horizon_s": 0.15,
        "cbf_max_prediction_buffer_m": 0.08,
        "cbf_max_joint_speed_rad_s": 2.0,
    }
    for name, expected in exact.items():
        actual = getattr(config, name, None)
        if isinstance(expected, float):
            valid = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        else:
            valid = actual == expected
        if not valid:
            raise ProtocolViolation(
                f"environment contract mismatch for {name}: expected {expected!r}, got {actual!r}"
            )
    cbf = getattr(env, "_cbf_filter", None)
    if cbf is None:
        raise ProtocolViolation("environment did not construct the CBF filter")
    cbf_config = getattr(cbf, "config", None)
    if cbf_config is None:
        raise ProtocolViolation("CBF configuration is unavailable")
    if not bool(cbf_config.fail_closed_on_invalid_active_hand):
        raise ProtocolViolation("CBF invalid-active-hand fail-stop is disabled")
    if not bool(cbf_config.stop_on_infeasible):
        raise ProtocolViolation("CBF infeasible-solution fail-stop is disabled")
    joint_names = tuple(str(name) for name in env.robot.dof_names)
    if joint_names != EXPECTED_PANDA_JOINT_NAMES:
        raise ProtocolViolation(
            f"unexpected Panda joint order: expected {EXPECTED_PANDA_JOINT_NAMES}, got {joint_names}"
        )
    arm_indices = tuple(range(7))
    return joint_names, arm_indices


def measured_joint_state(robot, *, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    positions = _finite_joint_vector(
        robot.get_joint_positions(), joint_count, "measured joint positions"
    )
    velocities = _finite_joint_vector(
        robot.get_joint_velocities(), joint_count, "measured joint velocities"
    )
    return positions, velocities


def apply_controlled_hold(robot, *, joint_count: int) -> dict[str, np.ndarray]:
    """Apply full current-position and zero-velocity targets, then verify readback."""

    positions, _ = measured_joint_state(robot, joint_count=joint_count)
    velocities = np.zeros(joint_count, dtype=np.float64)
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    robot.apply_action(
        ArticulationAction(
            joint_positions=positions.copy(),
            joint_velocities=velocities.copy(),
        )
    )
    getter = getattr(robot, "get_applied_action", None)
    if not callable(getter):
        raise ProtocolViolation("controlled hold cannot verify applied action")
    applied = getter()
    applied_positions = _finite_joint_vector(
        getattr(applied, "joint_positions", None),
        joint_count,
        "hold applied joint positions",
    )
    applied_velocities = _finite_joint_vector(
        getattr(applied, "joint_velocities", None),
        joint_count,
        "hold applied joint velocities",
    )
    if not np.allclose(applied_positions, positions, rtol=0.0, atol=1e-9):
        raise ProtocolViolation("controlled hold position readback mismatch")
    if not np.allclose(applied_velocities, 0.0, rtol=0.0, atol=1e-12):
        raise ProtocolViolation("controlled hold velocity readback is nonzero")
    return {
        "joint_positions": applied_positions,
        "joint_velocities": applied_velocities,
    }


def _finite_joint_vector(value: Any, size: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ProtocolViolation(f"{label} is not numeric") from error
    if result.shape != (int(size),) or not np.all(np.isfinite(result)):
        raise ProtocolViolation(f"{label} must be a finite ({int(size)},) vector")
    return result.copy()
