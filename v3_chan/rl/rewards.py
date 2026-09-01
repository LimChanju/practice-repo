from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_REWARD_VERSION = "reward_v0_hri_errp"
PLACEMENT_REWARD_VERSION = "reward_v1_placement_hri_errp"
GRASP_STABILITY_REWARD_VERSION = "reward_v2_grasp_stability_hri_errp"
RELEASE_PRECISION_REWARD_VERSION = "reward_v3_release_precision_hri_errp"
REWARD_VERSION = "reward_v4_post_release_stability_hri_errp"
MINIMAL_EVENT_POTENTIAL_REWARD_VERSION = "reward_v5_minimal_event_potential"
SUCCESS_ONLY_REWARD_VERSION = "reward_v6_strict_success_only"
ISAAC_FRANKA_DENSE_REWARD_VERSION = (
    "reward_v7_isaac_franka_cube_stack_adapted_dense"
)
SUPPORTED_REWARD_VERSIONS = {
    REWARD_VERSION,
    MINIMAL_EVENT_POTENTIAL_REWARD_VERSION,
    SUCCESS_ONLY_REWARD_VERSION,
    ISAAC_FRANKA_DENSE_REWARD_VERSION,
}


@dataclass(frozen=True)
class RewardWeights:
    ee_to_cube_progress: float = 2.0
    cube_to_target_progress: float = 1.5
    carrying_cube_to_target_progress: float = 7.0
    grasp_phase_distance_penalty: float = 0.35
    missed_grasp_transition_penalty: float = 2.0
    grasp_phase_target_dist: float = 0.058
    grasp_bonus: float = 0.03
    target_zone_bonus: float = 0.15
    success_bonus: float = 15.0
    placement_error_penalty: float = 0.08
    near_target_dist: float = 0.065
    near_target_progress_bonus: float = 0.8
    near_target_hold_bonus: float = 0.05
    near_target_regression_penalty: float = 2.0
    near_target_exit_penalty: float = 0.8
    release_inside_target_bonus: float = 2.0
    release_outside_target_penalty: float = 6.0
    post_release_hold_bonus: float = 0.08
    post_release_error_penalty: float = 0.6
    post_release_regression_penalty: float = 1.5
    action_penalty: float = 0.01
    near_human_penalty: float = 0.5
    human_collision_penalty: float = 5.0
    errp_penalty: float = 2.0


@dataclass(frozen=True)
class MinimalEventPotentialRewardWeights:
    approach_progress: float = 2.0
    transport_progress: float = 7.0
    new_grasp_bonus: float = 2.0
    valid_release_bonus: float = 5.0
    stable_success_bonus: float = 15.0
    missed_grasp_penalty: float = 2.0
    invalid_release_penalty: float = 6.0
    action_penalty: float = 0.01


@dataclass(frozen=True)
class IsaacFrankaDenseRewardWeights:
    """Weights from Isaac Gym's FrankaCubeStack dense reward.

    The source task stacks cube A on cube B.  The single-pick adaptation in
    this repository instead treats ``place_target_pos`` as the desired cube
    center and uses that target's z coordinate as the resting-height
    reference.  All gates remain physical-state gates; the scripted
    controller event and clock are deliberately excluded.
    """

    distance_reward_scale: float = 0.1
    lift_reward_scale: float = 1.5
    align_reward_scale: float = 2.0
    success_reward: float = 16.0
    distance_tanh_gain: float = 10.0
    lift_clearance_m: float = 0.04


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


DEFAULT_REWARD_WEIGHTS = RewardWeights()
DEFAULT_MINIMAL_REWARD_WEIGHTS = MinimalEventPotentialRewardWeights()
DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS = IsaacFrankaDenseRewardWeights()

_ISAAC_FRANKA_LEFT_FINGER_TO_CUBE_FIELD = (
    "_reward_isaac_franka_left_finger_to_cube"
)
_ISAAC_FRANKA_RIGHT_FINGER_TO_CUBE_FIELD = (
    "_reward_isaac_franka_right_finger_to_cube"
)
_ISAAC_FRANKA_DENSE_COMPONENT_NAMES = (
    "v7_isaac_distance_reward",
    "v7_isaac_lift_reward",
    "v7_isaac_align_reward",
    "v7_isaac_success_override",
)


def _compute_reward_v4(
    prev_obs: dict[str, np.ndarray] | None,
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    errp_feedback: float = 0.0,
    success: bool = False,
    success_dist: float = 0.06,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardResult:
    """Compute reward v4 from consecutive observations."""

    ee_cube_dist = _norm_field(obs, "ee_to_cube")
    cube_target_dist = _norm_field(obs, "cube_to_place_target")
    if prev_obs is None:
        ee_cube_progress = 0.0
        cube_target_progress = 0.0
        prev_has_grasped = 0.0
        prev_event = -1
    else:
        ee_cube_progress = _norm_field(prev_obs, "ee_to_cube") - ee_cube_dist
        cube_target_progress = _norm_field(prev_obs, "cube_to_place_target") - cube_target_dist
        prev_has_grasped = _scalar_field(prev_obs, "has_grasped_cube")
        prev_event = _controller_event(prev_obs)

    has_grasped = _scalar_field(obs, "has_grasped_cube")
    event = _controller_event(obs)
    post_grasp_phase = float(max(event, prev_event) >= 4)
    grasp_phase = float(event in (1, 2, 3) and has_grasped <= 0.5)
    placement_active = max(has_grasped, prev_has_grasped, post_grasp_phase)
    normalized_grasp_error = max(0.0, ee_cube_dist - weights.grasp_phase_target_dist) / max(
        weights.grasp_phase_target_dist,
        1e-6,
    )
    normalized_target_error = max(0.0, cube_target_dist - float(success_dist)) / max(float(success_dist), 1e-6)
    target_zone = placement_active * float(cube_target_dist <= float(success_dist))
    near_target_dist = max(float(weights.near_target_dist), float(success_dist))
    placement_precision_phase = placement_active * float(event in (5, 6, 7) or prev_event in (5, 6, 7))
    near_target_band = float(cube_target_dist <= near_target_dist)
    if prev_obs is None:
        was_near_target_band = 0.0
    else:
        was_near_target_band = float(_norm_field(prev_obs, "cube_to_place_target") <= near_target_dist)
    near_target_active = placement_precision_phase * max(near_target_band, was_near_target_band)
    near_target_hold = placement_precision_phase * near_target_band
    near_target_progress = max(0.0, cube_target_progress) / max(float(success_dist), 1e-6)
    near_target_regression = max(0.0, -cube_target_progress) / max(float(success_dist), 1e-6)
    exited_near_target_band = placement_precision_phase * float(
        was_near_target_band > 0.5 and near_target_band <= 0.5
    )
    entered_post_grasp_without_grasp = float(
        prev_obs is not None
        and prev_event <= 3
        and event >= 4
        and max(has_grasped, prev_has_grasped) <= 0.5
    )
    release_after_pick = (
        prev_obs is not None
        and prev_has_grasped > 0.5
        and has_grasped <= 0.5
        and max(event, prev_event) >= 7
    )
    release_inside_target = float(release_after_pick and cube_target_dist <= float(success_dist))
    release_outside_target = float(release_after_pick and cube_target_dist > float(success_dist))
    post_release_phase = placement_active * float(max(event, prev_event) >= 7 and has_grasped <= 0.5)
    post_release_inside_target = post_release_phase * float(cube_target_dist <= float(success_dist))
    post_release_outside_error = post_release_phase * normalized_target_error
    post_release_regression = post_release_phase * near_target_regression
    near_human = _scalar_field(obs, "near_human")
    human_collision = _scalar_field(obs, "human_robot_collision")
    action_norm = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1)))
    errp_feedback = float(np.clip(errp_feedback, 0.0, 1.0))

    components = {
        "ee_to_cube_progress": weights.ee_to_cube_progress * ee_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress * cube_target_progress,
        "carrying_cube_to_target_progress": (
            weights.carrying_cube_to_target_progress * placement_active * cube_target_progress
        ),
        "grasp_phase_distance_penalty": (
            -weights.grasp_phase_distance_penalty * grasp_phase * normalized_grasp_error
        ),
        "missed_grasp_transition_penalty": (
            -weights.missed_grasp_transition_penalty * entered_post_grasp_without_grasp
        ),
        "grasp_bonus": weights.grasp_bonus * has_grasped,
        "target_zone_bonus": weights.target_zone_bonus * target_zone,
        "success_bonus": weights.success_bonus * (1.0 if success else 0.0),
        "placement_error_penalty": -weights.placement_error_penalty * placement_active * normalized_target_error,
        "near_target_progress_bonus": (
            weights.near_target_progress_bonus * near_target_active * near_target_progress
        ),
        "near_target_hold_bonus": weights.near_target_hold_bonus * near_target_hold,
        "near_target_regression_penalty": (
            -weights.near_target_regression_penalty * near_target_active * near_target_regression
        ),
        "near_target_exit_penalty": -weights.near_target_exit_penalty * exited_near_target_band,
        "release_inside_target_bonus": weights.release_inside_target_bonus * release_inside_target,
        "release_outside_target_penalty": -weights.release_outside_target_penalty * release_outside_target,
        "post_release_hold_bonus": weights.post_release_hold_bonus * post_release_inside_target,
        "post_release_error_penalty": -weights.post_release_error_penalty * post_release_outside_error,
        "post_release_regression_penalty": (
            -weights.post_release_regression_penalty * post_release_regression
        ),
        "action_penalty": -weights.action_penalty * action_norm,
        "near_human_penalty": -weights.near_human_penalty * near_human,
        "human_collision_penalty": -weights.human_collision_penalty * human_collision,
        "errp_penalty": -weights.errp_penalty * errp_feedback,
    }
    return RewardResult(total=float(sum(components.values())), components=components)


def compute_minimal_event_potential_reward(
    prev_obs: dict[str, np.ndarray] | None,
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    success: bool = False,
    success_dist: float = 0.06,
    weights: MinimalEventPotentialRewardWeights = DEFAULT_MINIMAL_REWARD_WEIGHTS,
) -> RewardResult:
    """Minimal pick/place reward with transition-locked event bonuses."""

    ee_cube_dist = _norm_field(obs, "ee_to_cube")
    cube_target_dist = _norm_field(obs, "cube_to_place_target")
    has_grasped = _scalar_field(obs, "has_grasped_cube") > 0.5
    event = _controller_event(obs)
    if prev_obs is None:
        prev_has_grasped = has_grasped
        prev_event = event
        ee_progress = 0.0
        target_progress = 0.0
    else:
        prev_has_grasped = _scalar_field(prev_obs, "has_grasped_cube") > 0.5
        prev_event = _controller_event(prev_obs)
        ee_progress = _norm_field(prev_obs, "ee_to_cube") - ee_cube_dist
        target_progress = (
            _norm_field(prev_obs, "cube_to_place_target") - cube_target_dist
        )

    same_approach_regime = bool(
        prev_obs is not None
        and not prev_has_grasped
        and not has_grasped
        and max(prev_event, event) <= 3
    )
    same_transport_regime = bool(
        prev_obs is not None and prev_has_grasped and has_grasped
    )
    phase_progress = 0.0
    if same_approach_regime:
        phase_progress = float(weights.approach_progress) * ee_progress
    elif same_transport_regime:
        phase_progress = float(weights.transport_progress) * target_progress

    new_grasp = bool(prev_obs is not None and not prev_has_grasped and has_grasped)
    released = bool(prev_obs is not None and prev_has_grasped and not has_grasped)
    release_phase = bool(max(prev_event, event) >= 7)
    valid_release = bool(
        released and release_phase and cube_target_dist <= float(success_dist)
    )
    invalid_release = bool(
        released and cube_target_dist > float(success_dist)
    )
    missed_grasp = bool(
        prev_obs is not None
        and prev_event <= 3
        and event >= 4
        and not prev_has_grasped
        and not has_grasped
    )
    action_squared_norm = float(
        np.sum(np.asarray(action, dtype=np.float64).reshape(-1) ** 2)
    )
    components = {
        "v5_phase_progress": float(phase_progress),
        "v5_new_grasp_bonus": float(weights.new_grasp_bonus) * float(new_grasp),
        "v5_valid_release_bonus": (
            float(weights.valid_release_bonus) * float(valid_release)
        ),
        "v5_stable_success_bonus": (
            float(weights.stable_success_bonus) * float(bool(success))
        ),
        "v5_missed_grasp_penalty": (
            -float(weights.missed_grasp_penalty) * float(missed_grasp)
        ),
        "v5_invalid_release_penalty": (
            -float(weights.invalid_release_penalty) * float(invalid_release)
        ),
        "v5_action_penalty": (
            -float(weights.action_penalty) * action_squared_norm
        ),
    }
    return RewardResult(total=float(sum(components.values())), components=components)


def compute_strict_success_only_reward(*, success: bool = False) -> RewardResult:
    """Return +1 only for strict released-cube task success.

    The environment owns the definition of ``success``.  For Task-RLIHF runs
    it must be configured with ``require_release_for_success=True``.  Collision,
    distance, progress, action magnitude, and neural feedback are deliberately
    absent; delayed synthetic feedback is added to the PPO buffer by the
    trainer as a separate, auditable signal.
    """

    value = 1.0 if bool(success) else 0.0
    return RewardResult(
        total=value,
        components={"v6_strict_success_only": value},
    )


def compute_isaac_franka_dense_reward(
    obs: dict[str, np.ndarray],
    *,
    success: bool = False,
    weights: IsaacFrankaDenseRewardWeights = (
        DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS
    ),
) -> RewardResult:
    """Adapt Isaac Gym's ``FrankaCubeStack`` state reward to table placement.

    Isaac Gym averages cube distance to the grip site and both fingertips,
    gates target alignment on a physical lift, takes ``max(reach, align)``,
    and replaces all dense terms with the terminal reward on success.  This
    implementation preserves that composition.  The fixed marker replaces
    cube B, so its position is the desired final cube-center position and its
    z coordinate is also the resting-height reference.  Isaac Sim exposes the
    two finger-link origins used here as the available fingertip proxies.

    The two underscore-prefixed fingertip vectors are reward-only runtime
    fields added by ``IsaacPickPlaceEnv``.  Unit-test and offline callers that
    only have the frozen policy observation fall back to the grip-site proxy
    ``ee_to_cube`` rather than changing the 84-D observation contract.
    """

    ee_distance = _norm_field(obs, "ee_to_cube")
    finger_fields = (
        _ISAAC_FRANKA_LEFT_FINGER_TO_CUBE_FIELD,
        _ISAAC_FRANKA_RIGHT_FINGER_TO_CUBE_FIELD,
    )
    if all(name in obs for name in finger_fields):
        mean_reach_distance = float(
            (
                ee_distance
                + _norm_field(obs, finger_fields[0])
                + _norm_field(obs, finger_fields[1])
            )
            / 3.0
        )
    else:
        mean_reach_distance = ee_distance

    gain = float(weights.distance_tanh_gain)
    if not np.isfinite(mean_reach_distance) or not np.isfinite(gain) or gain < 0.0:
        raise ValueError("Isaac Franka dense reward requires finite distances/gain")
    reach_reward = float(1.0 - np.tanh(gain * mean_reach_distance))

    cube_position = np.asarray(obs["cube_pos"], dtype=float).reshape(-1)
    target_position = np.asarray(obs["place_target_pos"], dtype=float).reshape(-1)
    if cube_position.size < 3 or target_position.size < 3:
        raise ValueError("Isaac Franka dense reward requires 3-D cube/target poses")
    if not (
        np.all(np.isfinite(cube_position[:3]))
        and np.all(np.isfinite(target_position[:3]))
    ):
        raise ValueError("Isaac Franka dense reward requires finite cube/target poses")

    cube_clearance = float(cube_position[2] - target_position[2])
    lifted = float(cube_clearance > float(weights.lift_clearance_m))
    target_distance = _norm_field(obs, "cube_to_place_target")
    if not np.isfinite(target_distance):
        raise ValueError("Isaac Franka dense reward requires a finite target distance")
    align_reward = float(
        lifted * (1.0 - np.tanh(gain * target_distance))
    )
    distance_reward = max(reach_reward, align_reward)

    if bool(success):
        components = {
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[0]: 0.0,
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[1]: 0.0,
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[2]: 0.0,
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[3]: float(
                weights.success_reward
            ),
        }
    else:
        components = {
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[0]: (
                float(weights.distance_reward_scale) * distance_reward
            ),
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[1]: (
                float(weights.lift_reward_scale) * lifted
            ),
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[2]: (
                float(weights.align_reward_scale) * align_reward
            ),
            _ISAAC_FRANKA_DENSE_COMPONENT_NAMES[3]: 0.0,
        }
    return RewardResult(total=float(sum(components.values())), components=components)


def compute_reward(
    prev_obs: dict[str, np.ndarray] | None,
    obs: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    errp_feedback: float = 0.0,
    success: bool = False,
    success_dist: float = 0.06,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
    reward_version: str = REWARD_VERSION,
    minimal_weights: MinimalEventPotentialRewardWeights = (
        DEFAULT_MINIMAL_REWARD_WEIGHTS
    ),
    isaac_franka_dense_weights: IsaacFrankaDenseRewardWeights = (
        DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS
    ),
) -> RewardResult:
    """Dispatch a versioned reward while preserving v4 as the default."""

    if reward_version == REWARD_VERSION:
        return _compute_reward_v4(
            prev_obs,
            obs,
            action,
            errp_feedback=errp_feedback,
            success=success,
            success_dist=success_dist,
            weights=weights,
        )
    if reward_version == MINIMAL_EVENT_POTENTIAL_REWARD_VERSION:
        if float(errp_feedback) != 0.0:
            raise ValueError("The v5 task reward must not receive ErrP feedback")
        return compute_minimal_event_potential_reward(
            prev_obs,
            obs,
            action,
            success=success,
            success_dist=success_dist,
            weights=minimal_weights,
        )
    if reward_version == SUCCESS_ONLY_REWARD_VERSION:
        if float(errp_feedback) != 0.0:
            raise ValueError(
                "The v6 success-only base reward must receive neural feedback "
                "only through the external delayed-credit trainer"
            )
        return compute_strict_success_only_reward(success=success)
    if reward_version == ISAAC_FRANKA_DENSE_REWARD_VERSION:
        if float(errp_feedback) != 0.0:
            raise ValueError(
                "The v7 Isaac Franka task reward must not receive ErrP feedback"
            )
        return compute_isaac_franka_dense_reward(
            obs,
            success=success,
            weights=isaac_franka_dense_weights,
        )
    raise ValueError(f"Unsupported reward version: {reward_version}")


def is_success(obs: dict[str, np.ndarray], threshold_m: float = 0.06) -> bool:
    """Simple success: active cube is close enough to the place target."""

    return _norm_field(obs, "cube_to_place_target") <= float(threshold_m)


def reward_component_names(
    reward_version: str = REWARD_VERSION,
) -> tuple[str, ...]:
    if reward_version == ISAAC_FRANKA_DENSE_REWARD_VERSION:
        return _ISAAC_FRANKA_DENSE_COMPONENT_NAMES
    if reward_version == SUCCESS_ONLY_REWARD_VERSION:
        return ("v6_strict_success_only",)
    if reward_version == MINIMAL_EVENT_POTENTIAL_REWARD_VERSION:
        return (
            "v5_phase_progress",
            "v5_new_grasp_bonus",
            "v5_valid_release_bonus",
            "v5_stable_success_bonus",
            "v5_missed_grasp_penalty",
            "v5_invalid_release_penalty",
            "v5_action_penalty",
        )
    if reward_version != REWARD_VERSION:
        raise ValueError(f"Unsupported reward version: {reward_version}")
    return (
        "ee_to_cube_progress",
        "cube_to_target_progress",
        "carrying_cube_to_target_progress",
        "grasp_phase_distance_penalty",
        "missed_grasp_transition_penalty",
        "grasp_bonus",
        "target_zone_bonus",
        "success_bonus",
        "placement_error_penalty",
        "near_target_progress_bonus",
        "near_target_hold_bonus",
        "near_target_regression_penalty",
        "near_target_exit_penalty",
        "release_inside_target_bonus",
        "release_outside_target_penalty",
        "post_release_hold_bonus",
        "post_release_error_penalty",
        "post_release_regression_penalty",
        "action_penalty",
        "near_human_penalty",
        "human_collision_penalty",
        "errp_penalty",
    )


def reward_weights_dict(weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> dict[str, float]:
    return {
        "ee_to_cube_progress": weights.ee_to_cube_progress,
        "cube_to_target_progress": weights.cube_to_target_progress,
        "carrying_cube_to_target_progress": weights.carrying_cube_to_target_progress,
        "grasp_phase_distance_penalty": weights.grasp_phase_distance_penalty,
        "missed_grasp_transition_penalty": weights.missed_grasp_transition_penalty,
        "grasp_phase_target_dist": weights.grasp_phase_target_dist,
        "grasp_bonus": weights.grasp_bonus,
        "target_zone_bonus": weights.target_zone_bonus,
        "success_bonus": weights.success_bonus,
        "placement_error_penalty": weights.placement_error_penalty,
        "near_target_dist": weights.near_target_dist,
        "near_target_progress_bonus": weights.near_target_progress_bonus,
        "near_target_hold_bonus": weights.near_target_hold_bonus,
        "near_target_regression_penalty": weights.near_target_regression_penalty,
        "near_target_exit_penalty": weights.near_target_exit_penalty,
        "release_inside_target_bonus": weights.release_inside_target_bonus,
        "release_outside_target_penalty": weights.release_outside_target_penalty,
        "post_release_hold_bonus": weights.post_release_hold_bonus,
        "post_release_error_penalty": weights.post_release_error_penalty,
        "post_release_regression_penalty": weights.post_release_regression_penalty,
        "action_penalty": weights.action_penalty,
        "near_human_penalty": weights.near_human_penalty,
        "human_collision_penalty": weights.human_collision_penalty,
        "errp_penalty": weights.errp_penalty,
    }


def minimal_reward_weights_dict(
    weights: MinimalEventPotentialRewardWeights = DEFAULT_MINIMAL_REWARD_WEIGHTS,
) -> dict[str, float]:
    return {
        "approach_progress": float(weights.approach_progress),
        "transport_progress": float(weights.transport_progress),
        "new_grasp_bonus": float(weights.new_grasp_bonus),
        "valid_release_bonus": float(weights.valid_release_bonus),
        "stable_success_bonus": float(weights.stable_success_bonus),
        "missed_grasp_penalty": float(weights.missed_grasp_penalty),
        "invalid_release_penalty": float(weights.invalid_release_penalty),
        "action_penalty": float(weights.action_penalty),
    }


def isaac_franka_dense_reward_weights_dict(
    weights: IsaacFrankaDenseRewardWeights = (
        DEFAULT_ISAAC_FRANKA_DENSE_REWARD_WEIGHTS
    ),
) -> dict[str, float]:
    return {
        "distance_reward_scale": float(weights.distance_reward_scale),
        "lift_reward_scale": float(weights.lift_reward_scale),
        "align_reward_scale": float(weights.align_reward_scale),
        "success_reward": float(weights.success_reward),
        "distance_tanh_gain": float(weights.distance_tanh_gain),
        "lift_clearance_m": float(weights.lift_clearance_m),
    }


def _norm_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.linalg.norm(np.asarray(obs[name], dtype=float).reshape(-1)))


def _scalar_field(obs: dict[str, np.ndarray], name: str) -> float:
    return float(np.asarray(obs[name], dtype=float).reshape(-1)[0])


def _controller_event(obs: dict[str, np.ndarray]) -> int:
    value = obs.get("controller_event")
    if value is None:
        return -1
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return -1
    return int(np.argmax(arr))
