from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - Isaac runtime always has torch.
    torch = None
    nn = None

from .actions import ACTION_DIM, clip_action


BACKUP_ACTION_DIM = 3
BACKUP_ACTION_VERSION = "backup_xyz_replacement_v1"
BACKUP_XYZYAW_ACTION_DIM = 4
BACKUP_XYZYAW_ACTION_VERSION = "backup_xyz_yaw_replacement_v2"
RISK_ESTIMATOR_VERSION = "state_action_risk_v2_normalized"
SAFEMOTIONS_REFERENCE_COMMIT = "5558525a9bd63ff208bbf07970d31617bb27062c"
SAFEMOTIONS_PHASE_EXECUTION_CONTRACT = "task_phase_paused_during_backup_v1"
BackupRewardMode = Literal[
    "physical",
    "raw_errp",
    "uq_errp",
    "raw_errp_replace",
    "uq_errp_replace",
    "physical_plus_raw_errp",
    "physical_plus_uq_errp",
]
BackupTrainingContract = Literal[
    "legacy_gate_intervention",
    "safemotions_backup_episode",
]
BackupStartMode = Literal["encounter", "gate"]
BackupStartSource = Literal[
    "source_encounter",
    "encounter_jitter",
    "task_manifold",
]
RiskLabelMode = Literal["conservative_clearance", "collision_termination"]
RiskCandidateMode = Literal["task", "random", "mixed"]


@dataclass(frozen=True)
class ActionReplacementResult:
    executed_action: np.ndarray
    task_action: np.ndarray
    backup_action: np.ndarray
    intervened: bool
    replacement_norm: float


@dataclass(frozen=True)
class DecisionTimebase:
    """Shared physics, policy-decision, and finite-horizon contract.

    The decision interval must contain an integer number of physics steps and
    the horizon must contain an integer number of decisions. Enforcing that
    relationship prevents training, branch rollout, and evaluation from using
    subtly different definitions of a two-second safety horizon.
    """

    physics_dt_s: float
    decision_dt_s: float
    horizon_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("physics_dt_s", self.physics_dt_s),
            ("decision_dt_s", self.decision_dt_s),
            ("horizon_s", self.horizon_s),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.decision_dt_s) < float(self.physics_dt_s):
            raise ValueError("decision_dt_s must not be smaller than physics_dt_s")
        _integer_ratio(
            float(self.decision_dt_s) / float(self.physics_dt_s),
            name="decision_dt_s / physics_dt_s",
        )
        _integer_ratio(
            float(self.horizon_s) / float(self.decision_dt_s),
            name="horizon_s / decision_dt_s",
        )

    @property
    def action_repeat_steps(self) -> int:
        return _integer_ratio(
            float(self.decision_dt_s) / float(self.physics_dt_s),
            name="decision_dt_s / physics_dt_s",
        )

    @property
    def horizon_decisions(self) -> int:
        return _integer_ratio(
            float(self.horizon_s) / float(self.decision_dt_s),
            name="horizon_s / decision_dt_s",
        )

    @property
    def horizon_physics_steps(self) -> int:
        return int(self.action_repeat_steps * self.horizon_decisions)


@dataclass(frozen=True)
class BackupStartDistribution:
    """Mixture over valid sources for short backup-policy episodes."""

    source_encounter: float = 0.50
    encounter_jitter: float = 0.30
    task_manifold: float = 0.20

    def __post_init__(self) -> None:
        values = self.probabilities
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("backup start probabilities must be finite and non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one backup start probability must be positive")

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (
            float(self.source_encounter),
            float(self.encounter_jitter),
            float(self.task_manifold),
        )


@dataclass(frozen=True)
class BackupActionLimitConfig:
    """Finite-difference limits in normalized backup-command coordinates.

    Physical Cartesian limits remain the responsibility of the controller.
    These bounds stop the learned command from changing faster than the
    downstream controller can follow and make acceleration/jerk ablations
    explicit instead of hiding them inside a scalar action multiplier.
    """

    absolute_limit: float = 1.0
    max_delta_per_decision: float = 0.35
    max_second_delta_per_decision: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("absolute_limit", self.absolute_limit),
            ("max_delta_per_decision", self.max_delta_per_decision),
            (
                "max_second_delta_per_decision",
                self.max_second_delta_per_decision,
            ),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.absolute_limit) > 1.0:
            raise ValueError("absolute_limit must not exceed normalized action bound 1")


class BackupActionLimiter:
    """Stateful acceleration/jerk limiter for one backup-policy stream."""

    def __init__(
        self,
        action_dim: int,
        config: BackupActionLimitConfig = BackupActionLimitConfig(),
    ) -> None:
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        self.action_dim = int(action_dim)
        self.config = config
        self.reset()

    def reset(self, action: np.ndarray | Sequence[float] | None = None) -> None:
        initial = (
            np.zeros(self.action_dim, dtype=np.float32)
            if action is None
            else _finite_vector(action, name="initial_action", min_size=self.action_dim)[
                : self.action_dim
            ].astype(np.float32, copy=True)
        )
        initial = np.clip(
            initial,
            -float(self.config.absolute_limit),
            float(self.config.absolute_limit),
        ).astype(np.float32, copy=False)
        self.previous_action = initial.copy()
        self.previous_previous_action = initial.copy()

    def apply(
        self,
        candidate: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        proposed = _finite_vector(
            candidate,
            name="candidate_backup_action",
            min_size=self.action_dim,
        )[: self.action_dim]
        proposed = np.clip(
            proposed,
            -float(self.config.absolute_limit),
            float(self.config.absolute_limit),
        )
        previous_delta = self.previous_action - self.previous_previous_action
        max_delta = float(self.config.max_delta_per_decision)
        max_second_delta = float(self.config.max_second_delta_per_decision)
        lower = np.maximum(-max_delta, previous_delta - max_second_delta)
        upper = np.minimum(max_delta, previous_delta + max_second_delta)
        next_delta = np.clip(proposed - self.previous_action, lower, upper)
        limited = np.clip(
            self.previous_action + next_delta,
            -float(self.config.absolute_limit),
            float(self.config.absolute_limit),
        ).astype(np.float32)
        self.previous_previous_action = self.previous_action.copy()
        self.previous_action = limited.copy()
        return limited

    @property
    def previous_delta(self) -> np.ndarray:
        return (self.previous_action - self.previous_previous_action).astype(
            np.float32, copy=True
        )

    def observation_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the controller state required by the Markov observation."""

        return self.previous_action.copy(), self.previous_delta

    def capture_state(self) -> dict[str, np.ndarray]:
        return {
            "previous_action": self.previous_action.copy(),
            "previous_previous_action": self.previous_previous_action.copy(),
        }

    def restore_state(self, state: Mapping[str, np.ndarray]) -> None:
        previous = _finite_vector(
            state["previous_action"],
            name="previous_action",
            min_size=self.action_dim,
        )[: self.action_dim]
        previous_previous = _finite_vector(
            state["previous_previous_action"],
            name="previous_previous_action",
            min_size=self.action_dim,
        )[: self.action_dim]
        self.previous_action = previous.astype(np.float32, copy=True)
        self.previous_previous_action = previous_previous.astype(
            np.float32, copy=True
        )


@dataclass(frozen=True)
class InterventionHandoffConfig:
    """Temporal contract for switching between task and backup policies."""

    enter_threshold: float
    exit_threshold: float
    minimum_hold_steps: int = 8
    release_confirmation_steps: int = 4

    def __post_init__(self) -> None:
        if not np.isfinite(self.enter_threshold) or not 0.0 <= self.enter_threshold <= 1.0:
            raise ValueError("enter_threshold must be finite and in [0, 1]")
        if not np.isfinite(self.exit_threshold) or not 0.0 <= self.exit_threshold <= 1.0:
            raise ValueError("exit_threshold must be finite and in [0, 1]")
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must not exceed enter_threshold")
        if int(self.minimum_hold_steps) <= 0:
            raise ValueError("minimum_hold_steps must be positive")
        if int(self.release_confirmation_steps) <= 0:
            raise ValueError("release_confirmation_steps must be positive")


@dataclass(frozen=True)
class InterventionHandoffDecision:
    """One-step decision; release occurs after the current backup action."""

    intervene: bool
    raw_request: bool
    activated: bool
    release_after_step: bool
    active_after_step: bool
    active_steps: int
    clear_steps: int
    score: float


class InterventionHandoffController:
    """Latch noisy risk decisions and provide an explicit task re-entry edge.

    A release decision still executes the backup action for the current step.
    The caller can therefore reset/synchronize the task phase after that motion,
    and resume the task policy from a consistent observation on the next step.
    """

    def __init__(self, config: InterventionHandoffConfig) -> None:
        self.config = config
        self.reset()

    @property
    def active(self) -> bool:
        return bool(self._active)

    def reset(self) -> None:
        self._active = False
        self._active_steps = 0
        self._clear_steps = 0

    def update(
        self,
        score: float,
        *,
        geometry_valid: bool = True,
        counter_tick: bool = True,
    ) -> InterventionHandoffDecision:
        finite_score = float(score) if np.isfinite(score) else 0.0
        finite_score = float(np.clip(finite_score, 0.0, 1.0))
        raw_request = bool(
            geometry_valid and finite_score >= float(self.config.enter_threshold)
        )

        if not self._active:
            if not raw_request:
                return InterventionHandoffDecision(
                    intervene=False,
                    raw_request=False,
                    activated=False,
                    release_after_step=False,
                    active_after_step=False,
                    active_steps=0,
                    clear_steps=0,
                    score=finite_score,
                )
            self._active = True
            self._active_steps = int(bool(counter_tick))
            self._clear_steps = 0
            return InterventionHandoffDecision(
                intervene=True,
                raw_request=True,
                activated=True,
                release_after_step=False,
                active_after_step=True,
                active_steps=self._active_steps,
                clear_steps=self._clear_steps,
                score=finite_score,
            )

        if not bool(counter_tick):
            return InterventionHandoffDecision(
                intervene=True,
                raw_request=raw_request,
                activated=False,
                release_after_step=False,
                active_after_step=True,
                active_steps=int(self._active_steps),
                clear_steps=int(self._clear_steps),
                score=finite_score,
            )

        self._active_steps += 1
        exit_ready = bool(
            not geometry_valid
            or finite_score <= float(self.config.exit_threshold)
        )
        self._clear_steps = self._clear_steps + 1 if exit_ready else 0
        release = bool(
            self._active_steps >= int(self.config.minimum_hold_steps)
            and self._clear_steps
            >= int(self.config.release_confirmation_steps)
        )
        active_steps = int(self._active_steps)
        clear_steps = int(self._clear_steps)
        if release:
            self._active = False
            self._active_steps = 0
            self._clear_steps = 0
        return InterventionHandoffDecision(
            intervene=True,
            raw_request=raw_request,
            activated=False,
            release_after_step=release,
            active_after_step=bool(self._active),
            active_steps=active_steps,
            clear_steps=clear_steps,
            score=finite_score,
        )


@dataclass(frozen=True)
class HandoffBlendDecision:
    """One command from a finite decision-rate backup-to-task blend."""

    action: np.ndarray
    backup_weight: float
    active: bool
    finished_after_decision: bool
    decision_index: int


class HandoffActionBlender:
    """Hold and blend commands on the same decision clock as the backup policy."""

    def __init__(self, *, action_dim: int, blend_decisions: int = 2) -> None:
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if int(blend_decisions) <= 0:
            raise ValueError("blend_decisions must be positive")
        self.action_dim = int(action_dim)
        self.blend_decisions = int(blend_decisions)
        self.reset()

    @property
    def active(self) -> bool:
        return bool(self._active)

    def reset(self) -> None:
        self._active = False
        self._decision_index = 0
        self._anchor = np.zeros(self.action_dim, dtype=np.float32)
        self._cached = np.zeros(self.action_dim, dtype=np.float32)
        self._cached_weight = 0.0

    def begin(self, backup_action: np.ndarray | Sequence[float]) -> None:
        anchor = _finite_vector(
            backup_action,
            name="handoff_backup_action",
            min_size=self.action_dim,
        )[: self.action_dim].astype(np.float32, copy=True)
        self._active = True
        self._decision_index = 0
        self._anchor = anchor
        self._cached = anchor.copy()
        self._cached_weight = 1.0

    def cancel(self) -> None:
        self.reset()

    def update(
        self,
        task_action: np.ndarray | Sequence[float],
        *,
        decision_tick: bool,
    ) -> HandoffBlendDecision:
        task = _finite_vector(
            task_action,
            name="handoff_task_action",
            min_size=self.action_dim,
        )[: self.action_dim].astype(np.float32, copy=False)
        if not self._active:
            return HandoffBlendDecision(
                action=task.copy(),
                backup_weight=0.0,
                active=False,
                finished_after_decision=False,
                decision_index=0,
            )

        finished = False
        if bool(decision_tick):
            if self._decision_index >= self.blend_decisions:
                self._active = False
                self._cached = task.copy()
                self._cached_weight = 0.0
                finished = True
            else:
                self._decision_index += 1
                weight = float(
                    (self.blend_decisions - self._decision_index + 1)
                    / (self.blend_decisions + 1)
                )
                weight = float(np.clip(weight, 0.0, 1.0))
                self._cached = (
                    weight * self._anchor + (1.0 - weight) * task
                ).astype(np.float32)
                self._cached_weight = weight

        result = HandoffBlendDecision(
            action=self._cached.copy(),
            backup_weight=float(self._cached_weight),
            active=bool(self._active),
            finished_after_decision=bool(finished),
            decision_index=int(self._decision_index),
        )
        return result


@dataclass(frozen=True)
class BackupRewardConfig:
    """Reward terms used only while training the replacement policy.

    Collision is retained in every mode as a hard physical consequence. The
    physical mode additionally uses geometric proximity shaping, while the ErrP
    modes replace that soft term with decoder feedback.
    """

    task_weight: float = 1.0
    collision_penalty: float = 10.0
    physical_gate_penalty: float = 0.5
    clearance_progress_weight: float = 8.0
    clearance_progress_clip_m: float = 0.02
    errp_weight: float = 2.0
    smoothness_weight: float = 0.02
    action_weight: float = 0.005


@dataclass(frozen=True)
class BackupRewardResult:
    total: float
    components: dict[str, float]


@dataclass(frozen=True)
class SafeMotionsBackupRewardConfig:
    """Panda adaptation of the official collision-avoidance reward.

    The reference human backup receives a quadratic closest-distance reward,
    an action-quality reward, a safe-horizon bonus, and an early-collision
    penalty. Distances here are surface gaps in metres.
    """

    moving_obstacle_max_reward: float = 3.0
    moving_obstacle_max_reward_distance_m: float = 0.60
    static_obstacle_max_reward: float = 1.0
    static_obstacle_max_reward_distance_m: float = 0.10
    self_collision_max_reward: float = 1.0
    self_collision_max_reward_distance_m: float = 0.05
    action_max_reward: float = 0.40
    action_punishment_min_threshold: float = 0.95
    collision_penalty: float = 15.0
    safe_completion_bonus: float = 15.0
    joint_limit_max_penalty: float = 0.0


@dataclass(frozen=True)
class BackupEpisodeBoundary:
    terminal: bool
    successful: bool
    reason: str


@dataclass(frozen=True)
class RiskCandidateAction:
    action: np.ndarray
    source: str


@dataclass(frozen=True)
class TaskManifoldState:
    """One physically valid state from a robot-only task-policy rollout."""

    state: Any
    task_phase: str
    has_grasped_cube: bool
    source_episode: int
    source_step: int


def sample_task_manifold_candidates(
    pool: Sequence[TaskManifoldState],
    rng: np.random.Generator,
    *,
    task_phase: str,
    has_grasped_cube: bool,
    count: int,
) -> tuple[TaskManifoldState, ...]:
    """Sample phase/grasp-consistent states without inventing joint vectors."""

    if not pool:
        raise ValueError("task-manifold pool is empty")
    if int(count) <= 0:
        raise ValueError("task-manifold candidate count must be positive")
    exact = [
        sample
        for sample in pool
        if sample.task_phase == str(task_phase)
        and sample.has_grasped_cube == bool(has_grasped_cube)
    ]
    phase_only = [
        sample for sample in pool if sample.task_phase == str(task_phase)
    ]
    candidates = exact or phase_only or list(pool)
    sample_count = min(int(count), len(candidates))
    indices = np.asarray(
        rng.choice(len(candidates), size=sample_count, replace=False),
        dtype=np.int64,
    ).reshape(-1)
    return tuple(candidates[int(index)] for index in indices)


@dataclass(frozen=True)
class FutureRiskConfig:
    """Finite-horizon physical-risk endpoint used for supervised labels."""

    surface_gap_threshold_m: float = 0.02
    ttc_threshold_s: float = 0.25
    use_ttc_endpoint: bool = False
    label_mode: RiskLabelMode = "conservative_clearance"


@dataclass(frozen=True)
class FutureRiskResult:
    risky: bool
    surface_gap_violation: bool
    collision: bool
    ttc_violation: bool


def replace_unsafe_task_action(
    task_action: np.ndarray | Sequence[float],
    backup_action: np.ndarray | Sequence[float],
    *,
    intervene: bool,
    backup_action_scale: float = 0.2,
    controlled_dims: int = BACKUP_ACTION_DIM,
) -> ActionReplacementResult:
    """Replace task XYZ when unsafe while retaining task yaw and gripper.

    ``backup_action`` is normalized to ``[-1, 1]``. ``backup_action_scale``
    limits the replacement command in the normalized task-action coordinates;
    this changes command authority, not the replacement semantics.
    """

    task = _finite_vector(task_action, name="task_action", min_size=ACTION_DIM)
    backup = _finite_vector(
        backup_action,
        name="backup_action",
        min_size=int(controlled_dims),
    )
    if not 1 <= int(controlled_dims) <= ACTION_DIM:
        raise ValueError(f"controlled_dims must be in [1, {ACTION_DIM}]")
    if not np.isfinite(backup_action_scale) or not 0.0 < float(backup_action_scale) <= 1.0:
        raise ValueError("backup_action_scale must be finite and in (0, 1]")

    task = clip_action(task[:ACTION_DIM]).astype(np.float32, copy=False)
    normalized_backup = np.clip(
        backup[: int(controlled_dims)],
        -1.0,
        1.0,
    ).astype(np.float32, copy=False)
    scaled_backup = normalized_backup * float(backup_action_scale)
    executed = task.copy()
    if bool(intervene):
        executed[: int(controlled_dims)] = scaled_backup

    replacement = executed[: int(controlled_dims)] - task[: int(controlled_dims)]
    return ActionReplacementResult(
        executed_action=executed,
        task_action=task,
        backup_action=scaled_backup,
        intervened=bool(intervene),
        replacement_norm=float(np.linalg.norm(replacement)),
    )


def oracle_intervention_from_gate(
    distance_gate: float,
    *,
    threshold: float = 1e-6,
    geometry_valid: bool = True,
) -> bool:
    if not bool(geometry_valid) or not np.isfinite(distance_gate):
        return False
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return bool(float(distance_gate) >= float(threshold))


def classify_future_risk(
    *,
    surface_gap_m: float,
    collision: bool,
    minimum_ttc_s: float,
    ttc_valid: bool,
    config: FutureRiskConfig = FutureRiskConfig(),
) -> FutureRiskResult:
    """Classify one branch sample without hiding the contributing endpoint."""

    if config.label_mode not in (
        "conservative_clearance",
        "collision_termination",
    ):
        raise ValueError(f"Unknown risk label mode: {config.label_mode}")

    gap_violation = bool(
        np.isfinite(surface_gap_m)
        and float(surface_gap_m) <= float(config.surface_gap_threshold_m)
    )
    ttc_violation = bool(
        config.use_ttc_endpoint
        and ttc_valid
        and np.isfinite(minimum_ttc_s)
        and 0.0 <= float(minimum_ttc_s) <= float(config.ttc_threshold_s)
    )
    collision_value = bool(collision)
    risky = collision_value
    if config.label_mode == "conservative_clearance":
        risky = bool(risky or gap_violation or ttc_violation)
    return FutureRiskResult(
        risky=risky,
        surface_gap_violation=gap_violation,
        collision=collision_value,
        ttc_violation=ttc_violation,
    )


def backup_episode_should_start(
    *,
    encounter_active: bool,
    geometry_valid: bool,
    distance_gate: float,
    mode: BackupStartMode = "encounter",
    gate_threshold: float = 1e-6,
) -> bool:
    """Return whether a short backup-only safety episode should begin."""

    if mode not in ("encounter", "gate"):
        raise ValueError(f"Unknown backup start mode: {mode}")
    if not bool(encounter_active) or not bool(geometry_valid):
        return False
    if mode == "encounter":
        return True
    return oracle_intervention_from_gate(
        distance_gate,
        threshold=gate_threshold,
        geometry_valid=geometry_valid,
    )


def horizon_steps_from_seconds(duration_s: float, physics_dt_s: float) -> int:
    if not np.isfinite(duration_s) or float(duration_s) <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(physics_dt_s) or float(physics_dt_s) <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    return max(1, int(round(float(duration_s) / float(physics_dt_s))))


def time_steps_from_seconds(
    duration_s: float,
    step_dt_s: float,
    *,
    allow_zero: bool = False,
) -> int:
    """Convert a physical duration to steps without silently changing units."""

    if not np.isfinite(duration_s) or float(duration_s) < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    if not np.isfinite(step_dt_s) or float(step_dt_s) <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")
    if float(duration_s) == 0.0:
        if allow_zero:
            return 0
        raise ValueError("duration_s must be positive")
    return max(1, int(round(float(duration_s) / float(step_dt_s))))


def sample_backup_start_source(
    rng: np.random.Generator,
    config: BackupStartDistribution = BackupStartDistribution(),
) -> BackupStartSource:
    values: tuple[BackupStartSource, ...] = (
        "source_encounter",
        "encounter_jitter",
        "task_manifold",
    )
    probability = np.asarray(config.probabilities, dtype=np.float64)
    probability /= float(np.sum(probability))
    return values[int(rng.choice(len(values), p=probability))]


def scheduled_backup_start_source(
    rng: np.random.Generator,
    config: BackupStartDistribution,
    *,
    retry_source: str = "",
) -> tuple[BackupStartSource, bool]:
    """Return a new mixture draw or preserve an unrealized source slot.

    A task-manifold draw is not a completed training episode until a valid
    robot-state/human-encounter pairing is found. Re-sampling the mixture after
    every rejected pairing silently under-represents that source. Keeping the
    slot pending makes the configured distribution apply to completed Backup
    episodes instead of raw pairing attempts.
    """

    if retry_source:
        if retry_source not in (
            "source_encounter",
            "encounter_jitter",
            "task_manifold",
        ):
            raise ValueError(f"Unknown retry source: {retry_source}")
        return retry_source, True
    return sample_backup_start_source(rng, config), False


def backup_start_retry_source(
    source: BackupStartSource,
    *,
    realized: bool,
) -> str:
    """Return the source slot that must survive into the next encounter."""

    if not bool(realized):
        return str(source)
    return ""


def backup_episode_boundary(
    *,
    episode_steps: int,
    max_steps: int,
    collision: bool,
    encounter_active: bool,
    env_terminated: bool,
    env_truncated: bool,
) -> BackupEpisodeBoundary:
    """Define terminal conditions for an M2-aligned backup episode."""

    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if bool(collision):
        return BackupEpisodeBoundary(True, False, "collision")
    if bool(env_truncated):
        return BackupEpisodeBoundary(True, False, "environment_truncated")
    if bool(env_terminated):
        return BackupEpisodeBoundary(True, False, "task_completed_before_horizon")
    if int(episode_steps) >= int(max_steps):
        return BackupEpisodeBoundary(True, True, "horizon_completed")
    return BackupEpisodeBoundary(False, False, "running")


def quadratic_surface_distance_reward(
    surface_gap_m: float,
    threshold_m: float,
    maximum_reward: float,
    *,
    geometry_valid: bool,
    name: str,
) -> float:
    """Official safeMotionsRisk distance term with explicit validity."""

    if not bool(geometry_valid):
        raise ValueError(f"{name} geometry is invalid")
    if not np.isfinite(surface_gap_m):
        raise ValueError(f"{name} surface gap must be finite")
    if not np.isfinite(threshold_m) or float(threshold_m) <= 0.0:
        raise ValueError(f"{name} threshold must be finite and positive")
    if not np.isfinite(maximum_reward) or float(maximum_reward) < 0.0:
        raise ValueError(f"{name} maximum reward must be finite and non-negative")
    normalized = float(
        np.clip(max(float(surface_gap_m), 0.0) / float(threshold_m), 0.0, 1.0)
    )
    return float(maximum_reward) * normalized**2


def compute_safemotions_backup_reward(
    *,
    surface_gap_m: float,
    static_surface_gap_m: float,
    self_surface_gap_m: float,
    human_collision: bool,
    static_collision: bool,
    self_collision: bool,
    moving_geometry_valid: bool,
    static_geometry_valid: bool,
    self_geometry_valid: bool,
    safe_completion: bool,
    backup_action: np.ndarray | Sequence[float],
    include_clearance: bool = True,
    joint_limit_risk: float = 0.0,
    dense_step_scale: float = 1.0,
    config: SafeMotionsBackupRewardConfig = SafeMotionsBackupRewardConfig(),
) -> BackupRewardResult:
    """Compute the reference-style backup reward using Panda surface gap.

    ErrP variants keep the shared terminal and action terms but replace the
    dense geometric-clearance term with delayed decoder feedback.
    """

    action = _finite_vector(
        backup_action,
        name="backup_action",
        min_size=BACKUP_ACTION_DIM,
    )
    max_distance = float(config.moving_obstacle_max_reward_distance_m)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("moving obstacle reward distance must be positive")
    threshold = float(config.action_punishment_min_threshold)
    if not 0.0 <= threshold < 1.0:
        raise ValueError("action punishment threshold must be in [0, 1)")
    if not np.isfinite(dense_step_scale) or float(dense_step_scale) <= 0.0:
        raise ValueError("dense_step_scale must be finite and positive")
    if not np.isfinite(joint_limit_risk) or not 0.0 <= float(joint_limit_risk) <= 1.0:
        raise ValueError("joint_limit_risk must be finite and in [0, 1]")

    clearance = float(dense_step_scale) * (
        quadratic_surface_distance_reward(
            surface_gap_m,
            max_distance,
            config.moving_obstacle_max_reward,
            geometry_valid=moving_geometry_valid,
            name="moving_obstacle",
        )
        if bool(include_clearance)
        else 0.0
    )
    static_clearance = float(dense_step_scale) * quadratic_surface_distance_reward(
        static_surface_gap_m,
        config.static_obstacle_max_reward_distance_m,
        config.static_obstacle_max_reward,
        geometry_valid=static_geometry_valid,
        name="static_obstacle",
    )
    self_clearance = float(dense_step_scale) * quadratic_surface_distance_reward(
        self_surface_gap_m,
        config.self_collision_max_reward_distance_m,
        config.self_collision_max_reward,
        geometry_valid=self_geometry_valid,
        name="self_collision",
    )
    max_action = float(np.max(np.abs(action)))
    action_punishment = float(
        np.clip((max_action - threshold) / (1.0 - threshold), 0.0, 1.0)
        ** 2
    )
    action_quality = (
        float(dense_step_scale)
        * float(config.action_max_reward)
        * (1.0 - action_punishment)
    )
    components = {
        "moving_obstacle_clearance": clearance,
        "static_obstacle_clearance": static_clearance,
        "self_collision_clearance": self_clearance,
        "action_quality": action_quality,
        "joint_limit_proximity": -float(dense_step_scale)
        * float(config.joint_limit_max_penalty)
        * float(joint_limit_risk),
        "hard_collision": -float(config.collision_penalty)
        * float(bool(human_collision or static_collision or self_collision)),
        "safe_completion": float(config.safe_completion_bonus)
        * float(bool(safe_completion)),
    }
    return BackupRewardResult(
        total=float(sum(components.values())),
        components=components,
    )


def normalized_joint_limit_risk(
    joint_positions: np.ndarray | Sequence[float],
    lower_limits: np.ndarray | Sequence[float],
    upper_limits: np.ndarray | Sequence[float],
    *,
    margin_fraction: float = 0.05,
) -> float:
    """Return the worst normalized joint-limit proximity in ``[0, 1]``.

    Zero means every valid joint is at least ``margin_fraction`` of its range
    away from both limits. One means at least one joint is on or beyond a
    limit. Invalid or continuous joints are ignored.
    """

    if not np.isfinite(margin_fraction) or not 0.0 < float(margin_fraction) < 0.5:
        raise ValueError("margin_fraction must be finite and in (0, 0.5)")
    positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if positions.size != lower.size or positions.size != upper.size:
        raise ValueError("joint position and limit vectors must have the same size")
    valid = (
        np.isfinite(positions)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (upper > lower)
    )
    if not np.any(valid):
        return 0.0
    span = upper[valid] - lower[valid]
    clearance = np.minimum(
        positions[valid] - lower[valid],
        upper[valid] - positions[valid],
    ) / span
    risk = np.clip(
        (float(margin_fraction) - clearance) / float(margin_fraction),
        0.0,
        1.0,
    )
    return float(np.max(risk**2))


def risk_candidate_actions(
    task_action: np.ndarray | Sequence[float],
    rng: np.random.Generator,
    *,
    mode: RiskCandidateMode = "task",
    count: int = 1,
    random_fraction: float = 0.5,
    controlled_dims: int = BACKUP_ACTION_DIM,
) -> tuple[RiskCandidateAction, ...]:
    """Build task-distribution and exploratory actions for risk supervision.

    Random candidates vary the Panda XYZ command while retaining task yaw and
    gripper. This is the task-space counterpart of the reference repository's
    random-action risk-data generation.
    """

    if mode not in ("task", "random", "mixed"):
        raise ValueError(f"Unknown risk candidate mode: {mode}")
    if int(count) <= 0:
        raise ValueError("candidate count must be positive")
    if not 0.0 <= float(random_fraction) <= 1.0:
        raise ValueError("random_fraction must be in [0, 1]")
    if not 1 <= int(controlled_dims) <= ACTION_DIM:
        raise ValueError(f"controlled_dims must be in [1, {ACTION_DIM}]")

    task = clip_action(
        _finite_vector(task_action, name="task_action", min_size=ACTION_DIM)[
            :ACTION_DIM
        ]
    ).astype(np.float32, copy=False)
    if mode == "task":
        random_count = 0
    elif mode == "random":
        random_count = int(count)
    else:
        random_count = int(round(int(count) * float(random_fraction)))
        if random_fraction > 0.0:
            random_count = max(1, random_count)
        if random_fraction < 1.0:
            random_count = min(int(count) - 1, random_count)
    task_count = int(count) - random_count

    result: list[RiskCandidateAction] = []
    for _ in range(task_count):
        result.append(RiskCandidateAction(task.copy(), "task"))
    for _ in range(random_count):
        candidate = task.copy()
        candidate[: int(controlled_dims)] = rng.uniform(
            -1.0,
            1.0,
            size=int(controlled_dims),
        ).astype(np.float32)
        result.append(RiskCandidateAction(candidate, "random_xyz"))
    if len(result) > 1:
        order = rng.permutation(len(result))
        result = [result[int(index)] for index in order]
    return tuple(result)


def task_reward_without_human_terms(
    total_reward: float,
    reward_components: Mapping[str, float],
) -> float:
    """Remove legacy human terms so reward variants do not double count them."""

    human_terms = (
        "near_human_penalty",
        "human_collision_penalty",
        "errp_penalty",
    )
    return float(total_reward) - sum(
        float(reward_components.get(name, 0.0)) for name in human_terms
    )


def compute_backup_reward(
    *,
    mode: BackupRewardMode,
    task_reward: float,
    previous_surface_gap_m: float,
    surface_gap_m: float,
    distance_gate: float,
    collision: bool,
    backup_action: np.ndarray | Sequence[float],
    previous_backup_action: np.ndarray | Sequence[float] | None = None,
    intervened: bool = True,
    errp_probability: float = 0.0,
    errp_uncertainty: float = 0.0,
    config: BackupRewardConfig = BackupRewardConfig(),
) -> BackupRewardResult:
    if mode not in ("physical", "raw_errp", "uq_errp"):
        raise ValueError(f"Unknown backup reward mode: {mode}")

    action = _finite_vector(
        backup_action,
        name="backup_action",
        min_size=BACKUP_ACTION_DIM,
    )
    previous_action = (
        np.zeros_like(action)
        if previous_backup_action is None
        else _finite_vector(
            previous_backup_action,
            name="previous_backup_action",
            min_size=action.size,
        )[: action.size]
    )
    gate = float(np.clip(distance_gate, 0.0, 1.0))
    probability = float(np.clip(errp_probability, 0.0, 1.0))
    uncertainty = float(np.clip(errp_uncertainty, 0.0, 1.0))

    progress_m = 0.0
    if np.isfinite(previous_surface_gap_m) and np.isfinite(surface_gap_m):
        progress_m = float(surface_gap_m) - float(previous_surface_gap_m)
        clip_m = max(0.0, float(config.clearance_progress_clip_m))
        if clip_m > 0.0:
            progress_m = float(np.clip(progress_m, -clip_m, clip_m))

    components = {
        "task": float(config.task_weight) * float(task_reward),
        "hard_collision": -float(config.collision_penalty) * float(bool(collision)),
        "physical_gate": 0.0,
        "clearance_progress": 0.0,
        "errp": 0.0,
        "action": -float(config.action_weight)
        * float(bool(intervened))
        * float(np.linalg.norm(action)),
        "smoothness": -float(config.smoothness_weight)
        * float(bool(intervened))
        * float(np.linalg.norm(action - previous_action)),
    }
    if mode == "physical" and bool(intervened):
        components["physical_gate"] = -float(config.physical_gate_penalty) * gate
        components["clearance_progress"] = (
            float(config.clearance_progress_weight) * gate * progress_m
        )
    elif mode == "raw_errp" and bool(intervened):
        components["errp"] = -float(config.errp_weight) * probability
    elif mode == "uq_errp" and bool(intervened):
        components["errp"] = (
            -float(config.errp_weight) * probability * (1.0 - uncertainty)
        )
    return BackupRewardResult(
        total=float(sum(components.values())),
        components=components,
    )


def state_action_features(
    observation: np.ndarray | Sequence[float],
    candidate_action: np.ndarray | Sequence[float],
) -> np.ndarray:
    obs = _finite_vector(observation, name="observation", min_size=1)
    action = _finite_vector(candidate_action, name="candidate_action", min_size=1)
    return np.concatenate((obs, action), axis=0).astype(np.float32, copy=False)


if nn is not None:

    class StateActionRiskEstimator(nn.Module):
        def __init__(
            self,
            observation_dim: int,
            action_dim: int = ACTION_DIM,
            hidden_dims: tuple[int, ...] = (256, 256),
        ) -> None:
            super().__init__()
            if observation_dim <= 0 or action_dim <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)
            self.hidden_dims = tuple(int(value) for value in hidden_dims)
            layers: list[nn.Module] = []
            width = self.observation_dim + self.action_dim
            for hidden in self.hidden_dims:
                if hidden <= 0:
                    raise ValueError("hidden dimensions must be positive")
                layers.extend((nn.Linear(width, hidden), nn.ReLU()))
                width = hidden
            layers.append(nn.Linear(width, 1))
            self.network = nn.Sequential(*layers)

        def forward(
            self,
            observation: torch.Tensor,
            candidate_action: torch.Tensor,
        ) -> torch.Tensor:
            if observation.shape[-1] != self.observation_dim:
                raise ValueError(
                    f"Expected observation dim {self.observation_dim}, got "
                    f"{observation.shape[-1]}"
                )
            if candidate_action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected action dim {self.action_dim}, got "
                    f"{candidate_action.shape[-1]}"
                )
            features = torch.cat((observation, candidate_action), dim=-1)
            return self.network(features).squeeze(-1)

        def probability(
            self,
            observation: torch.Tensor,
            candidate_action: torch.Tensor,
        ) -> torch.Tensor:
            return torch.sigmoid(self.forward(observation, candidate_action))

else:  # pragma: no cover

    class StateActionRiskEstimator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("torch is required for StateActionRiskEstimator")


def risk_estimator_checkpoint(
    model: StateActionRiskEstimator,
    *,
    observation_version: str,
    action_version: str,
    backup_checkpoint: str,
    metrics: Mapping[str, float] | None = None,
    threshold: float | None = None,
    observation_mean: np.ndarray | Sequence[float] | None = None,
    observation_std: np.ndarray | Sequence[float] | None = None,
    action_mean: np.ndarray | Sequence[float] | None = None,
    action_std: np.ndarray | Sequence[float] | None = None,
    label_config: Mapping[str, Any] | None = None,
    artifact_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("torch is required for risk estimator checkpoints")
    obs_mean = _normalization_vector(
        observation_mean, model.observation_dim, fill_value=0.0
    )
    obs_std = _normalization_vector(
        observation_std, model.observation_dim, fill_value=1.0
    )
    candidate_mean = _normalization_vector(
        action_mean, model.action_dim, fill_value=0.0
    )
    candidate_std = _normalization_vector(
        action_std, model.action_dim, fill_value=1.0
    )
    return {
        "schema_version": RISK_ESTIMATOR_VERSION,
        "model_state_dict": model.state_dict(),
        "observation_dim": model.observation_dim,
        "action_dim": model.action_dim,
        "hidden_dims": model.hidden_dims,
        "observation_version": str(observation_version),
        "action_version": str(action_version),
        "backup_checkpoint": str(backup_checkpoint),
        "metrics": dict(metrics or {}),
        "threshold": None if threshold is None else float(threshold),
        "observation_mean": obs_mean,
        "observation_std": np.maximum(obs_std, 1e-6),
        "action_mean": candidate_mean,
        "action_std": np.maximum(candidate_std, 1e-6),
        "label_config": dict(label_config or {}),
        "artifact_contract": dict(artifact_contract or {}),
    }


def backup_reward_config_dict(config: BackupRewardConfig) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(config).items()}


def _integer_ratio(value: float, *, name: str, tolerance: float = 1e-6) -> int:
    rounded = int(round(float(value)))
    if rounded <= 0 or abs(float(value) - rounded) > float(tolerance):
        raise ValueError(f"{name} must be a positive integer ratio, got {value}")
    return rounded


def _finite_vector(
    value: np.ndarray | Sequence[float],
    *,
    name: str,
    min_size: int,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size < int(min_size):
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalization_vector(
    value: np.ndarray | Sequence[float] | None,
    expected_dim: int,
    *,
    fill_value: float,
) -> np.ndarray:
    if value is None:
        return np.full((int(expected_dim),), float(fill_value), dtype=np.float32)
    result = _finite_vector(value, name="normalization", min_size=expected_dim)
    if result.size != int(expected_dim):
        raise ValueError(
            f"normalization must contain exactly {expected_dim} values"
        )
    return result.astype(np.float32, copy=False)
