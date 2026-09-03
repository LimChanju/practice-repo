from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

import numpy as np


ERRP_FEEDBACK_VERSION = (
    "event_locked_errp_feedback_v4_realized_agency_one_query_per_hazard"
)


@dataclass(frozen=True)
class ErrPEvent:
    event_id: int
    event_step: int
    event_type: str
    expected_label: int
    physical_risk_score: float
    agency_label: str = "not_applicable"
    robot_approach_share: float = -1.0
    robot_approach_speed_mps: float = 0.0
    human_approach_speed_mps: float = 0.0
    eligibility_reason: str = ""


@dataclass(frozen=True)
class ErrPFeedbackSample:
    event: ErrPEvent
    source: str
    epoch_id: str
    decoded_probability: float
    uncertainty: float
    dataset_label: int
    decoder_prediction: int
    delivery_step: int

    @property
    def confidence(self) -> float:
        return float(1.0 - self.uncertainty)


@dataclass(frozen=True)
class CreditAssignment:
    transition_step: int
    weight: float
    penalty: float
    event_id: int


@dataclass(frozen=True)
class PseudoErrPConfig:
    minimum_evidence: float = 0.5
    maximum_evidence: float = 12.0
    probability_noise_std: float = 0.04
    uncertainty_noise_std: float = 0.02


@dataclass(frozen=True)
class EventDetectorConfig:
    threat_trigger: str = "near"
    refractory_steps: int = 60
    rearm_safe_steps: int = 12
    safe_probe_interval_steps: int = 180
    enable_safe_probes: bool = True
    agency_mode: str = "any_hazard"
    agency_minimum_closing_speed_mps: float = 0.02
    agency_robot_share_threshold: float = 0.60
    agency_history_steps: int = 12
    yield_assessment_steps: int = 12
    yield_recovery_m: float = 0.005
    failure_gap_decrease_m: float = 0.005


@dataclass(frozen=True)
class RealizedApproachAgency:
    label: str
    robot_share: float
    robot_approach_speed_mps: float
    human_approach_speed_mps: float
    valid: bool
    closest_hand: str


@dataclass(frozen=True)
class CreditConfig:
    latency_steps: int = 18
    history_steps: int = 30
    decay: float = 0.92
    penalty_scale: float = 2.0


class FeedbackSource(Protocol):
    source_name: str

    def sample(
        self,
        event: ErrPEvent,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, float, int, int, str]: ...


class PseudoErrPSource:
    source_name = "pseudo_dirichlet"

    def __init__(self, config: PseudoErrPConfig = PseudoErrPConfig()) -> None:
        self.config = config

    def sample(
        self,
        event: ErrPEvent,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, float, int, int, str]:
        risk = float(np.clip(event.physical_risk_score, 0.0, 1.0))
        # Evidence is strongest away from the ambiguous 0.5 boundary.
        certainty_shape = float(np.clip(2.0 * abs(risk - 0.5), 0.0, 1.0))
        evidence_strength = float(self.config.minimum_evidence) + (
            float(self.config.maximum_evidence) - float(self.config.minimum_evidence)
        ) * certainty_shape
        alpha_errp = 1.0 + evidence_strength * risk
        alpha_non_errp = 1.0 + evidence_strength * (1.0 - risk)
        strength = alpha_errp + alpha_non_errp
        probability = alpha_errp / strength
        uncertainty = 2.0 / strength
        probability += float(rng.normal(0.0, self.config.probability_noise_std))
        uncertainty += float(rng.normal(0.0, self.config.uncertainty_noise_std))
        probability = float(np.clip(probability, 0.0, 1.0))
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        prediction = int(probability >= 0.5)
        return probability, uncertainty, int(event.expected_label), prediction, (
            f"pseudo_event_{event.event_id:08d}"
        )


class CleanPseudoErrPSource:
    """Deterministic upper-bound channel for the ErrP reward path.

    This source is intentionally not a physiological decoder model.  It emits
    the simulator event label with zero uncertainty so that an experiment can
    first test whether event detection and delayed credit assignment are
    capable of teaching the Backup policy at all.
    """

    source_name = "clean_pseudo_errp"

    def sample(
        self,
        event: ErrPEvent,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, float, int, int, str]:
        del rng
        label = int(event.expected_label)
        if label not in (0, 1):
            raise ValueError(f"Clean pseudo-ErrP label must be binary: {label}")
        return (
            float(label),
            0.0,
            label,
            label,
            f"clean_pseudo_event_{event.event_id:08d}",
        )


class ErrPEventDetector:
    def __init__(
        self, config: EventDetectorConfig = EventDetectorConfig()
    ) -> None:
        self.config = config
        if self.config.threat_trigger not in (
            "gate",
            "near",
            "near_miss",
            "collision",
        ):
            raise ValueError(
                f"Unknown ErrP threat trigger: {self.config.threat_trigger}"
            )
        if int(self.config.refractory_steps) < 0:
            raise ValueError("refractory_steps must be non-negative")
        if int(self.config.rearm_safe_steps) <= 0:
            raise ValueError("rearm_safe_steps must be positive")
        if self.config.agency_mode not in ("any_hazard", "current_policy"):
            raise ValueError(
                "agency_mode must be 'any_hazard' or 'current_policy'"
            )
        if not 0.5 < float(self.config.agency_robot_share_threshold) <= 1.0:
            raise ValueError("agency_robot_share_threshold must be in (0.5, 1]")
        if int(self.config.agency_history_steps) <= 0:
            raise ValueError("agency_history_steps must be positive")
        if int(self.config.yield_assessment_steps) < 0:
            raise ValueError("yield_assessment_steps must be non-negative")
        self._next_event_id = 0
        self._diagnostics: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self._hazard_active = False
        self._hazard_event_emitted = False
        self._safe_streak_steps = 0
        self._last_threat_event_step = -10**9
        self._last_safe_probe_step = -10**9
        self._episode_observation_count = 0
        self._episode_had_safe_observation = False
        self._hazard_agency_samples: list[RealizedApproachAgency] = []
        self._pending_human_trigger_step: int | None = None
        self._pending_trigger_gap_m = 10.0
        self._pending_min_gap_m = 10.0
        self._pending_risk_score = 0.0
        self._pending_agency = _invalid_agency()

    def diagnostics(self) -> dict[str, int]:
        return dict(self._diagnostics)

    def _count(self, name: str) -> None:
        self._diagnostics[name] = int(self._diagnostics.get(name, 0)) + 1

    def observe(
        self,
        step: int,
        observation: Mapping[str, Any],
        *,
        encounter_active: bool,
        agency_context: Mapping[str, Any] | None = None,
    ) -> ErrPEvent | None:
        self._episode_observation_count += 1
        gate = _scalar(observation, "distance_gate", 0.0)
        near = _scalar(observation, "near_human", 0.0) > 0.5
        near_miss = _scalar(observation, "near_miss", 0.0) > 0.5
        collision = _scalar(observation, "human_robot_collision", 0.0) > 0.5
        gate_active = gate > 0.0
        hazard_signal = bool(gate_active or near or near_miss or collision)
        agency = realized_approach_agency(
            observation,
            agency_context=agency_context,
            minimum_closing_speed_mps=(
                self.config.agency_minimum_closing_speed_mps
            ),
            robot_share_threshold=self.config.agency_robot_share_threshold,
        )
        if hazard_signal:
            if not self._hazard_active:
                self._hazard_active = True
                self._hazard_event_emitted = False
                self._hazard_agency_samples = []
                self._pending_human_trigger_step = None
            self._safe_streak_steps = 0
            if agency.valid:
                self._hazard_agency_samples.append(agency)
                self._hazard_agency_samples = self._hazard_agency_samples[
                    -int(self.config.agency_history_steps) :
                ]
        elif self._hazard_active:
            if self._pending_human_trigger_step is not None:
                self._hazard_event_emitted = True
                self._pending_human_trigger_step = None
                self._count("suppressed_successful_yield")
            self._safe_streak_steps += 1
            if self._safe_streak_steps >= int(self.config.rearm_safe_steps):
                self._hazard_active = False
                self._hazard_event_emitted = False
                self._safe_streak_steps = 0
                self._hazard_agency_samples = []
        if not hazard_signal:
            self._episode_had_safe_observation = True

        risk_score = physical_risk_score(observation)
        event_type = ""
        expected_label = 0
        trigger_active = {
            "gate": gate_active,
            "near": near,
            "near_miss": near_miss,
            "collision": collision,
        }[self.config.threat_trigger]
        event_agency = _aggregate_agency(
            self._hazard_agency_samples,
            minimum_closing_speed_mps=self.config.agency_minimum_closing_speed_mps,
            robot_share_threshold=self.config.agency_robot_share_threshold,
        )
        eligibility_reason = ""
        if self.config.agency_mode == "current_policy":
            gap_m = _scalar(
                observation,
                "min_hand_end_effector_surface_gap",
                10.0,
            )
            if self._pending_human_trigger_step is not None:
                self._pending_min_gap_m = min(self._pending_min_gap_m, gap_m)
                elapsed = int(step) - int(self._pending_human_trigger_step)
                failure_to_yield = bool(
                    collision
                    or (
                        agency.valid
                        and agency.robot_approach_speed_mps
                        >= float(self.config.agency_minimum_closing_speed_mps)
                    )
                    or (
                        self._pending_trigger_gap_m - self._pending_min_gap_m
                        >= float(self.config.failure_gap_decrease_m)
                    )
                )
                if failure_to_yield or elapsed >= int(
                    self.config.yield_assessment_steps
                ):
                    recovered = bool(
                        gap_m - self._pending_trigger_gap_m
                        >= float(self.config.yield_recovery_m)
                    )
                    self._hazard_event_emitted = True
                    self._pending_human_trigger_step = None
                    event_agency = self._pending_agency
                    risk_score = max(risk_score, self._pending_risk_score)
                    if failure_to_yield and not recovered:
                        event_type = "failure_to_yield"
                        expected_label = 1
                        eligibility_reason = "human_initiated_failure_to_yield"
                        self._count("eligible_failure_to_yield")
                    else:
                        self._count(
                            "suppressed_successful_yield"
                            if recovered
                            else "suppressed_ambiguous_human_initiated"
                        )
            elif (
                self._hazard_active
                and not self._hazard_event_emitted
                and trigger_active
            ):
                initial_collision = bool(
                    collision
                    and self._episode_observation_count == 1
                    and not self._episode_had_safe_observation
                )
                if initial_collision:
                    self._hazard_event_emitted = True
                    self._count("suppressed_initial_collision")
                elif not event_agency.valid:
                    self._hazard_event_emitted = True
                    event_type = f"{self.config.threat_trigger}_onset"
                    expected_label = 1
                    eligibility_reason = "agency_unavailable_fallback"
                    self._count("eligible_agency_unavailable_fallback")
                elif event_agency.label in ("robot_initiated", "mixed"):
                    self._hazard_event_emitted = True
                    event_type = f"{self.config.threat_trigger}_onset"
                    expected_label = 1
                    eligibility_reason = event_agency.label
                    self._count(f"eligible_{event_agency.label}")
                else:
                    self._pending_human_trigger_step = int(step)
                    self._pending_trigger_gap_m = gap_m
                    self._pending_min_gap_m = gap_m
                    self._pending_risk_score = risk_score
                    self._pending_agency = event_agency
                    self._count(f"pending_{event_agency.label}")
        elif (
            self._hazard_active
            and not self._hazard_event_emitted
            and trigger_active
        ):
            # One physiological query per continuous hazard encounter. Nested
            # thresholds are physical severity outcomes, not three independent
            # EEG events.
            self._hazard_event_emitted = True
            event_type = f"{self.config.threat_trigger}_onset"
            expected_label = 1
            eligibility_reason = "any_hazard"

        if event_type and expected_label == 1:
            if (
                int(step) - self._last_threat_event_step
                < int(self.config.refractory_steps)
            ):
                event_type = ""
                self._count("suppressed_refractory")
            else:
                self._last_threat_event_step = int(step)
        elif (
            self.config.enable_safe_probes
            and encounter_active
            and not hazard_signal
            and int(step) - self._last_safe_probe_step
            >= int(self.config.safe_probe_interval_steps)
        ):
            event_type = "safe_probe"
            expected_label = 0
            risk_score = min(risk_score, 0.15)
            eligibility_reason = "safe_probe"

        if not event_type:
            return None
        event = ErrPEvent(
            event_id=self._next_event_id,
            event_step=int(step),
            event_type=event_type,
            expected_label=int(expected_label),
            physical_risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            agency_label=event_agency.label,
            robot_approach_share=float(event_agency.robot_share),
            robot_approach_speed_mps=float(
                event_agency.robot_approach_speed_mps
            ),
            human_approach_speed_mps=float(
                event_agency.human_approach_speed_mps
            ),
            eligibility_reason=eligibility_reason,
        )
        self._next_event_id += 1
        if event_type == "safe_probe":
            self._last_safe_probe_step = int(step)
        return event


class EventLockedFeedbackBridge:
    """Sample feedback at event onset and assign it to preceding actions.

    Events enter a pending queue at onset and become available only at their
    modeled delivery step. The returned credit still targets the causal action
    window preceding the event, without exposing future observations to the
    policy that produced those actions.
    """

    def __init__(
        self,
        source: FeedbackSource,
        *,
        detector_config: EventDetectorConfig = EventDetectorConfig(),
        credit_config: CreditConfig = CreditConfig(),
        seed: int = 0,
    ) -> None:
        self.source = source
        self.detector = ErrPEventDetector(detector_config)
        self.credit_config = credit_config
        self.rng = np.random.default_rng(seed)
        self.samples: list[ErrPFeedbackSample] = []
        self._pending: list[ErrPFeedbackSample] = []
        self._delivered_event_ids: set[int] = set()
        self._boundary_flushed_event_ids: set[int] = set()
        self.dropped_pending_on_reset = 0
        self._episode_start_step = 0

    def reset_episode(self, *, start_step: int = 0) -> None:
        self.dropped_pending_on_reset += len(self._pending)
        self._pending.clear()
        self.detector.reset()
        self._episode_start_step = int(start_step)

    def observe(
        self,
        step: int,
        observation: Mapping[str, Any],
        *,
        encounter_active: bool,
        agency_context: Mapping[str, Any] | None = None,
    ) -> tuple[ErrPFeedbackSample | None, tuple[CreditAssignment, ...]]:
        event = self.detector.observe(
            step,
            observation,
            encounter_active=encounter_active,
            agency_context=agency_context,
        )
        if event is not None:
            probability, uncertainty, dataset_label, prediction, epoch_id = (
                self.source.sample(event, rng=self.rng)
            )
            probability = _unit_interval(probability, "decoded_probability")
            uncertainty = _unit_interval(uncertainty, "uncertainty")
            queued_sample = ErrPFeedbackSample(
                event=event,
                source=str(self.source.source_name),
                epoch_id=str(epoch_id),
                decoded_probability=probability,
                uncertainty=uncertainty,
                dataset_label=int(dataset_label),
                decoder_prediction=int(prediction),
                delivery_step=int(
                    event.event_step + self.credit_config.latency_steps
                ),
            )
            self.samples.append(queued_sample)
            self._pending.append(queued_sample)

        delivered = self._deliver_oldest_due(int(step), boundary_flush=False)
        if delivered is None:
            return None, ()
        return delivered

    def flush_pending(
        self,
    ) -> tuple[tuple[ErrPFeedbackSample, tuple[CreditAssignment, ...]], ...]:
        """Deliver pending offline feedback before a training episode is closed.

        The decoder output is already available in the offline replay bundle.
        Flushing changes only when the learner receives retroactive credit; it
        never exposes future feedback to the policy that generated the action.
        """

        result = []
        while self._pending:
            delivery_step = min(sample.delivery_step for sample in self._pending)
            delivered = self._deliver_oldest_due(
                int(delivery_step), boundary_flush=True
            )
            if delivered is None:  # pragma: no cover - defensive invariant.
                raise RuntimeError("pending feedback could not be flushed")
            result.append(delivered)
        return tuple(result)

    def _deliver_oldest_due(
        self,
        step: int,
        *,
        boundary_flush: bool,
    ) -> tuple[ErrPFeedbackSample, tuple[CreditAssignment, ...]] | None:
        due = [
            sample
            for sample in self._pending
            if int(sample.delivery_step) <= int(step)
        ]
        if not due:
            return None
        # Simulation calls are step-contiguous and event steps are unique, so
        # fixed latency normally makes exactly one sample due. If a caller skips
        # steps, deliver the oldest sample now and leave the remainder queued.
        sample = min(due, key=lambda value: value.delivery_step)
        self._pending.remove(sample)
        event_id = int(sample.event.event_id)
        self._delivered_event_ids.add(event_id)
        if boundary_flush:
            self._boundary_flushed_event_ids.add(event_id)
        assignments = tuple(
            assignment
            for assignment in credit_assignments(sample, self.credit_config)
            if assignment.transition_step >= self._episode_start_step
        )
        assignments = _renormalize_boundary_credit(assignments)
        return sample, assignments

    def records(self) -> list[dict[str, Any]]:
        result = []
        for sample in self.samples:
            row = asdict(sample)
            row["event"] = asdict(sample.event)
            row["confidence"] = sample.confidence
            row["delivered"] = int(sample.event.event_id) in self._delivered_event_ids
            row["boundary_flushed"] = (
                int(sample.event.event_id) in self._boundary_flushed_event_ids
            )
            result.append(row)
        return result

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "feedback_version": ERRP_FEEDBACK_VERSION,
            "detector_config": asdict(self.detector.config),
            "mapping_contract": (
                "one_decoder_query_per_continuous_hazard; nested physical "
                "thresholds are not independent biological ErrP events"
            ),
            "observed_event_count": len(self.samples),
            "delivered_event_count": len(self._delivered_event_ids),
            "pending_event_count": len(self._pending),
            "dropped_pending_on_reset": int(self.dropped_pending_on_reset),
            "boundary_flushed_event_count": len(
                self._boundary_flushed_event_ids
            ),
            "agency_diagnostics": self.detector.diagnostics(),
        }


def credit_assignments(
    sample: ErrPFeedbackSample,
    config: CreditConfig = CreditConfig(),
) -> tuple[CreditAssignment, ...]:
    if config.history_steps <= 0:
        raise ValueError("history_steps must be positive")
    if not 0.0 < float(config.decay) <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    steps = np.arange(
        sample.event.event_step - int(config.history_steps) + 1,
        sample.event.event_step + 1,
        dtype=np.int64,
    )
    raw_weights = np.power(
        float(config.decay), sample.event.event_step - steps
    ).astype(np.float64)
    weights = raw_weights / max(float(raw_weights.sum()), 1e-12)
    total_penalty = -float(config.penalty_scale) * float(
        sample.decoded_probability
    )
    return tuple(
        CreditAssignment(
            transition_step=int(step),
            weight=float(weight),
            penalty=float(total_penalty * weight),
            event_id=int(sample.event.event_id),
        )
        for step, weight in zip(steps, weights)
    )


def _renormalize_boundary_credit(
    assignments: tuple[CreditAssignment, ...],
) -> tuple[CreditAssignment, ...]:
    """Preserve an event's total penalty after causal episode clipping."""
    if not assignments:
        return ()
    retained_weight = float(sum(item.weight for item in assignments))
    if retained_weight <= 0.0:
        raise ValueError("Retained credit weight must be positive")
    retained_penalty = float(sum(item.penalty for item in assignments))
    total_penalty = retained_penalty / retained_weight
    return tuple(
        CreditAssignment(
            transition_step=item.transition_step,
            weight=float(item.weight / retained_weight),
            penalty=float(total_penalty * item.weight / retained_weight),
            event_id=item.event_id,
        )
        for item in assignments
    )


def realized_approach_agency(
    observation: Mapping[str, Any],
    *,
    agency_context: Mapping[str, Any] | None = None,
    minimum_closing_speed_mps: float = 0.02,
    robot_share_threshold: float = 0.60,
) -> RealizedApproachAgency:
    """Attribute current gap closing to robot-surface and human motion."""

    context = {} if agency_context is None else agency_context
    candidates: list[tuple[float, str]] = []
    for side in ("left", "right"):
        gap = _scalar(
            observation,
            f"{side}_hand_end_effector_surface_gap",
            np.inf,
        )
        if np.isfinite(gap):
            candidates.append((gap, side))
    if not candidates:
        return _invalid_agency()
    _, side = min(candidates, key=lambda item: item[0])

    hand_position = _vector3(observation, f"human_{side}_hand_pos")
    surface_position = _vector3(
        context, f"{side}_closest_surface_point_world_pos"
    )
    robot_velocity = _vector3(
        context, f"{side}_closest_robot_velocity_world_mps"
    )
    hand_velocity = _vector3(observation, f"{side}_hand_vel_filtered_mps")
    robot_valid = _scalar(
        context, f"{side}_robot_surface_velocity_valid", 0.0
    ) > 0.5
    hand_valid = _scalar(
        context, f"{side}_hand_velocity_valid", 0.0
    ) > 0.5
    if (
        hand_position is None
        or surface_position is None
        or robot_velocity is None
        or hand_velocity is None
        or not robot_valid
        or not hand_valid
    ):
        return _invalid_agency(closest_hand=side)
    displacement = hand_position - surface_position
    distance = float(np.linalg.norm(displacement))
    if not np.isfinite(distance) or distance <= 1e-6:
        return _invalid_agency(closest_hand=side)
    normal_robot_to_hand = displacement / distance
    robot = max(float(np.dot(robot_velocity, normal_robot_to_hand)), 0.0)
    human = max(float(-np.dot(hand_velocity, normal_robot_to_hand)), 0.0)
    return _classify_agency(
        robot,
        human,
        minimum_closing_speed_mps=minimum_closing_speed_mps,
        robot_share_threshold=robot_share_threshold,
        closest_hand=side,
    )


def _aggregate_agency(
    samples: list[RealizedApproachAgency],
    *,
    minimum_closing_speed_mps: float,
    robot_share_threshold: float,
) -> RealizedApproachAgency:
    valid = [sample for sample in samples if sample.valid]
    if not valid:
        return _invalid_agency()
    robot = float(np.median([sample.robot_approach_speed_mps for sample in valid]))
    human = float(np.median([sample.human_approach_speed_mps for sample in valid]))
    closest_hands = [sample.closest_hand for sample in valid]
    closest_hand = max(set(closest_hands), key=closest_hands.count)
    return _classify_agency(
        robot,
        human,
        minimum_closing_speed_mps=minimum_closing_speed_mps,
        robot_share_threshold=robot_share_threshold,
        closest_hand=closest_hand,
    )


def _classify_agency(
    robot_speed_mps: float,
    human_speed_mps: float,
    *,
    minimum_closing_speed_mps: float,
    robot_share_threshold: float,
    closest_hand: str,
) -> RealizedApproachAgency:
    robot = max(float(robot_speed_mps), 0.0)
    human = max(float(human_speed_mps), 0.0)
    total = robot + human
    share = 0.5 if total <= 0.0 else robot / total
    if total < float(minimum_closing_speed_mps):
        label = "stationary_or_low_closing"
    elif share >= float(robot_share_threshold):
        label = "robot_initiated"
    elif share <= 1.0 - float(robot_share_threshold):
        label = "human_initiated"
    else:
        label = "mixed"
    return RealizedApproachAgency(
        label=label,
        robot_share=float(share),
        robot_approach_speed_mps=robot,
        human_approach_speed_mps=human,
        valid=True,
        closest_hand=str(closest_hand),
    )


def _invalid_agency(*, closest_hand: str = "") -> RealizedApproachAgency:
    return RealizedApproachAgency(
        label="unavailable",
        robot_share=-1.0,
        robot_approach_speed_mps=0.0,
        human_approach_speed_mps=0.0,
        valid=False,
        closest_hand=closest_hand,
    )


def physical_risk_score(observation: Mapping[str, Any]) -> float:
    gate = float(np.clip(_scalar(observation, "distance_gate", 0.0), 0.0, 1.0))
    collision = _scalar(observation, "human_robot_collision", 0.0) > 0.5
    near_miss = _scalar(observation, "near_miss", 0.0) > 0.5
    ttc_scores = []
    closing_scores = []
    for side in ("left", "right"):
        valid = _scalar(observation, f"{side}_ttc_valid", 0.0) > 0.5
        ttc = _scalar(observation, f"{side}_ttc_s", 10.0)
        if valid and np.isfinite(ttc):
            ttc_scores.append(float(np.clip((0.75 - ttc) / 0.75, 0.0, 1.0)))
        closing = _scalar(observation, f"{side}_closing_speed_mps", 0.0)
        if np.isfinite(closing):
            closing_scores.append(float(np.clip(closing / 1.0, 0.0, 1.0)))
    return float(
        max(
            gate,
            1.0 if collision else 0.0,
            0.9 if near_miss else 0.0,
            max(ttc_scores, default=0.0),
            0.5 * max(closing_scores, default=0.0),
        )
    )


def _scalar(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name)
    if value is None:
        return float(default)
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return float(default)
    result = float(array[0])
    return result if np.isfinite(result) else float(default)


def _vector3(values: Mapping[str, Any], name: str) -> np.ndarray | None:
    value = values.get(name)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3].copy()


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result
