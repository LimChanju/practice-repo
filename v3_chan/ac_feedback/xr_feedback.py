"""Controller-only XR explicit feedback for the BC+CBF collection v1 study.

The state machines in this module are deliberately Isaac-independent.  Kit/XR
objects are imported only by :class:`IsaacXRButtonSource` and
:class:`XRSceneViewTextRenderer` when those adapters are explicitly opened.
This keeps the input and questionnaire contracts unit-testable without a
running SimulationApp.

Only controller inputs are read.  This module never drives controller output.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


QUESTIONNAIRE_SCHEMA_VERSION = "ac_selective_smoothing_xr_feedback_v1"

Q1_ID = "q1_modification_needed"
Q1_TEXT_KO = "같은 상황에서 방금 로봇 반응을 다시 사용한다면 수정이 필요합니까?"
Q1_RESPONSE_CODES = (
    "needs_modification",
    "acceptable_as_is",
    "uncertain",
)
Q1_OPTION_LABELS_KO = {
    "needs_modification": "수정이 필요함",
    "acceptable_as_is": "그대로 사용 가능함",
    "uncertain": "잘 모르겠음",
}

LIKERT_QUESTIONS_KO = {
    "q2": "방금 로봇 반응에서 손과 충돌하거나 지나치게 가까워질 위험을 얼마나 느꼈습니까?",
    "q3": "방금 로봇 움직임이 갑작스럽거나 예측하기 어려웠습니까?",
    "q4": "방금 safety response가 필요 이상으로 오래 지속되었다고 느꼈습니까?",
    "q5": "방금 로봇 반응이 원래 pick-and-place 진행을 얼마나 방해했다고 느꼈습니까?",
    "q6": "방금 판단에 얼마나 확신합니까?",
}
LIKERT_VALUES = (1, 2, 3, 4, 5)

REJECTION_REASON_ID = "rejection_reasons"
REJECTION_REASON_IDS = (
    "too_close_or_late",
    "excessive_or_unnecessary_motion",
    "abrupt_or_unpredictable",
    "response_too_long",
    "inappropriate_direction",
    "recovery_or_task_resumption_problem",
    "grasp_place_release_disruption",
    "other",
)
REJECTION_REASON_LABELS_KO = {
    "too_close_or_late": "너무 가깝거나 회피가 늦었음",
    "excessive_or_unnecessary_motion": "움직임이 과도하거나 불필요했음",
    "abrupt_or_unpredictable": "움직임이 갑작스럽거나 예측하기 어려웠음",
    "response_too_long": "safety response가 너무 오래 지속되었음",
    "inappropriate_direction": "회피 방향이 적절하지 않았음",
    "recovery_or_task_resumption_problem": "Recovery 또는 작업 재개에 문제가 있었음",
    "grasp_place_release_disruption": "grasp/place/release를 방해했음",
    "other": "기타",
}

REALTIME_SAFETY_CONCERN = "realtime_safety_concern"
REALTIME_BEHAVIOR_ANOMALY = "realtime_behavior_anomaly"
REALTIME_MARKER_TYPES = (REALTIME_SAFETY_CONCERN, REALTIME_BEHAVIOR_ANOMALY)

_HANDS = ("left", "right")
_FACE_BUTTONS = {
    "left": ("x", "y"),
    "right": ("a", "b"),
}
_ALL_BUTTON_KEYS = ("left.x", "left.y", "right.a", "right.b")


def _exact_optional_bool(value: Any, *, name: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be True, False, 0, 1, or None")


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True, order=True)
class ButtonRef:
    """Canonical controller face-button reference."""

    hand: str
    button: str

    def __post_init__(self) -> None:
        hand = str(self.hand).strip().lower()
        button = str(self.button).strip().lower()
        if hand not in _HANDS:
            raise ValueError("button hand must be 'left' or 'right'")
        if button not in _FACE_BUTTONS[hand]:
            allowed = ", ".join(_FACE_BUTTONS[hand])
            raise ValueError(f"{hand} controller face button must be one of: {allowed}")
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "button", button)

    @property
    def key(self) -> str:
        return f"{self.hand}.{self.button}"

    @classmethod
    def parse(cls, value: "ButtonRef | str | Sequence[str]") -> "ButtonRef":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parts = value.strip().lower().replace("_", ".").split(".")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts = [str(part).strip().lower() for part in value]
        else:
            raise ValueError("button binding must be ButtonRef, 'hand.button', or (hand, button)")
        if len(parts) != 2:
            raise ValueError("button binding must identify hand and face button")
        return cls(parts[0], parts[1])


LEFT_X = ButtonRef("left", "x")
LEFT_Y = ButtonRef("left", "y")
RIGHT_A = ButtonRef("right", "a")
RIGHT_B = ButtonRef("right", "b")


@dataclass(frozen=True)
class ControllerButtonSnapshot:
    """One frame of tri-state XR face-button input.

    ``None`` means unknown/unavailable and is intentionally different from a
    known release (``False``).  A disconnected controller forces both of its
    face-button values to ``None``.
    """

    left_connected: Optional[bool] = None
    right_connected: Optional[bool] = None
    left_x: Optional[bool] = None
    left_y: Optional[bool] = None
    right_a: Optional[bool] = None
    right_b: Optional[bool] = None

    def __post_init__(self) -> None:
        values = {
            name: _exact_optional_bool(getattr(self, name), name=name)
            for name in (
                "left_connected",
                "right_connected",
                "left_x",
                "left_y",
                "right_a",
                "right_b",
            )
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if self.left_connected is not True:
            object.__setattr__(self, "left_x", None)
            object.__setattr__(self, "left_y", None)
        if self.right_connected is not True:
            object.__setattr__(self, "right_a", None)
            object.__setattr__(self, "right_b", None)

    def connected(self, hand: str) -> Optional[bool]:
        hand = str(hand).strip().lower()
        if hand not in _HANDS:
            raise ValueError("hand must be 'left' or 'right'")
        return getattr(self, f"{hand}_connected")

    def state(self, button: ButtonRef | str | Sequence[str]) -> Optional[bool]:
        ref = ButtonRef.parse(button)
        return getattr(self, f"{ref.hand}_{ref.button}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def coerce(cls, value: "ControllerButtonSnapshot | Mapping[str, Any]") -> "ControllerButtonSnapshot":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("button snapshot must be ControllerButtonSnapshot or a mapping")

        def nested(hand: str) -> Mapping[str, Any]:
            candidate = value.get(hand, {})
            return candidate if isinstance(candidate, Mapping) else {}

        def first(*keys: str) -> Any:
            for key in keys:
                if key in value:
                    return value[key]
            return None

        left = nested("left")
        right = nested("right")

        def face(hand: str, button: str, group: Mapping[str, Any]) -> Any:
            for key in (button, f"{hand}.{button}", f"{hand}_{button}"):
                if key in group:
                    return group[key]
                if key in value:
                    return value[key]
            return None

        left_x = face("left", "x", left)
        left_y = face("left", "y", left)
        right_a = face("right", "a", right)
        right_b = face("right", "b", right)
        left_connected = first("left_connected", "left.connected")
        right_connected = first("right_connected", "right.connected")
        if left_connected is None and "connected" in left:
            left_connected = left["connected"]
        if right_connected is None and "connected" in right:
            right_connected = right["connected"]
        if left_connected is None and any(value is not None for value in (left_x, left_y)):
            left_connected = True
        if right_connected is None and any(value is not None for value in (right_a, right_b)):
            right_connected = True
        return cls(
            left_connected=left_connected,
            right_connected=right_connected,
            left_x=left_x,
            left_y=left_y,
            right_a=right_a,
            right_b=right_b,
        )


@dataclass(frozen=True)
class FeedbackClocks:
    """Clock sample attached atomically to an input update."""

    sim_time_s: float
    monotonic_ns: int
    unix_ns: int
    control_step: int

    def __post_init__(self) -> None:
        sim_time_s = float(self.sim_time_s)
        if not math.isfinite(sim_time_s) or sim_time_s < 0.0:
            raise ValueError("sim_time_s must be finite and non-negative")
        object.__setattr__(self, "sim_time_s", sim_time_s)
        object.__setattr__(
            self,
            "monotonic_ns",
            _nonnegative_int(self.monotonic_ns, name="monotonic_ns"),
        )
        object.__setattr__(self, "unix_ns", _nonnegative_int(self.unix_ns, name="unix_ns"))
        object.__setattr__(
            self,
            "control_step",
            _nonnegative_int(self.control_step, name="control_step"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def now(cls, *, sim_time_s: float, control_step: int) -> "FeedbackClocks":
        return cls(
            sim_time_s=sim_time_s,
            monotonic_ns=time.monotonic_ns(),
            unix_ns=time.time_ns(),
            control_step=control_step,
        )

    @classmethod
    def coerce(cls, value: "FeedbackClocks | Mapping[str, Any]") -> "FeedbackClocks":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("clocks must be FeedbackClocks or a mapping")
        return cls(
            sim_time_s=value["sim_time_s"],
            monotonic_ns=value["monotonic_ns"],
            unix_ns=value["unix_ns"],
            control_step=value["control_step"],
        )


class DebouncedRisingEdge:
    """Fail-closed rising-edge detector for tri-state input.

    Startup, an unknown sample, and a disconnect all disarm the detector.  It
    must then observe ``release_frames`` consecutive known-release frames
    before a press can produce an edge.  A held button therefore cannot become
    a press merely because its controller reconnects.
    """

    def __init__(self, *, release_frames: int = 3) -> None:
        if isinstance(release_frames, bool) or int(release_frames) != release_frames:
            raise ValueError("release_frames must be an integer")
        if int(release_frames) < 1:
            raise ValueError("release_frames must be positive")
        self.release_frames = int(release_frames)
        self.reset()

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def stable_release_frames(self) -> int:
        return self._release_count

    def reset(self) -> None:
        self._armed = False
        self._release_count = 0

    def update(self, state: Optional[bool], *, connected: Optional[bool]) -> bool:
        connected = _exact_optional_bool(connected, name="connected")
        state = _exact_optional_bool(state, name="state")
        if connected is not True or state is None:
            self.reset()
            return False
        if state is False:
            self._release_count = min(self.release_frames, self._release_count + 1)
            if self._release_count >= self.release_frames:
                self._armed = True
            return False
        if not self._armed:
            self._release_count = 0
            return False
        self._armed = False
        self._release_count = 0
        return True


class ButtonEdgeBank:
    """Run one :class:`DebouncedRisingEdge` per face button."""

    def __init__(self, *, release_frames: int = 3) -> None:
        self.release_frames = int(release_frames)
        self._edges = {
            key: DebouncedRisingEdge(release_frames=self.release_frames)
            for key in _ALL_BUTTON_KEYS
        }

    def reset(self) -> None:
        for edge in self._edges.values():
            edge.reset()

    def update(
        self, snapshot: ControllerButtonSnapshot | Mapping[str, Any]
    ) -> tuple[str, ...]:
        sample = ControllerButtonSnapshot.coerce(snapshot)
        emitted = []
        for key in _ALL_BUTTON_KEYS:
            ref = ButtonRef.parse(key)
            if self._edges[key].update(
                sample.state(ref), connected=sample.connected(ref.hand)
            ):
                emitted.append(key)
        return tuple(emitted)


class ControllerInputUnavailable(RuntimeError):
    """Raised when a required XR controller or face button is unavailable."""

    def __init__(
        self,
        reasons: Iterable[str],
        snapshot: ControllerButtonSnapshot,
    ) -> None:
        self.reasons = tuple(str(reason) for reason in reasons)
        self.snapshot = snapshot
        super().__init__("controller input unavailable: " + ", ".join(self.reasons))


class ControllerInputWatchdog:
    """Require live, known input for both controllers and all face buttons.

    Pose validity is deliberately not used as a proxy for button health: an
    XR runtime can keep delivering controller or external-hand poses while
    its gesture/input channel is unavailable.
    """

    def __init__(
        self,
        *,
        required_buttons: Sequence[ButtonRef | str | Sequence[str]] = (
            LEFT_X,
            LEFT_Y,
            RIGHT_A,
            RIGHT_B,
        ),
    ) -> None:
        refs = tuple(ButtonRef.parse(value) for value in required_buttons)
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("required controller buttons must be non-empty and unique")
        required_hands = {ref.hand for ref in refs}
        if required_hands != set(_HANDS):
            raise ValueError("required controller inputs must cover both hands")
        self.required_buttons = refs

    def unavailable_reasons(
        self,
        snapshot: ControllerButtonSnapshot | Mapping[str, Any],
    ) -> tuple[str, ...]:
        sample = ControllerButtonSnapshot.coerce(snapshot)
        reasons: list[str] = []
        for hand in _HANDS:
            connected = sample.connected(hand)
            if connected is not True:
                status = "disconnected" if connected is False else "connection_unknown"
                reasons.append(f"{hand}_controller_{status}")
        for ref in self.required_buttons:
            if sample.connected(ref.hand) is True and sample.state(ref) is None:
                reasons.append(f"{ref.key}_state_unknown")
        return tuple(reasons)

    def require_available(
        self,
        snapshot: ControllerButtonSnapshot | Mapping[str, Any],
    ) -> ControllerButtonSnapshot:
        sample = ControllerButtonSnapshot.coerce(snapshot)
        reasons = self.unavailable_reasons(sample)
        if reasons:
            raise ControllerInputUnavailable(reasons, sample)
        return sample


class EmergencyAbortMonitor:
    """Require a configurable two-button hold before requesting an abort.

    Unknown input or either controller disconnect immediately clears the hold.
    The monitor is level based rather than edge based, so an accidental tap
    cannot terminate a participant session.
    """

    def __init__(
        self,
        *,
        buttons: Sequence[ButtonRef | str | Sequence[str]] = (RIGHT_A, RIGHT_B),
        hold_s: float = 2.0,
    ) -> None:
        refs = tuple(ButtonRef.parse(value) for value in buttons)
        if len(refs) != 2 or len(set(refs)) != 2:
            raise ValueError("emergency abort requires two distinct buttons")
        hold_s = float(hold_s)
        if not math.isfinite(hold_s) or hold_s < 0.5:
            raise ValueError("emergency abort hold_s must be at least 0.5 seconds")
        self.buttons = refs
        self.hold_ns = int(round(hold_s * 1e9))
        self._started_ns: Optional[int] = None
        self._latched = False

    def reset(self) -> None:
        self._started_ns = None
        self._latched = False

    def update(
        self,
        snapshot: ControllerButtonSnapshot | Mapping[str, Any],
        *,
        monotonic_ns: int,
    ) -> bool:
        sample = ControllerButtonSnapshot.coerce(snapshot)
        now = _nonnegative_int(monotonic_ns, name="monotonic_ns")
        states = [sample.state(ref) for ref in self.buttons]
        connected = [sample.connected(ref.hand) for ref in self.buttons]
        if any(value is not True for value in connected) or any(value is not True for value in states):
            self._started_ns = None
            return False
        if self._latched:
            return False
        if self._started_ns is None:
            self._started_ns = now
            return False
        if now - self._started_ns >= self.hold_ns:
            self._latched = True
            return True
        return False


@dataclass(frozen=True)
class RealtimeButtonMapping:
    """Opposite-controller mapping selected by the crossed hand."""

    crossing_left_safety: ButtonRef = RIGHT_A
    crossing_left_anomaly: ButtonRef = RIGHT_B
    crossing_right_safety: ButtonRef = LEFT_X
    crossing_right_anomaly: ButtonRef = LEFT_Y

    def __post_init__(self) -> None:
        fields = (
            "crossing_left_safety",
            "crossing_left_anomaly",
            "crossing_right_safety",
            "crossing_right_anomaly",
        )
        for field in fields:
            object.__setattr__(self, field, ButtonRef.parse(getattr(self, field)))
        for crossed_hand in _HANDS:
            safety, anomaly = self.for_crossing(crossed_hand)
            expected = "right" if crossed_hand == "left" else "left"
            if safety.hand != expected or anomaly.hand != expected:
                raise ValueError(
                    f"{crossed_hand}-hand crossing must use the opposite ({expected}) controller"
                )
            if safety == anomaly:
                raise ValueError("safety and anomaly markers require separate face buttons")

    def for_crossing(self, crossing_hand: str) -> tuple[ButtonRef, ButtonRef]:
        hand = str(crossing_hand).strip().lower()
        if hand == "left":
            return self.crossing_left_safety, self.crossing_left_anomaly
        if hand == "right":
            return self.crossing_right_safety, self.crossing_right_anomaly
        raise ValueError("crossing_hand must be 'left' or 'right'")

    def event_for_button(self, crossing_hand: str, button_key: str) -> Optional[str]:
        safety, anomaly = self.for_crossing(crossing_hand)
        if button_key == safety.key:
            return REALTIME_SAFETY_CONCERN
        if button_key == anomaly.key:
            return REALTIME_BEHAVIOR_ANOMALY
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RealtimeButtonMapping":
        if not isinstance(value, Mapping):
            raise TypeError("realtime mapping override must be a mapping")
        expected_hands = {"left", "right"}
        if set(value) != expected_hands:
            raise ValueError(
                "realtime mapping must contain exactly the left and right crossing groups"
            )
        expected_markers = {
            REALTIME_SAFETY_CONCERN,
            REALTIME_BEHAVIOR_ANOMALY,
        }
        groups: dict[str, Mapping[str, Any]] = {}
        for hand in sorted(expected_hands):
            group = value[hand]
            if not isinstance(group, Mapping) or set(group) != expected_markers:
                raise ValueError(
                    f"realtime mapping {hand} group must contain exactly "
                    f"{sorted(expected_markers)!r}"
                )
            groups[hand] = group
        return cls(
            crossing_left_safety=ButtonRef.parse(
                groups["left"][REALTIME_SAFETY_CONCERN]
            ),
            crossing_left_anomaly=ButtonRef.parse(
                groups["left"][REALTIME_BEHAVIOR_ANOMALY]
            ),
            crossing_right_safety=ButtonRef.parse(
                groups["right"][REALTIME_SAFETY_CONCERN]
            ),
            crossing_right_anomaly=ButtonRef.parse(
                groups["right"][REALTIME_BEHAVIOR_ANOMALY]
            ),
        )


@dataclass(frozen=True)
class RealtimeMarker:
    marker_id: str
    marker_type: str
    crossing_hand: str
    controller_hand: str
    button: str
    sim_time_s: float
    monotonic_ns: int
    unix_ns: int
    control_step: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealtimeMarkerInput:
    """Standalone realtime marker detector using the v1 opposite-hand mapping."""

    def __init__(
        self,
        *,
        mapping: RealtimeButtonMapping | Mapping[str, Any] | None = None,
        crossing_hand: Optional[str] = None,
        release_frames: int = 3,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.mapping = (
            RealtimeButtonMapping()
            if mapping is None
            else (
                mapping
                if isinstance(mapping, RealtimeButtonMapping)
                else RealtimeButtonMapping.from_mapping(mapping)
            )
        )
        self._buttons = ButtonEdgeBank(release_frames=release_frames)
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._crossing_hand: Optional[str] = None
        self.set_crossing_hand(crossing_hand)

    @property
    def crossing_hand(self) -> Optional[str]:
        return self._crossing_hand

    def reset_input(self) -> None:
        self._buttons.reset()

    def set_crossing_hand(self, hand: Optional[str]) -> None:
        normalized = None if hand is None else str(hand).strip().lower()
        if normalized is not None and normalized not in _HANDS:
            raise ValueError("crossing hand must be 'left', 'right', or None")
        if normalized != self._crossing_hand:
            self._crossing_hand = normalized
            self.reset_input()

    def records_for_edges(
        self, edges: Iterable[str], clocks: FeedbackClocks | Mapping[str, Any]
    ) -> tuple[RealtimeMarker, ...]:
        sample_time = FeedbackClocks.coerce(clocks)
        if self._crossing_hand is None:
            return ()
        records = []
        for button_key in edges:
            marker_type = self.mapping.event_for_button(self._crossing_hand, button_key)
            if marker_type is None:
                continue
            ref = ButtonRef.parse(button_key)
            records.append(
                RealtimeMarker(
                    marker_id=str(self._id_factory()),
                    marker_type=marker_type,
                    crossing_hand=self._crossing_hand,
                    controller_hand=ref.hand,
                    button=ref.button,
                    **sample_time.as_dict(),
                )
            )
        return tuple(records)

    def update(
        self,
        button_snapshot: ControllerButtonSnapshot | Mapping[str, Any],
        clocks: FeedbackClocks | Mapping[str, Any],
    ) -> tuple[RealtimeMarker, ...]:
        return self.records_for_edges(self._buttons.update(button_snapshot), clocks)


@dataclass(frozen=True)
class QuestionnaireButtonMapping:
    previous: ButtonRef = LEFT_X
    next: ButtonRef = LEFT_Y
    select: ButtonRef = RIGHT_A
    submit: ButtonRef = RIGHT_B
    back_chord: tuple[ButtonRef, ButtonRef] = (LEFT_X, LEFT_Y)

    def __post_init__(self) -> None:
        for field in ("previous", "next", "select", "submit"):
            object.__setattr__(self, field, ButtonRef.parse(getattr(self, field)))
        refs = (self.previous, self.next, self.select, self.submit)
        if len(set(refs)) != len(refs):
            raise ValueError("questionnaire controls must use four distinct face buttons")
        chord = tuple(ButtonRef.parse(value) for value in self.back_chord)
        if len(chord) != 2 or len(set(chord)) != 2:
            raise ValueError("back_chord must contain two distinct face buttons")
        object.__setattr__(self, "back_chord", chord)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuestionnaireButtonMapping":
        if not isinstance(value, Mapping):
            raise TypeError("questionnaire navigation must be a mapping")
        required = (
            "previous",
            "next",
            "select",
            "confirm",
            "back_chord",
            "emergency_abort_chord",
            "emergency_abort_hold_s",
        )
        missing = [name for name in required if name not in value]
        known = set(required)
        extra = sorted(set(value) - known)
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unknown " + ", ".join(extra))
            raise ValueError("invalid questionnaire navigation: " + "; ".join(detail))
        back = value["back_chord"]
        if not isinstance(back, Sequence) or isinstance(back, (str, bytes)):
            raise ValueError("back_chord must be a two-button sequence")
        return cls(
            previous=ButtonRef.parse(value["previous"]),
            next=ButtonRef.parse(value["next"]),
            select=ButtonRef.parse(value["select"]),
            submit=ButtonRef.parse(value["confirm"]),
            back_chord=tuple(ButtonRef.parse(item) for item in back),  # type: ignore[arg-type]
        )

    @property
    def controls_text(self) -> str:
        return (
            f"{self.previous.key.upper()}: 이전   "
            f"{self.next.key.upper()}: 다음   "
            f"{self.select.key.upper()}: 선택   "
            f"{self.submit.key.upper()}: 확인\n"
            f"{'+'.join(value.key.upper() for value in self.back_chord)}: 뒤로"
        )

    @property
    def back_keys(self) -> frozenset[str]:
        return frozenset(value.key for value in self.back_chord)

    def action_for_edge(self, edge: str) -> Optional[str]:
        for action in ("previous", "next", "select", "submit"):
            if edge == getattr(self, action).key:
                return action
        return None


@dataclass(frozen=True)
class QuestionnaireAnswer:
    question_id: str
    answer_status: str
    value: Any
    prompt_shown_simulation_time_s: float
    prompt_shown_monotonic_ns: int
    prompt_shown_unix_ns: int
    prompt_shown_control_step: int
    first_input_simulation_time_s: float
    first_input_monotonic_ns: int
    first_input_unix_ns: int
    first_input_control_step: int
    confirmed_simulation_time_s: float
    confirmed_monotonic_ns: int
    confirmed_unix_ns: int
    confirmed_control_step: int
    response_latency_ms: float
    input_device: str
    back_correction_count: int
    accidental_input_count: int

    def __post_init__(self) -> None:
        if self.question_id not in {
            Q1_ID,
            *LIKERT_QUESTIONS_KO,
            REJECTION_REASON_ID,
        }:
            raise ValueError("questionnaire answer has an unknown question_id")
        if self.answer_status not in {"confirmed", "unconfirmed"}:
            raise ValueError("questionnaire answer has an invalid answer_status")
        for prefix in ("prompt_shown",):
            simulation_time_s = float(
                getattr(self, f"{prefix}_simulation_time_s")
            )
            if not math.isfinite(simulation_time_s) or simulation_time_s < 0.0:
                raise ValueError(f"{prefix} simulation time is invalid")
            object.__setattr__(
                self, f"{prefix}_simulation_time_s", simulation_time_s
            )
            for suffix in ("monotonic_ns", "unix_ns", "control_step"):
                name = f"{prefix}_{suffix}"
                object.__setattr__(
                    self,
                    name,
                    _nonnegative_int(getattr(self, name), name=name),
                )
        latency_ms = float(self.response_latency_ms)
        if self.answer_status == "confirmed":
            for prefix in ("first_input", "confirmed"):
                simulation_time_s = float(
                    getattr(self, f"{prefix}_simulation_time_s")
                )
                if not math.isfinite(simulation_time_s) or simulation_time_s < 0.0:
                    raise ValueError(f"{prefix} simulation time is invalid")
                object.__setattr__(
                    self, f"{prefix}_simulation_time_s", simulation_time_s
                )
                for suffix in ("monotonic_ns", "unix_ns", "control_step"):
                    name = f"{prefix}_{suffix}"
                    object.__setattr__(
                        self,
                        name,
                        _nonnegative_int(getattr(self, name), name=name),
                    )
            for suffix in (
                "simulation_time_s",
                "monotonic_ns",
                "unix_ns",
                "control_step",
            ):
                shown = getattr(self, f"prompt_shown_{suffix}")
                first = getattr(self, f"first_input_{suffix}")
                confirmed = getattr(self, f"confirmed_{suffix}")
                if not shown <= first <= confirmed:
                    raise ValueError(
                        "questionnaire answer clocks must satisfy "
                        "prompt_shown <= first_input <= confirmed"
                    )
            expected_latency_ms = (
                self.confirmed_monotonic_ns - self.prompt_shown_monotonic_ns
            ) / 1e6
            if (
                not math.isfinite(latency_ms)
                or latency_ms < 0.0
                or not math.isclose(
                    latency_ms, expected_latency_ms, rel_tol=0.0, abs_tol=1e-6
                )
            ):
                raise ValueError("questionnaire answer latency disagrees with clocks")
        else:
            if self.value is not None:
                raise ValueError("unconfirmed questionnaire answer value must be null")
            no_first_input = (
                float(self.first_input_simulation_time_s) == -1.0
                and int(self.first_input_monotonic_ns) == 0
                and int(self.first_input_unix_ns) == 0
                and int(self.first_input_control_step) == -1
            )
            if not no_first_input:
                first_simulation_time_s = float(
                    self.first_input_simulation_time_s
                )
                if (
                    not math.isfinite(first_simulation_time_s)
                    or first_simulation_time_s
                    < self.prompt_shown_simulation_time_s
                ):
                    raise ValueError("unconfirmed first-input simulation time is invalid")
                object.__setattr__(
                    self,
                    "first_input_simulation_time_s",
                    first_simulation_time_s,
                )
                for suffix in ("monotonic_ns", "unix_ns", "control_step"):
                    name = f"first_input_{suffix}"
                    value = _nonnegative_int(getattr(self, name), name=name)
                    if value < getattr(self, f"prompt_shown_{suffix}"):
                        raise ValueError("unconfirmed first-input clock precedes prompt")
                    object.__setattr__(self, name, value)
            confirmed_sentinel = (
                float(self.confirmed_simulation_time_s) == -1.0
                and int(self.confirmed_monotonic_ns) == 0
                and int(self.confirmed_unix_ns) == 0
                and int(self.confirmed_control_step) == -1
            )
            if not confirmed_sentinel or latency_ms != -1.0:
                raise ValueError(
                    "unconfirmed questionnaire answer requires confirmation/latency sentinels"
                )
        object.__setattr__(self, "response_latency_ms", latency_ms)
        if self.input_device != "vr_controller":
            raise ValueError("questionnaire answer input_device must be vr_controller")
        for name in ("back_correction_count", "accidental_input_count"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionnaireResult:
    questionnaire_id: str
    context_id: str
    schema_version: str
    completion_status: str
    response_disposition: str
    q1_response: Optional[str]
    q2: Optional[int]
    q3: Optional[int]
    q4: Optional[int]
    q5: Optional[int]
    q6: Optional[int]
    rejection_reason_ids: Optional[tuple[str, ...]]
    missing_question_ids: tuple[str, ...]
    answers: tuple[QuestionnaireAnswer, ...]
    started_sim_time_s: float
    started_monotonic_ns: int
    started_unix_ns: int
    started_control_step: int
    first_input_sim_time_s: float
    first_input_monotonic_ns: int
    first_input_unix_ns: int
    first_input_control_step: int
    back_correction_count: int
    accidental_input_count: int
    deadline_monotonic_ns: int
    completed_sim_time_s: float
    completed_monotonic_ns: int
    completed_unix_ns: int
    completed_control_step: int

    @property
    def study_complete(self) -> bool:
        return self.completion_status == "completed"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["answers"] = [answer.as_dict() for answer in self.answers]
        payload["rejection_reason_ids"] = (
            None
            if self.rejection_reason_ids is None
            else list(self.rejection_reason_ids)
        )
        payload["missing_question_ids"] = list(self.missing_question_ids)
        return payload


class QuestionnaireFSM:
    """Six-question controller FSM with a conditional reason page."""

    _MAIN_STAGES = ("q1", "q2", "q3", "q4", "q5", "q6")

    def __init__(
        self,
        *,
        timeout_s: float = 60.0,
        release_frames: int = 3,
        button_mapping: QuestionnaireButtonMapping | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("questionnaire timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        self.button_mapping = button_mapping or QuestionnaireButtonMapping()
        self._buttons = ButtonEdgeBank(release_frames=release_frames)
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._state = "idle"
        self._questionnaire_id = ""
        self._context_id = ""
        self._started: Optional[FeedbackClocks] = None
        self._deadline_ns = 0
        self._answers: dict[str, QuestionnaireAnswer] = {}
        self._first_input: Optional[FeedbackClocks] = None
        self._back_correction_count = 0
        self._accidental_input_count = 0
        self._prompt_shown: Optional[FeedbackClocks] = None
        self._question_first_input: Optional[FeedbackClocks] = None
        self._question_accidental_input_count = 0
        self._question_back_correction_counts: dict[str, int] = {}
        self._cursor = 0
        self._pending_value: Any = None
        self._selected_reasons: set[str] = set()
        self._notice = ""
        self._result: Optional[QuestionnaireResult] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def active(self) -> bool:
        return self._state in (*self._MAIN_STAGES, REJECTION_REASON_ID)

    @property
    def result(self) -> Optional[QuestionnaireResult]:
        return self._result

    @property
    def current_question_id(self) -> Optional[str]:
        return self._state if self.active else None

    @property
    def selected_reasons(self) -> tuple[str, ...]:
        return tuple(reason for reason in REJECTION_REASON_IDS if reason in self._selected_reasons)

    @property
    def renderer_text(self) -> str:
        if self._state == "idle":
            return "명시적 피드백 설문을 기다리는 중입니다."
        if self._state == "completed":
            return "응답이 기록되었습니다."
        if self._state in ("timed_out", "abstained"):
            return "응답 없음으로 기록되었습니다."
        controls = self.button_mapping.controls_text
        notice = f"\n\n{self._notice}" if self._notice else ""
        if self._state == "q1":
            options = tuple(
                (code, Q1_OPTION_LABELS_KO[code]) for code in Q1_RESPONSE_CODES
            )
            return self._render_options("Q1/6", Q1_TEXT_KO, options, controls) + notice
        if self._state in LIKERT_QUESTIONS_KO:
            options = tuple((str(value), str(value)) for value in LIKERT_VALUES)
            anchors = (
                "1: 전혀 그렇지 않음   5: 매우 그러함"
                if self._state != "q6"
                else "1: 전혀 확신하지 못함   5: 매우 확신함"
            )
            ordinal = int(self._state[1:])
            return (
                self._render_options(
                    f"Q{ordinal}/6", LIKERT_QUESTIONS_KO[self._state], options, controls
                )
                + f"\n{anchors}"
                + notice
            )
        options = tuple(
            (
                reason,
                ("[선택] " if reason in self._selected_reasons else "[    ] ")
                + REJECTION_REASON_LABELS_KO[reason],
            )
            for reason in REJECTION_REASON_IDS
        )
        return (
            self._render_options(
                "수정 이유 (복수 선택)",
                "수정이 필요하다고 판단한 이유를 하나 이상 선택해 주세요.",
                options,
                controls,
            )
            + notice
        )

    def _render_options(
        self,
        title: str,
        question: str,
        options: Sequence[tuple[str, str]],
        controls: str,
    ) -> str:
        rendered = []
        for index, (value, label) in enumerate(options):
            pending = bool(
                self._state != REJECTION_REASON_ID
                and self._pending_value is not None
                and value == str(self._pending_value)
            )
            selection_prefix = (
                ""
                if self._state == REJECTION_REASON_ID
                else ("[선택] " if pending else "[    ] ")
            )
            rendered.append(
                ("> " if index == self._cursor else "  ")
                + selection_prefix
                + label
            )
        return f"{title}\n\n{question}\n\n" + "\n".join(rendered) + f"\n\n{controls}"

    def reset_input(self) -> None:
        self._buttons.reset()

    def dismiss_terminal_page(self) -> None:
        """Return a completed/timeout page to idle without erasing its audit result."""

        if self.active:
            raise RuntimeError("cannot dismiss an active questionnaire")
        if self._state != "idle":
            self._state = "idle"
            self.reset_input()

    def start(
        self,
        clocks: FeedbackClocks | Mapping[str, Any],
        *,
        context_id: str = "",
        timeout_s: Optional[float] = None,
    ) -> str:
        if self.active:
            raise RuntimeError("questionnaire is already active")
        duration = self.timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("questionnaire timeout_s must be finite and positive")
        sample_time = FeedbackClocks.coerce(clocks)
        self._state = "q1"
        self._questionnaire_id = str(self._id_factory())
        self._context_id = str(context_id)
        self._started = sample_time
        self._deadline_ns = sample_time.monotonic_ns + int(round(duration * 1e9))
        self._answers = {}
        self._first_input = None
        self._back_correction_count = 0
        self._accidental_input_count = 0
        self._prompt_shown = None
        self._question_first_input = None
        self._question_accidental_input_count = 0
        self._question_back_correction_counts = {}
        self._cursor = 0
        self._pending_value = None
        self._selected_reasons = set()
        self._notice = ""
        self._result = None
        self._begin_prompt(sample_time)
        self.reset_input()
        return self._questionnaire_id

    def update(
        self,
        button_snapshot: ControllerButtonSnapshot | Mapping[str, Any],
        clocks: FeedbackClocks | Mapping[str, Any],
    ) -> Optional[QuestionnaireResult]:
        edges = self._buttons.update(button_snapshot)
        return self.handle_edges(edges, clocks)

    def handle_edges(
        self,
        edges: Iterable[str],
        clocks: FeedbackClocks | Mapping[str, Any],
    ) -> Optional[QuestionnaireResult]:
        sample_time = FeedbackClocks.coerce(clocks)
        if not self.active:
            return None
        if sample_time.monotonic_ns >= self._deadline_ns:
            return self._finish("timed_out", sample_time)
        edges = tuple(str(edge) for edge in edges)
        if edges and self._first_input is None:
            self._first_input = sample_time
        if edges and self._question_first_input is None:
            self._question_first_input = sample_time
        edge_set = frozenset(edges)
        if edge_set == self.button_mapping.back_keys:
            self._go_back(sample_time)
            self._back_correction_count += 1
            return None
        actions = tuple(
            action
            for action in (self.button_mapping.action_for_edge(edge) for edge in edge_set)
            if action is not None
        )
        if not actions:
            return None
        if len(actions) != 1:
            self._notice = "동시에 한 버튼만 눌러 주세요."
            self._accidental_input_count += 1
            self._question_accidental_input_count += 1
            return None
        self._notice = ""
        action = actions[0]
        option_count = (
            len(Q1_RESPONSE_CODES)
            if self._state == "q1"
            else (
                len(LIKERT_VALUES)
                if self._state in LIKERT_QUESTIONS_KO
                else len(REJECTION_REASON_IDS)
            )
        )
        if action == "previous":
            self._cursor = (self._cursor - 1) % option_count
            return None
        if action == "next":
            self._cursor = (self._cursor + 1) % option_count
            return None
        if self._state == REJECTION_REASON_ID:
            if action == "select":
                reason = REJECTION_REASON_IDS[self._cursor]
                if reason in self._selected_reasons:
                    self._selected_reasons.remove(reason)
                else:
                    self._selected_reasons.add(reason)
                return None
            if not self._selected_reasons:
                self._notice = "수정 이유를 하나 이상 선택해야 합니다."
                self._accidental_input_count += 1
                self._question_accidental_input_count += 1
                return None
            self._record_answer(REJECTION_REASON_ID, self.selected_reasons, sample_time)
            return self._finish("completed", sample_time)

        if action == "select":
            self._pending_value = (
                Q1_RESPONSE_CODES[self._cursor]
                if self._state == "q1"
                else LIKERT_VALUES[self._cursor]
            )
            self._notice = "선택되었습니다. B 버튼으로 확인해 주세요."
            return None
        if action != "submit":
            return None
        if self._pending_value is None:
            self._notice = "A 버튼으로 응답을 먼저 선택해 주세요."
            self._accidental_input_count += 1
            self._question_accidental_input_count += 1
            return None
        value: Any = self._pending_value
        current = self._state
        self._record_answer(current, value, sample_time)
        current_index = self._MAIN_STAGES.index(current)
        if current_index + 1 < len(self._MAIN_STAGES):
            self._state = self._MAIN_STAGES[current_index + 1]
            self._cursor = 0
            self._pending_value = None
            self._begin_prompt(sample_time)
            return None
        if self._answers[Q1_ID].value == "needs_modification":
            self._state = REJECTION_REASON_ID
            self._cursor = 0
            self._pending_value = None
            self._begin_prompt(sample_time)
            return None
        return self._finish("completed", sample_time)

    def _go_back(self, clocks: FeedbackClocks) -> None:
        if self._state == "q1":
            self._notice = "첫 질문입니다."
            return
        if self._state == REJECTION_REASON_ID:
            target = "q6"
            self._selected_reasons.clear()
            self._answers.pop(REJECTION_REASON_ID, None)
        else:
            current_index = self._MAIN_STAGES.index(self._state)
            target = self._MAIN_STAGES[current_index - 1]
        target_index = self._MAIN_STAGES.index(target)
        for question_id in self._MAIN_STAGES[target_index:]:
            stored = Q1_ID if question_id == "q1" else question_id
            if self._answers.pop(stored, None) is not None:
                self._question_back_correction_counts[stored] = (
                    self._question_back_correction_counts.get(stored, 0) + 1
                )
        self._state = target
        self._cursor = 0
        self._pending_value = None
        self._notice = "이전 응답을 다시 선택해 주세요."
        self._begin_prompt(clocks)

    def abstain(
        self, clocks: FeedbackClocks | Mapping[str, Any]
    ) -> QuestionnaireResult:
        if not self.active:
            raise RuntimeError("no active questionnaire to abstain from")
        return self._finish("abstained", FeedbackClocks.coerce(clocks))

    def _begin_prompt(self, clocks: FeedbackClocks) -> None:
        """Start a new visible prompt cycle for the current question."""

        self._prompt_shown = clocks
        self._question_first_input = None
        self._question_accidental_input_count = 0

    def _record_answer(self, question_id: str, value: Any, clocks: FeedbackClocks) -> None:
        stored_id = Q1_ID if question_id == "q1" else question_id
        if self._prompt_shown is None or self._question_first_input is None:
            raise RuntimeError("answer audit clocks are incomplete")
        prompt = self._prompt_shown
        first = self._question_first_input
        self._answers[stored_id] = QuestionnaireAnswer(
            question_id=stored_id,
            answer_status="confirmed",
            value=value,
            prompt_shown_simulation_time_s=prompt.sim_time_s,
            prompt_shown_monotonic_ns=prompt.monotonic_ns,
            prompt_shown_unix_ns=prompt.unix_ns,
            prompt_shown_control_step=prompt.control_step,
            first_input_simulation_time_s=first.sim_time_s,
            first_input_monotonic_ns=first.monotonic_ns,
            first_input_unix_ns=first.unix_ns,
            first_input_control_step=first.control_step,
            confirmed_simulation_time_s=clocks.sim_time_s,
            confirmed_monotonic_ns=clocks.monotonic_ns,
            confirmed_unix_ns=clocks.unix_ns,
            confirmed_control_step=clocks.control_step,
            response_latency_ms=(clocks.monotonic_ns - prompt.monotonic_ns) / 1e6,
            input_device="vr_controller",
            back_correction_count=self._question_back_correction_counts.get(
                stored_id, 0
            ),
            accidental_input_count=self._question_accidental_input_count,
        )

    def _unconfirmed_answer(self) -> QuestionnaireAnswer:
        """Preserve the visible prompt cycle without fabricating an answer."""

        if not self.active or self._prompt_shown is None:
            raise RuntimeError("unconfirmed prompt audit is unavailable")
        stored_id = Q1_ID if self._state == "q1" else self._state
        prompt = self._prompt_shown
        first = self._question_first_input
        return QuestionnaireAnswer(
            question_id=stored_id,
            answer_status="unconfirmed",
            value=None,
            prompt_shown_simulation_time_s=prompt.sim_time_s,
            prompt_shown_monotonic_ns=prompt.monotonic_ns,
            prompt_shown_unix_ns=prompt.unix_ns,
            prompt_shown_control_step=prompt.control_step,
            first_input_simulation_time_s=(
                -1.0 if first is None else first.sim_time_s
            ),
            first_input_monotonic_ns=(0 if first is None else first.monotonic_ns),
            first_input_unix_ns=(0 if first is None else first.unix_ns),
            first_input_control_step=(-1 if first is None else first.control_step),
            confirmed_simulation_time_s=-1.0,
            confirmed_monotonic_ns=0,
            confirmed_unix_ns=0,
            confirmed_control_step=-1,
            response_latency_ms=-1.0,
            input_device="vr_controller",
            back_correction_count=self._question_back_correction_counts.get(
                stored_id, 0
            ),
            accidental_input_count=self._question_accidental_input_count,
        )

    def _finish(self, status: str, clocks: FeedbackClocks) -> QuestionnaireResult:
        if self._started is None:
            raise RuntimeError("questionnaire has no start clock")
        if status not in ("completed", "timed_out", "abstained"):
            raise ValueError(f"unsupported questionnaire terminal status: {status}")
        q1 = self._answers.get(Q1_ID)
        likert = {key: self._answers.get(key) for key in LIKERT_QUESTIONS_KO}
        reasons = self._answers.get(REJECTION_REASON_ID)
        required = [Q1_ID, *LIKERT_QUESTIONS_KO]
        if q1 is not None and q1.value == "needs_modification":
            required.append(REJECTION_REASON_ID)
        missing = tuple(question_id for question_id in required if question_id not in self._answers)
        if status == "completed" and missing:
            raise RuntimeError("cannot complete a questionnaire with missing required answers")
        ordered_confirmed_answers = tuple(
            self._answers[key]
            for key in (Q1_ID, *LIKERT_QUESTIONS_KO, REJECTION_REASON_ID)
            if key in self._answers
        )
        ordered_answers = (
            ordered_confirmed_answers
            if status == "completed"
            else (*ordered_confirmed_answers, self._unconfirmed_answer())
        )
        self._result = QuestionnaireResult(
            questionnaire_id=self._questionnaire_id,
            context_id=self._context_id,
            schema_version=QUESTIONNAIRE_SCHEMA_VERSION,
            completion_status=status,
            response_disposition=(
                "missing_abstain"
                if status != "completed"
                else (
                    "uncertain_abstain"
                    if q1 is not None and q1.value == "uncertain"
                    else "answered"
                )
            ),
            q1_response=None if q1 is None else str(q1.value),
            q2=None if likert["q2"] is None else int(likert["q2"].value),
            q3=None if likert["q3"] is None else int(likert["q3"].value),
            q4=None if likert["q4"] is None else int(likert["q4"].value),
            q5=None if likert["q5"] is None else int(likert["q5"].value),
            q6=None if likert["q6"] is None else int(likert["q6"].value),
            rejection_reason_ids=(
                tuple(reasons.value)
                if reasons is not None
                else (
                    None
                    if q1 is None or q1.value == "needs_modification"
                    else ()
                )
            ),
            missing_question_ids=missing,
            answers=ordered_answers,
            started_sim_time_s=self._started.sim_time_s,
            started_monotonic_ns=self._started.monotonic_ns,
            started_unix_ns=self._started.unix_ns,
            started_control_step=self._started.control_step,
            first_input_sim_time_s=(
                -1.0 if self._first_input is None else self._first_input.sim_time_s
            ),
            first_input_monotonic_ns=(
                0 if self._first_input is None else self._first_input.monotonic_ns
            ),
            first_input_unix_ns=(
                0 if self._first_input is None else self._first_input.unix_ns
            ),
            first_input_control_step=(
                -1 if self._first_input is None else self._first_input.control_step
            ),
            back_correction_count=self._back_correction_count,
            accidental_input_count=self._accidental_input_count,
            deadline_monotonic_ns=self._deadline_ns,
            completed_sim_time_s=clocks.sim_time_s,
            completed_monotonic_ns=clocks.monotonic_ns,
            completed_unix_ns=clocks.unix_ns,
            completed_control_step=clocks.control_step,
        )
        self._state = status
        self.reset_input()
        return self._result


@dataclass(frozen=True)
class FeedbackUpdate:
    rising_edges: tuple[str, ...]
    realtime_markers: tuple[RealtimeMarker, ...]
    questionnaire_result: Optional[QuestionnaireResult]
    questionnaire_state: str
    renderer_text: str


class XRFeedbackController:
    """Single update entry point for realtime markers and the questionnaire.

    Realtime marker input is disabled while a questionnaire is active, so the
    same physical button press cannot produce both kinds of feedback.
    """

    def __init__(
        self,
        *,
        realtime_mapping: RealtimeButtonMapping | Mapping[str, Any] | None = None,
        questionnaire_mapping: QuestionnaireButtonMapping | Mapping[str, Any] | None = None,
        questionnaire_timeout_s: float = 60.0,
        release_frames: int = 3,
        marker_id_factory: Callable[[], str] | None = None,
        questionnaire_id_factory: Callable[[], str] | None = None,
        renderer: Any = None,
    ) -> None:
        self._buttons = ButtonEdgeBank(release_frames=release_frames)
        self.realtime = RealtimeMarkerInput(
            mapping=realtime_mapping,
            release_frames=release_frames,
            id_factory=marker_id_factory,
        )
        if questionnaire_mapping is None:
            resolved_questionnaire_mapping = QuestionnaireButtonMapping()
        elif isinstance(questionnaire_mapping, QuestionnaireButtonMapping):
            resolved_questionnaire_mapping = questionnaire_mapping
        else:
            resolved_questionnaire_mapping = QuestionnaireButtonMapping.from_mapping(
                questionnaire_mapping
            )
        self.questionnaire = QuestionnaireFSM(
            timeout_s=questionnaire_timeout_s,
            release_frames=release_frames,
            button_mapping=resolved_questionnaire_mapping,
            id_factory=questionnaire_id_factory,
        )
        self._renderer = renderer
        self._instruction_text = ""
        self._last_monotonic_ns: Optional[int] = None
        self._last_rendered_text = ""
        self._render()

    @property
    def renderer_text(self) -> str:
        if self.questionnaire.state != "idle":
            return self.questionnaire.renderer_text
        hand = self.realtime.crossing_hand
        if hand is None:
            base = "실시간 피드백 입력 대기 중"
            return (
                f"{self._instruction_text}\n\n{base}"
                if self._instruction_text
                else base
            )
        safety, anomaly = self.realtime.mapping.for_crossing(hand)
        mapping_text = (
            f"실시간 피드백 ({hand} hand crossing)\n"
            f"{safety.key}: 안전 우려   {anomaly.key}: 행동 이상"
        )
        return (
            f"{self._instruction_text}\n\n{mapping_text}"
            if self._instruction_text
            else mapping_text
        )

    @property
    def latest_questionnaire_result(self) -> Optional[QuestionnaireResult]:
        return self.questionnaire.result

    def set_crossing_hand(self, hand: Optional[str]) -> None:
        self.realtime.set_crossing_hand(hand)
        self._buttons.reset()
        self._render()

    def set_instruction(self, text: str) -> None:
        """Set participant-visible cue text outside questionnaire pages."""

        self._instruction_text = str(text).strip()
        self._render()

    def dismiss_questionnaire(self) -> None:
        self.questionnaire.dismiss_terminal_page()
        self._buttons.reset()
        self._render()

    def start_questionnaire(
        self,
        clocks: FeedbackClocks | Mapping[str, Any],
        *,
        context_id: str = "",
        timeout_s: Optional[float] = None,
    ) -> str:
        sample_time = self._ordered_clocks(clocks)
        questionnaire_id = self.questionnaire.start(
            sample_time, context_id=context_id, timeout_s=timeout_s
        )
        self._buttons.reset()
        self._render()
        return questionnaire_id

    def abstain_questionnaire(
        self, clocks: FeedbackClocks | Mapping[str, Any]
    ) -> QuestionnaireResult:
        sample_time = self._ordered_clocks(clocks)
        result = self.questionnaire.abstain(sample_time)
        self._buttons.reset()
        self._render()
        return result

    def update(
        self,
        button_snapshot: ControllerButtonSnapshot | Mapping[str, Any],
        clocks: FeedbackClocks | Mapping[str, Any],
    ) -> FeedbackUpdate:
        sample_time = self._ordered_clocks(clocks)
        edges = self._buttons.update(button_snapshot)
        markers: tuple[RealtimeMarker, ...] = ()
        result: Optional[QuestionnaireResult] = None
        if self.questionnaire.active:
            result = self.questionnaire.handle_edges(edges, sample_time)
            if result is not None:
                self._buttons.reset()
        else:
            markers = self.realtime.records_for_edges(edges, sample_time)
        self._render()
        return FeedbackUpdate(
            rising_edges=edges,
            realtime_markers=markers,
            questionnaire_result=result,
            questionnaire_state=self.questionnaire.state,
            renderer_text=self.renderer_text,
        )

    def attach_xr_ui(self, **kwargs: Any) -> bool:
        """Create the optional XRSceneView text panel; return availability."""

        if self._renderer is not None:
            return bool(getattr(self._renderer, "available", True))
        renderer = XRSceneViewTextRenderer(**kwargs)
        if not renderer.open():
            return False
        self._renderer = renderer
        self._last_rendered_text = ""
        self._render()
        return True

    def close(self) -> None:
        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if callable(close):
                close()
        self._renderer = None

    def _ordered_clocks(
        self, clocks: FeedbackClocks | Mapping[str, Any]
    ) -> FeedbackClocks:
        sample_time = FeedbackClocks.coerce(clocks)
        if (
            self._last_monotonic_ns is not None
            and sample_time.monotonic_ns < self._last_monotonic_ns
        ):
            raise ValueError("feedback monotonic clock moved backwards")
        self._last_monotonic_ns = sample_time.monotonic_ns
        return sample_time

    def _render(self) -> None:
        text = self.renderer_text
        if text == self._last_rendered_text:
            return
        self._last_rendered_text = text
        if self._renderer is None:
            return
        update = getattr(self._renderer, "update", None)
        if callable(update):
            update(text)


class IsaacXRButtonSource:
    """Lazy XRCore face-button reader returning tri-state snapshots."""

    _INPUT_ALIASES = {
        "a": ("a", "button_a", "primary", "primary_button"),
        "b": ("b", "button_b", "secondary", "secondary_button"),
        "x": ("x", "button_x", "primary", "primary_button"),
        "y": ("y", "button_y", "secondary", "secondary_button"),
    }
    _GESTURE_PRIORITY = ("click", "press", "pressed", "activate", "value")

    def __init__(
        self,
        *,
        press_threshold: float = 0.5,
        xr_core: Any = None,
        avatar: Any = None,
    ) -> None:
        threshold = float(press_threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("press_threshold must be in (0, 1]")
        self.press_threshold = threshold
        self._xr_core = xr_core
        self._avatar = avatar
        self.last_error = ""

    @property
    def available(self) -> bool:
        return self._get_xr_core() is not None

    def read(self) -> ControllerButtonSnapshot:
        left_connected, left = self._read_hand("left")
        right_connected, right = self._read_hand("right")
        return ControllerButtonSnapshot(
            left_connected=left_connected,
            right_connected=right_connected,
            left_x=left.get("x"),
            left_y=left.get("y"),
            right_a=right.get("a"),
            right_b=right.get("b"),
        )

    def discover(self) -> dict[str, Any]:
        """Return runtime input/gesture names using the avatar's cached devices."""

        discovered: dict[str, Any] = {}
        for hand in _HANDS:
            device = self._cached_device(hand)
            if device is None:
                discovered[hand] = {"connected": False, "inputs": {}}
                continue
            try:
                names = tuple(str(value) for value in device.get_input_names())
            except Exception as error:
                discovered[hand] = {
                    "connected": True,
                    "error": f"{type(error).__name__}: {error}",
                    "inputs": {},
                }
                continue
            inputs: dict[str, list[str]] = {}
            for name in names:
                try:
                    inputs[name] = [
                        str(value)
                        for value in device.get_input_gesture_names(name)
                    ]
                except Exception:
                    inputs[name] = []
            discovered[hand] = {"connected": True, "inputs": inputs}
        return discovered

    def _cached_device(self, hand: str) -> Any:
        logical = getattr(
            self._avatar, "get_cached_xr_device_for_logical_hand", None
        )
        if callable(logical):
            return logical(hand)
        cached = getattr(self._avatar, "get_cached_xr_device", None)
        if callable(cached):
            return cached(f"/user/hand/{hand}")
        return None

    def _get_xr_core(self) -> Any:
        if self._xr_core is not None:
            return self._xr_core
        try:
            from omni.kit.xr.core import XRCore

            self._xr_core = XRCore.get_singleton()
            self.last_error = "" if self._xr_core is not None else "XRCore singleton unavailable"
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self._xr_core = None
        return self._xr_core

    def _read_hand(self, hand: str) -> tuple[Optional[bool], dict[str, Optional[bool]]]:
        buttons = {button: None for button in _FACE_BUTTONS[hand]}
        path = f"/user/hand/{hand}"
        logical_path_getter = getattr(
            self._avatar, "get_xr_path_for_logical_hand", None
        )
        if callable(logical_path_getter):
            try:
                path = str(logical_path_getter(hand))
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                return None, buttons
        core = self._get_xr_core()
        if core is not None:
            has_device = getattr(core, "has_input_device", None)
            if callable(has_device):
                try:
                    if not bool(has_device(path)):
                        self.last_error = f"XR input device unavailable: {path}"
                        return False, buttons
                except Exception as error:
                    self.last_error = f"{type(error).__name__}: {error}"
                    return None, buttons
        device = None
        logical_cached_getter = getattr(
            self._avatar, "get_cached_xr_device_for_logical_hand", None
        )
        cached_getter = getattr(self._avatar, "get_cached_xr_device", None)
        if callable(logical_cached_getter):
            try:
                # Use the same logical-to-physical hand mapping as avatar pose
                # tracking (including XR_SWAP_HANDS).
                device = logical_cached_getter(hand)
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                return None, buttons
        elif callable(cached_getter):
            try:
                # LiveVRTrackingProvider has already refreshed this wrapper in
                # the same control frame. Reusing it avoids a second
                # get_input_device() call while the XR runtime changes state.
                device = cached_getter(path)
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                return None, buttons
        else:
            core = self._get_xr_core()
            if core is None:
                return None, buttons
            try:
                has_device = getattr(core, "has_input_device", None)
                if callable(has_device) and not bool(has_device(path)):
                    return False, buttons
                device = core.get_input_device(path)
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                return None, buttons
        if device is None:
            self.last_error = f"XR input device unavailable: {path}"
            return False, buttons
        try:
            input_names = tuple(str(value) for value in device.get_input_names())
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return None, buttons
        for button in _FACE_BUTTONS[hand]:
            buttons[button] = self._read_button(device, input_names, button)
        unknown = [name for name, value in buttons.items() if value is None]
        self.last_error = (
            ""
            if not unknown
            else f"XR face-button states unavailable for {hand}: {','.join(unknown)}"
        )
        return True, buttons

    def _read_button(
        self, device: Any, input_names: Sequence[str], button: str
    ) -> Optional[bool]:
        candidates = [
            name
            for name in input_names
            if self._matches_input(name, self._INPUT_ALIASES[button])
        ]
        if not candidates:
            candidates = list(self._INPUT_ALIASES[button])
        for input_name in candidates:
            try:
                if input_names and input_name not in input_names:
                    continue
                gestures = tuple(
                    str(value) for value in device.get_input_gesture_names(input_name)
                )
            except Exception:
                gestures = ()
            # Face-button touch/proximity is not a press.  Some OpenXR
            # backends expose both ``touch`` and ``click`` and maxing across
            # every reported gesture would generate false markers merely from
            # resting a thumb on the button.
            for gesture in self._GESTURE_PRIORITY:
                if gestures and gesture not in gestures:
                    continue
                try:
                    value = float(device.get_input_gesture_value(input_name, gesture))
                except Exception:
                    continue
                if math.isfinite(value):
                    return bool(value >= self.press_threshold)
        return None

    @staticmethod
    def _matches_input(name: str, aliases: Sequence[str]) -> bool:
        normalized = str(name).strip().lower().replace("-", "_")
        pieces = tuple(part for part in normalized.replace("/", ".").split(".") if part)
        return any(alias == normalized or alias in pieces for alias in aliases)


class XRSceneViewTextRenderer:
    """Optional text-only panel rendered by Kit's XRSceneView.

    ``open`` is best-effort and never makes Isaac a dependency of the pure
    state-machine path.  The caller can inspect :attr:`last_error` when it
    returns ``False``.
    """

    def __init__(
        self,
        *,
        frame_name: str = "ac_selective_smoothing_feedback_xr_v1",
        width: int = 720,
        height: int = 520,
        world_position: Sequence[float] = (0.35, 0.0, 1.50),
    ) -> None:
        self.frame_name = str(frame_name)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("XR panel dimensions must be positive")
        if len(tuple(world_position)) != 3:
            raise ValueError("world_position must contain x, y, z")
        self.world_position = tuple(float(value) for value in world_position)
        if not all(math.isfinite(value) for value in self.world_position):
            raise ValueError("world_position must be finite")
        self.available = False
        self.last_error = ""
        self.text = ""
        self._viewport_window = None
        self._frame = None
        self._scene_view = None
        self._model = None

    def open(self) -> bool:
        if self.available:
            return True
        try:
            import omni.ui as ui
            import omni.ui.scene as sc
            from omni.kit.viewport.utility import get_active_viewport_window
            from omni.kit.xr.scene_view.core import XRSceneView

            viewport_window = get_active_viewport_window()
            if viewport_window is None or viewport_window.viewport_api is None:
                raise RuntimeError("active XR viewport is unavailable")
            frame = viewport_window.get_frame(self.frame_name)
            with frame:
                scene_view = XRSceneView()
            viewport_window.viewport_api.add_scene_view(scene_view)
            # Store lifecycle handles before constructing scene contents so a
            # partially failed open can still remove the registered scene view.
            self._viewport_window = viewport_window
            self._frame = frame
            self._scene_view = scene_view
            model = ui.SimpleStringModel(self.text)
            x, y, z = self.world_position
            with scene_view.scene:
                with sc.Transform(
                    transform=sc.Matrix44.get_translation_matrix(x, y, z),
                    look_at=sc.Transform.LookAt.CAMERA,
                ):
                    # Widget dimensions are scene units.  This repository's
                    # stage uses metres, so scale the 900x700 pixel-like panel
                    # to a comfortable 0.9x0.7 m surface in the HMD.
                    with sc.Transform(
                        transform=sc.Matrix44.get_scale_matrix(0.001, 0.001, 0.001)
                    ):
                        widget = sc.Widget(
                            self.width,
                            self.height,
                            update_policy=sc.Widget.UpdatePolicy.ALWAYS,
                        )
                        with widget.frame:
                            with ui.ZStack():
                                ui.Rectangle(
                                    style={
                                        "background_color": 0xE6221A13,
                                        "border_color": 0xFFB8C5D6,
                                        "border_width": 3,
                                        "border_radius": 16,
                                    }
                                )
                                with ui.VStack(spacing=12, style={"margin": 28}):
                                    ui.Label(
                                        "로봇 반응 피드백",
                                        style={"font_size": 36, "color": 0xFFFFFFFF},
                                    )
                                    ui.Label(
                                        model,
                                        word_wrap=True,
                                        style={"font_size": 27, "color": 0xFFF4F7FB},
                                    )
            self._model = model
            self.available = True
            self.last_error = ""
            return True
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.close()
            return False

    def update(self, text: str) -> None:
        self.text = str(text)
        if self._model is not None:
            self._model.set_value(self.text)

    def close(self) -> None:
        scene_view = self._scene_view
        viewport_window = self._viewport_window
        try:
            if scene_view is not None and viewport_window is not None:
                viewport_api = getattr(viewport_window, "viewport_api", None)
                if viewport_api is not None:
                    viewport_api.remove_scene_view(scene_view)
        except Exception:
            pass
        try:
            if scene_view is not None:
                scene_view.scene.clear()
                scene_view.destroy()
        except Exception:
            pass
        try:
            if self._frame is not None:
                self._frame.clear()
        except Exception:
            pass
        self._viewport_window = None
        self._frame = None
        self._scene_view = None
        self._model = None
        self.available = False


# Short aliases for integration code that names the aggregate by role.
XRFeedbackInput = XRFeedbackController
ClockSnapshot = FeedbackClocks


__all__ = [
    "QUESTIONNAIRE_SCHEMA_VERSION",
    "Q1_ID",
    "Q1_TEXT_KO",
    "Q1_RESPONSE_CODES",
    "Q1_OPTION_LABELS_KO",
    "LIKERT_QUESTIONS_KO",
    "LIKERT_VALUES",
    "REJECTION_REASON_ID",
    "REJECTION_REASON_IDS",
    "REJECTION_REASON_LABELS_KO",
    "REALTIME_SAFETY_CONCERN",
    "REALTIME_BEHAVIOR_ANOMALY",
    "REALTIME_MARKER_TYPES",
    "ButtonRef",
    "ControllerButtonSnapshot",
    "ControllerInputUnavailable",
    "ControllerInputWatchdog",
    "FeedbackClocks",
    "ClockSnapshot",
    "DebouncedRisingEdge",
    "ButtonEdgeBank",
    "EmergencyAbortMonitor",
    "RealtimeButtonMapping",
    "RealtimeMarker",
    "RealtimeMarkerInput",
    "QuestionnaireButtonMapping",
    "QuestionnaireAnswer",
    "QuestionnaireResult",
    "QuestionnaireFSM",
    "FeedbackUpdate",
    "XRFeedbackController",
    "XRFeedbackInput",
    "IsaacXRButtonSource",
    "XRSceneViewTextRenderer",
]
