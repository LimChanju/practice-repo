"""Lossless, fail-closed tracing around the frozen pick/place action path.

The production pick/place environment deliberately remains untouched.  A collector
can install :class:`ActionTraceRecorder` after environment construction and wrap
the three action-boundary methods used by the runtime::

    controller.forward -> cbf.filter_action -> robot.apply_action

Every captured action is immediately converted to a robot-DoF-sized vector plus a
validity mask for each articulation channel.  Immediate conversion is important:
the CBF filter mutates the RMPFlow ``ArticulationAction`` in place, and a gripper
command can replace the filtered arm command with a sparse full-robot command.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import ACTION_TRACE_SCHEMA_VERSION


_ACTION_CHANNELS = (
    "joint_positions",
    "joint_velocities",
    "joint_efforts",
)
_MISSING = object()


def _readonly_copy(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalJointAction:
    """A fixed-width articulation action whose masks preserve sparse commands.

    Invalid/missing entries have a stored value of zero and a false mask.  Consumers
    must always use the corresponding mask; zero is only a serialization-safe fill
    value and is not itself a command.
    """

    joint_positions: np.ndarray
    joint_positions_mask: np.ndarray
    joint_velocities: np.ndarray
    joint_velocities_mask: np.ndarray
    joint_efforts: np.ndarray
    joint_efforts_mask: np.ndarray
    raw_joint_indices: Optional[np.ndarray]

    @property
    def dof_count(self) -> int:
        return int(self.joint_positions.shape[0])

    @property
    def command_mask(self) -> np.ndarray:
        result = (
            self.joint_positions_mask
            | self.joint_velocities_mask
            | self.joint_efforts_mask
        )
        result.setflags(write=False)
        return result

    @property
    def joint_indices(self) -> Optional[np.ndarray]:
        """Alias exposing the source action's indices without canonical expansion."""

        return self.raw_joint_indices

    def as_dict(self) -> Dict[str, Any]:
        """Return writable copies suitable for an HDF5/NPZ row."""

        result = {
            "joint_positions": self.joint_positions.copy(),
            "joint_positions_mask": self.joint_positions_mask.copy(),
            "joint_velocities": self.joint_velocities.copy(),
            "joint_velocities_mask": self.joint_velocities_mask.copy(),
            "joint_efforts": self.joint_efforts.copy(),
            "joint_efforts_mask": self.joint_efforts_mask.copy(),
        }
        result["raw_joint_indices"] = (
            None
            if self.raw_joint_indices is None
            else self.raw_joint_indices.copy()
        )
        return result


@dataclass(frozen=True)
class ActionStepTrace:
    """The complete controller-to-articulation action trace for one physics step."""

    schema_version: str
    step_id: Any
    joint_names: Tuple[str, ...]
    nominal_rmpflow: CanonicalJointAction
    cbf_filtered: CanonicalJointAction
    submitted: CanonicalJointAction
    pre_applied: CanonicalJointAction
    post_applied: CanonicalJointAction
    timestamps_monotonic_ns: Mapping[str, int]
    rmpflow_output_monotonic_ns: int
    cbf_output_monotonic_ns: int
    action_command_monotonic_ns: int
    arm_action_submitted: bool
    pipeline_complete: bool

    @property
    def nominal(self) -> CanonicalJointAction:
        return self.nominal_rmpflow

    @property
    def filtered(self) -> CanonicalJointAction:
        return self.cbf_filtered

    @property
    def filtered_cbf(self) -> CanonicalJointAction:
        return self.cbf_filtered

    @property
    def submitted_partial(self) -> CanonicalJointAction:
        return self.submitted

    @property
    def pre_applied_full(self) -> CanonicalJointAction:
        return self.pre_applied

    @property
    def post_applied_full(self) -> CanonicalJointAction:
        return self.post_applied


def _action_field(action: Any, name: str) -> Any:
    if isinstance(action, Mapping):
        return action.get(name)
    return getattr(action, name, None)


def _host_value(value: Any) -> Any:
    """Best-effort conversion for CPU tensor-like Isaac action fields."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        try:
            candidate = numpy_method()
        except (TypeError, RuntimeError):
            pass
    return candidate


def _flat_object_vector(value: Any, label: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    result = np.asarray(_host_value(value), dtype=object)
    if result.ndim == 0:
        raise ValueError(f"{label} must be a vector, not a scalar")
    return result.reshape(-1)


def _joint_index_vector(value: Any, dof_count: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw = np.asarray(_host_value(value))
    if raw.ndim == 0:
        raise ValueError("joint_indices must be a vector, not a scalar")
    try:
        numeric = raw.astype(float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_indices must contain integers") from exc
    if numeric.size == 0:
        raise ValueError("joint_indices must not be empty")
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("joint_indices must contain finite integers")
    result = numeric.astype(np.int64)
    if np.any(result < 0) or np.any(result >= dof_count):
        raise ValueError(
            f"joint_indices are outside the robot's [0, {dof_count}) DoF range"
        )
    if np.unique(result).size != result.size:
        raise ValueError("joint_indices must not contain duplicates")
    return result


def canonicalize_joint_action(
    action: Any,
    joint_names: Sequence[str],
) -> CanonicalJointAction:
    """Map an Isaac-style partial action into fixed vectors and validity masks.

    When ``joint_indices`` is absent, Isaac applies an action field of length *k* to
    joints ``0..k-1``.  This also handles the gripper's full-width sparse list, in
    which arm entries are ``None`` and only the finger entries are commands.
    ``NaN`` is treated like ``None`` (uncommanded); infinities are rejected.
    """

    names = tuple(str(name) for name in joint_names)
    if not names:
        raise ValueError("joint_names must contain at least one robot DoF")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("joint_names must be non-empty and unique")
    if action is None:
        raise ValueError("action must not be None")

    dof_count = len(names)
    channel_values = {
        channel: _flat_object_vector(_action_field(action, channel), channel)
        for channel in _ACTION_CHANNELS
    }
    lengths = {value.size for value in channel_values.values() if value is not None}
    if len(lengths) > 1:
        raise ValueError("all present articulation action channels must have one size")
    target_size = next(iter(lengths), 0)

    raw_indices = _joint_index_vector(
        _action_field(action, "joint_indices"), dof_count
    )
    if raw_indices is None:
        if target_size > dof_count:
            raise ValueError(
                f"action width {target_size} exceeds robot DoF count {dof_count}"
            )
        indices = np.arange(target_size, dtype=np.int64)
    elif raw_indices.size != target_size:
        raise ValueError(
            "joint_indices length must match every present articulation channel"
        )
    else:
        indices = raw_indices

    canonical: Dict[str, np.ndarray] = {}
    for channel, raw_values in channel_values.items():
        values = np.zeros(dof_count, dtype=np.float64)
        mask = np.zeros(dof_count, dtype=np.bool_)
        if raw_values is not None:
            for source_offset, robot_index in enumerate(indices):
                raw_value = raw_values[source_offset]
                if raw_value is None:
                    continue
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{channel}[{source_offset}] is not numeric or None"
                    ) from exc
                if math.isnan(numeric):
                    continue
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"{channel}[{source_offset}] must be finite, None, or NaN"
                    )
                values[int(robot_index)] = numeric
                mask[int(robot_index)] = True
        canonical[channel] = _readonly_copy(values, dtype=np.float64)
        canonical[f"{channel}_mask"] = _readonly_copy(mask, dtype=np.bool_)

    canonical["raw_joint_indices"] = (
        None
        if raw_indices is None
        else _readonly_copy(raw_indices, dtype=np.int64)
    )
    return CanonicalJointAction(**canonical)


# Short spelling for collector call sites and backwards-compatible experiments.
canonicalize_action = canonicalize_joint_action


@dataclass
class _OpenStep:
    step_id: Any
    pre_applied: CanonicalJointAction
    nominal: Optional[CanonicalJointAction] = None
    filtered: Optional[CanonicalJointAction] = None
    submitted: Optional[CanonicalJointAction] = None
    post_applied: Optional[CanonicalJointAction] = None
    timestamps: Optional[Dict[str, int]] = None
    arm_action_submitted: Optional[bool] = None


class ActionTraceRecorder:
    """Install reversible external hooks and emit one strict trace per step.

    Calls outside a ``start_step``/``finish_step`` window pass through unchanged.
    During an open step, duplicate/out-of-order stages raise before an untraceable
    articulation command can be submitted.
    """

    SCHEMA_VERSION = ACTION_TRACE_SCHEMA_VERSION

    def __init__(
        self,
        *,
        robot: Any,
        controller: Any,
        cbf: Any,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.robot = robot
        self.controller = controller
        self.cbf = cbf
        self._clock_ns = clock_ns
        self.joint_names = self._read_joint_names(robot)
        self._patches = []
        self._installed = False
        self._open_step: Optional[_OpenStep] = None

        getter = getattr(robot, "get_applied_action", None)
        if not callable(getter):
            raise RuntimeError(
                "lossless action tracing requires robot.get_applied_action()"
            )
        self._get_applied_action = getter

    @staticmethod
    def _read_joint_names(robot: Any) -> Tuple[str, ...]:
        raw_names = getattr(robot, "dof_names", None)
        if raw_names is None:
            raise RuntimeError("lossless action tracing requires robot.dof_names")
        try:
            names = tuple(str(name) for name in raw_names)
        except TypeError as exc:
            raise RuntimeError("robot.dof_names must be an iterable") from exc
        if not names or any(not name for name in names):
            raise RuntimeError("robot.dof_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise RuntimeError("robot.dof_names must be unique")
        return names

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def step_active(self) -> bool:
        return self._open_step is not None

    def _now(self) -> int:
        value = self._clock_ns()
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError("action trace clock must return integer nanoseconds")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "action trace clock must return integer nanoseconds"
            ) from exc
        if result < 0:
            raise RuntimeError("action trace clock returned a negative timestamp")
        return result

    @staticmethod
    def _raw_instance_attribute(obj: Any, name: str) -> Any:
        namespace = getattr(obj, "__dict__", None)
        if isinstance(namespace, dict) and name in namespace:
            return namespace[name]
        return _MISSING

    @staticmethod
    def _restore_attribute(
        obj: Any,
        name: str,
        raw_original: Any,
        resolved_original: Callable[..., Any],
    ) -> None:
        if raw_original is not _MISSING:
            setattr(obj, name, raw_original)
            return
        try:
            delattr(obj, name)
        except (AttributeError, TypeError):
            # A non-standard extension object may not expose its instance storage.
            setattr(obj, name, resolved_original)

    def _install_method(
        self,
        obj: Any,
        name: str,
        wrapper: Callable[..., Any],
        *,
        boundary: str,
    ) -> None:
        original = getattr(obj, name, None)
        if not callable(original):
            raise RuntimeError(f"cannot trace {boundary}: method is unavailable")
        raw_original = self._raw_instance_attribute(obj, name)
        try:
            setattr(obj, name, wrapper)
            if getattr(obj, name, None) is not wrapper:
                raise TypeError("assigned wrapper was not retained")
        except Exception as exc:
            try:
                self._restore_attribute(obj, name, raw_original, original)
            except Exception:
                pass
            raise RuntimeError(
                f"cannot install fail-closed wrapper for {boundary}"
            ) from exc
        self._patches.append((obj, name, raw_original, original))

    def install(self) -> "ActionTraceRecorder":
        if self._installed:
            return self

        original_apply = getattr(self.robot, "apply_action", None)
        original_forward = getattr(self.controller, "forward", None)
        original_filter = getattr(self.cbf, "filter_action", None)
        if not callable(original_apply):
            raise RuntimeError(
                "cannot install fail-closed wrapper for robot.apply_action"
            )
        if not callable(original_forward):
            raise RuntimeError("cannot trace controller.forward: method is unavailable")
        if not callable(original_filter):
            raise RuntimeError("cannot trace cbf.filter_action: method is unavailable")

        @wraps(original_apply)
        def apply_wrapper(*args: Any, **kwargs: Any) -> Any:
            open_step = self._open_step
            if open_step is None:
                return original_apply(*args, **kwargs)
            if open_step.nominal is None or open_step.filtered is None:
                raise RuntimeError(
                    "refusing robot.apply_action before nominal and filtered actions "
                    "were captured"
                )
            if open_step.submitted is not None or open_step.post_applied is not None:
                raise RuntimeError(
                    "multiple robot.apply_action calls in one traced step"
                )
            submitted_action = self._extract_apply_action(args, kwargs)
            submitted = canonicalize_joint_action(
                submitted_action, self.joint_names
            )
            if not np.any(submitted.command_mask):
                raise RuntimeError("submitted articulation action contains no command")
            open_step.submitted = submitted
            open_step.timestamps["submitted"] = self._now()
            open_step.timestamps["apply_started"] = self._now()
            open_step.timestamps["action_command_monotonic_ns"] = (
                open_step.timestamps["apply_started"]
            )
            open_step.arm_action_submitted = self._contains_expected_action(
                submitted=submitted,
                expected=open_step.filtered,
            )

            result = original_apply(*args, **kwargs)

            post_action = self._get_applied_action()
            post_applied = canonicalize_joint_action(post_action, self.joint_names)
            self._validate_full_applied(post_applied, "post_applied")
            open_step.post_applied = post_applied
            open_step.timestamps["post_applied"] = self._now()
            return result

        @wraps(original_forward)
        def forward_wrapper(*args: Any, **kwargs: Any) -> Any:
            open_step = self._open_step
            if open_step is not None and open_step.nominal is not None:
                raise RuntimeError(
                    "multiple controller.forward calls in one traced step"
                )
            result = original_forward(*args, **kwargs)
            open_step = self._open_step
            if open_step is not None:
                # Canonicalization is the deep snapshot.  It must happen before the
                # same mutable object reaches the CBF filter.
                open_step.nominal = canonicalize_joint_action(
                    result, self.joint_names
                )
                open_step.timestamps["nominal"] = self._now()
                open_step.timestamps["rmpflow_output_monotonic_ns"] = (
                    open_step.timestamps["nominal"]
                )
            return result

        @wraps(original_filter)
        def filter_wrapper(*args: Any, **kwargs: Any) -> Any:
            open_step = self._open_step
            if open_step is not None:
                if open_step.nominal is None:
                    raise RuntimeError(
                        "refusing cbf.filter_action before nominal action capture"
                    )
                if open_step.filtered is not None:
                    raise RuntimeError(
                        "multiple cbf.filter_action calls in one traced step"
                    )
            result = original_filter(*args, **kwargs)
            open_step = self._open_step
            if open_step is not None:
                filtered_action = result[0] if isinstance(result, tuple) else result
                open_step.filtered = canonicalize_joint_action(
                    filtered_action, self.joint_names
                )
                open_step.timestamps["filtered"] = self._now()
                open_step.timestamps["cbf_output_monotonic_ns"] = (
                    open_step.timestamps["filtered"]
                )
            return result

        # Install the physical actuation boundary first.  If it cannot be wrapped,
        # no best-effort tracing mode is allowed.
        try:
            self._install_method(
                self.robot,
                "apply_action",
                apply_wrapper,
                boundary="robot.apply_action",
            )
            self._install_method(
                self.controller,
                "forward",
                forward_wrapper,
                boundary="controller.forward",
            )
            self._install_method(
                self.cbf,
                "filter_action",
                filter_wrapper,
                boundary="cbf.filter_action",
            )
        except Exception:
            self._rollback_patches()
            raise
        self._installed = True
        return self

    @staticmethod
    def _extract_apply_action(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if args:
            if "control_actions" in kwargs or "action" in kwargs:
                raise RuntimeError("ambiguous robot.apply_action invocation")
            return args[0]
        if "control_actions" in kwargs:
            return kwargs["control_actions"]
        if "action" in kwargs:
            return kwargs["action"]
        raise RuntimeError("robot.apply_action was called without an action")

    @staticmethod
    def _contains_expected_action(
        *,
        submitted: CanonicalJointAction,
        expected: CanonicalJointAction,
    ) -> bool:
        """Return whether submitted preserves every filtered arm command exactly."""

        expected_any = False
        for channel in _ACTION_CHANNELS:
            expected_values = getattr(expected, channel)
            expected_mask = getattr(expected, f"{channel}_mask")
            if not np.any(expected_mask):
                continue
            expected_any = True
            submitted_values = getattr(submitted, channel)
            submitted_mask = getattr(submitted, f"{channel}_mask")
            if not np.all(submitted_mask[expected_mask]):
                return False
            if not np.array_equal(
                submitted_values[expected_mask], expected_values[expected_mask]
            ):
                return False
        return expected_any

    @staticmethod
    def _validate_full_applied(
        stage: CanonicalJointAction,
        stage_name: str,
    ) -> None:
        # This runtime uses position/velocity drives.  Effort is intentionally unset
        # in reset and may therefore be either entirely absent or a full vector.
        for channel in ("joint_positions", "joint_velocities"):
            mask = getattr(stage, f"{channel}_mask")
            if not np.all(mask):
                raise RuntimeError(
                    f"{stage_name} is not a full-DoF applied {channel} target"
                )
        effort_mask = stage.joint_efforts_mask
        if np.any(effort_mask) and not np.all(effort_mask):
            raise RuntimeError(
                f"{stage_name} has a partially populated applied effort target"
            )

    def start_step(self, step_id: Any = None) -> None:
        if not self._installed:
            raise RuntimeError("install action trace hooks before start_step()")
        if self._open_step is not None:
            raise RuntimeError("an action trace step is already active")

        started = self._now()
        pre_action = self._get_applied_action()
        pre_applied = canonicalize_joint_action(pre_action, self.joint_names)
        self._validate_full_applied(pre_applied, "pre_applied")
        pre_timestamp = self._now()
        self._open_step = _OpenStep(
            step_id=step_id,
            pre_applied=pre_applied,
            timestamps={
                "step_started": started,
                "pre_applied": pre_timestamp,
            },
        )

    # A natural alias for collectors that refer to an episode loop as begin/finish.
    begin_step = start_step

    def finish_step(self) -> ActionStepTrace:
        open_step = self._open_step
        if open_step is None:
            raise RuntimeError("finish_step() called without an active trace step")

        missing = [
            name
            for name in ("nominal", "filtered", "submitted", "post_applied")
            if getattr(open_step, name) is None
        ]
        if missing:
            raise RuntimeError(
                "incomplete action trace; missing stages: " + ", ".join(missing)
            )
        if not np.any(open_step.nominal.command_mask):
            raise RuntimeError("nominal action trace contains no command")
        if not np.any(open_step.filtered.command_mask):
            raise RuntimeError("filtered action trace contains no command")
        if not np.any(open_step.submitted.command_mask):
            raise RuntimeError("submitted action trace contains no command")
        if open_step.arm_action_submitted is None:
            raise RuntimeError("action trace is missing arm submission provenance")
        self._validate_full_applied(open_step.pre_applied, "pre_applied")
        self._validate_full_applied(open_step.post_applied, "post_applied")

        required_timestamps = (
            "step_started",
            "pre_applied",
            "nominal",
            "filtered",
            "submitted",
            "apply_started",
            "post_applied",
        )
        missing_timestamps = [
            name for name in required_timestamps if name not in open_step.timestamps
        ]
        if missing_timestamps:
            raise RuntimeError(
                "incomplete action trace; missing timestamps: "
                + ", ".join(missing_timestamps)
            )
        ordered_values = [open_step.timestamps[name] for name in required_timestamps]
        pairs = zip(ordered_values, ordered_values[1:])
        if any(later < earlier for earlier, later in pairs):
            raise RuntimeError("action trace timestamps are not monotonic")

        timestamps = dict(open_step.timestamps)
        timestamps["step_finished"] = self._now()
        if timestamps["step_finished"] < ordered_values[-1]:
            raise RuntimeError("action trace finish timestamp is not monotonic")

        trace = ActionStepTrace(
            schema_version=self.SCHEMA_VERSION,
            step_id=open_step.step_id,
            joint_names=self.joint_names,
            nominal_rmpflow=open_step.nominal,
            cbf_filtered=open_step.filtered,
            submitted=open_step.submitted,
            pre_applied=open_step.pre_applied,
            post_applied=open_step.post_applied,
            timestamps_monotonic_ns=timestamps,
            rmpflow_output_monotonic_ns=timestamps[
                "rmpflow_output_monotonic_ns"
            ],
            cbf_output_monotonic_ns=timestamps["cbf_output_monotonic_ns"],
            action_command_monotonic_ns=timestamps[
                "action_command_monotonic_ns"
            ],
            arm_action_submitted=open_step.arm_action_submitted,
            pipeline_complete=True,
        )
        self._open_step = None
        return trace

    def abort_step(self) -> None:
        """Discard an incomplete step after the caller has handled its exception."""

        self._open_step = None

    def _rollback_patches(self) -> None:
        restoration_errors = []
        while self._patches:
            obj, name, raw_original, resolved_original = self._patches.pop()
            try:
                self._restore_attribute(
                    obj, name, raw_original, resolved_original
                )
            except Exception as exc:
                restoration_errors.append((name, exc))
        if restoration_errors:
            names = ", ".join(name for name, _ in restoration_errors)
            raise RuntimeError(f"could not restore action trace hooks: {names}")

    def uninstall(self) -> None:
        self._open_step = None
        if not self._patches:
            self._installed = False
            return
        self._rollback_patches()
        self._installed = False

    def __enter__(self) -> "ActionTraceRecorder":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.uninstall()


# Descriptive alias used by early explicit-feedback collector prototypes.
ActionTraceCapture = ActionTraceRecorder
ActionTrace = ActionStepTrace
StepActionTrace = ActionStepTrace


def install_action_trace(
    *,
    robot: Any,
    controller: Any,
    cbf: Any,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> ActionTraceRecorder:
    """Construct and install a fail-closed action trace recorder."""

    return ActionTraceRecorder(
        robot=robot,
        controller=controller,
        cbf=cbf,
        clock_ns=clock_ns,
    ).install()


__all__ = [
    "ActionStepTrace",
    "ActionTrace",
    "ActionTraceCapture",
    "ActionTraceRecorder",
    "CanonicalJointAction",
    "canonicalize_action",
    "canonicalize_joint_action",
    "install_action_trace",
    "StepActionTrace",
]
