"""Last-moment tracking guard at the articulation apply boundary."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

from .tracking import TrackingWatchdog


class PreApplyTrackingGuard:
    """Replace a control action with a verified full hold on late dropout.

    Install this *after* ActionTraceRecorder.  Its saved original apply method is
    therefore the trace wrapper, so even the stop override receives a complete
    submitted/post-applied audit snapshot before the exception escapes and
    prevents the physics step.
    """

    def __init__(
        self,
        *,
        robot,
        provider,
        watchdog: TrackingWatchdog,
        trace,
        abort_reason_fn: Callable[[], str] | None = None,
    ) -> None:
        self.robot = robot
        self.provider = provider
        self.watchdog = watchdog
        self.trace = trace
        self.abort_reason_fn = abort_reason_fn
        self._installed = False
        self._original_apply = None
        self._raw_original = None

    def install(self) -> "PreApplyTrackingGuard":
        if self._installed:
            return self
        if (
            getattr(self.trace, "robot", None) is not self.robot
            or not bool(getattr(self.trace, "installed", False))
        ):
            raise RuntimeError(
                "install ActionTraceRecorder on this robot before tracking guard"
            )
        original = getattr(self.robot, "apply_action", None)
        if not callable(original):
            raise RuntimeError("robot.apply_action is unavailable for tracking guard")
        namespace = getattr(self.robot, "__dict__", None)
        raw_original = (
            namespace.get("apply_action")
            if isinstance(namespace, dict) and "apply_action" in namespace
            else _MISSING
        )

        @wraps(original)
        def guarded_apply(*args: Any, **kwargs: Any) -> Any:
            if not self.trace.step_active:
                return original(*args, **kwargs)
            try:
                if self.abort_reason_fn is not None:
                    abort_reason = str(self.abort_reason_fn() or "").strip()
                    if abort_reason:
                        raise RuntimeError(
                            "control aborted before articulation apply: "
                            + abort_reason
                        )
                snapshot = self.provider.sample()
                require_current = getattr(
                    self.watchdog, "require_current", None
                )
                if callable(require_current):
                    require_current(snapshot)
                else:
                    self.watchdog.require_valid(snapshot)
                observation_snapshot = getattr(
                    self.provider, "observation_snapshot", None
                )
                if observation_snapshot is None:
                    raise RuntimeError(
                        "control observation has no tracking provenance snapshot"
                    )
                if callable(require_current):
                    require_current(observation_snapshot)
                else:
                    self.watchdog.require_valid(observation_snapshot)
            except BaseException:
                # Sampling failures are indistinguishable from invalid tracking at
                # the last safe action boundary.  Submit a verified hold through
                # the already-installed trace wrapper, then preserve the original
                # exception so PickPlaceEnv never reaches world.step().
                hold = _hold_action(self.robot)
                original(hold)
                _verify_hold(self.robot, hold, len(self.robot.dof_names))
                raise
            return original(*args, **kwargs)

        try:
            setattr(self.robot, "apply_action", guarded_apply)
            if getattr(self.robot, "apply_action", None) is not guarded_apply:
                raise TypeError("tracking guard wrapper was not retained")
        except Exception as error:
            try:
                _restore(self.robot, raw_original, original)
            except Exception:
                pass
            raise RuntimeError(
                "cannot install fail-closed pre-apply tracking guard"
            ) from error
        self._original_apply = original
        self._raw_original = raw_original
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        _restore(self.robot, self._raw_original, self._original_apply)
        self._installed = False


def _hold_action(robot):
    positions = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
    if (
        positions.shape != (len(robot.dof_names),)
        or not np.all(np.isfinite(positions))
    ):
        raise RuntimeError("cannot form a finite full-DoF tracking stop")
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        try:
            from omni.isaac.core.utils.types import ArticulationAction
        except ImportError:
            current = robot.get_applied_action()
            action_type = type(current)
            return action_type(
                joint_positions=positions,
                joint_velocities=np.zeros_like(positions),
            )
    return ArticulationAction(
        joint_positions=positions,
        joint_velocities=np.zeros_like(positions),
    )


def _verify_hold(robot, hold, joint_count: int) -> None:
    expected_positions = np.asarray(
        getattr(hold, "joint_positions", None), dtype=np.float64
    ).reshape(-1)
    expected_velocities = np.asarray(
        getattr(hold, "joint_velocities", None), dtype=np.float64
    ).reshape(-1)
    applied = robot.get_applied_action()
    positions = np.asarray(
        getattr(applied, "joint_positions", None), dtype=np.float64
    ).reshape(-1)
    velocities = np.asarray(
        getattr(applied, "joint_velocities", None), dtype=np.float64
    ).reshape(-1)
    if (
        expected_positions.shape != (joint_count,)
        or expected_velocities.shape != (joint_count,)
        or positions.shape != (joint_count,)
        or velocities.shape != (joint_count,)
        or not np.all(np.isfinite(expected_positions))
        or not np.all(np.isfinite(expected_velocities))
        or not np.all(np.isfinite(positions))
        or not np.allclose(
            positions, expected_positions, rtol=0.0, atol=1e-9
        )
        or not np.allclose(expected_velocities, 0.0, rtol=0.0, atol=1e-12)
        or not np.allclose(velocities, 0.0, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("tracking dropout hold readback verification failed")


def _restore(robot, raw_original, resolved_original) -> None:
    if raw_original is not _MISSING:
        setattr(robot, "apply_action", raw_original)
        return
    try:
        delattr(robot, "apply_action")
    except (AttributeError, TypeError):
        setattr(robot, "apply_action", resolved_original)


_MISSING = object()
