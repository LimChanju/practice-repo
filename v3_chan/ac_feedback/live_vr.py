"""Live VR tracking adapter preserving PoseSample validity and provenance."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from .tracking import TrackingDropout, TrackingSnapshot, TrackingWatchdog


class LiveVRTrackingProvider:
    """Bridge VRAvatar pose samples into PickPlaceEnv human-state payloads."""

    def __init__(self, watchdog: TrackingWatchdog) -> None:
        self.watchdog = watchdog
        self.avatar = None
        self._anchor_applied = False
        self._state: dict[str, Any] = {}
        self.observation_snapshot: TrackingSnapshot | None = None
        self.latest_snapshot: TrackingSnapshot | None = None

    @property
    def anchor_applied(self) -> bool:
        return self._anchor_applied

    def setup(self, world, simulation_app) -> None:
        start_live_vr_profile(simulation_app)
        try:
            from v3_chan.vr_avatar import VRAvatar
        except ImportError:
            from vr_avatar import VRAvatar

        self.avatar = VRAvatar()
        self.avatar.setup(world)

    def sample(self) -> TrackingSnapshot:
        if self.avatar is None:
            raise RuntimeError("LiveVRTrackingProvider.setup() has not been called")
        samples = self.avatar.update_with_status()
        if not self._anchor_applied:
            room_hmd_pos = self.avatar.capture_initial_hmd_pos()
            if room_hmd_pos is not None and set_live_vr_anchor(room_hmd_pos):
                self.avatar.notify_anchor_applied()
                self._anchor_applied = True
                samples = self.avatar.update_with_status()
        snapshot = self.watchdog.evaluate(samples)
        self.latest_snapshot = snapshot
        return snapshot

    def refresh_observation_state(self) -> TrackingSnapshot:
        snapshot = self.sample()
        self.observation_snapshot = snapshot
        self._state = snapshot.state_payload()
        return snapshot

    def __call__(self) -> dict[str, Any]:
        # PickPlaceEnv calls this while building the state that the next policy
        # action will consume.  Preserve that exact snapshot separately from a
        # newer pre-apply watchdog sample.
        self.refresh_observation_state()
        return dict(self._state)

    def wait_until_ready(
        self,
        simulation_app,
        *,
        timeout_s: float,
        stable_frames: int = 6,
        stop_requested: Callable[[], Any] | None = None,
    ) -> TrackingSnapshot:
        started = time.monotonic()
        stable = 0
        last: TrackingSnapshot | None = None
        while stable < max(1, int(stable_frames)):
            if stop_requested is not None:
                stop_requested()
            is_running = getattr(simulation_app, "is_running", None)
            if callable(is_running) and not bool(is_running()):
                raise RuntimeError("SimulationApp stopped during tracking readiness")
            last = self.refresh_observation_state()
            try:
                self.watchdog.require_valid(last)
            except TrackingDropout:
                stable = 0
            else:
                stable += 1
            simulation_app.update()
            if stop_requested is not None:
                stop_requested()
            if timeout_s > 0.0 and time.monotonic() - started >= timeout_s:
                reasons = () if last is None else last.invalid_reasons
                raise TimeoutError(
                    "live VR head and both hands did not become stably valid: "
                    + ", ".join(reasons)
                )
            time.sleep(0.01)
        if last is None:
            raise RuntimeError("tracking readiness loop produced no sample")
        if not self._anchor_applied:
            raise RuntimeError("XR anchor was not applied")
        return last


def start_live_vr_profile(simulation_app) -> None:
    """Enable the same XR profile used by the frozen evaluator."""

    import carb
    from omni.isaac.core.utils.extensions import enable_extension

    xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
    xr_backend = os.environ.get("ISAAC_XR_BACKEND", "OpenXR").strip()
    if xr_mode == "openxr":
        extension_ids = (
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.ar",
            "isaacsim.xr.openxr",
        )
        profile_name = "ar"
    elif xr_backend.lower() == "openxr":
        extension_ids = (
            "omni.kit.xr.system.openxr",
            "omni.kit.xr.profile.vr",
        )
        profile_name = "vr"
    else:
        extension_ids = (
            "omni.kit.xr.system.steamvr",
            "omni.kit.xr.profile.vr",
        )
        profile_name = "vr"
    enabled: list[str] = []
    for extension_id in extension_ids:
        try:
            enable_extension(extension_id)
            enabled.append(extension_id)
        except Exception:
            continue
    if not enabled:
        raise RuntimeError("No XR extension could be enabled for live VR collection")
    for _ in range(5):
        simulation_app.update()

    from omni.kit.xr.core import XRCore

    settings = carb.settings.get_settings()
    settings.set(f"/xr/profile/{profile_name}/adjustForUserHeight", False)
    settings.set(
        f"/defaults/xr/profile/{profile_name}/adjustForUserHeight", False
    )
    settings.set(f"/xr/profile/{profile_name}/system/display", xr_backend)
    settings.set(
        f"/defaults/xr/profile/{profile_name}/system/display", xr_backend
    )
    # This collector's questionnaire is an XRSceneView panel and therefore
    # must be composited into the headset.  Other runtime entry points keep
    # this disabled; the change is intentionally local to feedback sessions.
    settings.set("/xr/ui/enabled", True)
    if profile_name == "ar":
        settings.set("/xrstage/profile/ar/anchorMode", "scene origin")
    for key in (
        "/xr/profile/vr/enableControllerPhysics",
        "/xr/profile/vr/controllerPhysicsEnabled",
        "/xr/profile/vr/enablePhysicsInteraction",
        "/xr/profile/vr/pickAndPlace/enabled",
    ):
        settings.set(key, False)
    XRCore.request_enable_profile(profile_name)
    for _ in range(10):
        simulation_app.update()
    print(
        f"[ExplicitFeedback] requested XR profile={profile_name} "
        f"backend={xr_backend} extensions={','.join(enabled)}",
        flush=True,
    )


def set_live_vr_anchor(room_hmd_pos: np.ndarray) -> bool:
    import omni.usd
    from pxr import Gf, UsdGeom

    try:
        from v3_chan.vr_avatar import (
            AVATAR_EYE_POS,
            ROOM_TO_WORLD_MATRIX_ROWS,
            room_to_world_point,
        )
    except ImportError:
        from vr_avatar import (
            AVATAR_EYE_POS,
            ROOM_TO_WORLD_MATRIX_ROWS,
            room_to_world_point,
        )

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/_xr/stage/xrAnchor")
    if not prim.IsValid():
        return False
    try:
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translation = AVATAR_EYE_POS - room_to_world_point(room_hmd_pos)
        rows = [list(row) for row in ROOM_TO_WORLD_MATRIX_ROWS]
        rows[3][0:3] = [float(value) for value in translation]
        matrix = Gf.Matrix4d(*[value for row in rows for value in row])
        matrix_ops = [
            op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ]
        if matrix_ops:
            matrix_ops[0].Set(matrix)
            xformable.SetXformOpOrder([matrix_ops[0]])
        else:
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(matrix)
        return True
    except Exception:
        return False
