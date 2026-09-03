import os
import shutil
import subprocess

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux


def _parse_vec3(value: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = (value or "").strip()
    if not raw:
        return np.array(default, dtype=float)
    try:
        parts = [float(p.strip()) for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError
        return np.array(parts, dtype=float)
    except Exception:
        print(f"[HRIVideo] Invalid vec3 '{value}', using {default}.")
        return np.array(default, dtype=float)


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        parts = [int(p.strip()) for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError
        return max(1, parts[0]), max(1, parts[1])
    except Exception:
        print(f"[HRIVideo] Invalid resolution '{value}', using 1280,720.")
        return 1280, 720


class HRIOverviewVideoRecorder:
    """Fixed overview camera for reviewing HRI data collection sessions."""

    def __init__(
        self,
        prim_path: str = "/World/HRIOverviewCamera",
        enabled: bool = False,
        record_dir: str = "",
        resolution: str = "1280,720",
        interval_steps: int = 3,
        fps: int = 20,
        eye: str = "1.35,-1.15,1.75",
        target: str = "0.45,0.0,1.05",
        mp4_path: str = "",
    ):
        self.prim_path = prim_path
        self.enabled = bool(enabled)
        self.record_dir = os.path.abspath(record_dir) if record_dir else ""
        self.resolution = _parse_resolution(resolution)
        self.interval_steps = max(1, int(interval_steps))
        self.fps = max(1, int(fps))
        self.eye = _parse_vec3(eye, (1.35, -1.15, 1.75))
        self.target = _parse_vec3(target, (0.45, 0.0, 1.05))
        self.mp4_path = os.path.abspath(mp4_path) if mp4_path else ""
        self._rep = None
        self._render_product = None
        self._writer = None
        self._camera = None
        self._xform_op = None
        self._capture_count = 0
        self._step_count = 0

    def setup(self) -> None:
        if not self.enabled:
            print("[HRIVideo] disabled")
            return
        try:
            import omni.replicator.core as rep
            import omni.usd

            os.makedirs(self.record_dir, exist_ok=True)
            stage = omni.usd.get_context().get_stage()
            self._camera = UsdGeom.Camera.Define(stage, self.prim_path)
            self._camera.CreateFocalLengthAttr(24.0)
            self._camera.CreateHorizontalApertureAttr(28.0)
            self._camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
            xformable = UsdGeom.Xformable(self._camera.GetPrim())
            xformable.ClearXformOpOrder()
            self._xform_op = xformable.AddTransformOp()
            self._set_camera_pose()
            self._create_light(stage)

            self._rep = rep
            self._render_product = rep.create.render_product(
                self.prim_path,
                self.resolution,
            )
            self._writer = rep.WriterRegistry.get("BasicWriter")
            self._writer.initialize(output_dir=self.record_dir, rgb=True)
            self._writer.attach([self._render_product])
            try:
                rep.orchestrator.set_capture_on_play(False)
            except Exception:
                pass
            print(
                "[HRIVideo] recording enabled: "
                f"dir={self.record_dir} resolution={self.resolution[0]}x{self.resolution[1]} "
                f"interval_steps={self.interval_steps} fps={self.fps}"
            )
        except Exception as exc:
            self._rep = None
            self._render_product = None
            self._writer = None
            print(f"[HRIVideo] recording unavailable: {exc}")

    def _set_camera_pose(self) -> None:
        if self._xform_op is None:
            return
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        pose = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(float(self.eye[0]), float(self.eye[1]), float(self.eye[2])),
            Gf.Vec3d(float(self.target[0]), float(self.target[1]), float(self.target[2])),
            Gf.Vec3d(float(up[0]), float(up[1]), float(up[2])),
        ).GetInverse()
        self._xform_op.Set(pose)

    def _create_light(self, stage) -> None:
        try:
            light = UsdLux.DistantLight.Define(stage, f"{self.prim_path}_Light")
            light.CreateIntensityAttr(2500.0)
        except Exception as exc:
            print(f"[HRIVideo] light unavailable: {exc}")

    def capture(self) -> None:
        if self._rep is None or self._writer is None:
            return
        self._step_count += 1
        if self._step_count % self.interval_steps != 0:
            return
        try:
            # Refresh the render product without advancing the simulation clock.
            # Replicator's default delta advances the timeline and would inject an
            # extra physics step only when recording is enabled.
            self._rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            self._capture_count += 1
            if self._capture_count == 1:
                print(f"[HRIVideo] first frame captured: {self.record_dir}")
        except Exception as exc:
            if self._capture_count == 0:
                print(f"[HRIVideo] frame capture failed: {exc}")

    def close(self) -> None:
        if self._writer is not None and self._render_product is not None:
            try:
                self._writer.detach()
            except Exception:
                pass
        self._writer = None
        self._render_product = None
        if self.enabled and self._capture_count > 0:
            self._write_encode_script()
            self._try_encode_mp4()

    def _frame_glob(self) -> str:
        return os.path.join(self.record_dir, "*.png")

    def _default_mp4_path(self) -> str:
        if self.mp4_path:
            return self.mp4_path
        return os.path.join(os.path.dirname(self.record_dir), "hri_overview.mp4")

    def _write_encode_script(self) -> None:
        try:
            script_path = os.path.join(self.record_dir, "encode_mp4.sh")
            mp4_path = self._default_mp4_path()
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("set -euo pipefail\n")
                f.write(
                    "ffmpeg -y -framerate "
                    f"{self.fps} -pattern_type glob -i '{self._frame_glob()}' "
                    "-c:v libx264 -pix_fmt yuv420p "
                    f"'{mp4_path}'\n"
                )
            os.chmod(script_path, 0o755)
            print(f"[HRIVideo] encode script written: {script_path}")
        except Exception as exc:
            print(f"[HRIVideo] encode script failed: {exc}")

    def _try_encode_mp4(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if not ffmpeg:
            print(
                "[HRIVideo] ffmpeg not found; PNG frames were saved. "
                "Run encode_mp4.sh after installing ffmpeg."
            )
            return
        mp4_path = self._default_mp4_path()
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(self.fps),
            "-pattern_type",
            "glob",
            "-i",
            self._frame_glob(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            mp4_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[HRIVideo] mp4 saved: {mp4_path}")
        except Exception as exc:
            print(f"[HRIVideo] mp4 encode failed: {exc}")
