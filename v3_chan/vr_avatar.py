# vr_avatar.py - VR tracking avatar and visible hand spheres.
# Safety labels are computed separately from built-in Panda PhysX colliders.
#
# ── xrAnchor 일회 이동 전략 ──────────────────────────────────────────────
# get_virtual_world_pose() = xrAnchor + 물리_룸_위치
# 따라서 xrAnchor를 바꾸면 get_virtual_world_pose()도 바뀜.
# 매 프레임 "xrAnchor = AVATAR_EYE - raw" 를 적용하면
#   다음 프레임 raw = AVATAR_EYE → xrAnchor = [0,0,0] → 진동(oscillation) 발생.
# 해결: xrAnchor 이동은 최초 1회만 수행.
#   이후 get_virtual_world_pose()가 직접 시뮬 월드 좌표를 반환하므로 그대로 사용.

import os
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom

from end_effector_safety_geometry import DEFAULT_THRESHOLDS

try:
    from v3_chan.pose_tracking import (
        PoseSample,
        PoseSourceLatch,
        TRACKING_TRACKED,
        TRACKING_UNKNOWN,
    )
except ImportError:
    from pose_tracking import (
        PoseSample,
        PoseSourceLatch,
        TRACKING_TRACKED,
        TRACKING_UNKNOWN,
    )

# 아바타 머리 중심 위치 (시뮬 월드 좌표)
AVATAR_HEAD_INIT = np.array([1.1, 0.0, 1.5])

# 어깨 위치 (머리 기준 상대 오프셋)
SHOULDER_Y = 0.20   # 좌우 ±0.20 m
SHOULDER_Z = -0.20  # 머리보다 0.20 m 아래

HEAD_RADIUS = 0.10
HAND_RADIUS = DEFAULT_THRESHOLDS.hand_radius_m

# 눈 위치: 머리 구 중심에서 앞(테이블 방향, -x)으로 HEAD_RADIUS만큼 앞
AVATAR_EYE_OFFSET = np.array([-HEAD_RADIUS, 0.0, 0.0])
AVATAR_EYE_POS    = AVATAR_HEAD_INIT + AVATAR_EYE_OFFSET  # [1.00, 0.0, 1.5]

XR_ZERO_POSE_INVALID_DIST = float(os.environ.get("XR_ZERO_POSE_INVALID_DIST", "0.03"))
XR_STAGE_VISUAL_FALLBACK = os.environ.get("XR_STAGE_VISUAL_FALLBACK", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS = int(
    os.environ.get("XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS", "120")
)
XR_STAGE_VISUAL_MAX_CANDIDATES = int(
    os.environ.get("XR_STAGE_VISUAL_MAX_CANDIDATES", "80")
)
XR_VIRTUAL_WORLD_POSE_FALLBACK = os.environ.get(
    "XR_VIRTUAL_WORLD_POSE_FALLBACK", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_CONTROLLER_POSE_MODE = os.environ.get("XR_CONTROLLER_POSE_MODE", "auto").strip().lower()
if XR_CONTROLLER_POSE_MODE in ("stage", "stage_visual"):
    XR_CONTROLLER_POSE_MODE = "visual"
if XR_CONTROLLER_POSE_MODE not in ("auto", "physical", "virtual", "visual"):
    print(f"[Avatar] Invalid XR_CONTROLLER_POSE_MODE='{XR_CONTROLLER_POSE_MODE}', using auto.")
    XR_CONTROLLER_POSE_MODE = "auto"
XR_CONTROLLER_WORKSPACE_GUARD = os.environ.get(
    "XR_CONTROLLER_WORKSPACE_GUARD", "1"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_CONTROLLER_MAX_HEAD_DIST_M = float(os.environ.get("XR_CONTROLLER_MAX_HEAD_DIST_M", "1.85"))
XR_CONTROLLER_MIN_Z_M = float(os.environ.get("XR_CONTROLLER_MIN_Z_M", "0.35"))
XR_CONTROLLER_MAX_Z_M = float(os.environ.get("XR_CONTROLLER_MAX_Z_M", "2.25"))
XR_POSE_SOURCE_CONFIRMATION_FRAMES = max(
    1, int(os.environ.get("XR_POSE_SOURCE_CONFIRMATION_FRAMES", "3"))
)

# SteamVR/OpenVR room space is X-right, Y-up, -Z-forward.
# Isaac stage here is Z-up, with the user facing the table along -X.
# Row-vector USD matrix form for: world_delta = [room_z, room_x, room_y].
ROOM_TO_WORLD_MATRIX_ROWS = (
    (0.0,  1.0, 0.0, 0.0),
    (0.0,  0.0, 1.0, 0.0),
    (1.0,  0.0, 0.0, 0.0),
    (0.0,  0.0, 0.0, 1.0),
)


def room_to_world_delta(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return np.array([delta[2], delta[0], delta[1]], dtype=float)


def room_to_world_point(pos: np.ndarray) -> np.ndarray:
    return room_to_world_delta(pos)

COLOR_LEFT  = 0xFF88AAFF
COLOR_RIGHT = 0xFFFF8844
XR_DRAW_ARM_LINES = os.environ.get("XR_DRAW_ARM_LINES", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

XR_SWAP_HANDS = os.environ.get("XR_SWAP_HANDS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_USE_OPENXR_HAND_JOINTS = os.environ.get("XR_USE_OPENXR_HAND_JOINTS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
XR_VISUALIZE_OPENXR_HAND_JOINTS = os.environ.get(
    "XR_VISUALIZE_OPENXR_HAND_JOINTS", "1"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _env_vec3(name: str, default: "tuple[float, float, float]") -> np.ndarray:
    value = os.environ.get(name, "").strip()
    if not value:
        return np.array(default, dtype=float)
    try:
        parts = [float(p.strip()) for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError
        return np.array(parts, dtype=float)
    except Exception:
        print(f"[Avatar] Invalid {name}='{value}', expected 'x,y,z' meters. Using {default}.")
        return np.array(default, dtype=float)


HAND_VISUAL_OFFSET = _env_vec3("XR_HAND_VISUAL_OFFSET", (0.0, 0.0, 0.0))
LEFT_HAND_VISUAL_OFFSET = HAND_VISUAL_OFFSET + _env_vec3(
    "XR_LEFT_HAND_VISUAL_OFFSET", (0.0, 0.0, 0.0)
)
RIGHT_HAND_VISUAL_OFFSET = HAND_VISUAL_OFFSET + _env_vec3(
    "XR_RIGHT_HAND_VISUAL_OFFSET", (0.0, 0.0, 0.0)
)
HAND_PROXY_ENABLED = os.environ.get("XR_HAND_PROXY_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HAND_SPHERE_ENABLED = os.environ.get("XR_HAND_SPHERE_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HAND_HAPTIC_POINT_MODE = os.environ.get("XR_HAND_HAPTIC_POINT_MODE", "sphere").strip().lower()
EXTERNAL_HAND_TRACKING_ENABLED = os.environ.get("XR_EXTERNAL_HAND_TRACKING", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HAND_JOINT_VISUAL_RADIUS = 0.022

HAND_JOINT_NAMES = (
    "XR_HAND_JOINT_WRIST_EXT",
    "XR_HAND_JOINT_PALM_EXT",
    "XR_HAND_JOINT_THUMB_METACARPAL_EXT",
    "XR_HAND_JOINT_THUMB_PROXIMAL_EXT",
    "XR_HAND_JOINT_THUMB_DISTAL_EXT",
    "XR_HAND_JOINT_THUMB_TIP_EXT",
    "XR_HAND_JOINT_INDEX_METACARPAL_EXT",
    "XR_HAND_JOINT_INDEX_PROXIMAL_EXT",
    "XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_INDEX_DISTAL_EXT",
    "XR_HAND_JOINT_INDEX_TIP_EXT",
    "XR_HAND_JOINT_MIDDLE_METACARPAL_EXT",
    "XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT",
    "XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_MIDDLE_DISTAL_EXT",
    "XR_HAND_JOINT_MIDDLE_TIP_EXT",
    "XR_HAND_JOINT_RING_METACARPAL_EXT",
    "XR_HAND_JOINT_RING_PROXIMAL_EXT",
    "XR_HAND_JOINT_RING_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_RING_DISTAL_EXT",
    "XR_HAND_JOINT_RING_TIP_EXT",
    "XR_HAND_JOINT_LITTLE_METACARPAL_EXT",
    "XR_HAND_JOINT_LITTLE_PROXIMAL_EXT",
    "XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT",
    "XR_HAND_JOINT_LITTLE_DISTAL_EXT",
    "XR_HAND_JOINT_LITTLE_TIP_EXT",
)

FALLBACK_HAND_OFFSETS = (
    ("wrist", np.array([0.08, 0.00, -0.02])),
    ("palm", np.array([0.00, 0.00, 0.00])),
    ("thumb_1", np.array([-0.01, 0.055, -0.01])),
    ("thumb_2", np.array([-0.04, 0.085, -0.005])),
    ("thumb_tip", np.array([-0.07, 0.105, 0.000])),
    ("index_1", np.array([-0.035, 0.030, 0.005])),
    ("index_2", np.array([-0.075, 0.035, 0.010])),
    ("index_tip", np.array([-0.115, 0.040, 0.012])),
    ("middle_1", np.array([-0.040, 0.000, 0.006])),
    ("middle_2", np.array([-0.085, 0.000, 0.012])),
    ("middle_tip", np.array([-0.130, 0.000, 0.014])),
    ("ring_1", np.array([-0.035, -0.028, 0.005])),
    ("ring_2", np.array([-0.075, -0.034, 0.010])),
    ("ring_tip", np.array([-0.112, -0.040, 0.012])),
    ("little_1", np.array([-0.025, -0.052, 0.002])),
    ("little_2", np.array([-0.060, -0.065, 0.006])),
    ("little_tip", np.array([-0.092, -0.076, 0.008])),
)


def xr_source_hand_for_logical(hand: str) -> str:
    if XR_SWAP_HANDS:
        return "right" if hand == "left" else "left"
    return hand


def xr_path_for_hand(hand: str) -> str:
    return f"/user/hand/{xr_source_hand_for_logical(hand)}"


def _is_dummy_xr_pos(pos: "np.ndarray | None") -> bool:
    if pos is None:
        return False
    return bool(np.all(pos > 4.0) and np.all(pos < 6.0))


def _is_invalid_xr_pos(xr_path: str, pos: "np.ndarray | None") -> bool:
    if pos is None:
        return False
    if _is_dummy_xr_pos(pos):
        return True
    if xr_path.startswith("/user/hand/"):
        return float(np.linalg.norm(pos)) < XR_ZERO_POSE_INVALID_DIST
    return False


def _env_pose_priority(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    aliases = {"default": "", "empty": "", "none": ""}
    out = []
    for token in raw.split(","):
        pose = token.strip().lower()
        if not pose:
            continue
        pose = aliases.get(pose, pose)
        if pose not in out:
            out.append(pose)
    return tuple(out) if out else default


XR_CONTROLLER_POSE_PRIORITY = _env_pose_priority(
    "XR_CONTROLLER_POSE_PRIORITY",
    ("grip", "aim", "", "palm", "pinch", "poke"),
)


class VRAvatar:
    """
    VR 헤드셋/컨트롤러로 구동되는 인체 아바타.

    xrAnchor 이동 전략:
      - _p_hmd0: xrAnchor 이동 전 최초 HMD virtual_world_pose (= 물리 룸 좌표)
      - xrAnchor = AVATAR_EYE_POS - _p_hmd0 (1회만 이동)
      - 이후 get_virtual_world_pose() = xrAnchor + 물리_룸_pos = 올바른 시뮬 좌표
    """

    def __init__(self):
        self._head_prim  = None
        self._lhand_prim = None
        self._rhand_prim = None
        self._hand_proxy_prims = {"left": {}, "right": {}}

        self._xr_core     = None
        self._openxr      = None
        self._openxr_spec = None
        self._openxr_failed = False
        self._debug_draw  = None
        self._devices = {}
        self._last_pose_mats = {}
        self._last_pose_kind = {}
        self._stage_visual_candidates = {"left": [], "right": []}
        self._stage_visual_scan_ticks = {"left": 0, "right": 0}
        self._stage_visual_logged = {"left": False, "right": False}
        self._stage_visual_missing_logged = {"left": False, "right": False}
        self._external_hand_joints = {"left": {}, "right": {}}
        self._external_hand_logged = {"left": False, "right": False}
        self._external_hand_missing_logged = {"left": False, "right": False}
        self._last_controller_source = {"left": None, "right": None}
        self._last_stage_visual_path = {"left": "", "right": ""}
        self._last_openxr_tracking = {
            "left": (TRACKING_UNKNOWN, False),
            "right": (TRACKING_UNKNOWN, False),
        }
        self._pose_source_latches = {
            "left": PoseSourceLatch(XR_POSE_SOURCE_CONFIRMATION_FRAMES),
            "right": PoseSourceLatch(XR_POSE_SOURCE_CONFIRMATION_FRAMES),
        }

        # xrAnchor 이동 전 최초 HMD 실제 룸 좌표 (XR 트래킹 시작 후)
        self._p_hmd0: "np.ndarray | None" = None
        self._anchor_applied = False
        self._missing_paths = set()
        self._dummy_paths = set()
        self._seen_paths = set()
        self._announced_paths = set()
        self._pose_error_paths = set()
        self._physical_pose_paths = set()
        self._virtual_pose_paths = set()
        self._invalid_pose_paths = set()
        self._implausible_controller_pose_paths = set()
        self._accepted_controller_pose_paths = set()
        self._openxr_wait_logged = False
        self._coord_logged = False
        print(
            f"[Avatar] XR hand swap={XR_SWAP_HANDS} | "
            f"openxr-hand-joints={XR_USE_OPENXR_HAND_JOINTS} | "
            f"visualize-openxr-joints={XR_VISUALIZE_OPENXR_HAND_JOINTS} | "
            f"hand-proxy={HAND_PROXY_ENABLED} | "
            f"hand-sphere={HAND_SPHERE_ENABLED} | "
            f"haptic-point-mode={HAND_HAPTIC_POINT_MODE} | "
            f"external-hand-tracking={EXTERNAL_HAND_TRACKING_ENABLED} | "
            f"arm-lines={XR_DRAW_ARM_LINES} | "
            f"controller-pose-priority={XR_CONTROLLER_POSE_PRIORITY} | "
            f"controller-pose-mode={XR_CONTROLLER_POSE_MODE} | "
            f"workspace-guard={XR_CONTROLLER_WORKSPACE_GUARD} "
            f"z=[{XR_CONTROLLER_MIN_Z_M:.2f},{XR_CONTROLLER_MAX_Z_M:.2f}]m "
            f"max-head-dist={XR_CONTROLLER_MAX_HEAD_DIST_M:.2f}m | "
            f"source-confirmation={XR_POSE_SOURCE_CONFIRMATION_FRAMES}frames | "
            "safety-geometry=built-in-panda-physx (external runtime) | "
            f"hand-radius={HAND_RADIUS:.3f}m | "
            f"zero-pose-invalid<{XR_ZERO_POSE_INVALID_DIST:.3f}m | "
            f"virtual-world-pose-fallback={XR_VIRTUAL_WORLD_POSE_FALLBACK} | "
            f"stage-visual-fallback={XR_STAGE_VISUAL_FALLBACK} | "
            f"hand-offset L={np.round(LEFT_HAND_VISUAL_OFFSET, 3)} "
            f"R={np.round(RIGHT_HAND_VISUAL_OFFSET, 3)}"
        )

    @property
    def calibrated(self) -> bool:
        """_p_hmd0 가 캡처됐으면 True."""
        return self._p_hmd0 is not None

    # ── 프림 생성 (world.reset() 전에 호출) ───────────────────────────────
    def setup(self, world):
        init = AVATAR_HEAD_INIT
        if HAND_PROXY_ENABLED:
            self._create_hand_proxy(world, "left", init + np.array([0.0, SHOULDER_Y, SHOULDER_Z - 0.3]))
            self._create_hand_proxy(world, "right", init + np.array([0.0, -SHOULDER_Y, SHOULDER_Z - 0.3]))
            print(f"[Avatar] Hand proxy prims created. Eye={np.round(AVATAR_EYE_POS,3)}")
        elif HAND_SPHERE_ENABLED:
            from omni.isaac.core.objects import VisualSphere
            self._lhand_prim = world.scene.add(VisualSphere(
                prim_path="/World/Avatar/left_hand",
                name="avatar_left_hand",
                position=init + np.array([0.0,  SHOULDER_Y, SHOULDER_Z - 0.3]),
                radius=HAND_RADIUS,
                color=np.array([0.5, 0.7, 1.0]),
            ))
            self._rhand_prim = world.scene.add(VisualSphere(
                prim_path="/World/Avatar/right_hand",
                name="avatar_right_hand",
                position=init + np.array([0.0, -SHOULDER_Y, SHOULDER_Z - 0.3]),
                radius=HAND_RADIUS,
                color=np.array([1.0, 0.6, 0.3]),
            ))
            print(f"[Avatar] Hand sphere prims created. Eye={np.round(AVATAR_EYE_POS,3)}")
        else:
            print(f"[Avatar] Hand visual prims disabled. Using XR controller models only. Eye={np.round(AVATAR_EYE_POS,3)}")

    def _create_hand_proxy(self, world, hand: str, palm_pos: np.ndarray) -> None:
        from omni.isaac.core.objects import VisualSphere

        color = np.array([0.55, 0.72, 1.0]) if hand == "left" else np.array([1.0, 0.62, 0.36])
        for joint_name in HAND_JOINT_NAMES:
            safe_name = joint_name.replace("XR_HAND_JOINT_", "").replace("_EXT", "").lower()
            prim = world.scene.add(VisualSphere(
                prim_path=f"/World/Avatar/{hand}_{safe_name}",
                name=f"avatar_{hand}_{safe_name}",
                position=palm_pos,
                radius=HAND_JOINT_VISUAL_RADIUS,
                color=color,
            ))
            self._hand_proxy_prims[hand][joint_name] = prim

        for fallback_name, offset in FALLBACK_HAND_OFFSETS:
            prim = world.scene.add(VisualSphere(
                prim_path=f"/World/Avatar/{hand}_proxy_{fallback_name}",
                name=f"avatar_{hand}_proxy_{fallback_name}",
                position=palm_pos + self._fallback_hand_offset(hand, offset),
                radius=HAND_JOINT_VISUAL_RADIUS,
                color=color,
            ))
            self._hand_proxy_prims[hand][fallback_name] = prim

    # ── XRCore / DebugDraw 초기화 ──────────────────────────────────────────
    def _init_xr(self):
        try:
            from omni.kit.xr.core import XRCore
            self._xr_core = XRCore.get_singleton()
        except Exception as e:
            print(f"[Avatar] XRCore init failed: {e}")

    def _init_openxr(self):
        if self._openxr_failed:
            return
        try:
            from isaacsim.xr.openxr import OpenXRSpec, acquire_openxr_interface
            self._openxr = acquire_openxr_interface()
            if self._openxr is None:
                if not self._openxr_wait_logged:
                    self._openxr_wait_logged = True
                    print("[Avatar] OpenXR hand-joint API waiting for interface.")
                return
            self._openxr_spec = OpenXRSpec
            print("[Avatar] OpenXR hand-joint API ready.")
        except Exception as e:
            self._openxr_failed = True
            print(f"[Avatar] OpenXR hand-joint API unavailable: {e}")

    def _init_debug_draw(self):
        try:
            from omni.debugdraw import get_debug_draw_interface
            self._debug_draw = get_debug_draw_interface()
        except Exception as e:
            print(f"[Avatar] DebugDraw init failed: {e}")

    # ── XR 장치 위치 읽기 ──────────────────────────────────────────────────
    def _get_xr_device(self, xr_path: str):
        if xr_path in self._devices:
            return self._devices[xr_path]
        device = self._xr_core.get_input_device(xr_path)
        if device is not None:
            self._devices[xr_path] = device
        return device

    def get_cached_xr_device(self, xr_path: str):
        return self._devices.get(xr_path)

    def get_cached_xr_device_for_logical_hand(self, hand: str):
        """Return the pose-consistent controller device for a logical hand."""

        logical_hand = str(hand).strip().lower()
        if logical_hand not in ("left", "right"):
            raise ValueError("logical hand must be 'left' or 'right'")
        return self.get_cached_xr_device(xr_path_for_hand(logical_hand))

    def get_xr_path_for_logical_hand(self, hand: str) -> str:
        """Return the physical XR path after applying the hand-swap mapping."""

        logical_hand = str(hand).strip().lower()
        if logical_hand not in ("left", "right"):
            raise ValueError("logical hand must be 'left' or 'right'")
        return xr_path_for_hand(logical_hand)

    def _matrix_translation(self, mat) -> np.ndarray:
        p = mat.ExtractTranslation()
        return np.array([p[0], p[1], p[2]], dtype=float)

    def _raw_pos(self, xr_path: str, pose_names: "tuple[str, ...]" = ("",)) -> "np.ndarray | None":
        if self._xr_core is None:
            self._init_xr()
        if self._xr_core is None:
            return None
        try:
            device = self._get_xr_device(xr_path)
            if device is None:
                if xr_path not in self._missing_paths:
                    self._missing_paths.add(xr_path)
                    print(f"[Avatar] XR input device not found: {xr_path}")
                return None
            if xr_path not in self._announced_paths:
                self._announced_paths.add(xr_path)
                try:
                    pose_list = [str(p) for p in device.get_pose_names()]
                    print(f"[Avatar] XR input device present: {xr_path} poses={pose_list}")
                except Exception:
                    print(f"[Avatar] XR input device present: {xr_path}")
                    pose_list = []
            for pose_name in pose_names:
                label = pose_name or "default"
                try:
                    phys_mat = device.get_pose(pose_name)
                    phys_pos = self._matrix_translation(phys_mat)
                except Exception:
                    phys_pos = None
                if phys_pos is not None and np.all(np.isfinite(phys_pos)) and not _is_invalid_xr_pos(xr_path, phys_pos):
                    self._last_pose_mats[(xr_path, label)] = phys_mat
                    self._last_pose_kind[xr_path] = "physical"
                    key = f"{xr_path}:{label}:physical"
                    if key not in self._physical_pose_paths:
                        self._physical_pose_paths.add(key)
                        print(f"[Avatar] XR physical pose detected: {key} pos={np.round(phys_pos, 3)}")
                    return phys_pos
                if phys_pos is not None and np.all(np.isfinite(phys_pos)) and _is_invalid_xr_pos(xr_path, phys_pos):
                    key = f"{xr_path}:{label}:invalid"
                    if key not in self._invalid_pose_paths:
                        self._invalid_pose_paths.add(key)
                        print(f"[Avatar] XR pose ignored as invalid: {key} pos={np.round(phys_pos, 3)}")

                try:
                    if pose_name:
                        raw_mat = device.get_raw_pose(pose_name)
                    else:
                        raw_mat = device.get_raw_pose()
                    raw_phys_pos = self._matrix_translation(raw_mat)
                except Exception:
                    raw_phys_pos = None
                if (
                    raw_phys_pos is not None
                    and np.all(np.isfinite(raw_phys_pos))
                    and not _is_invalid_xr_pos(xr_path, raw_phys_pos)
                ):
                    self._last_pose_mats[(xr_path, label)] = raw_mat
                    self._last_pose_kind[xr_path] = "raw_physical"
                    key = f"{xr_path}:{label}:raw_physical"
                    if key not in self._physical_pose_paths:
                        self._physical_pose_paths.add(key)
                        print(
                            f"[Avatar] XR raw physical pose detected: "
                            f"{key} pos={np.round(raw_phys_pos, 3)}"
                        )
                    return raw_phys_pos
                if (
                    raw_phys_pos is not None
                    and np.all(np.isfinite(raw_phys_pos))
                    and _is_invalid_xr_pos(xr_path, raw_phys_pos)
                ):
                    key = f"{xr_path}:{label}:raw_invalid"
                    if key not in self._invalid_pose_paths:
                        self._invalid_pose_paths.add(key)
                        print(
                            f"[Avatar] XR raw pose ignored as invalid: "
                            f"{key} pos={np.round(raw_phys_pos, 3)}"
                        )

                # After calibration, virtual_world_pose already includes xrAnchor.
                # Falling back to it here would apply room->world conversion twice.
                if self._p_hmd0 is not None:
                    continue

                try:
                    mat = device.get_virtual_world_pose(pose_name)
                except Exception as exc:
                    key = f"{xr_path}:{pose_name or 'default'}"
                    if key not in self._pose_error_paths:
                        self._pose_error_paths.add(key)
                        print(f"[Avatar] XR pose unavailable: {key} ({exc})")
                    continue
                pos = self._matrix_translation(mat)
                if not np.all(np.isfinite(pos)):
                    continue
                is_dummy = _is_dummy_xr_pos(pos)
                if not is_dummy:
                    self._last_pose_mats[(xr_path, label)] = mat
                    self._last_pose_kind[xr_path] = "virtual_world"
                    self._seen_paths.add(xr_path)
                    print(f"[Avatar] XR input detected: {xr_path}:{label} raw={np.round(pos, 3)}")
                    return pos

                if xr_path not in self._dummy_paths:
                    self._dummy_paths.add(xr_path)
                    print(f"[Avatar] XR input has only dummy pose so far: {xr_path} pos={np.round(pos, 3)}")
                continue
            return None
        except Exception:
            return None

    def _virtual_world_pos(
        self,
        xr_path: str,
        pose_names: "tuple[str, ...]" = ("",),
        *,
        force: bool = False,
    ) -> "np.ndarray | None":
        if not (XR_VIRTUAL_WORLD_POSE_FALLBACK or force):
            return None
        if self._xr_core is None:
            self._init_xr()
        if self._xr_core is None:
            return None
        try:
            device = self._get_xr_device(xr_path)
            if device is None:
                return None
            for pose_name in pose_names:
                label = pose_name or "default"
                try:
                    mat = device.get_virtual_world_pose(pose_name)
                    pos = self._matrix_translation(mat)
                except Exception:
                    continue
                if not np.all(np.isfinite(pos)) or _is_invalid_xr_pos(xr_path, pos):
                    key = f"{xr_path}:{label}:virtual_invalid"
                    if key not in self._invalid_pose_paths:
                        self._invalid_pose_paths.add(key)
                        print(f"[Avatar] XR virtual pose ignored as invalid: {key} pos={np.round(pos, 3)}")
                    continue
                self._last_pose_mats[(xr_path, f"virtual_{label}")] = mat
                key = f"{xr_path}:{label}:virtual_world"
                if key not in self._virtual_pose_paths:
                    self._virtual_pose_paths.add(key)
                    print(f"[Avatar] XR virtual world pose detected: {key} pos={np.round(pos, 3)}")
                return pos
        except Exception:
            return None
        return None

    def _stage_visual_path_matches(self, path: str, hand: str) -> bool:
        lower = path.lower()
        source_hand = xr_source_hand_for_logical(hand)
        if "/_xr" not in lower:
            return False
        if any(token in lower for token in ("displaydevice", "camera", "hmd", "head")):
            return False
        has_hand = (
            f"/user/hand/{source_hand}" in lower
            or f"hand/{source_hand}" in lower
            or (source_hand in lower and "hand" in lower)
        )
        if not has_hand:
            return False
        return any(
            token in lower
            for token in ("controller_model", "hand", "joint", "skeleton", "mesh")
        )

    def _find_stage_visual_candidates(self, hand: str) -> list:
        if not XR_STAGE_VISUAL_FALLBACK:
            return []
        self._stage_visual_scan_ticks[hand] = self._stage_visual_scan_ticks.get(hand, 0) + 1
        existing = [p for p in self._stage_visual_candidates.get(hand, []) if p.IsValid()]
        interval = max(1, XR_STAGE_VISUAL_SEARCH_INTERVAL_STEPS)
        if existing and self._stage_visual_scan_ticks[hand] % interval != 0:
            return existing

        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        except Exception:
            return existing

        candidates = []
        xr_sample = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower = path.lower()
            if len(xr_sample) < 25 and ("/_xr" in lower or "hand" in lower):
                xr_sample.append(path)
            if self._stage_visual_path_matches(path, hand):
                candidates.append(prim)
                if len(candidates) >= XR_STAGE_VISUAL_MAX_CANDIDATES:
                    break

        self._stage_visual_candidates[hand] = candidates
        if candidates:
            if not self._stage_visual_logged.get(hand):
                self._stage_visual_logged[hand] = True
                preview = [str(p.GetPath()) for p in candidates[:8]]
                print(
                    f"[Avatar] XR stage visual candidates for {hand}: "
                    f"{len(candidates)} paths; sample={preview}"
                )
        elif not self._stage_visual_missing_logged.get(hand):
            self._stage_visual_missing_logged[hand] = True
            print(
                f"[Avatar] No XR stage visual candidates for {hand}. "
                f"Stage XR/hand sample={xr_sample}"
            )
        return candidates

    def _prim_world_bbox_center(self, prim) -> "tuple[np.ndarray, float] | None":
        try:
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            box_min_gf = box.GetMin()
            box_max_gf = box.GetMax()
            box_min = np.array([box_min_gf[0], box_min_gf[1], box_min_gf[2]], dtype=float)
            box_max = np.array([box_max_gf[0], box_max_gf[1], box_max_gf[2]], dtype=float)
            size = box_max - box_min
            if np.all(np.isfinite(size)) and np.all(size >= 0.0):
                diag = float(np.linalg.norm(size))
                if 0.01 <= diag <= 1.0:
                    center = 0.5 * (box_min + box_max)
                    volume = float(np.prod(np.maximum(size, 1e-4)))
                    if np.all(np.isfinite(center)) and not _is_dummy_xr_pos(center):
                        return center, volume
        except Exception:
            pass

        try:
            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = mat.ExtractTranslation()
            center = np.array([p[0], p[1], p[2]], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(center)) or _is_dummy_xr_pos(center):
            return None
        return center, 0.0

    def _stage_visual_hand_pos(self, hand: str) -> "np.ndarray | None":
        candidates = self._find_stage_visual_candidates(hand)
        best = None
        best_path = None
        for prim in candidates:
            result = self._prim_world_bbox_center(prim)
            if result is None:
                continue
            center, volume = result
            if _is_invalid_xr_pos(xr_path_for_hand(hand), center):
                continue
            if best is None or volume > best[1]:
                best = (center, volume)
                best_path = str(prim.GetPath())
        if best is None:
            self._last_stage_visual_path[hand] = ""
            return None
        pos = best[0]
        self._last_stage_visual_path[hand] = str(best_path or "")
        key = f"stage_visual:{hand}"
        if key not in self._seen_paths:
            self._seen_paths.add(key)
            print(
                f"[Avatar] XR stage visual fallback detected: {hand} "
                f"pos={np.round(pos, 3)} prim={best_path}"
            )
        return pos

    def _room_to_world_from_hmd(self, room_pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
        if not self._coord_logged:
            self._coord_logged = True
            print("[Avatar] Coordinate map: SteamVR room [x,y,z] -> Isaac delta [z,x,y]")
        return origin + room_to_world_delta(room_pos - self._p_hmd0)

    def _controller_world_pos_is_plausible(self, pos: "np.ndarray | None") -> bool:
        if pos is None:
            return False
        arr = np.asarray(pos, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            return False
        if not XR_CONTROLLER_WORKSPACE_GUARD:
            return True
        if not (XR_CONTROLLER_MIN_Z_M <= float(arr[2]) <= XR_CONTROLLER_MAX_Z_M):
            return False
        return float(np.linalg.norm(arr - AVATAR_EYE_POS)) <= XR_CONTROLLER_MAX_HEAD_DIST_M

    def _log_implausible_controller_pose(
        self,
        hand: str,
        source: str,
        pos: np.ndarray,
    ) -> None:
        key = f"{hand}:{source}"
        if key in self._implausible_controller_pose_paths:
            return
        self._implausible_controller_pose_paths.add(key)
        dist = float(np.linalg.norm(np.asarray(pos, dtype=float) - AVATAR_EYE_POS))
        print(
            f"[Avatar] {hand} controller {source} pose outside workspace: "
            f"pos={np.round(pos, 3)} dist_from_eye={dist:.3f}m; trying fallback"
        )

    def _accept_controller_world_pos(
        self,
        hand: str,
        source: str,
        pos: "np.ndarray | None",
    ) -> "np.ndarray | None":
        if pos is None:
            return None
        arr = self._with_hand_visual_offset(hand, np.asarray(pos, dtype=float))
        if self._controller_world_pos_is_plausible(arr):
            key = f"{hand}:{source}"
            if key not in self._accepted_controller_pose_paths:
                self._accepted_controller_pose_paths.add(key)
                print(
                    f"[Avatar] {hand} controller using {source} pose "
                    f"pos={np.round(arr, 3)}"
                )
            source_name = {
                "physical": "xr_physical",
                "raw_physical": "xr_raw_physical",
                "virtual_world": "xr_virtual_world",
                "stage_visual": "xr_stage_visual",
                "openxr_joint": "openxr_joint",
            }.get(source, source)
            source_path = xr_path_for_hand(hand)
            tracked, known = TRACKING_UNKNOWN, False
            if source == "stage_visual":
                source_path = self._last_stage_visual_path.get(hand, "")
            elif source == "openxr_joint":
                source_path = f"{xr_path_for_hand(hand)}/palm"
                tracked, known = self._last_openxr_tracking.get(
                    hand, (TRACKING_UNKNOWN, False)
                )
            self._last_controller_source[hand] = (
                source_name,
                source_path,
                int(tracked),
                bool(known),
            )
            return arr
        self._log_implausible_controller_pose(hand, source, arr)
        return None

    def _controller_world_pos_from_physical(
        self,
        hand: str,
        xr_path: str,
        pose_names: "tuple[str, ...]",
    ) -> "np.ndarray | None":
        if self._p_hmd0 is None:
            return None
        raw = self._raw_pos(xr_path, pose_names)
        if raw is None:
            return None
        world_pos = self._room_to_world_from_hmd(raw, AVATAR_EYE_POS)
        source = self._last_pose_kind.get(xr_path, "physical")
        return self._accept_controller_world_pos(hand, source, world_pos)

    def _controller_world_pos_from_virtual(
        self,
        hand: str,
        xr_path: str,
        pose_names: "tuple[str, ...]",
    ) -> "np.ndarray | None":
        virtual_world_pos = self._virtual_world_pos(xr_path, pose_names, force=True)
        return self._accept_controller_world_pos(hand, "virtual_world", virtual_world_pos)

    def _controller_world_pos(
        self,
        hand: str,
        pose_names: "tuple[str, ...]",
        *,
        allow_openxr_joint: bool = True,
        allow_stage_visual: bool = True,
    ) -> "np.ndarray | None":
        xr_path = xr_path_for_hand(hand)

        if XR_CONTROLLER_POSE_MODE == "visual":
            if allow_stage_visual:
                stage_visual_pos = self._stage_visual_hand_pos(hand)
                pos = self._accept_controller_world_pos(hand, "stage_visual", stage_visual_pos)
                if pos is not None:
                    return pos
            pos = self._controller_world_pos_from_virtual(hand, xr_path, pose_names)
            if pos is not None:
                return pos
        elif XR_CONTROLLER_POSE_MODE == "virtual":
            pos = self._controller_world_pos_from_virtual(hand, xr_path, pose_names)
            if pos is not None:
                return pos
        else:
            pos = self._controller_world_pos_from_physical(hand, xr_path, pose_names)
            if pos is not None:
                return pos

        if XR_CONTROLLER_POSE_MODE == "auto":
            pos = self._controller_world_pos_from_virtual(hand, xr_path, pose_names)
            if pos is not None:
                return pos
        elif XR_CONTROLLER_POSE_MODE == "physical":
            pos = self._virtual_world_pos(xr_path, pose_names)
            pos = self._accept_controller_world_pos(hand, "virtual_world", pos)
            if pos is not None:
                return pos

        if allow_openxr_joint and XR_USE_OPENXR_HAND_JOINTS:
            openxr_pos = self._openxr_hand_joint_pos(hand, "XR_HAND_JOINT_PALM_EXT")
            pos = self._accept_controller_world_pos(hand, "openxr_joint", openxr_pos)
            if pos is not None:
                return pos

        if allow_stage_visual:
            stage_visual_pos = self._stage_visual_hand_pos(hand)
            pos = self._accept_controller_world_pos(hand, "stage_visual", stage_visual_pos)
            if pos is not None:
                return pos

        if XR_CONTROLLER_POSE_MODE in ("virtual", "visual"):
            pos = self._controller_world_pos_from_physical(hand, xr_path, pose_names)
            if pos is not None:
                return pos

        return None

    def _openxr_hand_joint_pos(
        self,
        hand: str,
        joint_name: str = "XR_HAND_JOINT_PALM_EXT",
    ) -> "np.ndarray | None":
        if self._openxr is None and not self._openxr_failed:
            self._init_openxr()
        if self._openxr is None or self._openxr_spec is None:
            return None
        try:
            hand_enum = (
                self._openxr_spec.XrHandEXT.XR_HAND_LEFT_EXT
                if xr_source_hand_for_logical(hand) == "left"
                else self._openxr_spec.XrHandEXT.XR_HAND_RIGHT_EXT
            )
            joints = self._openxr.locate_hand_joints(hand_enum, stage_axis=True)
            if not joints:
                return None
            joint_idx = int(getattr(self._openxr_spec.HandJointEXT, joint_name))
            joint = joints[joint_idx]
            if joint is None:
                return None
            flags = joint.locationFlags
            valid = flags & self._openxr_spec.XR_SPACE_LOCATION_POSITION_VALID_BIT
            if not valid:
                self._last_openxr_tracking[hand] = (0, True)
                return None
            tracked_bit = getattr(
                self._openxr_spec,
                "XR_SPACE_LOCATION_POSITION_TRACKED_BIT",
                None,
            )
            if tracked_bit is None:
                self._last_openxr_tracking[hand] = (TRACKING_UNKNOWN, False)
            else:
                self._last_openxr_tracking[hand] = (
                    TRACKING_TRACKED if flags & tracked_bit else 0,
                    True,
                )
            p = joint.pose.position
            pos = np.array([p.x, p.y, p.z], dtype=float)
            if not np.all(np.isfinite(pos)):
                return None
            key = f"openxr:{hand}:{joint_name}"
            if key not in self._seen_paths:
                self._seen_paths.add(key)
                print(f"[Avatar] OpenXR hand joint detected: {key} pos={np.round(pos, 3)}")
            return pos
        except Exception:
            return None

    def _openxr_hand_joint_positions(self, hand: str) -> "dict[str, np.ndarray]":
        if not XR_VISUALIZE_OPENXR_HAND_JOINTS:
            return {}
        if self._openxr is None and not self._openxr_failed:
            self._init_openxr()
        if self._openxr is None or self._openxr_spec is None:
            return {}
        try:
            hand_enum = (
                self._openxr_spec.XrHandEXT.XR_HAND_LEFT_EXT
                if xr_source_hand_for_logical(hand) == "left"
                else self._openxr_spec.XrHandEXT.XR_HAND_RIGHT_EXT
            )
            joints = self._openxr.locate_hand_joints(hand_enum, stage_axis=True)
            if not joints:
                return {}
            out = {}
            for joint_name in HAND_JOINT_NAMES:
                try:
                    joint_idx = int(getattr(self._openxr_spec.HandJointEXT, joint_name))
                    joint = joints[joint_idx]
                except Exception:
                    continue
                if joint is None:
                    continue
                flags = joint.locationFlags
                valid = flags & self._openxr_spec.XR_SPACE_LOCATION_POSITION_VALID_BIT
                if not valid:
                    continue
                p = joint.pose.position
                pos = np.array([p.x, p.y, p.z], dtype=float)
                if np.all(np.isfinite(pos)):
                    out[joint_name] = self._with_hand_visual_offset(hand, pos)
            return out
        except Exception:
            return {}

    # ── xrAnchor 이동 전 초기 HMD 위치 캡처 ──────────────────────────────
    def capture_initial_hmd_pos(self) -> "np.ndarray | None":
        """
        XR이 초기화 중엔 [5,5,5] 같은 더미값을 반환함.
        실제 트래킹이 시작된 후에만 캡처 (z < 3.0m 조건으로 판별).
        """
        if self._p_hmd0 is None:
            raw = self._raw_pos("displayDevice", ("", "raw"))
            if raw is not None and not _is_dummy_xr_pos(raw) and float(raw[2]) < 3.0:
                self._p_hmd0 = raw.copy()
                print(f"[Avatar] Initial HMD pos captured (real tracking): {np.round(raw, 3)}")
        return self._p_hmd0

    def notify_anchor_applied(self):
        """xrAnchor 이동 완료 후 main.py 에서 호출."""
        self._anchor_applied = True
        print("[Avatar] xrAnchor applied.")

    # ── 보정된 위치 읽기 ───────────────────────────────────────────────────
    def get_hmd_pos(self) -> "np.ndarray | None":
        if self._p_hmd0 is None:
            return None
        raw = self._raw_pos("displayDevice", ("", "raw"))
        if raw is None:
            return None
        return self._room_to_world_from_hmd(raw, AVATAR_HEAD_INIT)

    def get_hmd_pose_sample(self) -> PoseSample:
        position = self.get_hmd_pos()
        acquired_ns = time.monotonic_ns()
        if position is None:
            return PoseSample.invalid(acquisition_monotonic_ns=acquired_ns)
        return PoseSample(
            position_world=position,
            pose_valid=True,
            position_tracked=TRACKING_UNKNOWN,
            tracking_status_known=False,
            source_name="hmd_xr_physical",
            source_path="displayDevice",
            acquisition_monotonic_ns=acquired_ns,
            pose_age_ms=0.0,
        )

    def get_hmd_forward(self) -> "np.ndarray | None":
        """HMD forward direction in Isaac world coordinates."""
        if self._p_hmd0 is None:
            return None
        mat = self._last_pose_mats.get(("displayDevice", "default"))
        if mat is None:
            mat = self._last_pose_mats.get(("displayDevice", "raw"))
        if mat is None:
            self._raw_pos("displayDevice", ("", "raw"))
            mat = self._last_pose_mats.get(("displayDevice", "default"))
            if mat is None:
                mat = self._last_pose_mats.get(("displayDevice", "raw"))
        if mat is None:
            return None
        try:
            # OpenXR/SteamVR camera forward is local -Z.
            room_forward = np.array(
                [-mat[2][0], -mat[2][1], -mat[2][2]],
                dtype=float,
            )
        except Exception:
            return None
        if not np.all(np.isfinite(room_forward)):
            return None
        world_forward = room_to_world_delta(room_forward)
        norm = float(np.linalg.norm(world_forward))
        if norm < 1e-6:
            return None
        return world_forward / norm

    def set_external_hand_joints(
        self,
        joints_by_hand: "dict[str, dict[str, np.ndarray]] | None",
    ) -> None:
        if not EXTERNAL_HAND_TRACKING_ENABLED:
            return
        clean = {"left": {}, "right": {}}
        for hand in ("left", "right"):
            points = (joints_by_hand or {}).get(hand, {})
            for name, pos in points.items():
                arr = np.asarray(pos, dtype=float)
                if arr.shape == (3,) and np.all(np.isfinite(arr)):
                    clean[hand][name] = arr
            if clean[hand] and not self._external_hand_logged.get(hand):
                self._external_hand_logged[hand] = True
                preview = list(clean[hand].keys())[:6]
                print(
                    f"[Avatar] External {hand} hand joints active: "
                    f"{len(clean[hand])} joints sample={preview}"
                )
        self._external_hand_joints = clean

    def _external_hand_joint_positions(self, hand: str) -> "dict[str, np.ndarray]":
        if not EXTERNAL_HAND_TRACKING_ENABLED:
            return {}
        return self._external_hand_joints.get(hand, {})

    def _external_hand_center(self, hand: str) -> "np.ndarray | None":
        points = self._external_hand_joint_positions(hand)
        for name in (
            "XR_HAND_JOINT_PALM_EXT",
            "XR_HAND_JOINT_WRIST_EXT",
            "XR_HAND_JOINT_MIDDLE_METACARPAL_EXT",
        ):
            pos = points.get(name)
            if pos is not None:
                return self._with_hand_visual_offset(hand, pos)
        if points:
            arr = np.stack(list(points.values()), axis=0)
            return self._with_hand_visual_offset(hand, np.mean(arr, axis=0))
        return None

    def get_controller_pos(self, hand: str) -> "np.ndarray | None":
        return self.get_controller_pose_sample(hand).position_world

    def get_controller_pose_sample(self, hand: str) -> PoseSample:
        self._last_controller_source[hand] = None
        external_pos = self._external_hand_center(hand)
        if external_pos is not None:
            position = external_pos
            source = (
                "external_hand_tracking",
                f"udp:{hand}",
                TRACKING_UNKNOWN,
                False,
            )
        else:
            position = self._controller_world_pos(hand, XR_CONTROLLER_POSE_PRIORITY)
            source = self._last_controller_source.get(hand)

        acquired_ns = time.monotonic_ns()
        if position is None or source is None:
            candidate = PoseSample.invalid(acquisition_monotonic_ns=acquired_ns)
        else:
            source_name, source_path, tracked, tracking_known = source
            candidate = PoseSample(
                position_world=position,
                pose_valid=True,
                position_tracked=tracked,
                tracking_status_known=tracking_known,
                source_name=source_name,
                source_path=source_path,
                acquisition_monotonic_ns=acquired_ns,
                pose_age_ms=0.0,
            )
        return self._pose_source_latches[hand].update(candidate)

    def get_hand_pose_pos(self, hand: str, pose_name: str) -> "np.ndarray | None":
        external_points = self._external_hand_joint_positions(hand)
        if external_points:
            pose_to_joint = {
                "pinch": ("XR_HAND_JOINT_THUMB_TIP_EXT", "XR_HAND_JOINT_INDEX_TIP_EXT"),
                "palm": ("XR_HAND_JOINT_PALM_EXT",),
                "grip": ("XR_HAND_JOINT_PALM_EXT",),
                "aim": ("XR_HAND_JOINT_INDEX_TIP_EXT",),
            }
            names = pose_to_joint.get(pose_name, ())
            pts = [external_points[name] for name in names if name in external_points]
            if pts:
                return self._with_hand_visual_offset(hand, np.mean(np.stack(pts, axis=0), axis=0))
        return self._controller_world_pos(
            hand,
            (pose_name,),
            allow_openxr_joint=False,
            allow_stage_visual=False,
        )

    def _with_hand_visual_offset(self, hand: str, pos: np.ndarray) -> np.ndarray:
        offset = LEFT_HAND_VISUAL_OFFSET if hand == "left" else RIGHT_HAND_VISUAL_OFFSET
        return pos + offset

    def _fallback_hand_offset(self, hand: str, offset: np.ndarray) -> np.ndarray:
        out = np.asarray(offset, dtype=float).copy()
        if hand == "right":
            out[1] *= -1.0
        return out

    def _park_hand_joint_prims(self, hand: str, active_names: set[str]) -> None:
        parked = np.array([0.0, 0.0, -100.0])
        for name, prim in self._hand_proxy_prims.get(hand, {}).items():
            if name in active_names:
                continue
            try:
                prim.set_world_pose(position=parked)
            except Exception:
                pass

    def _update_hand_proxy(self, hand: str, palm_pos: "np.ndarray | None") -> None:
        if not HAND_PROXY_ENABLED:
            return
        prims = self._hand_proxy_prims.get(hand, {})
        if not prims:
            return

        joint_positions = self._external_hand_joint_positions(hand)
        if not joint_positions:
            joint_positions = self._openxr_hand_joint_positions(hand)
        if joint_positions:
            active = set()
            for joint_name, pos in joint_positions.items():
                prim = prims.get(joint_name)
                if prim is None:
                    continue
                try:
                    prim.set_world_pose(position=self._with_hand_visual_offset(hand, pos))
                    active.add(joint_name)
                except Exception:
                    pass
            self._park_hand_joint_prims(hand, active)
            return

        if EXTERNAL_HAND_TRACKING_ENABLED or XR_USE_OPENXR_HAND_JOINTS:
            if not self._external_hand_missing_logged.get(hand):
                self._external_hand_missing_logged[hand] = True
                print(
                    f"[Avatar] Waiting for real {hand} hand joints; "
                    "parking fallback hand proxy."
                )
            self._park_hand_joint_prims(hand, set())
            return

        if palm_pos is None:
            self._park_hand_joint_prims(hand, set())
            return

        active = set()
        for fallback_name, offset in FALLBACK_HAND_OFFSETS:
            prim = prims.get(fallback_name)
            if prim is None:
                continue
            try:
                prim.set_world_pose(
                    position=palm_pos + self._fallback_hand_offset(hand, offset)
                )
                active.add(fallback_name)
            except Exception:
                pass
        self._park_hand_joint_prims(hand, active)

    def shoulder_pos(self, hand: str, head_pos: np.ndarray) -> "np.ndarray | None":
        if head_pos is None:
            return None
        sign = 1.0 if hand == "left" else -1.0
        return head_pos + np.array([0.0, sign * SHOULDER_Y, SHOULDER_Z])

    # ── 매 프레임 갱신 ─────────────────────────────────────────────────────
    def update_with_status(self) -> "tuple[PoseSample, PoseSample, PoseSample]":
        head_sample = self.get_hmd_pose_sample()
        left_sample = self.get_controller_pose_sample("left")
        right_sample = self.get_controller_pose_sample("right")
        head_pos = head_sample.position_world
        left_pos = left_sample.position_world
        right_pos = right_sample.position_world
        if head_pos is None and (left_pos is not None or right_pos is not None):
            head_pos = AVATAR_HEAD_INIT
            head_sample = PoseSample(
                position_world=head_pos,
                pose_valid=False,
                source_name="synthetic_head_fallback",
                acquisition_monotonic_ns=time.monotonic_ns(),
            )

        if left_pos  is not None and self._lhand_prim is not None:
            self._lhand_prim.set_world_pose(position=left_pos)
        if right_pos is not None and self._rhand_prim is not None:
            self._rhand_prim.set_world_pose(position=right_pos)
        self._update_hand_proxy("left", left_pos)
        self._update_hand_proxy("right", right_pos)

        if XR_DRAW_ARM_LINES and self._debug_draw is None:
            self._init_debug_draw()
        dd = self._debug_draw if XR_DRAW_ARM_LINES else None
        if XR_DRAW_ARM_LINES and dd is not None and head_pos is not None:
            for hand, ctrl_pos, color in (
                ("left",  left_pos,  COLOR_LEFT),
                ("right", right_pos, COLOR_RIGHT),
            ):
                sh = self.shoulder_pos(hand, head_pos)
                if sh is not None and ctrl_pos is not None:
                    try:
                        dd.draw_line(
                            Gf.Vec3f(*sh.tolist()),       color,
                            Gf.Vec3f(*ctrl_pos.tolist()), color,
                        )
                    except Exception:
                        pass

        return head_sample, left_sample, right_sample

    def update(self) -> "tuple[np.ndarray|None, np.ndarray|None, np.ndarray|None]":
        head, left, right = self.update_with_status()
        return head.position_world, left.position_world, right.position_world

    def get_avatar_prims(self) -> list:
        prims = [p for p in (self._lhand_prim, self._rhand_prim) if p is not None]
        if HAND_PROXY_ENABLED:
            for hand_prims in self._hand_proxy_prims.values():
                prims.extend(hand_prims.values())
        return prims
