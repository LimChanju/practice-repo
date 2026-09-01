from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RL_DIR = os.path.join(SCRIPT_DIR, "rl")
PYTHON_PACKAGE_DIR = os.path.join(SCRIPT_DIR, ".python_packages")
ISAAC_TORCH_BUNDLE = os.environ.get(
    "ISAAC_TORCH_BUNDLE",
    os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"),
)

for path in (RL_DIR, PYTHON_PACKAGE_DIR, SCRIPT_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from end_effector_safety_geometry import (  # noqa: E402
    SafetyThresholds,
    distance_gate,
)
from scene_randomization import scene_layout_id  # noqa: E402
from physical_safety_controllers import PHYSICAL_SAFETY_MODES  # noqa: E402
from trajectory_metrics import CartesianMotionTracker  # noqa: E402


SAFETY_THRESHOLDS = SafetyThresholds.from_env()
TASK_FAILURE_DIAGNOSTIC_SCHEMA = "task_cbf_failure_diagnostics_v1"


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_phase_name(controller_event: int) -> str:
    event = int(controller_event)
    if event <= 0:
        return "approach_cube"
    if event <= 3:
        return "grasp_cube"
    if event <= 6:
        return "move_to_target"
    return "release_cube"


def _json_mapping(value: Any) -> str:
    mapping = _mapping_or_empty(value)
    return json.dumps(
        dict(mapping),
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: (
            item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item)
        ),
    )


def _parse_args() -> argparse.Namespace:
    default_json = os.path.join(
        SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.json"
    )
    default_csv = os.path.join(
        SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval.csv"
    )
    default_step_csv = os.path.join(
        SCRIPT_DIR, "eval_results", "bc_pick_place_v1_rollout_eval_steps.csv"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC policy by rolling it out in Isaac Sim."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(SCRIPT_DIR, "policies", "bc_pick_place_v1_100eps.pt"),
        help="PyTorch checkpoint produced by v2/rl/train_bc.py.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--render", action="store_true", help="Render the rollout window."
    )
    parser.add_argument(
        "--render-step-delay-sec",
        type=float,
        default=0.0,
        help="Optional wall-clock delay after each rendered simulation step.",
    )
    parser.add_argument(
        "--live-vr",
        action="store_true",
        help="Use live VRAvatar head/hand sphere positions instead of recorded human replay.",
    )
    parser.add_argument(
        "--vr-tracking-timeout-sec",
        type=float,
        default=120.0,
        help="Seconds to wait for a live HMD and at least one hand. Zero waits indefinitely.",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--task-reward-version",
        default="",
        help=(
            "Evaluation-only reward override. Empty uses checkpoint metadata. "
            "This does not change policy actions and is required when evaluating "
            "legacy checkpoints whose recorded reward implementation was retired."
        ),
    )
    parser.add_argument(
        "--residual-gate-mode",
        choices=("checkpoint", "none", "distance"),
        default="checkpoint",
        help=(
            "Override gating embedded in a residual task checkpoint. "
            "Legacy checkpoints default to none."
        ),
    )
    parser.add_argument(
        "--mask-human-obs-for-policy",
        action="store_true",
        help=(
            "Feed robot-only-compatible observations to the policy by zeroing live/replayed "
            "human fields and setting min_hand_gripper_dist to the missing-human value. "
            "Safety/ErrP logging still uses the environment's real HRI observation."
        ),
    )
    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument(
        "--fixed-orientation",
        dest="fixed_orientation",
        action="store_true",
        help="Use the expert's fixed top-down gripper orientation.",
    )
    orientation_group.add_argument(
        "--free-orientation",
        dest="fixed_orientation",
        action="store_false",
        help="Leave the gripper orientation unconstrained.",
    )
    parser.set_defaults(fixed_orientation=True)
    parser.add_argument(
        "--gripper-mode", choices=("event", "rule", "policy"), default="event"
    )
    parser.add_argument("--success-dist", type=float, default=0.06)
    parser.add_argument(
        "--safety-gate-start-dist",
        type=float,
        default=SAFETY_THRESHOLDS.gate_start_gap_m,
        help="Surface gap in meters where safety residual gate starts opening.",
    )
    parser.add_argument(
        "--safety-gate-full-dist",
        type=float,
        default=SAFETY_THRESHOLDS.gate_full_gap_m,
        help="Surface gap in meters where safety residual gate reaches 1.0.",
    )
    parser.add_argument(
        "--safety-residual-checkpoint",
        default="",
        help=(
            "Optional pi_safe checkpoint. It receives the v4 HRI safety observation "
            "and outputs a 5D residual."
        ),
    )
    parser.add_argument(
        "--safety-residual-alpha",
        type=float,
        default=0.1,
        help="Maximum scale for the safety residual action when --safety-residual-checkpoint is set.",
    )
    parser.add_argument(
        "--analytic-avoidance-offset-m",
        type=float,
        default=0.0,
        help=(
            "Use a non-learned XYZ residual pointing away from the nearest hand. "
            "The value is the requested position offset at a fully open gate."
        ),
    )
    parser.add_argument(
        "--physical-safety-controller",
        choices=PHYSICAL_SAFETY_MODES,
        default="none",
        help=(
            "Physical safety backend applied after the frozen task policy. "
            "rmpflow registers tracked hands as dynamic obstacles; cbf filters "
            "the nominal joint command; curobo uses optional MPPI/MPC."
        ),
    )
    parser.add_argument(
        "--rmpflow-human-safety-margin-m",
        type=float,
        default=0.05,
        help="Inflation added to each tracked hand radius for RMPflow/cuRobo.",
    )
    parser.add_argument(
        "--visualize-physical-safety",
        action="store_true",
        help="Show the inflated RMPflow hand obstacle spheres while rendering.",
    )
    parser.add_argument("--cbf-safe-gap-m", type=float, default=0.05)
    parser.add_argument("--cbf-activation-gap-m", type=float, default=0.13)
    parser.add_argument("--cbf-gamma-per-s", type=float, default=8.0)
    parser.add_argument("--cbf-prediction-horizon-s", type=float, default=0.15)
    parser.add_argument("--cbf-max-prediction-buffer-m", type=float, default=0.08)
    parser.add_argument("--cbf-max-joint-speed-rad-s", type=float, default=2.0)
    parser.add_argument("--phase-gate-close-dist", type=float, default=0.075)
    parser.add_argument("--phase-gate-max-hold", type=int, default=320)
    pseudo_errp_group = parser.add_mutually_exclusive_group()
    pseudo_errp_group.add_argument(
        "--pseudo-errp",
        dest="pseudo_errp_enabled",
        action="store_true",
        help="Enable pseudo-ErrP feedback from configured task/HRI flags.",
    )
    pseudo_errp_group.add_argument(
        "--no-pseudo-errp",
        dest="pseudo_errp_enabled",
        action="store_false",
        help="Disable pseudo-ErrP feedback while still reporting source flags.",
    )
    parser.set_defaults(pseudo_errp_enabled=True)
    parser.add_argument(
        "--pseudo-errp-sources",
        default="all",
        help=(
            "Comma-separated pseudo-ErrP sources, 'all', or 'none'. "
            "Known sources include human_robot_collision, near_human, collision_green, "
            "pick_miss_recent, drop_throw_recent, gripper_camera_occluded."
        ),
    )
    parser.add_argument(
        "--human-replay-data",
        default="",
        help="Optional HDF5 trajectory file containing recorded human head/hand motion.",
    )
    parser.add_argument(
        "--human-replay-mode",
        choices=("step", "loop"),
        default="step",
        help="step holds the last human sample after replay ends; loop repeats it.",
    )
    parser.add_argument(
        "--human-replay-episode-policy",
        choices=("cycle", "random"),
        default="cycle",
        help="How to choose a recorded human episode for each rollout episode.",
    )
    parser.add_argument(
        "--encounter-manifest",
        default="",
        help=(
            "Optional encounter JSON. One scenario is inserted at its matching "
            "task phase/event per rollout; source safety labels remain metadata."
        ),
    )
    parser.add_argument(
        "--encounter-policy",
        choices=("cycle", "random"),
        default="cycle",
        help="Use cycle for deterministic held-out evaluation.",
    )
    parser.add_argument(
        "--encounter-severity-mix",
        default="safe=0.40,gate_only=0.25,near=0.20,near_miss=0.10,collision=0.05",
    )
    parser.add_argument(
        "--encounter-anchor-mode",
        choices=("ee", "world"),
        default="ee",
    )
    parser.add_argument(
        "--encounter-timebase",
        choices=("recorded", "step"),
        default="recorded",
    )
    parser.add_argument("--encounter-playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--allow-legacy-source-configuration",
        action="store_true",
        help=(
            "Explicitly allow encounter screening with a new random layout when "
            "the source HDF5 has neither exact poses nor a layout seed. Disabled "
            "by default because such results are not source-aligned."
        ),
    )
    parser.add_argument(
        "--no-encounter-phase-match",
        dest="encounter_phase_match",
        action="store_false",
    )
    parser.add_argument(
        "--no-encounter-event-match",
        dest="encounter_event_match",
        action="store_false",
    )
    parser.set_defaults(
        encounter_phase_match=True,
        encounter_event_match=True,
    )
    parser.add_argument(
        "--visualize-human-replay",
        action="store_true",
        help="Show replayed head/hand positions as spheres when rendering.",
    )
    parser.add_argument(
        "--human-replay-visual-z-offset",
        type=float,
        default=0.0,
        help="Visualization-only z offset in meters for replayed head/hand spheres.",
    )
    parser.add_argument(
        "--synthetic-human",
        action="store_true",
        help="Inject a random synthetic hand sweep near the gripper for pseudo-ErrP stress tests.",
    )
    parser.add_argument("--synthetic-human-episode-prob", type=float, default=0.35)
    parser.add_argument("--synthetic-human-start-min-step", type=int, default=120)
    parser.add_argument("--synthetic-human-start-max-step", type=int, default=520)
    parser.add_argument("--synthetic-human-duration-steps", type=int, default=90)
    parser.add_argument("--synthetic-human-near-dist", type=float, default=0.12)
    parser.add_argument("--synthetic-human-collision-dist", type=float, default=0.035)
    parser.add_argument(
        "--early-close-on-grasp-gate",
        action="store_true",
        help="Close the gripper as soon as the grasp gate distance is reached during grasp approach.",
    )
    parser.add_argument(
        "--fast-forward-grasp-gate",
        action="store_true",
        help="When early close triggers, jump the event clock to close_gripper to avoid lingering in approach.",
    )
    parser.add_argument(
        "--release-gate-dist",
        type=float,
        default=-1.0,
        help="Hold release until this cube-target distance. Negative disables release gating.",
    )
    parser.add_argument("--release-gate-max-hold", type=int, default=240)
    parser.add_argument(
        "--strict-task-semantics",
        action="store_true",
        help=(
            "Opt in to the event-driven grasp/place/release controller contract. "
            "Legacy behavior remains the default for artifact compatibility."
        ),
    )
    parser.add_argument(
        "--strict-place-xy-tolerance-m",
        type=float,
        default=0.04,
        help=(
            "Horizontal released-cube target tolerance used by strict task "
            "semantics. It is recorded in the result contract."
        ),
    )
    parser.add_argument(
        "--blend-bc-checkpoint",
        default="",
        help="Optional BC checkpoint to blend into the evaluated policy for selected controller events.",
    )
    parser.add_argument(
        "--blend-bc-events",
        default="",
        help="Comma-separated controller events that should use the BC blend, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--blend-bc-alpha",
        type=float,
        default=1.0,
        help="Blend weight for --blend-bc-checkpoint. 1.0 means replace policy action with BC action.",
    )
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument(
        "--require-release-for-success",
        dest="require_release_for_success",
        action="store_true",
        help="Count success only after the cube has been released inside the target radius.",
    )
    success_group.add_argument(
        "--allow-success-before-release",
        dest="require_release_for_success",
        action="store_false",
        help="Count success when the cube reaches the target radius, even before release.",
    )
    parser.set_defaults(require_release_for_success=False)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Episode logging interval. 0 disables logs.",
    )
    parser.add_argument(
        "--output-json",
        default=default_json,
        help="Path to save summary and per-episode JSON.",
    )
    parser.add_argument(
        "--output-csv", default=default_csv, help="Path to save per-episode CSV."
    )
    parser.add_argument(
        "--output-step-csv",
        default=default_step_csv,
        help="Path to save v4 geometry and residual metrics for every rollout step.",
    )
    return parser.parse_args()


args = _parse_args()


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if os.path.isdir(ISAAC_TORCH_BUNDLE) and ISAAC_TORCH_BUNDLE not in sys.path:
            sys.path.insert(0, ISAAC_TORCH_BUNDLE)
        import torch  # noqa: F401


_ensure_torch()

from omni.isaac.kit import SimulationApp

_simulation_config = {
    "headless": False if args.live_vr else not args.render,
    "width": 1280,
    "height": 720,
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
    "max_gpu_count": 1,
}
if args.live_vr:
    isaac_root = os.environ.get(
        "ISAACSIM_ROOT", os.path.expanduser("~/isaac-sim-4.5.0")
    )
    xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
    experience_name = (
        "isaacsim.exp.base.xr.openxr.kit"
        if xr_mode == "openxr"
        else "isaacsim.exp.base.xr.vr.kit"
    )
    experience_path = os.path.join(isaac_root, "apps", experience_name)
    if not os.path.exists(experience_path):
        raise FileNotFoundError(f"Live VR experience not found: {experience_path}")
    _simulation_config["experience"] = experience_path

simulation_app = SimulationApp(
    _simulation_config,
    experience=os.environ.get("ISAAC_SIM_EXPERIENCE", ""),
)
print(
    f"[EvalRollout] SimulationApp headless={_simulation_config['headless']} live_vr={args.live_vr}",
    flush=True,
)

import torch  # noqa: E402
import carb  # noqa: E402
from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402

from rl import (  # noqa: E402
    HumanEncounterReplay,
    HumanTrajectoryReplay,
    IsaacPickPlaceEnv,
    PickPlaceEnvConfig,
    REWARD_VERSION,
    parse_pseudo_errp_sources,
)
from rl.actions import (
    ACTION_DIM,
    CONTROLLER_TARGET_MAX_DELTA_M,
    clip_action,
)  # noqa: E402
from rl.observations import (  # noqa: E402
    DYNAMIC_HRI_OBS_DIM,
    DYNAMIC_HRI_OBS_FIELD_NAMES,
    DYNAMIC_HRI_OBSERVATION_VERSION,
    HRI_OBS_DIM,
    HRI_OBS_FIELD_NAMES,
    MISSING_DISTANCE_M,
    OBSERVATION_DIM,
    flatten_hri_observation,
    observation_slices,
)
from rl.policies import MLPPolicy  # noqa: E402


def _start_live_vr_profile() -> None:
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

    enabled = []
    for extension_id in extension_ids:
        try:
            enable_extension(extension_id)
            enabled.append(extension_id)
        except Exception:
            continue
    if not enabled:
        raise RuntimeError("No XR extension could be enabled for live VR evaluation.")
    for _ in range(5):
        simulation_app.update()

    from omni.kit.xr.core import XRCore

    settings = carb.settings.get_settings()
    settings.set(f"/xr/profile/{profile_name}/adjustForUserHeight", False)
    settings.set(f"/defaults/xr/profile/{profile_name}/adjustForUserHeight", False)
    settings.set(f"/xr/profile/{profile_name}/system/display", xr_backend)
    settings.set(f"/defaults/xr/profile/{profile_name}/system/display", xr_backend)
    settings.set("/xr/ui/enabled", False)
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
        f"[EvalRollout] requested XR profile={profile_name} backend={xr_backend} "
        f"extensions={','.join(enabled)}",
        flush=True,
    )


def _set_live_vr_anchor(room_hmd_pos: np.ndarray) -> bool:
    import omni.usd
    from pxr import Gf, UsdGeom
    from vr_avatar import AVATAR_EYE_POS, ROOM_TO_WORLD_MATRIX_ROWS, room_to_world_point

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
        print(
            f"[EvalRollout] XR anchor applied translation={np.round(translation, 3)}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[EvalRollout] XR anchor update failed: {exc}", flush=True)
        return False


class LiveVRHumanProvider:
    def __init__(self) -> None:
        self.avatar = None
        self._state: dict[str, np.ndarray] = {}
        self._anchor_applied = False

    def setup(self, world) -> None:
        _start_live_vr_profile()
        from vr_avatar import VRAvatar

        self.avatar = VRAvatar()
        self.avatar.setup(world)

    def update(self) -> bool:
        if self.avatar is None:
            return False
        head_pos, left_pos, right_pos = self.avatar.update()
        if not self._anchor_applied:
            room_hmd_pos = self.avatar.capture_initial_hmd_pos()
            if room_hmd_pos is not None and _set_live_vr_anchor(room_hmd_pos):
                self.avatar.notify_anchor_applied()
                self._anchor_applied = True
                head_pos, left_pos, right_pos = self.avatar.update()

        state: dict[str, np.ndarray] = {}
        if head_pos is not None:
            state["human_head_pos"] = np.asarray(head_pos, dtype=np.float32)
        if left_pos is not None:
            state["human_left_hand_pos"] = np.asarray(left_pos, dtype=np.float32)
        if right_pos is not None:
            state["human_right_hand_pos"] = np.asarray(right_pos, dtype=np.float32)
        self._state = state
        return head_pos is not None and (left_pos is not None or right_pos is not None)

    def wait_until_ready(self, timeout_sec: float) -> None:
        started = time.time()
        while not self.update():
            simulation_app.update()
            if timeout_sec > 0.0 and time.time() - started >= timeout_sec:
                raise TimeoutError(
                    "Live VR tracking did not provide an HMD and hand pose before timeout."
                )
            time.sleep(0.01)
        print("[EvalRollout] live VR tracking ready", flush=True)

    def __call__(self) -> dict[str, np.ndarray]:
        return dict(self._state)


class PolicyRunner:
    def __init__(
        self,
        checkpoint_path: str,
        device_name: str,
        residual_gate_override: str = "checkpoint",
    ) -> None:
        self.checkpoint_path = _resolve_project_path(checkpoint_path)
        self.device = _select_device(device_name)
        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self.target_version = str(
            checkpoint.get("target_version", "task_space_action_v0")
        )
        self.action_version = str(
            checkpoint.get("action_version", "action_v0_task_space")
        )
        self.policy_mode = str(checkpoint.get("policy_mode", "direct"))
        self.output_activation = str(checkpoint.get("output_activation", "clip"))
        self.residual_scale = float(checkpoint.get("residual_scale", 1.0))
        self.xyz_only_residual = bool(checkpoint.get("xyz_only_residual", False))
        checkpoint_gate_mode = str(checkpoint.get("residual_gate_mode", "none"))
        self.residual_gate_mode = (
            checkpoint_gate_mode
            if residual_gate_override == "checkpoint"
            else residual_gate_override
        )
        if self.residual_gate_mode not in ("none", "distance"):
            raise ValueError(f"Unknown residual gate mode: {self.residual_gate_mode}")
        self.last_residual_norm = 0.0
        self.last_residual_gate = 0.0
        if self.target_version == "expert_arm_joint_action_v0":
            raise ValueError(
                "evaluate_rollout_policy.py currently supports 5D task-space policies only."
            )

        self.obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
        self.obs_std = _tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1)
        self.target_mean = None
        self.target_std = None
        self.target_min = None
        self.target_max = None
        if "target_mean" in checkpoint and "target_std" in checkpoint:
            self.target_mean = _tensor_to_numpy(checkpoint["target_mean"]).reshape(
                1, -1
            )
            self.target_std = _tensor_to_numpy(checkpoint["target_std"]).reshape(1, -1)
        if "target_min" in checkpoint and "target_max" in checkpoint:
            self.target_min = _tensor_to_numpy(checkpoint["target_min"]).reshape(1, -1)
            self.target_max = _tensor_to_numpy(checkpoint["target_max"]).reshape(1, -1)

        hidden_dims = tuple(
            int(value) for value in checkpoint.get("hidden_dims", (256, 256))
        )
        self.obs_dim = int(checkpoint.get("obs_dim", OBSERVATION_DIM))
        self.observation_fields = tuple(checkpoint.get("observation_fields", ()))
        action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
        if action_dim != ACTION_DIM:
            raise ValueError(
                f"Checkpoint action dim {action_dim} != runtime action dim {ACTION_DIM}"
            )

        self.model = MLPPolicy(self.obs_dim, action_dim, hidden_dims=hidden_dims).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.base_model = None
        self.base_obs_mean = None
        self.base_obs_std = None
        self.base_obs_dim = None
        self.base_checkpoint_path = ""
        if self.policy_mode == "residual":
            self._load_residual_base(checkpoint)
        self.metadata = {
            "checkpoint": self.checkpoint_path,
            "target_version": self.target_version,
            "action_version": self.action_version,
            "policy_mode": self.policy_mode,
            "output_activation": self.output_activation,
            "residual_scale": self.residual_scale,
            "xyz_only_residual": self.xyz_only_residual,
            "residual_gate_mode": self.residual_gate_mode,
            "source_bc_checkpoint": self.base_checkpoint_path,
            "observation_version": str(checkpoint.get("observation_version", "")),
            "reward_version": str(checkpoint.get("reward_version", "")),
            "obs_dim": self.obs_dim,
            "observation_fields": list(self.observation_fields),
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "torch_version": torch.__version__,
            "device": str(self.device),
        }
        print(
            f"[EvalRollout] loaded checkpoint={self.checkpoint_path} device={self.device} "
            f"torch={torch.__version__} action={self.action_version} obs_dim={self.obs_dim} "
            f"policy_mode={self.policy_mode} residual_gate={self.residual_gate_mode}",
            flush=True,
        )
        if self.device.type == "cuda":
            print(f"[EvalRollout] cuda={torch.cuda.get_device_name(0)}", flush=True)

    def predict(
        self,
        obs: np.ndarray,
        *,
        safety_gate: float | None = None,
        mask_human_obs: bool = False,
    ) -> np.ndarray:
        if mask_human_obs:
            obs = _mask_human_obs_for_policy(obs)
        residual_or_action = self._predict_model(
            self.model,
            obs,
            self.obs_mean,
            self.obs_std,
            self.obs_dim,
        )
        if self.xyz_only_residual:
            residual_or_action = residual_or_action.copy()
            residual_or_action[3:] = 0.0
        if self.policy_mode == "residual":
            if (
                self.base_model is None
                or self.base_obs_mean is None
                or self.base_obs_std is None
            ):
                raise RuntimeError(
                    "Residual policy checkpoint is missing a loadable source BC checkpoint."
                )
            base_action = self._predict_model(
                self.base_model,
                obs,
                self.base_obs_mean,
                self.base_obs_std,
                int(self.base_obs_dim),
            )
            gate = 1.0
            if self.residual_gate_mode == "distance":
                gate = (
                    float(np.clip(safety_gate, 0.0, 1.0))
                    if safety_gate is not None
                    else _distance_gate_from_flat_obs(obs)
                )
            scaled_residual = gate * float(self.residual_scale) * residual_or_action
            self.last_residual_gate = float(gate)
            self.last_residual_norm = float(np.linalg.norm(scaled_residual))
            return clip_action(base_action + scaled_residual)
        self.last_residual_gate = 0.0
        self.last_residual_norm = 0.0
        return residual_or_action

    def _predict_model(
        self,
        model: MLPPolicy,
        obs: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        obs_dim: int,
    ) -> np.ndarray:
        obs_policy = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_policy = _align_obs_dim(obs_policy, obs_dim)
        obs_norm = (obs_policy - obs_mean) / np.maximum(obs_std, 1e-6)
        with torch.no_grad():
            tensor = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)
            action = model(tensor).detach().cpu().numpy()[0]
        if self.output_activation == "tanh":
            action = np.tanh(action)
        if self.target_mean is not None and self.target_std is not None:
            action = (action.reshape(1, -1) * self.target_std + self.target_mean)[0]
            if self.target_min is not None and self.target_max is not None:
                action = np.clip(
                    action.reshape(1, -1), self.target_min, self.target_max
                )[0]
            return np.asarray(action, dtype=np.float32)
        return clip_action(action)

    def _load_residual_base(self, checkpoint: dict[str, Any]) -> None:
        source_path = str(checkpoint.get("source_bc_checkpoint", ""))
        if not source_path:
            raise ValueError(
                "Residual checkpoint is missing source_bc_checkpoint metadata."
            )
        self.base_checkpoint_path = _resolve_project_path(source_path)
        base_checkpoint = _torch_load(self.base_checkpoint_path, self.device)
        base_hidden_dims = tuple(
            int(value) for value in base_checkpoint.get("hidden_dims", (256, 256))
        )
        self.base_obs_dim = int(base_checkpoint.get("obs_dim", OBSERVATION_DIM))
        base_action_dim = int(base_checkpoint.get("action_dim", ACTION_DIM))
        if base_action_dim != ACTION_DIM:
            raise ValueError(
                f"Residual base checkpoint action dim {base_action_dim} != runtime action dim {ACTION_DIM}"
            )
        self.base_obs_mean = _tensor_to_numpy(base_checkpoint["obs_mean"]).reshape(
            1, -1
        )
        self.base_obs_std = _tensor_to_numpy(base_checkpoint["obs_std"]).reshape(1, -1)
        self.base_model = MLPPolicy(
            int(self.base_obs_dim),
            base_action_dim,
            hidden_dims=base_hidden_dims,
        ).to(self.device)
        self.base_model.load_state_dict(base_checkpoint["model_state_dict"])
        self.base_model.eval()


def _select_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda, but torch.cuda.is_available() is False"
        )
    return torch.device(requested)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_path = os.path.abspath(os.path.join(PROJECT_DIR, path))
    if os.path.exists(project_path):
        return project_path
    return project_path


def _resolve_output_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _align_obs_dim(obs_policy: np.ndarray, expected_dim: int) -> np.ndarray:
    if obs_policy.shape[1] == expected_dim:
        return obs_policy
    if obs_policy.shape[1] > expected_dim:
        return obs_policy[:, :expected_dim]
    pad = np.zeros(
        (obs_policy.shape[0], expected_dim - obs_policy.shape[1]),
        dtype=obs_policy.dtype,
    )
    return np.concatenate([obs_policy, pad], axis=1)


def _distance_gate_from_flat_obs(obs: np.ndarray) -> float:
    obs_policy = np.asarray(obs, dtype=np.float32).reshape(-1)
    gap_index = observation_slices()["min_hand_gripper_dist"].start
    if gap_index >= obs_policy.size:
        return 0.0
    return distance_gate(float(obs_policy[gap_index]), SAFETY_THRESHOLDS)


def _maybe_load_human_replay() -> HumanTrajectoryReplay | HumanEncounterReplay | None:
    if args.live_vr:
        return None
    if args.encounter_manifest:
        path = _resolve_project_path(args.encounter_manifest)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--encounter-manifest not found: {path}")
        return HumanEncounterReplay(
            path,
            episode_policy=args.encounter_policy,
            severity_mix=args.encounter_severity_mix,
            anchor_mode=args.encounter_anchor_mode,
            phase_match=args.encounter_phase_match,
            event_match=args.encounter_event_match,
            playback_timebase=args.encounter_timebase,
            playback_speed=args.encounter_playback_speed,
            seed=args.seed,
        )
    if not args.human_replay_data:
        return None
    path = _resolve_project_path(args.human_replay_data)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--human-replay-data not found: {path}")
    return HumanTrajectoryReplay(
        path,
        mode=args.human_replay_mode,
        episode_policy=args.human_replay_episode_policy,
        seed=args.seed,
    )


def _safety_observation_fields(runner: PolicyRunner) -> tuple[str, ...]:
    static_fields = tuple(HRI_OBS_FIELD_NAMES)
    dynamic_fields = tuple(DYNAMIC_HRI_OBS_FIELD_NAMES)
    fields = tuple(runner.observation_fields)
    if not fields:
        if int(runner.obs_dim) == int(HRI_OBS_DIM):
            return static_fields
        if int(runner.obs_dim) == int(DYNAMIC_HRI_OBS_DIM):
            return dynamic_fields
        raise ValueError(
            f"Safety checkpoint obs_dim={runner.obs_dim} has no observation_fields metadata."
        )
    if fields == static_fields and int(runner.obs_dim) == int(HRI_OBS_DIM):
        return fields
    if fields == dynamic_fields and int(runner.obs_dim) == int(DYNAMIC_HRI_OBS_DIM):
        return fields
    raise ValueError(
        "Safety checkpoint observation_fields do not match the supported "
        "83-D static or 109-D dynamic safety schema."
    )


def _run() -> None:
    if args.analytic_avoidance_offset_m < 0.0:
        raise ValueError("--analytic-avoidance-offset-m must be non-negative")
    if args.analytic_avoidance_offset_m > 0.0 and args.safety_residual_checkpoint:
        raise ValueError(
            "Use either --analytic-avoidance-offset-m or --safety-residual-checkpoint, not both."
        )
    if args.live_vr and args.synthetic_human:
        raise ValueError("--live-vr and --synthetic-human cannot be used together.")
    if args.rmpflow_human_safety_margin_m < 0.0:
        raise ValueError("--rmpflow-human-safety-margin-m must be non-negative")
    np.random.seed(args.seed)
    started_at = time.time()
    runner = PolicyRunner(
        args.checkpoint,
        args.device,
        residual_gate_override=args.residual_gate_mode,
    )
    safety_runner = (
        PolicyRunner(
            args.safety_residual_checkpoint,
            args.device,
            residual_gate_override="none",
        )
        if args.safety_residual_checkpoint
        else None
    )
    safety_observation_fields = (
        _safety_observation_fields(safety_runner)
        if safety_runner is not None
        else tuple(DYNAMIC_HRI_OBS_FIELD_NAMES)
    )
    safety_observation_dim = (
        int(safety_runner.obs_dim)
        if safety_runner is not None
        else int(DYNAMIC_HRI_OBS_DIM)
    )
    safety_observation_version = (
        str(safety_runner.metadata.get("observation_version", ""))
        if safety_runner is not None
        else DYNAMIC_HRI_OBSERVATION_VERSION
    )
    blend_events = _parse_event_set(args.blend_bc_events)
    blend_runner = (
        PolicyRunner(
            args.blend_bc_checkpoint,
            args.device,
            residual_gate_override=args.residual_gate_mode,
        )
        if args.blend_bc_checkpoint and blend_events
        else None
    )
    blend_alpha = float(np.clip(args.blend_bc_alpha, 0.0, 1.0))
    release_gate_dist = (
        None if args.release_gate_dist < 0.0 else float(args.release_gate_dist)
    )
    checkpoint_reward_version = str(
        runner.metadata.get("reward_version", "") or REWARD_VERSION
    )
    effective_task_reward_version = str(
        args.task_reward_version or checkpoint_reward_version
    )
    pseudo_errp_sources = parse_pseudo_errp_sources(args.pseudo_errp_sources)
    human_replay = _maybe_load_human_replay()
    evaluation_episodes = int(args.episodes)
    if evaluation_episodes <= 0:
        if isinstance(human_replay, HumanEncounterReplay):
            evaluation_episodes = int(human_replay.info.scenario_count)
        else:
            raise ValueError(
                "--episodes must be positive unless --encounter-manifest is used."
            )
    live_vr = LiveVRHumanProvider() if args.live_vr else None
    env = IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            max_episode_steps=args.max_steps,
            success_dist=args.success_dist,
            action_scale=args.action_scale,
            action_version=runner.action_version,
            fixed_orientation=args.fixed_orientation,
            gripper_mode=args.gripper_mode,
            phase_gate_close_dist=args.phase_gate_close_dist,
            phase_gate_max_hold=args.phase_gate_max_hold,
            early_close_on_grasp_gate=args.early_close_on_grasp_gate,
            fast_forward_grasp_gate=args.fast_forward_grasp_gate,
            release_gate_dist=release_gate_dist,
            release_gate_max_hold=args.release_gate_max_hold,
            require_release_for_success=args.require_release_for_success,
            strict_task_semantics=bool(args.strict_task_semantics),
            strict_place_xy_tolerance_m=float(
                args.strict_place_xy_tolerance_m
            ),
            reward_version=effective_task_reward_version,
            observation_mode="flat",
            seed=args.seed,
            render=args.render or args.live_vr,
            pseudo_errp_enabled=args.pseudo_errp_enabled,
            pseudo_errp_sources=pseudo_errp_sources,
            visualize_human_replay=args.visualize_human_replay or args.live_vr,
            human_replay_visual_z_offset=args.human_replay_visual_z_offset,
            synthetic_human_enabled=args.synthetic_human,
            synthetic_human_episode_prob=args.synthetic_human_episode_prob,
            synthetic_human_start_min_step=args.synthetic_human_start_min_step,
            synthetic_human_start_max_step=args.synthetic_human_start_max_step,
            synthetic_human_duration_steps=args.synthetic_human_duration_steps,
            synthetic_human_near_dist=args.synthetic_human_near_dist,
            synthetic_human_collision_dist=args.synthetic_human_collision_dist,
            physical_safety_controller=args.physical_safety_controller,
            rmpflow_human_safety_margin_m=args.rmpflow_human_safety_margin_m,
            visualize_physical_safety=args.visualize_physical_safety,
            cbf_safe_gap_m=args.cbf_safe_gap_m,
            cbf_activation_gap_m=args.cbf_activation_gap_m,
            cbf_gamma_per_s=args.cbf_gamma_per_s,
            cbf_prediction_horizon_s=args.cbf_prediction_horizon_s,
            cbf_max_prediction_buffer_m=args.cbf_max_prediction_buffer_m,
            cbf_max_joint_speed_rad_s=args.cbf_max_joint_speed_rad_s,
        ),
        human_state_fn=live_vr if live_vr is not None else human_replay,
    )
    if live_vr is not None:
        live_vr.setup(env.world)
        env.world.reset()
        env.world.play()
        env.controller.reset()
        live_vr.wait_until_ready(float(args.vr_tracking_timeout_sec))

    rows = []
    step_rows = []
    print(
        f"[EvalRollout] checkpoint={args.checkpoint} episodes={evaluation_episodes} "
        f"max_steps={args.max_steps} render={args.render} gripper_mode={args.gripper_mode} "
        f"human_replay={human_replay.path if human_replay is not None else 'off'} "
        f"live_vr={args.live_vr} synthetic_human={args.synthetic_human} "
        f"physical_safety={args.physical_safety_controller}",
        flush=True,
    )
    try:
        for episode_idx in range(evaluation_episodes):
            episode_seed = int(args.seed + episode_idx)
            source_restoration: dict[str, Any] | None = None
            if human_replay is not None:
                human_replay.reset(episode_idx, seed=episode_seed)
            human_replay_episode = (
                human_replay.episode_name if human_replay is not None else ""
            )
            encounter = (
                human_replay.current_scenario
                if isinstance(human_replay, HumanEncounterReplay)
                else {}
            )
            if isinstance(human_replay, HumanEncounterReplay):
                source_restoration = human_replay.source_restoration(
                    screening_seed=episode_seed,
                    allow_legacy_fallback=args.allow_legacy_source_configuration,
                )
                if not bool(
                    source_restoration.get(
                        "source_configuration_available",
                        False,
                    )
                ):
                    rows.append(
                        _unavailable_source_episode_row(
                            episode_idx=episode_idx,
                            episode_seed=episode_seed,
                            human_replay_episode=human_replay_episode,
                            encounter=encounter,
                            restoration=source_restoration,
                        )
                    )
                    print(
                        f"[EvalRollout] episode={episode_idx:04d} "
                        "skipped=source_configuration_unavailable "
                        f"reason={source_restoration.get('restoration_reason', '')}",
                        flush=True,
                    )
                    continue
            if live_vr is not None:
                live_vr.wait_until_ready(float(args.vr_tracking_timeout_sec))
            try:
                obs, info = env.reset(
                    seed=episode_seed,
                    source_restoration=source_restoration,
                )
            except ValueError as exc:
                if source_restoration is None or not str(exc).startswith(
                    ("source_configuration_unavailable", "source_configuration_pose_mismatch")
                ):
                    raise
                failed_restoration = dict(source_restoration)
                failed_restoration["source_configuration_available"] = False
                failed_restoration["restoration_reason"] = str(exc)
                failed_restoration["pose_mismatch"] = (
                    "pose_mismatch" in str(exc)
                )
                failed_restoration["pose_mismatch_reason"] = str(exc)
                rows.append(
                    _unavailable_source_episode_row(
                        episode_idx=episode_idx,
                        episode_seed=episode_seed,
                        human_replay_episode=human_replay_episode,
                        encounter=encounter,
                        restoration=failed_restoration,
                    )
                )
                print(
                    f"[EvalRollout] episode={episode_idx:04d} "
                    f"skipped={exc}",
                    flush=True,
                )
                continue
            cube_poses = [cube.get_world_pose() for cube in env.cubes]
            initial_cube_positions = np.asarray(
                [pose[0] for pose in cube_poses],
                dtype=float,
            )
            initial_cube_orientations = np.asarray(
                [pose[1] for pose in cube_poses],
                dtype=float,
            )
            target_position, target_orientation = env.place_target.get_world_pose()
            initial_scene_layout_id = scene_layout_id(
                initial_cube_positions,
                initial_cube_orientations,
                target_position,
                target_orientation,
            )
            initial_obs_dict = info.get("obs_dict", {})
            initial_cube_position = _obs_vector3(initial_obs_dict, "cube_pos")
            if initial_cube_position is None:
                initial_cube_position = np.full(3, np.nan, dtype=float)
            place_target_position = _obs_vector3(initial_obs_dict, "place_target_pos")
            if place_target_position is None:
                place_target_position = np.full(3, np.nan, dtype=float)
            motion_tracker = CartesianMotionTracker(
                _obs_vector3(initial_obs_dict, "ee_pos"),
                float(info.get("sim_time", 0.0)),
            )
            restoration_diagnostics = dict(
                info.get(
                    "source_restoration",
                    source_restoration or {},
                )
            )
            total_reward = 0.0
            min_cube_target_dist = float(info["cube_target_dist"])
            min_ee_cube_dist = float(info["ee_cube_dist"])
            min_hand_gripper_dist = _obs_scalar(
                info.get("obs_dict", {}),
                "min_hand_gripper_dist",
                default=MISSING_DISTANCE_M,
            )
            grasped_any = bool(info["has_grasped_cube"])
            errp_count = 0
            errp_feedback_sum = 0.0
            errp_uncertainty_sum = 0.0
            max_errp_feedback = 0.0
            max_errp_uncertainty = 0.0
            errp_source_code = 0
            errp_source_counts: dict[str, int] = {}
            reward_components_total: dict[str, float] = {}
            safety_gate_sum = 0.0
            safety_gate_max = 0.0
            safety_gate_active_count = 0
            safety_gate_full_count = 0
            safety_gate_near_overlap_count = 0
            safety_gate_collision_overlap_count = 0
            near_human_count = 0
            human_collision_count = 0
            near_miss_count = 0
            min_hand_gripper_center_dist = MISSING_DISTANCE_M
            terminated = False
            truncated = False
            bc_blend_count = 0
            safety_residual_count = 0
            safety_residual_norm_sum = 0.0
            encounter_active_count = 0
            safety_applied_position_m_sum = 0.0
            safety_applied_position_m_max = 0.0
            collision_steps = 0
            near_steps = 0
            near_miss_steps = 0
            geometry_valid_steps = 0
            collision_event_count = 0
            collision_was_active = False
            collision_consecutive_steps = 0
            collision_max_consecutive_steps = 0
            min_surface_gap = MISSING_DISTANCE_M
            minimum_ttc_s = 10.0
            minimum_ttc_valid = False
            safety_query_time_ms_sum = 0.0
            closest_link_counts: dict[str, int] = {}
            collision_link_counts: dict[str, int] = {}
            physical_safety_active_count = 0
            physical_safety_feasible_count = 0
            physical_safety_intervention_available_count = 0
            physical_safety_intervention_count = 0
            physical_safety_intervention_norm_sum = 0.0
            physical_safety_intervention_norm_max = 0.0
            physical_safety_slack_sum = 0.0
            physical_safety_slack_max = 0.0
            physical_safety_violation_before_sum = 0.0
            physical_safety_violation_before_max = 0.0
            physical_safety_violation_after_sum = 0.0
            physical_safety_violation_after_max = 0.0
            physical_safety_solve_time_ms_sum = 0.0
            gate_ee_acceleration_norms: list[float] = []
            gate_ee_jerk_norms: list[float] = []
            initial_controller_event = int(info["controller_event"])
            maximum_controller_event_reached = initial_controller_event
            controller_lift_phase_reached = bool(
                4 <= initial_controller_event < 10
            )
            controller_place_phase_reached = bool(
                5 <= initial_controller_event < 10
            )
            controller_release_phase_reached = bool(
                7 <= initial_controller_event < 10
            )
            cube_entered_target_tolerance = bool(
                float(info["cube_target_dist"]) <= float(args.success_dist)
            )
            released_after_grasp = False
            grasp_was_observed = bool(info["has_grasped_cube"])
            has_grasped_previous = bool(info["has_grasped_cube"])
            gripper_closed_previous = bool(info.get("gripper_closed", False))
            first_grasp_step = 0 if has_grasped_previous else -1
            first_grasp_loss_step = -1
            first_target_entry_step = (
                0 if cube_entered_target_tolerance else -1
            )
            first_release_step = -1
            first_success_step = -1
            grasp_acquisition_count = 0
            grasp_loss_count = 0
            release_command_count = 0
            task_phase_paused_steps = 0
            task_phase_reentry_count = 0
            physical_intervention_consecutive_steps = 0
            max_consecutive_physical_intervention_steps = 0
            physical_intervention_steps_before_grasp = 0
            physical_intervention_steps_during_transport = 0
            physical_intervention_steps_during_place = 0
            initial_strict_payload = _mapping_or_empty(
                info.get("strict_task_semantics")
            )
            strict_semantics_config: dict[str, Any] = dict(
                _mapping_or_empty(initial_strict_payload.get("config"))
            )
            strict_semantics_final_state: dict[str, Any] = dict(
                _mapping_or_empty(initial_strict_payload.get("state"))
            )
            strict_semantics_last_decision: dict[str, Any] = dict(
                _mapping_or_empty(initial_strict_payload.get("last_decision"))
            )

            for _ in range(args.max_steps):
                if live_vr is not None:
                    live_vr.update()
                current_obs_dict = info.get("obs_dict", {})
                current_safety_gate = _safety_distance_gate(
                    current_obs_dict,
                    start_dist=args.safety_gate_start_dist,
                    full_dist=args.safety_gate_full_dist,
                )
                action = runner.predict(
                    np.asarray(obs, dtype=np.float32),
                    safety_gate=current_safety_gate,
                    mask_human_obs=args.mask_human_obs_for_policy,
                )
                pre_surface_gap = _obs_scalar(
                    current_obs_dict,
                    "min_hand_end_effector_surface_gap",
                    default=_obs_scalar(
                        current_obs_dict,
                        "min_hand_gripper_dist",
                        default=MISSING_DISTANCE_M,
                    ),
                )
                pre_center_dist = _obs_scalar(
                    current_obs_dict,
                    "min_hand_gripper_center_dist",
                    default=MISSING_DISTANCE_M,
                )
                safety_residual = None
                raw_residual = np.zeros(ACTION_DIM, dtype=np.float32)
                applied_residual = np.zeros(ACTION_DIM, dtype=np.float32)
                if safety_runner is not None:
                    hri_obs = flatten_hri_observation(
                        current_obs_dict,
                        field_names=safety_observation_fields,
                    )
                    safety_residual = safety_runner.predict(hri_obs)
                elif args.analytic_avoidance_offset_m > 0.0:
                    safety_residual = _analytic_avoidance_residual(
                        current_obs_dict,
                        full_gate_offset_m=float(args.analytic_avoidance_offset_m),
                        residual_alpha=float(args.safety_residual_alpha),
                        action_scale=float(args.action_scale),
                    )
                if safety_residual is not None:
                    raw_residual = np.asarray(safety_residual, dtype=np.float32).copy()
                    applied_residual = (
                        float(current_safety_gate)
                        * float(args.safety_residual_alpha)
                        * safety_residual
                    )
                    action = clip_action(action + applied_residual)
                    if current_safety_gate > 0.0:
                        applied_position_m = (
                            float(np.linalg.norm(applied_residual[:3]))
                            * float(CONTROLLER_TARGET_MAX_DELTA_M)
                            * float(args.action_scale)
                        )
                        safety_residual_count += 1
                        safety_residual_norm_sum += float(
                            np.linalg.norm(safety_residual)
                        )
                        safety_applied_position_m_sum += applied_position_m
                        safety_applied_position_m_max = max(
                            safety_applied_position_m_max,
                            applied_position_m,
                        )
                if (
                    blend_runner is not None
                    and int(info["controller_event"]) in blend_events
                ):
                    bc_action = blend_runner.predict(
                        np.asarray(obs, dtype=np.float32),
                        safety_gate=current_safety_gate,
                        mask_human_obs=args.mask_human_obs_for_policy,
                    )
                    action = clip_action(
                        (1.0 - blend_alpha) * action + blend_alpha * bc_action
                    )
                    bc_blend_count += 1
                controller_event_before = int(info.get("controller_event", -1))
                controller_t_before = float(info.get("controller_t", 0.0))
                task_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
                obs, reward, terminated, truncated, info = env.step(action)
                if args.render and args.render_step_delay_sec > 0.0:
                    time.sleep(float(args.render_step_delay_sec))
                total_reward += float(reward)
                min_cube_target_dist = min(
                    min_cube_target_dist, float(info["cube_target_dist"])
                )
                min_ee_cube_dist = min(min_ee_cube_dist, float(info["ee_cube_dist"]))
                grasped_any = grasped_any or bool(info["has_grasped_cube"])
                has_grasped_now = bool(info["has_grasped_cube"])
                step_index = int(info["step"])
                grasp_acquired = bool(
                    has_grasped_now and not has_grasped_previous
                )
                grasp_lost = bool(
                    has_grasped_previous and not has_grasped_now
                )
                if grasp_acquired:
                    grasp_acquisition_count += 1
                    if first_grasp_step < 0:
                        first_grasp_step = step_index
                if grasp_lost:
                    grasp_loss_count += 1
                    if first_grasp_loss_step < 0:
                        first_grasp_loss_step = step_index
                released_after_grasp = released_after_grasp or bool(
                    grasp_was_observed and not has_grasped_now
                )
                grasp_was_observed = grasp_was_observed or has_grasped_now
                gripper_closed_now = bool(info.get("gripper_closed", False))
                gripper_command = str(info.get("gripper_command", "") or "")
                release_commanded = bool(
                    (gripper_command == "open" and gripper_closed_previous)
                    or (gripper_closed_previous and not gripper_closed_now)
                )
                if release_commanded:
                    release_command_count += 1
                    if first_release_step < 0:
                        first_release_step = step_index
                controller_event = int(info["controller_event"])
                if not (
                    args.strict_task_semantics and controller_event >= 10
                ):
                    maximum_controller_event_reached = max(
                        maximum_controller_event_reached, controller_event
                    )
                controller_lift_phase_reached = bool(
                    controller_lift_phase_reached
                    or 4 <= controller_event < 10
                )
                controller_place_phase_reached = bool(
                    controller_place_phase_reached
                    or 5 <= controller_event < 10
                )
                controller_release_phase_reached = bool(
                    controller_release_phase_reached
                    or 7 <= controller_event < 10
                )
                target_entered_now = bool(
                    float(info["cube_target_dist"]) <= float(args.success_dist)
                )
                if target_entered_now and first_target_entry_step < 0:
                    first_target_entry_step = step_index
                cube_entered_target_tolerance = bool(
                    cube_entered_target_tolerance or target_entered_now
                )
                step_success = bool(info.get("success", False))
                if step_success and first_success_step < 0:
                    first_success_step = step_index
                phase_control = _mapping_or_empty(info.get("task_phase_control"))
                task_phase_paused = bool(
                    info.get(
                        "task_phase_paused",
                        phase_control.get("paused", False),
                    )
                )
                task_phase_reentry = bool(
                    info.get(
                        "task_phase_reentry",
                        phase_control.get("reentry", False),
                    )
                )
                task_phase_paused_steps += int(task_phase_paused)
                task_phase_reentry_count += int(task_phase_reentry)
                strict_payload = _mapping_or_empty(
                    info.get("strict_task_semantics")
                )
                strict_config = _mapping_or_empty(strict_payload.get("config"))
                strict_state = _mapping_or_empty(strict_payload.get("state"))
                strict_decision = _mapping_or_empty(
                    strict_payload.get("last_decision")
                )
                if strict_config:
                    strict_semantics_config = dict(strict_config)
                if strict_state:
                    strict_semantics_final_state = dict(strict_state)
                if strict_decision:
                    strict_semantics_last_decision = dict(strict_decision)
                errp_feedback = float(info["errp_feedback"])
                errp_uncertainty = float(info.get("errp_uncertainty", 0.0))
                errp_count += int(info.get("errp_label", errp_feedback >= 0.5))
                errp_feedback_sum += errp_feedback
                errp_uncertainty_sum += errp_uncertainty
                max_errp_feedback = max(max_errp_feedback, errp_feedback)
                max_errp_uncertainty = max(max_errp_uncertainty, errp_uncertainty)
                errp_source_code |= int(info.get("errp_source_code", 0))
                for source_name in info.get("errp_source_names", ()):
                    errp_source_counts[source_name] = (
                        errp_source_counts.get(source_name, 0) + 1
                    )
                for name, value in info["reward_components"].items():
                    reward_components_total[name] = reward_components_total.get(
                        name, 0.0
                    ) + float(value)
                safety_gate = _safety_distance_gate(
                    info.get("obs_dict", {}),
                    start_dist=args.safety_gate_start_dist,
                    full_dist=args.safety_gate_full_dist,
                )
                safety_gate_sum += safety_gate
                safety_gate_max = max(safety_gate_max, safety_gate)
                safety_gate_active_count += int(safety_gate > 0.0)
                safety_gate_full_count += int(safety_gate >= 1.0)
                obs_dict = info.get("obs_dict", {})
                near_human = _obs_flag(obs_dict, "near_human")
                human_collision = _obs_flag(obs_dict, "human_robot_collision")
                geometry_valid = _obs_flag(obs_dict, "geometry_valid")
                collision_steps += int(human_collision)
                near_steps += int(near_human)
                near_miss_steps += int(_obs_flag(obs_dict, "near_miss"))
                geometry_valid_steps += int(geometry_valid)
                if human_collision and not collision_was_active:
                    collision_event_count += 1
                collision_consecutive_steps = (
                    collision_consecutive_steps + 1 if human_collision else 0
                )
                collision_max_consecutive_steps = max(
                    collision_max_consecutive_steps,
                    collision_consecutive_steps,
                )
                collision_was_active = bool(human_collision)
                min_hand_gripper_dist = min(
                    min_hand_gripper_dist,
                    _obs_scalar(
                        obs_dict,
                        "min_hand_gripper_dist",
                        default=MISSING_DISTANCE_M,
                    ),
                )
                min_hand_gripper_center_dist = min(
                    min_hand_gripper_center_dist,
                    _obs_scalar(
                        obs_dict,
                        "min_hand_gripper_center_dist",
                        default=MISSING_DISTANCE_M,
                    ),
                )
                near_human_count += int(near_human)
                human_collision_count += int(human_collision)
                near_miss_count += int(_obs_flag(obs_dict, "near_miss"))
                safety_gate_near_overlap_count += int(safety_gate > 0.0 and near_human)
                safety_gate_collision_overlap_count += int(
                    safety_gate > 0.0 and human_collision
                )
                post_surface_gap = _obs_scalar(
                    obs_dict,
                    "min_hand_end_effector_surface_gap",
                    default=_obs_scalar(
                        obs_dict,
                        "min_hand_gripper_dist",
                        default=MISSING_DISTANCE_M,
                    ),
                )
                if geometry_valid:
                    min_surface_gap = min(min_surface_gap, post_surface_gap)
                step_ttc_s, step_ttc_valid = _minimum_ttc(obs_dict)
                if step_ttc_valid:
                    minimum_ttc_s = min(minimum_ttc_s, step_ttc_s)
                    minimum_ttc_valid = True
                safety_query_time_ms_sum += float(
                    info.get("safety_query_time_ms", 0.0)
                )
                for key in ("closest_link_left", "closest_link_right"):
                    link_name = str(info.get(key, ""))
                    if link_name:
                        closest_link_counts[link_name] = (
                            closest_link_counts.get(link_name, 0) + 1
                        )
                for contact_key, link_key in (
                    ("contact_left", "closest_link_left"),
                    ("contact_right", "closest_link_right"),
                ):
                    if bool(info.get(contact_key, False)):
                        link_name = str(info.get(link_key, ""))
                        if link_name:
                            collision_link_counts[link_name] = (
                                collision_link_counts.get(link_name, 0) + 1
                            )
                post_ee_position = _obs_vector3(obs_dict, "ee_pos")
                post_cube_position = _obs_vector3(obs_dict, "cube_pos")
                cube_to_target = _obs_vector3(
                    obs_dict, "cube_to_place_target"
                )
                post_left_hand_position = _obs_vector3(obs_dict, "human_left_hand_pos")
                post_right_hand_position = _obs_vector3(
                    obs_dict, "human_right_hand_pos"
                )
                if post_ee_position is None:
                    post_ee_position = np.full(3, np.nan, dtype=float)
                if post_cube_position is None:
                    post_cube_position = np.full(3, np.nan, dtype=float)
                if cube_to_target is None:
                    cube_to_target = np.full(3, np.nan, dtype=float)
                if post_left_hand_position is None:
                    post_left_hand_position = np.full(3, np.nan, dtype=float)
                if post_right_hand_position is None:
                    post_right_hand_position = np.full(3, np.nan, dtype=float)
                try:
                    cube_linear_velocity = np.asarray(
                        env.active_cube.get_linear_velocity(), dtype=float
                    ).reshape(-1)
                    cube_speed_valid = bool(
                        cube_linear_velocity.size >= 3
                        and np.all(np.isfinite(cube_linear_velocity[:3]))
                    )
                    cube_speed_mps = (
                        float(np.linalg.norm(cube_linear_velocity[:3]))
                        if cube_speed_valid
                        else 0.0
                    )
                except Exception:
                    cube_speed_valid = False
                    cube_speed_mps = 0.0
                motion_sample = motion_tracker.update(
                    post_ee_position,
                    float(info.get("sim_time", 0.0)),
                )
                reward_components = info.get("reward_components", {})
                physical_safety = info.get("physical_safety", {})
                physical_active = bool(physical_safety.get("active", False))
                physical_feasible = bool(physical_safety.get("feasible", True))
                intervention_available = bool(
                    physical_safety.get("intervention_available", False)
                )
                intervention_norm = float(
                    physical_safety.get("intervention_norm_radps", 0.0)
                )
                physical_slack = float(
                    physical_safety.get(
                        "slack_mps", physical_safety.get("slack_radps", 0.0)
                    )
                )
                physical_violation_before = float(
                    physical_safety.get(
                        "max_constraint_violation_before_mps",
                        physical_safety.get("max_constraint_violation_before", 0.0),
                    )
                )
                physical_violation_after = float(
                    physical_safety.get(
                        "max_constraint_violation_after_mps",
                        physical_safety.get("max_constraint_violation_after", 0.0),
                    )
                )
                physical_solve_time_ms = float(
                    physical_safety.get("solve_time_ms", 0.0)
                )
                physical_numeric = {
                    "intervention_norm_radps": intervention_norm,
                    "slack_mps": physical_slack,
                    "constraint_violation_before_mps": physical_violation_before,
                    "constraint_violation_after_mps": physical_violation_after,
                    "solve_time_ms": physical_solve_time_ms,
                }
                invalid_physical = [
                    name
                    for name, value in physical_numeric.items()
                    if not np.isfinite(value) or value < 0.0
                ]
                if invalid_physical:
                    raise RuntimeError(
                        "Physical-safety diagnostics must be finite and "
                        f"non-negative: {invalid_physical}"
                    )
                physical_safety_active_count += int(physical_active)
                physical_safety_feasible_count += int(physical_feasible)
                physical_safety_intervention_available_count += int(
                    intervention_available
                )
                physical_intervened = bool(
                    intervention_available and intervention_norm > 1e-8
                )
                physical_safety_intervention_count += int(physical_intervened)
                if physical_intervened:
                    physical_intervention_consecutive_steps += 1
                    if first_grasp_step < 0:
                        physical_intervention_steps_before_grasp += 1
                    elif controller_event in (4, 5):
                        physical_intervention_steps_during_transport += 1
                    elif controller_event >= 6:
                        physical_intervention_steps_during_place += 1
                else:
                    physical_intervention_consecutive_steps = 0
                max_consecutive_physical_intervention_steps = max(
                    max_consecutive_physical_intervention_steps,
                    physical_intervention_consecutive_steps,
                )
                physical_safety_intervention_norm_sum += intervention_norm
                physical_safety_intervention_norm_max = max(
                    physical_safety_intervention_norm_max, intervention_norm
                )
                physical_safety_slack_sum += physical_slack
                physical_safety_slack_max = max(
                    physical_safety_slack_max, physical_slack
                )
                physical_safety_violation_before_sum += physical_violation_before
                physical_safety_violation_before_max = max(
                    physical_safety_violation_before_max,
                    physical_violation_before,
                )
                physical_safety_violation_after_sum += physical_violation_after
                physical_safety_violation_after_max = max(
                    physical_safety_violation_after_max,
                    physical_violation_after,
                )
                physical_safety_solve_time_ms_sum += physical_solve_time_ms
                encounter_aux = info.get("human_replay_aux_state", {})
                encounter_active = int(
                    float(encounter_aux.get("encounter_active", 0.0)) > 0.5
                )
                encounter_active_count += encounter_active
                if current_safety_gate > 0.0 and motion_sample.acceleration_valid:
                    gate_ee_acceleration_norms.append(
                        motion_sample.acceleration_norm_mps2
                    )
                if current_safety_gate > 0.0 and motion_sample.jerk_valid:
                    gate_ee_jerk_norms.append(motion_sample.jerk_norm_mps3)
                left_hand_velocity = _obs_vector3(
                    obs_dict, "left_hand_vel_filtered_mps"
                )
                right_hand_velocity = _obs_vector3(
                    obs_dict, "right_hand_vel_filtered_mps"
                )
                left_robot_velocity = _obs_vector3(
                    obs_dict, "left_closest_robot_velocity_world_mps"
                )
                right_robot_velocity = _obs_vector3(
                    obs_dict, "right_closest_robot_velocity_world_mps"
                )
                left_relative_velocity = _obs_vector3(
                    obs_dict, "left_relative_velocity_world_mps"
                )
                right_relative_velocity = _obs_vector3(
                    obs_dict, "right_relative_velocity_world_mps"
                )

                step_rows.append(
                    {
                        "episode": int(episode_idx),
                        "seed": int(episode_seed),
                        "step": int(info["step"]),
                        "sim_time": float(info.get("sim_time", 0.0)),
                        "controller_event_before": int(
                            phase_control.get(
                                "event_before", controller_event_before
                            )
                        ),
                        "controller_event_after": int(
                            phase_control.get("event_after", controller_event)
                        ),
                        "controller_t_before": float(
                            phase_control.get(
                                "controller_t_before", controller_t_before
                            )
                        ),
                        "controller_t_after": float(
                            phase_control.get(
                                "controller_t_after",
                                info.get("controller_t", 0.0),
                            )
                        ),
                        "task_phase_before": _task_phase_name(
                            controller_event_before
                        ),
                        "task_phase_after": _task_phase_name(controller_event),
                        "phase_hold_steps": int(info.get("phase_hold_steps", 0)),
                        "task_phase_advanced": int(
                            bool(
                                phase_control.get(
                                    "phase_changed",
                                    info.get("task_phase_advanced", True),
                                )
                            )
                        ),
                        "task_phase_paused": int(task_phase_paused),
                        "task_phase_reentry": int(task_phase_reentry),
                        "task_phase_control_reason": str(
                            phase_control.get("reason", "")
                        ),
                        "task_action_x": float(task_action[0]),
                        "task_action_y": float(task_action[1]),
                        "task_action_z": float(task_action[2]),
                        "task_action_yaw": float(task_action[3]),
                        "task_action_gripper": float(task_action[4]),
                        "gripper_command": gripper_command,
                        "gripper_closed": int(gripper_closed_now),
                        "gripper_width_m": _obs_scalar(
                            obs_dict, "gripper_width", default=0.0
                        ),
                        "has_grasped_cube": int(has_grasped_now),
                        "grasp_acquired": int(grasp_acquired),
                        "grasp_lost": int(grasp_lost),
                        "release_commanded": int(release_commanded),
                        "ee_cube_dist_m": float(info["ee_cube_dist"]),
                        "cube_target_dist_m": float(info["cube_target_dist"]),
                        "cube_target_xy_error_m": float(
                            np.linalg.norm(cube_to_target[:2])
                        ),
                        "cube_target_z_error_m": float(abs(cube_to_target[2])),
                        "post_cube_x": float(post_cube_position[0]),
                        "post_cube_y": float(post_cube_position[1]),
                        "post_cube_z": float(post_cube_position[2]),
                        "cube_speed_mps": cube_speed_mps,
                        "cube_speed_valid": int(cube_speed_valid),
                        "step_success": int(step_success),
                        "step_truncated": int(bool(truncated)),
                        "task_terminal_reason": str(
                            info.get("task_terminal_reason", "") or ""
                        ),
                        "strict_task_semantics_enabled": int(
                            bool(
                                strict_payload.get(
                                    "enabled", args.strict_task_semantics
                                )
                            )
                        ),
                        "strict_success_latched": int(
                            bool(strict_state.get("success_latched", False))
                        ),
                        "strict_failure_reason": str(
                            strict_state.get("failure_reason", "") or ""
                        ),
                        "strict_transition_reason": str(
                            strict_decision.get("reason", "") or ""
                        ),
                        "strict_transition_held": int(
                            bool(strict_decision.get("held", False))
                        ),
                        "strict_retry_started": int(
                            bool(strict_decision.get("retry_started", False))
                        ),
                        "strict_transition_reentry": int(
                            bool(strict_decision.get("reentry", False))
                        ),
                        "strict_grasp_retry_count": int(
                            strict_state.get("grasp_retry_count", 0) or 0
                        ),
                        "strict_phase_reentry_count": int(
                            strict_state.get("phase_reentry_count", 0) or 0
                        ),
                        "strict_phase_hold_total_steps": int(
                            strict_state.get("phase_hold_total_steps", 0) or 0
                        ),
                        "strict_place_ready_latched": int(
                            bool(strict_state.get("place_ready_latched", False))
                        ),
                        "strict_place_ready_streak": int(
                            strict_state.get("place_ready_streak", 0) or 0
                        ),
                        "strict_release_settle_streak": int(
                            strict_state.get("release_settle_streak", 0) or 0
                        ),
                        "strict_cbf_pause_steps": int(
                            strict_state.get("cbf_pause_steps", 0) or 0
                        ),
                        "strict_task_semantics_state_json": _json_mapping(
                            strict_state
                        ),
                        "strict_task_semantics_last_decision_json": _json_mapping(
                            strict_decision
                        ),
                        "encounter_id": str(encounter.get("id", "")),
                        "encounter_target_severity": str(
                            encounter.get("target_severity", "")
                        ),
                        "encounter_active": encounter_active,
                        "encounter_source_step": int(
                            encounter_aux.get("encounter_source_step", -1)
                        ),
                        "encounter_source_time_s": float(
                            encounter_aux.get("encounter_source_time_s", -1.0)
                        ),
                        "pre_surface_gap_m": float(pre_surface_gap),
                        "post_surface_gap_m": float(post_surface_gap),
                        "surface_gap_delta_m": float(
                            post_surface_gap - pre_surface_gap
                        ),
                        "pre_center_dist_m": float(pre_center_dist),
                        "gate": float(current_safety_gate),
                        "gate_active": int(current_safety_gate > 0.0),
                        "raw_residual_x": float(raw_residual[0]),
                        "raw_residual_y": float(raw_residual[1]),
                        "raw_residual_z": float(raw_residual[2]),
                        "raw_residual_norm": float(np.linalg.norm(raw_residual[:3])),
                        "applied_residual_x": float(applied_residual[0]),
                        "applied_residual_y": float(applied_residual[1]),
                        "applied_residual_z": float(applied_residual[2]),
                        "applied_residual_norm": float(
                            np.linalg.norm(applied_residual[:3])
                        ),
                        "applied_position_m": float(
                            np.linalg.norm(applied_residual[:3])
                            * float(CONTROLLER_TARGET_MAX_DELTA_M)
                            * float(args.action_scale)
                        ),
                        "near_human": int(near_human),
                        "near_miss": int(_obs_flag(obs_dict, "near_miss")),
                        "human_collision": int(human_collision),
                        "geometry_valid": int(geometry_valid),
                        "physical_safety_controller": str(
                            info.get("physical_safety_controller", "none")
                        ),
                        "physical_safety_active": int(physical_active),
                        "physical_safety_intervened": int(physical_intervened),
                        "physical_safety_intervention_available": int(
                            intervention_available
                        ),
                        "physical_safety_valid_hand_count": int(
                            physical_safety.get("valid_hand_count", 0)
                        ),
                        "physical_safety_tracked_hand_count": int(
                            physical_safety.get("tracked_hand_count", 0)
                        ),
                        "physical_safety_fallback_applied": int(
                            bool(physical_safety.get("fallback_applied", False))
                        ),
                        "physical_safety_failure_reasons_json": json.dumps(
                            list(physical_safety.get("failure_reasons", ())),
                            separators=(",", ":"),
                        ),
                        "intentional_human_absence": int(
                            bool(
                                info.get(
                                    "intentional_human_absence",
                                    physical_safety.get(
                                        "intentional_human_absence", False
                                    ),
                                )
                            )
                        ),
                        "physical_safety_constraint_count": int(
                            physical_safety.get("constraint_count", 0)
                        ),
                        "physical_safety_intervention_norm_radps": intervention_norm,
                        "physical_safety_nominal_velocity_norm_radps": float(
                            physical_safety.get(
                                "nominal_velocity_norm_radps", 0.0
                            )
                        ),
                        "physical_safety_filtered_velocity_norm_radps": float(
                            physical_safety.get(
                                "filtered_velocity_norm_radps", 0.0
                            )
                        ),
                        "physical_safety_constraint_violation_before": float(
                            physical_violation_before
                        ),
                        "physical_safety_constraint_violation_before_mps": float(
                            physical_violation_before
                        ),
                        "physical_safety_constraint_violation_after": float(
                            physical_violation_after
                        ),
                        "physical_safety_constraint_violation_after_mps": float(
                            physical_violation_after
                        ),
                        "physical_safety_slack_radps": physical_slack,
                        "physical_safety_slack_mps": physical_slack,
                        "physical_safety_min_predicted_gap_m": float(
                            physical_safety.get("min_predicted_gap_m", 10.0)
                        ),
                        "physical_safety_feasible": int(physical_feasible),
                        "physical_safety_status": str(
                            physical_safety.get("status", "")
                        ),
                        "physical_safety_solve_time_ms": physical_solve_time_ms,
                        "rmpflow_valid_hand_obstacles": int(
                            info.get("rmpflow_valid_hand_obstacles", 0)
                        ),
                        "left_hand_speed_mps": _vector_norm_or_nan(
                            left_hand_velocity
                        ),
                        "right_hand_speed_mps": _vector_norm_or_nan(
                            right_hand_velocity
                        ),
                        "left_robot_surface_speed_mps": _vector_norm_or_nan(
                            left_robot_velocity
                        ),
                        "right_robot_surface_speed_mps": _vector_norm_or_nan(
                            right_robot_velocity
                        ),
                        "left_relative_speed_mps": _vector_norm_or_nan(
                            left_relative_velocity
                        ),
                        "right_relative_speed_mps": _vector_norm_or_nan(
                            right_relative_velocity
                        ),
                        "left_closing_speed_mps": _obs_scalar(
                            obs_dict, "left_closing_speed_mps", default=0.0
                        ),
                        "right_closing_speed_mps": _obs_scalar(
                            obs_dict, "right_closing_speed_mps", default=0.0
                        ),
                        "left_ttc_s": _obs_scalar(
                            obs_dict, "left_ttc_s", default=10.0
                        ),
                        "right_ttc_s": _obs_scalar(
                            obs_dict, "right_ttc_s", default=10.0
                        ),
                        "left_dynamic_measurement_valid": int(
                            _obs_flag(obs_dict, "left_dynamic_measurement_valid")
                        ),
                        "right_dynamic_measurement_valid": int(
                            _obs_flag(obs_dict, "right_dynamic_measurement_valid")
                        ),
                        "left_ttc_valid": int(
                            _obs_flag(obs_dict, "left_ttc_valid")
                        ),
                        "right_ttc_valid": int(
                            _obs_flag(obs_dict, "right_ttc_valid")
                        ),
                        "closest_human_hand": str(
                            info.get("closest_human_hand", "")
                        ),
                        "closest_robot_link": str(
                            info.get("closest_robot_link", "")
                        ),
                        "closest_collider_prim": str(
                            info.get("closest_collider", "")
                        ),
                        "contact_active": int(info.get("contact_active", False)),
                        "contact_left": int(info.get("contact_left", False)),
                        "contact_right": int(info.get("contact_right", False)),
                        "penetration_depth_m": float(
                            info.get("penetration_depth_m", 0.0)
                        ),
                        "safety_query_time_ms": float(
                            info.get("safety_query_time_ms", 0.0)
                        ),
                        "post_ee_x": float(post_ee_position[0]),
                        "post_ee_y": float(post_ee_position[1]),
                        "post_ee_z": float(post_ee_position[2]),
                        "ee_velocity_x_mps": float(motion_sample.velocity_mps[0]),
                        "ee_velocity_y_mps": float(motion_sample.velocity_mps[1]),
                        "ee_velocity_z_mps": float(motion_sample.velocity_mps[2]),
                        "ee_speed_mps": float(motion_sample.speed_mps),
                        "ee_velocity_valid": int(motion_sample.velocity_valid),
                        "ee_acceleration_x_mps2": float(
                            motion_sample.acceleration_mps2[0]
                        ),
                        "ee_acceleration_y_mps2": float(
                            motion_sample.acceleration_mps2[1]
                        ),
                        "ee_acceleration_z_mps2": float(
                            motion_sample.acceleration_mps2[2]
                        ),
                        "ee_acceleration_norm_mps2": float(
                            motion_sample.acceleration_norm_mps2
                        ),
                        "ee_acceleration_valid": int(
                            motion_sample.acceleration_valid
                        ),
                        "ee_jerk_x_mps3": float(motion_sample.jerk_mps3[0]),
                        "ee_jerk_y_mps3": float(motion_sample.jerk_mps3[1]),
                        "ee_jerk_z_mps3": float(motion_sample.jerk_mps3[2]),
                        "ee_jerk_norm_mps3": float(motion_sample.jerk_norm_mps3),
                        "ee_jerk_valid": int(motion_sample.jerk_valid),
                        "post_left_hand_x": float(post_left_hand_position[0]),
                        "post_left_hand_y": float(post_left_hand_position[1]),
                        "post_left_hand_z": float(post_left_hand_position[2]),
                        "post_right_hand_x": float(post_right_hand_position[0]),
                        "post_right_hand_y": float(post_right_hand_position[1]),
                        "post_right_hand_z": float(post_right_hand_position[2]),
                        "errp_feedback": float(info.get("errp_feedback", 0.0)),
                        "errp_uncertainty": float(info.get("errp_uncertainty", 0.0)),
                        "reward_total": float(reward),
                        "distance_progress_reward": float(
                            reward_components.get("distance_progress", 0.0)
                        ),
                        "near_human_penalty": float(
                            reward_components.get("near_human_penalty", 0.0)
                        ),
                        "human_collision_penalty": float(
                            reward_components.get("human_collision_penalty", 0.0)
                        ),
                        "errp_penalty": float(
                            reward_components.get("errp_penalty", 0.0)
                        ),
                    }
                )
                has_grasped_previous = has_grasped_now
                gripper_closed_previous = gripper_closed_now
                if terminated or truncated:
                    break

            episode_steps = max(1, int(info["step"]))
            episode_success = bool(info.get("success", False))
            task_terminal_reason = str(
                info.get("task_terminal_reason", "")
                or strict_semantics_final_state.get("failure_reason", "")
                or (
                    "success"
                    if episode_success
                    else "terminated_unspecified"
                    if terminated
                    else "max_episode_steps"
                    if truncated
                    else "incomplete"
                )
            )
            row = {
                "episode": episode_idx,
                "seed": episode_seed,
                "active_cube": info["active_cube"],
                "human_replay_episode": human_replay_episode,
                "encounter_id": str(encounter.get("id", "")),
                "encounter_target_severity": str(
                    encounter.get("target_severity", "")
                ),
                "encounter_target_phase": str(
                    encounter.get("task_phase", "")
                ),
                "encounter_target_event": int(
                    encounter.get("controller_event", -1)
                ),
                "encounter_source_session": str(
                    encounter.get("session_id", "")
                ),
                "encounter_active_count": int(encounter_active_count),
                **_source_restoration_row_fields(restoration_diagnostics),
                "scene_layout_id": initial_scene_layout_id,
                "screening_layout_id": initial_scene_layout_id,
                "initial_cube_positions": initial_cube_positions.tolist(),
                "initial_active_cube_position": [
                    float(value) for value in initial_cube_position
                ],
                "place_target_position": [
                    float(value) for value in place_target_position
                ],
                "success": episode_success,
                "terminated": bool(terminated),
                "collision": bool(human_collision_count > 0),
                "truncated": bool(truncated),
                "task_terminal_reason": task_terminal_reason,
                "task_failure_diagnostic_schema": (
                    TASK_FAILURE_DIAGNOSTIC_SCHEMA
                ),
                "strict_task_semantics_enabled": bool(
                    args.strict_task_semantics
                ),
                "steps": int(info["step"]),
                "total_reward": float(total_reward),
                "final_cube_target_dist": float(info["cube_target_dist"]),
                "min_cube_target_dist": float(min_cube_target_dist),
                "final_ee_cube_dist": float(info["ee_cube_dist"]),
                "min_ee_cube_dist": float(min_ee_cube_dist),
                "min_hand_gripper_dist": float(min_hand_gripper_dist),
                "min_hand_gripper_center_dist": float(min_hand_gripper_center_dist),
                "min_hand_gripper_surface_gap": float(min_hand_gripper_dist),
                "grasped_any": bool(grasped_any),
                "final_has_grasped": bool(info["has_grasped_cube"]),
                "released_after_grasp": bool(released_after_grasp),
                "cube_entered_target_tolerance": bool(
                    cube_entered_target_tolerance
                ),
                "maximum_controller_event_reached": int(
                    maximum_controller_event_reached
                ),
                "controller_lift_phase_reached": bool(
                    controller_lift_phase_reached
                ),
                "controller_place_phase_reached": bool(
                    controller_place_phase_reached
                ),
                "controller_release_phase_reached": bool(
                    controller_release_phase_reached
                ),
                "first_grasp_step": int(first_grasp_step),
                "first_grasp_loss_step": int(first_grasp_loss_step),
                "first_target_entry_step": int(first_target_entry_step),
                "first_release_step": int(first_release_step),
                "first_success_step": int(first_success_step),
                "grasp_acquisition_count": int(grasp_acquisition_count),
                "grasp_loss_count": int(grasp_loss_count),
                "release_command_count": int(release_command_count),
                "task_phase_paused_steps": int(task_phase_paused_steps),
                "task_phase_reentry_count": int(task_phase_reentry_count),
                "max_consecutive_physical_safety_intervention_steps": int(
                    max_consecutive_physical_intervention_steps
                ),
                "physical_safety_intervention_steps_before_grasp": int(
                    physical_intervention_steps_before_grasp
                ),
                "physical_safety_intervention_steps_during_transport": int(
                    physical_intervention_steps_during_transport
                ),
                "physical_safety_intervention_steps_during_place": int(
                    physical_intervention_steps_during_place
                ),
                "strict_task_semantics_config": strict_semantics_config,
                "strict_task_semantics_final_state": (
                    strict_semantics_final_state
                ),
                "strict_task_semantics_last_decision": (
                    strict_semantics_last_decision
                ),
                "errp_count": int(errp_count),
                "errp_feedback_sum": float(errp_feedback_sum),
                "mean_errp_feedback": float(
                    errp_feedback_sum / max(1, int(info["step"]))
                ),
                "max_errp_feedback": float(max_errp_feedback),
                "mean_errp_uncertainty": float(
                    errp_uncertainty_sum / max(1, int(info["step"]))
                ),
                "max_errp_uncertainty": float(max_errp_uncertainty),
                "errp_source_code": int(errp_source_code),
                "errp_sources": sorted(errp_source_counts),
                "errp_source_counts": errp_source_counts,
                "safety_gate_sum": float(safety_gate_sum),
                "safety_gate_mean": float(safety_gate_sum / max(1, int(info["step"]))),
                "safety_gate_max": float(safety_gate_max),
                "safety_gate_active_count": int(safety_gate_active_count),
                "safety_gate_full_count": int(safety_gate_full_count),
                "safety_gate_near_overlap_count": int(safety_gate_near_overlap_count),
                "safety_gate_collision_overlap_count": int(
                    safety_gate_collision_overlap_count
                ),
                "near_human_count": int(near_human_count),
                "near_miss_count": int(near_miss_count),
                "human_collision_count": int(human_collision_count),
                "encounter_realized_severity": _realized_severity(
                    collision_count=human_collision_count,
                    near_miss_count=near_miss_count,
                    near_count=near_human_count,
                    gate_active_count=safety_gate_active_count,
                ),
                "final_controller_event": int(info["controller_event"]),
                "final_controller_t": float(info["controller_t"]),
                "phase_hold_steps": int(info["phase_hold_steps"]),
                "bc_blend_count": int(bc_blend_count),
                "safety_residual_count": int(safety_residual_count),
                "safety_residual_norm_sum": float(safety_residual_norm_sum),
                "mean_safety_residual_norm": float(
                    safety_residual_norm_sum / max(1, int(safety_residual_count))
                ),
                "mean_safety_applied_position_m": float(
                    safety_applied_position_m_sum / max(1, int(safety_residual_count))
                ),
                "max_safety_applied_position_m": float(safety_applied_position_m_max),
                "collision_steps": int(collision_steps),
                "near_steps": int(near_steps),
                "near_miss_steps": int(near_miss_steps),
                "geometry_valid_steps": int(geometry_valid_steps),
                "collision_event_count": int(collision_event_count),
                "collision_max_consecutive_steps": int(
                    collision_max_consecutive_steps
                ),
                "collision_rate": float(collision_steps / episode_steps),
                "near_rate": float(near_steps / episode_steps),
                "near_miss_rate": float(near_miss_steps / episode_steps),
                "gate_activation_rate": float(
                    safety_gate_active_count / episode_steps
                ),
                "geometry_valid_rate": float(geometry_valid_steps / episode_steps),
                "min_surface_gap": float(min_surface_gap),
                "minimum_end_effector_surface_gap_m": float(min_surface_gap),
                "minimum_ttc_s": float(minimum_ttc_s),
                "minimum_ttc_valid": bool(minimum_ttc_valid),
                "physics_dt_s": float(env.physics_dt_s),
                "completion_time_s": float(episode_steps * env.physics_dt_s),
                "collision_duration_s": float(collision_steps * env.physics_dt_s),
                "collision_max_consecutive_duration_s": float(
                    collision_max_consecutive_steps * env.physics_dt_s
                ),
                "near_human_duration_s": float(near_steps * env.physics_dt_s),
                "mean_safety_query_time_ms": float(
                    safety_query_time_ms_sum / episode_steps
                ),
                "closest_link_counts": closest_link_counts,
                "collision_link_counts": collision_link_counts,
                "physical_safety_controller": args.physical_safety_controller,
                "physical_safety_active_count": int(
                    physical_safety_active_count
                ),
                "physical_safety_active_rate": float(
                    physical_safety_active_count / episode_steps
                ),
                "physical_safety_feasible_rate": float(
                    physical_safety_feasible_count / episode_steps
                ),
                "physical_safety_intervention_available_count": int(
                    physical_safety_intervention_available_count
                ),
                "physical_safety_intervention_available_rate": float(
                    physical_safety_intervention_available_count / episode_steps
                ),
                "physical_safety_intervention_count": int(
                    physical_safety_intervention_count
                ),
                "physical_safety_intervention_rate": float(
                    physical_safety_intervention_count / episode_steps
                ),
                "mean_physical_safety_intervention_norm_radps": float(
                    physical_safety_intervention_norm_sum / episode_steps
                ),
                "max_physical_safety_intervention_norm_radps": float(
                    physical_safety_intervention_norm_max
                ),
                "mean_physical_safety_slack_radps": float(
                    physical_safety_slack_sum / episode_steps
                ),
                "max_physical_safety_slack_radps": float(
                    physical_safety_slack_max
                ),
                "mean_physical_safety_slack_mps": float(
                    physical_safety_slack_sum / episode_steps
                ),
                "max_physical_safety_slack_mps": float(
                    physical_safety_slack_max
                ),
                "mean_physical_safety_constraint_violation_before_mps": float(
                    physical_safety_violation_before_sum / episode_steps
                ),
                "max_physical_safety_constraint_violation_before_mps": float(
                    physical_safety_violation_before_max
                ),
                "mean_physical_safety_constraint_violation_after_mps": float(
                    physical_safety_violation_after_sum / episode_steps
                ),
                "max_physical_safety_constraint_violation_after_mps": float(
                    physical_safety_violation_after_max
                ),
                "mean_physical_safety_solve_time_ms": float(
                    physical_safety_solve_time_ms_sum / episode_steps
                ),
                **motion_tracker.summary(),
                "rms_gate_ee_acceleration_mps2": _rms_or_zero(
                    gate_ee_acceleration_norms
                ),
                "p95_gate_ee_acceleration_mps2": _percentile_or_zero(
                    gate_ee_acceleration_norms, 95.0
                ),
                "rms_gate_ee_jerk_mps3": _rms_or_zero(gate_ee_jerk_norms),
                "p95_gate_ee_jerk_mps3": _percentile_or_zero(
                    gate_ee_jerk_norms, 95.0
                ),
                "max_gate_ee_jerk_mps3": _max_or_zero(gate_ee_jerk_norms),
                "gate_ee_jerk_sample_count": len(gate_ee_jerk_norms),
                "reward_components_total": reward_components_total,
            }
            rows.append(row)

            if args.log_every > 0 and (
                episode_idx == 0
                or (episode_idx + 1) % args.log_every == 0
                or episode_idx + 1 == evaluation_episodes
            ):
                print(
                    f"[EvalRollout] episode={episode_idx:04d} success={row['success']} "
                    f"steps={row['steps']} reward={row['total_reward']:.3f} "
                    f"cube_target={row['final_cube_target_dist']:.3f} "
                    f"grasped_any={int(row['grasped_any'])} "
                    f"collision_rate={row['collision_rate']:.3f} "
                    f"gate_rate={row['gate_activation_rate']:.3f}",
                    flush=True,
                )
    finally:
        env.close()
        if human_replay is not None:
            human_replay.close()

    summary = _summarize(rows)
    result = {
        "created_unix": time.time(),
        "duration_sec": time.time() - started_at,
        "config": {
            "checkpoint": _resolve_project_path(args.checkpoint),
            "episodes": evaluation_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "render": args.render,
            "live_vr": args.live_vr,
            "vr_tracking_timeout_sec": args.vr_tracking_timeout_sec,
            "action_scale": args.action_scale,
            "checkpoint_task_reward_version": checkpoint_reward_version,
            "task_reward_version_override": str(args.task_reward_version),
            "effective_task_reward_version": effective_task_reward_version,
            "residual_gate_mode_override": args.residual_gate_mode,
            "mask_human_obs_for_policy": args.mask_human_obs_for_policy,
            "fixed_orientation": args.fixed_orientation,
            "gripper_mode": args.gripper_mode,
            "success_dist": args.success_dist,
            "physics_dt_s": float(env.physics_dt_s),
            "safety_gate_start_dist": args.safety_gate_start_dist,
            "safety_gate_full_dist": args.safety_gate_full_dist,
            "safety_residual_checkpoint": (
                _resolve_project_path(args.safety_residual_checkpoint)
                if args.safety_residual_checkpoint
                else ""
            ),
            "safety_residual_alpha": args.safety_residual_alpha,
            "safety_observation_dim": safety_observation_dim,
            "safety_observation_version": safety_observation_version,
            "safety_observation_fields": list(safety_observation_fields),
            "analytic_avoidance_offset_m": args.analytic_avoidance_offset_m,
            "physical_safety_controller": args.physical_safety_controller,
            "rmpflow_human_safety_margin_m": args.rmpflow_human_safety_margin_m,
            "visualize_physical_safety": args.visualize_physical_safety,
            "cbf_safe_gap_m": args.cbf_safe_gap_m,
            "cbf_activation_gap_m": args.cbf_activation_gap_m,
            "cbf_gamma_per_s": args.cbf_gamma_per_s,
            "cbf_prediction_horizon_s": args.cbf_prediction_horizon_s,
            "cbf_max_prediction_buffer_m": args.cbf_max_prediction_buffer_m,
            "cbf_max_joint_speed_rad_s": args.cbf_max_joint_speed_rad_s,
            "phase_gate_close_dist": args.phase_gate_close_dist,
            "phase_gate_max_hold": args.phase_gate_max_hold,
            "pseudo_errp_enabled": args.pseudo_errp_enabled,
            "pseudo_errp_sources": list(pseudo_errp_sources),
            "human_replay_data": human_replay.path if human_replay is not None else "",
            "human_replay_mode": args.human_replay_mode,
            "human_replay_episode_policy": args.human_replay_episode_policy,
            "encounter_manifest": (
                _resolve_project_path(args.encounter_manifest)
                if args.encounter_manifest
                else ""
            ),
            "encounter_policy": args.encounter_policy,
            "encounter_severity_mix": args.encounter_severity_mix,
            "encounter_anchor_mode": args.encounter_anchor_mode,
            "encounter_timebase": args.encounter_timebase,
            "encounter_playback_speed": args.encounter_playback_speed,
            "allow_legacy_source_configuration": (
                args.allow_legacy_source_configuration
            ),
            "encounter_phase_match": args.encounter_phase_match,
            "encounter_event_match": args.encounter_event_match,
            "visualize_human_replay": args.visualize_human_replay,
            "human_replay_visual_z_offset": args.human_replay_visual_z_offset,
            "early_close_on_grasp_gate": args.early_close_on_grasp_gate,
            "fast_forward_grasp_gate": args.fast_forward_grasp_gate,
            "release_gate_dist": release_gate_dist,
            "release_gate_max_hold": args.release_gate_max_hold,
            "require_release_for_success": args.require_release_for_success,
            "strict_task_semantics": bool(args.strict_task_semantics),
            "strict_task_semantics_config": (
                dict(rows[0].get("strict_task_semantics_config", {}))
                if rows
                else {}
            ),
            "task_failure_diagnostic_schema": (
                TASK_FAILURE_DIAGNOSTIC_SCHEMA
            ),
            "blend_bc_checkpoint": (
                _resolve_project_path(args.blend_bc_checkpoint)
                if args.blend_bc_checkpoint
                else ""
            ),
            "blend_bc_events": sorted(blend_events),
            "blend_bc_alpha": blend_alpha,
            "safety_geometry_source": env.safety_geometry.GEOMETRY_SOURCE,
            "safety_geometry_metadata": env.safety_geometry.metadata(),
        },
        "policy": runner.metadata,
        "safety_policy": safety_runner.metadata if safety_runner is not None else {},
        "blend_policy": blend_runner.metadata if blend_runner is not None else {},
        "summary": summary,
        "episodes": rows,
    }
    output_json = _resolve_output_path(args.output_json)
    output_csv = _resolve_output_path(args.output_csv)
    output_step_csv = _resolve_output_path(args.output_step_csv)
    _write_json(output_json, result)
    _write_csv(output_csv, rows)
    _write_step_csv(output_step_csv, step_rows)
    print(
        f"[EvalRollout] success_rate={summary['success_rate']:.3f} "
        f"successes={summary['successes']}/{summary['episodes']} "
        f"mean_steps={summary['mean_steps']:.1f} "
        f"mean_final_cube_target_dist={summary['mean_final_cube_target_dist']:.4f} "
        f"collision_rate={summary['collision_rate']:.4f} "
        f"gate_activation_rate={summary['gate_activation_rate']:.4f}",
        flush=True,
    )
    print(f"[EvalRollout] saved json={output_json}", flush=True)
    print(f"[EvalRollout] saved csv={output_csv}", flush=True)
    print(f"[EvalRollout] saved step csv={output_step_csv}", flush=True)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
        }
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float32)
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=np.float32)
    final_dist = np.asarray(
        [row["final_cube_target_dist"] for row in rows], dtype=np.float32
    )
    min_dist = np.asarray(
        [row["min_cube_target_dist"] for row in rows], dtype=np.float32
    )
    grasped = np.asarray([row["grasped_any"] for row in rows], dtype=np.float32)
    truncated = np.asarray([row["truncated"] for row in rows], dtype=np.float32)
    errp_counts = np.asarray(
        [row.get("errp_count", 0) for row in rows], dtype=np.float32
    )
    mean_errp_feedback = np.asarray(
        [row.get("mean_errp_feedback", 0.0) for row in rows],
        dtype=np.float32,
    )
    max_errp_feedback = np.asarray(
        [row.get("max_errp_feedback", 0.0) for row in rows],
        dtype=np.float32,
    )
    mean_errp_uncertainty = np.asarray(
        [row.get("mean_errp_uncertainty", 0.0) for row in rows],
        dtype=np.float32,
    )
    max_errp_uncertainty = np.asarray(
        [row.get("max_errp_uncertainty", 0.0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_mean = np.asarray(
        [row.get("safety_gate_mean", 0.0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_max = np.asarray(
        [row.get("safety_gate_max", 0.0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_active_counts = np.asarray(
        [row.get("safety_gate_active_count", 0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_full_counts = np.asarray(
        [row.get("safety_gate_full_count", 0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_near_overlap_counts = np.asarray(
        [row.get("safety_gate_near_overlap_count", 0) for row in rows],
        dtype=np.float32,
    )
    safety_gate_collision_overlap_counts = np.asarray(
        [row.get("safety_gate_collision_overlap_count", 0) for row in rows],
        dtype=np.float32,
    )
    min_hand_gripper_dist = np.asarray(
        [row.get("min_hand_gripper_dist", MISSING_DISTANCE_M) for row in rows],
        dtype=np.float32,
    )
    min_hand_gripper_center_dist = np.asarray(
        [row.get("min_hand_gripper_center_dist", MISSING_DISTANCE_M) for row in rows],
        dtype=np.float32,
    )
    near_human_counts = np.asarray(
        [row.get("near_human_count", 0) for row in rows],
        dtype=np.float32,
    )
    human_collision_counts = np.asarray(
        [row.get("human_collision_count", 0) for row in rows],
        dtype=np.float32,
    )
    near_miss_counts = np.asarray(
        [row.get("near_miss_count", 0) for row in rows],
        dtype=np.float32,
    )
    mean_applied_position_m = np.asarray(
        [row.get("mean_safety_applied_position_m", 0.0) for row in rows],
        dtype=np.float32,
    )
    max_applied_position_m = np.asarray(
        [row.get("max_safety_applied_position_m", 0.0) for row in rows],
        dtype=np.float32,
    )
    total_steps = max(1, int(np.sum(steps)))
    collision_steps = int(sum(row.get("collision_steps", 0) for row in rows))
    near_steps = int(sum(row.get("near_steps", 0) for row in rows))
    near_miss_steps = int(sum(row.get("near_miss_steps", 0) for row in rows))
    geometry_valid_steps = int(
        sum(row.get("geometry_valid_steps", 0) for row in rows)
    )
    gate_active_steps = int(
        sum(row.get("safety_gate_active_count", 0) for row in rows)
    )
    collision_event_count = int(
        sum(row.get("collision_event_count", 0) for row in rows)
    )
    physical_active_steps = int(
        sum(row.get("physical_safety_active_count", 0) for row in rows)
    )
    physical_intervention_steps = int(
        sum(row.get("physical_safety_intervention_count", 0) for row in rows)
    )
    physical_intervention_available_steps = int(
        sum(
            row.get("physical_safety_intervention_available_count", 0)
            for row in rows
        )
    )
    physical_feasible_steps = float(
        sum(
            row.get("physical_safety_feasible_rate", 1.0) * row["steps"]
            for row in rows
        )
    )
    physical_intervention_norm_sum = float(
        sum(
            row.get("mean_physical_safety_intervention_norm_radps", 0.0)
            * row["steps"]
            for row in rows
        )
    )
    physical_solve_time_ms_sum = float(
        sum(
            row.get("mean_physical_safety_solve_time_ms", 0.0) * row["steps"]
            for row in rows
        )
    )
    physical_slack_sum = float(
        sum(
            row.get(
                "mean_physical_safety_slack_mps",
                row.get("mean_physical_safety_slack_radps", 0.0),
            )
            * row["steps"]
            for row in rows
        )
    )
    physical_violation_before_sum = float(
        sum(
            row.get(
                "mean_physical_safety_constraint_violation_before_mps", 0.0
            )
            * row["steps"]
            for row in rows
        )
    )
    physical_violation_after_sum = float(
        sum(
            row.get(
                "mean_physical_safety_constraint_violation_after_mps", 0.0
            )
            * row["steps"]
            for row in rows
        )
    )
    closest_link_counts: dict[str, int] = {}
    collision_link_counts: dict[str, int] = {}
    for row in rows:
        for name, count in row.get("closest_link_counts", {}).items():
            closest_link_counts[name] = closest_link_counts.get(name, 0) + int(count)
        for name, count in row.get("collision_link_counts", {}).items():
            collision_link_counts[name] = collision_link_counts.get(name, 0) + int(
                count
            )
    restoration_mode_counts: dict[str, int] = {}
    source_cube_matches = []
    for row in rows:
        mode = str(row.get("restoration_mode", "not_requested"))
        restoration_mode_counts[mode] = restoration_mode_counts.get(mode, 0) + 1
        source_cube = row.get("source_cube_index")
        screening_cube = row.get("screening_cube_index")
        if source_cube is not None and screening_cube is not None:
            source_cube_matches.append(int(source_cube) == int(screening_cube))
    return {
        "episodes": int(len(rows)),
        "restoration_mode_counts": restoration_mode_counts,
        "source_configuration_unavailable_count": int(
            sum(
                not bool(row.get("source_configuration_available", True))
                for row in rows
            )
        ),
        "source_cube_match_rate": (
            float(np.mean(source_cube_matches)) if source_cube_matches else None
        ),
        "successes": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "released_after_grasp_rate": float(
            np.mean([bool(row.get("released_after_grasp", False)) for row in rows])
        ),
        "cube_entered_target_tolerance_rate": float(
            np.mean(
                [
                    bool(row.get("cube_entered_target_tolerance", False))
                    for row in rows
                ]
            )
        ),
        "controller_lift_phase_reached_rate": float(
            np.mean(
                [
                    bool(row.get("controller_lift_phase_reached", False))
                    for row in rows
                ]
            )
        ),
        "controller_place_phase_reached_rate": float(
            np.mean(
                [
                    bool(row.get("controller_place_phase_reached", False))
                    for row in rows
                ]
            )
        ),
        "controller_release_phase_reached_rate": float(
            np.mean(
                [
                    bool(row.get("controller_release_phase_reached", False))
                    for row in rows
                ]
            )
        ),
        "collision_episode_rate": float(
            np.mean([bool(row.get("collision", False)) for row in rows])
        ),
        "truncated_rate": float(np.mean(truncated)),
        "grasp_rate": float(np.mean(grasped)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_total_reward": float(np.mean(rewards)),
        "std_total_reward": float(np.std(rewards)),
        "mean_final_cube_target_dist": float(np.mean(final_dist)),
        "std_final_cube_target_dist": float(np.std(final_dist)),
        "mean_min_cube_target_dist": float(np.mean(min_dist)),
        "std_min_cube_target_dist": float(np.std(min_dist)),
        "mean_errp_count": float(np.mean(errp_counts)),
        "max_errp_count": int(np.max(errp_counts)),
        "mean_episode_errp_feedback": float(np.mean(mean_errp_feedback)),
        "max_episode_errp_feedback": float(np.max(max_errp_feedback)),
        "mean_episode_errp_uncertainty": float(np.mean(mean_errp_uncertainty)),
        "max_episode_errp_uncertainty": float(np.max(max_errp_uncertainty)),
        "mean_safety_gate": float(np.mean(safety_gate_mean)),
        "max_safety_gate": float(np.max(safety_gate_max)),
        "mean_safety_gate_active_count": float(np.mean(safety_gate_active_counts)),
        "mean_safety_gate_full_count": float(np.mean(safety_gate_full_counts)),
        "mean_safety_gate_near_overlap_count": float(
            np.mean(safety_gate_near_overlap_counts)
        ),
        "mean_safety_gate_collision_overlap_count": float(
            np.mean(safety_gate_collision_overlap_counts)
        ),
        "mean_min_hand_gripper_dist": float(np.mean(min_hand_gripper_dist)),
        "min_hand_gripper_dist": float(np.min(min_hand_gripper_dist)),
        "mean_min_hand_gripper_surface_gap": float(np.mean(min_hand_gripper_dist)),
        "min_hand_gripper_surface_gap": float(np.min(min_hand_gripper_dist)),
        "mean_min_hand_gripper_center_dist": float(
            np.mean(min_hand_gripper_center_dist)
        ),
        "min_hand_gripper_center_dist": float(np.min(min_hand_gripper_center_dist)),
        "mean_near_human_count": float(np.mean(near_human_counts)),
        "mean_near_miss_count": float(np.mean(near_miss_counts)),
        "mean_human_collision_count": float(np.mean(human_collision_counts)),
        "mean_safety_applied_position_m": float(np.mean(mean_applied_position_m)),
        "max_safety_applied_position_m": float(np.max(max_applied_position_m)),
        "collision_steps": collision_steps,
        "near_steps": near_steps,
        "near_miss_steps": near_miss_steps,
        "geometry_valid_steps": geometry_valid_steps,
        "collision_event_count": collision_event_count,
        "collision_duration_s": float(
            sum(row.get("collision_duration_s", 0.0) for row in rows)
        ),
        "near_human_duration_s": float(
            sum(row.get("near_human_duration_s", 0.0) for row in rows)
        ),
        "max_collision_consecutive_duration_s": float(
            max(
                row.get("collision_max_consecutive_duration_s", 0.0)
                for row in rows
            )
        ),
        "gate_active_steps": gate_active_steps,
        "collision_rate": float(collision_steps / total_steps),
        "near_rate": float(near_steps / total_steps),
        "near_miss_rate": float(near_miss_steps / total_steps),
        "gate_activation_rate": float(gate_active_steps / total_steps),
        "geometry_valid_rate": float(geometry_valid_steps / total_steps),
        "physical_safety_active_steps": physical_active_steps,
        "physical_safety_active_rate": float(physical_active_steps / total_steps),
        "physical_safety_intervention_available_steps": (
            physical_intervention_available_steps
        ),
        "physical_safety_intervention_available_rate": float(
            physical_intervention_available_steps / total_steps
        ),
        "physical_safety_intervention_steps": physical_intervention_steps,
        "physical_safety_intervention_rate": float(
            physical_intervention_steps / total_steps
        ),
        "physical_safety_feasible_rate": float(
            physical_feasible_steps / total_steps
        ),
        "mean_physical_safety_intervention_norm_radps": float(
            physical_intervention_norm_sum / total_steps
        ),
        "max_physical_safety_intervention_norm_radps": float(
            max(
                row.get("max_physical_safety_intervention_norm_radps", 0.0)
                for row in rows
            )
        ),
        "mean_physical_safety_slack_radps": float(
            physical_slack_sum / total_steps
        ),
        "max_physical_safety_slack_radps": float(
            max(row.get("max_physical_safety_slack_radps", 0.0) for row in rows)
        ),
        "mean_physical_safety_slack_mps": float(
            physical_slack_sum / total_steps
        ),
        "max_physical_safety_slack_mps": float(
            max(
                row.get(
                    "max_physical_safety_slack_mps",
                    row.get("max_physical_safety_slack_radps", 0.0),
                )
                for row in rows
            )
        ),
        "mean_physical_safety_constraint_violation_before_mps": float(
            physical_violation_before_sum / total_steps
        ),
        "max_physical_safety_constraint_violation_before_mps": float(
            max(
                row.get(
                    "max_physical_safety_constraint_violation_before_mps", 0.0
                )
                for row in rows
            )
        ),
        "mean_physical_safety_constraint_violation_after_mps": float(
            physical_violation_after_sum / total_steps
        ),
        "max_physical_safety_constraint_violation_after_mps": float(
            max(
                row.get(
                    "max_physical_safety_constraint_violation_after_mps", 0.0
                )
                for row in rows
            )
        ),
        "mean_physical_safety_solve_time_ms": float(
            physical_solve_time_ms_sum / total_steps
        ),
        "mean_ee_path_length_m": float(
            np.mean([row.get("ee_path_length_m", 0.0) for row in rows])
        ),
        "mean_ee_acceleration_mps2": float(
            np.mean([row.get("mean_ee_acceleration_mps2", 0.0) for row in rows])
        ),
        "mean_rms_ee_acceleration_mps2": float(
            np.mean([row.get("rms_ee_acceleration_mps2", 0.0) for row in rows])
        ),
        "mean_ee_jerk_mps3": float(
            np.mean([row.get("mean_ee_jerk_mps3", 0.0) for row in rows])
        ),
        "mean_rms_ee_jerk_mps3": float(
            np.mean([row.get("rms_ee_jerk_mps3", 0.0) for row in rows])
        ),
        "mean_p95_ee_jerk_mps3": float(
            np.mean([row.get("p95_ee_jerk_mps3", 0.0) for row in rows])
        ),
        "max_ee_jerk_mps3": float(
            max(row.get("max_ee_jerk_mps3", 0.0) for row in rows)
        ),
        "mean_integrated_squared_ee_jerk_m2ps5": float(
            np.mean(
                [
                    row.get("integrated_squared_ee_jerk_m2ps5", 0.0)
                    for row in rows
                ]
            )
        ),
        "mean_rms_gate_ee_acceleration_mps2": float(
            np.mean([row.get("rms_gate_ee_acceleration_mps2", 0.0) for row in rows])
        ),
        "mean_p95_gate_ee_jerk_mps3": float(
            np.mean([row.get("p95_gate_ee_jerk_mps3", 0.0) for row in rows])
        ),
        "mean_rms_gate_ee_jerk_mps3": float(
            np.mean([row.get("rms_gate_ee_jerk_mps3", 0.0) for row in rows])
        ),
        "min_surface_gap": float(
            min(row.get("min_surface_gap", MISSING_DISTANCE_M) for row in rows)
        ),
        "minimum_end_effector_surface_gap_m": float(
            min(row.get("min_surface_gap", MISSING_DISTANCE_M) for row in rows)
        ),
        "minimum_ttc_valid_episode_rate": float(
            np.mean([bool(row.get("minimum_ttc_valid", False)) for row in rows])
        ),
        "mean_minimum_ttc_s_valid_only": float(
            np.mean(
                [
                    float(row["minimum_ttc_s"])
                    for row in rows
                    if bool(row.get("minimum_ttc_valid", False))
                ]
            )
            if any(bool(row.get("minimum_ttc_valid", False)) for row in rows)
            else 0.0
        ),
        "mean_safety_query_time_ms": float(
            sum(
                row.get("mean_safety_query_time_ms", 0.0) * row["steps"]
                for row in rows
            )
            / total_steps
        ),
        "closest_link_counts": closest_link_counts,
        "collision_link_counts": collision_link_counts,
    }


def _parse_event_set(text: str) -> set[int]:
    events: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        events.add(int(part))
    return events


def _mask_human_obs_for_policy(obs: np.ndarray) -> np.ndarray:
    obs_policy = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
    slices = observation_slices()
    zero_fields = (
        "human_head_pos",
        "human_left_hand_pos",
        "human_right_hand_pos",
        "ee_to_left_hand",
        "ee_to_right_hand",
        "human_robot_collision",
        "near_human",
    )
    for field_name in zero_fields:
        field_slice = slices.get(field_name)
        if field_slice is not None and field_slice.stop <= obs_policy.size:
            obs_policy[field_slice] = 0.0
    dist_slice = slices.get("min_hand_gripper_dist")
    if dist_slice is not None and dist_slice.stop <= obs_policy.size:
        obs_policy[dist_slice] = float(MISSING_DISTANCE_M)
    return obs_policy


def _safety_distance_gate(
    obs: dict[str, np.ndarray],
    *,
    start_dist: float,
    full_dist: float,
) -> float:
    dist = _obs_scalar(obs, "min_hand_gripper_dist", default=MISSING_DISTANCE_M)
    start_dist = float(start_dist)
    full_dist = float(full_dist)
    denom = start_dist - full_dist
    if denom <= 1e-6:
        return 0.0
    return float(np.clip((start_dist - dist) / denom, 0.0, 1.0))


def _realized_severity(
    *,
    collision_count: int,
    near_miss_count: int,
    near_count: int,
    gate_active_count: int,
) -> str:
    if collision_count > 0:
        return "collision"
    if near_miss_count > 0:
        return "near_miss"
    if near_count > 0:
        return "near"
    if gate_active_count > 0:
        return "gate_only"
    return "safe"


def _analytic_avoidance_residual(
    obs: dict[str, np.ndarray],
    *,
    full_gate_offset_m: float,
    residual_alpha: float,
    action_scale: float,
) -> np.ndarray:
    ee_pos = _obs_vector3(obs, "ee_pos")
    if ee_pos is None:
        return np.zeros(ACTION_DIM, dtype=np.float32)

    nearest_hand = None
    nearest_dist = float("inf")
    for field_name in ("human_left_hand_pos", "human_right_hand_pos"):
        hand_pos = _obs_vector3(obs, field_name)
        if hand_pos is None or float(np.linalg.norm(hand_pos)) <= 1e-6:
            continue
        distance = float(np.linalg.norm(ee_pos - hand_pos))
        if distance < nearest_dist:
            nearest_dist = distance
            nearest_hand = hand_pos
    if nearest_hand is None or nearest_dist <= 1e-6:
        return np.zeros(ACTION_DIM, dtype=np.float32)

    denominator = (
        max(float(residual_alpha), 1e-6)
        * float(CONTROLLER_TARGET_MAX_DELTA_M)
        * max(float(action_scale), 1e-6)
    )
    direction = (ee_pos - nearest_hand) / nearest_dist
    normalized_magnitude = float(full_gate_offset_m) / denominator
    residual = np.zeros(ACTION_DIM, dtype=np.float32)
    residual[:3] = direction.astype(np.float32) * normalized_magnitude
    return clip_action(residual)


def _obs_vector3(obs: Any, field_name: str) -> np.ndarray | None:
    if not isinstance(obs, dict) or field_name not in obs:
        return None
    value = np.asarray(obs[field_name], dtype=float).reshape(-1)
    if value.size < 3 or not np.all(np.isfinite(value[:3])):
        return None
    return value[:3].copy()


def _vector_norm_or_nan(value) -> float:
    return float(np.linalg.norm(value)) if value is not None else float("nan")


def _rms_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


def _percentile_or_zero(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _obs_scalar(obs: Any, field_name: str, *, default: float = 0.0) -> float:
    if not isinstance(obs, dict) or field_name not in obs:
        return float(default)
    value = np.asarray(obs[field_name], dtype=float).reshape(-1)
    if value.size == 0 or not np.isfinite(value[0]):
        return float(default)
    return float(value[0])


def _obs_flag(obs: Any, field_name: str) -> bool:
    return _obs_scalar(obs, field_name, default=0.0) > 0.5


def _minimum_ttc(obs: Any) -> tuple[float, bool]:
    values: list[float] = []
    for hand in ("left", "right"):
        if not _obs_flag(obs, f"{hand}_ttc_valid"):
            continue
        value = _obs_scalar(obs, f"{hand}_ttc_s", default=10.0)
        if np.isfinite(value) and value >= 0.0:
            values.append(float(value))
    if not values:
        return 10.0, False
    return float(min(values)), True


def _source_restoration_row_fields(restoration: dict[str, Any]) -> dict[str, Any]:
    missing = restoration.get("missing_fields", [])
    if not isinstance(missing, list):
        missing = list(missing) if missing else []
    return {
        "source_configuration_available": bool(
            restoration.get("source_configuration_available", False)
        ),
        "restoration_mode": str(restoration.get("restoration_mode", "unavailable")),
        "restoration_reason": str(restoration.get("restoration_reason", "")),
        "source_cube_index": restoration.get("source_cube_index"),
        "screening_cube_index": restoration.get("screening_cube_index"),
        "source_cube_name": restoration.get("source_cube_name"),
        "screening_cube_name": restoration.get("screening_cube_name"),
        "collection_seed": restoration.get("collection_seed"),
        "source_layout_seed": restoration.get("layout_seed"),
        "screening_seed": restoration.get("screening_seed"),
        "source_layout_id": restoration.get("source_layout_id"),
        "cube_pose_restored": bool(restoration.get("cube_pose_restored", False)),
        "target_pose_restored": bool(restoration.get("target_pose_restored", False)),
        "robot_initial_state_restored": bool(
            restoration.get("robot_initial_state_restored", False)
        ),
        "pose_mismatch": bool(restoration.get("pose_mismatch", False)),
        "pose_mismatch_reason": str(
            restoration.get("pose_mismatch_reason", "")
        ),
        "source_configuration_missing_fields": missing,
        "max_cube_position_error_m": restoration.get(
            "max_cube_position_error_m"
        ),
        "max_cube_orientation_error_rad": restoration.get(
            "max_cube_orientation_error_rad"
        ),
        "target_position_error_m": restoration.get("target_position_error_m"),
        "target_orientation_error_rad": restoration.get(
            "target_orientation_error_rad"
        ),
    }


def _unavailable_source_episode_row(
    *,
    episode_idx: int,
    episode_seed: int,
    human_replay_episode: str,
    encounter: dict[str, Any],
    restoration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode": int(episode_idx),
        "seed": int(episode_seed),
        "active_cube": "",
        "human_replay_episode": human_replay_episode,
        "encounter_id": str(encounter.get("id", "")),
        "encounter_target_severity": str(encounter.get("target_severity", "")),
        "encounter_target_phase": str(encounter.get("task_phase", "")),
        "encounter_target_event": int(encounter.get("controller_event", -1)),
        "encounter_source_session": str(encounter.get("session_id", "")),
        "encounter_active_count": 0,
        "encounter_realized_severity": "unknown",
        **_source_restoration_row_fields(restoration),
        "scene_layout_id": "",
        "screening_layout_id": "",
        "initial_active_cube_position": [],
        "place_target_position": [],
        "success": False,
        "terminated": False,
        "truncated": True,
        "steps": 0,
        "total_reward": 0.0,
        "final_cube_target_dist": 10.0,
        "min_cube_target_dist": 10.0,
        "grasped_any": False,
        "safety_gate_active_count": 0,
        "geometry_valid_steps": 0,
        "collision_steps": 0,
        "near_steps": 0,
        "near_miss_steps": 0,
        "collision_rate": 0.0,
        "near_rate": 0.0,
        "near_miss_rate": 0.0,
        "gate_activation_rate": 0.0,
        "geometry_valid_rate": 0.0,
        "min_surface_gap": 10.0,
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "episode",
        "seed",
        "active_cube",
        "human_replay_episode",
        "encounter_id",
        "encounter_target_severity",
        "encounter_target_phase",
        "encounter_target_event",
        "encounter_source_session",
        "encounter_active_count",
        "encounter_realized_severity",
        "source_configuration_available",
        "restoration_mode",
        "restoration_reason",
        "source_cube_index",
        "screening_cube_index",
        "source_cube_name",
        "screening_cube_name",
        "collection_seed",
        "source_layout_seed",
        "screening_seed",
        "source_layout_id",
        "screening_layout_id",
        "cube_pose_restored",
        "target_pose_restored",
        "robot_initial_state_restored",
        "pose_mismatch",
        "pose_mismatch_reason",
        "source_configuration_missing_fields",
        "max_cube_position_error_m",
        "max_cube_orientation_error_rad",
        "target_position_error_m",
        "target_orientation_error_rad",
        "initial_active_cube_position",
        "place_target_position",
        "success",
        "terminated",
        "collision",
        "truncated",
        "task_terminal_reason",
        "task_failure_diagnostic_schema",
        "strict_task_semantics_enabled",
        "steps",
        "total_reward",
        "final_cube_target_dist",
        "min_cube_target_dist",
        "final_ee_cube_dist",
        "min_ee_cube_dist",
        "min_hand_gripper_dist",
        "min_hand_gripper_center_dist",
        "min_hand_gripper_surface_gap",
        "grasped_any",
        "final_has_grasped",
        "released_after_grasp",
        "cube_entered_target_tolerance",
        "maximum_controller_event_reached",
        "controller_lift_phase_reached",
        "controller_place_phase_reached",
        "controller_release_phase_reached",
        "first_grasp_step",
        "first_grasp_loss_step",
        "first_target_entry_step",
        "first_release_step",
        "first_success_step",
        "grasp_acquisition_count",
        "grasp_loss_count",
        "release_command_count",
        "task_phase_paused_steps",
        "task_phase_reentry_count",
        "max_consecutive_physical_safety_intervention_steps",
        "physical_safety_intervention_steps_before_grasp",
        "physical_safety_intervention_steps_during_transport",
        "physical_safety_intervention_steps_during_place",
        "strict_task_semantics_config",
        "strict_task_semantics_final_state",
        "strict_task_semantics_last_decision",
        "errp_count",
        "errp_feedback_sum",
        "mean_errp_feedback",
        "max_errp_feedback",
        "mean_errp_uncertainty",
        "max_errp_uncertainty",
        "errp_source_code",
        "errp_sources",
        "safety_gate_sum",
        "safety_gate_mean",
        "safety_gate_max",
        "safety_gate_active_count",
        "safety_gate_full_count",
        "safety_gate_near_overlap_count",
        "safety_gate_collision_overlap_count",
        "near_human_count",
        "near_miss_count",
        "human_collision_count",
        "final_controller_event",
        "final_controller_t",
        "phase_hold_steps",
        "bc_blend_count",
        "safety_residual_count",
        "safety_residual_norm_sum",
        "mean_safety_residual_norm",
        "mean_safety_applied_position_m",
        "max_safety_applied_position_m",
        "collision_steps",
        "near_steps",
        "near_miss_steps",
        "geometry_valid_steps",
        "collision_event_count",
        "collision_max_consecutive_steps",
        "collision_rate",
        "near_rate",
        "near_miss_rate",
        "gate_activation_rate",
        "geometry_valid_rate",
        "min_surface_gap",
        "minimum_end_effector_surface_gap_m",
        "minimum_ttc_s",
        "minimum_ttc_valid",
        "physics_dt_s",
        "completion_time_s",
        "collision_duration_s",
        "collision_max_consecutive_duration_s",
        "near_human_duration_s",
        "mean_safety_query_time_ms",
        "closest_link_counts",
        "collision_link_counts",
        "physical_safety_controller",
        "physical_safety_active_count",
        "physical_safety_active_rate",
        "physical_safety_feasible_rate",
        "physical_safety_intervention_available_count",
        "physical_safety_intervention_available_rate",
        "physical_safety_intervention_count",
        "physical_safety_intervention_rate",
        "mean_physical_safety_intervention_norm_radps",
        "max_physical_safety_intervention_norm_radps",
        "mean_physical_safety_slack_radps",
        "max_physical_safety_slack_radps",
        "mean_physical_safety_slack_mps",
        "max_physical_safety_slack_mps",
        "mean_physical_safety_constraint_violation_before_mps",
        "max_physical_safety_constraint_violation_before_mps",
        "mean_physical_safety_constraint_violation_after_mps",
        "max_physical_safety_constraint_violation_after_mps",
        "mean_physical_safety_solve_time_ms",
        "ee_path_length_m",
        "ee_motion_duration_s",
        "mean_ee_speed_mps",
        "max_ee_speed_mps",
        "mean_ee_acceleration_mps2",
        "rms_ee_acceleration_mps2",
        "max_ee_acceleration_mps2",
        "mean_ee_jerk_mps3",
        "rms_ee_jerk_mps3",
        "p95_ee_jerk_mps3",
        "max_ee_jerk_mps3",
        "integrated_squared_ee_jerk_m2ps5",
        "ee_velocity_sample_count",
        "ee_acceleration_sample_count",
        "ee_jerk_sample_count",
        "rms_gate_ee_acceleration_mps2",
        "p95_gate_ee_acceleration_mps2",
        "rms_gate_ee_jerk_mps3",
        "p95_gate_ee_jerk_mps3",
        "max_gate_ee_jerk_mps3",
        "gate_ee_jerk_sample_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field, "") for field in fields}
            csv_row["errp_sources"] = ",".join(row.get("errp_sources", []))
            csv_row["initial_active_cube_position"] = json.dumps(
                row.get("initial_active_cube_position", []), separators=(",", ":")
            )
            csv_row["place_target_position"] = json.dumps(
                row.get("place_target_position", []), separators=(",", ":")
            )
            csv_row["source_configuration_missing_fields"] = json.dumps(
                row.get("source_configuration_missing_fields", []),
                separators=(",", ":"),
            )
            csv_row["closest_link_counts"] = json.dumps(
                row.get("closest_link_counts", {}), sort_keys=True
            )
            csv_row["collision_link_counts"] = json.dumps(
                row.get("collision_link_counts", {}), sort_keys=True
            )
            for diagnostic_field in (
                "strict_task_semantics_config",
                "strict_task_semantics_final_state",
                "strict_task_semantics_last_decision",
            ):
                csv_row[diagnostic_field] = json.dumps(
                    row.get(diagnostic_field, {}), sort_keys=True
                )
            writer.writerow(csv_row)


def _write_step_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "episode",
        "seed",
        "step",
        "sim_time",
        "controller_event_before",
        "controller_event_after",
        "controller_t_before",
        "controller_t_after",
        "task_phase_before",
        "task_phase_after",
        "phase_hold_steps",
        "task_phase_advanced",
        "task_phase_paused",
        "task_phase_reentry",
        "task_phase_control_reason",
        "task_action_x",
        "task_action_y",
        "task_action_z",
        "task_action_yaw",
        "task_action_gripper",
        "gripper_command",
        "gripper_closed",
        "gripper_width_m",
        "has_grasped_cube",
        "grasp_acquired",
        "grasp_lost",
        "release_commanded",
        "ee_cube_dist_m",
        "cube_target_dist_m",
        "cube_target_xy_error_m",
        "cube_target_z_error_m",
        "post_cube_x",
        "post_cube_y",
        "post_cube_z",
        "cube_speed_mps",
        "cube_speed_valid",
        "step_success",
        "step_truncated",
        "task_terminal_reason",
        "strict_task_semantics_enabled",
        "strict_success_latched",
        "strict_failure_reason",
        "strict_transition_reason",
        "strict_transition_held",
        "strict_retry_started",
        "strict_transition_reentry",
        "strict_grasp_retry_count",
        "strict_phase_reentry_count",
        "strict_phase_hold_total_steps",
        "strict_place_ready_latched",
        "strict_place_ready_streak",
        "strict_release_settle_streak",
        "strict_cbf_pause_steps",
        "strict_task_semantics_state_json",
        "strict_task_semantics_last_decision_json",
        "encounter_id",
        "encounter_target_severity",
        "encounter_active",
        "encounter_source_step",
        "encounter_source_time_s",
        "pre_surface_gap_m",
        "post_surface_gap_m",
        "surface_gap_delta_m",
        "pre_center_dist_m",
        "gate",
        "gate_active",
        "raw_residual_x",
        "raw_residual_y",
        "raw_residual_z",
        "raw_residual_norm",
        "applied_residual_x",
        "applied_residual_y",
        "applied_residual_z",
        "applied_residual_norm",
        "applied_position_m",
        "near_human",
        "near_miss",
        "human_collision",
        "geometry_valid",
        "left_hand_speed_mps",
        "right_hand_speed_mps",
        "left_robot_surface_speed_mps",
        "right_robot_surface_speed_mps",
        "left_relative_speed_mps",
        "right_relative_speed_mps",
        "left_closing_speed_mps",
        "right_closing_speed_mps",
        "left_ttc_s",
        "right_ttc_s",
        "left_dynamic_measurement_valid",
        "right_dynamic_measurement_valid",
        "left_ttc_valid",
        "right_ttc_valid",
        "closest_human_hand",
        "closest_robot_link",
        "closest_collider_prim",
        "contact_active",
        "contact_left",
        "contact_right",
        "penetration_depth_m",
        "safety_query_time_ms",
        "post_ee_x",
        "post_ee_y",
        "post_ee_z",
        "ee_velocity_x_mps",
        "ee_velocity_y_mps",
        "ee_velocity_z_mps",
        "ee_speed_mps",
        "ee_velocity_valid",
        "ee_acceleration_x_mps2",
        "ee_acceleration_y_mps2",
        "ee_acceleration_z_mps2",
        "ee_acceleration_norm_mps2",
        "ee_acceleration_valid",
        "ee_jerk_x_mps3",
        "ee_jerk_y_mps3",
        "ee_jerk_z_mps3",
        "ee_jerk_norm_mps3",
        "ee_jerk_valid",
        "post_left_hand_x",
        "post_left_hand_y",
        "post_left_hand_z",
        "post_right_hand_x",
        "post_right_hand_y",
        "post_right_hand_z",
        "errp_feedback",
        "errp_uncertainty",
        "reward_total",
        "distance_progress_reward",
        "near_human_penalty",
        "human_collision_penalty",
        "errp_penalty",
        "physical_safety_controller",
        "physical_safety_active",
        "physical_safety_intervened",
        "physical_safety_intervention_available",
        "physical_safety_valid_hand_count",
        "physical_safety_tracked_hand_count",
        "physical_safety_fallback_applied",
        "physical_safety_failure_reasons_json",
        "intentional_human_absence",
        "physical_safety_constraint_count",
        "physical_safety_intervention_norm_radps",
        "physical_safety_nominal_velocity_norm_radps",
        "physical_safety_filtered_velocity_norm_radps",
        "physical_safety_constraint_violation_before",
        "physical_safety_constraint_violation_before_mps",
        "physical_safety_constraint_violation_after",
        "physical_safety_constraint_violation_after_mps",
        "physical_safety_slack_radps",
        "physical_safety_slack_mps",
        "physical_safety_min_predicted_gap_m",
        "physical_safety_feasible",
        "physical_safety_status",
        "physical_safety_solve_time_ms",
        "rmpflow_valid_hand_obstacles",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


try:
    _run()
except BaseException as exc:
    print(f"[EvalRollout] terminated by {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
