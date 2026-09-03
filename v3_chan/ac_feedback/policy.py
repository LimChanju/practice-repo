"""Strict loader for the frozen direct-BC checkpoint."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import POLICY_SHA256, POLICY_SIZE_BYTES


@dataclass(frozen=True)
class PolicyOutput:
    policy_input: np.ndarray
    action: np.ndarray
    inference_started_monotonic_ns: int
    inference_completed_monotonic_ns: int


class FrozenBCPolicy:
    """Load only the policy metadata frozen by the runtime handoff."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != POLICY_SIZE_BYTES:
            raise RuntimeError(
                f"checkpoint size mismatch: expected {POLICY_SIZE_BYTES}, got {path.stat().st_size}"
            )
        actual_sha = _sha256(path)
        if actual_sha != POLICY_SHA256:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch: expected {POLICY_SHA256}, got {actual_sha}"
            )
        import torch

        from v3_chan.rl.policies import MLPPolicy

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "obs_dim": 84,
            "observation_version": "obs_v1_state_controller_phase",
            "action_dim": 5,
            "action_version": "action_v1_controller_target_delta",
            "hidden_dims": (256, 256),
            "reward_version": "reward_v4_post_release_stability_hri_errp",
        }
        observed = {
            "obs_dim": int(checkpoint.get("obs_dim", -1)),
            "observation_version": str(checkpoint.get("observation_version", "")),
            "action_dim": int(checkpoint.get("action_dim", -1)),
            "action_version": str(checkpoint.get("action_version", "")),
            "hidden_dims": tuple(int(v) for v in checkpoint.get("hidden_dims", ())),
            "reward_version": str(checkpoint.get("reward_version", "")),
        }
        if observed != expected:
            raise RuntimeError(
                f"checkpoint metadata mismatch: expected {expected}, got {observed}"
            )
        self.obs_dim = 84
        self.action_dim = 5
        self.action_version = expected["action_version"]
        self.obs_mean = _numpy(checkpoint["obs_mean"]).reshape(1, self.obs_dim)
        self.obs_std = _numpy(checkpoint["obs_std"]).reshape(1, self.obs_dim)
        if not np.all(np.isfinite(self.obs_mean)) or not np.all(
            np.isfinite(self.obs_std)
        ):
            raise RuntimeError("checkpoint normalization contains non-finite values")
        self.model = MLPPolicy(
            self.obs_dim,
            self.action_dim,
            hidden_dims=expected["hidden_dims"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.checkpoint_path = path
        self.metadata: dict[str, Any] = {
            **observed,
            "checkpoint_path": str(path),
            "checkpoint_sha256": actual_sha,
            "checkpoint_size_bytes": path.stat().st_size,
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "mask_human_observations": True,
            "policy_mode": "direct_bc",
        }

    def predict(self, observation: np.ndarray) -> PolicyOutput:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(
                f"BC observation shape must be ({self.obs_dim},), got {obs.shape}"
            )
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("BC observation contains non-finite values")
        policy_input = mask_human_observations(obs)
        normalized = (
            policy_input.reshape(1, -1) - self.obs_mean
        ) / np.maximum(self.obs_std, 1e-6)
        started_ns = time.monotonic_ns()
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(normalized.astype(np.float32)).to(
                self.device
            )
            action = self.model(tensor).detach().cpu().numpy()[0]
        completed_ns = time.monotonic_ns()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise RuntimeError("BC produced an invalid action")
        if np.any(action < -1.000001) or np.any(action > 1.000001):
            raise RuntimeError("BC action is outside [-1, 1]")
        return PolicyOutput(
            policy_input=policy_input,
            action=np.clip(action, -1.0, 1.0),
            inference_started_monotonic_ns=started_ns,
            inference_completed_monotonic_ns=completed_ns,
        )


def mask_human_observations(observation: np.ndarray) -> np.ndarray:
    """Apply the exact mask used by the frozen handoff evaluator."""

    from v3_chan.rl.observations import MISSING_DISTANCE_M, observation_slices

    result = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    if result.shape != (84,):
        raise ValueError(f"expected an 84-D observation, got {result.shape}")
    slices = observation_slices()
    for field in (
        "human_head_pos",
        "human_left_hand_pos",
        "human_right_hand_pos",
        "ee_to_left_hand",
        "ee_to_right_hand",
        "human_robot_collision",
        "near_human",
    ):
        result[slices[field]] = 0.0
    result[slices["min_hand_gripper_dist"]] = float(MISSING_DISTANCE_M)
    return result


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
