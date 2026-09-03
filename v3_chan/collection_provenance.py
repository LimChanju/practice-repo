"""Source-version metadata and production collection validation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


_SOURCE_SUFFIXES = {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}
_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "eval_results",
    "logs",
    "policies",
    "trajectories",
    "videos",
    "gripper_camera_recording",
    "legacy_pre_distal_collider",
    "ac_selective_smoothing_feedback_data",
}
_INVALID_VERSION_VALUES = {"", "unknown", "unspecified", "none", "null"}


def source_tree_sha256(project_root: str | os.PathLike[str]) -> str:
    """Hash runnable source files without including datasets or outputs."""

    root = Path(project_root).resolve()
    candidates: list[Path] = []
    launch_script = root / "launch_isaac.sh"
    if launch_script.is_file():
        candidates.append(launch_script)
    source_root = root / "v3_chan"
    if source_root.is_dir():
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if any(part in _EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No source files found below {root}")

    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def resolve_code_version(
    project_root: str | os.PathLike[str], explicit_version: str | None = None
) -> tuple[str, str, str]:
    """Return code version, provenance source, and source-tree SHA-256."""

    tree_hash = source_tree_sha256(project_root)
    explicit = str(explicit_version or "").strip()
    if explicit.lower() not in _INVALID_VERSION_VALUES:
        return explicit, "HRI_CODE_VERSION", tree_hash
    return f"source-sha256:{tree_hash}", "source_tree_sha256", tree_hash


def validate_production_metadata(
    *, production_mode: bool, participant_id: str, code_version: str
) -> None:
    if not production_mode:
        return
    participant = str(participant_id or "").strip().lower()
    version = str(code_version or "").strip().lower()
    if participant in _INVALID_VERSION_VALUES:
        raise RuntimeError(
            "Production collection requires a non-placeholder HRI_PARTICIPANT_ID"
        )
    if version in _INVALID_VERSION_VALUES:
        raise RuntimeError(
            "Production collection requires HRI_CODE_VERSION or a source-tree hash"
        )
