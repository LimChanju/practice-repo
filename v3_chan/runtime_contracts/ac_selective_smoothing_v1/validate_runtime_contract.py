from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).with_name("runtime_config.json")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime_config.json must contain an object")
    return payload


def validate_runtime_contract(
    project_root: Path = PROJECT_ROOT,
    *,
    check_checkpoint_metadata: bool = False,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    contract = _contract()
    failures: list[str] = []
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in contract["files"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        actual_sha = _sha256(path)
        observed[relative] = {
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
        }
        if actual_sha != expected_sha:
            failures.append(
                f"SHA mismatch: {relative}: expected {expected_sha}, got {actual_sha}"
            )

    policy = contract["policy"]
    checkpoint = root / policy["path"]
    if checkpoint.is_file() and checkpoint.stat().st_size != policy["size_bytes"]:
        failures.append("checkpoint size mismatch")

    responses = contract["responses"]
    expected_responses = {
        "A_reactive": ("joint_nominal", 0.0),
        "C_smooth": ("smooth_intervention", 4.0),
    }
    for condition, (mode, weight) in expected_responses.items():
        actual = responses.get(condition, {})
        if actual.get("objective_mode") != mode:
            failures.append(f"{condition} objective mode mismatch")
        if float(actual.get("correction_smoothness_weight", -1.0)) != weight:
            failures.append(f"{condition} smoothing weight mismatch")
    if responses.get("allowed_collection_conditions") != [
        "A_reactive",
        "C_smooth",
    ]:
        failures.append("collector condition allow-list mismatch")
    if responses.get("online_adaptation_during_collection") is not False:
        failures.append("online adaptation must be disabled during collection")
    if contract["task_runtime"].get("haptics_enabled") is not False:
        failures.append("haptics must be disabled")

    checkpoint_metadata: dict[str, Any] | None = None
    if check_checkpoint_metadata and checkpoint.is_file():
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_metadata = {
                "observation_dim": int(payload.get("obs_dim", -1)),
                "observation_version": str(payload.get("observation_version", "")),
                "action_dim": int(payload.get("action_dim", -1)),
                "action_version": str(payload.get("action_version", "")),
                "hidden_dims": list(payload.get("hidden_dims", ())),
                "reward_version": str(payload.get("reward_version", "")),
            }
            expected = {
                "observation_dim": policy["observation_dim"],
                "observation_version": policy["observation_version"],
                "action_dim": policy["action_dim"],
                "action_version": policy["action_version"],
                "hidden_dims": policy["hidden_dims"],
                "reward_version": policy["reward_version"],
            }
            if checkpoint_metadata != expected:
                failures.append(
                    f"checkpoint metadata mismatch: expected {expected}, "
                    f"got {checkpoint_metadata}"
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(f"checkpoint metadata inspection failed: {error}")

    return {
        "schema_version": "ac_selective_smoothing_runtime_validation_v1",
        "valid": not failures,
        "project_root": str(root),
        "checked_file_count": len(observed),
        "checkpoint_metadata": checkpoint_metadata,
        "runtime_handoff_ready": bool(
            contract["collection_readiness"]["runtime_handoff_ready"]
        ),
        "production_collection_ready": bool(
            contract["collection_readiness"]["production_collection_ready"]
        ),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen A/C selective-smoothing runtime handoff."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--check-checkpoint-metadata", action="store_true")
    args = parser.parse_args()
    report = validate_runtime_contract(
        args.project_root,
        check_checkpoint_metadata=args.check_checkpoint_metadata,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
