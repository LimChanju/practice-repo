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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime_config.json must contain a JSON object")
    return payload


def validate_runtime_contract(
    project_root: Path = PROJECT_ROOT,
    *,
    check_checkpoint_metadata: bool = False,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    contract = _load_contract()
    failures: list[str] = []
    observed: dict[str, dict[str, Any]] = {}

    for relative, expected_sha in contract.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        actual_sha = _sha256(path)
        observed[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha,
        }
        if actual_sha != expected_sha:
            failures.append(
                f"SHA mismatch: {relative}: expected {expected_sha}, got {actual_sha}"
            )

    policy = contract["policy"]
    policy_path = root / policy["path"]
    if policy_path.is_file() and policy_path.stat().st_size != policy["size_bytes"]:
        failures.append(
            f"checkpoint size mismatch: expected {policy['size_bytes']}, "
            f"got {policy_path.stat().st_size}"
        )

    checkpoint_metadata: dict[str, Any] | None = None
    if check_checkpoint_metadata and policy_path.is_file():
        try:
            import torch

            checkpoint = torch.load(
                policy_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_metadata = {
                "observation_dim": int(checkpoint.get("obs_dim", -1)),
                "observation_version": str(
                    checkpoint.get("observation_version", "")
                ),
                "action_dim": int(checkpoint.get("action_dim", -1)),
                "action_version": str(checkpoint.get("action_version", "")),
                "hidden_dims": list(checkpoint.get("hidden_dims", ())),
                "reward_version": str(checkpoint.get("reward_version", "")),
            }
            expected_metadata = {
                "observation_dim": policy["observation_dim"],
                "observation_version": policy["observation_version"],
                "action_dim": policy["action_dim"],
                "action_version": policy["action_version"],
                "hidden_dims": policy["hidden_dims"],
                "reward_version": contract["evaluation_reward"][
                    "checkpoint_metadata_version"
                ],
            }
            if checkpoint_metadata != expected_metadata:
                failures.append(
                    "checkpoint metadata mismatch: "
                    f"expected {expected_metadata}, got {checkpoint_metadata}"
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(f"checkpoint metadata inspection failed: {error}")

    return {
        "schema_version": "bc_cbf_runtime_contract_validation_v1",
        "valid": not failures,
        "project_root": str(root),
        "contract": str(CONTRACT_PATH),
        "checked_file_count": len(observed),
        "observed": observed,
        "checkpoint_metadata": checkpoint_metadata,
        "failures": failures,
        "production_collection_ready": bool(
            contract["collection_readiness"]["production_collection_ready"]
        ),
        "runtime_handoff_ready": bool(
            contract["collection_readiness"]["runtime_handoff_ready"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen BC+CBF public-server handoff."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--check-checkpoint-metadata",
        action="store_true",
        help="Also load the checkpoint with torch and validate its contract fields.",
    )
    args = parser.parse_args()
    report = validate_runtime_contract(
        args.project_root,
        check_checkpoint_metadata=args.check_checkpoint_metadata,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
