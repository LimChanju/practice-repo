from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from v3_chan import run_cbf_response_family_experiment as runner


def _condition(family: str) -> runner.Condition:
    return next(
        item for item in runner.development_conditions() if item.family == family
    )


def _pair_episode(*, controller: str = "cbf") -> dict:
    return {
        "episode": 0,
        "seed": runner.SEED,
        "encounter_id": "encounter-1",
        "source_layout_id": "layout-1",
        "scene_layout_id": "layout-1",
        "source_layout_seed": 123,
        "collection_seed": 456,
        "screening_seed": runner.SEED,
        "active_cube": "cube_0",
        "source_cube_index": 0,
        "initial_cube_positions": [[0.4, 0.0, 0.97]],
        "initial_active_cube_position": [0.4, 0.0, 0.97],
        "place_target_position": [0.6, -0.25, 0.97],
        "restoration_mode": "exact_pose",
        "cube_pose_restored": True,
        "target_pose_restored": True,
        "robot_initial_state_restored": True,
        "pose_mismatch": False,
        "physical_safety_controller": controller,
        "logged_surface_gap_below_configured_margin_steps": 0,
        "logged_surface_gap_below_configured_margin_episode": False,
        "configured_safe_gap_m": 0.05,
        "static_collision_steps": 0,
        "static_collision_episode": False,
        "static_geometry_valid_steps": 10,
        "self_collision_steps": 0,
        "self_collision_episode": False,
        "self_geometry_valid_steps": 10,
    }


def _result_config(condition: runner.Condition, *, episodes: int = 1) -> dict:
    return {
        "checkpoint": str(runner.CHECKPOINT.resolve()),
        "episodes": episodes,
        "max_steps": runner.MAX_STEPS,
        "seed": runner.SEED,
        "mask_human_obs_for_policy": True,
        "fixed_orientation": True,
        "gripper_mode": "policy",
        "pseudo_errp_enabled": False,
        "encounter_manifest": "unused-in-cross-contract",
        "encounter_policy": "cycle",
        "encounter_anchor_mode": "world",
        "encounter_timebase": "recorded",
        "encounter_playback_speed": 1.0,
        "require_release_for_success": True,
        "strict_task_semantics": True,
        "strict_task_semantics_config": {
            "schema_version": runner.STRICT_TASK_SCHEMA,
            "enabled": True,
            "place_xy_tolerance_m": runner.STRICT_PLACE_XY_TOLERANCE_M,
            "state_aware_recovery": True,
        },
        "state_aware_recovery": True,
        "state_aware_recovery_config": {
            "schema_version": runner.RECOVERY_SCHEMA,
            "enabled": True,
            "maximum_recovery_steps": 600,
        },
        "task_failure_diagnostic_schema": "task_cbf_failure_diagnostics_v1",
        "physical_safety_controller": condition.controller,
        "extended_safety_logging": True,
        "effective_task_reward_version": runner.TASK_REWARD_VERSION,
        "safety_geometry_source": "test_geometry",
        "safety_geometry_metadata": {"links": ["panda_link6"]},
        **runner.SAFETY_CONSTRAINT_CONFIG,
        **condition.objective_config(),
    }


def _payload(condition: runner.Condition) -> dict:
    return {
        "config": _result_config(condition),
        "episodes": [_pair_episode(controller=condition.controller)],
    }


def _write_result_files(
    tmp_path: Path, condition: runner.Condition, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    encounter = tmp_path / "encounters.json"
    encounter.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "CHECKPOINT", checkpoint)
    payload = _payload(condition)
    payload["config"]["checkpoint"] = str(checkpoint.resolve())
    payload["config"]["encounter_manifest"] = str(encounter.resolve())
    result = tmp_path / "result.json"
    episodes_csv = tmp_path / "episodes.csv"
    steps_csv = tmp_path / "steps.csv"
    result.write_text(json.dumps(payload), encoding="utf-8")
    episodes_csv.write_text("episode\n0\n", encoding="utf-8")
    with steps_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runner.REQUIRED_STEP_COLUMNS))
        writer.writeheader()
        writer.writerow({name: 0 for name in runner.REQUIRED_STEP_COLUMNS})
    return result, episodes_csv, steps_csv, encounter


def test_development_grid_is_fixed_and_complete() -> None:
    conditions = runner.development_conditions()

    assert len(conditions) == 8
    assert [item.family for item in conditions].count("B") == 3
    assert [item.family for item in conditions].count("C") == 3
    assert {
        item.joint_regularization_epsilon for item in conditions if item.family == "B"
    } == {0.01, 0.05, 0.20}
    assert {
        item.correction_smoothness_weight for item in conditions if item.family == "C"
    } == {0.25, 1.0, 4.0}
    assert all(
        item.task_space_weight == 1.0 and item.task_yaw_length_scale_m_per_rad == 0.10
        for item in conditions
        if item.family == "B"
    )


def test_command_freezes_safety_recovery_and_objective(tmp_path: Path) -> None:
    condition = next(
        item
        for item in runner.development_conditions()
        if item.condition_id == "B_task_consistent_eps_0p01"
    )
    command = runner._build_command(
        condition=condition,
        encounter_manifest=runner.DEV_MANIFEST,
        output_json=tmp_path / "result.json",
        output_csv=tmp_path / "episodes.csv",
        output_steps=tmp_path / "steps.csv",
        device="cuda",
        eval_log_every=10,
    )

    assert command[0].endswith("launch_isaac.sh")
    assert command[command.index("--physical-safety-controller") + 1] == "cbf"
    assert command[command.index("--cbf-objective-mode") + 1] == "task_consistent"
    assert command[command.index("--cbf-joint-regularization-epsilon") + 1] == "0.01"
    assert command[command.index("--cbf-safe-gap-m") + 1] == "0.05"
    assert command[command.index("--cbf-activation-gap-m") + 1] == "0.13"
    assert "--extended-safety-logging" in command
    assert "--strict-task-semantics" in command
    assert "--state-aware-recovery" in command
    assert "--mask-human-obs-for-policy" in command
    assert command[command.index("--gripper-mode") + 1] == "policy"
    assert command[command.index("--seed") + 1] == "11"
    assert command[command.index("--max-steps") + 1] == "1200"


def test_cli_never_opens_heldout_without_frozen_selection() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(["--stage", "heldout"])
    with pytest.raises(SystemExit):
        runner._parse_args(["--stage", "all", "--selection-json", "x", "--force"])


def test_result_validator_accepts_exact_objective_and_extended_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    condition = _condition("A")
    result, episodes_csv, steps_csv, encounter = _write_result_files(
        tmp_path, condition, monkeypatch
    )

    payload = runner._validate_result(
        output_json=result,
        output_csv=episodes_csv,
        output_steps=steps_csv,
        encounter_manifest=encounter,
        expected_episodes=1,
        condition=condition,
    )

    assert payload["config"]["cbf_objective_mode"] == "joint_nominal"


def test_result_validator_fails_closed_on_objective_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    condition = _condition("A")
    result, episodes_csv, steps_csv, encounter = _write_result_files(
        tmp_path, condition, monkeypatch
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["config"]["cbf_objective_mode"] = "task_consistent"
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.ExperimentContractError, match="cbf_objective_mode"):
        runner._validate_result(
            output_json=result,
            output_csv=episodes_csv,
            output_steps=steps_csv,
            encounter_manifest=encounter,
            expected_episodes=1,
            condition=condition,
        )


def test_cross_condition_contract_accepts_objective_only_differences() -> None:
    conditions = (_condition("A"), _condition("B"), _condition("C"))
    payloads = {condition.condition_id: _payload(condition) for condition in conditions}

    report = runner._validate_cross_condition_contracts(payloads, conditions)

    assert report["status"] == "PASS"
    assert len(set(report["condition_safety_constraint_fingerprints"].values())) == 1
    assert len(set(report["condition_recovery_fingerprints"].values())) == 1


@pytest.mark.parametrize("drift", ["safety", "recovery", "pairing"])
def test_cross_condition_contract_rejects_nonobjective_drift(drift: str) -> None:
    conditions = (_condition("A"), _condition("B"), _condition("C"))
    payloads = {condition.condition_id: _payload(condition) for condition in conditions}
    target = payloads[conditions[1].condition_id]
    if drift == "safety":
        target["config"]["cbf_safe_gap_m"] = 0.04
    elif drift == "recovery":
        target["config"]["state_aware_recovery_config"]["maximum_recovery_steps"] = 1
    else:
        target["episodes"][0]["initial_active_cube_position"][0] = 0.5

    with pytest.raises(runner.ExperimentContractError):
        runner._validate_cross_condition_contracts(payloads, conditions)


def test_frozen_selection_is_sha_bound_and_family_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    run_manifest = dev_dir / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": runner.RUN_MANIFEST_SCHEMA,
                "split": "dev",
                "heldout_opened": False,
                "runtime_sources": {},
                "conditions": {
                    item.condition_id: {"family": item.family}
                    for item in runner.development_conditions()
                },
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": runner.SELECTION_SCHEMA,
                "development_run_manifest": {
                    "path": str(run_manifest),
                    "sha256": runner._sha256(run_manifest),
                },
                "selected_conditions": {
                    "B": "B_task_consistent_eps_0p05",
                    "C": "C_smooth_intervention_lambda_1p00",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_validate_runtime_sources_against_dev", lambda _: None)
    monkeypatch.setattr(
        runner, "_validate_frozen_dev_artifacts", lambda _dev, output_root: None
    )

    loaded = runner._load_frozen_selection(selection, output_root=tmp_path)
    heldout = runner._selected_heldout_conditions(loaded)

    assert [item.family for item in heldout] == ["BC_ONLY", "A", "B", "C"]

    data = json.loads(selection.read_text(encoding="utf-8"))
    data["development_run_manifest"]["sha256"] = "0" * 64
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(runner.ExperimentContractError, match="SHA-256"):
        runner._load_frozen_selection(selection, output_root=tmp_path)


def test_frozen_qualified_inputs_are_unchanged() -> None:
    runner._validate_frozen_inputs()
