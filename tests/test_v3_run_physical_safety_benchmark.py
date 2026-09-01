from __future__ import annotations

import json
from pathlib import Path

import pytest

from v3_chan import run_physical_safety_benchmark as benchmark


def _episode(*, encounter_id: str = "encounter_a") -> dict:
    steps = 10
    dt = 0.1
    return {
        "episode": 0,
        "encounter_id": encounter_id,
        "scene_layout_id": "layout_a",
        "restoration_mode": "exact_pose",
        "source_configuration_available": True,
        "pose_mismatch": False,
        "physical_safety_controller": "cbf",
        "success": True,
        "terminated": True,
        "truncated": False,
        "task_terminal_reason": "success",
        "collision": True,
        "grasped_any": True,
        "released_after_grasp": True,
        "cube_entered_target_tolerance": True,
        "controller_lift_phase_reached": True,
        "controller_place_phase_reached": True,
        "controller_release_phase_reached": True,
        "minimum_ttc_valid": True,
        "steps": steps,
        "collision_steps": 2,
        "near_steps": 3,
        "collision_event_count": 1,
        "physics_dt_s": dt,
        "total_reward": 1.0,
        "collision_rate": 0.2,
        "near_rate": 0.3,
        "near_miss_rate": 0.1,
        "gate_activation_rate": 0.5,
        "min_surface_gap": -0.01,
        "minimum_ttc_s": 0.2,
        "completion_time_s": steps * dt,
        "collision_duration_s": 2 * dt,
        "collision_max_consecutive_duration_s": 2 * dt,
        "near_human_duration_s": 3 * dt,
        "physical_safety_active_rate": 0.5,
        "physical_safety_intervention_rate": 0.4,
        "physical_safety_feasible_rate": 1.0,
        "mean_physical_safety_intervention_norm_radps": 0.2,
        "max_physical_safety_intervention_norm_radps": 0.4,
        "mean_physical_safety_slack_mps": 0.0,
        "max_physical_safety_slack_mps": 0.0,
        "mean_physical_safety_constraint_violation_before_mps": 0.01,
        "max_physical_safety_constraint_violation_before_mps": 0.02,
        "mean_physical_safety_constraint_violation_after_mps": 0.0,
        "max_physical_safety_constraint_violation_after_mps": 0.0,
        "mean_physical_safety_solve_time_ms": 0.2,
        "ee_path_length_m": 0.5,
        "rms_ee_acceleration_mps2": 0.3,
        "p95_ee_jerk_mps3": 0.4,
        "rms_ee_jerk_mps3": 0.3,
        "max_ee_jerk_mps3": 0.5,
        "integrated_squared_ee_jerk_m2ps5": 0.1,
        "rms_gate_ee_acceleration_mps2": 0.2,
        "p95_gate_ee_jerk_mps3": 0.3,
        "rms_gate_ee_jerk_mps3": 0.2,
        "max_gate_ee_jerk_mps3": 0.4,
    }


def _result(*, require_release: bool = True) -> dict:
    return {
        "config": {
            "physical_safety_controller": "cbf",
            "mask_human_obs_for_policy": True,
            "pseudo_errp_enabled": False,
            "encounter_timebase": "recorded",
            "require_release_for_success": require_release,
        },
        "episodes": [_episode()],
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_single_pick_result_contract_accepts_released_success(tmp_path) -> None:
    path = tmp_path / "result.json"
    _write(path, _result())

    benchmark._validate_result(path, 1, expected_controller="cbf")


def test_strict_single_pick_result_contract_rejects_target_entry_success(
    tmp_path,
) -> None:
    path = tmp_path / "result.json"
    _write(path, _result(require_release=False))

    with pytest.raises(RuntimeError, match="released-pick"):
        benchmark._validate_result(path, 1, expected_controller="cbf")


def test_runner_requests_strict_release_success() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    assert '"--require-release-for-success"' in source
    assert "collision_free_success" in source


def test_runner_forwards_opt_in_strict_task_semantics() -> None:
    benchmark_source = Path(benchmark.__file__).read_text(encoding="utf-8")
    evaluator_source = (benchmark.SCRIPT_DIR / "evaluate_rollout_policy.py").read_text(
        encoding="utf-8"
    )
    assert 'command.append("--strict-task-semantics")' in benchmark_source
    assert "strict_task_semantics=bool(args.strict_task_semantics)" in evaluator_source
    assert '"strict_task_semantics": bool(args.strict_task_semantics)' in evaluator_source


def test_runner_forwards_strict_place_xy_tolerance() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    evaluator_source = (benchmark.SCRIPT_DIR / "evaluate_rollout_policy.py").read_text(
        encoding="utf-8"
    )

    assert '"--strict-place-xy-tolerance-m"' in source
    assert "str(args.strict_place_xy_tolerance_m)" in source
    assert '"--strict-place-xy-tolerance-m"' in evaluator_source
    assert "strict_place_xy_tolerance_m=float(" in evaluator_source


def test_runner_can_override_retired_checkpoint_reward_for_bc_evaluation() -> None:
    benchmark_source = Path(benchmark.__file__).read_text(encoding="utf-8")
    evaluator_source = (benchmark.SCRIPT_DIR / "evaluate_rollout_policy.py").read_text(
        encoding="utf-8"
    )
    assert '"--task-reward-version"' in benchmark_source
    assert 'command.extend(\n                    ["--task-reward-version"' in benchmark_source
    assert "effective_task_reward_version" in evaluator_source
    assert '"checkpoint_task_reward_version"' in evaluator_source
    assert '"task_reward_version_override"' in evaluator_source


def test_strict_result_contract_requires_diagnostic_provenance(tmp_path) -> None:
    path = tmp_path / "result.json"
    payload = _result()
    payload["config"]["strict_task_semantics"] = True
    payload["config"]["task_failure_diagnostic_schema"] = (
        benchmark.TASK_FAILURE_DIAGNOSTIC_SCHEMA
    )
    payload["episodes"][0]["strict_task_semantics_enabled"] = True
    payload["episodes"][0]["task_failure_diagnostic_schema"] = (
        benchmark.TASK_FAILURE_DIAGNOSTIC_SCHEMA
    )
    _write(path, payload)

    benchmark._validate_result(
        path,
        1,
        expected_controller="cbf",
        require_episode_metrics=True,
        expected_strict_task_semantics=True,
    )

    payload["episodes"][0]["strict_task_semantics_enabled"] = False
    _write(path, payload)
    with pytest.raises(RuntimeError, match="strict task diagnostics"):
        benchmark._validate_result(
            path,
            1,
            expected_controller="cbf",
            require_episode_metrics=True,
            expected_strict_task_semantics=True,
        )


def test_strict_result_contract_rejects_terminal_semantic_conflation(
    tmp_path,
) -> None:
    path = tmp_path / "result.json"
    payload = _result()
    payload["config"]["strict_task_semantics"] = True
    payload["config"]["task_failure_diagnostic_schema"] = (
        benchmark.TASK_FAILURE_DIAGNOSTIC_SCHEMA
    )
    episode = payload["episodes"][0]
    episode["strict_task_semantics_enabled"] = True
    episode["task_failure_diagnostic_schema"] = (
        benchmark.TASK_FAILURE_DIAGNOSTIC_SCHEMA
    )
    episode["task_terminal_reason"] = "place_readiness_timeout"
    episode["success"] = False
    episode["terminated"] = False
    episode["truncated"] = True
    _write(path, payload)

    with pytest.raises(RuntimeError, match="strict terminal semantics drift"):
        benchmark._validate_result(
            path,
            1,
            expected_controller="cbf",
            require_episode_metrics=True,
            expected_strict_task_semantics=True,
        )


def test_rollout_exports_stage2_raw_metrics() -> None:
    source = (benchmark.SCRIPT_DIR / "evaluate_rollout_policy.py").read_text(
        encoding="utf-8"
    )
    for field in (
        '"released_after_grasp"',
        '"controller_lift_phase_reached"',
        '"collision_duration_s"',
        '"collision_max_consecutive_duration_s"',
        '"near_human_duration_s"',
        '"minimum_ttc_s"',
        '"mean_physical_safety_constraint_violation_before_mps"',
        '"mean_physical_safety_constraint_violation_after_mps"',
    ):
        assert field in source


def test_result_contract_rejects_nonfinite_metric(tmp_path) -> None:
    path = tmp_path / "result.json"
    payload = _result()
    payload["episodes"][0]["minimum_ttc_s"] = float("nan")
    _write(path, payload)

    with pytest.raises(RuntimeError, match="minimum_ttc_s must be finite"):
        benchmark._validate_result(
            path,
            1,
            expected_controller="cbf",
            require_episode_metrics=True,
        )


def test_pairing_rejects_duplicate_keys() -> None:
    rows = [
        {
            "controller": "none",
            "eval_seed": 11,
            "encounter_id": "a",
            "scene_layout_id": "layout",
        },
        {
            "controller": "none",
            "eval_seed": 11,
            "encounter_id": "a",
            "scene_layout_id": "layout",
        },
        {
            "controller": "cbf",
            "eval_seed": 11,
            "encounter_id": "a",
            "scene_layout_id": "layout",
        },
    ]
    with pytest.raises(RuntimeError, match="duplicate pairing keys"):
        benchmark._validate_pairing(rows, ("none", "cbf"))


def _gate_summary(*, split: str = "calibration") -> dict:
    return {
        "evaluation_split": split,
        "pairing": {"exact": True},
        "controllers": {
            "none": {"collision_episode": 0.5, "success": 0.8},
            "cbf": {"collision_episode": 0.2, "success": 0.7},
        },
    }


def _thresholds() -> dict:
    return {
        "schema_version": benchmark.FEASIBILITY_THRESHOLD_SCHEMA,
        "protocol_id": "frozen_protocol_v1",
        "frozen_before_evaluation": True,
        "evaluation_split": "calibration",
        "reference_controller": "none",
        "candidate_controller": "cbf",
        "minimum_collision_episode_rate_reduction": 0.2,
        "maximum_success_rate_drop": 0.15,
        "minimum_candidate_success_rate": 0.6,
    }


def test_frozen_physical_feasibility_gate_can_pass() -> None:
    report = benchmark.decide_physical_feasibility(_gate_summary(), _thresholds())
    assert report["decision"] == "GO"
    assert report["unlocks_constrained_ppo"] is True
    assert report["checks"]["human_collision_episode_rate_reduction"]["passed"] is True


def test_physical_feasibility_gate_fails_closed_without_frozen_contract() -> None:
    thresholds = _thresholds()
    thresholds["frozen_before_evaluation"] = False
    report = benchmark.decide_physical_feasibility(_gate_summary(), thresholds)
    assert report["decision"] == "NO_GO"
    assert report["status"] == "INVALID_CONTRACT"
    assert "thresholds_not_declared_frozen" in report["reason_codes"]


def test_heldout_evidence_cannot_unlock_calibration_gate() -> None:
    thresholds = _thresholds()
    thresholds["evaluation_split"] = "heldout"
    report = benchmark.decide_physical_feasibility(
        _gate_summary(split="heldout"), thresholds
    )
    assert report["decision"] == "NO_GO"
    assert "heldout_cannot_unlock_calibration_gate" in report["reason_codes"]


def test_summary_propagates_metrics_and_writes_machine_gate(tmp_path) -> None:
    result_dir = tmp_path / "benchmark"
    for controller, collision_steps in (("none", 5), ("cbf", 0)):
        payload = _result()
        payload["config"]["physical_safety_controller"] = controller
        episode = payload["episodes"][0]
        episode["physical_safety_controller"] = controller
        episode["collision_steps"] = collision_steps
        episode["collision"] = collision_steps > 0
        episode["collision_rate"] = collision_steps / episode["steps"]
        episode["collision_duration_s"] = collision_steps * episode["physics_dt_s"]
        episode["collision_max_consecutive_duration_s"] = (
            min(collision_steps, 2) * episode["physics_dt_s"]
        )
        episode["collision_event_count"] = int(collision_steps > 0)
        output_dir = result_dir / controller
        output_dir.mkdir(parents=True)
        _write(output_dir / "seed_11.json", payload)
    thresholds_path = tmp_path / "thresholds.json"
    _write(thresholds_path, _thresholds())

    summary = benchmark._summarize(
        ("none", "cbf"),
        (11,),
        result_dir,
        evaluation_split="calibration",
        expected_episodes=1,
        thresholds_path=thresholds_path,
    )

    assert summary["evaluation_split"] == "calibration"
    assert summary["collision_signal_scope"]["static"] == ("unavailable_not_inferred")
    assert summary["controllers"]["cbf"]["minimum_ttc_s"] == pytest.approx(0.2)
    assert summary["controllers"]["cbf"][
        "max_constraint_violation_before_mps"
    ] == pytest.approx(0.02)
    assert summary["physical_feasibility"]["decision"] == "GO"
    assert (result_dir / "physical_feasibility.json").exists()


def test_summary_rejects_noncontroller_config_drift(tmp_path) -> None:
    result_dir = tmp_path / "benchmark"
    for controller, max_steps in (("none", 4500), ("cbf", 1200)):
        payload = _result()
        payload["config"]["physical_safety_controller"] = controller
        payload["config"]["max_steps"] = max_steps
        payload["episodes"][0]["physical_safety_controller"] = controller
        output_dir = result_dir / controller
        output_dir.mkdir(parents=True)
        _write(output_dir / "seed_11.json", payload)

    with pytest.raises(RuntimeError, match="Non-controller evaluation config"):
        benchmark._summarize(
            ("none", "cbf"),
            (11,),
            result_dir,
            evaluation_split="calibration",
            expected_episodes=1,
        )


def test_calibration_rejects_locked_eval_manifest_before_rollout(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": "hri_encounter_manifest_v2",
            "split_metadata": {"role": "eval"},
            "scenarios": [{"id": "a"}],
        },
    )

    with pytest.raises(RuntimeError, match="Manifest/evaluation split mismatch"):
        benchmark._validate_manifest_split(manifest, "calibration")


def test_open_development_manifest_is_valid_calibration_input(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": "hri_encounter_manifest_v2",
            "split_metadata": {"role": "development"},
            "scenarios": [{"id": "a"}],
        },
    )

    contract = benchmark._validate_manifest_split(manifest, "calibration")

    assert contract["native_role"] == "development"
    assert contract["scenario_count"] == 1
