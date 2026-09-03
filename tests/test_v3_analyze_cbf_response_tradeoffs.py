from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from v3_chan.analyze_cbf_response_tradeoffs import (
    WINDOWS,
    _condition_specs_from_root,
    _resolve_prefix,
    analyze,
    condition_tradeoff_summary,
    main,
    pairing_verification,
    pareto_flags,
    response_window_masks,
    task_progress_v1,
)


def _write_run(
    directory: Path,
    name: str,
    *,
    condition: str,
    intervention_norms: list[float],
    recovery: list[int],
    progress_events: list[int],
    ee_x: list[float],
    ee_jerk: list[float],
    completion_time_s: float,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / name
    episode = {
        "episode": 0,
        "seed": 17,
        "encounter_id": "encounter-abc123",
        "scene_layout_id": "layout-fixed",
        "active_cube": 0,
        "success": True,
        "steps": len(intervention_norms),
        "completion_time_s": completion_time_s,
        "task_terminal_reason": "success",
        "grasped_any": True,
        "released_after_grasp": True,
        "object_drop": False,
        "collision": False,
        "near_miss_count": 0,
        "state_aware_recovery_timeout_count": 0,
        "minimum_ttc_s": 0.3,
        "initial_state_fingerprint": "initial-identical",
        "response_branch_state_fingerprint": "branch-identical",
        "response_branch_step": 3,
        "exact_prefix_digest_through_branch": "prefix-identical",
    }
    (prefix.with_suffix(".json")).write_text(
        json.dumps(
            {
                "config": {"physics_dt_s": 0.1},
                "episodes": [episode],
            }
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    for index, norm in enumerate(intervention_norms):
        correction = [norm, 0.0]
        rows.append(
            {
                "episode": 0,
                "seed": 17,
                "step": index + 1,
                "sim_time": (index + 1) * 0.1,
                "encounter_id": "encounter-abc123",
                "controller_event_after": progress_events[index],
                "controller_t_after": 0.0,
                "task_phase_before": "grasp_cube",
                "step_success": int(index == len(intervention_norms) - 1),
                "physical_safety_intervened": int(norm > 0.0),
                "physical_safety_intervention_norm_radps": norm,
                "physical_safety_correction_rate_norm_radps2": 0.0,
                "physical_safety_diagnostics_json": json.dumps(
                    {"correction_radps": correction}
                ),
                "physical_command_provenance_json": json.dumps(
                    {
                        "post_joint_velocities_radps": [
                            0.1 * index,
                            0.02 * index * index,
                        ],
                        "physics_dt_s": 0.1,
                    }
                ),
                "state_aware_recovery_control_authority": recovery[index],
                "state_aware_recovery_active": recovery[index],
                "post_ee_x": ee_x[index],
                "post_ee_y": 0.0,
                "post_ee_z": 0.2,
                "post_cube_x": 0.4 + 0.01 * index,
                "post_cube_y": 0.0,
                "post_cube_z": 0.03,
                "ee_cube_dist_m": 0.5 + (0.05 if index == 3 else 0.0),
                "cube_target_dist_m": 0.3,
                "has_grasped_cube": 0,
                "post_surface_gap_m": 0.06,
                "logged_surface_gap_below_configured_margin": 0,
                "logged_surface_gap_below_2cm": 0,
                "human_collision": 0,
                "near_miss": 0,
                "static_collision": 0,
                "static_geometry_valid": 1,
                "static_closest_robot_collider": "",
                "static_closest_environment_collider": "",
                "self_collision": 0,
                "self_geometry_valid": 1,
                "self_closest_first_collider": "",
                "self_closest_second_collider": "",
                "ee_jerk_norm_mps3": ee_jerk[index],
                "ee_jerk_valid": 1,
                "condition_hint": condition,
            }
        )
    fields = sorted({field for row in rows for field in row})
    with Path(str(prefix) + "_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(prefix)


@pytest.fixture
def synthetic_conditions(tmp_path: Path) -> dict[str, list[str]]:
    count = 8
    zeros = [0.0] * count
    no_recovery = [0] * count
    bc = _write_run(
        tmp_path / "bc_only",
        "seed_17",
        condition="BC-only",
        intervention_norms=zeros,
        recovery=no_recovery,
        progress_events=list(range(count)),
        ee_x=[0.1 * index for index in range(count)],
        ee_jerk=[1.0] * count,
        completion_time_s=0.8,
    )
    common_recovery = [0, 0, 0, 0, 1, 0, 0, 0]
    a = _write_run(
        tmp_path / "A",
        "seed_17",
        condition="A",
        intervention_norms=[0, 0, 2, 2, 0, 0, 0, 0],
        recovery=common_recovery,
        progress_events=[0, 1, 1, 1, 2, 4, 6, 7],
        ee_x=[0.0, 0.1, 0.15, 0.12, 0.18, 0.3, 0.5, 0.7],
        ee_jerk=[1, 1, 8, 6, 4, 3, 2, 1],
        completion_time_s=1.0,
    )
    b = _write_run(
        tmp_path / "B",
        "seed_17",
        condition="B",
        intervention_norms=[0, 0, 1.5, 1.5, 0, 0, 0, 0],
        recovery=common_recovery,
        progress_events=[0, 1, 2, 2, 3, 5, 6, 7],
        ee_x=[0.0, 0.1, 0.2, 0.25, 0.32, 0.42, 0.55, 0.7],
        ee_jerk=[1, 1, 7, 6, 4, 3, 2, 1],
        completion_time_s=0.9,
    )
    c = _write_run(
        tmp_path / "C",
        "seed_17",
        condition="C",
        intervention_norms=[0, 0, 1, 1, 0, 0, 0, 0],
        recovery=common_recovery,
        progress_events=[0, 1, 1, 1, 2, 4, 6, 7],
        ee_x=[0.0, 0.1, 0.14, 0.12, 0.18, 0.3, 0.5, 0.7],
        ee_jerk=[1, 1, 3, 3, 3, 2, 2, 1],
        completion_time_s=1.0,
    )
    return {"BC-only": [bc], "A": [a], "B": [b], "C": [c]}


def _rewrite_static_contacts(prefix: str, contacts: dict[int, tuple[str, str]]) -> None:
    path = Path(prefix + "_steps.csv")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    for index, row in enumerate(rows):
        pair = contacts.get(index)
        row["static_collision"] = int(pair is not None)
        row["static_closest_robot_collider"] = pair[0] if pair else ""
        row["static_closest_environment_collider"] = pair[1] if pair else ""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_task_progress_is_named_controller_clock_proxy() -> None:
    assert task_progress_v1(
        {"controller_event_after": "4", "controller_t_after": "0.5"}
    ) == pytest.approx(4.5 / 8.0)
    assert task_progress_v1({"controller_event_after": "3", "step_success": "1"}) == 1.0


def test_response_windows_are_explicit_and_exclusive_at_boundaries() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "physical_safety_intervened": int(index in (2, 3)),
                "physical_safety_intervention_norm_radps": int(index in (2, 3)),
                "state_aware_recovery_active": int(index == 4),
            }
        )
    masks = response_window_masks(rows, dt_s=0.1, resumed_duration_s=0.2)
    assert set(WINDOWS).issubset(masks)
    assert masks["PRE_INTERVENTION"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks["CBF_ACTIVE"].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["RECOVERY_ACTIVE"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert masks["BC_RESUMED"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_analysis_computes_paired_disruption_and_windowed_jerk(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    summaries, windows, qualification = analyze(
        synthetic_conditions, resumed_duration_s=0.2
    )
    a = next(row for row in summaries if row["condition"] == "A")
    assert a["paired_valid"] is True
    assert a["integrated_intervention_rad"] == pytest.approx(0.4)
    assert a["recovery_duration_s"] == pytest.approx(0.1)
    assert a["retreat_distance_m"] == pytest.approx(0.05)
    assert a["delta_completion_time_s"] == pytest.approx(0.2)
    assert a["progress_deficit_response_auc_s"] > 0.0
    assert a["cbf_ee_jerk_rms_mps3"] == pytest.approx(math.sqrt(50.0))
    assert a["correction_variation_source"] == "diagnostic_correction_vector"
    a_windows = [row for row in windows if row["condition"] == "A"]
    assert {row["window"] for row in a_windows} == set(WINDOWS)
    assert qualification["A"]["physical_safety_qualification_pass"] is True
    assert qualification["A"]["task_success_count"] == 1
    assert qualification["A"]["global_min_logged_surface_gap_m"] == pytest.approx(0.06)
    assert qualification["A"]["mean_episode_min_surface_gap_m"] == pytest.approx(0.06)
    assert any(
        row["pareto_progress_vs_jerk"]
        for row in summaries
        if row["condition"] != "BC-only"
    )

    pairing_rows, pairing_summary = pairing_verification(summaries, "BC-only")
    assert pairing_summary["all_response_pairs_exactly_verified"] is True
    assert pairing_rows[0]["exact_response_pairing_verified"] is True


def test_pareto_flags_reject_dominated_points() -> None:
    flags = pareto_flags([[1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [math.nan, 0.0]])
    assert flags.tolist() == [True, False, True, False]


def test_nonexact_response_branch_is_excluded_from_tradeoff_aggregate(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    c_json = Path(synthetic_conditions["C"][0] + ".json")
    payload = json.loads(c_json.read_text(encoding="utf-8"))
    payload["episodes"][0]["response_branch_step"] = 4
    c_json.write_text(json.dumps(payload), encoding="utf-8")

    summaries, _, _ = analyze(synthetic_conditions, resumed_duration_s=0.2)
    response_rows = [row for row in summaries if row["condition"] != "BC-only"]
    assert all(not row["exact_response_pairing_verified"] for row in response_rows)
    assert all(not row["pareto_overall_task_motion"] for row in response_rows)

    aggregates = condition_tradeoff_summary(summaries, "BC-only")
    a = next(row for row in aggregates if row["condition"] == "A")
    assert a["raw_response_episode_count"] == 1
    assert a["response_episode_count"] == 0
    assert a["excluded_nonexact_response_episode_count"] == 1


def test_first_applied_response_step_evidence_supersedes_legacy_active_marker(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    for condition, prefixes in synthetic_conditions.items():
        step_path = Path(prefixes[0] + "_steps.csv")
        with step_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = list(reader.fieldnames or ())
        fields.append("pre_step_state_fingerprint")
        for index, row in enumerate(rows):
            row["pre_step_state_fingerprint"] = f"shared-state-{index}"
        with step_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        if condition == "C":
            payload_path = Path(prefixes[0] + ".json")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["episodes"][0]["response_branch_step"] = 99
            payload["episodes"][0]["response_branch_state_fingerprint"] = "legacy-mismatch"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

    summaries, _, _ = analyze(synthetic_conditions, resumed_duration_s=0.2)
    response_rows = [row for row in summaries if row["condition"] != "BC-only"]
    assert all(row["exact_response_pairing_verified"] for row in response_rows)
    pairing_rows, pairing_summary = pairing_verification(summaries, "BC-only")
    assert pairing_summary["all_response_pairs_exactly_verified"] is True
    assert pairing_rows[0]["pairing_evidence_source"] == (
        "first_applied_response_step_log_v1"
    )


def test_persistent_task_contact_duration_is_not_a_new_collision_failure(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    task_pair = ("/robot/left_finger", "/world/table")
    _rewrite_static_contacts(synthetic_conditions["BC-only"][0], {2: task_pair})
    _rewrite_static_contacts(synthetic_conditions["A"][0], {2: task_pair, 3: task_pair})
    summaries, _, qualification = analyze(synthetic_conditions, resumed_duration_s=0.2)
    a = next(row for row in summaries if row["condition"] == "A")
    assert a["delta_static_collision_steps"] == 1
    assert a["increased_static_collision_steps_vs_bc"] is True
    assert a["new_static_collision_episode_vs_bc"] is False
    assert a["new_static_contact_pair_vs_bc"] is False
    assert a["new_static_collision_vs_bc"] is False
    assert a["response_candidate_episode_qualified"] is True
    assert (
        qualification["A"]["paired_increased_static_collision_steps_episode_count"] == 1
    )
    assert qualification["A"]["paired_new_static_collision_episode_count"] == 0


def test_new_contact_pair_is_qualification_failure_even_if_bc_already_contacts(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    task_pair = ("/robot/left_finger", "/world/table")
    new_pair = ("/robot/link7", "/world/table")
    _rewrite_static_contacts(synthetic_conditions["BC-only"][0], {2: task_pair})
    _rewrite_static_contacts(synthetic_conditions["A"][0], {2: task_pair, 3: new_pair})
    summaries, _, qualification = analyze(synthetic_conditions, resumed_duration_s=0.2)
    a = next(row for row in summaries if row["condition"] == "A")
    assert a["new_static_collision_episode_vs_bc"] is False
    assert a["new_static_contact_pair_vs_bc"] is True
    assert a["new_static_contact_pairs_vs_bc"] == ["/robot/link7||/world/table"]
    assert a["new_static_collision_vs_bc"] is True
    assert a["response_candidate_episode_qualified"] is False
    assert qualification["A"]["paired_new_static_contact_pair_episode_count"] == 1


def test_left_right_finger_table_raw_switch_is_same_task_contact_class(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    left = ("/World/Franka/panda_leftfinger/geometry", "/World/table")
    right = ("/World/Franka/panda_rightfinger/geometry", "/World/table")
    _rewrite_static_contacts(synthetic_conditions["BC-only"][0], {2: left})
    _rewrite_static_contacts(synthetic_conditions["A"][0], {2: right})
    summaries, _, qualification = analyze(synthetic_conditions, resumed_duration_s=0.2)
    a = next(row for row in summaries if row["condition"] == "A")
    assert a["new_static_contact_pairs_vs_bc"] == [
        "/World/Franka/panda_rightfinger/geometry||/World/table"
    ]
    assert a["raw_new_static_contact_pair_vs_bc"] is True
    assert a["new_static_contact_classes_vs_bc"] == []
    assert a["new_static_contact_pair_vs_bc"] is False
    assert a["new_static_collision_vs_bc"] is False
    assert a["response_candidate_episode_qualified"] is True
    assert qualification["A"]["paired_new_static_contact_pair_episode_count"] == 0
    assert qualification["A"]["paired_raw_new_static_contact_pair_episode_count"] == 1


def test_collision_episode_absent_in_bc_is_new_failure(
    synthetic_conditions: dict[str, list[str]],
) -> None:
    _rewrite_static_contacts(
        synthetic_conditions["A"][0],
        {3: ("/robot/link7", "/world/table")},
    )
    summaries, _, _ = analyze(synthetic_conditions, resumed_duration_s=0.2)
    a = next(row for row in summaries if row["condition"] == "A")
    assert a["new_static_collision_episode_vs_bc"] is True
    assert a["increased_static_collision_steps_vs_bc"] is True
    assert a["new_static_collision_vs_bc"] is True
    assert a["response_candidate_episode_qualified"] is False


def test_cli_writes_tables_json_and_four_plots(
    tmp_path: Path, synthetic_conditions: dict[str, list[str]]
) -> None:
    output = tmp_path / "analysis"
    arguments: list[str] = []
    for condition, specs in synthetic_conditions.items():
        arguments.extend(("--condition", f"{condition}={specs[0]}"))
    arguments.extend(("--resume-window-s", "0.2", "--output-directory", str(output)))
    assert main(arguments) == 0
    assert (output / "per_episode_paired_tradeoffs.csv").is_file()
    assert (output / "windowed_jerk.csv").is_file()
    assert (output / "qualification_summary.csv").is_file()
    assert (output / "condition_tradeoff_summary.csv").is_file()
    assert (output / "pairing_verification.csv").is_file()
    report = json.loads((output / "tradeoff_analysis.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "cbf_response_tradeoff_analysis_v1"
    assert report["pairing_verification"]["all_response_pairs_exactly_verified"] is True
    assert len(report["plot_files"]) == 4
    assert all(Path(path).is_file() for path in report["plot_files"])


def test_result_root_convention_discovers_all_conditions(
    tmp_path: Path, synthetic_conditions: dict[str, list[str]]
) -> None:
    del synthetic_conditions
    discovered = _condition_specs_from_root(str(tmp_path))
    assert {condition for condition, _ in discovered} == {"BC-only", "A", "B", "C"}


def test_runner_result_root_and_explicit_path_formats_preserve_condition_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runner"
    specifications = {
        "BC_ONLY": ([0.0] * 8, [0] * 8),
        "B_task_consistent_eps_0p05": ([0, 0, 1, 1, 0, 0, 0, 0], [0] * 8),
    }
    for condition_id, (intervention, recovery) in specifications.items():
        condition_dir = root / "dev" / condition_id
        prefix = _write_run(
            condition_dir,
            "result",
            condition=condition_id,
            intervention_norms=intervention,
            recovery=recovery,
            progress_events=list(range(8)),
            ee_x=[0.1 * index for index in range(8)],
            ee_jerk=[1.0] * 8,
            completion_time_s=0.8,
        )
        Path(prefix + "_steps.csv").rename(condition_dir / "steps.csv")

    discovered = _condition_specs_from_root(str(root))
    assert [condition for condition, _ in discovered] == [
        "BC_ONLY",
        "B_task_consistent_eps_0p05",
    ]
    condition_dir = root / "dev" / "BC_ONLY"
    assert _resolve_prefix(str(condition_dir)) == (
        (condition_dir / "result.json").resolve(),
        (condition_dir / "steps.csv").resolve(),
    )
    explicit = f"{condition_dir / 'result.json'}::{condition_dir / 'steps.csv'}"
    assert _resolve_prefix(explicit) == (
        (condition_dir / "result.json").resolve(),
        (condition_dir / "steps.csv").resolve(),
    )

    output = tmp_path / "runner_analysis"
    assert (
        main(
            [
                "--result-root",
                str(root),
                "--output-directory",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "tradeoff_analysis.json").read_text(encoding="utf-8"))
    assert report["baseline_condition"] == "BC_ONLY"
    assert {row["condition"] for row in report["episode_metrics"]} == set(
        specifications
    )
