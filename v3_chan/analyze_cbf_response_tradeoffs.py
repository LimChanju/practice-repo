"""Paired task--motion trade-off analysis for BC and CBF response families.

The script is deliberately independent of Isaac Sim.  It consumes the episode
JSON and step CSV written by :mod:`v3_chan.evaluate_rollout_policy`, pairs each
safety response with the matching BC-only counterfactual, and writes auditable
tables plus four Pareto scatter plots.

Examples
--------

Explicit run prefixes (``.json`` and ``_steps.csv`` are inferred)::

    python v3_chan/analyze_cbf_response_tradeoffs.py \
      --condition BC-only=results/none/seed_11 \
      --condition A=results/cbf/seed_11 \
      --condition B=results/cbf_task_consistent/seed_11 \
      --condition C=results/cbf_smooth_intervention/seed_11 \
      --output-directory results/tradeoff_analysis

Or use the benchmark directory convention::

    python v3_chan/analyze_cbf_response_tradeoffs.py \
      --result-root results/final_abc \
      --output-directory results/final_abc/tradeoff_analysis

The task-progress quantity is a *controller-clock proxy*, not geometric task
completion: ``min(1, (clip(event, 0, 8) + clip(t, 0, 1)) / 8)`` (explicit
success is one).  ``progress_deficit`` is BC-only minus the response value, so
positive values mean that the safety response is behind its paired nominal
rollout.  Physical cube/goal regression is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_SCHEMA = "cbf_response_tradeoff_analysis_v1"
WINDOWS = (
    "PRE_INTERVENTION",
    "CBF_ACTIVE",
    "RECOVERY_ACTIVE",
    "BC_RESUMED",
    "WHOLE_TASK",
)
DEFAULT_DT_S = 1.0 / 60.0
INTERVENTION_EPS = 1e-8


def _artifact_reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


@dataclass
class EpisodeTrace:
    condition: str
    pair_key: str
    episode: dict[str, Any]
    steps: list[dict[str, str]]
    source_json: str
    source_steps_csv: str
    physics_dt_s: float
    configured_safe_gap_m: float


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _first_finite(mapping: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = _finite_float(mapping.get(name))
        if math.isfinite(value):
            return value
    return math.nan


def task_progress_v1(row: Mapping[str, Any]) -> float:
    """Return the explicitly named controller-clock progress proxy."""

    logged = _finite_float(row.get("task_progress_v1"))
    if math.isfinite(logged):
        return float(np.clip(logged, 0.0, 1.0))
    if _boolean(row.get("step_success")) or _boolean(row.get("success")):
        return 1.0
    event = _integer(
        row.get("controller_event_after", row.get("final_controller_event", 0))
    )
    controller_t = _finite_float(
        row.get("controller_t_after", row.get("final_controller_t", 0.0)), 0.0
    )
    return float(
        min(1.0, (min(8, max(0, event)) + np.clip(controller_t, 0.0, 1.0)) / 8.0)
    )


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            parsed = _json_value(stripped, stripped)
            if parsed is not stripped:
                value = parsed
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def episode_pair_key(episode: Mapping[str, Any]) -> str:
    """Build a deterministic counterfactual key without using condition name."""

    seed = str(episode.get("seed", ""))
    encounter = str(episode.get("encounter_id", "")).strip()
    if not encounter:
        encounter = "human=" + str(episode.get("human_replay_episode", ""))
    layout = str(episode.get("scene_layout_id", episode.get("screening_layout_id", "")))
    active_cube = str(episode.get("active_cube", ""))
    return "|".join((seed, encounter, layout, active_cube))


def _resolve_prefix(spec: str) -> tuple[Path, Path]:
    if "::" in spec:
        json_text, steps_text = spec.split("::", 1)
        if not json_text.strip() or not steps_text.strip():
            raise ValueError(
                f"Explicit condition paths must be JSON::STEPS, got: {spec!r}"
            )
        return (
            Path(json_text.strip()).expanduser().resolve(),
            Path(steps_text.strip()).expanduser().resolve(),
        )
    path = Path(spec).expanduser().resolve()
    if path.is_dir():
        conventional_json = path / "result.json"
        conventional_steps = path / "steps.csv"
        if conventional_json.is_file() and conventional_steps.is_file():
            return conventional_json, conventional_steps
        json_candidates = sorted(path.glob("*.json"))
        pairs = []
        for json_path in json_candidates:
            step_path = Path(str(json_path)[: -len(".json")] + "_steps.csv")
            if step_path.is_file():
                pairs.append((json_path, step_path))
        if len(pairs) == 1:
            return pairs[0]
        raise ValueError(
            "Directory condition spec must contain result.json + steps.csv or "
            f"exactly one <prefix>.json + <prefix>_steps.csv pair: {path}"
        )
    text = str(path)
    if text.endswith("_steps.csv"):
        return Path(text[: -len("_steps.csv")] + ".json"), path
    if path.name == "steps.csv":
        return path.with_name("result.json"), path
    if path.suffix.lower() == ".json":
        inferred = Path(text[: -len(".json")] + "_steps.csv")
        if inferred.is_file() or path.name != "result.json":
            return path, inferred
        return path, path.with_name("steps.csv")
    if path.suffix.lower() == ".csv":
        raise ValueError(
            f"Step CSV must end in '_steps.csv' so its episode JSON can be inferred: {path}"
        )
    return Path(text + ".json"), Path(text + "_steps.csv")


def _condition_specs_from_root(root: str) -> list[tuple[str, str]]:
    root_path = Path(root).expanduser().resolve()
    runner_base = root_path / "dev" if (root_path / "dev").is_dir() else root_path
    runner_results = [
        directory
        for directory in sorted(runner_base.iterdir())
        if directory.is_dir()
        and (directory / "result.json").is_file()
        and (directory / "steps.csv").is_file()
    ]
    if runner_results:
        # Preserve the exact condition_id directory name: parameter-sweep
        # settings must remain distinct instead of being collapsed into B/C.
        return [(directory.name, str(directory)) for directory in runner_results]

    candidates = {
        "BC-only": ("bc_only", "BC-only", "bc", "none"),
        "A": ("A", "a", "cbf"),
        "B": ("B", "b", "cbf_task_consistent"),
        "C": ("C", "c", "cbf_smooth_intervention"),
    }
    discovered: list[tuple[str, str]] = []
    for name, directory_names in candidates.items():
        directory = next(
            (
                root_path / item
                for item in directory_names
                if (root_path / item).is_dir()
            ),
            None,
        )
        if directory is None:
            continue
        for json_path in sorted(directory.glob("*.json")):
            step_path = Path(str(json_path)[: -len(".json")] + "_steps.csv")
            if step_path.is_file():
                discovered.append((name, str(json_path)))
    if not discovered:
        raise FileNotFoundError(
            "No runner dev/<condition_id>/result.json + steps.csv or paired "
            "<prefix>.json + <prefix>_steps.csv files were found under: "
            f"{root_path}"
        )
    return discovered


def _read_steps(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(_integer(row.get("episode"), -1), []).append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda item: _integer(item.get("step"), 0))
    return grouped


def _infer_dt(
    episode: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    for mapping in (episode, config):
        value = _finite_float(mapping.get("physics_dt_s"))
        if math.isfinite(value) and value > 0.0:
            return value
    provenance_dts: list[float] = []
    for row in rows[:10]:
        provenance = _json_value(row.get("physical_command_provenance_json"), {})
        value = _finite_float(
            provenance.get("physics_dt_s") if isinstance(provenance, Mapping) else None
        )
        if math.isfinite(value) and value > 0.0:
            provenance_dts.append(value)
    if provenance_dts:
        return float(np.median(provenance_dts))
    times = np.asarray(
        [_finite_float(row.get("sim_time")) for row in rows], dtype=float
    )
    if times.size >= 2:
        deltas = np.diff(times)
        valid = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if valid.size:
            return float(np.median(valid))
    return DEFAULT_DT_S


def load_condition(condition: str, specs: Sequence[str]) -> list[EpisodeTrace]:
    traces: list[EpisodeTrace] = []
    seen: set[str] = set()
    for spec in specs:
        json_path, steps_path = _resolve_prefix(spec)
        if not json_path.is_file():
            raise FileNotFoundError(f"Episode JSON does not exist: {json_path}")
        if not steps_path.is_file():
            raise FileNotFoundError(f"Step CSV does not exist: {steps_path}")
        with json_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
        if not isinstance(episodes, list):
            raise ValueError(f"JSON has no episode list: {json_path}")
        config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(config, Mapping):
            config = {}
        grouped_steps = _read_steps(steps_path)
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            episode_dict = dict(episode)
            episode_index = _integer(episode_dict.get("episode"), -1)
            rows = grouped_steps.get(episode_index, [])
            if not rows:
                raise ValueError(
                    f"No step rows for episode {episode_index} in {steps_path}"
                )
            pair_key = episode_pair_key(episode_dict)
            if pair_key in seen:
                raise ValueError(
                    f"Ambiguous duplicate paired key in condition {condition!r}: {pair_key}"
                )
            seen.add(pair_key)
            traces.append(
                EpisodeTrace(
                    condition=condition,
                    pair_key=pair_key,
                    episode=episode_dict,
                    steps=rows,
                    source_json=str(json_path),
                    source_steps_csv=str(steps_path),
                    physics_dt_s=_infer_dt(episode_dict, rows, config),
                    configured_safe_gap_m=_finite_float(config.get("cbf_safe_gap_m")),
                )
            )
    return traces


def _vector(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray | None:
    values = np.asarray([_finite_float(row.get(name)) for name in names], dtype=float)
    return values if np.all(np.isfinite(values)) else None


def _provenance_joint_velocity(row: Mapping[str, Any]) -> np.ndarray | None:
    payload = _json_value(row.get("physical_command_provenance_json"), {})
    if not isinstance(payload, Mapping):
        return None
    values = np.asarray(
        payload.get("post_joint_velocities_radps", []), dtype=float
    ).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return values


def _diagnostic_correction(row: Mapping[str, Any]) -> np.ndarray | None:
    payload = _json_value(row.get("physical_safety_diagnostics_json"), {})
    if not isinstance(payload, Mapping):
        return None
    values = np.asarray(payload.get("correction_radps", []), dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return values


def _diagnostic_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _json_value(row.get("physical_safety_diagnostics_json"), {})
    return payload if isinstance(payload, Mapping) else {}


def _norm_statistics(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"sample_count": 0, "rms": math.nan, "p95": math.nan, "peak": math.nan}
    return {
        "sample_count": int(finite.size),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "p95": float(np.percentile(finite, 95.0)),
        "peak": float(np.max(finite)),
    }


def _joint_jerk(rows: Sequence[Mapping[str, Any]], dt_s: float) -> np.ndarray:
    velocities = [_provenance_joint_velocity(row) for row in rows]
    result = np.full(len(rows), np.nan, dtype=float)
    if len(rows) < 3:
        return result
    for index in range(2, len(rows)):
        before, previous, current = velocities[index - 2 : index + 1]
        if before is None or previous is None or current is None:
            continue
        if before.shape != previous.shape or previous.shape != current.shape:
            continue
        previous_acceleration = (previous - before) / dt_s
        current_acceleration = (current - previous) / dt_s
        result[index] = float(
            np.linalg.norm((current_acceleration - previous_acceleration) / dt_s)
        )
    return result


def _ee_jerk(rows: Sequence[Mapping[str, Any]], dt_s: float) -> np.ndarray:
    logged = np.asarray(
        [_finite_float(row.get("ee_jerk_norm_mps3")) for row in rows], dtype=float
    )
    valid_field_present = any("ee_jerk_valid" in row for row in rows)
    if valid_field_present:
        valid = np.asarray([_boolean(row.get("ee_jerk_valid")) for row in rows])
        logged[~valid] = np.nan
    if np.any(np.isfinite(logged)):
        return logged
    positions = [_vector(row, ("post_ee_x", "post_ee_y", "post_ee_z")) for row in rows]
    result = np.full(len(rows), np.nan, dtype=float)
    if len(rows) < 4:
        return result
    for index in range(3, len(rows)):
        sample = positions[index - 3 : index + 1]
        if any(value is None for value in sample):
            continue
        p0, p1, p2, p3 = sample
        result[index] = float(np.linalg.norm((p3 - 3 * p2 + 3 * p1 - p0) / dt_s**3))
    return result


def response_window_masks(
    rows: Sequence[Mapping[str, Any]], dt_s: float, resumed_duration_s: float = 0.5
) -> dict[str, np.ndarray]:
    """Return explicit window masks, merging response gaps shorter than 0.5 s."""

    count = len(rows)
    intervention = np.asarray(
        [
            _boolean(row.get("physical_safety_intervened"))
            or _finite_float(row.get("physical_safety_intervention_norm_radps"), 0.0)
            > INTERVENTION_EPS
            for row in rows
        ],
        dtype=bool,
    )
    recovery = np.asarray(
        [
            _boolean(row.get("state_aware_recovery_control_authority"))
            or _boolean(row.get("state_aware_recovery_active"))
            for row in rows
        ],
        dtype=bool,
    )
    raw_response = intervention | recovery
    bridge_steps = max(1, int(round(resumed_duration_s / dt_s)))
    response = raw_response.copy()
    active_indices = np.flatnonzero(raw_response)
    if active_indices.size:
        for left, right in zip(active_indices[:-1], active_indices[1:]):
            if 1 < right - left <= bridge_steps + 1:
                response[left : right + 1] = True
    segments: list[tuple[int, int]] = []
    index = 0
    while index < count:
        if not response[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and response[end + 1]:
            end += 1
        segments.append((index, end))
        index = end + 1

    pre = np.zeros(count, dtype=bool)
    resumed = np.zeros(count, dtype=bool)
    for start, end in segments:
        pre[max(0, start - bridge_steps) : start] = True
        resumed[end + 1 : min(count, end + 1 + bridge_steps)] = True
    pre &= ~response
    resumed &= ~response
    resumed &= ~pre  # an imminent intervention is classified as PRE, not resumed.
    return {
        "PRE_INTERVENTION": pre,
        "CBF_ACTIVE": intervention,
        "RECOVERY_ACTIVE": recovery & ~intervention,
        "BC_RESUMED": resumed,
        "WHOLE_TASK": np.ones(count, dtype=bool),
        "_RESPONSE": response,
    }


def _path_length(positions: Sequence[np.ndarray | None], mask: np.ndarray) -> float:
    total = 0.0
    for index in range(1, len(positions)):
        before, current = positions[index - 1], positions[index]
        if not mask[index] or not mask[index - 1] or before is None or current is None:
            continue
        total += float(np.linalg.norm(current - before))
    return total


def _response_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(mask) and mask[end + 1]:
            end += 1
        segments.append((index, end))
        index = end + 1
    return segments


def _contact_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    collision_field: str,
    first_field: str,
    second_field: str,
    undirected: bool,
) -> tuple[list[str], bool]:
    """Return canonical collider identities observed on collision steps."""

    identity_fields_available = any(
        first_field in row and second_field in row for row in rows
    )
    pairs: set[str] = set()
    for row in rows:
        if not _boolean(row.get(collision_field)):
            continue
        first = str(row.get(first_field, "")).strip()
        second = str(row.get(second_field, "")).strip()
        if not first or not second:
            continue
        if undirected and second < first:
            first, second = second, first
        pairs.add(f"{first}||{second}")
    return sorted(pairs), identity_fields_available


def _static_contact_class(pair: str) -> str:
    """Collapse symmetric finger/table pairs into one task-contact class."""

    try:
        robot_path, environment_path = str(pair).split("||", 1)
    except ValueError:
        return str(pair)
    robot_class = robot_path
    if "panda_leftfinger" in robot_path or "panda_rightfinger" in robot_path:
        robot_class = "gripper_finger"
    environment_class = environment_path
    if environment_path == "/World/table" or environment_path.startswith(
        "/World/table/"
    ):
        environment_class = "table"
    return f"{robot_class}||{environment_class}"


def summarize_trace(
    trace: EpisodeTrace, resumed_duration_s: float = 0.5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = trace.steps
    episode = trace.episode
    dt_s = trace.physics_dt_s
    masks = response_window_masks(rows, dt_s, resumed_duration_s)
    response_mask = masks["_RESPONSE"]
    intervention_norm = np.asarray(
        [
            _finite_float(row.get("physical_safety_intervention_norm_radps"), 0.0)
            for row in rows
        ],
        dtype=float,
    )
    correction_rate = np.asarray(
        [
            _finite_float(row.get("physical_safety_correction_rate_norm_radps2"))
            for row in rows
        ],
        dtype=float,
    )
    corrections = [_diagnostic_correction(row) for row in rows]
    correction_tv = 0.0
    correction_direction_change = 0.0
    vector_pairs = 0
    for before, current in zip(corrections[:-1], corrections[1:]):
        if before is None or current is None or before.shape != current.shape:
            continue
        correction_tv += float(np.linalg.norm(current - before))
        before_norm = float(np.linalg.norm(before))
        current_norm = float(np.linalg.norm(current))
        if before_norm > INTERVENTION_EPS and current_norm > INTERVENTION_EPS:
            cosine = float(
                np.clip(np.dot(before, current) / (before_norm * current_norm), -1, 1)
            )
            correction_direction_change += float(math.acos(cosine))
        vector_pairs += 1
    if vector_pairs:
        correction_source = "diagnostic_correction_vector"
    elif np.any(np.isfinite(correction_rate)):
        correction_tv = float(np.nansum(correction_rate) * dt_s)
        correction_direction_change = math.nan
        correction_source = "logged_correction_rate_norm"
    else:
        correction_tv = float(np.sum(np.abs(np.diff(intervention_norm))))
        correction_direction_change = math.nan
        correction_source = "scalar_intervention_norm_fallback"

    ee_positions = [
        _vector(row, ("post_ee_x", "post_ee_y", "post_ee_z")) for row in rows
    ]
    cube_positions = [
        _vector(row, ("post_cube_x", "post_cube_y", "post_cube_z")) for row in rows
    ]
    ee_jerk = _ee_jerk(rows, dt_s)
    joint_jerk = _joint_jerk(rows, dt_s)
    window_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        mask = masks[window]
        ee_stats = _norm_statistics(ee_jerk[mask])
        joint_stats = _norm_statistics(joint_jerk[mask])
        window_rows.append(
            {
                "condition": trace.condition,
                "pair_key": trace.pair_key,
                "encounter_id": str(episode.get("encounter_id", "")),
                "task_phase": _response_task_phase(rows, masks),
                "window": window,
                "duration_s": float(np.count_nonzero(mask) * dt_s),
                "ee_jerk_sample_count": ee_stats["sample_count"],
                "ee_jerk_rms_mps3": ee_stats["rms"],
                "ee_jerk_p95_mps3": ee_stats["p95"],
                "ee_jerk_peak_mps3": ee_stats["peak"],
                "joint_jerk_sample_count": joint_stats["sample_count"],
                "joint_jerk_rms_radps3": joint_stats["rms"],
                "joint_jerk_p95_radps3": joint_stats["p95"],
                "joint_jerk_peak_radps3": joint_stats["peak"],
            }
        )

    retreat_distance_m = 0.0
    cube_goal_regression_m = 0.0
    ee_displacement_m = 0.0
    cube_displacement_m = 0.0
    surface_gap_increase_m = 0.0
    for start, end in _response_segments(response_mask):
        grasped_at_onset = _boolean(rows[start].get("has_grasped_cube"))
        error_field = "cube_target_dist_m" if grasped_at_onset else "ee_cube_dist_m"
        errors = np.asarray(
            [
                _finite_float(rows[index].get(error_field))
                for index in range(start, end + 1)
            ]
        )
        if errors.size and math.isfinite(errors[0]):
            retreat_distance_m = max(
                retreat_distance_m,
                float(max(0.0, np.nanmax(errors) - errors[0])),
            )
        cube_errors = np.asarray(
            [
                _finite_float(rows[index].get("cube_target_dist_m"))
                for index in range(start, end + 1)
            ]
        )
        if cube_errors.size and math.isfinite(cube_errors[0]):
            cube_goal_regression_m = max(
                cube_goal_regression_m,
                float(max(0.0, np.nanmax(cube_errors) - cube_errors[0])),
            )
        onset_ee = ee_positions[start]
        onset_cube = cube_positions[start]
        if onset_ee is not None:
            ee_displacement_m = max(
                ee_displacement_m,
                max(
                    (
                        float(np.linalg.norm(value - onset_ee))
                        for value in ee_positions[start : end + 1]
                        if value is not None
                    ),
                    default=0.0,
                ),
            )
        if onset_cube is not None:
            cube_displacement_m = max(
                cube_displacement_m,
                max(
                    (
                        float(np.linalg.norm(value - onset_cube))
                        for value in cube_positions[start : end + 1]
                        if value is not None
                    ),
                    default=0.0,
                ),
            )
        gaps = np.asarray(
            [
                _finite_float(rows[index].get("post_surface_gap_m"))
                for index in range(start, end + 1)
            ]
        )
        if gaps.size and math.isfinite(gaps[0]):
            surface_gap_increase_m = max(
                surface_gap_increase_m,
                float(max(0.0, np.nanmax(gaps) - gaps[0])),
            )

    whole_mask = masks["WHOLE_TASK"]
    response_phase_row = next(
        (row for row in window_rows if row["window"] == "CBF_ACTIVE"), {}
    )
    first_response_indices = np.flatnonzero(masks["CBF_ACTIVE"])
    first_response_step = -1
    first_response_pre_state_fingerprint = ""
    first_response_prefix_digest = ""
    if first_response_indices.size:
        first_response_index = int(first_response_indices[0])
        first_response_step = _integer(
            rows[first_response_index].get("step"), first_response_index + 1
        )
        first_response_pre_state_fingerprint = str(
            rows[first_response_index].get("pre_step_state_fingerprint", "")
        )
        prefix_fingerprints = [
            str(row.get("pre_step_state_fingerprint", ""))
            for row in rows[: first_response_index + 1]
        ]
        if prefix_fingerprints and all(prefix_fingerprints):
            prefix_hasher = hashlib.sha256()
            for fingerprint in prefix_fingerprints:
                prefix_hasher.update((fingerprint + "\n").encode("utf-8"))
            first_response_prefix_digest = prefix_hasher.hexdigest()
    completion_time = _finite_float(episode.get("completion_time_s"))
    if not math.isfinite(completion_time):
        completion_time = float(len(rows) * dt_s)
    configured_margin = _first_finite(
        rows[0] if rows else {}, ("configured_safe_gap_m",)
    )
    if not math.isfinite(configured_margin):
        configured_margin = _first_finite(episode, ("configured_safe_gap_m",))
    if not math.isfinite(configured_margin):
        configured_margin = trace.configured_safe_gap_m
    if any("logged_surface_gap_below_configured_margin" in row for row in rows):
        configured_margin_steps = sum(
            _boolean(row.get("logged_surface_gap_below_configured_margin"))
            for row in rows
        )
    elif "logged_surface_gap_below_configured_margin_steps" in episode:
        configured_margin_steps = _integer(
            episode.get("logged_surface_gap_below_configured_margin_steps"), 0
        )
    elif math.isfinite(configured_margin):
        configured_margin_steps = sum(
            math.isfinite(_finite_float(row.get("post_surface_gap_m")))
            and _finite_float(row.get("post_surface_gap_m")) < configured_margin
            for row in rows
        )
    else:
        configured_margin_steps = 0
    if any("logged_surface_gap_below_2cm" in row for row in rows):
        below_2cm_steps = sum(
            _boolean(row.get("logged_surface_gap_below_2cm")) for row in rows
        )
    elif "logged_surface_gap_below_2cm_steps" in episode:
        below_2cm_steps = _integer(episode.get("logged_surface_gap_below_2cm_steps"), 0)
    else:
        below_2cm_steps = sum(
            math.isfinite(_finite_float(row.get("post_surface_gap_m")))
            and _finite_float(row.get("post_surface_gap_m")) < 0.02
            for row in rows
        )
    minimum_gap = min(
        (
            _finite_float(row.get("post_surface_gap_m"))
            for row in rows
            if math.isfinite(_finite_float(row.get("post_surface_gap_m")))
        ),
        default=_finite_float(
            episode.get("min_surface_gap", episode.get("min_hand_gripper_surface_gap"))
        ),
    )
    terminal_reason = str(episode.get("task_terminal_reason", ""))
    horizon_timeout = _boolean(episode.get("truncated")) or any(
        token in terminal_reason.lower() for token in ("horizon", "max_episode", "1200")
    )
    static_valid_steps = sum(_boolean(row.get("static_geometry_valid")) for row in rows)
    self_valid_steps = sum(_boolean(row.get("self_geometry_valid")) for row in rows)
    static_logging_available = static_valid_steps > 0
    self_logging_available = self_valid_steps > 0
    static_contact_pairs, static_pair_identity_available = _contact_pairs(
        rows,
        collision_field="static_collision",
        first_field="static_closest_robot_collider",
        second_field="static_closest_environment_collider",
        undirected=False,
    )
    self_contact_pairs, self_pair_identity_available = _contact_pairs(
        rows,
        collision_field="self_collision",
        first_field="self_closest_first_collider",
        second_field="self_closest_second_collider",
        undirected=True,
    )
    static_contact_classes = sorted(
        {_static_contact_class(pair) for pair in static_contact_pairs}
    )
    result: dict[str, Any] = {
        "condition": trace.condition,
        "pair_key": trace.pair_key,
        "episode": _integer(episode.get("episode"), -1),
        "seed": _integer(episode.get("seed"), 0),
        "encounter_id": str(episode.get("encounter_id", "")),
        "task_phase": _response_task_phase(rows, masks),
        "success": _boolean(episode.get("success")),
        "steps": _integer(episode.get("steps"), len(rows)),
        "completion_time_s": completion_time,
        "horizon_timeout": horizon_timeout,
        "failure_reason": terminal_reason,
        "grasp_success": _boolean(episode.get("grasped_any")),
        "release_success": _boolean(episode.get("released_after_grasp")),
        "object_drop": _boolean(episode.get("object_drop")),
        "human_collision": _boolean(episode.get("collision"))
        or any(_boolean(row.get("human_collision")) for row in rows),
        "near_miss": _integer(episode.get("near_miss_count"), 0) > 0
        or any(_boolean(row.get("near_miss")) for row in rows),
        "static_collision": _boolean(episode.get("static_collision_episode"))
        or any(_boolean(row.get("static_collision")) for row in rows),
        "self_collision": _boolean(episode.get("self_collision_episode"))
        or any(_boolean(row.get("self_collision")) for row in rows),
        "static_collision_steps": max(
            _integer(episode.get("static_collision_steps"), 0),
            sum(_boolean(row.get("static_collision")) for row in rows),
        ),
        "self_collision_steps": max(
            _integer(episode.get("self_collision_steps"), 0),
            sum(_boolean(row.get("self_collision")) for row in rows),
        ),
        "static_contact_pairs": static_contact_pairs,
        "static_contact_classes": static_contact_classes,
        "self_contact_pairs": self_contact_pairs,
        "static_contact_pair_identity_available": static_pair_identity_available,
        "self_contact_pair_identity_available": self_pair_identity_available,
        "static_logging_available": static_logging_available,
        "self_logging_available": self_logging_available,
        "static_geometry_valid_steps": static_valid_steps,
        "self_geometry_valid_steps": self_valid_steps,
        "minimum_logged_surface_gap_m": minimum_gap,
        "logged_surface_gap_below_2cm_steps": below_2cm_steps,
        "logged_surface_gap_below_configured_margin_steps": configured_margin_steps,
        "configured_safe_gap_m": configured_margin,
        "minimum_ttc_s": _finite_float(episode.get("minimum_ttc_s")),
        "response_present": bool(np.any(response_mask)),
        "intervention_steps": int(np.count_nonzero(masks["CBF_ACTIVE"])),
        "intervention_duration_s": float(np.count_nonzero(masks["CBF_ACTIVE"]) * dt_s),
        "recovery_duration_s": float(
            sum(
                _boolean(row.get("state_aware_recovery_control_authority"))
                or _boolean(row.get("state_aware_recovery_active"))
                for row in rows
            )
            * dt_s
        ),
        "recovery_internal_timeout": _integer(
            episode.get("state_aware_recovery_timeout_count"), 0
        )
        > 0,
        "solver_infeasible_steps": sum(
            str(row.get("physical_safety_status", ""))
            == "fallback_stop_infeasible"
            or _boolean(_diagnostic_payload(row).get("infeasibility_proven"))
            for row in rows
        ),
        "solver_fallback_steps": sum(
            str(row.get("physical_safety_status", ""))
            == "fallback_stop_infeasible"
            or _boolean(_diagnostic_payload(row).get("objective_solver_fallback"))
            for row in rows
        ),
        "invalid_hand_fail_closed_steps": sum(
            str(row.get("physical_safety_status", ""))
            == "fallback_stop_invalid_active_hand"
            for row in rows
        ),
        "all_fail_closed_stop_steps": sum(
            _boolean(row.get("physical_safety_fallback_applied")) for row in rows
        ),
        "integrated_intervention_rad": float(np.sum(intervention_norm) * dt_s),
        "correction_total_variation_radps": correction_tv,
        "correction_direction_change_rad": correction_direction_change,
        "correction_variation_source": correction_source,
        "ee_path_length_m": _path_length(ee_positions, whole_mask),
        "response_ee_path_length_m": _path_length(ee_positions, response_mask),
        "retreat_distance_m": retreat_distance_m,
        "retreat_definition": (
            "max positive increase from each response onset in EE-to-cube error "
            "if initially ungrasped, otherwise cube-to-target error"
        ),
        "cube_goal_regression_m": cube_goal_regression_m,
        "max_ee_displacement_from_response_onset_m": ee_displacement_m,
        "max_cube_displacement_from_response_onset_m": cube_displacement_m,
        "max_surface_gap_increase_from_response_onset_m": surface_gap_increase_m,
        "cbf_ee_jerk_rms_mps3": response_phase_row.get("ee_jerk_rms_mps3", math.nan),
        "cbf_ee_jerk_peak_mps3": response_phase_row.get("ee_jerk_peak_mps3", math.nan),
        "cbf_joint_jerk_rms_radps3": response_phase_row.get(
            "joint_jerk_rms_radps3", math.nan
        ),
        "cbf_joint_jerk_peak_radps3": response_phase_row.get(
            "joint_jerk_peak_radps3", math.nan
        ),
        "physics_dt_s": dt_s,
        "initial_state_fingerprint": str(episode.get("initial_state_fingerprint", "")),
        "response_branch_state_fingerprint": str(
            episode.get("response_branch_state_fingerprint", "")
        ),
        "response_branch_step": _integer(episode.get("response_branch_step"), -1),
        "exact_prefix_digest_through_branch": str(
            episode.get("exact_prefix_digest_through_branch", "")
        ),
        "first_applied_response_step": first_response_step,
        "first_applied_response_pre_state_fingerprint": (
            first_response_pre_state_fingerprint
        ),
        "first_applied_response_prefix_digest": first_response_prefix_digest,
        "source_json": trace.source_json,
        "source_steps_csv": trace.source_steps_csv,
    }
    result["absolute_contact_free_episode"] = bool(
        not result["human_collision"]
        and not result["static_collision"]
        and not result["self_collision"]
    )
    # Static/self contact is also judged against BC-only after pairing.  Some
    # qualified legacy scenes contain persistent gripper/table contact in every
    # condition, so absolute incidence is retained but is not silently treated
    # as a response-induced failure.
    result["physical_safety_episode_qualified"] = bool(
        not result["human_collision"]
        and below_2cm_steps == 0
        and static_logging_available
        and self_logging_available
    )
    result["response_candidate_episode_qualified"] = bool(
        result["physical_safety_episode_qualified"]
        and result["success"]
        and not result["horizon_timeout"]
        and not result["recovery_internal_timeout"]
        and result["solver_infeasible_steps"] == 0
        and result["solver_fallback_steps"] == 0
    )
    return result, window_rows


def _response_task_phase(
    rows: Sequence[Mapping[str, Any]], masks: Mapping[str, np.ndarray]
) -> str:
    active = masks["CBF_ACTIVE"]
    indices = np.flatnonzero(active)
    if indices.size == 0:
        indices = np.flatnonzero(masks["_RESPONSE"])
    if indices.size == 0:
        return str(rows[0].get("task_phase_after", "none")) if rows else "none"
    row = rows[int(indices[0])]
    explicit = str(row.get("task_phase_before", row.get("task_phase_after", "")))
    if explicit:
        return explicit
    event = _integer(row.get("controller_event_before"), 0)
    if event <= 0:
        return "approach_cube"
    if event <= 3:
        return "grasp_cube"
    if event <= 6:
        return "move_to_target"
    return "release_cube"


def _aligned_bc_progress(
    bc_rows: Sequence[Mapping[str, Any]], response_rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    bc_by_step = {
        _integer(row.get("step"), index + 1): task_progress_v1(row)
        for index, row in enumerate(bc_rows)
    }
    sorted_steps = sorted(bc_by_step)
    if not sorted_steps:
        return np.full(len(response_rows), np.nan)
    final_progress = bc_by_step[sorted_steps[-1]]
    result: list[float] = []
    for index, row in enumerate(response_rows):
        step = _integer(row.get("step"), index + 1)
        if step in bc_by_step:
            result.append(bc_by_step[step])
            continue
        prior = [item for item in sorted_steps if item <= step]
        result.append(bc_by_step[prior[-1]] if prior else final_progress)
    return np.asarray(result, dtype=float)


def add_paired_metrics(
    summaries: list[dict[str, Any]],
    traces: Sequence[EpisodeTrace],
    baseline_condition: str,
    resumed_duration_s: float,
) -> None:
    trace_index = {(trace.condition, trace.pair_key): trace for trace in traces}
    summary_index = {(row["condition"], row["pair_key"]): row for row in summaries}
    bc_keys = {
        row["pair_key"] for row in summaries if row["condition"] == baseline_condition
    }
    for row in summaries:
        if row["condition"] == baseline_condition:
            row.update(
                {
                    "paired_valid": True,
                    "paired_baseline_condition": baseline_condition,
                    "delta_path_m": 0.0,
                    "delta_completion_time_s": 0.0,
                    "delta_progress_variant_minus_bc_final": 0.0,
                    "progress_deficit_response_auc_s": 0.0,
                    "progress_deficit_response_max": 0.0,
                    "progress_deficit_response_mean": 0.0,
                    "delta_static_collision_steps": 0,
                    "delta_self_collision_steps": 0,
                    "increased_static_collision_steps_vs_bc": False,
                    "increased_self_collision_steps_vs_bc": False,
                    "new_static_collision_episode_vs_bc": False,
                    "new_self_collision_episode_vs_bc": False,
                    "new_static_contact_pairs_vs_bc": [],
                    "new_self_contact_pairs_vs_bc": [],
                    "raw_new_static_contact_pair_vs_bc": False,
                    "new_static_contact_classes_vs_bc": [],
                    "new_static_contact_class_vs_bc": False,
                    "new_static_contact_pair_vs_bc": False,
                    "new_self_contact_pair_vs_bc": False,
                    "new_static_collision_vs_bc": False,
                    "new_self_collision_vs_bc": False,
                }
            )
            continue
        key = row["pair_key"]
        bc_summary = summary_index.get((baseline_condition, key))
        response_trace = trace_index.get((row["condition"], key))
        bc_trace = trace_index.get((baseline_condition, key))
        paired_valid = bool(
            key in bc_keys
            and bc_summary is not None
            and response_trace is not None
            and bc_trace is not None
        )
        row["paired_valid"] = paired_valid
        row["paired_baseline_condition"] = baseline_condition
        if not paired_valid:
            for name in (
                "delta_path_m",
                "delta_completion_time_s",
                "delta_progress_variant_minus_bc_final",
                "progress_deficit_response_auc_s",
                "progress_deficit_response_max",
                "progress_deficit_response_mean",
                "delta_static_collision_steps",
                "delta_self_collision_steps",
            ):
                row[name] = math.nan
            row["new_static_collision_vs_bc"] = False
            row["new_self_collision_vs_bc"] = False
            row["increased_static_collision_steps_vs_bc"] = False
            row["increased_self_collision_steps_vs_bc"] = False
            row["new_static_collision_episode_vs_bc"] = False
            row["new_self_collision_episode_vs_bc"] = False
            row["new_static_contact_pairs_vs_bc"] = []
            row["new_self_contact_pairs_vs_bc"] = []
            row["raw_new_static_contact_pair_vs_bc"] = False
            row["new_static_contact_classes_vs_bc"] = []
            row["new_static_contact_class_vs_bc"] = False
            row["new_static_contact_pair_vs_bc"] = False
            row["new_self_contact_pair_vs_bc"] = False
            row["physical_safety_episode_qualified"] = False
            row["response_candidate_episode_qualified"] = False
            continue
        row["delta_path_m"] = float(
            row["ee_path_length_m"] - bc_summary["ee_path_length_m"]
        )
        row["delta_completion_time_s"] = float(
            row["completion_time_s"] - bc_summary["completion_time_s"]
        )
        row["delta_static_collision_steps"] = int(
            row["static_collision_steps"] - bc_summary["static_collision_steps"]
        )
        row["delta_self_collision_steps"] = int(
            row["self_collision_steps"] - bc_summary["self_collision_steps"]
        )
        row["increased_static_collision_steps_vs_bc"] = bool(
            row["delta_static_collision_steps"] > 0
        )
        row["increased_self_collision_steps_vs_bc"] = bool(
            row["delta_self_collision_steps"] > 0
        )
        row["new_static_collision_episode_vs_bc"] = bool(
            row["static_collision"] and not bc_summary["static_collision"]
        )
        row["new_self_collision_episode_vs_bc"] = bool(
            row["self_collision"] and not bc_summary["self_collision"]
        )
        new_static_pairs = sorted(
            set(row.get("static_contact_pairs", ()))
            - set(bc_summary.get("static_contact_pairs", ()))
        )
        new_self_pairs = sorted(
            set(row.get("self_contact_pairs", ()))
            - set(bc_summary.get("self_contact_pairs", ()))
        )
        new_static_classes = sorted(
            set(row.get("static_contact_classes", ()))
            - set(bc_summary.get("static_contact_classes", ()))
        )
        row["new_static_contact_pairs_vs_bc"] = new_static_pairs
        row["new_self_contact_pairs_vs_bc"] = new_self_pairs
        row["raw_new_static_contact_pair_vs_bc"] = bool(new_static_pairs)
        row["new_static_contact_classes_vs_bc"] = new_static_classes
        row["new_static_contact_class_vs_bc"] = bool(new_static_classes)
        # Backward-compatible boolean uses semantic contact equivalence. The
        # exact raw path set remains in new_static_contact_pairs_vs_bc.
        row["new_static_contact_pair_vs_bc"] = bool(new_static_classes)
        row["new_self_contact_pair_vs_bc"] = bool(new_self_pairs)
        # Backward-compatible broad aliases now mean a genuinely new collision
        # episode or collider pair, never a duration-only increase.
        row["new_static_collision_vs_bc"] = bool(
            row["new_static_collision_episode_vs_bc"]
            or row["new_static_contact_pair_vs_bc"]
        )
        row["new_self_collision_vs_bc"] = bool(
            row["new_self_collision_episode_vs_bc"]
            or row["new_self_contact_pair_vs_bc"]
        )
        row["physical_safety_episode_qualified"] = bool(
            row["physical_safety_episode_qualified"]
            and not row["new_static_collision_vs_bc"]
            and not row["new_self_collision_vs_bc"]
        )
        row["response_candidate_episode_qualified"] = bool(
            row["physical_safety_episode_qualified"]
            and row["success"]
            and not row["horizon_timeout"]
            and not row["recovery_internal_timeout"]
            and row["solver_infeasible_steps"] == 0
            and row["solver_fallback_steps"] == 0
        )
        bc_progress = _aligned_bc_progress(bc_trace.steps, response_trace.steps)
        response_progress = np.asarray(
            [task_progress_v1(item) for item in response_trace.steps], dtype=float
        )
        masks = response_window_masks(
            response_trace.steps, response_trace.physics_dt_s, resumed_duration_s
        )
        response_mask = masks["_RESPONSE"]
        deficits = np.maximum(0.0, bc_progress - response_progress)
        selected = deficits[response_mask]
        row["delta_progress_variant_minus_bc_final"] = float(
            response_progress[-1] - bc_progress[-1]
        )
        row["progress_deficit_response_auc_s"] = float(
            np.sum(selected) * response_trace.physics_dt_s
        )
        row["progress_deficit_response_max"] = (
            float(np.max(selected)) if selected.size else 0.0
        )
        row["progress_deficit_response_mean"] = (
            float(np.mean(selected)) if selected.size else 0.0
        )


def pareto_flags(points: np.ndarray) -> np.ndarray:
    """Return non-dominated flags for finite minimization objectives."""

    values = np.asarray(points, dtype=float)
    flags = np.zeros(len(values), dtype=bool)
    finite = np.all(np.isfinite(values), axis=1)
    for index in np.flatnonzero(finite):
        candidate = values[index]
        dominated = False
        for other_index in np.flatnonzero(finite):
            if other_index == index:
                continue
            other = values[other_index]
            if np.all(other <= candidate) and np.any(other < candidate):
                dominated = True
                break
        flags[index] = not dominated
    return flags


PARETO_AXES = {
    "path_vs_jerk": ("delta_path_m", "cbf_ee_jerk_rms_mps3"),
    "intervention_vs_recovery": (
        "integrated_intervention_rad",
        "recovery_duration_s",
    ),
    "retreat_vs_completion": ("retreat_distance_m", "delta_completion_time_s"),
    "progress_vs_jerk": (
        "progress_deficit_response_auc_s",
        "cbf_ee_jerk_rms_mps3",
    ),
}

TRADEOFF_FIELDS = (
    "delta_path_m",
    "delta_completion_time_s",
    "integrated_intervention_rad",
    "progress_deficit_response_auc_s",
    "progress_deficit_response_max",
    "retreat_distance_m",
    "recovery_duration_s",
    "cbf_ee_jerk_rms_mps3",
    "cbf_joint_jerk_rms_radps3",
    "correction_total_variation_radps",
    "delta_static_collision_steps",
    "delta_self_collision_steps",
)


def mark_pareto(summaries: list[dict[str, Any]], baseline_condition: str) -> None:
    eligible_indices = [
        index
        for index, row in enumerate(summaries)
        if row["condition"] != baseline_condition
        and _boolean(row.get("paired_valid"))
        and _boolean(row.get("response_present"))
        and _boolean(row.get("exact_response_pairing_verified"))
        and _boolean(row.get("response_candidate_episode_qualified"))
    ]
    for name, (x_field, y_field) in PARETO_AXES.items():
        values = np.asarray(
            [
                [
                    _finite_float(summaries[index].get(x_field)),
                    _finite_float(summaries[index].get(y_field)),
                ]
                for index in eligible_indices
            ],
            dtype=float,
        ).reshape((-1, 2))
        flags = pareto_flags(values) if len(values) else np.zeros(0, dtype=bool)
        for row in summaries:
            row[f"pareto_{name}"] = False
        for index, flag in zip(eligible_indices, flags):
            summaries[index][f"pareto_{name}"] = bool(flag)

    overall_fields = (
        "delta_path_m",
        "progress_deficit_response_auc_s",
        "cbf_ee_jerk_rms_mps3",
        "correction_total_variation_radps",
        "recovery_duration_s",
    )
    values = np.asarray(
        [
            [_finite_float(summaries[index].get(field)) for field in overall_fields]
            for index in eligible_indices
        ],
        dtype=float,
    ).reshape((-1, len(overall_fields)))
    flags = pareto_flags(values) if len(values) else np.zeros(0, dtype=bool)
    for row in summaries:
        row["pareto_overall_task_motion"] = False
    for index, flag in zip(eligible_indices, flags):
        summaries[index]["pareto_overall_task_motion"] = bool(flag)


def pairing_verification(
    rows: Sequence[Mapping[str, Any]], baseline_condition: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit initial-state and first-active-prefix evidence for every pair.

    The first-applied-response proof applies to the response families (A/B/C),
    since a BC-only rollout has no CBF response branch. Source-key alignment
    alone is never labelled exact. A pair is verified only when at least two
    response conditions provide identical, non-empty initial, pre-response
    state, response-step and prefix-digest evidence. Legacy artifacts that do
    not log per-step state fingerprints fall back to their first-active-
    constraint episode fingerprints and expose that evidence source.
    """

    audits: list[dict[str, Any]] = []
    for pair_key in sorted({str(row["pair_key"]) for row in rows}):
        selected = [row for row in rows if row["pair_key"] == pair_key]
        response_rows = [
            row
            for row in selected
            if row["condition"] != baseline_condition
            and _boolean(row.get("response_present"))
        ]
        initial_values = [
            str(row.get("initial_state_fingerprint", "")) for row in response_rows
        ]
        applied_evidence_complete = all(
            str(row.get("first_applied_response_pre_state_fingerprint", ""))
            and str(row.get("first_applied_response_prefix_digest", ""))
            and _integer(row.get("first_applied_response_step"), -1) >= 0
            for row in response_rows
        )
        evidence_source = (
            "first_applied_response_step_log_v1"
            if response_rows and applied_evidence_complete
            else "legacy_first_active_constraint_episode_v1"
        )
        if evidence_source == "first_applied_response_step_log_v1":
            branch_values = [
                str(row.get("first_applied_response_pre_state_fingerprint", ""))
                for row in response_rows
            ]
            branch_steps = [
                _integer(row.get("first_applied_response_step"), -1)
                for row in response_rows
            ]
            prefix_values = [
                str(row.get("first_applied_response_prefix_digest", ""))
                for row in response_rows
            ]
        else:
            branch_values = [
                str(row.get("response_branch_state_fingerprint", ""))
                for row in response_rows
            ]
            branch_steps = [
                _integer(row.get("response_branch_step"), -1)
                for row in response_rows
            ]
            prefix_values = [
                str(row.get("exact_prefix_digest_through_branch", ""))
                for row in response_rows
            ]
        enough = len(response_rows) >= 2
        initial_match = bool(
            enough and all(initial_values) and len(set(initial_values)) == 1
        )
        branch_match = bool(
            enough and all(branch_values) and len(set(branch_values)) == 1
        )
        branch_step_match = bool(
            enough
            and all(step >= 0 for step in branch_steps)
            and len(set(branch_steps)) == 1
        )
        prefix_match = bool(
            enough and all(prefix_values) and len(set(prefix_values)) == 1
        )
        verified = bool(
            initial_match and branch_match and branch_step_match and prefix_match
        )
        audits.append(
            {
                "pair_key": pair_key,
                "encounter_id": str(selected[0].get("encounter_id", "")),
                "response_condition_count": len(response_rows),
                "response_conditions": sorted(
                    str(row["condition"]) for row in response_rows
                ),
                "pairing_evidence_source": evidence_source,
                "initial_state_fingerprint_match": initial_match,
                "response_branch_state_fingerprint_match": branch_match,
                "response_branch_step_match": branch_step_match,
                "exact_prefix_digest_match": prefix_match,
                "exact_response_pairing_verified": verified,
                "evidence_complete": bool(
                    enough
                    and all(initial_values)
                    and all(branch_values)
                    and all(prefix_values)
                    and all(step >= 0 for step in branch_steps)
                ),
            }
        )
    eligible = [audit for audit in audits if audit["response_condition_count"] >= 2]
    summary = {
        "schema_version": "paired_fingerprint_verification_v2",
        "pair_count": len(audits),
        "verifiable_response_pair_count": len(eligible),
        "verified_response_pair_count": sum(
            _boolean(audit["exact_response_pairing_verified"]) for audit in eligible
        ),
        "all_response_pairs_exactly_verified": bool(
            eligible
            and all(
                _boolean(audit["exact_response_pairing_verified"]) for audit in eligible
            )
        ),
        "claim_boundary": (
            "Exact pairing is claimed only for A/B/C response families with "
            "matching initial-state, first-applied-response pre-state, response-"
            "step, and prefix-digest fingerprints. Legacy artifacts fall back "
            "to first-active-constraint evidence. BC-only remains a source-"
            "aligned counterfactual unless separate branch-state proof is supplied."
        ),
    }
    return audits, summary


def annotate_exact_response_pairing(
    rows: Sequence[dict[str, Any]], baseline_condition: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach the response-family fingerprint verdict to every episode row.

    Pair-key alignment is enough to compute a diagnostic BC counterfactual, but
    it is not enough to claim an exact A/B/C branch comparison.  Downstream
    Pareto plots and response-family aggregates therefore consume only rows
    whose first active branch is fingerprint-identical across the response
    families.  Raw episode and qualification tables still retain every run.
    """

    audits, summary = pairing_verification(rows, baseline_condition)
    verified_by_key = {
        str(audit["pair_key"]): _boolean(
            audit.get("exact_response_pairing_verified")
        )
        for audit in audits
    }
    for row in rows:
        row["exact_response_pairing_verified"] = bool(
            verified_by_key.get(str(row.get("pair_key", "")), False)
        )
    return audits, summary


def qualification_summary(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        minimum_gaps = np.asarray(
            [_finite_float(row.get("minimum_logged_surface_gap_m")) for row in selected]
        )
        finite_gaps = minimum_gaps[np.isfinite(minimum_gaps)]
        paired_rows = [row for row in selected if _boolean(row.get("paired_valid"))]
        result[condition] = {
            "episodes": len(selected),
            "task_success_count": sum(_boolean(row.get("success")) for row in selected),
            "human_collision_episode_count": sum(
                _boolean(row.get("human_collision")) for row in selected
            ),
            "static_collision_episode_count": sum(
                _boolean(row.get("static_collision")) for row in selected
            ),
            "static_collision_steps_total": sum(
                _integer(row.get("static_collision_steps"), 0) for row in selected
            ),
            "self_collision_episode_count": sum(
                _boolean(row.get("self_collision")) for row in selected
            ),
            "self_collision_steps_total": sum(
                _integer(row.get("self_collision_steps"), 0) for row in selected
            ),
            "near_miss_episode_count": sum(
                _boolean(row.get("near_miss")) for row in selected
            ),
            "logged_gap_below_2cm_episode_count": sum(
                _integer(row.get("logged_surface_gap_below_2cm_steps"), 0) > 0
                for row in selected
            ),
            "logged_gap_below_configured_margin_episode_count": sum(
                _integer(row.get("logged_surface_gap_below_configured_margin_steps"), 0)
                > 0
                for row in selected
            ),
            "global_min_logged_surface_gap_m": (
                float(np.min(finite_gaps)) if finite_gaps.size else math.nan
            ),
            "mean_episode_min_surface_gap_m": (
                float(np.mean(finite_gaps)) if finite_gaps.size else math.nan
            ),
            "global_horizon_timeout_count": sum(
                _boolean(row.get("horizon_timeout")) for row in selected
            ),
            "recovery_internal_timeout_count": sum(
                _boolean(row.get("recovery_internal_timeout")) for row in selected
            ),
            "object_drop_episode_count": sum(
                _boolean(row.get("object_drop")) for row in selected
            ),
            "response_episode_count": sum(
                _boolean(row.get("response_present")) for row in selected
            ),
            "physical_safety_qualified_episode_count": sum(
                _boolean(row.get("physical_safety_episode_qualified"))
                for row in selected
            ),
            "response_candidate_qualified_episode_count": sum(
                _boolean(row.get("response_present"))
                and _boolean(row.get("response_candidate_episode_qualified"))
                for row in selected
            ),
            "solver_infeasible_episode_count": sum(
                _integer(row.get("solver_infeasible_steps"), 0) > 0 for row in selected
            ),
            "solver_fallback_episode_count": sum(
                _integer(row.get("solver_fallback_steps"), 0) > 0 for row in selected
            ),
            "invalid_hand_fail_closed_episode_count": sum(
                _integer(row.get("invalid_hand_fail_closed_steps"), 0) > 0
                for row in selected
            ),
            "invalid_hand_fail_closed_steps_total": sum(
                _integer(row.get("invalid_hand_fail_closed_steps"), 0)
                for row in selected
            ),
            "all_fail_closed_stop_steps_total": sum(
                _integer(row.get("all_fail_closed_stop_steps"), 0)
                for row in selected
            ),
            "static_logger_available_all_episodes": all(
                _boolean(row.get("static_logging_available")) for row in selected
            ),
            "self_logger_available_all_episodes": all(
                _boolean(row.get("self_logging_available")) for row in selected
            ),
            "paired_static_collision_step_increase_total": sum(
                max(0, _integer(row.get("delta_static_collision_steps"), 0))
                for row in paired_rows
            ),
            "paired_self_collision_step_increase_total": sum(
                max(0, _integer(row.get("delta_self_collision_steps"), 0))
                for row in paired_rows
            ),
            "paired_static_collision_step_delta_net": sum(
                _integer(row.get("delta_static_collision_steps"), 0)
                for row in paired_rows
            ),
            "paired_self_collision_step_delta_net": sum(
                _integer(row.get("delta_self_collision_steps"), 0)
                for row in paired_rows
            ),
            "paired_new_static_collision_episode_count": sum(
                _boolean(row.get("new_static_collision_episode_vs_bc"))
                for row in paired_rows
            ),
            "paired_new_self_collision_episode_count": sum(
                _boolean(row.get("new_self_collision_episode_vs_bc"))
                for row in paired_rows
            ),
            "paired_increased_static_collision_steps_episode_count": sum(
                _boolean(row.get("increased_static_collision_steps_vs_bc"))
                for row in paired_rows
            ),
            "paired_increased_self_collision_steps_episode_count": sum(
                _boolean(row.get("increased_self_collision_steps_vs_bc"))
                for row in paired_rows
            ),
            "paired_new_static_contact_pair_episode_count": sum(
                _boolean(row.get("new_static_contact_pair_vs_bc"))
                for row in paired_rows
            ),
            "paired_raw_new_static_contact_pair_episode_count": sum(
                _boolean(row.get("raw_new_static_contact_pair_vs_bc"))
                for row in paired_rows
            ),
            "paired_new_self_contact_pair_episode_count": sum(
                _boolean(row.get("new_self_contact_pair_vs_bc")) for row in paired_rows
            ),
        }
        summary = result[condition]
        summary["physical_safety_qualification_pass"] = bool(
            summary["human_collision_episode_count"] == 0
            and summary["logged_gap_below_2cm_episode_count"] == 0
            and summary["static_logger_available_all_episodes"]
            and summary["self_logger_available_all_episodes"]
            and summary["paired_new_static_collision_episode_count"] == 0
            and summary["paired_new_self_collision_episode_count"] == 0
            and summary["paired_new_static_contact_pair_episode_count"] == 0
            and summary["paired_new_self_contact_pair_episode_count"] == 0
            and summary["solver_infeasible_episode_count"] == 0
            and summary["solver_fallback_episode_count"] == 0
        )
    return result


def condition_tradeoff_summary(
    rows: Sequence[Mapping[str, Any]], baseline_condition: str
) -> list[dict[str, Any]]:
    """Aggregate paired response metrics without hiding episode-level values."""

    result: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        paired = [row for row in selected if _boolean(row.get("paired_valid"))]
        raw_response = [
            row for row in paired if _boolean(row.get("response_present"))
        ]
        response = [
            row
            for row in raw_response
            if _boolean(row.get("exact_response_pairing_verified"))
        ]
        aggregate: dict[str, Any] = {
            "condition": condition,
            "is_baseline": condition == baseline_condition,
            "episode_count": len(selected),
            "paired_episode_count": len(paired),
            "response_episode_count": len(response),
            "raw_response_episode_count": len(raw_response),
            "excluded_nonexact_response_episode_count": len(raw_response)
            - len(response),
            "qualified_response_episode_count": sum(
                _boolean(row.get("response_candidate_episode_qualified"))
                for row in response
            ),
            "new_static_collision_episode_vs_bc_count": sum(
                _boolean(row.get("new_static_collision_episode_vs_bc"))
                for row in response
            ),
            "new_self_collision_episode_vs_bc_count": sum(
                _boolean(row.get("new_self_collision_episode_vs_bc"))
                for row in response
            ),
            "new_static_contact_pair_vs_bc_count": sum(
                _boolean(row.get("new_static_contact_pair_vs_bc")) for row in response
            ),
            "raw_new_static_contact_pair_vs_bc_count": sum(
                _boolean(row.get("raw_new_static_contact_pair_vs_bc"))
                for row in response
            ),
            "new_self_contact_pair_vs_bc_count": sum(
                _boolean(row.get("new_self_contact_pair_vs_bc")) for row in response
            ),
            "increased_static_collision_steps_vs_bc_count": sum(
                _boolean(row.get("increased_static_collision_steps_vs_bc"))
                for row in response
            ),
            "increased_self_collision_steps_vs_bc_count": sum(
                _boolean(row.get("increased_self_collision_steps_vs_bc"))
                for row in response
            ),
        }
        metric_rows = paired if condition == baseline_condition else response
        for field in TRADEOFF_FIELDS:
            values = np.asarray(
                [_finite_float(row.get(field)) for row in metric_rows], dtype=float
            )
            values = values[np.isfinite(values)]
            aggregate[f"{field}_count"] = int(values.size)
            aggregate[f"{field}_mean"] = (
                float(np.mean(values)) if values.size else math.nan
            )
            aggregate[f"{field}_median"] = (
                float(np.median(values)) if values.size else math.nan
            )
            aggregate[f"{field}_p95"] = (
                float(np.percentile(values, 95.0)) if values.size else math.nan
            )
        for name in (*PARETO_AXES, "overall_task_motion"):
            aggregate[f"pareto_{name}_count"] = sum(
                _boolean(row.get(f"pareto_{name}")) for row in response
            )
        result.append(aggregate)
    return result


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_plots(
    output_directory: Path,
    summaries: Sequence[Mapping[str, Any]],
    baseline_condition: str,
    annotate: bool,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [
        row
        for row in summaries
        if row["condition"] != baseline_condition
        and _boolean(row.get("paired_valid"))
        and _boolean(row.get("response_present"))
        and _boolean(row.get("exact_response_pairing_verified"))
    ]
    condition_names = sorted({str(row["condition"]) for row in eligible})
    colors = {
        name: plt.get_cmap("tab10")(index % 10)
        for index, name in enumerate(condition_names)
    }
    markers = {
        "approach_cube": "o",
        "grasp_cube": "s",
        "move_to_target": "^",
        "release_cube": "D",
    }
    labels = {
        "delta_path_m": "Paired ΔPath (m; response - BC-only)",
        "cbf_ee_jerk_rms_mps3": "CBF-active EE RMS jerk (m/s³)",
        "integrated_intervention_rad": "Integrated intervention (rad)",
        "recovery_duration_s": "Recovery duration (s)",
        "retreat_distance_m": "Task-relative retreat (m)",
        "delta_completion_time_s": "Paired ΔCompletionTime (s)",
        "progress_deficit_response_auc_s": "Response progress-deficit AUC (s)",
    }
    paths: list[str] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    for plot_name, (x_field, y_field) in PARETO_AXES.items():
        fig, axis = plt.subplots(figsize=(9.0, 6.0))
        for condition in condition_names:
            condition_rows = [row for row in eligible if row["condition"] == condition]
            for phase, marker in markers.items():
                phase_rows = [
                    row for row in condition_rows if row.get("task_phase") == phase
                ]
                xs = np.asarray([_finite_float(row.get(x_field)) for row in phase_rows])
                ys = np.asarray([_finite_float(row.get(y_field)) for row in phase_rows])
                finite = np.isfinite(xs) & np.isfinite(ys)
                if not np.any(finite):
                    continue
                axis.scatter(
                    xs[finite],
                    ys[finite],
                    c=[colors[condition]],
                    marker=marker,
                    alpha=0.78,
                    edgecolors="black",
                    linewidths=0.35,
                    label=f"{condition} / {phase}",
                )
                if annotate:
                    for x_value, y_value, row in zip(
                        xs[finite],
                        ys[finite],
                        np.asarray(phase_rows, dtype=object)[finite],
                    ):
                        encounter = str(row.get("encounter_id", ""))
                        axis.annotate(
                            f"{condition}:{encounter[:8]}",
                            (x_value, y_value),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6,
                            alpha=0.8,
                        )
        frontier = [
            row
            for row in eligible
            if _boolean(row.get(f"pareto_{plot_name}"))
            and math.isfinite(_finite_float(row.get(x_field)))
            and math.isfinite(_finite_float(row.get(y_field)))
        ]
        frontier.sort(key=lambda row: _finite_float(row.get(x_field)))
        if frontier:
            axis.plot(
                [_finite_float(row.get(x_field)) for row in frontier],
                [_finite_float(row.get(y_field)) for row in frontier],
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label="non-dominated frontier",
            )
        axis.set_xlabel(labels.get(x_field, x_field))
        axis.set_ylabel(labels.get(y_field, y_field))
        axis.set_title(plot_name.replace("_", " ").title())
        axis.grid(True, alpha=0.25)
        handles, legend_labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, legend_labels):
            unique.setdefault(label, handle)
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
        fig.tight_layout()
        path = output_directory / f"pareto_{plot_name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def analyze(
    condition_specs: Mapping[str, Sequence[str]],
    baseline_condition: str = "BC-only",
    resumed_duration_s: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    traces: list[EpisodeTrace] = []
    for condition, specs in condition_specs.items():
        traces.extend(load_condition(condition, specs))
    if baseline_condition not in {trace.condition for trace in traces}:
        raise ValueError(f"Baseline condition {baseline_condition!r} was not loaded")
    summaries: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for trace in traces:
        summary, window_rows = summarize_trace(trace, resumed_duration_s)
        summaries.append(summary)
        windows.extend(window_rows)
    add_paired_metrics(
        summaries, traces, baseline_condition, resumed_duration_s=resumed_duration_s
    )
    annotate_exact_response_pairing(summaries, baseline_condition)
    mark_pareto(summaries, baseline_condition)
    summaries.sort(key=lambda row: (str(row["condition"]), str(row["pair_key"])))
    windows.sort(
        key=lambda row: (
            str(row["condition"]),
            str(row["pair_key"]),
            WINDOWS.index(str(row["window"])),
        )
    )
    return summaries, windows, qualification_summary(summaries)


def _parse_condition_specs(values: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--condition must be NAME=JSON_OR_STEP_PREFIX, got: {value!r}"
            )
        name, spec = value.split("=", 1)
        name, spec = name.strip(), spec.strip()
        if not name or not spec:
            raise ValueError(f"Invalid --condition value: {value!r}")
        result.setdefault(name, []).append(spec)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=JSON_OR_STEP_PREFIX",
        help="Repeat for BC-only, A, B and C; repeat a NAME to aggregate seeds.",
    )
    parser.add_argument(
        "--result-root",
        default="",
        help="Auto-discover <root>/{bc_only|none,A|cbf,B|cbf_task_consistent,C|cbf_smooth_intervention}.",
    )
    parser.add_argument("--baseline-condition", default="BC-only")
    parser.add_argument("--resume-window-s", type=float, default=0.5)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--annotate-points",
        action="store_true",
        help="Write condition and short encounter ID beside every scatter point.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    parsed = _parse_condition_specs(args.condition)
    if args.result_root:
        for condition, spec in _condition_specs_from_root(args.result_root):
            parsed.setdefault(condition, []).append(spec)
    if not parsed:
        raise ValueError("Provide repeated --condition entries or --result-root")
    if args.resume_window_s <= 0.0:
        raise ValueError("--resume-window-s must be positive")
    baseline_condition = args.baseline_condition
    if baseline_condition not in parsed and baseline_condition == "BC-only":
        if "BC_ONLY" in parsed:
            baseline_condition = "BC_ONLY"
    summaries, windows, qualification = analyze(
        parsed,
        baseline_condition=baseline_condition,
        resumed_duration_s=args.resume_window_s,
    )
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_episode_paired_tradeoffs.csv", summaries)
    write_csv(output / "windowed_jerk.csv", windows)
    qualification_rows = [
        {"condition": condition, **values}
        for condition, values in qualification.items()
    ]
    write_csv(output / "qualification_summary.csv", qualification_rows)
    tradeoff_summary = condition_tradeoff_summary(summaries, baseline_condition)
    write_csv(output / "condition_tradeoff_summary.csv", tradeoff_summary)
    pairing_rows, pairing_summary = pairing_verification(summaries, baseline_condition)
    write_csv(output / "pairing_verification.csv", pairing_rows)
    plots = write_plots(
        output,
        summaries,
        baseline_condition=baseline_condition,
        annotate=bool(args.annotate_points),
    )
    input_artifacts = {}
    for condition, specs in sorted(parsed.items()):
        input_artifacts[condition] = []
        for spec in specs:
            json_path, steps_path = _resolve_prefix(spec)
            input_artifacts[condition].append(
                {
                    "result": _artifact_reference(json_path),
                    "steps": _artifact_reference(steps_path),
                }
            )
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "analyzer_source": _artifact_reference(Path(__file__)),
        "input_artifacts": input_artifacts,
        "baseline_condition": baseline_condition,
        "resume_window_s": float(args.resume_window_s),
        "task_progress_definition": (
            "task_progress_v1=min(1,(clip(controller_event,0,8)+"
            "clip(controller_t,0,1))/8); explicit success=1"
        ),
        "progress_deficit_definition": (
            "max(0, paired_BC_only_progress - response_progress), integrated "
            "over the merged CBF/recovery response window"
        ),
        "retreat_definition": (
            "maximum positive task-error regression from each response onset: "
            "EE-to-cube if ungrasped at onset, cube-to-target if grasped"
        ),
        "static_self_contact_comparison_definition": {
            "new_collision_episode": (
                "response condition has collision contact while paired BC-only does not"
            ),
            "new_contact_pair": (
                "a semantic contact class reported on a response collision step "
                "is absent from paired BC-only; left/right Panda finger contact "
                "with /World/table is one gripper_finger/table task-contact class"
            ),
            "raw_contact_pair_diagnostic": (
                "exact collider-path set differences are retained in "
                "new_static_contact_pairs_vs_bc and "
                "raw_new_static_contact_pair_vs_bc but do not alone disqualify"
            ),
            "increased_collision_steps": (
                "response collision-step count exceeds paired BC-only; retained as "
                "duration burden but does not alone disqualify a Pareto candidate"
            ),
            "qualification_rule": (
                "new collision episodes and new reported collider pairs disqualify; "
                "duration-only increases remain visible in the selection table"
            ),
            "solver_vs_measurement_fallback": (
                "fallback_stop_invalid_active_hand is counted separately as a "
                "conservative measurement fail-closed stop; only proven/explicit "
                "constraint infeasibility or objective-solver fallback enters the "
                "solver-infeasible/fallback qualification fields"
            ),
        },
        "window_definition": {
            "PRE_INTERVENTION": f"{args.resume_window_s:g}s before response onset",
            "CBF_ACTIVE": "steps with non-zero CBF intervention",
            "RECOVERY_ACTIVE": "recovery-authority steps excluding CBF-active overlap",
            "BC_RESUMED": f"first {args.resume_window_s:g}s after response end",
            "WHOLE_TASK": "all episode steps",
            "merge_rule": (
                f"response gaps <= {args.resume_window_s:g}s are merged to avoid "
                "classifying one blinking intervention as multiple responses"
            ),
        },
        "pareto_objectives": {
            name: {"x": fields[0], "y": fields[1], "direction": "minimize"}
            for name, fields in PARETO_AXES.items()
        },
        "qualification": qualification,
        "condition_tradeoff_summary": tradeoff_summary,
        "pairing_verification": pairing_summary,
        "pairing_verification_by_encounter": pairing_rows,
        "episode_metrics": summaries,
        "windowed_jerk": windows,
        "plot_files": plots,
    }
    with (output / "tradeoff_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(
            _strict_json(report), stream, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "schema_version": ANALYSIS_SCHEMA,
                "output_directory": str(output),
                "episode_count": len(summaries),
                "qualification": _strict_json(qualification),
                "pairing_verification": _strict_json(pairing_summary),
                "plots": plots,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
