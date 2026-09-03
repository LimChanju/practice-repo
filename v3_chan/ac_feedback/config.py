"""Loader and fail-closed checks for the A/C pilot configuration."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .study import SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ac_selective_smoothing_pilot.yaml"
RUNTIME_CONFIG = (
    PROJECT_ROOT
    / "v3_chan"
    / "runtime_contracts"
    / "ac_selective_smoothing_v1"
    / "runtime_config.json"
)


FROZEN_REALTIME_MARKER_MAPPING: dict[str, dict[str, str]] = {
    "crossing_left": {
        "realtime_safety_concern": "right.a",
        "realtime_behavior_anomaly": "right.b",
    },
    "crossing_right": {
        "realtime_safety_concern": "left.x",
        "realtime_behavior_anomaly": "left.y",
    },
}
FROZEN_QUESTIONNAIRE_NAVIGATION: dict[str, Any] = {
    "previous": "left.x",
    "next": "left.y",
    "select": "right.a",
    "confirm": "right.b",
    "back_chord": ["left.x", "left.y"],
    "emergency_abort_chord": ["right.a", "right.b"],
    "emergency_abort_hold_s": 2.0,
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError(
                "config is not JSON-compatible YAML and PyYAML is unavailable"
            ) from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {resolved}")
    validate_config(value)
    return value


def runtime_contract() -> dict[str, Any]:
    value = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime contract must be a JSON object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    contract = runtime_contract()
    required: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study.pilot": True,
        "study.practice_trials": 4,
        "study.feedback_changes_schedule": False,
        "study.condition_blinding": True,
        "policy.checkpoint": contract["policy"]["path"],
        "policy.sha256": contract["policy"]["sha256"],
        "policy.mode": "frozen_direct_bc",
        "policy.frozen": True,
        "policy.observation_dim": 84,
        "policy.observation_version": "obs_v1_state_controller_phase",
        "policy.action_dim": 5,
        "policy.action_version": "action_v1_controller_target_delta",
        "policy.human_observations_masked": True,
        "policy.online_updates": False,
        "policy.fine_tuning": False,
        "conditions.A_reactive.objective_mode": "joint_nominal",
        "conditions.A_reactive.lambda_s": 0.0,
        "conditions.C_smooth.objective_mode": "smooth_intervention",
        "conditions.C_smooth.lambda_s": 4.0,
        "shared_cbf.controller": "cbf",
        "shared_cbf.safe_gap_m": 0.05,
        "shared_cbf.activation_gap_m": 0.13,
        "shared_cbf.gamma_per_s": 8.0,
        "shared_cbf.prediction_horizon_s": 0.15,
        "shared_cbf.max_prediction_buffer_m": 0.08,
        "shared_cbf.max_joint_speed_rad_s": 2.0,
        "shared_cbf.fail_closed_on_invalid_active_hand": True,
        "shared_cbf.stop_on_infeasible": True,
        "shared_cbf.parameter_adaptation": False,
        "task.cube_count": 1,
        "task.bc_feasible_layout_required": True,
        "task.strict_semantics": "physical_event_driven_pick_place_v5",
        "task.state_aware_recovery": True,
        "task.state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
        "task.recovery_fixed": True,
        "task.pseudo_errp_enabled": False,
        "task.task_progression_paused_during_query": True,
        "task.physics_remains_on_during_query": True,
        "task.cbf_remains_on_during_query": True,
        "task.safe_hold_during_query": True,
        "response_episode.onset_intervention_rad_s": 0.05,
        "response_episode.onset_confirmation_frames": 3,
        "response_episode.stable_intervention_rad_s": 0.01,
        "response_episode.stable_duration_s": 0.5,
        "feedback.haptics_enabled": False,
        "feedback.input_device": "vr_controller",
        "feedback.condition_identity_visible": False,
        "feedback.no_marker_semantics": "missing_evidence_not_acceptable",
        "feedback.uncertain_semantics": "abstain",
        "feedback.timeout_semantics": "missing_abstain",
        "feedback.realtime_markers": FROZEN_REALTIME_MARKER_MAPPING,
        "feedback.questionnaire_navigation": FROZEN_QUESTIONNAIRE_NAVIGATION,
    }
    for path, expected in required.items():
        actual = get_path(config, path)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"frozen config mismatch at {path}: expected {expected!r}, got {actual!r}"
            )
    mode = str(get_path(config, "study.mode"))
    evaluated = int(get_path(config, "study.evaluated_trials"))
    anchors = int(get_path(config, "study.anchor_repeat_trials"))
    expected_counts = {
        "minimal_pilot": (16, 0),
        "pilot_with_anchors": (20, 4),
    }
    if mode not in expected_counts or (evaluated, anchors) != expected_counts[mode]:
        raise ValueError("study mode and trial counts disagree")
    available_modes = get_path(config, "study.available_modes")
    if not isinstance(available_modes, Mapping):
        raise ValueError("study.available_modes must be a mapping")
    normalized_modes = {
        str(name): (
            int(values["evaluated_trials"]),
            int(values["anchor_repeat_trials"]),
        )
        for name, values in available_modes.items()
        if isinstance(values, Mapping)
    }
    if normalized_modes != expected_counts:
        raise ValueError("study.available_modes drift")
    expected_policy = contract["policy"]
    for name in ("observation_dim", "observation_version", "action_dim", "action_version"):
        if get_path(config, f"policy.{name}") != expected_policy[name]:
            raise ValueError(f"policy metadata drift at {name}")
    expected_shared = contract["shared_cbf"]
    for name in (
        "safe_gap_m", "activation_gap_m", "gamma_per_s", "prediction_horizon_s",
        "max_prediction_buffer_m", "max_joint_speed_rad_s",
        "fail_closed_on_invalid_active_hand", "stop_on_infeasible",
    ):
        if get_path(config, f"shared_cbf.{name}") != expected_shared[name]:
            raise ValueError(f"shared CBF drift at {name}")
    if list(get_path(config, "feedback.q1_responses")) != [
        "needs_modification", "acceptable_as_is", "uncertain"
    ]:
        raise ValueError("Q1 response vocabulary drift")
    if list(get_path(config, "feedback.modification_reasons")) != [
        "too_close_or_late", "excessive_or_unnecessary_motion",
        "abrupt_or_unpredictable", "response_too_long",
        "inappropriate_direction", "recovery_or_task_resumption_problem",
        "grasp_place_release_disruption", "other",
    ]:
        raise ValueError("modification reason vocabulary drift")


def actual_cbf_config(cbf: Any, *, condition_id: str) -> dict[str, Any]:
    config = getattr(cbf, "config", None)
    if config is None:
        raise RuntimeError("runtime CBF has no config")
    if is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = dict(vars(config))
    result = {str(key): _json_value(item) for key, item in value.items()}
    frozen = load_config()
    expected_condition = get_path(frozen, f"conditions.{condition_id}")
    if result.get("objective_mode") != expected_condition["objective_mode"]:
        raise RuntimeError("runtime CBF objective mode drift")
    if float(result.get("correction_smoothness_weight", -1.0)) != float(expected_condition["lambda_s"]):
        raise RuntimeError("runtime CBF lambda_s drift")
    for name in (
        "safe_gap_m", "activation_gap_m", "gamma_per_s", "prediction_horizon_s",
        "max_prediction_buffer_m", "max_joint_speed_rad_s",
        "fail_closed_on_invalid_active_hand", "stop_on_infeasible",
    ):
        if result.get(name) != get_path(frozen, f"shared_cbf.{name}"):
            raise RuntimeError(f"runtime shared CBF drift at {name}")
    return result


def get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"missing configuration key: {path}")
        current = current[part]
    return current


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


__all__ = [
    "DEFAULT_CONFIG", "FROZEN_QUESTIONNAIRE_NAVIGATION",
    "FROZEN_REALTIME_MARKER_MAPPING", "PROJECT_ROOT", "RUNTIME_CONFIG",
    "actual_cbf_config",
    "canonical_json", "canonical_sha256", "file_sha256", "get_path",
    "load_config", "runtime_contract", "validate_config",
]
