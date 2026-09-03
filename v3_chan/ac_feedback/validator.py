"""Fail-closed validation for the A/C selective-smoothing feedback study.

The static half of this module only needs the Python standard library.  HDF5
support is imported lazily so configuration and schedule audits still work on
machines that do not have ``h5py`` installed.  A malformed or incomplete
artifact is untrusted input: every unexpected parsing/shape error becomes a
validation issue, never a successful result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .config import (
    DEFAULT_CONFIG,
    FROZEN_QUESTIONNAIRE_NAVIGATION,
    FROZEN_REALTIME_MARKER_MAPPING,
    PROJECT_ROOT,
    RUNTIME_CONFIG,
    canonical_sha256,
    file_sha256,
    load_config,
    runtime_contract,
)
from .online_schema import (
    ACTUAL_CROSSING_DIRECTIONS,
    DIRECTIONS,
    FEATURE_PROVENANCE_VERSION,
    LIKERT_QUESTIONS,
    MARKER_TYPES,
    MODIFICATION_REASONS,
    ONLINE_PROTOCOL_VERSION,
    ONLINE_ROW_SEMANTICS,
    ONLINE_SCHEMA_VERSION,
    Q1_ID,
    Q1_RESPONSES,
    Q1_TEXT_KO,
    SEVERITIES,
    SPEEDS,
    TASK_PHASES,
    EncounterRecordV1,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
)
from .online_protocol import resolve_actual_crossing_direction
from .schema import (
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    POLICY_SIZE_BYTES,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from .study import (
    CONDITIONS,
    FORBIDDEN_DECISION_CONTEXT_KEYS,
    ResponsePhase,
    TrialState,
    build_participant_schedule,
    validate_decision_context,
)
from .video import (
    DEFAULT_CAMERA_EYE_M,
    DEFAULT_CAMERA_TARGET_M,
    DEFAULT_CAMERA_UP,
    VIDEO_SYNC_SCHEMA_VERSION,
    VideoFrameRecord,
)
from .xr_feedback import QuestionnaireAnswer


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS = (
    "PRE_RESPONSE",
    "CBF_ACTIVE",
    "SMOOTH_TAIL",
    "RECOVERY_ACTIVE",
    "BC_RESUMED_FIRST_0_5_S",
    "WHOLE_TASK",
)
_RESPONSE_PHASES = tuple(item.value for item in ResponsePhase)
_TRIAL_STATES = tuple(item.value for item in TrialState)
_CONFIG_CONDITION_FIELDS = frozenset(("objective_mode", "lambda_s"))
_TRIAL_CONDITION_FIELDS = frozenset(("condition_id", "objective_mode", "lambda_s"))
_SHARED_CBF_FIELDS = (
    "safe_gap_m",
    "activation_gap_m",
    "gamma_per_s",
    "prediction_horizon_s",
    "max_prediction_buffer_m",
    "max_joint_speed_rad_s",
    "fail_closed_on_invalid_active_hand",
    "stop_on_infeasible",
)
_EXPECTED_CBF_RUNTIME_SHARED: Mapping[str, Any] = {
    "safe_gap_m": 0.05,
    "activation_gap_m": 0.13,
    "gamma_per_s": 8.0,
    "prediction_horizon_s": 0.15,
    "max_prediction_buffer_m": 0.08,
    "max_joint_speed_rad_s": 2.0,
    "task_space_weight": 1.0,
    "task_yaw_length_scale_m_per_rad": 0.1,
    "joint_regularization_epsilon": 0.05,
    "progress_retention_rho": 0.7,
    "progress_penalty_weight": 50.0,
    "progress_nominal_threshold_mps": 0.01,
    "projection_iterations": 80,
    "projection_tolerance": 1e-5,
    "fail_closed_on_invalid_active_hand": True,
    "stop_on_infeasible": True,
    "require_valid_recorded_hand_tracking": False,
}
_EXPECTED_PHYSICS_CONFIG: Mapping[str, Any] = {
    "physics_dt_s": 1.0 / 60.0,
    "max_episode_steps": 12_000,
    "action_scale": 1.0,
    "action_version": "action_v1_controller_target_delta",
    "fixed_orientation": True,
    "gripper_mode": "policy",
    "cube_count": 1,
}
_EXPECTED_RECOVERY_CONFIG: Mapping[str, Any] = {
    "schema_version": "state_aware_pick_place_recovery_bridge_v3",
    "enabled": True,
    "controller_target_y_offset_m": 0.005,
    "prepose_height_m": 0.17425,
    "observed_ee_frame_offset_x_m": 0.0,
    "observed_ee_frame_offset_y_m": 0.04,
    "observed_ee_frame_offset_z_m": 0.042,
    "place_handoff_progress": 1.0,
    "placement_z_offset_m": 0.015,
    "placement_xy_tolerance_m": 0.02,
    "placement_z_tolerance_m": 0.01,
    "minimum_transport_clearance_m": 0.1,
    "anchor_position_tolerance_m": 0.035,
    "maximum_cube_speed_mps": 0.05,
    "maximum_joint_speed_radps": 0.25,
    "anchor_confirmation_steps": 6,
    "maximum_recovery_steps": 600,
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    path: str
    config_path: str
    valid: bool
    study_eligible: bool
    static_valid: bool
    schedule_valid: bool
    collection_checked: bool
    config_sha256: str
    schedule_seed: int
    counterbalancing_group: str
    trial_count: int
    evaluated_trial_count: int
    practice_trial_count: int
    query_count: int
    encounter_count: int
    realtime_marker_count: int
    excluded_trial_count: int
    issues: tuple[ValidationIssue, ...]

    # Compatibility with the pre-A/C validator/report surface.
    @property
    def episode_count(self) -> int:
        return self.trial_count

    @property
    def prompt_count(self) -> int:
        return self.query_count

    @property
    def response_count(self) -> int:
        return self.query_count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = ONLINE_SCHEMA_VERSION
        result["issues"] = [issue.to_dict() for issue in self.issues]
        return result


class CollectionValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        detail = "; ".join(
            f"[{issue.code}] {issue.path}: {issue.message}"
            for issue in report.issues[:12]
        )
        super().__init__("A/C collection validation failed: " + detail)


class _IssueSink:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def issue(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(str(code), str(path), str(message)))


def _missing_resume_outcome_matches(
    *,
    states: Sequence[str],
    step_success: Sequence[int],
    step_terminated: Sequence[int],
    step_truncated: Sequence[int],
    terminal_reasons: Sequence[str],
    summary_success: bool,
    completion_observed: bool,
    failure_reason: str,
) -> bool:
    """Bind a missing resume row to an outcome seen by query completion.

    The collector keeps physics and the safety filter active while the XR
    questionnaire is displayed.  Consequently, the task can first reach a
    terminal state on any questionnaire row, including the row that confirms
    the final answer.  In that case there is no legal environment step left to
    record under ``RESUME_TASK_TO_COMPLETION_OR_TERMINAL``.  Only evidence at
    or before the last query row can justify that missing state, and the trial
    summary must still agree exactly with the observed outcome.
    """

    lengths = {
        len(states),
        len(step_success),
        len(step_terminated),
        len(step_truncated),
        len(terminal_reasons),
    }
    if len(lengths) != 1:
        return False
    query_state = TrialState.SHOW_MANDATORY_FEEDBACK_UI.value
    query_rows = [index for index, state in enumerate(states) if state == query_state]
    if not query_rows:
        return False
    evidence_end_exclusive = query_rows[-1] + 1
    success_observed = any(
        int(value) == 1 for value in step_success[:evidence_end_exclusive]
    )
    terminal_rows = [
        index
        for index in range(evidence_end_exclusive)
        if int(step_terminated[index]) == 1
        or int(step_truncated[index]) == 1
        or bool(terminal_reasons[index])
    ]
    outcome_observed = bool(success_observed or terminal_rows)
    if completion_observed != outcome_observed or not outcome_observed:
        return False
    if summary_success != success_observed:
        return False
    if success_observed:
        return not failure_reason

    first_terminal = terminal_rows[0]
    expected_reason = str(terminal_reasons[first_terminal])
    if not expected_reason:
        expected_reason = (
            "environment_terminated_without_reason"
            if int(step_terminated[first_terminal]) == 1
            else "environment_truncated_without_reason"
        )
    return failure_reason == expected_reason


def validate_static_contract(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    participant_id: str = "STATIC-P00",
    session_id: str = "STATIC-S00",
    seed: int = 11,
    mode: str | None = None,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate the frozen files, configuration, and one generated schedule."""

    validator = _Validator(
        config_path=Path(config_path),
        collection_path=None,
        participant_id=participant_id,
        session_id=session_id,
        seed=seed,
        mode=mode,
        allow_partial_path=False,
    )
    report = validator.run()
    if raise_on_error and not report.valid:
        raise CollectionValidationError(report)
    return report


def validate_collection(
    path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    participant_id: str | None = None,
    session_id: str | None = None,
    seed: int | None = None,
    mode: str | None = None,
    allow_partial_path: bool = False,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate a finalized (or explicitly internal sealed-partial) artifact."""

    validator = _Validator(
        config_path=Path(config_path),
        collection_path=Path(path),
        participant_id=participant_id,
        session_id=session_id,
        seed=seed,
        mode=mode,
        allow_partial_path=allow_partial_path,
    )
    report = validator.run()
    if raise_on_error and not report.valid:
        raise CollectionValidationError(report)
    return report


# Explicit name for callers that should not be confused with the old v1
# explicit-feedback validator.
validate_ac_selective_smoothing_collection = validate_collection


class _Validator(_IssueSink):
    def __init__(
        self,
        *,
        config_path: Path,
        collection_path: Path | None,
        participant_id: str | None,
        session_id: str | None,
        seed: int | None,
        mode: str | None,
        allow_partial_path: bool,
    ) -> None:
        super().__init__()
        self.config_path = config_path.expanduser().resolve(strict=False)
        self.collection_path = (
            collection_path.expanduser().resolve(strict=False)
            if collection_path is not None
            else None
        )
        self.requested_participant_id = participant_id
        self.requested_session_id = session_id
        self.requested_seed = seed
        self.requested_mode = mode
        self.allow_partial_path = bool(allow_partial_path)
        self.config: dict[str, Any] = {}
        self.config_sha256 = ""
        self.static_issue_count = 0
        self.schedule_issue_count = 0
        self.schedule_seed = -1
        self.counterbalancing_group = ""
        self.counts = {
            "trial_count": 0,
            "evaluated_trial_count": 0,
            "practice_trial_count": 0,
            "query_count": 0,
            "encounter_count": 0,
            "realtime_marker_count": 0,
            "excluded_trial_count": 0,
        }
        self.study_eligible = True
        self.file: Any = None

    def run(self) -> ValidationReport:
        try:
            self._validate_static()
        except Exception as error:
            self.issue(
                "validator.static_failure",
                str(self.config_path),
                f"{type(error).__name__}: {error}",
            )
        self.static_issue_count = len(self.issues)
        try:
            self._validate_requested_schedule()
        except Exception as error:
            self.issue(
                "validator.schedule_failure",
                "/schedule",
                f"{type(error).__name__}: {error}",
            )
        self.schedule_issue_count = len(self.issues) - self.static_issue_count
        if self.collection_path is not None:
            self._validate_hdf5()
        return self._report()

    def _report(self) -> ValidationReport:
        return ValidationReport(
            path=str(self.collection_path or ""),
            config_path=str(self.config_path),
            valid=not self.issues,
            study_eligible=bool(not self.issues and self.study_eligible),
            static_valid=self.static_issue_count == 0,
            schedule_valid=self.schedule_issue_count == 0,
            collection_checked=self.collection_path is not None,
            config_sha256=self.config_sha256,
            schedule_seed=self.schedule_seed,
            counterbalancing_group=self.counterbalancing_group,
            issues=tuple(self.issues),
            **self.counts,
        )

    def _validate_static(self) -> None:
        if not self.config_path.is_file():
            self.issue("config.not_found", str(self.config_path), "configuration is not a regular file")
            return
        try:
            self.config = load_config(self.config_path)
            self.config_sha256 = canonical_sha256(self.config)
        except Exception as error:
            self.issue("config.frozen", str(self.config_path), f"{type(error).__name__}: {error}")
            return
        self._validate_static_policy_and_runtime()
        self._validate_static_config_semantics()

    def _validate_static_policy_and_runtime(self) -> None:
        try:
            contract = runtime_contract()
        except Exception as error:
            self.issue("runtime.contract", str(RUNTIME_CONFIG), f"cannot parse runtime contract: {error}")
            return
        try:
            actual_contract_sha = file_sha256(RUNTIME_CONFIG)
        except OSError as error:
            self.issue("runtime.contract", str(RUNTIME_CONFIG), str(error))
            actual_contract_sha = ""
        if actual_contract_sha != RUNTIME_CONTRACT_SHA256:
            self.issue(
                "runtime.contract_sha256",
                str(RUNTIME_CONFIG),
                f"expected {RUNTIME_CONTRACT_SHA256}, got {actual_contract_sha or '<unavailable>'}",
            )
        policy_rel = str(self.config.get("policy", {}).get("checkpoint", ""))
        policy_path = (PROJECT_ROOT / policy_rel).resolve(strict=False)
        try:
            policy_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            self.issue("policy.path_escape", policy_rel, "checkpoint escapes the project root")
            return
        if policy_rel != POLICY_RELATIVE_PATH:
            self.issue("policy.path", "config/policy/checkpoint", f"expected {POLICY_RELATIVE_PATH!r}, got {policy_rel!r}")
        if not policy_path.is_file():
            self.issue("policy.not_found", str(policy_path), "frozen BC checkpoint is missing")
        else:
            try:
                actual_sha = file_sha256(policy_path)
                actual_size = policy_path.stat().st_size
            except OSError as error:
                self.issue("policy.read", str(policy_path), str(error))
            else:
                if actual_sha != POLICY_SHA256:
                    self.issue("policy.sha256", str(policy_path), f"expected {POLICY_SHA256}, got {actual_sha}")
                if actual_size != POLICY_SIZE_BYTES:
                    self.issue("policy.size", str(policy_path), f"expected {POLICY_SIZE_BYTES}, got {actual_size}")
        if contract.get("policy", {}).get("sha256") != POLICY_SHA256:
            self.issue("runtime.policy", str(RUNTIME_CONFIG), "runtime contract does not pin the required checkpoint")
        if contract.get("responses", {}).get("allowed_collection_conditions") != [
            "A_reactive", "C_smooth"
        ]:
            self.issue("runtime.conditions", str(RUNTIME_CONFIG), "runtime contract condition allowlist drifted")
        if contract.get("responses", {}).get("online_adaptation_during_collection") is not False:
            self.issue("runtime.online_adaptation", str(RUNTIME_CONFIG), "online A/C adaptation must be false")
        task_runtime = contract.get("task_runtime", {})
        expected_task = {
            "single_pick": True,
            "strict_task_semantics_schema": "physical_event_driven_pick_place_v5",
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "gripper_mode": "policy",
            "pseudo_errp_enabled": False,
            "haptics_enabled": False,
        }
        for name, expected in expected_task.items():
            if task_runtime.get(name) != expected:
                self.issue("runtime.task", f"{RUNTIME_CONFIG}/{name}", f"expected {expected!r}, got {task_runtime.get(name)!r}")
        files = contract.get("files")
        if not isinstance(files, Mapping) or not files:
            self.issue("runtime.files", str(RUNTIME_CONFIG), "pinned source-file map is absent")
            return
        for relative, expected_sha in files.items():
            source = (PROJECT_ROOT / str(relative)).resolve(strict=False)
            try:
                source.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                self.issue("runtime.file_path_escape", str(relative), "pinned source path escapes project root")
                continue
            if not _SHA256_RE.fullmatch(str(expected_sha)):
                self.issue("runtime.file_sha_format", str(relative), "pinned source SHA-256 is malformed")
                continue
            if not source.is_file():
                self.issue("runtime.file_missing", str(source), "pinned runtime source is missing")
                continue
            try:
                actual = file_sha256(source)
            except OSError as error:
                self.issue("runtime.file_read", str(source), str(error))
                continue
            if actual != str(expected_sha):
                self.issue("runtime.file_drift", str(source), f"expected {expected_sha}, got {actual}")

    def _validate_static_config_semantics(self) -> None:
        config = self.config
        conditions = config.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
            self.issue("config.conditions", "config/conditions", "conditions must be exactly A_reactive and C_smooth")
        else:
            for condition_id, expected in CONDITIONS.items():
                value = conditions.get(condition_id)
                if not isinstance(value, Mapping) or set(value) != _CONFIG_CONDITION_FIELDS:
                    self.issue("config.condition_fields", f"config/conditions/{condition_id}", "only objective_mode and lambda_s may differ")
                    continue
                if value.get("objective_mode") != expected.objective_mode or not _same_number(value.get("lambda_s"), expected.lambda_s):
                    self.issue("config.condition_value", f"config/conditions/{condition_id}", "condition objective/lambda violates the frozen A/C contract")
        shared = config.get("shared_cbf", {})
        if not isinstance(shared, Mapping):
            self.issue("config.shared_cbf", "config/shared_cbf", "shared CBF config must be a mapping")
        else:
            missing = [name for name in _SHARED_CBF_FIELDS if name not in shared]
            if missing:
                self.issue("config.shared_cbf", "config/shared_cbf", "missing shared fields: " + ", ".join(missing))
            if shared.get("parameter_adaptation") is not False:
                self.issue("config.cbf_adaptation", "config/shared_cbf/parameter_adaptation", "CBF parameter adaptation must be false")
        study = config.get("study", {})
        if study.get("feedback_changes_schedule") is not False:
            self.issue("config.feedback_schedule", "config/study/feedback_changes_schedule", "feedback-driven schedule changes are forbidden")
        if study.get("condition_blinding") is not True:
            self.issue("config.blinding", "config/study/condition_blinding", "participant condition blinding is required")
        if study.get("participant_count_target") != 5:
            self.issue("config.pilot_n", "config/study/participant_count_target", "feasibility pilot target must be five people")
        task = config.get("task", {})
        required_task = {
            "cube_count": 1,
            "bc_feasible_layout_required": True,
            "strict_semantics": "physical_event_driven_pick_place_v5",
            "state_aware_recovery": True,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "recovery_fixed": True,
            "pseudo_errp_enabled": False,
            "task_progression_paused_during_query": True,
            "physics_remains_on_during_query": True,
            "cbf_remains_on_during_query": True,
            "safe_hold_during_query": True,
        }
        for name, expected in required_task.items():
            if task.get(name) != expected:
                self.issue("config.task", f"config/task/{name}", f"expected {expected!r}, got {task.get(name)!r}")
        feedback = config.get("feedback", {})
        if feedback.get("haptics_enabled") is not False:
            self.issue("config.haptics", "config/feedback/haptics_enabled", "haptics must be exactly false")
        if feedback.get("condition_identity_visible") is not False:
            self.issue("config.blinding", "config/feedback/condition_identity_visible", "condition identity must not be visible")
        markers = feedback.get("realtime_markers", {})
        if markers != FROZEN_REALTIME_MARKER_MAPPING:
            self.issue(
                "config.marker_mapping",
                "config/feedback/realtime_markers",
                "realtime marker keys and controller buttons must match the frozen mapping exactly",
            )
        navigation = feedback.get("questionnaire_navigation", {})
        if navigation != FROZEN_QUESTIONNAIRE_NAVIGATION:
            self.issue(
                "config.questionnaire_navigation",
                "config/feedback/questionnaire_navigation",
                "questionnaire navigation and emergency-abort keys/values must match the frozen mapping exactly",
            )
        if feedback.get("no_marker_semantics") != "missing_evidence_not_acceptable":
            self.issue("config.no_marker", "config/feedback/no_marker_semantics", "no marker must not mean acceptable")
        if feedback.get("uncertain_semantics") != "abstain" or feedback.get("timeout_semantics") != "missing_abstain":
            self.issue("config.abstain", "config/feedback", "uncertain/timeout must remain abstain")
        questions = feedback.get("questions", {})
        expected_questions = {"q1": Q1_TEXT_KO, **{
            f"q{index}": text for index, text in enumerate(LIKERT_QUESTIONS.values(), start=2)
        }}
        if questions != expected_questions:
            self.issue("config.questions", "config/feedback/questions", "mandatory Q1-Q6 wording drifted")
        if list(feedback.get("q1_responses", ())) != list(Q1_RESPONSES):
            self.issue("config.q1", "config/feedback/q1_responses", "Q1 vocabulary drifted")
        if list(feedback.get("modification_reasons", ())) != list(MODIFICATION_REASONS):
            self.issue("config.reasons", "config/feedback/modification_reasons", "modification reason vocabulary drifted")
        recording = config.get("recording", {})
        if recording.get("hdf5_enabled") is not True:
            self.issue("config.recording", "config/recording/hdf5_enabled", "HDF5 logging must be enabled")
        if type(recording.get("spectator_video_enabled")) is not bool:
            self.issue("config.video", "config/recording/spectator_video_enabled", "must be an exact boolean")
        camera_path = str(recording.get("camera_prim_path", ""))
        resolution = recording.get("resolution", ())
        try:
            valid_resolution = (
                isinstance(resolution, Sequence)
                and not isinstance(resolution, (str, bytes))
                and len(resolution) == 2
                and all(
                    not isinstance(value, bool)
                    and int(value) == value
                    and int(value) > 0
                    for value in resolution
                )
            )
            valid_rate = all(
                not isinstance(recording.get(name), bool)
                and int(recording.get(name)) == recording.get(name)
                and int(recording.get(name)) > 0
                for name in ("fps", "capture_interval_steps")
            )
            eye = tuple(float(value) for value in recording.get("eye_m", DEFAULT_CAMERA_EYE_M))
            target = tuple(float(value) for value in recording.get("target_m", DEFAULT_CAMERA_TARGET_M))
            valid_pose = len(eye) == 3 and len(target) == 3 and all(math.isfinite(value) for value in (*eye, *target)) and eye != target
        except (TypeError, ValueError, OverflowError):
            valid_resolution = valid_rate = valid_pose = False
        if not camera_path.startswith("/") or not valid_resolution or not valid_rate or not valid_pose:
            self.issue("config.video", "config/recording", "fixed spectator camera path, pose, resolution, FPS, or capture cadence is invalid")

    def _validate_requested_schedule(self) -> None:
        if not self.config:
            return
        participant_id = str(self.requested_participant_id or "STATIC-P00").strip()
        session_id = str(self.requested_session_id or "STATIC-S00").strip()
        seed = 11 if self.requested_seed is None else self.requested_seed
        mode = str(self.requested_mode or self.config.get("study", {}).get("mode", ""))
        try:
            schedule = build_participant_schedule(
                participant_id,
                session_id=session_id,
                seed=int(seed),
                mode=mode,
                practice_trials=4,
            )
        except Exception as error:
            self.issue("schedule.generate", "schedule", f"{type(error).__name__}: {error}")
            return
        self.schedule_seed = schedule.schedule_seed
        self.counterbalancing_group = schedule.counterbalancing_group
        self.counts["trial_count"] = len(schedule.trials)
        self.counts["practice_trial_count"] = len(schedule.practice_trials)
        self.counts["evaluated_trial_count"] = len(schedule.evaluated_trials)
        self.counts["excluded_trial_count"] = len(schedule.practice_trials)
        self._validate_schedule_records([trial.as_dict() for trial in schedule], mode=mode, practice_only=False)

    def _validate_schedule_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        mode: str,
        practice_only: bool,
    ) -> None:
        """Validate a generated or serialized schedule without trusting its builder."""

        if not records:
            self.issue("schedule.empty", "/schedule", "schedule must not be empty")
            return
        indices = [_int(record.get("trial_index", record.get("order_index", -1))) for record in records]
        if indices != list(range(len(records))):
            self.issue("schedule.order", "/schedule/trial_index", "trial order must be contiguous and immutable")
        for identifier in ("trial_id", "query_id"):
            values = [str(record.get(identifier, "")) for record in records]
            if any(not value for value in values) or len(set(values)) != len(values):
                self.issue("schedule.identifiers", f"/schedule/{identifier}", "identifiers must be non-empty and unique")
        seeds = {_int(record.get("schedule_seed", -1)) for record in records}
        groups = {str(record.get("counterbalancing_group", "")) for record in records}
        if len(seeds) != 1 or next(iter(seeds), -1) < 0:
            self.issue("schedule.seed", "/schedule/schedule_seed", "one non-negative schedule_seed is required")
        else:
            self.schedule_seed = next(iter(seeds))
        if len(groups) != 1 or not re.fullmatch(r"G0[0-7]", next(iter(groups), "")):
            self.issue("schedule.group", "/schedule/counterbalancing_group", "one G00-G07 counterbalancing group is required")
        else:
            self.counterbalancing_group = next(iter(groups))
        practice = [record for record in records if _bool(record.get("practice", 0))]
        evaluated = [record for record in records if not _bool(record.get("practice", 0))]
        core = [record for record in evaluated if not _bool(record.get("anchor_repeat", 0))]
        anchors = [record for record in evaluated if _bool(record.get("anchor_repeat", 0))]
        if practice_only:
            if not (1 <= len(practice) <= 4) or evaluated:
                self.issue("schedule.practice_only", "/schedule", "practice-only sessions require one to four practice trials and no evaluated trials")
            self.study_eligible = False
        else:
            expected_evaluated = 16 if mode == "minimal_pilot" else 20 if mode == "pilot_with_anchors" else -1
            expected_anchors = 0 if mode == "minimal_pilot" else 4 if mode == "pilot_with_anchors" else -1
            if len(practice) != 4 or len(evaluated) != expected_evaluated or len(anchors) != expected_anchors:
                self.issue(
                    "schedule.count",
                    "/schedule",
                    f"{mode} requires practice=4, evaluated={expected_evaluated}, anchors={expected_anchors}; got {len(practice)}, {len(evaluated)}, {len(anchors)}",
                )
        for index, record in enumerate(records):
            path = f"/schedule[{index}]"
            condition_id = str(record.get("condition_id", ""))
            condition = CONDITIONS.get(condition_id)
            if condition is None:
                self.issue("schedule.condition", path, f"unknown condition {condition_id!r}")
            elif (
                record.get("objective_mode") != condition.objective_mode
                or not _same_number(record.get("lambda_s"), condition.lambda_s)
            ):
                self.issue("schedule.condition_config", path, "condition objective/lambda does not match A or C")
            phase = str(record.get("task_phase", ""))
            severity = str(record.get("severity", ""))
            direction = str(record.get("direction", record.get("crossing_direction", "")))
            speed = str(record.get("speed", record.get("crossing_speed", "")))
            crossing_hand = str(record.get("crossing_hand", ""))
            if phase not in TASK_PHASES or severity not in SEVERITIES or direction not in DIRECTIONS or speed not in SPEEDS:
                self.issue("schedule.factor", path, "unknown task phase, severity, direction, or speed")
            expected_hand = "left" if direction == "left_to_right" else "right"
            if direction in DIRECTIONS and crossing_hand != expected_hand:
                self.issue("schedule.crossing_hand", path, "crossing hand and direction disagree")
            is_practice = _bool(record.get("practice", 0))
            analysis_exclude = _bool(record.get("analysis_exclude", 0))
            if is_practice and not analysis_exclude:
                self.issue("schedule.practice_exclusion", path, "practice trial must be analysis_exclude=true")
            if not str(record.get("block_id", "")):
                self.issue("schedule.block", path, "block_id is required")
        if len(core) == 16:
            condition_counts = Counter(str(record.get("condition_id", "")) for record in core)
            if condition_counts != Counter({"A_reactive": 8, "C_smooth": 8}):
                self.issue("schedule.condition_balance", "/schedule", f"core A/C counts are {dict(condition_counts)}")
            for phase in TASK_PHASES:
                counts = Counter(
                    str(record.get("condition_id", ""))
                    for record in core if record.get("task_phase") == phase
                )
                if counts != Counter({"A_reactive": 2, "C_smooth": 2}):
                    self.issue("schedule.phase_balance", f"/schedule/{phase}", f"A/C counts are {dict(counts)}")
            for severity in SEVERITIES:
                counts = Counter(
                    str(record.get("condition_id", ""))
                    for record in core if record.get("severity") == severity
                )
                if counts != Counter({"A_reactive": 4, "C_smooth": 4}):
                    self.issue("schedule.severity_balance", f"/schedule/{severity}", f"A/C counts are {dict(counts)}")
            direction_counts = Counter(str(record.get("direction", record.get("crossing_direction", ""))) for record in core)
            speed_counts = Counter(str(record.get("speed", record.get("crossing_speed", ""))) for record in core)
            if direction_counts != Counter({"left_to_right": 8, "right_to_left": 8}):
                self.issue("schedule.direction_balance", "/schedule/direction", f"counts are {dict(direction_counts)}")
            if speed_counts != Counter({"slow": 8, "fast": 8}):
                self.issue("schedule.speed_balance", "/schedule/speed", f"counts are {dict(speed_counts)}")
            run = _maximum_run([str(record.get("condition_id", "")) for record in core])
            if run > 2:
                self.issue("schedule.condition_run", "/schedule/condition_id", f"maximum run is {run}, expected <=2")
            if self.counterbalancing_group:
                group_index = int(self.counterbalancing_group[1:])
                expected_first = ("A_reactive", "C_smooth")[group_index & 1]
                if str(core[0].get("condition_id", "")) != expected_first:
                    self.issue("schedule.first_condition", "/schedule", "first production condition disagrees with counterbalancing group")
        elif not practice_only:
            self.issue("schedule.core_count", "/schedule", f"core factorial has {len(core)} trials, expected 16")
        if anchors:
            condition_counts = Counter(str(record.get("condition_id", "")) for record in anchors)
            contexts: dict[str, list[str]] = defaultdict(list)
            for record in anchors:
                context = str(record.get("anchor_context_id", "") or record.get("source_trial_id", ""))
                if not context:
                    self.issue("schedule.anchor_context", "/schedule", "anchor repeat lacks a context identifier")
                contexts[context].append(str(record.get("condition_id", "")))
            if condition_counts != Counter({"A_reactive": 2, "C_smooth": 2}) or any(
                Counter(values) != Counter({"A_reactive": 1, "C_smooth": 1}) for values in contexts.values()
            ):
                self.issue("schedule.anchor_balance", "/schedule", "each of two anchor contexts must repeat once under A and once under C")

    def _validate_hdf5(self) -> None:
        assert self.collection_path is not None
        suffixes = [suffix.lower() for suffix in self.collection_path.suffixes]
        final_suffix = suffixes[-1] if suffixes else ""
        if final_suffix == ".partial" and not self.allow_partial_path:
            self.issue("file.partial", str(self.collection_path), "public validation rejects .partial artifacts")
        if final_suffix not in {".h5", ".hdf5", ".partial"}:
            self.issue("file.extension", str(self.collection_path), "expected .h5 or .hdf5")
        if not self.collection_path.is_file():
            self.issue("file.not_found", str(self.collection_path), "collection is not a regular file")
            return
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError:
            self.issue("file.h5py_unavailable", str(self.collection_path), "HDF5 validation requires h5py")
            return
        try:
            with h5py.File(self.collection_path, "r") as handle:
                self.file = handle
                self._validate_root()
                self._validate_tables_and_schedule()
                self._validate_trials()
                self._validate_relations()
        except OSError as error:
            self.issue("file.open", str(self.collection_path), str(error))
        except Exception as error:
            self.issue("validator.malformed_artifact", "/", f"{type(error).__name__}: {error}")
        finally:
            self.file = None

    def _validate_root(self) -> None:
        attrs = self.file.attrs
        exact = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "protocol_version": ONLINE_PROTOCOL_VERSION,
            "row_semantics": ONLINE_ROW_SEMANTICS,
            "feature_provenance_version": FEATURE_PROVENANCE_VERSION,
            "collection_complete": 1,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "frozen_direct_bc",
            "policy_frozen": 1,
            "policy_online_updates": 0,
            "policy_fine_tuning": 0,
            "policy_observation_dim": 84,
            "policy_observation_version": "obs_v1_state_controller_phase",
            "policy_action_dim": 5,
            "policy_action_version": "action_v1_controller_target_delta",
            "human_observations_masked": 1,
            "feedback_is_policy_input": 0,
            "feedback_changes_schedule": 0,
            "condition_schedule_preassigned": 1,
            "condition_blinding": 1,
            "condition_identity_visible": 0,
            "pseudo_errp_enabled": 0,
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "cbf_frozen": 1,
            "cbf_feedback_adaptation": 0,
            "cbf_parameter_adaptation": 0,
            "cbf_participant_activation_display": 0,
            "cbf_safe_gap_m": 0.05,
            "cbf_activation_gap_m": 0.13,
            "cbf_gamma_per_s": 8.0,
            "cbf_prediction_horizon_s": 0.15,
            "cbf_max_prediction_buffer_m": 0.08,
            "cbf_max_joint_speed_rad_s": 2.0,
            "single_pick": 1,
            "cube_count": 1,
            "intended_crossings_per_trial": 1,
            "queries_per_trial": 1,
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v5",
            "state_aware_recovery_enabled": 1,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "recovery_fixed": 1,
            "physics_dt_s": 1.0 / 60.0,
            "task_max_episode_steps": 12_000,
            "task_progression_paused_during_query": 1,
            "physics_remains_on_during_query": 1,
            "cbf_remains_on_during_query": 1,
            "safe_hold_during_query": 1,
            "emergency_stop_remains_on_during_query": 1,
            "pilot": 1,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "q1_id": Q1_ID,
        }
        for name, expected in exact.items():
            self._require_exact_attr(attrs, name, expected)
        for name in ("participant_id", "session_id", "source_tree_sha256", "frozen_config_sha256"):
            value = str(_scalar(attrs.get(name, "")))
            if not value:
                self.issue("root.missing", f"/@{name}", "required metadata is missing")
            elif name.endswith("sha256") and not _SHA256_RE.fullmatch(value):
                self.issue("integrity.sha256", f"/@{name}", "must be a lowercase SHA-256")
        participant_id = str(_scalar(attrs.get("participant_id", "")))
        session_id = str(_scalar(attrs.get("session_id", "")))
        if str(_scalar(attrs.get("participant_split_key", ""))) != participant_id:
            self.issue("split.participant", "/@participant_split_key", "split key must equal pseudonymous participant_id")
        if self.requested_participant_id is not None and participant_id != self.requested_participant_id:
            self.issue("identity.participant", "/@participant_id", "does not match CLI participant_id")
        if self.requested_session_id is not None and session_id != self.requested_session_id:
            self.issue("identity.session", "/@session_id", "does not match CLI session_id")
        self._validate_code_provenance(attrs)
        self._validate_root_config(attrs)
        self._validate_root_questionnaire(attrs)
        self._validate_root_geometry(attrs)
        self._validate_runtime_dynamics(attrs)
        self._validate_spectator_video(attrs)
        start_mono, end_mono = _int(attrs.get("session_start_monotonic_ns", 0)), _int(attrs.get("session_end_monotonic_ns", 0))
        start_unix, end_unix = _int(attrs.get("session_start_unix_ns", 0)), _int(attrs.get("session_end_unix_ns", 0))
        if not (0 < start_mono <= end_mono and 0 < start_unix <= end_unix):
            self.issue("clock.session", "/", "session timestamps are absent or out of order")
        practice_only = _bool(attrs.get("practice_only", 0))
        if _int(attrs.get("production_mode", -1)) != int(not practice_only):
            self.issue("root.production_mode", "/@production_mode", "must be the inverse of practice_only")
        if practice_only:
            self.study_eligible = False
        else:
            if not _bool(attrs.get("runtime_production_collection_ready", 0)):
                self.issue("runtime.production_readiness", "/@runtime_production_collection_ready", "evaluated collection requires qualified production readiness")
            if not _bool(attrs.get("live_hmd_qualified", 0)):
                self.issue("runtime.live_hmd", "/@live_hmd_qualified", "evaluated collection requires manual live-HMD qualification")
            if not _bool(attrs.get("qualification_source_lineage_reconciled", 0)):
                self.issue(
                    "runtime.qualification_lineage",
                    "/@qualification_source_lineage_reconciled",
                    "evaluated collection requires explicit reconciliation of held-out versus handoff source SHAs",
                )

    def _validate_runtime_dynamics(self, attrs: Any) -> None:
        contracts = (
            (
                "state_aware_recovery_config",
                _EXPECTED_RECOVERY_CONFIG,
            ),
            ("physics_config", _EXPECTED_PHYSICS_CONFIG),
        )
        parsed: dict[str, Mapping[str, Any]] = {}
        for prefix, expected in contracts:
            json_name = f"{prefix}_json"
            sha_name = f"{prefix}_sha256"
            try:
                value = json.loads(str(_scalar(attrs.get(json_name, ""))))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if not isinstance(value, Mapping):
                self.issue("runtime.config", f"/@{json_name}", "required runtime config is not a JSON object")
                continue
            parsed[prefix] = value
            actual_sha = str(_scalar(attrs.get(sha_name, "")))
            if actual_sha != canonical_sha256(value):
                self.issue("runtime.config_hash", f"/@{sha_name}", f"does not bind to {json_name}")
            if canonical_sha256(value) != canonical_sha256(expected):
                self.issue("runtime.config", f"/@{json_name}", "runtime config differs from the frozen study contract")
        physics = parsed.get("physics_config", {})
        if physics:
            if not _same_number(attrs.get("physics_dt_s", math.nan), physics.get("physics_dt_s", math.nan)):
                self.issue("runtime.physics", "/@physics_dt_s", "root physics_dt_s differs from physics_config_json")
            if _int(attrs.get("task_max_episode_steps", -1)) != _int(physics.get("max_episode_steps", -2)):
                self.issue("runtime.physics", "/@task_max_episode_steps", "root task horizon differs from physics_config_json")

    def _validate_spectator_video(self, attrs: Any) -> None:
        raw = str(_scalar(attrs.get("spectator_video_metadata_json", "")))
        try:
            metadata = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = None
        if not isinstance(metadata, Mapping):
            self.issue("video.metadata", "/@spectator_video_metadata_json", "required spectator-video metadata is not a JSON object")
            return
        expected_keys = {
            "schema_version",
            "enabled",
            "available",
            "fixed_camera_pose",
            "condition_identity_visible",
            "camera_prim_path",
            "camera_eye_m",
            "camera_target_m",
            "camera_up",
            "resolution",
            "fps",
            "capture_interval_steps",
            "video_start_sim_time_s",
            "video_start_control_step",
            "video_start_trial_id",
            "frame_count",
            "capture_call_count",
            "timestamp_semantics",
            "record_dir",
            "mp4_path",
            "frame_csv_path",
            "frame_json_path",
            "setup_error",
            "capture_error",
        }
        if set(metadata) != expected_keys:
            self.issue("video.metadata_schema", "/@spectator_video_metadata_json", "spectator-video metadata fields differ from the frozen schema")
        recording = self.config.get("recording", {})
        if not isinstance(recording, Mapping):
            self.issue("video.config", "config/recording", "recording config is not a mapping")
            return
        enabled = bool(recording.get("spectator_video_enabled", False))
        expected_values = {
            "schema_version": VIDEO_SYNC_SCHEMA_VERSION,
            "enabled": enabled,
            "available": enabled,
            "fixed_camera_pose": True,
            "condition_identity_visible": False,
            "camera_prim_path": str(recording.get("camera_prim_path", "/World/ACFeedbackSpectatorCamera")),
            "camera_eye_m": [float(value) for value in recording.get("eye_m", DEFAULT_CAMERA_EYE_M)],
            "camera_target_m": [float(value) for value in recording.get("target_m", DEFAULT_CAMERA_TARGET_M)],
            "camera_up": [float(value) for value in DEFAULT_CAMERA_UP],
            "resolution": [int(value) for value in recording.get("resolution", (1280, 720))],
            "fps": int(recording.get("fps", 20)),
            "capture_interval_steps": int(recording.get("capture_interval_steps", 3)),
            "timestamp_semantics": "caller_supplied_simulation_state_rendered_with_delta_time_zero",
            "setup_error": "",
            "capture_error": "",
        }
        for name, expected in expected_values.items():
            if metadata.get(name) != expected:
                self.issue("video.metadata_value", f"/@spectator_video_metadata_json/{name}", f"expected {expected!r}, got {metadata.get(name)!r}")

        assert self.collection_path is not None
        base = self.collection_path
        if base.suffix.lower() == ".partial":
            base = base.with_suffix("")
        if base.suffix.lower() in {".h5", ".hdf5"}:
            base = base.with_suffix("")
        spectator = base.parent / f"{base.name}_spectator"
        expected_paths = {
            "record_dir": spectator.parent / f"{spectator.name}_frames",
            "mp4_path": spectator.with_suffix(".mp4"),
            "frame_csv_path": spectator.with_name(f"{spectator.name}_frames.csv"),
            "frame_json_path": spectator.with_name(f"{spectator.name}_frames.json"),
        }
        for name, expected_path in expected_paths.items():
            if str(metadata.get(name, "")) != str(expected_path):
                self.issue("video.path", f"/@spectator_video_metadata_json/{name}", "path is not bound to the collection artifact")
        frame_count = _int(metadata.get("frame_count", -1))
        capture_count = _int(metadata.get("capture_call_count", -1))
        interval = _int(metadata.get("capture_interval_steps", -1))
        csv_path = expected_paths["frame_csv_path"]
        json_path = expected_paths["frame_json_path"]
        if not enabled:
            if frame_count != 0 or capture_count != 0:
                self.issue("video.disabled", "/@spectator_video_metadata_json", "disabled video must have zero captures and frames")
            if any(metadata.get(name) is not None for name in ("video_start_sim_time_s", "video_start_control_step")) or metadata.get("video_start_trial_id", "") != "":
                self.issue("video.disabled", "/@spectator_video_metadata_json", "disabled video must not claim a start frame")
            if any(path.exists() for path in expected_paths.values()):
                self.issue("video.disabled_sidecar", str(spectator), "disabled video must not create frame/video sidecars")
            return
        if frame_count < 1 or capture_count < frame_count or interval < 1 or frame_count != capture_count // interval:
            self.issue("video.frame_count", "/@spectator_video_metadata_json", "enabled video frame/capture counts violate the frozen cadence")
        if not csv_path.is_file() or not json_path.is_file():
            self.issue("video.sidecar_missing", str(spectator), "enabled video requires both CSV and JSON frame sidecars")
            return
        frame_fields = tuple(VideoFrameRecord.__dataclass_fields__)
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                csv_fields = tuple(reader.fieldnames or ())
                csv_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as error:
            self.issue("video.csv", str(csv_path), f"cannot parse frame sidecar: {error}")
            return
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.issue("video.json", str(json_path), f"cannot parse frame sidecar: {error}")
            return
        if csv_fields != frame_fields:
            self.issue("video.csv_schema", str(csv_path), f"expected columns {frame_fields}, got {csv_fields}")
        if not isinstance(payload, Mapping) or set(payload) != {"metadata", "frames"}:
            self.issue("video.json_schema", str(json_path), "JSON sidecar must contain exactly metadata and frames")
            return
        if payload.get("metadata") != metadata:
            self.issue("video.metadata_binding", str(json_path), "JSON-sidecar metadata differs from HDF5 metadata")
        json_rows = payload.get("frames")
        if not isinstance(json_rows, list):
            self.issue("video.json_schema", str(json_path), "frames must be a list")
            return
        if len(csv_rows) != frame_count or len(json_rows) != frame_count:
            self.issue("video.sidecar_count", str(spectator), "HDF5, CSV, and JSON frame counts disagree")
        normalized: list[dict[str, Any]] = []
        for index in range(min(len(csv_rows), len(json_rows))):
            csv_row = csv_rows[index]
            json_row = json_rows[index]
            path = f"{json_path}#frames[{index}]"
            if set(csv_row) != set(frame_fields) or not isinstance(json_row, Mapping) or set(json_row) != set(frame_fields):
                self.issue("video.frame_schema", path, "frame fields differ from the frozen sidecar schema")
                continue
            try:
                from_csv = {
                    "frame_index": int(csv_row["frame_index"]),
                    "trial_frame_index": int(csv_row["trial_frame_index"]),
                    "simulation_time_s": float(csv_row["simulation_time_s"]),
                    "control_step": int(csv_row["control_step"]),
                    "trial_id": str(csv_row["trial_id"]),
                    "monotonic_ns": int(csv_row["monotonic_ns"]),
                    "unix_ns": int(csv_row["unix_ns"]),
                }
                from_json = {
                    "frame_index": int(json_row["frame_index"]),
                    "trial_frame_index": int(json_row["trial_frame_index"]),
                    "simulation_time_s": float(json_row["simulation_time_s"]),
                    "control_step": int(json_row["control_step"]),
                    "trial_id": str(json_row["trial_id"]),
                    "monotonic_ns": int(json_row["monotonic_ns"]),
                    "unix_ns": int(json_row["unix_ns"]),
                }
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self.issue("video.frame_value", path, f"cannot parse frame: {error}")
                continue
            if from_csv != from_json:
                self.issue("video.sidecar_binding", path, "CSV and JSON frame records disagree")
                continue
            normalized.append(from_json)
        self._validate_video_frames(normalized, metadata, json_path)

    def _validate_video_frames(
        self,
        frames: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        sidecar_path: Path,
    ) -> None:
        if not frames:
            return
        if [_int(frame.get("frame_index", -1)) for frame in frames] != list(range(len(frames))):
            self.issue("video.frame_index", str(sidecar_path), "global frame indices must be contiguous from zero")
        trial_counts: Counter[str] = Counter()
        previous_mono = previous_unix = 0
        schedule_order = {
            trial_id: index
            for index, trial_id in enumerate(_strings(self.file.get("schedule/trial_id")))
        }
        previous_trial_order = -1
        trials_by_id: dict[str, Any] = {}
        trials = self.file.get("trials")
        if trials is not None and hasattr(trials, "values"):
            for group in trials.values():
                trials_by_id[str(_scalar(group.attrs.get("trial_id", "")))] = group
        for index, frame in enumerate(frames):
            path = f"{sidecar_path}#frames[{index}]"
            trial_id = str(frame.get("trial_id", ""))
            trial_frame_index = _int(frame.get("trial_frame_index", -1))
            if trial_frame_index != trial_counts[trial_id]:
                self.issue("video.trial_frame_index", path, "per-trial frame index is not contiguous from zero")
            trial_counts[trial_id] += 1
            order = schedule_order.get(trial_id, -1)
            if order < previous_trial_order or order < 0:
                self.issue("video.trial_order", path, "frame trial IDs are unknown or regress against schedule order")
            previous_trial_order = max(previous_trial_order, order)
            sim = _float(frame.get("simulation_time_s", math.nan))
            step = _int(frame.get("control_step", -1))
            mono = _int(frame.get("monotonic_ns", 0))
            unix = _int(frame.get("unix_ns", 0))
            if not math.isfinite(sim) or sim < 0 or step < 0 or mono <= previous_mono or unix <= previous_unix:
                self.issue("video.frame_clock", path, "frame clocks are absent, non-finite, or non-monotonic")
            previous_mono, previous_unix = mono, unix
            group = trials_by_id.get(trial_id)
            if group is None:
                self.issue("video.frame_trial", path, "frame references an unknown trial")
                continue
            try:
                source_steps = [int(value) for value in group["control/source_step"][()]]
                matches = [row for row, value in enumerate(source_steps) if value == step]
                if len(matches) != 1:
                    raise ValueError("control step does not map exactly once")
                row = matches[0]
                row_sim = _float(group["control/sim_time"][row])
                row_mono = _int(group["control/post_step_monotonic_ns"][row])
                row_unix = _int(group["control/unix_time_ns"][row])
                if not math.isclose(sim, row_sim, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError("simulation time differs from the HDF5 row")
                if mono < row_mono or unix < row_unix:
                    raise ValueError("frame capture predates its logged simulation row")
                if mono > _int(group.attrs.get("end_monotonic_ns", 0)) or unix > _int(group.attrs.get("end_unix_ns", 0)):
                    raise ValueError("frame capture falls outside trial bounds")
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self.issue("video.frame_mapping", path, str(error))
        first = frames[0]
        start_values = (
            _same_number(metadata.get("video_start_sim_time_s"), first.get("simulation_time_s")),
            _int(metadata.get("video_start_control_step", -1)) == _int(first.get("control_step", -2)),
            str(metadata.get("video_start_trial_id", "")) == str(first.get("trial_id", "")),
        )
        if not all(start_values):
            self.issue("video.start", "/@spectator_video_metadata_json", "video start metadata differs from frame zero")

    def _require_exact_attr(self, attrs: Any, name: str, expected: Any) -> None:
        path = f"/@{name}"
        if name not in attrs:
            self.issue("root.missing", path, "required attribute is missing")
            return
        actual = _scalar(attrs[name])
        matches = _same_number(actual, expected) if isinstance(expected, float) else actual == expected
        if not matches:
            self.issue("root.value", path, f"expected {expected!r}, got {actual!r}")

    def _validate_code_provenance(self, attrs: Any) -> None:
        code_version = str(_scalar(attrs.get("code_version", "")))
        source = str(_scalar(attrs.get("code_version_source", "")))
        verification = str(_scalar(attrs.get("code_commit_verification", "")))
        commit = str(_scalar(attrs.get("code_commit_sha", "")))
        tree_sha = str(_scalar(attrs.get("source_tree_sha256", "")))
        practice_only = _bool(attrs.get("practice_only", 0))
        if source == "verified_git_head":
            dirty = _int(attrs.get("repository_dirty", -1))
            expected_verification = {
                0: "verified_git_head_clean",
                1: "git_head_dirty",
            }.get(dirty, "")
            if (
                not _COMMIT_RE.fullmatch(commit)
                or code_version != commit
                or verification != expected_verification
            ):
                self.issue("integrity.code_provenance", "/@code_version", "verified Git HEAD provenance fields are inconsistent")
            if not practice_only and dirty != 0:
                self.issue("integrity.code_provenance", "/@repository_dirty", "evaluated collection requires a clean verified Git HEAD")
        elif source == "source_tree_sha256":
            if (
                not _SHA256_RE.fullmatch(tree_sha)
                or code_version != f"source-sha256:{tree_sha}"
                or commit
                or verification != "source_tree_only"
                or _int(attrs.get("repository_dirty", -1)) != 1
            ):
                self.issue("integrity.code_provenance", "/@code_version", "source-only version is not bound to source_tree_sha256")
            if not practice_only:
                self.issue("integrity.code_provenance", "/@code_version_source", "source-tree-only provenance is practice-only")
        else:
            self.issue("integrity.code_provenance", "/@code_version_source", "unknown or unverified code provenance")

    def _validate_root_config(self, attrs: Any) -> None:
        raw = str(_scalar(attrs.get("frozen_config_json", "")))
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.issue("integrity.config_json", "/@frozen_config_json", "missing or invalid JSON object")
            return
        if not isinstance(stored, dict):
            self.issue("integrity.config_json", "/@frozen_config_json", "frozen config must be an object")
            return
        stored_sha = str(_scalar(attrs.get("frozen_config_sha256", "")))
        calculated = canonical_sha256(stored)
        if stored_sha != calculated:
            self.issue("integrity.config_hash", "/@frozen_config_sha256", "does not hash frozen_config_json canonically")
        if self.config and stored != self.config:
            self.issue("integrity.config_drift", "/@frozen_config_json", "collection config differs from the validator's frozen config")
        if self.config_sha256 and stored_sha != self.config_sha256:
            self.issue("integrity.config_drift", "/@frozen_config_sha256", f"expected {self.config_sha256}, got {stored_sha}")
        if self.config_sha256 and str(_scalar(attrs.get("collector_config_canonical_sha256", ""))) != self.config_sha256:
            self.issue("integrity.config_drift", "/@collector_config_canonical_sha256", "collector canonical config hash differs from the frozen config")
        try:
            expected_file_sha = file_sha256(self.config_path)
        except OSError:
            expected_file_sha = ""
        if not expected_file_sha or str(_scalar(attrs.get("collector_config_file_sha256", ""))) != expected_file_sha:
            self.issue("integrity.config_drift", "/@collector_config_file_sha256", "collector config file hash differs from the validator input")
        # Hashes for derived runtime fragments must bind to the JSON actually
        # stored in the artifact when those fragments are present.
        for prefix in ("protocol", "scenario", "cbf"):
            json_name, sha_name = f"{prefix}_config_json", f"{prefix}_config_sha256"
            if json_name not in attrs or sha_name not in attrs:
                self.issue("integrity.derived_config", f"/@{json_name}", "derived runtime config and hash are required")
                continue
            try:
                value = json.loads(str(_scalar(attrs[json_name])))
            except (TypeError, ValueError, json.JSONDecodeError):
                self.issue("integrity.derived_config", f"/@{json_name}", "invalid JSON")
                continue
            if not isinstance(value, dict) or canonical_sha256(value) != str(_scalar(attrs[sha_name])):
                self.issue("integrity.derived_config", f"/@{sha_name}", "does not bind to the stored JSON object")
            if prefix == "cbf" and isinstance(value, dict):
                expected_cbf = {
                    **_EXPECTED_CBF_RUNTIME_SHARED,
                    "objective_mode": "joint_nominal",
                    "correction_smoothness_weight": 0.0,
                }
                if canonical_sha256(value) != canonical_sha256(expected_cbf):
                    self.issue("integrity.derived_config", f"/@{json_name}", "session CBF baseline differs from the exact frozen A config")

    def _validate_root_questionnaire(self, attrs: Any) -> None:
        if str(_scalar(attrs.get("q1_text_ko", ""))) != Q1_TEXT_KO:
            self.issue("feedback.questionnaire", "/@q1_text_ko", "Q1 wording drifted")
        expected_json = {
            "q1_responses_json": list(Q1_RESPONSES),
            "likert_questions_json": LIKERT_QUESTIONS,
            "modification_reasons_json": list(MODIFICATION_REASONS),
            "realtime_marker_types_json": list(MARKER_TYPES),
        }
        for name, expected in expected_json.items():
            try:
                value = json.loads(str(_scalar(attrs.get(name, ""))))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if value != expected:
                self.issue("feedback.questionnaire", f"/@{name}", "feedback vocabulary or wording metadata drifted")
        if str(_scalar(attrs.get("no_marker_semantics", ""))) != "missing_evidence_not_acceptable":
            self.issue("feedback.no_marker_semantics", "/@no_marker_semantics", "no marker must not be encoded as acceptable")
        if str(_scalar(attrs.get("uncertain_semantics", ""))) != "abstain":
            self.issue("feedback.uncertain_semantics", "/@uncertain_semantics", "uncertain must remain abstain")
        if str(_scalar(attrs.get("timeout_semantics", ""))) != "missing_abstain":
            self.issue("feedback.timeout_semantics", "/@timeout_semantics", "timeout/no-response must remain missing abstain")

    def _validate_root_geometry(self, attrs: Any) -> None:
        try:
            links = json.loads(str(_scalar(attrs.get("cbf_protected_links_json", ""))))
            colliders = json.loads(str(_scalar(attrs.get("cbf_protected_colliders_json", ""))))
        except (TypeError, ValueError, json.JSONDecodeError):
            links, colliders = [], []
        expected_tokens = tuple(self.config.get("crossing", {}).get("protected_distal_links", ()))
        if not isinstance(links, list) or not links or not isinstance(colliders, list) or not colliders:
            self.issue("geometry.protected", "/@cbf_protected_links_json", "runtime-discovered protected geometry is required")
        elif expected_tokens and any(not any(token in str(link) for token in expected_tokens) for link in links):
            self.issue("geometry.scope", "/@cbf_protected_links_json", "protected geometry includes an unapproved link")
        elif expected_tokens:
            required_present = set(expected_tokens) - {"panda_link8"}
            if not required_present.issubset(set(str(value) for value in links)):
                self.issue("geometry.incomplete", "/@cbf_protected_links_json", "link6/link7/panda_hand/finger protected geometry is incomplete")
        missing_raw = str(_scalar(attrs.get("cbf_missing_links_json", "[]")))
        try:
            missing = json.loads(missing_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            missing = ["invalid"]
        if missing not in ([], ["panda_link8"]):
            self.issue("geometry.missing", "/@cbf_missing_links_json", "only the documented link8-with-panda_hand-equivalent omission is allowed")

    def _validate_tables_and_schedule(self) -> None:
        required_tables: dict[str, tuple[str, ...]] = {
            "schedule": tuple(field.name for field in fields(TrialPlan)),
            "realtime_markers": tuple(field.name for field in fields(RealtimeMarkerRecord)),
            "queries": tuple(
                "modification_reasons_json" if field.name == "modification_reasons" else field.name
                for field in fields(QueryRecord)
            ),
            "query_answers": (
                "query_id",
                "trial_id",
                *(
                    "value_json" if field.name == "value" else field.name
                    for field in fields(QuestionnaireAnswer)
                ),
            ),
            "encounters": tuple(field.name for field in fields(EncounterRecordV1)),
            "layout_prechecks": (
                "candidate_index",
                "layout_id",
                "seed",
                "controller",
                "policy_sha256",
                "cbf_enabled",
                "human_absent_or_clear",
                "strict_success",
                "release_observed",
                "nominal_phase_references_complete",
                "nominal_phase_references_json",
                "nominal_corridor_bank_complete",
                "nominal_phase_trajectories_json",
                "nominal_corridor_bank_json",
                "steps",
                "diagnostic_group",
                "failure_reason",
                "started_monotonic_ns",
                "ended_monotonic_ns",
                "source_restoration_json",
            ),
        }
        lengths: dict[str, int] = {}
        for table_name, columns in required_tables.items():
            group = self.file.get(table_name)
            if group is None or not hasattr(group, "keys"):
                self.issue("schema.table", f"/{table_name}", "required table group is missing")
                lengths[table_name] = 0
                continue
            actual_columns = set(group.keys())
            expected_columns = set(columns)
            if actual_columns != expected_columns:
                self.issue(
                    "schema.table_columns",
                    f"/{table_name}",
                    "table columns differ from the frozen schema: "
                    f"missing={sorted(expected_columns - actual_columns)}, "
                    f"extra={sorted(actual_columns - expected_columns)}",
                )
            column_lengths: set[int] = set()
            for column in columns:
                dataset = group.get(column)
                if dataset is None or not hasattr(dataset, "shape"):
                    self.issue("schema.column", f"/{table_name}/{column}", "required dataset is missing")
                    continue
                if len(dataset.shape) != 1:
                    self.issue("schema.column_shape", f"/{table_name}/{column}", f"expected a one-dimensional table column, got {dataset.shape}")
                else:
                    column_lengths.add(int(dataset.shape[0]))
            if len(column_lengths) > 1:
                self.issue("schema.table_length", f"/{table_name}", f"column lengths disagree: {sorted(column_lengths)}")
            lengths[table_name] = next(iter(column_lengths), 0)
        self.counts["trial_count"] = lengths.get("schedule", 0)
        self.counts["query_count"] = lengths.get("queries", 0)
        self.counts["encounter_count"] = lengths.get("encounters", 0)
        self.counts["realtime_marker_count"] = lengths.get("realtime_markers", 0)
        for attr, table_name in (
            ("scheduled_trial_count", "schedule"),
            ("trial_count", "schedule"),
            ("query_count", "queries"),
            ("query_answer_count", "query_answers"),
            ("encounter_count", "encounters"),
            ("realtime_marker_count", "realtime_markers"),
            ("layout_precheck_count", "layout_prechecks"),
        ):
            if _int(self.file.attrs.get(attr, -1)) != lengths.get(table_name, 0):
                self.issue("schema.count", f"/@{attr}", f"does not equal /{table_name} length")
        if lengths.get("schedule", 0):
            records = [_table_record(self.file["schedule"], index) for index in range(lengths["schedule"])]
            try:
                trial_order = json.loads(str(_scalar(self.file.attrs.get("trial_order_json", ""))))
            except (TypeError, ValueError, json.JSONDecodeError):
                trial_order = None
            expected_order = [str(record.get("trial_id", "")) for record in records]
            if trial_order != expected_order:
                self.issue("schedule.trial_order", "/@trial_order_json", "does not match the immutable schedule order")
            schedule_seeds = {_int(record.get("schedule_seed", -1)) for record in records}
            groups = {str(record.get("counterbalancing_group", "")) for record in records}
            if schedule_seeds != {_int(self.file.attrs.get("schedule_seed", -2))}:
                self.issue("schedule.root_seed", "/@schedule_seed", "does not match schedule table")
            if groups != {str(_scalar(self.file.attrs.get("counterbalancing_group", "")))}:
                self.issue("schedule.root_group", "/@counterbalancing_group", "does not match schedule table")
            practice_only = _bool(self.file.attrs.get("practice_only", 0))
            mode = str(_scalar(self.file.attrs.get("study_mode", "")))
            if not mode:
                evaluated = sum(not _bool(record.get("practice", 0)) for record in records)
                mode = "pilot_with_anchors" if evaluated == 20 else "minimal_pilot"
            previous_issue_count = len(self.issues)
            self._validate_schedule_records(records, mode=mode, practice_only=practice_only)
            # A full session is generated deterministically before collection;
            # compare it field-for-field to rule out feedback-driven mutation.
            if not practice_only:
                participant = str(_scalar(self.file.attrs.get("participant_id", "")))
                session = str(_scalar(self.file.attrs.get("session_id", "")))
                base_seed = _int(self.file.attrs.get("session_seed", -1))
                if base_seed < 0:
                    self.issue("schedule.base_seed", "/@session_seed", "base schedule seed is required")
                else:
                    try:
                        expected = build_participant_schedule(participant, session_id=session, seed=base_seed, mode=mode, practice_trials=4)
                        expected_records = [_normalize_schedule_record(trial.as_dict()) for trial in expected]
                        actual_records = [_normalize_schedule_record(record) for record in records]
                        if actual_records != expected_records:
                            self.issue("schedule.preassignment", "/schedule", "stored schedule differs from deterministic pre-session assignment")
                    except Exception as error:
                        self.issue("schedule.reconstruct", "/schedule", str(error))
            if len(self.issues) > previous_issue_count:
                self.study_eligible = False
        self._validate_layout_prechecks()

    def _validate_layout_prechecks(self) -> None:
        records = self._table_records("layout_prechecks")
        accepted: set[int] = set()
        for index, record in enumerate(records):
            path = f"/layout_prechecks[{index}]"
            if _int(record.get("candidate_index", -1)) != index:
                self.issue("layout.index", f"{path}/candidate_index", "layout precheck indices must be contiguous table-row indices")
            common = {
                "controller": "bc_only",
                "policy_sha256": POLICY_SHA256,
                "cbf_enabled": 0,
            }
            for name, expected in common.items():
                if _scalar(record.get(name, "")) != expected:
                    self.issue("layout.precheck", f"{path}/{name}", f"expected {expected!r}, got {record.get(name)!r}")
            for name in (
                "human_absent_or_clear",
                "strict_success",
                "release_observed",
                "nominal_phase_references_complete",
                "nominal_corridor_bank_complete",
            ):
                if _int(record.get(name, -1)) not in (0, 1):
                    self.issue("layout.precheck", f"{path}/{name}", "must be an exact boolean")
            human_clear = _bool(record.get("human_absent_or_clear", 0))
            strict_success = _bool(record.get("strict_success", 0))
            release_observed = _bool(record.get("release_observed", 0))
            references_complete = _bool(record.get("nominal_phase_references_complete", 0))
            corridor_complete = _bool(record.get("nominal_corridor_bank_complete", 0))
            diagnostic_group = str(record.get("diagnostic_group", ""))
            failure_reason = str(record.get("failure_reason", ""))
            is_accepted = bool(
                human_clear
                and strict_success
                and release_observed
                and references_complete
                and corridor_complete
                and diagnostic_group == "bc_feasible_main"
                and not failure_reason
            )
            if is_accepted:
                accepted.add(index)
                exact = {
                    "controller": "bc_only",
                    "policy_sha256": POLICY_SHA256,
                    "cbf_enabled": 0,
                    "human_absent_or_clear": 1,
                    "release_observed": 1,
                    "nominal_phase_references_complete": 1,
                    "nominal_corridor_bank_complete": 1,
                    "diagnostic_group": "bc_feasible_main",
                    "failure_reason": "",
                }
                for name, expected in exact.items():
                    if _scalar(record.get(name, "")) != expected:
                        self.issue("layout.precheck", f"{path}/{name}", f"expected {expected!r}, got {record.get(name)!r}")
            else:
                if not failure_reason:
                    self.issue("layout.diagnostic", f"{path}/failure_reason", "every rejected precheck needs an explicit failure reason")
                if not human_clear:
                    expected_group = "precheck_invalid_human_intrusion"
                elif strict_success and references_complete:
                    expected_group = "nominal_sweep_infeasible_diagnostic"
                else:
                    expected_group = "bc_infeasible_diagnostic"
                if diagnostic_group != expected_group:
                    self.issue("layout.diagnostic", f"{path}/diagnostic_group", f"expected rejected-row group {expected_group!r}, got {diagnostic_group!r}")
            started = _int(record.get("started_monotonic_ns", 0))
            ended = _int(record.get("ended_monotonic_ns", 0))
            if not (0 < started <= ended) or _int(record.get("steps", 0)) <= 0:
                self.issue("layout.clock", path, "precheck steps/timestamps are invalid")
            for name in ("nominal_phase_references_json", "nominal_phase_trajectories_json", "nominal_corridor_bank_json", "source_restoration_json"):
                try:
                    value = json.loads(str(record.get(name, "")))
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = None
                if not isinstance(value, dict):
                    self.issue("layout.json", f"{path}/{name}", "must be a JSON object")
        trials = self.file.get("trials")
        if trials is not None and hasattr(trials, "keys"):
            for name, group in trials.items():
                index = _int(group.attrs.get("layout_precheck_index", -1))
                if index not in accepted:
                    self.issue("layout.trial_reference", f"/trials/{name}/@layout_precheck_index", "trial does not reference a successful BC-only layout precheck")
                elif records[index].get("layout_id") != _scalar(group.attrs.get("layout_id", "")):
                    self.issue("layout.trial_reference", f"/trials/{name}/@layout_id", "trial layout differs from its successful precheck")

    def _validate_trials(self) -> None:
        trials = self.file.get("trials")
        schedule = self.file.get("schedule")
        if trials is None or not hasattr(trials, "keys"):
            self.issue("schema.trials", "/trials", "required trials group is missing")
            return
        expected_names = [f"trial_{index:06d}" for index in range(self.counts["trial_count"])]
        if sorted(trials.keys()) != expected_names:
            self.issue("schema.trial_groups", "/trials", "trial groups must exactly match contiguous schedule rows")
        if schedule is None or not hasattr(schedule, "keys"):
            return
        excluded = 0
        evaluated = 0
        practice = 0
        frozen_runtime_cbf_shared: Mapping[str, Any] | None = None
        for index, name in enumerate(expected_names):
            if name not in trials:
                continue
            group = trials[name]
            path = f"/trials/{name}"
            schedule_record = _table_record(schedule, index)
            self._validate_trial_plan_attrs(group.attrs, schedule_record, path)
            is_practice = _bool(group.attrs.get("practice", 0))
            analysis_exclude = _bool(group.attrs.get("analysis_exclude", 0))
            protocol_valid = _bool(group.attrs.get("protocol_valid", 0))
            off_protocol = _bool(group.attrs.get("off_protocol", 0))
            practice += int(is_practice)
            evaluated += int(not is_practice)
            excluded += int(analysis_exclude)
            if _int(group.attrs.get("trial_complete", 0)) != 1:
                self.issue("trial.incomplete", f"{path}/@trial_complete", "sealed collection contains incomplete trial")
            if _int(group.attrs.get("single_pick", -1)) != 1 or _int(group.attrs.get("cube_count", -1)) != 1:
                self.issue("trial.single_pick", path, "trial must contain exactly one cube and one pick")
            if _int(group.attrs.get("intended_crossing_count", -1)) != 1:
                self.issue("trial.crossing_intended", f"{path}/@intended_crossing_count", "exactly one intended crossing is required")
            actual_crossings = _int(group.attrs.get("actual_crossing_count", -1))
            query_present = _int(group.attrs.get("query_present", -1))
            if query_present != 1:
                self.issue("trial.query", f"{path}/@query_present", "exactly one mandatory query must be recorded")
            if protocol_valid and (actual_crossings != 1 or off_protocol or query_present != 1):
                self.issue("trial.protocol_valid", path, "protocol_valid trial violates crossing/query/off-protocol cardinality")
            expected_exclude = bool(is_practice or off_protocol or not protocol_valid)
            if analysis_exclude != expected_exclude:
                self.issue("trial.analysis_exclude", f"{path}/@analysis_exclude", f"expected {int(expected_exclude)}, got {int(analysis_exclude)}")
            # Practice rows are intentionally analysis-excluded and are part
            # of every complete production session.  They do not make the
            # evaluated portion ineligible; only a bad evaluated row does.
            if not is_practice and (off_protocol or not protocol_valid):
                self.study_eligible = False
            if _int(group.attrs.get("layout_bc_feasible", 0)) != 1:
                self.issue("trial.layout", f"{path}/@layout_bc_feasible", "human-feedback trial must use a BC-feasible layout")
            if str(_scalar(group.attrs.get("layout_group", ""))) != "bc_feasible_main":
                self.issue("trial.layout", f"{path}/@layout_group", "diagnostic layouts are forbidden in human-feedback trials")
            if _int(group.attrs.get("haptics_enabled", -1)) != 0 or _int(group.attrs.get("haptic_event_count", 0)) != 0:
                self.issue("trial.haptics", path, "any configured or actual haptic activity invalidates the trial")
            self._validate_trial_condition(group, path)
            try:
                actual_cbf = json.loads(
                    str(_scalar(group.attrs.get("actual_cbf_config_json", "")))
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                actual_cbf = None
            if isinstance(actual_cbf, Mapping) and actual_cbf:
                shared_runtime = {
                    str(key): value
                    for key, value in actual_cbf.items()
                    if key not in {
                        "objective_mode",
                        "correction_smoothness_weight",
                        "lambda_s",
                    }
                }
                if frozen_runtime_cbf_shared is None:
                    frozen_runtime_cbf_shared = shared_runtime
                elif canonical_sha256(shared_runtime) != canonical_sha256(
                    frozen_runtime_cbf_shared
                ):
                    self.issue(
                        "condition.runtime_invariance",
                        f"{path}/@actual_cbf_config_json",
                        "runtime CBF fields other than objective/lambda changed across A/C trials",
                    )
            self._validate_trial_summary(group, path)
            self._validate_trial_steps(group, path)
        self.counts["practice_trial_count"] = practice
        self.counts["evaluated_trial_count"] = evaluated
        self.counts["excluded_trial_count"] = excluded

    def _validate_trial_plan_attrs(self, attrs: Any, schedule: Mapping[str, Any], path: str) -> None:
        # ``analysis_exclude`` is initialized by the plan but becomes an
        # outcome at seal time (e.g. a tracked off-protocol crossing).  The
        # assignment itself remains immutable; exclusion is checked from the
        # final protocol flags in ``_validate_trials``.
        plan_fields = tuple(
            field.name for field in fields(TrialPlan)
            if field.name != "analysis_exclude"
        )
        for name in plan_fields:
            expected = schedule.get(name)
            if name not in attrs:
                self.issue("trial.plan_attr", f"{path}/@{name}", "scheduled assignment attribute is missing")
                continue
            actual = _scalar(attrs[name])
            if isinstance(expected, float):
                matches = _same_number(actual, expected)
            else:
                matches = actual == expected
            if not matches:
                self.issue("trial.schedule_mutation", f"{path}/@{name}", f"scheduled {expected!r}, trial logged {actual!r}")
        severity_name = str(schedule.get("severity", ""))
        severity = self.config.get("severity", {}).get(severity_name, {})
        planned_range = severity.get("planned_minimum_gap_range_m", ()) if isinstance(severity, Mapping) else ()
        if not isinstance(planned_range, Sequence) or len(planned_range) != 2:
            self.issue("trial.severity_config", path, "assigned severity has no frozen planned range")
        else:
            expected = (
                _float(planned_range[0]),
                _float(planned_range[1]),
                _float(severity.get("target_gap_m", math.nan)),
            )
            actual = (
                _float(attrs.get("planned_gap_min_m", math.nan)),
                _float(attrs.get("planned_gap_max_m", math.nan)),
                _float(attrs.get("planned_target_gap_m", math.nan)),
            )
            if any(not _same_number(first, second) for first, second in zip(actual, expected)):
                self.issue("trial.severity_plan", path, f"planned gap {actual} differs from frozen severity {expected}")

    def _validate_trial_condition(self, group: Any, path: str) -> None:
        attrs = group.attrs
        condition_id = str(_scalar(attrs.get("condition_id", "")))
        condition = CONDITIONS.get(condition_id)
        if condition is None:
            self.issue("condition.id", f"{path}/@condition_id", f"unknown condition {condition_id!r}")
            return
        if str(_scalar(attrs.get("objective_mode", ""))) != condition.objective_mode:
            self.issue("condition.objective", f"{path}/@objective_mode", "objective mode does not match assigned condition")
        if not _same_number(attrs.get("lambda_s", math.nan), condition.lambda_s):
            self.issue("condition.lambda", f"{path}/@lambda_s", "lambda_s does not match assigned condition")
        parsed: dict[str, dict[str, Any]] = {}
        for attr_name in ("condition_config_json", "actual_cbf_config_json"):
            try:
                value = json.loads(str(_scalar(attrs.get(attr_name, ""))))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if not isinstance(value, dict):
                self.issue("condition.config_json", f"{path}/@{attr_name}", "required condition config is invalid JSON")
            else:
                parsed[attr_name] = value
        assigned = parsed.get("condition_config_json", {})
        if set(assigned) != _TRIAL_CONDITION_FIELDS or assigned.get("condition_id") != condition_id or assigned.get("objective_mode") != condition.objective_mode or not _same_number(assigned.get("lambda_s"), condition.lambda_s):
            self.issue("condition.assignment_config", f"{path}/@condition_config_json", "must contain exactly condition_id, objective_mode, and lambda_s")
        elif str(_scalar(attrs.get("condition_config_sha256", ""))) != canonical_sha256(assigned):
            self.issue("condition.assignment_hash", f"{path}/@condition_config_sha256", "does not bind to condition_config_json")
        actual = parsed.get("actual_cbf_config_json", {})
        if not actual:
            self.issue("condition.runtime_config", f"{path}/@actual_cbf_config_json", "runtime CBF configuration must be a non-empty object")
        else:
            if str(_scalar(attrs.get("actual_cbf_config_sha256", ""))) != canonical_sha256(actual):
                self.issue("condition.runtime_hash", f"{path}/@actual_cbf_config_sha256", "does not bind to actual_cbf_config_json")
            expected_actual = {
                **_EXPECTED_CBF_RUNTIME_SHARED,
                "objective_mode": condition.objective_mode,
                "correction_smoothness_weight": condition.lambda_s,
            }
            if canonical_sha256(actual) != canonical_sha256(expected_actual):
                self.issue("condition.runtime_config", f"{path}/@actual_cbf_config_json", "runtime CBF config must equal the exact frozen A/C config with no aliases or extra fields")
            for name in _SHARED_CBF_FIELDS:
                expected = self.config.get("shared_cbf", {}).get(name)
                if actual.get(name) != expected:
                    self.issue("condition.shared_cbf", f"{path}/@actual_cbf_config_json/{name}", f"expected shared value {expected!r}, got {actual.get(name)!r}")

    def _validate_trial_summary(self, group: Any, path: str) -> None:
        import numpy as np

        attrs = group.attrs
        response_mode = self._response_mode(group)
        required = (
            "response_onset_simulation_time_s",
            "response_end_simulation_time_s",
            "response_onset_step",
            "response_onset_confirmed_step",
            "response_end_step_exclusive",
            "recovery_onset_step",
            "recovery_end_step_exclusive",
            "recovery_onset_simulation_time_s",
            "recovery_end_simulation_time_s",
            "stable_task_resumption_step",
            "cbf_active_duration_s",
            "smooth_tail_duration_s",
            "recovery_duration_s",
            "bc_resumed_duration_s",
            "total_safety_response_duration_s",
            "integrated_intervention_rad",
            "correction_total_variation_rad_s",
            "windowed_jerk_json",
            "delta_path_m",
            "delta_completion_time_s",
            "decision_context_json",
            "trial_fsm_history_json",
            "task_success",
            "task_failure_reason",
            "object_drop",
            "grasp_outcome",
            "release_outcome",
            "task_completion_observed",
            "completion_time_s",
            "completion_time_semantics",
            "actual_crossing_direction",
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "actual_minimum_gap_m",
            "tracking_validity_rate",
            "realtime_safety_marker_count",
            "realtime_behavior_anomaly_marker_count",
            "response_phase_history_json",
            "response_phase_durations_json",
        )
        for name in required:
            if name not in attrs:
                self.issue("trial.summary", f"{path}/@{name}", "required trial outcome is missing")
        onset = _float(attrs.get("response_onset_simulation_time_s", math.nan))
        end = _float(attrs.get("response_end_simulation_time_s", math.nan))
        durations = {
            name: _float(attrs.get(name, math.nan))
            for name in (
                "cbf_active_duration_s",
                "smooth_tail_duration_s",
                "recovery_duration_s",
                "bc_resumed_duration_s",
                "total_safety_response_duration_s",
                "integrated_intervention_rad",
                "correction_total_variation_rad_s",
                "delta_path_m",
                "delta_completion_time_s",
            )
        }
        if response_mode == "no_onset":
            sentinel_values = {
                "response_onset_simulation_time_s": -1.0,
                "response_end_simulation_time_s": -1.0,
                "response_onset_step": -1,
                "response_onset_confirmed_step": -1,
                "response_end_step_exclusive": -1,
                "recovery_onset_step": -1,
                "recovery_end_step_exclusive": -1,
                "recovery_onset_simulation_time_s": -1.0,
                "recovery_end_simulation_time_s": -1.0,
                "stable_task_resumption_step": -1,
            }
            for name, expected in sentinel_values.items():
                actual = attrs.get(name, math.nan)
                if not _same_number(actual, expected):
                    self.issue(
                        "response.partial_sentinel",
                        f"{path}/@{name}",
                        f"unconfirmed-onset trial requires sentinel {expected!r}",
                    )
        elif response_mode == "no_stable":
            if not math.isfinite(onset) or onset < 0:
                self.issue(
                    "response.clock",
                    f"{path}/@response_onset_simulation_time_s",
                    "confirmed partial response requires a valid onset clock",
                )
            for name, expected in (
                ("response_end_simulation_time_s", -1.0),
                ("response_end_step_exclusive", -1),
                ("stable_task_resumption_step", -1),
            ):
                if not _same_number(attrs.get(name, math.nan), expected):
                    self.issue(
                        "response.partial_sentinel",
                        f"{path}/@{name}",
                        f"stable-timeout trial requires sentinel {expected!r}",
                    )
        elif not (
            math.isfinite(onset)
            and math.isfinite(end)
            and 0 <= onset <= end
        ):
            self.issue("response.clock", path, "response onset/end are absent or out of order")
        for name, value in durations.items():
            if not math.isfinite(value):
                self.issue("trial.summary_finite", f"{path}/@{name}", "value must be finite")
            elif name not in {"delta_path_m", "delta_completion_time_s"} and value < 0:
                self.issue("trial.summary_nonnegative", f"{path}/@{name}", "value must be non-negative")
        total = durations["total_safety_response_duration_s"]
        # Recovery is reconstructed from its raw activity/authority mask and
        # may overlap the mutually-exclusive CBF/tail phase labels.
        categorical_components = sum(
            durations[name]
            for name in ("cbf_active_duration_s", "smooth_tail_duration_s")
        )
        if response_mode in {"no_onset", "no_stable"}:
            for name in (
                "total_safety_response_duration_s",
                "integrated_intervention_rad",
                "correction_total_variation_rad_s",
            ):
                if not _same_number(attrs.get(name, math.nan), 0.0):
                    self.issue(
                        "response.partial_metric_sentinel",
                        f"{path}/@{name}",
                        "incomplete safety-response episodes preserve raw rows and use zero for non-final metrics",
                    )
        elif (
            math.isfinite(total)
            and math.isfinite(categorical_components)
            and total + 1e-6 < categorical_components
        ):
            self.issue(
                "response.duration",
                path,
                "total safety-response duration is shorter than its disjoint categorical component durations",
            )
        try:
            history = json.loads(str(_scalar(attrs.get("trial_fsm_history_json", ""))))
        except (TypeError, ValueError, json.JSONDecodeError):
            history = None
        if history != list(_TRIAL_STATES):
            self.issue("trial.fsm", f"{path}/@trial_fsm_history_json", "FSM history must contain each of the 17 states exactly once and in order")
        try:
            context = json.loads(str(_scalar(attrs.get("decision_context_json", ""))))
            if not isinstance(context, dict):
                raise ValueError("decision context is not an object")
            if response_mode == "no_onset":
                if context:
                    raise ValueError(
                        "unconfirmed-onset trial must not promote a candidate decision context"
                    )
            else:
                validate_decision_context(context)
        except Exception as error:
            context = {}
            self.issue("decision_context.schema", f"{path}/@decision_context_json", str(error))
        forbidden = sorted(_recursive_keys(context).intersection(FORBIDDEN_DECISION_CONTEXT_KEYS))
        if forbidden:
            self.issue("decision_context.future_leak", f"{path}/@decision_context_json", "future outcomes present: " + ", ".join(forbidden))
        condition_id = str(_scalar(attrs.get("condition_id", "")))
        condition = CONDITIONS.get(condition_id)
        if context and condition and (
            context.get("condition_id") != condition_id
            or not _same_number(context.get("lambda_s"), condition.lambda_s)
        ):
            self.issue("decision_context.assignment", f"{path}/@decision_context_json", "condition/lambda differs from trial assignment")
        self._validate_windowed_jerk(attrs.get("windowed_jerk_json", ""), path)
        tracking_rate = _float(attrs.get("tracking_validity_rate", math.nan))
        minimum_rate = _float(self.config.get("tracking", {}).get("minimum_valid_fraction", math.nan))
        off_protocol = _bool(attrs.get("off_protocol", 0))
        reasons = _json_string_list(attrs.get("off_protocol_reasons_json", ""))
        if reasons is None or off_protocol != bool(reasons):
            self.issue("trial.off_protocol_summary", f"{path}/@off_protocol_reasons_json", "off_protocol flag must exactly match a JSON list of reasons")
        if math.isfinite(tracking_rate) and math.isfinite(minimum_rate) and tracking_rate < minimum_rate and not off_protocol:
            self.issue("tracking.off_protocol", path, "low tracking validity was not marked off_protocol")
        deviation = _float(attrs.get("actual_path_deviation_max_m", math.nan))
        max_deviation = _float(self.config.get("crossing", {}).get("maximum_path_deviation_m", math.nan))
        if math.isfinite(deviation) and math.isfinite(max_deviation) and deviation > max_deviation and not off_protocol:
            self.issue("tracking.path_deviation", path, "excessive path deviation was not marked off_protocol")
        actual_direction = str(_scalar(attrs.get("actual_crossing_direction", "")))
        planned_direction = str(_scalar(attrs.get("direction", "")))
        crossing_hand = str(_scalar(attrs.get("crossing_hand", "")))
        expected_non_crossing_hand = (
            "right" if crossing_hand == "left" else "left"
        )
        intended_hands = _strings(group["crossing/intended_hand"])
        non_crossing_hands = _strings(group["crossing/non_crossing_hand"])
        if (
            crossing_hand not in {"left", "right"}
            or any(value != crossing_hand for value in intended_hands)
            or any(
                value != expected_non_crossing_hand
                for value in non_crossing_hands
            )
        ):
            self.issue(
                "crossing.hand_identity",
                f"{path}/crossing",
                "intended/non-crossing hand rows must be the exact left-right complement of the assigned crossing hand",
            )
        intended_completed = np.asarray(
            group["crossing/intended_completed_now"][()], dtype=int
        )
        reverse_completed = np.asarray(
            group["crossing/reverse_completed_now"][()], dtype=int
        )
        if (
            np.any((intended_completed != 0) & (intended_completed != 1))
            or np.any((reverse_completed != 0) & (reverse_completed != 1))
        ):
            self.issue(
                "crossing.direction_evidence",
                f"{path}/crossing",
                "crossing completion flags must be exact booleans",
            )
        simultaneous = np.flatnonzero(
            (intended_completed == 1) & (reverse_completed == 1)
        )
        if simultaneous.size:
            self.issue(
                "crossing.direction_evidence",
                f"{path}/crossing",
                "one control row cannot complete both traversal directions",
            )
        completion_rows = np.flatnonzero(
            (intended_completed == 1) | (reverse_completed == 1)
        )
        first_sign = 0
        if completion_rows.size:
            first_row = int(completion_rows[0])
            first_sign = 1 if intended_completed[first_row] == 1 else -1
        expected_actual_direction = (
            resolve_actual_crossing_direction(planned_direction, first_sign)
            if planned_direction in DIRECTIONS
            else ""
        )
        if (
            actual_direction not in ACTUAL_CROSSING_DIRECTIONS
            or actual_direction != expected_actual_direction
        ):
            self.issue(
                "crossing.direction",
                f"{path}/@actual_crossing_direction",
                "actual direction must equal the first completed signed "
                "traversal in the step rows",
            )
        missing_direction_reason = (
            "actual_crossing_direction_not_observed" in set(reasons or ())
        )
        if (actual_direction == "not_observed") != missing_direction_reason:
            self.issue(
                "crossing.direction_summary",
                f"{path}/@off_protocol_reasons_json",
                "not_observed direction must exactly match its off-protocol reason",
            )
        if actual_direction == "not_observed" and not off_protocol:
            self.issue(
                "crossing.direction_summary",
                path,
                "not_observed direction must be marked off_protocol",
            )
        speed_name = str(_scalar(attrs.get("speed", "")))
        speed_target = _float(self.config.get("crossing", {}).get("speed_target_m_s", {}).get(speed_name, math.nan))
        speed_tolerance = _float(self.config.get("crossing", {}).get("speed_tolerance_fraction", math.nan))
        actual_speed = _float(attrs.get("actual_crossing_speed_m_s", math.nan))
        if math.isfinite(speed_target) and math.isfinite(speed_tolerance):
            within_speed = speed_target * (1.0 - speed_tolerance) <= actual_speed <= speed_target * (1.0 + speed_tolerance)
            if not within_speed and not off_protocol:
                self.issue("crossing.speed", path, "crossing speed deviation was not marked off_protocol")
        task_success = _int(attrs.get("task_success", -1))
        object_drop = _int(attrs.get("object_drop", -1))
        if task_success not in (0, 1) or object_drop not in (0, 1):
            self.issue("task.outcome", path, "task_success/object_drop must be exact booleans")
        step_success = [int(value) for value in group["task/success"][()]]
        step_terminated = [
            int(value) for value in group["task/terminated"][()]
        ]
        step_truncated = [
            int(value) for value in group["task/truncated"][()]
        ]
        terminal_reasons = _strings(group["task/terminal_reason"])
        if any(
            value not in (0, 1)
            for values in (step_success, step_terminated, step_truncated)
            for value in values
        ):
            self.issue(
                "task.step_outcome",
                f"{path}/task",
                "success/terminated/truncated row flags must be exact booleans",
            )
        ever_success = any(value == 1 for value in step_success)
        terminal_rows = [
            index
            for index, (terminated, truncated, reason) in enumerate(
                zip(step_terminated, step_truncated, terminal_reasons)
            )
            if terminated == 1 or truncated == 1 or bool(reason)
        ]
        completion_observed = bool(ever_success or terminal_rows)
        if task_success != int(ever_success):
            self.issue(
                "task.success_summary",
                f"{path}/@task_success",
                "summary must equal whether any control row observed success",
            )
        if _int(attrs.get("task_completion_observed", -1)) != int(
            completion_observed
        ):
            self.issue(
                "task.completion_summary",
                f"{path}/@task_completion_observed",
                "summary must equal success/terminal evidence in the control rows",
            )
        if not completion_observed:
            self.issue(
                "task.outcome_missing",
                path,
                "a sealed trial must continue through observed task success or terminal outcome",
            )
        failure_reason = str(_scalar(attrs.get("task_failure_reason", "")))
        if ever_success:
            if failure_reason:
                self.issue("task.outcome", path, "successful trial cannot carry a failure reason")
        elif terminal_rows:
            first_terminal = terminal_rows[0]
            expected_reason = terminal_reasons[first_terminal]
            if not expected_reason:
                expected_reason = (
                    "environment_terminated_without_reason"
                    if step_terminated[first_terminal] == 1
                    else "environment_truncated_without_reason"
                )
            if failure_reason != expected_reason:
                self.issue(
                    "task.failure_reason",
                    f"{path}/@task_failure_reason",
                    f"expected first observed terminal reason {expected_reason!r}, got {failure_reason!r}",
                )
        elif failure_reason:
            self.issue(
                "task.failure_reason",
                f"{path}/@task_failure_reason",
                "failure reason lacks a corresponding terminal row",
            )
        if object_drop != int(any(int(value) for value in group["task/object_drop"][()])):
            self.issue("task.object_drop", path, "step and trial object-drop outcomes disagree")

    def _response_mode(self, group: Any) -> str:
        """Return the only two admissible incomplete-response modes.

        An incomplete episode is data, not a fabricated successful episode.
        Relaxation therefore requires the recorder's full exclusion triad and
        explicit reasons.  Any other attribute combination remains on the
        complete/strict path and fails against missing onset or stable data.
        """

        attrs = group.attrs
        reasons = _json_string_list(
            attrs.get("off_protocol_reasons_json", "")
        )
        excluded = bool(
            not _bool(attrs.get("protocol_valid", 0))
            and _bool(attrs.get("off_protocol", 0))
            and _bool(attrs.get("analysis_exclude", 0))
            and reasons is not None
        )
        reason_set = set(reasons or ())
        confirmed = _int(attrs.get("response_onset_confirmed_step", -1))
        stable = _int(attrs.get("stable_task_resumption_step", -1))
        if (
            excluded
            and confirmed < 0
            and stable < 0
            and {
                "response_onset_not_confirmed",
                "stable_task_resumption_not_confirmed",
            }.issubset(reason_set)
        ):
            return "no_onset"
        if (
            excluded
            and confirmed >= 0
            and stable < 0
            and "stable_task_resumption_not_confirmed" in reason_set
        ):
            return "no_stable"
        return "complete"

    def _validate_windowed_jerk(self, raw: Any, path: str) -> None:
        try:
            value = json.loads(str(_scalar(raw)))
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
        if not isinstance(value, dict):
            self.issue("metrics.windowed_jerk", f"{path}/@windowed_jerk_json", "must be a JSON object")
            return
        if set(value) != set(_WINDOWS):
            self.issue("metrics.windowed_jerk", f"{path}/@windowed_jerk_json", f"windows must be exactly {list(_WINDOWS)}")
        for window in _WINDOWS:
            metrics = value.get(window)
            if not isinstance(metrics, Mapping) or set(metrics) != {"ee", "joint_space"}:
                self.issue("metrics.windowed_jerk", f"{path}/@windowed_jerk_json/{window}", "window must contain exactly ee and joint_space metrics")
                continue
            expected = {
                "ee": {"sample_count", "rms_norm_m_s3", "peak_norm_m_s3"},
                "joint_space": {"sample_count", "rms_norm_rad_s3", "peak_norm_rad_s3"},
            }
            for space, required_keys in expected.items():
                item = metrics.get(space)
                item_path = f"{path}/@windowed_jerk_json/{window}/{space}"
                if not isinstance(item, Mapping) or set(item) != required_keys:
                    self.issue("metrics.windowed_jerk", item_path, f"metrics must be exactly {sorted(required_keys)}")
                    continue
                sample_count = _int(item.get("sample_count", -1))
                numeric = [_float(item.get(name, math.nan)) for name in required_keys if name != "sample_count"]
                if sample_count < 0 or any(not math.isfinite(metric) or metric < 0 for metric in numeric):
                    self.issue("metrics.windowed_jerk", item_path, "sample count and jerk metrics must be finite and non-negative")

    def _validate_trial_steps(self, group: Any, path: str) -> None:
        try:
            import numpy as np
            from .online_rows import online_row_specs
        except Exception as error:
            self.issue("schema.row_import", path, f"cannot load row schema: {error}")
            return
        try:
            joint_names = _strings(group["actions/all_joint_names"])
            arm_indices = np.asarray(group["actions/arm_joint_indices"][()], dtype=int).reshape(-1)
            arm_names = _strings(group["actions/arm_joint_names"])
        except Exception as error:
            self.issue("schema.action_metadata", f"{path}/actions", str(error))
            return
        if len(joint_names) != 9 or len(set(joint_names)) != 9:
            self.issue("schema.joint_names", f"{path}/actions/all_joint_names", "Franka trial requires nine unique joint names")
        if arm_indices.tolist() != list(range(7)) or arm_names != joint_names[:7]:
            self.issue("schema.arm_indices", f"{path}/actions/arm_joint_indices", "Franka arm action mapping must be joints 0..6")
        specs = online_row_specs(len(joint_names), len(arm_indices))
        step_count = _int(group.attrs.get("step_count", -1))
        if step_count <= 0:
            self.issue("trial.step_count", f"{path}/@step_count", "trial must contain control rows")
            return
        for dataset_path, spec in specs.items():
            dataset = group.get(dataset_path)
            full_path = f"{path}/{dataset_path}"
            if dataset is None or not hasattr(dataset, "shape"):
                self.issue("schema.step_dataset", full_path, "required step-level dataset is missing")
                continue
            expected_shape = (step_count, *tuple(spec.tail_shape))
            if tuple(dataset.shape) != expected_shape:
                self.issue("schema.step_shape", full_path, f"expected {expected_shape}, got {tuple(dataset.shape)}")
                continue
            if getattr(dataset.dtype, "kind", "") not in {"O", "S", "U"} and not _dataset_all_finite(dataset):
                self.issue("schema.step_finite", full_path, "numeric step values must all be finite")
        required_exact_paths = (
            "controller/condition_id",
            "controller/objective_mode",
            "controller/lambda_s",
            "policy/bc_raw_action_5d",
            "policy/bc_checkpoint_sha256",
            "cbf/current_correction_rad_s",
            "cbf/previous_correction_rad_s",
            "cbf/correction_delta_rad_s",
            "cbf/correction_memory_valid",
            "response/phase",
            "response/smooth_tail_active",
            "response/time_since_response_onset_s",
            "response/time_since_recovery_onset_s",
            "recovery/active",
            "recovery/onset_now",
            "recovery/end_now",
            "recovery/state",
            "recovery/control_authority",
            "recovery/bc_resumed",
            "recovery/stable_task_resumption",
            "crossing/intended_hand",
            "crossing/non_crossing_hand",
            "object/cube_linear_velocity_world_m_s",
            "safety/self_collision",
            "safety/static_collision",
            "safety/static_contact_semantic_class",
            "provenance/decision_context_available",
        )
        if any(group.get(name) is None for name in required_exact_paths):
            # The schema loop normally reports these too; this code remains
            # explicit so a future permissive row schema cannot weaken the
            # research contract silently.
            for name in required_exact_paths:
                if group.get(name) is None:
                    self.issue("schema.ac_step_dataset", f"{path}/{name}", "A/C study semantic is required")
            return
        condition_id = str(_scalar(group.attrs.get("condition_id", "")))
        condition = CONDITIONS.get(condition_id)
        condition_rows = _strings(group["controller/condition_id"])
        objective_rows = _strings(group["controller/objective_mode"])
        lambda_rows = np.asarray(group["controller/lambda_s"][()], dtype=float)
        policy_sha_rows = _strings(group["policy/bc_checkpoint_sha256"])
        if any(value != condition_id for value in condition_rows):
            self.issue("condition.step_mutation", f"{path}/controller/condition_id", "condition changed within the preassigned trial")
        if condition and (
            any(value != condition.objective_mode for value in objective_rows)
            or not np.allclose(lambda_rows, condition.lambda_s, rtol=0.0, atol=0.0)
        ):
            self.issue("condition.step_config", f"{path}/controller", "objective/lambda changed or disagrees with the condition")
        if any(value != POLICY_SHA256 for value in policy_sha_rows):
            self.issue("policy.step_sha", f"{path}/policy/bc_checkpoint_sha256", "step did not use the pinned Frozen BC checkpoint")
        bc_raw = np.asarray(group["policy/bc_raw_action_5d"][()], dtype=float)
        base_bc = np.asarray(group["actions/bc_task_action"][()], dtype=float)
        if bc_raw.shape != base_bc.shape or not np.allclose(bc_raw, base_bc, rtol=0.0, atol=1e-7):
            self.issue("policy.action_provenance", f"{path}/policy/bc_raw_action_5d", "BC raw action and action-trace policy command disagree")
        self._validate_step_clocks(group, path, step_count)
        self._validate_action_provenance(group, path, step_count, len(joint_names), arm_indices)
        self._validate_cbf_rows(group, path, condition_id)
        self._validate_tracking_rows(group, path)
        self._validate_recovery_rows(group, path)
        self._validate_response_rows(group, path)
        self._validate_fsm_rows(group, path)

    def _validate_step_clocks(self, group: Any, path: str, step_count: int) -> None:
        import numpy as np

        source = np.asarray(group["control/source_step"][()], dtype=np.int64)
        result = np.asarray(group["control/result_step"][()], dtype=np.int64)
        sim = np.asarray(group["control/sim_time"][()], dtype=float)
        if source.shape != (step_count,) or np.any(np.diff(source) != 1) or np.any(result != source + 1):
            self.issue("clock.step", f"{path}/control", "source/result control steps must be contiguous t/t+1")
        if np.any(np.diff(sim) < -1e-12) or np.any(sim < 0):
            self.issue("clock.simulation", f"{path}/control/sim_time", "simulation time must be non-negative and monotonic")
        expected_dt = _float(self.file.attrs.get("physics_dt_s", math.nan))
        if (
            sim.size > 1
            and (
                not math.isfinite(expected_dt)
                or not np.allclose(
                    np.diff(sim), expected_dt, rtol=0.0, atol=1e-6
                )
            )
        ):
            self.issue("clock.physics_dt", f"{path}/control/sim_time", "control-row simulation increments differ from the frozen physics timestep")
        ordered_names = (
            "observation_monotonic_ns",
            "preflight_monotonic_ns",
            "policy_inference_started_monotonic_ns",
            "policy_output_monotonic_ns",
            "rmpflow_output_monotonic_ns",
            "cbf_output_monotonic_ns",
            "action_command_monotonic_ns",
            "next_observation_monotonic_ns",
            "post_step_monotonic_ns",
        )
        try:
            clocks = [np.asarray(group[f"control/{name}"][()], dtype=np.int64) for name in ordered_names]
        except Exception as error:
            self.issue("clock.datasets", f"{path}/control", str(error))
            return
        if any(np.any(clock <= 0) for clock in clocks) or any(np.any(later < earlier) for earlier, later in zip(clocks, clocks[1:])):
            self.issue("clock.order", f"{path}/control", "per-step monotonic timestamps are absent or out of pipeline order")

    def _validate_action_provenance(
        self,
        group: Any,
        path: str,
        step_count: int,
        joint_count: int,
        arm_indices: Any,
    ) -> None:
        import numpy as np

        required = {
            "nominal": "actions/nominal_rmpflow_joint_velocities_rad_s",
            "filtered": "actions/cbf_filtered_joint_velocities_rad_s",
            "applied": "actions/applied_joint_velocities_rad_s",
        }
        commands: dict[str, Any] = {}
        for label, name in required.items():
            dataset = group.get(name)
            mask = group.get(name + "_mask")
            if dataset is None or mask is None:
                self.issue("action.provenance", f"{path}/{name}", f"full-vector {label} command and mask are required")
                continue
            values = np.asarray(dataset[()], dtype=float)
            masks = np.asarray(mask[()], dtype=int)
            if values.shape != (step_count, joint_count) or masks.shape != values.shape:
                self.issue("action.dimension", f"{path}/{name}", "command must use the full joint vector")
                continue
            if np.any((masks[:, arm_indices] != 1)):
                self.issue("action.arm_mask", f"{path}/{name}_mask", "all seven arm velocities must be present")
            commands[label] = values
        arm_applied = np.asarray(group["actions/applied_arm_joint_velocities_rad_s"][()], dtype=float)
        arm_mask = np.asarray(group["actions/applied_arm_joint_velocities_mask"][()], dtype=int)
        if arm_applied.shape != (step_count, 7) or np.any(arm_mask != 1):
            self.issue("action.applied_arm", f"{path}/actions/applied_arm_joint_velocities_rad_s", "applied arm action must be a valid seven-vector in rad/s")
        elif "applied" in commands and not np.allclose(arm_applied, commands["applied"][:, arm_indices], rtol=0.0, atol=1e-9):
            self.issue("action.applied_relation", f"{path}/actions", "applied arm projection disagrees with full applied command")
        if np.any(np.abs(arm_applied) > 2.0 + 1e-6):
            self.issue("action.speed_limit", f"{path}/actions/applied_arm_joint_velocities_rad_s", "applied action exceeds 2 rad/s")
        if np.any(np.asarray(group["actions/action_pipeline_complete"][()], dtype=int) != 1):
            self.issue("action.pipeline", f"{path}/actions/action_pipeline_complete", "every row must complete BC -> RMPFlow -> CBF -> applied provenance")

    def _validate_cbf_rows(self, group: Any, path: str, condition_id: str) -> None:
        import numpy as np

        required_one = (
            "cbf/intervention_available",
            "cbf/feasible",
            "cbf/solver_converged",
            "safety/geometry_valid",
            "actions/action_pipeline_complete",
        )
        required_zero = (
            "cbf/fallback_applied",
            "cbf/infeasibility_proven",
            "cbf/relaxed_solution_applied",
            "cbf/intentional_human_absence",
            "safety/collision",
            "safety/self_collision",
        )
        for name in required_one:
            values = np.asarray(group[name][()], dtype=int)
            if np.any(values != 1):
                self.issue("safety.unsafe_row", f"{path}/{name}", "all rows must equal one")
        for name in required_zero:
            values = np.asarray(group[name][()], dtype=int)
            if np.any(values != 0):
                self.issue("safety.unsafe_row", f"{path}/{name}", "all rows must equal zero")
        static_collision = np.asarray(group["safety/static_collision"][()], dtype=int)
        static_class = _strings(group["safety/static_contact_semantic_class"])
        if any(not value for value in static_class):
            self.issue("safety.static_semantics", f"{path}/safety/static_contact_semantic_class", "every row needs an explicit static-contact semantic class")
        if any(flag and value in {"none", "identity_unavailable", ""} for flag, value in zip(static_collision, static_class)):
            self.issue("safety.static_semantics", f"{path}/safety/static_contact_semantic_class", "static collision lacks exact semantic identity")
        active = np.asarray(group["cbf/active"][()], dtype=int)
        count = np.asarray(group["cbf/constraint_count"][()], dtype=int)
        tracked = np.asarray(group["cbf/tracked_hand_count"][()], dtype=int)
        valid = np.asarray(group["cbf/valid_hand_count"][()], dtype=int)
        if np.any((active != (count > 0).astype(int)) | (count < 0) | (count > 2) | (tracked != 2) | (valid != 2)):
            self.issue("cbf.diagnostics", f"{path}/cbf", "active/count/tracked/valid hand diagnostics are inconsistent")
        for name in ("cbf/fail_closed_on_invalid_active_hand", "cbf/stop_on_infeasible"):
            if np.any(np.asarray(group[name][()], dtype=int) != 1):
                self.issue("cbf.fail_closed_config", f"{path}/{name}", "fail-closed safety setting changed within trial")
        if np.any(np.asarray(group["cbf/max_prediction_buffer_m"][()], dtype=float) != 0.08):
            self.issue("cbf.prediction_buffer_config", f"{path}/cbf/max_prediction_buffer_m", "maximum prediction buffer must remain 0.08 m")
        if any(value != "safe_solution_no_fallback" for value in _strings(group["cbf/fail_closed_status"])):
            self.issue("cbf.fail_closed_status", f"{path}/cbf/fail_closed_status", "every collected row must have a safe solved command without fallback")
        self._validate_constraint_evidence(group, path, count, active)
        current = np.asarray(group["cbf/current_correction_rad_s"][()], dtype=float)
        previous = np.asarray(group["cbf/previous_correction_rad_s"][()], dtype=float)
        delta = np.asarray(group["cbf/correction_delta_rad_s"][()], dtype=float)
        if not np.allclose(delta, current - previous, rtol=0.0, atol=1e-8):
            self.issue("cbf.correction_delta", f"{path}/cbf/correction_delta_rad_s", "delta must equal current correction minus previous correction")
        memory_valid = np.asarray(group["cbf/correction_memory_valid"][()], dtype=int)
        tail = np.asarray(group["response/smooth_tail_active"][()], dtype=int)
        if np.any(tail & (active != 0)) or np.any(tail & (memory_valid != 1)):
            self.issue("cbf.smooth_tail", f"{path}/response/smooth_tail_active", "smooth tail requires inactive constraints and valid correction memory")
        if condition_id == "A_reactive" and np.any(tail != 0):
            self.issue("cbf.a_tail", f"{path}/response/smooth_tail_active", "A must not retain a smooth correction tail")
        nominal = np.asarray(group["actions/nominal_rmpflow_joint_velocities_rad_s"][()], dtype=float)
        filtered = np.asarray(group["actions/cbf_filtered_joint_velocities_rad_s"][()], dtype=float)
        inactive_no_tail = (active == 0) & (tail == 0)
        if np.any(inactive_no_tail) and not np.allclose(filtered[inactive_no_tail], nominal[inactive_no_tail], rtol=0.0, atol=1e-9):
            self.issue("cbf.inactive_passthrough", f"{path}/actions/cbf_filtered_joint_velocities_rad_s", "inactive non-tail CBF rows must equal nominal RMPFlow")

    def _validate_constraint_evidence(self, group: Any, path: str, counts: Any, active: Any) -> None:
        import numpy as np

        evidence_rows = _strings(group["cbf/constraint_evidence_json"])
        min_barriers = np.asarray(group["cbf/min_barrier_value_m"][()], dtype=float)
        max_buffers = np.asarray(group["cbf/max_active_prediction_buffer_m"][()], dtype=float)
        identity = np.asarray(group["cbf/active_constraint_identity_available"][()], dtype=int)
        active_hands = _strings(group["cbf/active_hand"])
        active_links = _strings(group["cbf/active_robot_link"])
        active_colliders = _strings(group["cbf/active_robot_collider"])
        identity_semantics = _strings(group["cbf/identity_semantics"])
        summaries = _strings(group["cbf/active_constraint_summary_json"])
        required = {
            "hand",
            "closest_link",
            "closest_collider_path",
            "raw_surface_gap_m",
            "prediction_buffer_m",
            "effective_safe_gap_m",
            "barrier_value_m",
            "buffered_current_gap_m",
            "closing_speed_mps",
            "lower_bound_mps",
            "nominal_residual_mps",
            "filtered_residual_mps",
        }
        for index, raw in enumerate(evidence_rows):
            try:
                evidence = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                evidence = None
            row_path = f"{path}/cbf/constraint_evidence_json[{index}]"
            if not isinstance(evidence, list) or len(evidence) != int(counts[index]):
                self.issue("cbf.constraint_evidence", row_path, "evidence list length must equal constraint_count")
                continue
            if identity[index] != int(bool(active[index])):
                self.issue("cbf.constraint_identity", row_path, "exact identity availability must equal CBF constraint activity")
            barriers: list[float] = []
            buffers: list[float] = []
            expected_constraints: list[dict[str, Any]] = []
            for item in evidence:
                if not isinstance(item, Mapping) or not required.issubset(item):
                    self.issue("cbf.constraint_evidence", row_path, "constraint evidence fields are incomplete")
                    continue
                if item.get("hand") not in {"left", "right"} or not str(item.get("closest_link", "")) or not str(item.get("closest_collider_path", "")):
                    self.issue("cbf.constraint_identity", row_path, "constraint hand/link/collider identity is invalid")
                numeric = {name: _float(item.get(name, math.nan)) for name in required - {"hand", "closest_link", "closest_collider_path"}}
                if any(not math.isfinite(value) for value in numeric.values()):
                    self.issue("cbf.constraint_evidence", row_path, "constraint evidence contains a non-finite value")
                    continue
                prediction_buffer = numeric["prediction_buffer_m"]
                if not 0.0 <= prediction_buffer <= 0.08 + 1e-12:
                    self.issue("cbf.prediction_buffer", row_path, "prediction buffer lies outside [0, 0.08] m")
                if not math.isclose(numeric["effective_safe_gap_m"], 0.05 + prediction_buffer, rel_tol=0.0, abs_tol=1e-7):
                    self.issue("cbf.effective_gap", row_path, "effective safe gap must equal 0.05 m plus prediction buffer")
                if numeric["filtered_residual_mps"] < -1e-7:
                    self.issue("cbf.constraint_residual", row_path, "filtered command violates a CBF constraint")
                barriers.append(numeric["barrier_value_m"])
                buffers.append(prediction_buffer)
                expected_constraints.append(
                    {
                        "hand": item.get("hand"),
                        "robot_link": item.get("closest_link"),
                        "collider_path": item.get("closest_collider_path"),
                        "surface_gap_m": numeric["raw_surface_gap_m"],
                    }
                )
            expected_identity_semantics = (
                "exact_reconstruction_from_frozen_cbf_filter_input_v1"
                if active[index]
                else "inactive_or_identity_unavailable"
            )
            if (
                active_hands[index] != ",".join(str(item["hand"]) for item in expected_constraints)
                or active_links[index] != ",".join(str(item["robot_link"]) for item in expected_constraints)
                or active_colliders[index] != ",".join(str(item["collider_path"]) for item in expected_constraints)
                or identity_semantics[index] != expected_identity_semantics
            ):
                self.issue("cbf.constraint_summary", row_path, "active hand/link/collider summaries differ from lossless evidence")
            try:
                summary = json.loads(summaries[index])
            except (TypeError, ValueError, json.JSONDecodeError):
                summary = None
            if (
                not isinstance(summary, Mapping)
                or _int(summary.get("constraint_count", -1)) != int(counts[index])
                or _bool(summary.get("exact_identity_available", 0)) != bool(active[index])
                or summary.get("constraints") != expected_constraints
                or not str(summary.get("derivation", ""))
            ):
                self.issue("cbf.constraint_summary", f"{path}/cbf/active_constraint_summary_json[{index}]", "summary does not reconstruct the lossless constraint evidence")
            expected_barrier = min(barriers, default=10.0)
            expected_buffer = max(buffers, default=0.0)
            if not math.isclose(min_barriers[index], expected_barrier, rel_tol=0.0, abs_tol=1e-9):
                self.issue("cbf.barrier_summary", row_path, "minimum barrier summary disagrees with evidence")
            if not math.isclose(max_buffers[index], expected_buffer, rel_tol=0.0, abs_tol=1e-9):
                self.issue("cbf.buffer_summary", row_path, "prediction buffer summary disagrees with evidence")

    def _validate_tracking_rows(self, group: Any, path: str) -> None:
        import numpy as np

        for namespace in ("human", "human_next"):
            valid_mask = np.asarray(group[f"{namespace}/valid_mask"][()], dtype=int)
            if np.any(valid_mask != 1):
                self.issue("tracking.invalid", f"{path}/{namespace}/valid_mask", "head and both controllers must be valid on every collected control row")
            for entity in ("head", "left", "right"):
                pose_valid = np.asarray(group[f"{namespace}/{entity}_pose_valid"][()], dtype=int)
                tracked = np.asarray(group[f"{namespace}/{entity}_position_tracked"][()], dtype=int)
                known = np.asarray(group[f"{namespace}/{entity}_tracking_status_known"][()], dtype=int)
                switched = np.asarray(group[f"{namespace}/{entity}_source_switched"][()], dtype=int)
                ages = np.asarray(group[f"{namespace}/{entity}_pose_age_ms"][()], dtype=float)
                max_age = _float(self.config.get("tracking", {}).get("maximum_pose_age_ms", math.nan))
                status_valid = ((known == 1) & (tracked == 1)) | ((known == 0) & (tracked == -1))
                if np.any(pose_valid != 1) or np.any(~status_valid):
                    self.issue("tracking.invalid", f"{path}/{namespace}/{entity}", "pose/status fields violate tracked-or-explicitly-unknown semantics")
                if np.any(switched != 0):
                    self.issue("tracking.source_switch", f"{path}/{namespace}/{entity}_source_switched", "tracking source switched inside a trial")
                if math.isfinite(max_age) and np.any((ages < 0) | (ages > max_age + 1e-9)):
                    self.issue("tracking.stale", f"{path}/{namespace}/{entity}_pose_age_ms", f"pose age exceeds {max_age} ms")
        for hand in ("left", "right"):
            if np.any(np.asarray(group[f"human/{hand}_velocity_valid"][()], dtype=int) != 1):
                self.issue("tracking.velocity", f"{path}/human/{hand}_velocity_valid", "controller velocity is invalid")

    def _validate_recovery_rows(self, group: Any, path: str) -> None:
        """Bind the logged Recovery labels to the bridge's ownership semantics.

        ``info_after`` is sampled after the environment commits a Recovery
        handoff.  Consequently, the handoff transition has an ``inactive``
        post-step bridge state while the command applied during that same
        transition still had Recovery control authority.  That single
        boundary row is intentional; treating state alone as activity would
        silently drop it from the response episode.
        """

        import numpy as np

        active = np.asarray(group["recovery/active"][()], dtype=int)
        authority = np.asarray(
            group["recovery/control_authority"][()], dtype=int
        )
        states = _strings(group["recovery/state"])
        control_modes = _strings(group["control/control_mode"])
        allowed_states = {
            "inactive",
            "place_lift",
            "place_transport",
            "place_descend",
            "regrasp_prepose",
        }
        if (
            np.any((active != 0) & (active != 1))
            or np.any((authority != 0) & (authority != 1))
        ):
            self.issue(
                "recovery.binary",
                f"{path}/recovery",
                "Recovery activity and control-authority flags must be binary",
            )
        invalid_states = sorted(set(states) - allowed_states)
        if invalid_states:
            self.issue(
                "recovery.state",
                f"{path}/recovery/state",
                f"unknown state-aware Recovery stages: {invalid_states}",
            )
            return

        state_active = np.asarray(
            [state != "inactive" for state in states], dtype=bool
        )
        expected_active = state_active | (authority == 1)
        if np.any((active == 1) != expected_active):
            self.issue(
                "recovery.activity_semantics",
                f"{path}/recovery/active",
                "active must equal (post-step bridge state active OR applied Recovery control authority)",
            )
        if np.any((authority == 1) & (np.asarray(control_modes) != "nominal_bc")):
            self.issue(
                "recovery.authority_mode",
                f"{path}/recovery/control_authority",
                "Recovery control authority is incompatible with protocol_hold",
            )

        # A state-active/authority-off row is the activation boundary: the
        # request was raised after the current nominal command.  It may occur
        # only at the start of a Recovery-active run and must hand authority
        # to Recovery on the following row.
        activation_rows = np.flatnonzero(state_active & (authority == 0))
        for raw_index in activation_rows:
            index = int(raw_index)
            previous_active = index > 0 and active[index - 1] == 1
            next_has_authority = (
                index + 1 < len(active) and authority[index + 1] == 1
            )
            if previous_active or not next_has_authority:
                self.issue(
                    "recovery.activation_boundary",
                    f"{path}/recovery/control_authority[{index}]",
                    "an active bridge without authority must be a one-row request boundary followed by Recovery authority",
                )

        # Conversely, state-inactive/authority-on is the post-action handoff
        # boundary (or a Recovery timeout, which makes the trial invalid via
        # its outcome metadata).  It must close an existing active run and
        # cannot retain authority on the next transition.
        handoff_rows = np.flatnonzero((~state_active) & (authority == 1))
        for raw_index in handoff_rows:
            index = int(raw_index)
            previous_bridge_active = (
                index > 0
                and active[index - 1] == 1
                and states[index - 1] != "inactive"
            )
            next_released = (
                index + 1 >= len(active)
                or (active[index + 1] == 0 and authority[index + 1] == 0)
            )
            if not previous_bridge_active or not next_released:
                self.issue(
                    "recovery.handoff_boundary",
                    f"{path}/recovery/state[{index}]",
                    "inactive state with Recovery authority must be the single closing row of an active bridge run",
                )

    def _validate_response_rows(self, group: Any, path: str) -> None:
        import numpy as np

        response_mode = self._response_mode(group)
        partial_response = response_mode in {"no_onset", "no_stable"}
        phases = _strings(group["response/phase"])
        invalid = sorted(set(phases) - set(_RESPONSE_PHASES))
        if invalid:
            self.issue("response.phase_value", f"{path}/response/phase", f"unknown phases: {invalid}")
            return
        if not phases or phases[0] != ResponsePhase.PRE_RESPONSE.value:
            self.issue("response.phase_start", f"{path}/response/phase", "response timeline must start PRE_RESPONSE")
        trial_states = _strings(group["control/trial_state"])
        if partial_response:
            phase_window_end = next(
                (
                    index
                    for index, state in enumerate(trial_states)
                    if state
                    == TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value
                ),
                len(phases),
            )
        else:
            phase_window_end = len(phases)
        episode_phases = phases[:phase_window_end]
        first_non_pre = next((value for value in episode_phases if value != ResponsePhase.PRE_RESPONSE.value), "")
        if first_non_pre and first_non_pre != ResponsePhase.CBF_ACTIVE.value:
            self.issue("response.phase_onset", f"{path}/response/phase", "first post-onset phase must be CBF_ACTIVE")
        if response_mode == "complete" and (
            ResponsePhase.CBF_ACTIVE.value not in episode_phases
            or ResponsePhase.STABLE_TASK_RESUMPTION.value not in episode_phases
        ):
            self.issue("response.phase_coverage", f"{path}/response/phase", "confirmed CBF_ACTIVE and STABLE_TASK_RESUMPTION are required")
        elif response_mode == "no_stable" and ResponsePhase.CBF_ACTIVE.value not in episode_phases:
            self.issue(
                "response.phase_coverage",
                f"{path}/response/phase",
                "a stable-timeout trial still requires its confirmed CBF_ACTIVE rows",
            )
        if partial_response and ResponsePhase.STABLE_TASK_RESUMPTION.value in episode_phases:
            self.issue(
                "response.partial_stable",
                f"{path}/response/phase",
                "an incomplete-response trial cannot contain a stable-resumption phase",
            )
        seen_stable = False
        for index, phase in enumerate(phases):
            if phase == ResponsePhase.STABLE_TASK_RESUMPTION.value:
                seen_stable = True
            elif seen_stable:
                self.issue("response.stable_regression", f"{path}/response/phase[{index}]", "timeline left stable task resumption")
                break
        active = np.asarray(group["cbf/active"][()], dtype=int)
        intervention = np.asarray(group["cbf/intervention_norm_radps"][()], dtype=float)
        tail = np.asarray(group["response/smooth_tail_active"][()], dtype=int)
        recovery = np.asarray(group["recovery/active"][()], dtype=int)
        recovery_onset_now = np.asarray(
            group["recovery/onset_now"][()], dtype=int
        )
        recovery_end_now = np.asarray(
            group["recovery/end_now"][()], dtype=int
        )
        resumed = np.asarray(group["recovery/bc_resumed"][()], dtype=int)
        stable = np.asarray(group["recovery/stable_task_resumption"][()], dtype=int)
        control_modes = _strings(group["control/control_mode"])
        expected_recovery_onset_now = (recovery == 1) & np.concatenate(
            (np.asarray([True]), recovery[:-1] == 0)
        )
        expected_recovery_end_now = (recovery == 0) & np.concatenate(
            (np.asarray([False]), recovery[:-1] == 1)
        )
        if (
            np.any((recovery_onset_now != 0) & (recovery_onset_now != 1))
            or np.any((recovery_end_now != 0) & (recovery_end_now != 1))
            or np.any(
                recovery_onset_now
                != expected_recovery_onset_now.astype(int)
            )
            or np.any(
                recovery_end_now != expected_recovery_end_now.astype(int)
            )
        ):
            self.issue(
                "response.recovery_edges",
                f"{path}/recovery",
                "onset_now/end_now must exactly encode rising/falling edges of the raw Recovery activity/authority mask",
            )
        first_stable = next((index for index, phase in enumerate(episode_phases) if phase == ResponsePhase.STABLE_TASK_RESUMPTION.value), -1)
        expected_resumed = np.asarray(
            [
                mode == "nominal_bc"
                and phase
                in {
                    ResponsePhase.BC_RESUMED.value,
                    ResponsePhase.STABLE_TASK_RESUMPTION.value,
                }
                for mode, phase in zip(control_modes, phases)
            ],
            dtype=int,
        )
        expected_stable = np.asarray(
            [
                phase == ResponsePhase.STABLE_TASK_RESUMPTION.value
                for phase in phases
            ],
            dtype=int,
        )
        if (
            np.any((resumed != 0) & (resumed != 1))
            or np.any(resumed != expected_resumed)
        ):
            self.issue(
                "response.bc_resumed_semantics",
                f"{path}/recovery/bc_resumed",
                "BC-resumed must reflect nominal-BC authority and the BC_RESUMED/STABLE phase exactly",
            )
        if (
            np.any((stable != 0) & (stable != 1))
            or np.any(stable != expected_stable)
        ):
            self.issue(
                "response.stable_semantics",
                f"{path}/recovery/stable_task_resumption",
                "stable-task flag must equal the terminal response phase exactly",
            )
        for index, phase in enumerate(phases):
            valid = True
            if phase == ResponsePhase.CBF_ACTIVE.value:
                valid = active[index] == 1
            elif phase == ResponsePhase.SMOOTH_TAIL.value:
                valid = active[index] == 0 and tail[index] == 1
            elif phase == ResponsePhase.RECOVERY_ACTIVE.value:
                valid = active[index] == 0 and tail[index] == 0 and recovery[index] == 1
            elif phase == ResponsePhase.BC_RESUMED.value:
                clear_safety_authority = bool(
                    active[index] == 0
                    and tail[index] == 0
                    and recovery[index] == 0
                )
                if index < phase_window_end:
                    valid = bool(
                        clear_safety_authority
                        and resumed[index] == 1
                        and control_modes[index] == "nominal_bc"
                    )
                elif control_modes[index] == "nominal_bc":
                    # Once a timed-out response window has closed, the
                    # detector's last phase remains raw diagnostic state.  A
                    # nominal neutral-return row still has BC authority.
                    valid = bool(clear_safety_authority and resumed[index] == 1)
                elif control_modes[index] == "protocol_hold":
                    # PAUSE/query rows deliberately remove BC authority while
                    # retaining the detector's post-window BC_RESUMED label.
                    valid = bool(clear_safety_authority and resumed[index] == 0)
                else:
                    valid = False
            elif phase == ResponsePhase.STABLE_TASK_RESUMPTION.value:
                valid = (
                    active[index] == 0
                    and tail[index] == 0
                    and recovery[index] == 0
                    and stable[index] == 1
                )
                if index == first_stable:
                    # The transition itself proves 0.5 s of clean nominal-BC
                    # resumption.  The terminal label is then latched; later
                    # query/hold rows legitimately have no BC authority.
                    valid = bool(
                        valid
                        and resumed[index] == 1
                        and control_modes[index] == "nominal_bc"
                        and intervention[index] <= 0.01 + 1e-9
                    )
                elif control_modes[index] == "nominal_bc":
                    valid = bool(valid and resumed[index] == 1)
                elif control_modes[index] == "protocol_hold":
                    valid = bool(valid and resumed[index] == 0)
                else:
                    valid = False
            if not valid:
                self.issue("response.phase_diagnostics", f"{path}/response/phase[{index}]", "phase conflicts with CBF/tail/Recovery/BC diagnostics")
                break
        active_signal = (active == 1) & (intervention >= 0.05 - 1e-12)
        episode_active_signal = active_signal[:phase_window_end]
        # The detector labels the onset-candidate frame CBF_ACTIVE immediately;
        # confirmation is reached only on the third consecutive qualifying
        # frame.  Derive both boundaries from the raw signal, rather than
        # treating the first CBF_ACTIVE label as the confirmation frame.
        candidate = next(
            (
                index
                for index in range(max(0, len(episode_active_signal) - 2))
                if np.all(episode_active_signal[index:index + 3])
            ),
            -1,
        )
        confirmed = candidate + 2 if candidate >= 0 else -1
        decision_available = np.asarray(
            group["provenance/decision_context_available"][()], dtype=int
        )
        candidate_starts = episode_active_signal & np.concatenate(
            (
                np.asarray([True], dtype=bool),
                ~episode_active_signal[:-1],
            )
        )
        candidate_cutoff = (
            candidate if candidate >= 0 else len(episode_active_signal) - 1
        )
        expected_decision = candidate_starts & (
            np.arange(len(episode_active_signal)) <= candidate_cutoff
        )
        if np.any((decision_available != 0) & (decision_available != 1)) or np.any(
            (decision_available[:phase_window_end] == 1) != expected_decision
        ):
            self.issue(
                "decision_context.cutoff",
                f"{path}/provenance/decision_context_available",
                "decision context availability must mark pre-confirmation raw onset candidates exactly",
            )
        if candidate < 0 and response_mode != "no_onset":
            self.issue("response.onset_confirmation", f"{path}/response/phase", "confirmed onset requires intervention>=0.05 for three consecutive active frames")
        elif candidate >= 0 and response_mode == "no_onset":
            self.issue(
                "response.partial_onset",
                f"{path}/response/phase[{candidate}]",
                "unconfirmed-onset sentinel conflicts with a qualifying three-frame onset run",
            )
        else:
            if candidate >= 0 and decision_available[candidate] != 1:
                self.issue("decision_context.onset", f"{path}/provenance/decision_context_available[{candidate}]", "confirmed episode lacks a pre-outcome decision context at its onset candidate")
            # Before confirmation the phase mirrors the raw onset signal, so
            # failed one/two-frame candidates may be preserved as
            # CBF_ACTIVE -> PRE_RESPONSE transitions.  Once the first
            # qualifying run begins, PRE_RESPONSE can never recur.
            for index in range(max(0, confirmed)):
                expected = (
                    ResponsePhase.CBF_ACTIVE.value
                    if active_signal[index]
                    else ResponsePhase.PRE_RESPONSE.value
                )
                if phases[index] != expected:
                    self.issue("response.transient_candidate", f"{path}/response/phase[{index}]", "pre-confirmation phase does not preserve the raw onset-candidate signal")
                    break
            if confirmed >= 0 and phases[confirmed] != ResponsePhase.CBF_ACTIVE.value:
                self.issue("response.onset_candidate", f"{path}/response/phase[{confirmed}]", "third qualifying frame must remain CBF_ACTIVE at confirmation")
            later_pre = next(
                (
                    index
                    for index in range(max(0, candidate), len(phases))
                    if phases[index] == ResponsePhase.PRE_RESPONSE.value
                ),
                -1,
            )
            if candidate >= 0 and later_pre >= 0:
                self.issue("response.phase_regression", f"{path}/response/phase[{later_pre}]", "timeline returned to PRE_RESPONSE after the confirmed episode's onset candidate")
        if response_mode == "no_onset":
            for index, signal in enumerate(episode_active_signal):
                expected = (
                    ResponsePhase.CBF_ACTIVE.value
                    if signal
                    else ResponsePhase.PRE_RESPONSE.value
                )
                if phases[index] != expected:
                    self.issue(
                        "response.transient_candidate",
                        f"{path}/response/phase[{index}]",
                        "an unconfirmed timeline may contain only raw one/two-frame CBF candidates and PRE_RESPONSE regressions",
                    )
                    break
        sim = np.asarray(group["control/sim_time"][()], dtype=float)
        source = np.asarray(group["control/source_step"][()], dtype=np.int64)
        if first_stable >= 0:
            stable_signal = (active == 0) & (tail == 0) & (recovery == 0) & (resumed == 1) & (intervention <= 0.01 + 1e-9)
            start = first_stable
            while start > 0 and stable_signal[start - 1]:
                start -= 1
            if sim[first_stable] - sim[start] < 0.5 - 1e-6:
                self.issue("response.stable_duration", f"{path}/response/phase", "stable signal was not held for 0.5 seconds before confirmation")
        elapsed_response = np.asarray(group["response/time_since_response_onset_s"][()], dtype=float)
        elapsed_recovery = np.asarray(group["response/time_since_recovery_onset_s"][()], dtype=float)
        if np.any(elapsed_response < -1.0) or np.any(elapsed_recovery < -1.0):
            self.issue("response.elapsed", f"{path}/response", "elapsed clocks must use -1 before onset and non-negative values after onset")
        if response_mode == "no_onset":
            if np.any(elapsed_response[:phase_window_end] != -1.0):
                self.issue(
                    "response.elapsed_onset",
                    f"{path}/response/time_since_response_onset_s",
                    "unconfirmed-onset trial must retain the -1 elapsed sentinel",
                )
        elif candidate >= 0:
            expected_elapsed = sim - sim[candidate]
            confirmed_and_after = np.arange(len(sim)) >= confirmed
            if np.any(elapsed_response[~confirmed_and_after] != -1.0) or not np.allclose(elapsed_response[confirmed_and_after], expected_elapsed[confirmed_and_after], rtol=0.0, atol=1e-6):
                self.issue("response.elapsed_onset", f"{path}/response/time_since_response_onset_s", "must be -1 until confirmation, then map to the onset-candidate timestamp")
        recovery_window_start = candidate if candidate >= 0 else phase_window_end
        recovery_indices = np.flatnonzero(
            recovery[recovery_window_start:phase_window_end] == 1
        ) + recovery_window_start
        first_recovery = -1
        last_recovery = -1
        if response_mode == "no_onset":
            if np.any(elapsed_recovery[:phase_window_end] != -1.0):
                self.issue(
                    "response.elapsed_recovery",
                    f"{path}/response/time_since_recovery_onset_s",
                    "Recovery elapsed time remains undefined before a confirmed response onset",
                )
        elif recovery_indices.size:
            first_recovery = int(recovery_indices[0])
            last_recovery = int(recovery_indices[-1])
            expected_elapsed = sim - sim[first_recovery]
            # An onset inside the two pre-confirmation candidate frames is
            # promoted only when the third CBF frame confirms the scientific
            # episode.  Its clock still references the first raw active row.
            post = np.arange(len(sim)) >= max(first_recovery, confirmed)
            if np.any(elapsed_recovery[~post] != -1.0) or not np.allclose(elapsed_recovery[post], expected_elapsed[post], rtol=0.0, atol=1e-6):
                self.issue("response.elapsed_recovery", f"{path}/response/time_since_recovery_onset_s", "does not map to Recovery onset")
        elif partial_response:
            # A later Recovery activation while finishing the task is useful
            # raw diagnostic data, but it occurs after the timeout boundary
            # and must not rewrite the scientific response summary.
            if np.any(elapsed_recovery[:phase_window_end] != -1.0):
                self.issue(
                    "response.elapsed_recovery",
                    f"{path}/response/time_since_recovery_onset_s",
                    "Recovery elapsed time must remain undefined inside a response window with no Recovery phase",
                )
        elif np.any(elapsed_recovery != -1.0):
            self.issue("response.elapsed_recovery", f"{path}/response/time_since_recovery_onset_s", "must remain -1 when Recovery never activates")
        onset_attr = _float(group.attrs.get("response_onset_simulation_time_s", math.nan))
        end_attr = _float(group.attrs.get("response_end_simulation_time_s", math.nan))
        if candidate >= 0 and not math.isclose(onset_attr, sim[candidate], rel_tol=0.0, abs_tol=1e-6):
            self.issue("response.summary_onset", f"{path}/@response_onset_simulation_time_s", "does not match step-level onset candidate")
        if candidate >= 0 and _int(group.attrs.get("response_onset_step", -1)) != int(source[candidate]):
            self.issue("response.summary_onset_step", f"{path}/@response_onset_step", "does not match the first frame of the qualifying onset run")
        if confirmed >= 0 and _int(group.attrs.get("response_onset_confirmed_step", -1)) != int(source[confirmed]):
            self.issue("response.summary_confirmed_step", f"{path}/@response_onset_confirmed_step", "does not match the third frame of the qualifying onset run")
        if first_stable >= 0 and not math.isclose(end_attr, sim[first_stable], rel_tol=0.0, abs_tol=1e-6):
            self.issue("response.summary_end", f"{path}/@response_end_simulation_time_s", "does not match stable task resumption")
        if candidate >= 0 and first_stable >= 0 and not math.isclose(
            _float(
                group.attrs.get(
                    "total_safety_response_duration_s", math.nan
                )
            ),
            sim[first_stable] - sim[candidate],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            self.issue(
                "response.summary_total_duration",
                f"{path}/@total_safety_response_duration_s",
                "must span the onset-candidate timestamp through first stable task resumption",
            )
        if first_stable >= 0 and (
            _int(group.attrs.get("stable_task_resumption_step", -1)) != int(source[first_stable])
            or _int(group.attrs.get("response_end_step_exclusive", -1)) != int(source[first_stable]) + 1
        ):
            self.issue("response.summary_end_step", path, "stable/end-exclusive response steps do not match the first stable row")
        expected_recovery_step = -1 if first_recovery < 0 else int(source[first_recovery])
        if _int(group.attrs.get("recovery_onset_step", -2)) != expected_recovery_step:
            self.issue("response.summary_recovery_step", f"{path}/@recovery_onset_step", "does not match the first Recovery-active row")
        expected_recovery_onset_sim = (
            -1.0 if first_recovery < 0 else float(sim[first_recovery])
        )
        if not math.isclose(
            _float(
                group.attrs.get(
                    "recovery_onset_simulation_time_s", math.nan
                )
            ),
            expected_recovery_onset_sim,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            self.issue(
                "response.summary_recovery_onset",
                f"{path}/@recovery_onset_simulation_time_s",
                "does not match the first Recovery-active row",
            )
        if (
            last_recovery >= 0
            and last_recovery + 1 < phase_window_end
            and recovery[last_recovery + 1] == 0
        ):
            expected_recovery_end_step = int(source[last_recovery + 1])
            expected_recovery_end_sim = float(sim[last_recovery + 1])
        else:
            expected_recovery_end_step = -1
            expected_recovery_end_sim = -1.0
        if (
            _int(group.attrs.get("recovery_end_step_exclusive", -2))
            != expected_recovery_end_step
            or not math.isclose(
                _float(
                    group.attrs.get(
                        "recovery_end_simulation_time_s", math.nan
                    )
                ),
                expected_recovery_end_sim,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            self.issue(
                "response.summary_recovery_end",
                f"{path}/@recovery_end_step_exclusive",
                "does not match the falling edge after the last Recovery-active row",
            )
        dt_s = _float(self.file.attrs.get("physics_dt_s", math.nan))
        expected_recovery_duration = float(
            np.sum(recovery[recovery_window_start:phase_window_end]) * dt_s
        )
        if not math.isclose(
            _float(group.attrs.get("recovery_duration_s", math.nan)),
            expected_recovery_duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            self.issue(
                "response.summary_recovery_duration",
                f"{path}/@recovery_duration_s",
                "must equal raw Recovery-active row count times physics_dt_s inside the response window",
            )
        if confirmed >= 0 and response_mode == "complete":
            dt_s = _float(self.file.attrs.get("physics_dt_s", math.nan))
            metric_end = first_stable if first_stable >= 0 else len(intervention) - 1
            metric_window = slice(candidate, metric_end + 1)
            expected_integrated = float(
                np.sum(intervention[metric_window]) * dt_s
            )
            if not math.isclose(
                _float(group.attrs.get("integrated_intervention_rad", math.nan)),
                expected_integrated,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                self.issue("response.integrated_intervention", f"{path}/@integrated_intervention_rad", "does not integrate the onset-candidate through stable-resumption window at the frozen physics timestep")
            correction = np.asarray(group["cbf/current_correction_rad_s"][()], dtype=float)
            windowed_correction = correction[metric_window]
            expected_variation = float(
                np.sum(
                    np.linalg.norm(
                        np.diff(windowed_correction, axis=0), axis=1
                    )
                )
            ) if len(windowed_correction) > 1 else 0.0
            if not math.isclose(
                _float(group.attrs.get("correction_total_variation_rad_s", math.nan)),
                expected_variation,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                self.issue("response.correction_variation", f"{path}/@correction_total_variation_rad_s", "does not match onset-candidate through stable-resumption correction total variation")
        if first_stable >= 0:
            phase_window_end = first_stable + 1
        episode_phases = phases[:phase_window_end]
        try:
            history = json.loads(str(_scalar(group.attrs.get("response_phase_history_json", ""))))
        except (TypeError, ValueError, json.JSONDecodeError):
            history = None
        if not isinstance(history, list) or not history:
            self.issue("response.phase_history", f"{path}/@response_phase_history_json", "phase transition history is missing")
        else:
            history_phases: list[str] = []
            history_steps: list[int] = []
            history_sim: list[float] = []
            last_step = -1
            last_sim = -math.inf
            for item in history:
                if not isinstance(item, Mapping):
                    self.issue("response.phase_history", f"{path}/@response_phase_history_json", "history entries must be objects")
                    break
                phase = str(item.get("phase", ""))
                step = _int(item.get("control_step", -1))
                item_sim = _float(item.get("simulation_time_s", math.nan))
                if phase not in _RESPONSE_PHASES or step < last_step or not math.isfinite(item_sim) or item_sim < last_sim:
                    self.issue("response.phase_history", f"{path}/@response_phase_history_json", "phase history values or clocks are invalid")
                    break
                history_phases.append(phase)
                history_steps.append(step)
                history_sim.append(item_sim)
                last_step, last_sim = step, item_sim
            transition_indices = [
                index
                for index, phase in enumerate(episode_phases)
                if index == 0 or phase != episode_phases[index - 1]
            ]
            expected_history_phases = [
                episode_phases[index] for index in transition_indices
            ]
            bad_end = bool(
                history_phases
                and (
                    history_phases[-1]
                    != ResponsePhase.STABLE_TASK_RESUMPTION.value
                )
                and response_mode == "complete"
            )
            partial_claims_stable = bool(
                history_phases
                and ResponsePhase.STABLE_TASK_RESUMPTION.value
                in history_phases
                and partial_response
            )
            if history_phases and (
                history_phases[0] != ResponsePhase.PRE_RESPONSE.value
                or bad_end
                or partial_claims_stable
                or history_phases != expected_history_phases
                or any(first == second for first, second in zip(history_phases, history_phases[1:]))
            ):
                self.issue("response.phase_history", f"{path}/@response_phase_history_json", "history must exactly encode response-window transitions and use STABLE_TASK_RESUMPTION iff the episode completed")
            elif transition_indices:
                first_ok = bool(
                    history_steps[0] == int(source[0])
                    and 0.0 <= history_sim[0] <= float(sim[0]) + 1e-6
                )
                later_ok = all(
                    history_steps[item_index] == int(source[row_index])
                    and math.isclose(
                        history_sim[item_index],
                        float(sim[row_index]),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    for item_index, row_index in enumerate(
                        transition_indices[1:], start=1
                    )
                )
                if not first_ok or not later_ok:
                    self.issue(
                        "response.phase_history_clock",
                        f"{path}/@response_phase_history_json",
                        "phase-history transitions do not map to their first control rows",
                    )
        try:
            durations = json.loads(str(_scalar(group.attrs.get("response_phase_durations_json", ""))))
        except (TypeError, ValueError, json.JSONDecodeError):
            durations = None
        if not isinstance(durations, dict) or set(durations) != set(_RESPONSE_PHASES):
            self.issue("response.phase_durations", f"{path}/@response_phase_durations_json", "durations must cover exactly all six response phases")
        else:
            if any(not math.isfinite(_float(value)) or _float(value) < 0 for value in durations.values()):
                self.issue("response.phase_durations", f"{path}/@response_phase_durations_json", "phase durations must be finite and non-negative")
            dt_s = _float(self.file.attrs.get("physics_dt_s", math.nan))
            # STABLE_TASK_RESUMPTION is a terminal detector label and is
            # deliberately retained on neutral-return, questionnaire, and
            # task-completion rows.  Those later rows are outside the safety-
            # response episode; phase duration accounting stops at the first
            # stable row (inclusive).
            for phase in _RESPONSE_PHASES:
                if phase == ResponsePhase.PRE_RESPONSE.value:
                    duration_phases = episode_phases
                elif candidate >= 0:
                    duration_phases = episode_phases[candidate:]
                else:
                    duration_phases = []
                expected_duration = duration_phases.count(phase) * dt_s
                if not math.isclose(_float(durations[phase]), expected_duration, rel_tol=0.0, abs_tol=1e-6):
                    self.issue("response.phase_durations", f"{path}/@response_phase_durations_json/{phase}", "duration does not equal its scientific response-window row count times physics_dt_s (PRE_RESPONSE retains whole pre-window audit rows)")
            summary_map = {
                "CBF_ACTIVE": "cbf_active_duration_s",
                "SMOOTH_TAIL": "smooth_tail_duration_s",
                "BC_RESUMED": "bc_resumed_duration_s",
            }
            for phase, attr_name in summary_map.items():
                expected = min(0.5, _float(durations[phase])) if phase == "BC_RESUMED" else _float(durations[phase])
                if not _same_number(group.attrs.get(attr_name, math.nan), expected):
                    self.issue("response.phase_duration_summary", f"{path}/@{attr_name}", f"does not match {phase} duration")

    def _validate_fsm_rows(self, group: Any, path: str) -> None:
        import numpy as np

        states = _strings(group["control/trial_state"])
        invalid = sorted(set(states) - set(_TRIAL_STATES))
        if invalid:
            self.issue("trial.fsm_rows", f"{path}/control/trial_state", f"unknown FSM states: {invalid}")
            return
        indices = [_TRIAL_STATES.index(state) for state in states]
        if any(later < earlier for earlier, later in zip(indices, indices[1:])):
            self.issue("trial.fsm_rows", f"{path}/control/trial_state", "step-level FSM regressed")
        present = set(states)
        row_states = {
            TrialState.WAIT_FOR_TARGET_TASK_PHASE.value,
            TrialState.SHOW_HAND_CROSSING_CUE.value,
            TrialState.TRACK_SAFETY_RESPONSE_EPISODE.value,
            TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
        }
        if not row_states.issubset(present):
            self.issue("trial.fsm_rows", f"{path}/control/trial_state", "control timeline omits a required observable FSM state")
        hold = np.asarray(group["control/control_mode"].asstr()[()] == "protocol_hold", dtype=bool)
        phase_advance = np.asarray(group["task/phase_advance_enabled"][()], dtype=int)
        query = np.asarray([state == TrialState.SHOW_MANDATORY_FEEDBACK_UI.value for state in states], dtype=bool)
        pause = np.asarray([state == TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value for state in states], dtype=bool)
        required_hold = query | pause
        if not np.any(query) or np.any(~hold[required_hold]) or np.any(phase_advance[required_hold] != 0):
            self.issue("trial.query_hold", f"{path}/control", "query/pause rows require protocol_hold and phase progression disabled")
        controller_actions = np.asarray(
            group["actions/controller_task_action"][()], dtype=float
        )
        controller_reasons = _strings(
            group["actions/controller_task_action_reason"]
        )
        recovery_authority = np.asarray(
            group["recovery/control_authority"][()], dtype=int
        )
        pipeline_complete = np.asarray(
            group["actions/action_pipeline_complete"][()], dtype=int
        )
        intervention_available = np.asarray(
            group["cbf/intervention_available"][()], dtype=int
        )
        geometry_valid = np.asarray(
            group["safety/geometry_valid"][()], dtype=int
        )
        hold_reasons_valid = np.asarray(
            [reason == "current_ee_protocol_hold" for reason in controller_reasons],
            dtype=bool,
        )
        if np.any(required_hold) and (
            np.any(controller_actions[required_hold] != 0.0)
            or np.any(~hold_reasons_valid[required_hold])
            or np.any(recovery_authority[required_hold] != 0)
            or np.any(pipeline_complete[required_hold] != 1)
            or np.any(intervention_available[required_hold] != 1)
            or np.any(geometry_valid[required_hold] != 1)
        ):
            self.issue(
                "trial.query_hold",
                f"{path}/actions",
                "query/pause rows require an exact zero current-EE hold command, its exact reason, no Recovery authority, and complete valid CBF-pipeline evidence",
            )
        stable = np.asarray(group["recovery/stable_task_resumption"][()], dtype=int)
        first_query = int(np.flatnonzero(query)[0]) if np.any(query) else -1
        first_stable = int(np.flatnonzero(stable == 1)[0]) if np.any(stable == 1) else -1
        response_mode = self._response_mode(group)
        if response_mode == "complete":
            if first_query < 0 or first_stable < 0 or first_query <= first_stable:
                self.issue("trial.query_order", f"{path}/control", "mandatory query must begin after stable task resumption")
        else:
            return_neutral = np.asarray(
                [
                    state
                    == TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value
                    for state in states
                ],
                dtype=bool,
            )
            first_return = (
                int(np.flatnonzero(return_neutral)[0])
                if np.any(return_neutral)
                else -1
            )
            if (
                first_query < 0
                or first_return < 0
                or first_query <= first_return
                or (
                    first_return > 0
                    and np.any(stable[:first_return] == 1)
                )
            ):
                self.issue(
                    "trial.query_order",
                    f"{path}/control",
                    "partial response must time out into neutral return before the mandatory query without claiming stable resumption",
                )
        resume_state = TrialState.RESUME_TASK_TO_COMPLETION_OR_TERMINAL.value
        if resume_state not in present:
            completion_observed = _bool(group.attrs.get("task_completion_observed", 0))
            summary_success = _bool(group.attrs.get("task_success", 0))
            step_success = np.asarray(group["task/success"][()], dtype=int)
            terminated = np.asarray(group["task/terminated"][()], dtype=int)
            truncated = np.asarray(group["task/truncated"][()], dtype=int)
            terminal_reasons = _strings(group["task/terminal_reason"])
            failure_reason = str(_scalar(group.attrs.get("task_failure_reason", "")))
            if not _missing_resume_outcome_matches(
                states=states,
                step_success=step_success,
                step_terminated=terminated,
                step_truncated=truncated,
                terminal_reasons=terminal_reasons,
                summary_success=summary_success,
                completion_observed=completion_observed,
                failure_reason=failure_reason,
            ):
                self.issue(
                    "trial.resume_rows",
                    f"{path}/control/trial_state",
                    "missing resume-task row is allowed only when an exactly "
                    "bound task success/first-terminal outcome was observed "
                    "at or before query completion",
                )

    def _validate_relations(self) -> None:
        if self.file.get("schedule") is None:
            return
        schedule = self.file["schedule"]
        schedule_trial_ids = _strings(schedule.get("trial_id"))
        schedule_query_ids = _strings(schedule.get("query_id"))
        query_records = self._table_records("queries")
        query_answer_records = self._table_records("query_answers")
        encounter_records = self._table_records("encounters")
        marker_records = self._table_records("realtime_markers")
        marker_ids = [str(record.get("marker_id", "")) for record in marker_records]
        if any(not value for value in marker_ids) or len(marker_ids) != len(set(marker_ids)):
            self.issue("marker.identifier", "/realtime_markers/marker_id", "marker IDs must be non-empty and unique")
        queries_by_trial: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        encounters_by_trial: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        markers_by_trial: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in query_records:
            queries_by_trial[str(record.get("trial_id", ""))].append(record)
        for record in encounter_records:
            encounters_by_trial[str(record.get("trial_id", ""))].append(record)
        for record in marker_records:
            markers_by_trial[str(record.get("trial_id", ""))].append(record)
        unknown_query = sorted(set(queries_by_trial) - set(schedule_trial_ids))
        unknown_encounter = sorted(set(encounters_by_trial) - set(schedule_trial_ids))
        unknown_marker = sorted(set(markers_by_trial) - set(schedule_trial_ids))
        if unknown_query or unknown_encounter or unknown_marker:
            self.issue("relation.unknown_trial", "/", f"orphan trials: queries={unknown_query}, encounters={unknown_encounter}, markers={unknown_marker}")
        self._validate_query_answer_audit(query_records, query_answer_records)
        participant_id = str(_scalar(self.file.attrs.get("participant_id", "")))
        session_id = str(_scalar(self.file.attrs.get("session_id", "")))
        for index, trial_id in enumerate(schedule_trial_ids):
            name = f"trial_{index:06d}"
            group = self.file.get(f"trials/{name}")
            if group is None:
                continue
            path = f"/trials/{name}"
            queries = queries_by_trial.get(trial_id, [])
            encounters = encounters_by_trial.get(trial_id, [])
            if len(queries) != 1:
                self.issue("relation.query_cardinality", path, f"expected one mandatory query, found {len(queries)}")
            else:
                self._validate_query(queries[0], group, schedule_query_ids[index], path)
            response_mode = self._response_mode(group)
            if response_mode == "no_onset" and len(encounters) == 0:
                # A failed risk/onset candidate must not be promoted into a
                # synthetic encounter merely to satisfy table cardinality.
                pass
            elif len(encounters) != 1:
                self.issue("relation.encounter_cardinality", path, f"expected one observed safety-response encounter, found {len(encounters)}")
            else:
                self._validate_encounter(encounters[0], group, path)
            encounter_id = (
                str(encounters[0].get("encounter_id", ""))
                if len(encounters) == 1
                else str(_scalar(group.attrs.get("encounter_id", "")))
            )
            trial_markers = markers_by_trial.get(trial_id, [])
            for marker in trial_markers:
                self._validate_marker(marker, group, path, participant_id, session_id, encounter_id)
            safety_count = sum(record.get("marker_type") == "realtime_safety_concern" for record in trial_markers)
            anomaly_count = sum(record.get("marker_type") == "realtime_behavior_anomaly" for record in trial_markers)
            if safety_count != _int(group.attrs.get("realtime_safety_marker_count", -1)) or anomaly_count != _int(group.attrs.get("realtime_behavior_anomaly_marker_count", -1)):
                self.issue("marker.summary", path, "per-trial marker counts disagree with realtime marker table")

    def _validate_query_answer_audit(
        self,
        queries: Sequence[Mapping[str, Any]],
        answers: Sequence[Mapping[str, Any]],
    ) -> None:
        query_by_id = {str(record.get("query_id", "")): record for record in queries}
        trial_groups: dict[str, Any] = {}
        trials = self.file.get("trials")
        if trials is not None and hasattr(trials, "values"):
            for group in trials.values():
                trial_groups[str(_scalar(group.attrs.get("trial_id", "")))] = group
        allowed = {
            Q1_ID,
            *LIKERT_QUESTIONS.keys(),
            "modification_reasons",
        }
        by_query: dict[str, dict[str, Any]] = defaultdict(dict)
        answer_audits_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        answer_back_counts: dict[str, list[int]] = defaultdict(list)
        answer_accidental_counts: dict[str, list[int]] = defaultdict(list)
        for index, answer in enumerate(answers):
            path = f"/query_answers[{index}]"
            query_id = str(answer.get("query_id", ""))
            query = query_by_id.get(query_id)
            if query is None:
                self.issue("feedback.answer_query", path, "answer references an unknown mandatory query")
                continue
            trial_id = str(answer.get("trial_id", ""))
            if trial_id != str(query.get("trial_id", "")):
                self.issue("feedback.answer_trial", path, "answer trial differs from its query")
            question_id = str(answer.get("question_id", ""))
            if question_id not in allowed:
                self.issue("feedback.answer_question", path, f"unknown question identifier {question_id!r}")
                continue
            if question_id in by_query[query_id]:
                self.issue("feedback.answer_duplicate", path, "one query has duplicate final answer events for a question")
            answer_status = str(answer.get("answer_status", ""))
            if answer_status not in {"confirmed", "unconfirmed"}:
                self.issue(
                    "feedback.answer_status",
                    f"{path}/answer_status",
                    "answer audit status must be confirmed or unconfirmed",
                )
            try:
                value = json.loads(str(answer.get("value_json", "")))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
                self.issue("feedback.answer_value", path, "answer value is not valid JSON")
            if answer_status == "confirmed":
                by_query[query_id][question_id] = value
                if question_id == Q1_ID and value not in Q1_RESPONSES:
                    self.issue("feedback.answer_value", path, "confirmed Q1 value is invalid")
                elif question_id in LIKERT_QUESTIONS and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in range(1, 6)
                ):
                    self.issue("feedback.answer_value", path, "confirmed Likert value must be an integer in 1..5")
                elif question_id == "modification_reasons" and (
                    not isinstance(value, list)
                    or not value
                    or len(value) != len(set(value))
                    or any(item not in MODIFICATION_REASONS for item in value)
                ):
                    self.issue("feedback.answer_value", path, "confirmed modification reasons are invalid")
            elif value is not None:
                self.issue(
                    "feedback.answer_value",
                    path,
                    "unconfirmed prompt audit must carry JSON null, never a provisional answer",
                )
            prompt = (
                _float(
                    answer.get("prompt_shown_simulation_time_s", math.nan)
                ),
                _int(answer.get("prompt_shown_monotonic_ns", 0)),
                _int(answer.get("prompt_shown_unix_ns", 0)),
                _int(answer.get("prompt_shown_control_step", -1)),
            )
            first_input = (
                _float(answer.get("first_input_simulation_time_s", math.nan)),
                _int(answer.get("first_input_monotonic_ns", 0)),
                _int(answer.get("first_input_unix_ns", 0)),
                _int(answer.get("first_input_control_step", -1)),
            )
            confirmed = (
                _float(answer.get("confirmed_simulation_time_s", math.nan)),
                _int(answer.get("confirmed_monotonic_ns", 0)),
                _int(answer.get("confirmed_unix_ns", 0)),
                _int(answer.get("confirmed_control_step", -1)),
            )
            issued = (
                _float(query.get("issued_simulation_time_s", math.nan)),
                _int(query.get("issued_monotonic_ns", 0)),
                _int(query.get("issued_unix_ns", 0)),
                _int(query.get("issued_control_step", -1)),
            )
            completed = (
                _float(query.get("completed_simulation_time_s", math.nan)),
                _int(query.get("completed_monotonic_ns", 0)),
                _int(query.get("completed_unix_ns", 0)),
                _int(query.get("completed_control_step", -1)),
            )
            latency_ms = _float(answer.get("response_latency_ms", math.nan))
            no_first_input = first_input == (-1.0, 0, 0, -1)
            if answer_status == "confirmed":
                if any(
                    not (lower <= shown <= first <= accepted <= upper)
                    for lower, shown, first, accepted, upper in zip(
                        issued, prompt, first_input, confirmed, completed
                    )
                ):
                    self.issue(
                        "feedback.answer_clock",
                        path,
                        "confirmed per-question clocks must be ordered inside the query interval",
                    )
                expected_latency_ms = (confirmed[1] - prompt[1]) / 1e6
                if (
                    not math.isfinite(latency_ms)
                    or latency_ms < 0.0
                    or not math.isclose(
                        latency_ms,
                        expected_latency_ms,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                ):
                    self.issue(
                        "feedback.answer_latency",
                        f"{path}/response_latency_ms",
                        "confirmed per-question latency must equal confirmation minus prompt",
                    )
            else:
                prompt_in_query = all(
                    lower <= shown <= upper
                    for lower, shown, upper in zip(issued, prompt, completed)
                )
                first_in_query = no_first_input or all(
                    shown <= first <= upper
                    for shown, first, upper in zip(prompt, first_input, completed)
                )
                if not prompt_in_query or not first_in_query:
                    self.issue(
                        "feedback.answer_clock",
                        path,
                        "unconfirmed prompt/optional first-input clocks must be inside the query interval",
                    )
                if confirmed != (-1.0, 0, 0, -1) or latency_ms != -1.0:
                    self.issue(
                        "feedback.answer_unconfirmed_sentinel",
                        path,
                        "unconfirmed prompt requires explicit confirmation and latency sentinels",
                    )
            if str(answer.get("input_device", "")) != "vr_controller":
                self.issue(
                    "feedback.answer_input_device",
                    f"{path}/input_device",
                    "per-question input device must be vr_controller",
                )
            back_count = _int(answer.get("back_correction_count", -1))
            accidental_count = _int(answer.get("accidental_input_count", -1))
            if back_count < 0 or accidental_count < 0:
                self.issue(
                    "feedback.answer_input_audit",
                    path,
                    "per-question back/accidental counts must be non-negative",
                )
            answer_back_counts[query_id].append(back_count)
            answer_accidental_counts[query_id].append(accidental_count)
            answer_audits_by_query[query_id].append(
                {
                    "question_id": question_id,
                    "answer_status": answer_status,
                    "prompt": prompt,
                    "first_input": first_input,
                    "confirmed": confirmed,
                }
            )
            group = trial_groups.get(trial_id)
            if group is not None:
                result_steps = set(int(item) for item in group["control/result_step"][()])
                mapped_events = [("prompt_shown", prompt)]
                if not no_first_input:
                    mapped_events.append(("first_input", first_input))
                if answer_status == "confirmed":
                    mapped_events.append(("confirmed", confirmed))
                for event_name, event in mapped_events:
                    if event[3] not in result_steps:
                        self.issue(
                            "feedback.answer_step",
                            f"{path}/{event_name}_control_step",
                            "per-question audit event does not map to a recorded post-step transition",
                        )
        for query_id, query in query_by_id.items():
            audits = answer_audits_by_query.get(query_id, [])
            for prior, current in zip(audits, audits[1:]):
                prior_confirmed = prior["confirmed"]
                current_prompt = current["prompt"]
                if prior["answer_status"] != "confirmed" or any(
                    later < earlier
                    for earlier, later in zip(prior_confirmed, current_prompt)
                ):
                    self.issue(
                        "feedback.answer_sequence",
                        f"/queries/{query_id}",
                        "final per-question prompt cycles are not chronological",
                    )
                    break
            query_back_count = _int(query.get("back_correction_count", -1))
            query_accidental_count = _int(
                query.get("accidental_input_count", -1)
            )
            per_answer_back = answer_back_counts.get(query_id, [])
            per_answer_accidental = answer_accidental_counts.get(query_id, [])
            if (
                query_back_count < 0
                or sum(per_answer_back) > query_back_count
                or any(value > query_back_count for value in per_answer_back)
            ):
                self.issue(
                    "feedback.answer_back_audit",
                    f"/queries/{query_id}",
                    "per-question back-correction counts exceed the query-wide audit",
                )
            if (
                query_accidental_count < 0
                or sum(per_answer_accidental) > query_accidental_count
                or any(
                    value > query_accidental_count
                    for value in per_answer_accidental
                )
            ):
                self.issue(
                    "feedback.answer_accidental_audit",
                    f"/queries/{query_id}",
                    "per-question accidental-input counts exceed the query-wide audit",
                )
            completed_status = str(query.get("response_status", "")) == "completed"
            statuses = [str(audit["answer_status"]) for audit in audits]
            question_ids = [str(audit["question_id"]) for audit in audits]
            expected_question_ids = [Q1_ID, *LIKERT_QUESTIONS.keys()]
            if by_query.get(query_id, {}).get(Q1_ID) == "needs_modification":
                expected_question_ids.append("modification_reasons")
            if completed_status:
                if (
                    not audits
                    or any(status != "confirmed" for status in statuses)
                    or question_ids != expected_question_ids
                ):
                    self.issue(
                        "feedback.answer_cardinality",
                        f"/queries/{query_id}",
                        "completed query requires exactly six confirmed prompts, plus one confirmed reason prompt when required",
                    )
            elif (
                not audits
                or statuses[-1:] != ["unconfirmed"]
                or statuses.count("unconfirmed") != 1
                or question_ids != expected_question_ids[: len(question_ids)]
            ):
                self.issue(
                    "feedback.answer_cardinality",
                    f"/queries/{query_id}",
                    "incomplete query must preserve its confirmed prefix and exactly one final unconfirmed visible prompt",
                )
            if completed_status and audits:
                final_confirmed = audits[-1]["confirmed"]
                query_completed = (
                    _float(query.get("completed_simulation_time_s", math.nan)),
                    _int(query.get("completed_monotonic_ns", 0)),
                    _int(query.get("completed_unix_ns", 0)),
                    _int(query.get("completed_control_step", -1)),
                )
                if final_confirmed != query_completed:
                    self.issue(
                        "feedback.answer_completion_clock",
                        f"/queries/{query_id}",
                        "the final answer confirmation must equal query completion",
                    )
            if not completed_status:
                continue
            expected: dict[str, Any] = {
                Q1_ID: str(query.get("q1_response", "")),
                **{
                    name: _int(query.get(name, 0))
                    for name in LIKERT_QUESTIONS
                },
            }
            if expected[Q1_ID] == "needs_modification":
                expected["modification_reasons"] = _json_string_list(
                    query.get("modification_reasons_json", "")
                )
            if by_query.get(query_id, {}) != expected:
                self.issue("feedback.answer_audit", f"/queries/{query_id}", "completed query fields do not exactly match its timestamped answer events")

    def _table_records(self, name: str) -> list[dict[str, Any]]:
        group = self.file.get(name)
        if group is None or not hasattr(group, "keys") or not group.keys():
            return []
        lengths = [int(group[column].shape[0]) for column in group.keys() if hasattr(group[column], "shape")]
        if not lengths or len(set(lengths)) != 1:
            return []
        return [_table_record(group, index) for index in range(lengths[0])]

    def _validate_query(self, record: Mapping[str, Any], group: Any, expected_query_id: str, path: str) -> None:
        query_id = str(record.get("query_id", ""))
        if query_id != expected_query_id or query_id != str(_scalar(group.attrs.get("query_id", ""))):
            self.issue("feedback.query_id", f"{path}/query", "query ID does not match the immutable schedule")
        encounter_id = str(record.get("encounter_id", ""))
        if not encounter_id or encounter_id != str(_scalar(group.attrs.get("encounter_id", ""))):
            self.issue("feedback.encounter_id", f"{path}/query", "query must point to the trial's safety-response encounter")
        issued_sim = _float(record.get("issued_simulation_time_s"))
        completed_sim = _float(record.get("completed_simulation_time_s"))
        issued_mono = _int(record.get("issued_monotonic_ns"))
        completed_mono = _int(record.get("completed_monotonic_ns"))
        issued_unix = _int(record.get("issued_unix_ns"))
        completed_unix = _int(record.get("completed_unix_ns"))
        if not (0 <= issued_sim <= completed_sim and 0 < issued_mono <= completed_mono and 0 < issued_unix <= completed_unix):
            self.issue("feedback.query_clock", f"{path}/query", "prompt/confirmation timestamps are absent or out of order")
        issued_step = _int(record.get("issued_control_step", -1))
        completed_step = _int(record.get("completed_control_step", -1))
        if issued_step < 0 or completed_step < issued_step:
            self.issue("feedback.query_step", f"{path}/query", "issued/completed control steps are absent or out of order")
        else:
            logged_steps = set(int(value) for value in group["control/source_step"][()]) | set(
                int(value) for value in group["control/result_step"][()]
            )
            if issued_step not in logged_steps or completed_step not in logged_steps:
                self.issue("feedback.query_step", f"{path}/query", "query timestamps do not map to trial transition rows")
        stable_sim = _float(group.attrs.get("response_end_simulation_time_s", math.nan))
        if math.isfinite(stable_sim) and issued_sim + 1e-6 < stable_sim:
            self.issue("feedback.query_order", f"{path}/query", "query was shown before stable task resumption")
        status = str(record.get("response_status", ""))
        disposition = str(record.get("response_disposition", ""))
        q1 = str(record.get("q1_response", ""))
        ratings = tuple(_int(record.get(name, 0)) for name in (
            "q2_perceived_danger",
            "q3_abruptness",
            "q4_excessive_duration",
            "q5_task_disruption",
            "q6_confidence",
        ))
        reasons = _json_string_list(record.get("modification_reasons_json", ""))
        protocol_valid = _bool(group.attrs.get("protocol_valid", 0))
        if status == "completed":
            if q1 not in Q1_RESPONSES:
                self.issue("feedback.q1", f"{path}/query", "completed query requires one exact Q1 response")
            if any(value not in range(1, 6) for value in ratings):
                self.issue("feedback.likert", f"{path}/query", "completed query requires Q2-Q6 values in 1..5")
            if q1 == "needs_modification" and not reasons:
                self.issue("feedback.reasons", f"{path}/query", "needs_modification requires at least one reason")
            if q1 != "needs_modification" and reasons:
                self.issue("feedback.reasons", f"{path}/query", "reasons are only valid for needs_modification")
        elif status in {"timeout", "no_response"}:
            if q1 or any(ratings) or reasons:
                self.issue("feedback.missing_inference", f"{path}/query", "timeout/no-response must not carry inferred labels or ratings")
            if protocol_valid:
                self.issue("feedback.valid_missing", f"{path}/query", "protocol_valid trial cannot have missing feedback")
        else:
            self.issue("feedback.status", f"{path}/query", f"unknown response status {status!r}")
        expected_disposition = (
            "missing_abstain"
            if status in {"timeout", "no_response"}
            else "uncertain_abstain"
            if status == "completed" and q1 == "uncertain"
            else "answered"
            if status == "completed"
            else ""
        )
        if disposition != expected_disposition:
            self.issue(
                "feedback.disposition",
                f"{path}/query/response_disposition",
                "response disposition must preserve definite, uncertain-abstain, "
                "and missing-abstain semantics exactly",
            )
        if reasons is None or len(set(reasons or ())) != len(reasons or ()) or any(reason not in MODIFICATION_REASONS for reason in reasons or ()):
            self.issue("feedback.reasons", f"{path}/query", "modification reasons are malformed, duplicate, or unknown")
        if str(record.get("input_device", "")) != "vr_controller":
            self.issue("feedback.input_device", f"{path}/query", "mandatory response must use a VR controller")
        latency_ms = _float(record.get("response_latency_ms", math.nan))
        if not math.isfinite(latency_ms) or latency_ms < 0 or not math.isclose(latency_ms, (completed_mono - issued_mono) / 1e6, rel_tol=0.0, abs_tol=2.0):
            self.issue("feedback.latency", f"{path}/query", "response latency does not match monotonic timestamps")
        first_sim = _float(record.get("first_input_simulation_time_s", math.nan))
        first_mono = _int(record.get("first_input_monotonic_ns", 0))
        first_unix = _int(record.get("first_input_unix_ns", 0))
        first_step = _int(record.get("first_input_control_step", -1))
        no_input = first_sim == -1.0 and first_mono == 0 and first_unix == 0 and first_step == -1
        in_window_input = (
            issued_sim <= first_sim <= completed_sim
            and issued_mono <= first_mono <= completed_mono
            and issued_unix <= first_unix <= completed_unix
            and issued_step <= first_step <= completed_step
        )
        if status == "completed" and not in_window_input:
            self.issue("feedback.first_input", f"{path}/query", "completed response requires an in-window first-input timestamp and step")
        elif status != "completed" and not (no_input or in_window_input):
            self.issue("feedback.first_input", f"{path}/query", "partial timeout input must be preserved in-window or use explicit no-input sentinels")
        if _int(record.get("back_correction_count", -1)) < 0 or _int(record.get("accidental_input_count", -1)) < 0:
            self.issue("feedback.input_audit", f"{path}/query", "back/accidental input counters must be non-negative")
        if (
            str(_scalar(group.attrs.get("query_response_status", ""))) != status
            or str(
                _scalar(group.attrs.get("query_response_disposition", ""))
            )
            != disposition
            or str(_scalar(group.attrs.get("q1_primary_response", ""))) != q1
        ):
            self.issue("feedback.trial_summary", path, "query table and trial feedback summary disagree")
        summary_values = {
            "q2_perceived_danger": ratings[0],
            "q3_abruptness": ratings[1],
            "q4_excessive_duration": ratings[2],
            "q5_task_disruption": ratings[3],
            "q6_confidence": ratings[4],
            "feedback_input_device": str(record.get("input_device", "")),
            "back_correction_count": _int(record.get("back_correction_count", -1)),
            "accidental_input_count": _int(record.get("accidental_input_count", -1)),
        }
        if any(_scalar(group.attrs.get(name, None)) != value for name, value in summary_values.items()):
            self.issue("feedback.trial_summary", path, "Q2-Q6, input device, or input-audit summary differs from the query table")
        trial_reasons = _json_string_list(group.attrs.get("modification_reasons_json", ""))
        if trial_reasons != reasons:
            self.issue("feedback.trial_summary", f"{path}/@modification_reasons_json", "modification reasons differ from the query table")
        summary_clocks = (
            ("prompt_shown_simulation_time_s", issued_sim),
            ("first_input_simulation_time_s", first_sim),
            ("confirmed_simulation_time_s", completed_sim),
            ("response_latency_ms", latency_ms),
        )
        if any(
            not math.isclose(
                _float(group.attrs.get(name, math.nan)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for name, expected in summary_clocks
        ):
            self.issue("feedback.trial_summary", path, "query timing summary differs from the query table")

    def _validate_encounter(self, record: Mapping[str, Any], group: Any, path: str) -> None:
        response_mode = self._response_mode(group)
        encounter_id = str(record.get("encounter_id", ""))
        if not encounter_id or encounter_id != str(_scalar(group.attrs.get("encounter_id", ""))):
            self.issue("encounter.id", f"{path}/encounter", "encounter ID does not match trial")
        onset = _int(record.get("onset_step", -1))
        confirmed = _int(record.get("onset_confirmed_step", -1))
        offset = _int(record.get("offset_step_exclusive", -1))
        window_start = _int(record.get("window_start_step", -1))
        window_end = _int(record.get("window_end_step_exclusive", -1))
        if not (0 <= window_start <= onset <= confirmed < offset <= window_end):
            self.issue("encounter.window", f"{path}/encounter", "onset/confirmation/offset/window steps are out of order")
        onset_sim = _float(record.get("onset_simulation_time_s", math.nan))
        offset_sim = _float(record.get("offset_simulation_time_s", math.nan))
        onset_mono = _int(record.get("onset_monotonic_ns", 0))
        offset_mono = _int(record.get("offset_monotonic_ns", 0))
        if not (0 <= onset_sim <= offset_sim and 0 < onset_mono <= offset_mono):
            self.issue("encounter.clock", f"{path}/encounter", "encounter timestamps are absent or out of order")
        if _bool(record.get("encounter_timeout", 0)) and _bool(group.attrs.get("protocol_valid", 0)):
            self.issue("encounter.timeout", f"{path}/encounter", "timed-out response episode cannot be protocol_valid")
        cbf_start = _int(record.get("cbf_intervention_start_step", -1))
        cbf_confirmed = _int(record.get("cbf_intervention_confirmed_step", -1))
        expected_cbf_start = _int(group.attrs.get("response_onset_step", -2))
        expected_cbf_confirmed = _int(
            group.attrs.get("response_onset_confirmed_step", -2)
        )
        if cbf_start != expected_cbf_start or cbf_confirmed != expected_cbf_confirmed:
            self.issue("encounter.response_mapping", f"{path}/encounter", "encounter CBF onset fields differ from response-detector summary")
        minimum = _float(record.get("minimum_surface_gap_m", math.nan))
        maximum_intervention = _float(record.get("maximum_intervention_norm_rad_s", math.nan))
        minimum_intervention = 0.0 if response_mode == "no_onset" else 0.05
        if not (
            math.isfinite(minimum)
            and math.isfinite(maximum_intervention)
            and maximum_intervention >= minimum_intervention
        ):
            self.issue("encounter.metrics", f"{path}/encounter", "encounter gap/intervention metrics are invalid")

    def _validate_marker(
        self,
        record: Mapping[str, Any],
        group: Any,
        path: str,
        participant_id: str,
        session_id: str,
        encounter_id: str,
    ) -> None:
        import numpy as np

        marker_type = str(record.get("marker_type", ""))
        if marker_type not in MARKER_TYPES:
            self.issue("marker.type", f"{path}/marker", f"unknown marker type {marker_type!r}")
        if str(record.get("participant_id", "")) != participant_id or str(record.get("session_id", "")) != session_id:
            self.issue("marker.identity", f"{path}/marker", "participant/session identity differs from root")
        if str(record.get("encounter_id", "")) != encounter_id:
            self.issue("marker.encounter", f"{path}/marker", "marker does not map to the trial encounter")
        condition_id = str(_scalar(group.attrs.get("condition_id", "")))
        condition = CONDITIONS.get(condition_id)
        if str(record.get("condition_id", "")) != condition_id or not condition or not _same_number(record.get("lambda_s"), condition.lambda_s):
            self.issue("marker.condition", f"{path}/marker", "condition/lambda differs from trial assignment")
        phase = str(record.get("response_phase", ""))
        if phase not in _RESPONSE_PHASES:
            self.issue("marker.phase", f"{path}/marker", "marker response phase is unknown")
        result_steps = np.asarray(group["control/result_step"][()], dtype=int)
        step = _int(record.get("control_step", -1))
        matches = np.flatnonzero(result_steps == step)
        if matches.size != 1:
            self.issue("marker.step", f"{path}/marker", "post-step marker control_step does not map to exactly one transition result")
            return
        index = int(matches[0])
        row_phase = _strings(group["response/phase"])[index]
        if phase != row_phase:
            self.issue("marker.phase", f"{path}/marker", "marker phase differs from the referenced control row")
        sim = _float(record.get("simulation_time_s", math.nan))
        mono = _int(record.get("monotonic_ns", 0))
        unix = _int(record.get("unix_ns", 0))
        row_sim = _float(group["control/sim_time"][index])
        row_mono = _int(group["control/post_step_monotonic_ns"][index])
        row_unix = _int(group["control/unix_time_ns"][index])
        trial_start_mono = _int(group.attrs.get("start_monotonic_ns", 0))
        trial_end_mono = _int(group.attrs.get("end_monotonic_ns", 0))
        trial_start_unix = _int(group.attrs.get("start_unix_ns", 0))
        trial_end_unix = _int(group.attrs.get("end_unix_ns", 0))
        if (
            not math.isclose(sim, row_sim, rel_tol=0.0, abs_tol=1e-6)
            or mono != row_mono
            or unix != row_unix
        ):
            self.issue("marker.clock", f"{path}/marker", "marker simulation/monotonic clock does not map to the referenced row")
        if not (trial_start_mono <= mono <= trial_end_mono and trial_start_unix <= unix <= trial_end_unix):
            self.issue("marker.clock", f"{path}/marker", "marker clock falls outside trial bounds")
        for field_name, row_name in (
            ("time_since_response_onset_s", "response/time_since_response_onset_s"),
            ("time_since_recovery_onset_s", "response/time_since_recovery_onset_s"),
        ):
            if not math.isclose(_float(record.get(field_name, math.nan)), _float(group[row_name][index]), rel_tol=0.0, abs_tol=1e-6):
                self.issue("marker.elapsed", f"{path}/marker/{field_name}", "does not match referenced response row")
        controller_hand = str(record.get("controller_hand", ""))
        crossing_hand = str(_scalar(group.attrs.get("crossing_hand", "")))
        if controller_hand not in {"left", "right"} or controller_hand == crossing_hand:
            self.issue("marker.controller", f"{path}/marker", "marker must use the non-crossing controller")
        button = str(record.get("button", ""))
        identifier = str(record.get("button_identifier", ""))
        if not button or identifier != f"{controller_hand}.{button}":
            self.issue("marker.button", f"{path}/marker", "button_identifier must be controller_hand.button")
        configured = (
            self.config.get("feedback", {})
            .get("realtime_markers", {})
            .get(f"crossing_{crossing_hand}", {})
            .get(marker_type, "")
        )
        if identifier != configured:
            self.issue("marker.mapping", f"{path}/marker", "button does not match the frozen opposite-controller marker mapping")


def _normalize_schedule_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize TrialSpec aliases to the exact serialized TrialPlan surface."""

    result: dict[str, Any] = {}
    for field in fields(TrialPlan):
        name = field.name
        source_name = {
            "trial_index": "order_index",
            "direction": "crossing_direction",
            "speed": "crossing_speed",
            "source_trial_id": "anchor_context_id",
        }.get(name, name)
        result[name] = _scalar(value.get(name, value.get(source_name, "")))
    return result


def _table_record(group: Any, index: int) -> dict[str, Any]:
    return {str(name): _dataset_scalar(dataset, index) for name, dataset in group.items() if hasattr(dataset, "shape")}


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return value


def _dataset_scalar(dataset: Any, index: int) -> Any:
    try:
        if hasattr(dataset, "asstr") and getattr(dataset.dtype, "kind", "") in {"O", "S", "U"}:
            return str(dataset.asstr()[index])
    except (TypeError, ValueError):
        pass
    return _scalar(dataset[index])


def _strings(dataset: Any) -> list[str]:
    if dataset is None:
        return []
    try:
        values = dataset.asstr()[()] if hasattr(dataset, "asstr") else dataset[()]
    except Exception:
        return []
    try:
        import numpy as np

        return [str(_scalar(value)) for value in np.asarray(values).reshape(-1)]
    except ImportError:
        return [str(_scalar(value)) for value in values]


def _int(value: Any) -> int:
    try:
        return int(_scalar(value))
    except (TypeError, ValueError, OverflowError):
        return -1


def _float(value: Any) -> float:
    try:
        return float(_scalar(value))
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _bool(value: Any) -> bool:
    return _int(value) == 1


def _same_number(first: Any, second: Any) -> bool:
    try:
        return math.isclose(float(_scalar(first)), float(_scalar(second)), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError, OverflowError):
        return False


def _maximum_run(values: Sequence[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def _json_string_list(raw: Any) -> list[str] | None:
    try:
        value = json.loads(str(_scalar(raw)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return value


def _recursive_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.add(str(key))
            result.update(_recursive_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_recursive_keys(item))
    return result


def _dataset_all_finite(dataset: Any, *, chunk_rows: int = 2048) -> bool:
    try:
        import numpy as np

        length = int(dataset.shape[0]) if dataset.shape else 1
        if not dataset.shape:
            return bool(np.all(np.isfinite(np.asarray(dataset[()], dtype=float))))
        for start in range(0, length, chunk_rows):
            if not np.all(np.isfinite(np.asarray(dataset[start:start + chunk_rows], dtype=float))):
                return False
        return True
    except (TypeError, ValueError, OverflowError):
        return False


__all__ = [
    "CollectionValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_ac_selective_smoothing_collection",
    "validate_collection",
    "validate_static_contract",
]
