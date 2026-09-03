"""Atomic HDF5 recorder for the one-crossing online explicit-feedback study."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .online_rows import online_row_specs
from .online_schema import (
    ACTUAL_CROSSING_DIRECTIONS,
    FEATURE_PROVENANCE_VERSION,
    FUTURE_OUTCOME_FIELDS,
    LIKERT_QUESTIONS,
    MARKER_TYPES,
    MODIFICATION_REASONS,
    ONLINE_PROTOCOL_VERSION,
    ONLINE_ROW_SEMANTICS,
    ONLINE_SCHEMA_VERSION,
    Q1_ID,
    Q1_RESPONSES,
    Q1_TEXT_KO,
    RESPONSE_PHASES,
    RUNTIME_OBSERVABLE_FIELDS,
    EncounterRecordV1,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
    exact_bool,
    validate_query_record,
)
from .study import TrialState, validate_decision_context
from .recorder import (
    DatasetSpec,
    _append_table_row,
    _attribute_value,
    _coerce_value,
    _create_extendable_dataset,
    _write_static_value,
)
from .schema import (
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    POLICY_SIZE_BYTES,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)


class OnlineExplicitFeedbackRecorder:
    """Write to ``.partial`` until the new fail-closed validator approves it."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any],
        joint_names: Sequence[str],
        arm_joint_indices: Sequence[int],
        protocol_config: Mapping[str, Any],
        scenario_config: Mapping[str, Any],
        frozen_config: Mapping[str, Any],
        flush_every_steps: int = 1,
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("online collection requires h5py; NPZ fallback is forbidden") from error
        self.h5py = h5py
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output_path must end in .h5 or .hdf5")
        self.partial_path = Path(str(self.output_path) + ".partial")
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite collection output: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = tuple(str(value) for value in joint_names)
        self.arm_joint_indices = tuple(int(value) for value in arm_joint_indices)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if (
            not self.arm_joint_indices
            or len(set(self.arm_joint_indices)) != len(self.arm_joint_indices)
            or any(
                value < 0 or value >= len(self.joint_names)
                for value in self.arm_joint_indices
            )
        ):
            raise ValueError("arm_joint_indices are invalid")
        self.flush_every_steps = max(1, int(flush_every_steps))
        self._file = h5py.File(self.partial_path, "x", libver="latest")
        self._datasets: dict[str, Any] = {}
        self._active_trial = None
        self._active_plan: TrialPlan | None = None
        self._trial_length = 0
        self._trial_count = 0
        self._marker_count = 0
        self._marker_ids: set[str] = set()
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._query_count = 0
        self._query_answer_count = 0
        self._encounter_count = 0
        self._layout_precheck_count = 0
        self._schedule: tuple[TrialPlan, ...] = ()
        self._participant_id = str(metadata.get("participant_id", "")).strip()
        self._session_id = str(metadata.get("session_id", "")).strip()
        self._sealed = False
        self._closed = False
        self._write_root_metadata(
            metadata, protocol_config, scenario_config, frozen_config
        )
        self._file.create_group("trials")
        self._file.create_group("bc_infeasible_diagnostic")
        self._create_tables()
        self._file.flush()

    def write_schedule(self, plans: Sequence[TrialPlan]) -> None:
        self._require_open()
        table = self._file["schedule"]
        if table["trial_id"].shape[0] != 0:
            raise RuntimeError("schedule was already written")
        frozen_plans = tuple(plans)
        if not frozen_plans:
            raise ValueError("the preassigned trial schedule cannot be empty")
        if tuple(plan.trial_index for plan in frozen_plans) != tuple(
            range(len(frozen_plans))
        ):
            raise ValueError("scheduled trial indices must be contiguous from zero")
        for field in ("trial_id", "query_id", "encounter_id"):
            values = [str(getattr(plan, field)) for plan in frozen_plans]
            if len(set(values)) != len(values):
                raise ValueError(f"scheduled {field} values must be unique")
        if len({plan.schedule_seed for plan in frozen_plans}) != 1:
            raise ValueError("every trial must preserve the same preassigned schedule_seed")
        if len({plan.counterbalancing_group for plan in frozen_plans}) != 1:
            raise ValueError("counterbalancing_group cannot change within a session")
        for plan in frozen_plans:
            _append_table_row(
                table,
                {
                    "trial_index": int(plan.trial_index),
                    "trial_id": plan.trial_id,
                    "query_id": plan.query_id,
                    "encounter_id": plan.encounter_id,
                    "task_phase": plan.task_phase,
                    "severity": plan.severity,
                    "direction": plan.direction,
                    "speed": plan.speed,
                    "crossing_hand": plan.crossing_hand,
                    "condition_id": plan.condition_id,
                    "objective_mode": plan.objective_mode,
                    "lambda_s": float(plan.lambda_s),
                    "schedule_seed": int(plan.schedule_seed),
                    "block_id": plan.block_id,
                    "counterbalancing_group": plan.counterbalancing_group,
                    "practice": int(plan.practice),
                    "anchor_repeat": int(plan.anchor_repeat),
                    "pilot": int(plan.pilot),
                    "analysis_exclude": int(plan.analysis_exclude),
                    "source_trial_id": plan.source_trial_id,
                },
            )
        self._schedule = frozen_plans
        self._file.attrs["scheduled_trial_count"] = len(frozen_plans)
        self._file.attrs["schedule_seed"] = int(frozen_plans[0].schedule_seed)
        self._file.attrs["counterbalancing_group"] = str(
            frozen_plans[0].counterbalancing_group
        )
        self._file.attrs["trial_order_json"] = _json(
            [plan.trial_id for plan in frozen_plans]
        )
        self._file.flush()

    def append_layout_precheck(self, record: Mapping[str, Any]) -> None:
        self._require_open()
        fields = {
            "candidate_index": int(record["candidate_index"]),
            "layout_id": str(record["layout_id"]),
            "seed": int(record["seed"]),
            "controller": str(record.get("controller", "bc_only")),
            "policy_sha256": str(record["policy_sha256"]),
            "cbf_enabled": int(bool(record.get("cbf_enabled", False))),
            "human_absent_or_clear": int(bool(record.get("human_absent_or_clear", False))),
            "strict_success": int(bool(record.get("strict_success", False))),
            "release_observed": int(bool(record.get("release_observed", False))),
            "nominal_phase_references_complete": int(
                bool(record.get("nominal_phase_references_complete", False))
            ),
            "nominal_phase_references_json": _json(
                record.get("nominal_phase_references", {})
            ),
            "nominal_corridor_bank_complete": int(
                bool(record.get("nominal_corridor_bank_complete", False))
            ),
            "nominal_phase_trajectories_json": _json(
                record.get("nominal_phase_trajectories", {})
            ),
            "nominal_corridor_bank_json": _json(
                record.get("nominal_corridor_bank", {})
            ),
            "steps": int(record.get("steps", 0)),
            "diagnostic_group": str(record.get("diagnostic_group", "")),
            "failure_reason": str(record.get("failure_reason", "")),
            "started_monotonic_ns": int(record.get("started_monotonic_ns", 0)),
            "ended_monotonic_ns": int(record.get("ended_monotonic_ns", 0)),
            "source_restoration_json": _json(record.get("source_restoration", {})),
        }
        _append_table_row(self._file["layout_prechecks"], fields)
        if (
            not fields["strict_success"]
            and fields["diagnostic_group"] == "bc_infeasible_diagnostic"
        ):
            diagnostic = self._file["bc_infeasible_diagnostic"].create_group(
                f"candidate_{self._layout_precheck_count:06d}"
            )
            for name, value in fields.items():
                diagnostic.attrs[name] = value
        self._layout_precheck_count += 1
        self._file.attrs["layout_precheck_count"] = self._layout_precheck_count
        self._file.flush()

    def start_trial(
        self,
        plan: TrialPlan,
        *,
        seed: int,
        layout_id: str,
        layout_precheck_index: int,
        planned_gap_min_m: float,
        planned_gap_max_m: float,
        planned_target_gap_m: float,
        start_simulation_time_s: float,
        start_monotonic_ns: int,
        start_unix_ns: int,
        initial_scene: Mapping[str, Any],
        condition_config: Mapping[str, Any],
        actual_cbf_config: Mapping[str, Any],
    ) -> None:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("a trial is already active")
        if int(plan.trial_index) != self._trial_count:
            raise ValueError(
                f"trial indices must be contiguous; expected {self._trial_count}, got {plan.trial_index}"
            )
        if not self._schedule:
            raise RuntimeError("write the preassigned schedule before starting trials")
        if plan != self._schedule[self._trial_count]:
            raise RuntimeError(
                "runtime trial plan differs from the immutable preassigned schedule"
            )
        condition_payload = dict(condition_config)
        if set(condition_payload) != {"condition_id", "objective_mode", "lambda_s"}:
            raise RuntimeError(
                "condition config must contain exactly "
                "condition_id/objective_mode/lambda_s"
            )
        if (
            str(condition_payload.get("condition_id", plan.condition_id))
            != plan.condition_id
            or str(condition_payload.get("objective_mode", ""))
            != plan.objective_mode
            or float(condition_payload.get("lambda_s", -1.0)) != float(plan.lambda_s)
        ):
            raise RuntimeError("condition config differs from the scheduled A/C assignment")
        actual_cbf_payload = dict(actual_cbf_config)
        if (
            str(actual_cbf_payload.get("objective_mode", "")) != plan.objective_mode
            or float(
                actual_cbf_payload.get("correction_smoothness_weight", -1.0)
            )
            != float(plan.lambda_s)
        ):
            raise RuntimeError("actual runtime CBF config differs from scheduled A/C assignment")
        group = self._file["trials"].create_group(f"trial_{plan.trial_index:06d}")
        attrs = {
            **plan.as_dict(),
            "trial_seed": int(seed),
            "trial_complete": 0,
            "protocol_valid": 0,
            "single_pick": 1,
            "cube_count": 1,
            "intended_crossing_count": 1,
            "actual_crossing_count": 0,
            "query_present": 0,
            "query_count": 0,
            "encounter_record_count": 0,
            "encounter_id": plan.encounter_id,
            "layout_id": str(layout_id),
            "layout_group": "bc_feasible_main",
            "layout_precheck_index": int(layout_precheck_index),
            "layout_bc_feasible": 1,
            "planned_gap_min_m": float(planned_gap_min_m),
            "planned_gap_max_m": float(planned_gap_max_m),
            "planned_target_gap_m": float(planned_target_gap_m),
            "start_simulation_time_s": float(start_simulation_time_s),
            "start_monotonic_ns": int(start_monotonic_ns),
            "start_unix_ns": int(start_unix_ns),
            "end_simulation_time_s": 0.0,
            "end_monotonic_ns": 0,
            "end_unix_ns": 0,
            "step_count": 0,
            "off_protocol": 0,
            "off_protocol_reasons_json": "[]",
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "query_response_status": "",
            "query_response_disposition": "",
            "q1_primary_response": "",
            "condition_config_json": _json(condition_payload),
            "condition_config_sha256": _sha256_json(condition_payload),
            "actual_cbf_config_json": _json(actual_cbf_payload),
            "actual_cbf_config_sha256": _sha256_json(actual_cbf_payload),
            "trial_fsm_history_json": "[]",
            "decision_context_json": "{}",
        }
        for name, value in attrs.items():
            group.attrs[name] = _attribute_value(value)
        action_group = group.create_group("actions")
        string_dtype = self.h5py.string_dtype("utf-8")
        action_group.create_dataset(
            "all_joint_names", data=np.asarray(self.joint_names, dtype=string_dtype)
        )
        action_group.create_dataset(
            "arm_joint_indices", data=np.asarray(self.arm_joint_indices, dtype=np.int32)
        )
        action_group.create_dataset(
            "arm_joint_names",
            data=np.asarray(
                [self.joint_names[index] for index in self.arm_joint_indices],
                dtype=string_dtype,
            ),
        )
        scene = group.create_group("initial_scene")
        for name, value in initial_scene.items():
            _write_static_value(scene, str(name), value, h5py=self.h5py)
        specs = online_row_specs(len(self.joint_names), len(self.arm_joint_indices))
        self._datasets = {
            path: _create_extendable_dataset(group, path, spec, h5py=self.h5py)
            for path, spec in specs.items()
        }
        self._active_trial = group
        self._active_plan = plan
        self._trial_length = 0
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._file.flush()

    def append_transition(self, row: Mapping[str, Any]) -> None:
        self._require_active_trial()
        assert self._active_plan is not None
        specs = online_row_specs(len(self.joint_names), len(self.arm_joint_indices))
        missing = sorted(set(specs) - set(row))
        extra = sorted(set(row) - set(specs))
        if missing or extra:
            raise RuntimeError(f"transition columns mismatch: missing={missing}, extra={extra}")
        if (
            str(row["control/trial_id"]) != self._active_plan.trial_id
            or str(row["control/query_id"]) != self._active_plan.query_id
            or str(row["controller/condition_id"]) != self._active_plan.condition_id
            or str(row["controller/objective_mode"]) != self._active_plan.objective_mode
            or float(row["controller/lambda_s"]) != float(self._active_plan.lambda_s)
            or str(row["policy/bc_checkpoint_sha256"]) != POLICY_SHA256
            or str(row["crossing/intended_hand"])
            != self._active_plan.crossing_hand
            or str(row["crossing/non_crossing_hand"])
            != (
                "right"
                if self._active_plan.crossing_hand == "left"
                else "left"
            )
        ):
            raise RuntimeError("transition provenance differs from active trial assignment")
        converted = {
            path: _coerce_value(row[path], spec, path=path)
            for path, spec in specs.items()
        }
        index = self._trial_length
        for path, value in converted.items():
            dataset = self._datasets[path]
            dataset.resize((index + 1, *dataset.shape[1:]))
            dataset[index] = value
        self._trial_length += 1
        self._active_trial.attrs["step_count"] = self._trial_length
        if self._trial_length % self.flush_every_steps == 0:
            self._file.flush()

    def append_marker(self, record: RealtimeMarkerRecord) -> None:
        self._require_active_trial()
        if record.marker_type not in MARKER_TYPES:
            raise ValueError(f"invalid marker type: {record.marker_type}")
        assert self._active_plan is not None
        if (
            record.participant_id != self._participant_id
            or record.session_id != self._session_id
            or record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
            or record.condition_id != self._active_plan.condition_id
            or float(record.lambda_s) != float(self._active_plan.lambda_s)
            or record.controller_hand == self._active_plan.crossing_hand
        ):
            raise RuntimeError("realtime marker provenance does not match active trial")
        if record.marker_id in self._marker_ids:
            raise RuntimeError(f"duplicate realtime marker_id: {record.marker_id}")
        _append_table_row(self._file["realtime_markers"], record.as_dict())
        self._marker_ids.add(record.marker_id)
        self._marker_count += 1
        self._trial_marker_counts[record.marker_type] += 1
        self._file.attrs["realtime_marker_count"] = self._marker_count
        self._file.flush()

    def append_query(self, record: QueryRecord) -> None:
        self._require_active_trial()
        validate_query_record(record)
        assert self._active_plan is not None
        if (
            record.query_id != self._active_plan.query_id
            or record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
        ):
            raise RuntimeError("query identifiers do not match the active trial plan")
        if int(self._active_trial.attrs.get("query_count", 0)) != 0:
            raise RuntimeError("exactly one mandatory query is allowed per trial")
        _append_table_row(self._file["queries"], record.as_dict())
        self._query_count += 1
        self._file.attrs["query_count"] = self._query_count
        self._active_trial.attrs["query_present"] = 1
        self._active_trial.attrs["query_count"] = 1
        self._active_trial.attrs["query_response_status"] = record.response_status
        self._active_trial.attrs["query_response_disposition"] = (
            record.response_disposition
        )
        self._active_trial.attrs["q1_primary_response"] = record.q1_response
        self._active_trial.attrs["encounter_id"] = record.encounter_id
        self._active_trial.attrs["q2_perceived_danger"] = record.q2_perceived_danger
        self._active_trial.attrs["q3_abruptness"] = record.q3_abruptness
        self._active_trial.attrs["q4_excessive_duration"] = record.q4_excessive_duration
        self._active_trial.attrs["q5_task_disruption"] = record.q5_task_disruption
        self._active_trial.attrs["q6_confidence"] = record.q6_confidence
        self._active_trial.attrs["modification_reasons_json"] = _json(
            record.modification_reasons
        )
        self._active_trial.attrs["response_latency_ms"] = record.response_latency_ms
        self._active_trial.attrs["prompt_shown_simulation_time_s"] = (
            record.issued_simulation_time_s
        )
        self._active_trial.attrs["first_input_simulation_time_s"] = (
            record.first_input_simulation_time_s
        )
        self._active_trial.attrs["confirmed_simulation_time_s"] = (
            record.completed_simulation_time_s
        )
        self._active_trial.attrs["feedback_input_device"] = record.input_device
        self._active_trial.attrs["back_correction_count"] = (
            record.back_correction_count
        )
        self._active_trial.attrs["accidental_input_count"] = (
            record.accidental_input_count
        )
        self._file.flush()

    def append_query_answer(
        self, *, query_id: str, trial_id: str, answer: Mapping[str, Any]
    ) -> None:
        """Preserve controller answer events, including partial timeout audits."""

        self._require_active_trial()
        assert self._active_plan is not None
        if (
            str(query_id) != self._active_plan.query_id
            or str(trial_id) != self._active_plan.trial_id
        ):
            raise RuntimeError("query-answer identifiers do not match active trial")
        _append_table_row(
            self._file["query_answers"],
            {
                "query_id": str(query_id),
                "trial_id": str(trial_id),
                "question_id": str(answer["question_id"]),
                "answer_status": str(answer["answer_status"]),
                "value_json": _json(answer.get("value")),
                "prompt_shown_simulation_time_s": float(
                    answer["prompt_shown_simulation_time_s"]
                ),
                "prompt_shown_monotonic_ns": int(
                    answer["prompt_shown_monotonic_ns"]
                ),
                "prompt_shown_unix_ns": int(
                    answer["prompt_shown_unix_ns"]
                ),
                "prompt_shown_control_step": int(
                    answer["prompt_shown_control_step"]
                ),
                "first_input_simulation_time_s": float(
                    answer["first_input_simulation_time_s"]
                ),
                "first_input_monotonic_ns": int(
                    answer["first_input_monotonic_ns"]
                ),
                "first_input_unix_ns": int(answer["first_input_unix_ns"]),
                "first_input_control_step": int(
                    answer["first_input_control_step"]
                ),
                "confirmed_simulation_time_s": float(
                    answer["confirmed_simulation_time_s"]
                ),
                "confirmed_monotonic_ns": int(
                    answer["confirmed_monotonic_ns"]
                ),
                "confirmed_unix_ns": int(answer["confirmed_unix_ns"]),
                "confirmed_control_step": int(
                    answer["confirmed_control_step"]
                ),
                "response_latency_ms": float(answer["response_latency_ms"]),
                "input_device": str(answer["input_device"]),
                "back_correction_count": int(answer["back_correction_count"]),
                "accidental_input_count": int(answer["accidental_input_count"]),
            },
        )
        self._query_answer_count += 1
        self._file.attrs["query_answer_count"] = self._query_answer_count
        self._file.flush()

    def append_encounter(self, record: EncounterRecordV1) -> None:
        self._require_active_trial()
        assert self._active_plan is not None
        if (
            record.trial_id != self._active_plan.trial_id
            or record.encounter_id != self._active_plan.encounter_id
        ):
            raise RuntimeError("encounter identifiers do not match active trial")
        if int(self._active_trial.attrs.get("encounter_record_count", 0)) != 0:
            raise RuntimeError("exactly one safety-response encounter is allowed per trial")
        fields = record.as_dict()
        fields["encounter_timeout"] = int(record.encounter_timeout)
        _append_table_row(self._file["encounters"], fields)
        self._active_trial.attrs["encounter_record_count"] = 1
        self._encounter_count += 1
        self._file.attrs["encounter_count"] = self._encounter_count
        self._file.flush()

    def end_trial(self, summary: Mapping[str, Any]) -> None:
        self._require_active_trial()
        required = {
            "protocol_valid",
            "actual_crossing_count",
            "actual_crossing_direction",
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "tracking_validity_rate",
            "actual_minimum_gap_m",
            "actual_minimum_any_hand_gap_m",
            "actual_minimum_proximal_gap_m",
            "closest_proximal_collider",
            "planned_corridor_json",
            "off_protocol",
            "off_protocol_reasons",
            "cbf_activated",
            "response_onset_step",
            "response_onset_confirmed_step",
            "response_end_step_exclusive",
            "response_onset_simulation_time_s",
            "response_end_simulation_time_s",
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
            "cbf_max_intervention_norm_rad_s",
            "cbf_intervention_count",
            "windowed_jerk_json",
            "response_phase_history_json",
            "response_phase_durations_json",
            "delta_path_m",
            "delta_completion_time_s",
            "decision_context_json",
            "trial_fsm_history_json",
            "realtime_safety_marker_count",
            "realtime_behavior_anomaly_marker_count",
            "task_success",
            "task_failure_reason",
            "object_drop",
            "grasp_outcome",
            "release_outcome",
            "task_completion_observed",
            "completion_time_s",
            "completion_time_semantics",
            "trial_simulation_duration_s",
            "trial_wall_duration_s",
            "risk_onset_step",
            "cbf_intervention_start_step",
            "cbf_intervention_confirmed_step",
            "end_simulation_time_s",
            "end_monotonic_ns",
            "end_unix_ns",
        }
        missing = sorted(required - set(summary))
        if missing:
            raise RuntimeError(f"trial summary missing fields: {missing}")
        summary_bools = {
            name: bool(exact_bool(summary[name], name=f"trial summary {name}"))
            for name in (
                "protocol_valid", "off_protocol", "cbf_activated",
                "task_success", "object_drop", "task_completion_observed",
            )
        }
        raw_off_protocol_reasons = summary["off_protocol_reasons"]
        if not isinstance(raw_off_protocol_reasons, (list, tuple)) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in raw_off_protocol_reasons
        ):
            raise RuntimeError(
                "off_protocol_reasons must be a list of non-empty strings"
            )
        off_protocol_reasons = tuple(sorted(set(raw_off_protocol_reasons)))
        missing_onset = "response_onset_not_confirmed" in off_protocol_reasons
        missing_stable = (
            "stable_task_resumption_not_confirmed" in off_protocol_reasons
        )
        if missing_onset and not missing_stable:
            raise RuntimeError(
                "missing response onset also requires missing stable resumption"
            )
        if (missing_onset or missing_stable) and (
            not summary_bools["off_protocol"]
            or summary_bools["protocol_valid"]
        ):
            raise RuntimeError(
                "incomplete response episodes must be explicit invalid "
                "off-protocol trials"
            )
        assert self._active_trial is not None and self._active_plan is not None
        for name, expected in (
            ("condition_id", self._active_plan.condition_id),
            ("objective_mode", self._active_plan.objective_mode),
            ("lambda_s", float(self._active_plan.lambda_s)),
        ):
            if name in summary and summary[name] != expected:
                raise RuntimeError(f"final trial {name} differs from its assignment")
        if "actual_cbf_config_json" in summary:
            final_cbf_config = _decoded_json(
                summary["actual_cbf_config_json"], expected_type=dict
            )
            initial_cbf_config = json.loads(
                str(self._active_trial.attrs["actual_cbf_config_json"])
            )
            if final_cbf_config != initial_cbf_config:
                raise RuntimeError("actual CBF configuration changed within the trial")
        history = _decoded_json(summary["trial_fsm_history_json"], expected_type=list)
        expected_history = [state.value for state in TrialState]
        if history != expected_history:
            raise RuntimeError("trial FSM history is not the exact frozen 17-state sequence")
        decision_context = _decoded_json(
            summary["decision_context_json"], expected_type=dict
        )
        if decision_context:
            validate_decision_context(decision_context)
            if (
                str(decision_context.get("condition_id"))
                != self._active_plan.condition_id
                or float(decision_context.get("lambda_s", -1.0))
                != float(self._active_plan.lambda_s)
            ):
                raise RuntimeError(
                    "decision context condition differs from active trial"
                )
        elif summary_bools["protocol_valid"]:
            raise RuntimeError(
                "a protocol-valid trial requires a complete pre-response decision context"
            )
        response_phase_history = _decoded_json(
            summary["response_phase_history_json"], expected_type=list
        )
        history_phases: list[str] = []
        previous_step = -1
        previous_simulation_time_s = -math.inf
        for index, item in enumerate(response_phase_history):
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"response phase history entry {index} must be an object"
                )
            phase_name = str(item.get("phase", ""))
            control_step = item.get("control_step")
            simulation_time_s = float(item.get("simulation_time_s", math.nan))
            if phase_name not in RESPONSE_PHASES:
                raise RuntimeError(
                    f"response phase history entry {index} has an invalid phase"
                )
            if (
                isinstance(control_step, bool)
                or not isinstance(control_step, (int, np.integer))
                or int(control_step) < previous_step
                or not math.isfinite(simulation_time_s)
                or simulation_time_s < previous_simulation_time_s
            ):
                raise RuntimeError(
                    f"response phase history entry {index} has invalid clocks"
                )
            if history_phases and phase_name == history_phases[-1]:
                raise RuntimeError(
                    "response phase history must contain transitions, not repeated phases"
                )
            history_phases.append(phase_name)
            previous_step = int(control_step)
            previous_simulation_time_s = simulation_time_s
        if not history_phases or history_phases[0] != "PRE_RESPONSE":
            raise RuntimeError("response phase history must start PRE_RESPONSE")
        if missing_stable:
            if "STABLE_TASK_RESUMPTION" in history_phases:
                raise RuntimeError(
                    "missing-stable trial cannot contain a stable response transition"
                )
        elif history_phases[-1] != "STABLE_TASK_RESUMPTION":
            raise RuntimeError(
                "complete response phase history must end "
                "STABLE_TASK_RESUMPTION"
            )
        windowed_jerk = _decoded_json(
            summary["windowed_jerk_json"], expected_type=dict
        )
        required_windows = {
            "PRE_RESPONSE",
            "CBF_ACTIVE",
            "SMOOTH_TAIL",
            "RECOVERY_ACTIVE",
            "BC_RESUMED_FIRST_0_5_S",
            "WHOLE_TASK",
        }
        if set(windowed_jerk) != required_windows:
            raise RuntimeError(
                "windowed jerk summary must contain exactly: "
                + ", ".join(sorted(required_windows))
            )
        jerk_channel_fields = {
            "ee": ("sample_count", "rms_norm_m_s3", "peak_norm_m_s3"),
            "joint_space": (
                "sample_count", "rms_norm_rad_s3", "peak_norm_rad_s3"
            ),
        }
        for window_name in required_windows:
            window = windowed_jerk.get(window_name)
            if not isinstance(window, Mapping) or set(window) != set(jerk_channel_fields):
                raise RuntimeError(
                    f"windowed jerk {window_name} must contain exact ee/joint_space channels"
                )
            for channel_name, field_names in jerk_channel_fields.items():
                channel = window[channel_name]
                if not isinstance(channel, Mapping) or set(channel) != set(field_names):
                    raise RuntimeError(
                        f"windowed jerk {window_name}/{channel_name} has invalid fields"
                    )
                sample_count = channel["sample_count"]
                if (
                    isinstance(sample_count, bool)
                    or not isinstance(sample_count, (int, np.integer))
                    or int(sample_count) < 0
                ):
                    raise RuntimeError(
                        f"windowed jerk {window_name}/{channel_name} sample_count is invalid"
                    )
                for field_name in field_names[1:]:
                    value = float(channel[field_name])
                    if not math.isfinite(value) or value < 0.0:
                        raise RuntimeError(
                            f"windowed jerk {window_name}/{channel_name}/{field_name} is invalid"
                        )
        numeric_nonnegative = (
            "actual_path_deviation_rms_m",
            "actual_path_deviation_max_m",
            "actual_crossing_speed_m_s",
            "cbf_active_duration_s",
            "smooth_tail_duration_s",
            "recovery_duration_s",
            "bc_resumed_duration_s",
            "total_safety_response_duration_s",
            "integrated_intervention_rad",
            "correction_total_variation_rad_s",
            "cbf_max_intervention_norm_rad_s",
            "trial_simulation_duration_s",
            "trial_wall_duration_s",
        )
        for name in numeric_nonnegative:
            value = float(summary[name])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"trial summary {name} must be finite and non-negative")
        tracking_validity_rate = float(summary["tracking_validity_rate"])
        if not math.isfinite(tracking_validity_rate) or not 0.0 <= tracking_validity_rate <= 1.0:
            raise RuntimeError("tracking_validity_rate must be inside [0, 1]")
        for name in (
            "actual_minimum_gap_m", "actual_minimum_any_hand_gap_m",
            "actual_minimum_proximal_gap_m", "delta_path_m",
            "delta_completion_time_s",
        ):
            if not math.isfinite(float(summary[name])):
                raise RuntimeError(f"trial summary {name} must be finite")
        actual_crossing_direction = str(summary["actual_crossing_direction"])
        if actual_crossing_direction not in ACTUAL_CROSSING_DIRECTIONS:
            raise RuntimeError("actual_crossing_direction is invalid")
        direction_not_observed = actual_crossing_direction == "not_observed"
        direction_not_observed_reason = (
            "actual_crossing_direction_not_observed" in off_protocol_reasons
        )
        if direction_not_observed != direction_not_observed_reason:
            raise RuntimeError(
                "not-observed crossing direction must exactly match its "
                "off-protocol reason"
            )
        if direction_not_observed and not summary_bools["off_protocol"]:
            raise RuntimeError(
                "not-observed crossing direction must be off-protocol"
            )
        if int(summary["actual_crossing_count"]) < 0:
            raise RuntimeError("actual_crossing_count must be non-negative")
        for name in (
            "cbf_intervention_count", "realtime_safety_marker_count",
            "realtime_behavior_anomaly_marker_count",
        ):
            if isinstance(summary[name], bool) or int(summary[name]) < 0:
                raise RuntimeError(f"trial summary {name} must be non-negative")
        if (
            int(summary["realtime_safety_marker_count"])
            != self._trial_marker_counts["realtime_safety_concern"]
            or int(summary["realtime_behavior_anomaly_marker_count"])
            != self._trial_marker_counts["realtime_behavior_anomaly"]
        ):
            raise RuntimeError("trial marker summary disagrees with recorded marker events")
        onset_sim = float(summary["response_onset_simulation_time_s"])
        end_sim = float(summary["response_end_simulation_time_s"])
        if not all(math.isfinite(value) for value in (onset_sim, end_sim)):
            raise RuntimeError("response timestamps must be finite")
        onset_step = int(summary["response_onset_step"])
        confirmed_step = int(summary["response_onset_confirmed_step"])
        end_step = int(summary["response_end_step_exclusive"])
        stable_step = int(summary["stable_task_resumption_step"])
        recovery_step = int(summary["recovery_onset_step"])
        recovery_end_step = int(summary["recovery_end_step_exclusive"])
        recovery_onset_sim = float(
            summary["recovery_onset_simulation_time_s"]
        )
        recovery_end_sim = float(
            summary["recovery_end_simulation_time_s"]
        )
        if not all(
            math.isfinite(value)
            for value in (recovery_onset_sim, recovery_end_sim)
        ):
            raise RuntimeError("Recovery timestamps must be finite")
        if missing_onset:
            if (
                (onset_sim, end_sim) != (-1.0, -1.0)
                or (recovery_onset_sim, recovery_end_sim) != (-1.0, -1.0)
                or (
                    onset_step,
                    confirmed_step,
                    end_step,
                    stable_step,
                    recovery_step,
                    recovery_end_step,
                ) != (-1, -1, -1, -1, -1, -1)
                or any(
                    float(summary[name]) != 0.0
                    for name in (
                        "total_safety_response_duration_s",
                        "integrated_intervention_rad",
                        "correction_total_variation_rad_s",
                        "recovery_duration_s",
                    )
                )
            ):
                raise RuntimeError(
                    "unconfirmed response onset requires exact absent-response sentinels"
                )
        elif missing_stable:
            if (
                onset_sim < 0.0
                or onset_step < 0
                or confirmed_step < onset_step
                or end_sim != -1.0
                or end_step != -1
                or stable_step != -1
                or any(
                    float(summary[name]) != 0.0
                    for name in (
                        "total_safety_response_duration_s",
                        "integrated_intervention_rad",
                        "correction_total_variation_rad_s",
                    )
                )
            ):
                raise RuntimeError(
                    "unstable response timeout requires valid onset and exact missing-end sentinels"
                )
        elif not (
            0.0 <= onset_sim <= end_sim
            and 0 <= onset_step <= confirmed_step < end_step
            and stable_step == end_step - 1
        ):
            raise RuntimeError("complete response bounds are invalid")

        recovery_active = np.asarray(
            self._active_trial["recovery/active"][()], dtype=np.int8
        )
        recovery_onset_rows = np.asarray(
            self._active_trial["recovery/onset_now"][()], dtype=np.int8
        )
        recovery_end_rows = np.asarray(
            self._active_trial["recovery/end_now"][()], dtype=np.int8
        )
        expected_recovery_onsets = (recovery_active == 1) & np.concatenate(
            (np.asarray([True]), recovery_active[:-1] == 0)
        )
        expected_recovery_ends = (recovery_active == 0) & np.concatenate(
            (np.asarray([False]), recovery_active[:-1] == 1)
        )
        if (
            np.any(recovery_onset_rows != expected_recovery_onsets.astype(np.int8))
            or np.any(recovery_end_rows != expected_recovery_ends.astype(np.int8))
        ):
            raise RuntimeError(
                "Recovery onset/end row evidence must exactly encode raw active-mask edges"
            )
        if not missing_onset:
            source_steps = np.asarray(
                self._active_trial["control/source_step"][()], dtype=np.int64
            )
            simulation_times = np.asarray(
                self._active_trial["control/sim_time"][()], dtype=np.float64
            )
            trial_states = np.asarray(
                self._active_trial["control/trial_state"].asstr()[()]
            )
            start_rows = np.flatnonzero(source_steps == onset_step)
            if start_rows.size != 1:
                raise RuntimeError("response onset step does not map to one control row")
            start_index = int(start_rows[0])
            if missing_stable:
                return_rows = np.flatnonzero(
                    trial_states
                    == TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value
                )
                window_end = (
                    int(return_rows[0]) if return_rows.size else len(source_steps)
                )
            else:
                stable_rows = np.flatnonzero(source_steps == stable_step)
                if stable_rows.size != 1:
                    raise RuntimeError(
                        "stable response step does not map to one control row"
                    )
                window_end = int(stable_rows[0]) + 1
            response_recovery = recovery_active[start_index:window_end]
            active_offsets = np.flatnonzero(response_recovery == 1)
            if active_offsets.size:
                first_recovery_index = start_index + int(active_offsets[0])
                last_recovery_index = start_index + int(active_offsets[-1])
                expected_recovery_step = int(source_steps[first_recovery_index])
                expected_recovery_onset_sim = float(
                    simulation_times[first_recovery_index]
                )
                if (
                    last_recovery_index + 1 < window_end
                    and recovery_active[last_recovery_index + 1] == 0
                ):
                    expected_recovery_end_step = int(
                        source_steps[last_recovery_index + 1]
                    )
                    expected_recovery_end_sim = float(
                        simulation_times[last_recovery_index + 1]
                    )
                else:
                    expected_recovery_end_step = -1
                    expected_recovery_end_sim = -1.0
            else:
                expected_recovery_step = -1
                expected_recovery_onset_sim = -1.0
                expected_recovery_end_step = -1
                expected_recovery_end_sim = -1.0
            expected_recovery_duration = float(
                np.sum(response_recovery)
                * float(self._file.attrs["physics_dt_s"])
            )
            if (
                recovery_step != expected_recovery_step
                or recovery_end_step != expected_recovery_end_step
                or not math.isclose(
                    recovery_onset_sim,
                    expected_recovery_onset_sim,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    recovery_end_sim,
                    expected_recovery_end_sim,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(summary["recovery_duration_s"]),
                    expected_recovery_duration,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "Recovery summary does not exactly reconstruct from its raw mask"
                )

        all_trial_states = np.asarray(
            self._active_trial["control/trial_state"].asstr()[()]
        )
        required_hold = np.isin(
            all_trial_states,
            (
                TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            ),
        )
        if np.any(required_hold):
            hold_actions = np.asarray(
                self._active_trial["actions/controller_task_action"][()],
                dtype=np.float64,
            )
            hold_reasons = np.asarray(
                self._active_trial[
                    "actions/controller_task_action_reason"
                ].asstr()[()]
            )
            hold_modes = np.asarray(
                self._active_trial["control/control_mode"].asstr()[()]
            )
            hold_phase_advance = np.asarray(
                self._active_trial["task/phase_advance_enabled"][()],
                dtype=np.int8,
            )
            hold_recovery_authority = np.asarray(
                self._active_trial["recovery/control_authority"][()],
                dtype=np.int8,
            )
            hold_pipeline_complete = np.asarray(
                self._active_trial["actions/action_pipeline_complete"][()],
                dtype=np.int8,
            )
            hold_intervention_available = np.asarray(
                self._active_trial["cbf/intervention_available"][()],
                dtype=np.int8,
            )
            hold_geometry_valid = np.asarray(
                self._active_trial["safety/geometry_valid"][()],
                dtype=np.int8,
            )
            if (
                np.any(hold_actions[required_hold] != 0.0)
                or np.any(
                    hold_reasons[required_hold]
                    != "current_ee_protocol_hold"
                )
                or np.any(hold_modes[required_hold] != "protocol_hold")
                or np.any(hold_phase_advance[required_hold] != 0)
                or np.any(hold_recovery_authority[required_hold] != 0)
                or np.any(hold_pipeline_complete[required_hold] != 1)
                or np.any(hold_intervention_available[required_hold] != 1)
                or np.any(hold_geometry_valid[required_hold] != 1)
            ):
                raise RuntimeError(
                    "query/pause rows do not prove the exact current-EE hold through the complete valid CBF pipeline"
                )
        off_protocol = summary_bools["off_protocol"]
        response_status = str(self._active_trial.attrs.get("query_response_status", ""))
        protocol_valid = summary_bools["protocol_valid"]
        if int(summary["actual_crossing_count"]) != 1 or off_protocol:
            protocol_valid = False
        if response_status != "completed":
            protocol_valid = False
        if int(self._active_trial.attrs.get("query_count", 0)) != 1:
            protocol_valid = False
        encounter_record_count = int(
            self._active_trial.attrs.get("encounter_record_count", 0)
        )
        if missing_onset:
            if encounter_record_count not in (0, 1):
                raise RuntimeError(
                    "unconfirmed response trial may contain at most one observed risk encounter"
                )
        elif encounter_record_count != 1:
            raise RuntimeError(
                "a confirmed response trial requires exactly one encounter"
            )
        if encounter_record_count != 1:
            protocol_valid = False
        if (
            onset_sim < 0.0
            or end_sim < onset_sim
            or int(summary["response_onset_step"]) < 0
            or int(summary["response_end_step_exclusive"])
            <= int(summary["response_onset_step"])
        ):
            protocol_valid = False
        attrs = dict(summary)
        attrs["protocol_valid"] = int(protocol_valid)
        attrs["trial_complete"] = 1
        attrs["analysis_exclude"] = int(
            bool(self._active_plan.analysis_exclude) or not protocol_valid
        )
        attrs["off_protocol"] = int(off_protocol)
        attrs["off_protocol_reasons_json"] = _json(off_protocol_reasons)
        del attrs["off_protocol_reasons"]
        attrs["trial_fsm_history_json"] = _json(history)
        attrs["decision_context_json"] = _json(decision_context)
        attrs["response_phase_history_json"] = _json(response_phase_history)
        attrs["windowed_jerk_json"] = _json(windowed_jerk)
        phase_durations = _decoded_json(
            summary["response_phase_durations_json"], expected_type=dict
        )
        if set(phase_durations) != set(RESPONSE_PHASES):
            raise RuntimeError(
                "response phase duration summary must contain exactly all six phases"
            )
        for phase_name in RESPONSE_PHASES:
            duration = float(phase_durations[phase_name])
            if not math.isfinite(duration) or duration < 0.0:
                raise RuntimeError(
                    f"response phase duration {phase_name} must be finite and non-negative"
                )
        for phase_name, summary_name, maximum_duration_s in (
            ("CBF_ACTIVE", "cbf_active_duration_s", None),
            ("SMOOTH_TAIL", "smooth_tail_duration_s", None),
            ("BC_RESUMED", "bc_resumed_duration_s", 0.5),
        ):
            expected_duration = float(phase_durations[phase_name])
            if maximum_duration_s is not None:
                expected_duration = min(maximum_duration_s, expected_duration)
            if not math.isclose(
                expected_duration,
                float(summary[summary_name]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"response phase duration {phase_name} disagrees with {summary_name}"
                )
        attrs["response_phase_durations_json"] = _json(phase_durations)
        for name, value in attrs.items():
            self._active_trial.attrs[name] = _attribute_value(value)
        self._file.flush()
        self._active_trial = None
        self._active_plan = None
        self._datasets = {}
        self._trial_length = 0
        self._trial_marker_counts = {marker_type: 0 for marker_type in MARKER_TYPES}
        self._trial_count += 1
        self._file.attrs["trial_count"] = self._trial_count

    def update_spectator_video_metadata(
        self, metadata: Mapping[str, Any]
    ) -> None:
        """Bind the finalized video/frame-clock summary before sealing."""

        self._require_open()
        payload = dict(metadata)
        if str(payload.get("schema_version", "")) != (
            "ac_spectator_video_sync_v1"
        ):
            raise ValueError("unexpected spectator video metadata schema")
        self._file.attrs["spectator_video_metadata_json"] = _json(payload)
        self._file.flush()

    def seal(self) -> Path:
        self._require_open()
        if self._active_trial is not None:
            raise RuntimeError("cannot seal while a trial is active")
        if self._trial_count <= 0:
            raise RuntimeError("cannot seal an empty collection")
        if not self._schedule or self._trial_count != len(self._schedule):
            raise RuntimeError(
                "cannot seal before every preassigned schedule entry is recorded"
            )
        trials = self._file["trials"]
        expected_encounter_count = sum(
            int(group.attrs.get("encounter_record_count", 0))
            for group in trials.values()
        )
        if (
            self._query_count != self._trial_count
            or not 0 <= self._encounter_count <= self._trial_count
            or self._encounter_count != expected_encounter_count
        ):
            raise RuntimeError(
                "cannot seal without one query per trial and an exact "
                "observed-encounter count"
            )
        self._file.attrs.update(
            {
                "collection_complete": 1,
                "trial_count": self._trial_count,
                "query_count": self._query_count,
                "query_answer_count": self._query_answer_count,
                "encounter_count": self._encounter_count,
                "realtime_marker_count": self._marker_count,
                "layout_precheck_count": self._layout_precheck_count,
                "session_end_monotonic_ns": time.monotonic_ns(),
                "session_end_unix_ns": time.time_ns(),
            }
        )
        self._file.flush()
        self._file.close()
        self._closed = True
        self._sealed = True
        return self.partial_path

    def commit_validated(self) -> Path:
        if not self._sealed or not self._closed:
            raise RuntimeError("seal and validate before commit")
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        os.replace(self.partial_path, self.output_path)
        return self.output_path

    def abort(self, reason: str) -> Path:
        if self._closed:
            return self.partial_path
        if self._active_trial is not None:
            self._active_trial.attrs.update(
                {
                    "trial_complete": 0,
                    "protocol_valid": 0,
                    "analysis_exclude": 1,
                    "abort_reason": str(reason),
                    "step_count": self._trial_length,
                }
            )
        self._file.attrs.update(
            {
                "collection_complete": 0,
                "collection_abort_reason": str(reason).strip() or "unspecified_abort",
                "session_end_monotonic_ns": time.monotonic_ns(),
                "session_end_unix_ns": time.time_ns(),
            }
        )
        self._file.flush()
        self._file.close()
        self._closed = True
        return self.partial_path

    def _write_root_metadata(
        self,
        metadata: Mapping[str, Any],
        protocol_config: Mapping[str, Any],
        scenario_config: Mapping[str, Any],
        frozen_config: Mapping[str, Any],
    ) -> None:
        required = (
            "participant_id",
            "session_id",
            "code_version",
            "code_version_source",
            "code_commit_verification",
            "source_tree_sha256",
            "cbf_config_json",
            "cbf_config_sha256",
            "scenario_config_sha256",
            "protocol_config_sha256",
        )
        missing = [name for name in required if not str(metadata.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required session metadata: {missing}")
        protocol_config_json = _json(dict(protocol_config))
        scenario_config_json = _json(dict(scenario_config))
        cbf_config = _decoded_json(
            metadata["cbf_config_json"], expected_type=dict
        )
        cbf_config_json = _json(cbf_config)
        derived_configs = {
            "protocol": (protocol_config_json, _sha256_json(protocol_config)),
            "scenario": (scenario_config_json, _sha256_json(scenario_config)),
            "cbf": (cbf_config_json, _sha256_json(cbf_config)),
        }
        for prefix, (_, expected_sha256) in derived_configs.items():
            actual_sha256 = str(metadata.get(f"{prefix}_config_sha256", ""))
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{prefix}_config_sha256 does not bind to the supplied config"
                )
        frozen_config_json = _json(dict(frozen_config))
        frozen_config_sha256 = hashlib.sha256(
            frozen_config_json.encode("utf-8")
        ).hexdigest()
        attrs: dict[str, Any] = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "protocol_version": ONLINE_PROTOCOL_VERSION,
            "row_semantics": ONLINE_ROW_SEMANTICS,
            "feature_provenance_version": FEATURE_PROVENANCE_VERSION,
            "collection_complete": 0,
            "production_mode": int(not bool(metadata.get("practice_only", False))),
            "pilot": 1,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_mode": "frozen_direct_bc",
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
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
            "cbf_scope": "tracked_hands_to_runtime_discovered_distal_panda_colliders",
            "cbf_constraint_identity_semantics": "exact_reconstruction_from_frozen_cbf_filter_input_v1",
            "single_pick": 1,
            "cube_count": 1,
            "intended_crossings_per_trial": 1,
            "queries_per_trial": 1,
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v5",
            "state_aware_recovery_enabled": 1,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "recovery_fixed": 1,
            "task_require_release": 1,
            "strict_place_xy_tolerance_m": 0.05,
            "task_max_episode_steps": 12000,
            "task_phase_clock_latched_during_pre_crossing_cue": 1,
            "task_progression_during_crossing_and_recovery": 1,
            "task_progression_paused_during_query": 1,
            "physics_remains_on_during_query": 1,
            "cbf_remains_on_during_query": 1,
            "safe_hold_during_query": 1,
            "emergency_stop_remains_on_during_query": 1,
            "query_encounter_nullable": 0,
            "null_string_encoding": "empty_utf8",
            "q1_id": Q1_ID,
            "q1_text_ko": Q1_TEXT_KO,
            "q1_responses_json": _json(Q1_RESPONSES),
            "no_marker_semantics": "missing_evidence_not_acceptable",
            "uncertain_semantics": "abstain",
            "timeout_semantics": "missing_abstain",
            "likert_questions_json": _json(LIKERT_QUESTIONS),
            "modification_reasons_json": _json(MODIFICATION_REASONS),
            "realtime_marker_types_json": _json(MARKER_TYPES),
            "runtime_observable_fields_json": _json(RUNTIME_OBSERVABLE_FIELDS),
            "future_outcome_fields_json": _json(FUTURE_OUTCOME_FIELDS),
            "protocol_config_json": protocol_config_json,
            "protocol_config_sha256": derived_configs["protocol"][1],
            "scenario_config_json": scenario_config_json,
            "scenario_config_sha256": derived_configs["scenario"][1],
            "cbf_config_json": cbf_config_json,
            "cbf_config_sha256": derived_configs["cbf"][1],
            "frozen_config_json": frozen_config_json,
            "frozen_config_sha256": frozen_config_sha256,
            "trial_count": 0,
            "query_count": 0,
            "query_answer_count": 0,
            "encounter_count": 0,
            "realtime_marker_count": 0,
            "layout_precheck_count": 0,
            "session_start_monotonic_ns": time.monotonic_ns(),
            "session_start_unix_ns": time.time_ns(),
        }
        locked = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "policy_path": POLICY_RELATIVE_PATH,
            "policy_sha256": POLICY_SHA256,
            "policy_size_bytes": POLICY_SIZE_BYTES,
            "policy_frozen": 1,
            "policy_online_updates": 0,
            "policy_fine_tuning": 0,
            "pilot": 1,
            "policy_mode": "frozen_direct_bc",
            "policy_observation_version": "obs_v1_state_controller_phase",
            "policy_action_version": "action_v1_controller_target_delta",
            "human_observations_masked": 1,
            "feedback_changes_schedule": 0,
            "condition_schedule_preassigned": 1,
            "condition_blinding": 1,
            "condition_identity_visible": 0,
            "haptics_enabled": 0,
            "haptic_event_count": 0,
            "cbf_frozen": 1,
            "cbf_feedback_adaptation": 0,
            "cbf_parameter_adaptation": 0,
            "recovery_fixed": 1,
            "state_aware_recovery_enabled": 1,
            "state_aware_recovery_schema": "state_aware_pick_place_recovery_bridge_v3",
            "strict_task_semantics_schema_version": "physical_event_driven_pick_place_v5",
            "physics_remains_on_during_query": 1,
            "cbf_remains_on_during_query": 1,
            "safe_hold_during_query": 1,
            "emergency_stop_remains_on_during_query": 1,
            "no_marker_semantics": "missing_evidence_not_acceptable",
            "uncertain_semantics": "abstain",
            "timeout_semantics": "missing_abstain",
            "frozen_config_json": frozen_config_json,
            "frozen_config_sha256": frozen_config_sha256,
            "protocol_config_json": protocol_config_json,
            "protocol_config_sha256": derived_configs["protocol"][1],
            "scenario_config_json": scenario_config_json,
            "scenario_config_sha256": derived_configs["scenario"][1],
            "cbf_config_json": cbf_config_json,
            "cbf_config_sha256": derived_configs["cbf"][1],
        }
        for name, expected in locked.items():
            if name in metadata and metadata[name] != expected:
                raise ValueError(f"session metadata attempts to override frozen {name}")
        attrs.update(dict(metadata))
        attrs.update(locked)
        for name, value in attrs.items():
            self._file.attrs[str(name)] = _attribute_value(value)

    def _create_tables(self) -> None:
        tables: dict[str, dict[str, DatasetSpec]] = {
            "schedule": {
                "trial_index": DatasetSpec(np.int64),
                "trial_id": DatasetSpec("str"),
                "query_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "task_phase": DatasetSpec("str"),
                "severity": DatasetSpec("str"),
                "direction": DatasetSpec("str"),
                "speed": DatasetSpec("str"),
                "crossing_hand": DatasetSpec("str"),
                "condition_id": DatasetSpec("str"),
                "objective_mode": DatasetSpec("str"),
                "lambda_s": DatasetSpec(np.float64),
                "schedule_seed": DatasetSpec(np.int64),
                "block_id": DatasetSpec("str"),
                "counterbalancing_group": DatasetSpec("str"),
                "practice": DatasetSpec(np.int8),
                "anchor_repeat": DatasetSpec(np.int8),
                "pilot": DatasetSpec(np.int8),
                "analysis_exclude": DatasetSpec(np.int8),
                "source_trial_id": DatasetSpec("str"),
            },
            "layout_prechecks": {
                "candidate_index": DatasetSpec(np.int64),
                "layout_id": DatasetSpec("str"),
                "seed": DatasetSpec(np.int64),
                "controller": DatasetSpec("str"),
                "policy_sha256": DatasetSpec("str"),
                "cbf_enabled": DatasetSpec(np.int8),
                "human_absent_or_clear": DatasetSpec(np.int8),
                "strict_success": DatasetSpec(np.int8),
                "release_observed": DatasetSpec(np.int8),
                "nominal_phase_references_complete": DatasetSpec(np.int8),
                "nominal_phase_references_json": DatasetSpec("str"),
                "nominal_corridor_bank_complete": DatasetSpec(np.int8),
                "nominal_phase_trajectories_json": DatasetSpec("str"),
                "nominal_corridor_bank_json": DatasetSpec("str"),
                "steps": DatasetSpec(np.int64),
                "diagnostic_group": DatasetSpec("str"),
                "failure_reason": DatasetSpec("str"),
                "started_monotonic_ns": DatasetSpec(np.int64),
                "ended_monotonic_ns": DatasetSpec(np.int64),
                "source_restoration_json": DatasetSpec("str"),
            },
            "realtime_markers": {
                "marker_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "participant_id": DatasetSpec("str"),
                "session_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "condition_id": DatasetSpec("str"),
                "lambda_s": DatasetSpec(np.float64),
                "marker_type": DatasetSpec("str"),
                "simulation_time_s": DatasetSpec(np.float64),
                "monotonic_ns": DatasetSpec(np.int64),
                "unix_ns": DatasetSpec(np.int64),
                "control_step": DatasetSpec(np.int64),
                "response_phase": DatasetSpec("str"),
                "time_since_response_onset_s": DatasetSpec(np.float64),
                "time_since_recovery_onset_s": DatasetSpec(np.float64),
                "controller_hand": DatasetSpec("str"),
                "button": DatasetSpec("str"),
                "button_identifier": DatasetSpec("str"),
            },
            "queries": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "encounter_id": DatasetSpec("str"),
                "issued_simulation_time_s": DatasetSpec(np.float64),
                "issued_monotonic_ns": DatasetSpec(np.int64),
                "issued_unix_ns": DatasetSpec(np.int64),
                "issued_control_step": DatasetSpec(np.int64),
                "completed_simulation_time_s": DatasetSpec(np.float64),
                "completed_monotonic_ns": DatasetSpec(np.int64),
                "completed_unix_ns": DatasetSpec(np.int64),
                "completed_control_step": DatasetSpec(np.int64),
                "response_status": DatasetSpec("str"),
                "response_disposition": DatasetSpec("str"),
                "q1_response": DatasetSpec("str"),
                "q2_perceived_danger": DatasetSpec(np.int8),
                "q3_abruptness": DatasetSpec(np.int8),
                "q4_excessive_duration": DatasetSpec(np.int8),
                "q5_task_disruption": DatasetSpec(np.int8),
                "q6_confidence": DatasetSpec(np.int8),
                "modification_reasons_json": DatasetSpec("str"),
                "response_latency_ms": DatasetSpec(np.float64),
                "input_device": DatasetSpec("str"),
                "first_input_simulation_time_s": DatasetSpec(np.float64),
                "first_input_monotonic_ns": DatasetSpec(np.int64),
                "first_input_unix_ns": DatasetSpec(np.int64),
                "first_input_control_step": DatasetSpec(np.int64),
                "back_correction_count": DatasetSpec(np.int32),
                "accidental_input_count": DatasetSpec(np.int32),
            },
            "query_answers": {
                "query_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "question_id": DatasetSpec("str"),
                "answer_status": DatasetSpec("str"),
                "value_json": DatasetSpec("str"),
                "prompt_shown_simulation_time_s": DatasetSpec(np.float64),
                "prompt_shown_monotonic_ns": DatasetSpec(np.int64),
                "prompt_shown_unix_ns": DatasetSpec(np.int64),
                "prompt_shown_control_step": DatasetSpec(np.int64),
                "first_input_simulation_time_s": DatasetSpec(np.float64),
                "first_input_monotonic_ns": DatasetSpec(np.int64),
                "first_input_unix_ns": DatasetSpec(np.int64),
                "first_input_control_step": DatasetSpec(np.int64),
                "confirmed_simulation_time_s": DatasetSpec(np.float64),
                "confirmed_monotonic_ns": DatasetSpec(np.int64),
                "confirmed_unix_ns": DatasetSpec(np.int64),
                "confirmed_control_step": DatasetSpec(np.int64),
                "response_latency_ms": DatasetSpec(np.float64),
                "input_device": DatasetSpec("str"),
                "back_correction_count": DatasetSpec(np.int32),
                "accidental_input_count": DatasetSpec(np.int32),
            },
            "encounters": {
                "encounter_id": DatasetSpec("str"),
                "trial_id": DatasetSpec("str"),
                "onset_step": DatasetSpec(np.int64),
                "onset_confirmed_step": DatasetSpec(np.int64),
                "offset_step_exclusive": DatasetSpec(np.int64),
                "onset_simulation_time_s": DatasetSpec(np.float64),
                "offset_simulation_time_s": DatasetSpec(np.float64),
                "onset_monotonic_ns": DatasetSpec(np.int64),
                "offset_monotonic_ns": DatasetSpec(np.int64),
                "window_start_step": DatasetSpec(np.int64),
                "window_end_step_exclusive": DatasetSpec(np.int64),
                "minimum_surface_gap_m": DatasetSpec(np.float64),
                "maximum_intervention_norm_rad_s": DatasetSpec(np.float64),
                "risk_onset_step": DatasetSpec(np.int64),
                "cbf_intervention_start_step": DatasetSpec(np.int64),
                "cbf_intervention_confirmed_step": DatasetSpec(np.int64),
                "encounter_timeout": DatasetSpec(np.int8),
                "merged_reentry_count": DatasetSpec(np.int32),
            },
        }
        for group_name, specs in tables.items():
            group = self._file.require_group(group_name)
            for name, spec in specs.items():
                _create_extendable_dataset(group, name, spec, h5py=self.h5py)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")

    def _require_active_trial(self) -> None:
        self._require_open()
        if self._active_trial is None or self._active_plan is None:
            raise RuntimeError("no active trial")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _decoded_json(value: Any, *, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError(
            f"expected {expected_type.__name__} or its JSON encoding, got {type(value).__name__}"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid JSON trial summary field") from error
    if not isinstance(decoded, expected_type):
        raise RuntimeError(
            f"JSON trial summary field must decode to {expected_type.__name__}"
        )
    return decoded


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(dict(value)).encode("utf-8")).hexdigest()
