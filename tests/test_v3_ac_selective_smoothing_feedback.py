"""Pure-Python contract tests for the A/C VR feedback collector.

These tests intentionally avoid Isaac Sim, Torch, HDF5, and a live XR runtime.
They protect the study design and controller-input state machines that must be
stable before an operator starts an in-HMD practice trial.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest
from unittest.mock import patch

from v3_chan.collect_ac_selective_smoothing_feedback import (
    CollectionAbort,
    _read_required_controller_inputs,
)
import numpy as np

from v3_chan.ac_feedback.config import DEFAULT_CONFIG, load_config, validate_config
from v3_chan.ac_feedback.online_protocol import (
    CrossingPathMonitor,
    resolve_actual_crossing_direction,
)
from v3_chan.ac_feedback.online_schema import ACTUAL_CROSSING_DIRECTIONS
from v3_chan.ac_feedback.online_schema import Q1_ID as ONLINE_Q1_ID
from v3_chan.ac_feedback.study import (
    ResponsePhase,
    ResponsePhaseConfig,
    SafetyResponsePhaseDetector,
    TASK_PHASES,
    TrialLifecycle,
    TrialState,
    build_participant_schedule,
    validate_decision_context,
)
from v3_chan.ac_feedback.xr_feedback import (
    ControllerButtonSnapshot,
    ControllerInputUnavailable,
    ControllerInputWatchdog,
    EmergencyAbortMonitor,
    FeedbackClocks,
    LIKERT_QUESTIONS_KO,
    Q1_RESPONSE_CODES,
    QuestionnaireButtonMapping,
    QuestionnaireFSM,
    REALTIME_BEHAVIOR_ANOMALY,
    REALTIME_SAFETY_CONCERN,
    REJECTION_REASON_IDS,
    RealtimeButtonMapping,
    RealtimeMarkerInput,
)
from v3_chan.ac_feedback.validator import (
    _Validator,
    _missing_resume_outcome_matches,
)


def _maximum_run(values: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def _clock(index: int) -> FeedbackClocks:
    return FeedbackClocks(
        sim_time_s=index / 10.0,
        monotonic_ns=1_000_000_000 + index * 100_000_000,
        unix_ns=2_000_000_000 + index * 100_000_000,
        control_step=index,
    )


class ScheduleContractTests(unittest.TestCase):
    def test_minimal_schedule_is_deterministic_and_balanced(self) -> None:
        schedule = build_participant_schedule(
            "pilot_p01", session_id="session_01", seed=11,
            mode="minimal_pilot",
        )
        repeated = build_participant_schedule(
            "pilot_p01", session_id="session_01", seed=11,
            mode="minimal_pilot",
        )

        self.assertEqual(schedule.trial_order, repeated.trial_order)
        self.assertEqual(schedule.schedule_seed, repeated.schedule_seed)
        self.assertEqual(len(schedule.practice_trials), 4)
        self.assertEqual(len(schedule.evaluated_trials), 16)
        self.assertTrue(all(trial.analysis_exclude for trial in schedule.practice_trials))
        core = list(schedule.evaluated_trials)
        self.assertEqual(
            Counter(trial.condition_id for trial in core),
            Counter({"A_reactive": 8, "C_smooth": 8}),
        )
        for phase in TASK_PHASES:
            phase_trials = [trial for trial in core if trial.task_phase == phase]
            self.assertEqual(
                Counter(trial.condition_id for trial in phase_trials),
                Counter({"A_reactive": 2, "C_smooth": 2}),
            )
        for severity in ("shallow", "threat"):
            severity_trials = [trial for trial in core if trial.severity == severity]
            self.assertEqual(
                Counter(trial.condition_id for trial in severity_trials),
                Counter({"A_reactive": 4, "C_smooth": 4}),
            )
        self.assertEqual(
            Counter(trial.crossing_direction for trial in core),
            Counter({"left_to_right": 8, "right_to_left": 8}),
        )
        self.assertEqual(
            Counter(trial.crossing_speed for trial in core),
            Counter({"slow": 8, "fast": 8}),
        )
        self.assertLessEqual(_maximum_run([trial.condition_id for trial in core]), 2)
        self.assertEqual(
            [trial.order_index for trial in schedule], list(range(len(schedule)))
        )

    def test_first_condition_is_counterbalanced_across_participant_ids(self) -> None:
        first_conditions = {
            build_participant_schedule(
                f"pilot_p{index:02d}",
                session_id="session_01",
                seed=11,
                mode="minimal_pilot",
            ).evaluated_trials[0].condition_id
            for index in range(1, 33)
        }
        self.assertEqual(first_conditions, {"A_reactive", "C_smooth"})

    def test_anchor_mode_adds_two_contexts_repeated_under_a_and_c(self) -> None:
        schedule = build_participant_schedule(
            "pilot_p01", session_id="session_anchors", seed=11,
            mode="pilot_with_anchors",
        )
        anchors = [trial for trial in schedule.evaluated_trials if trial.anchor_repeat]
        self.assertEqual(len(schedule.practice_trials), 4)
        self.assertEqual(len(schedule.evaluated_trials), 20)
        self.assertEqual(len(anchors), 4)
        self.assertEqual(len({trial.anchor_context_id for trial in anchors}), 2)
        for context_id in {trial.anchor_context_id for trial in anchors}:
            context = [trial for trial in anchors if trial.anchor_context_id == context_id]
            self.assertEqual(
                Counter(trial.condition_id for trial in context),
                Counter({"A_reactive": 1, "C_smooth": 1}),
            )


class TrialLifecycleTests(unittest.TestCase):
    def test_exact_seventeen_state_lifecycle_and_cardinality(self) -> None:
        lifecycle = TrialLifecycle("A_reactive")
        for state in list(TrialState)[1:]:
            lifecycle.advance(state)
            if state is TrialState.EXECUTE_SINGLE_CROSSING:
                lifecycle.record_crossing()
            elif state is TrialState.SHOW_MANDATORY_FEEDBACK_UI:
                lifecycle.record_query()
            elif state is TrialState.SAVE_FEEDBACK:
                lifecycle.record_feedback_saved()
        lifecycle.validate_complete()
        self.assertEqual(lifecycle.history, [state.value for state in TrialState])

    def test_illegal_skip_and_duplicate_crossing_fail_closed(self) -> None:
        lifecycle = TrialLifecycle("C_smooth")
        with self.assertRaisesRegex(RuntimeError, "illegal trial transition"):
            lifecycle.advance(TrialState.START_FROZEN_BC)
        for state in list(TrialState)[1:6]:
            lifecycle.advance(state)
        lifecycle.record_crossing()
        with self.assertRaisesRegex(RuntimeError, "more than one intended crossing"):
            lifecycle.record_crossing()

    def test_missing_resume_accepts_terminal_on_final_query_row(self) -> None:
        states = [
            TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
        ]
        self.assertTrue(
            _missing_resume_outcome_matches(
                states=states,
                step_success=[0, 0, 0],
                step_terminated=[0, 0, 1],
                step_truncated=[0, 0, 0],
                terminal_reasons=["", "", "object_drop"],
                summary_success=False,
                completion_observed=True,
                failure_reason="object_drop",
            )
        )
        self.assertTrue(
            _missing_resume_outcome_matches(
                states=states,
                step_success=[0, 0, 1],
                step_terminated=[0, 0, 0],
                step_truncated=[0, 0, 0],
                terminal_reasons=["", "", ""],
                summary_success=True,
                completion_observed=True,
                failure_reason="",
            )
        )

    def test_missing_resume_rejects_late_or_misbound_terminal_evidence(self) -> None:
        states = [
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.SAVE_FEEDBACK.value,
        ]
        common = {
            "states": states,
            "step_success": [0, 0],
            "step_terminated": [0, 1],
            "step_truncated": [0, 0],
            "terminal_reasons": ["", "object_drop"],
            "summary_success": False,
            "completion_observed": True,
            "failure_reason": "object_drop",
        }
        self.assertFalse(_missing_resume_outcome_matches(**common))

        first_reason_mismatch = {
            **common,
            "states": [
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            ],
            "step_terminated": [1, 1],
            "terminal_reasons": ["grasp_lost", "object_drop"],
        }
        self.assertFalse(
            _missing_resume_outcome_matches(**first_reason_mismatch)
        )
        self.assertTrue(
            _missing_resume_outcome_matches(
                **{
                    **first_reason_mismatch,
                    "failure_reason": "grasp_lost",
                }
            )
        )


class ResponsePhaseDetectorTests(unittest.TestCase):
    def test_onset_tail_recovery_and_stable_resumption_are_separate(self) -> None:
        detector = SafetyResponsePhaseDetector(ResponsePhaseConfig())

        low = detector.update(
            control_step=0, simulation_time_s=0.00,
            cbf_constraint_active=True, intervention_norm_rad_s=0.049,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(low.phase, ResponsePhase.PRE_RESPONSE)
        candidate = detector.update(
            control_step=1, simulation_time_s=0.01,
            cbf_constraint_active=True, intervention_norm_rad_s=0.05,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertTrue(candidate.onset_candidate_now)
        detector.update(
            control_step=2, simulation_time_s=0.02,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        confirmed = detector.update(
            control_step=3, simulation_time_s=0.03,
            cbf_constraint_active=True, intervention_norm_rad_s=0.07,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertTrue(confirmed.onset_confirmed_now)
        self.assertEqual(confirmed.onset_candidate_step, 1)
        self.assertEqual(confirmed.onset_confirmed_step, 3)
        self.assertEqual(confirmed.phase, ResponsePhase.CBF_ACTIVE)

        tail = detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=False, intervention_norm_rad_s=0.03,
            smooth_tail_active=True, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(tail.phase, ResponsePhase.SMOOTH_TAIL)
        recovery = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=False, intervention_norm_rad_s=0.02,
            smooth_tail_active=False, recovery_active=True, bc_resumed=False,
        )
        self.assertEqual(recovery.phase, ResponsePhase.RECOVERY_ACTIVE)
        self.assertEqual(recovery.recovery_onset_step, 5)
        resumed = detector.update(
            control_step=6, simulation_time_s=0.10,
            cbf_constraint_active=False, intervention_norm_rad_s=0.01,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertEqual(resumed.phase, ResponsePhase.BC_RESUMED)
        not_stable = detector.update(
            control_step=7, simulation_time_s=0.59,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertFalse(not_stable.stable_resumption_now)
        stable = detector.update(
            control_step=8, simulation_time_s=0.61,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertTrue(stable.stable_resumption_now)
        self.assertEqual(stable.phase, ResponsePhase.STABLE_TASK_RESUMPTION)

        terminal = detector.update(
            control_step=99, simulation_time_s=2.0,
            cbf_constraint_active=True, intervention_norm_rad_s=0.2,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(terminal.phase, ResponsePhase.STABLE_TASK_RESUMPTION)
        self.assertFalse(terminal.stable_resumption_now)

    def test_interrupted_candidate_requires_three_new_consecutive_frames(self) -> None:
        detector = SafetyResponsePhaseDetector()
        common = dict(
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        first = detector.update(
            control_step=1, simulation_time_s=0.01,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(first.onset_candidate_now)
        detector.update(
            control_step=2, simulation_time_s=0.02,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0, **common,
        )
        second = detector.update(
            control_step=3, simulation_time_s=0.03,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(second.onset_candidate_now)
        self.assertEqual(second.onset_candidate_step, 3)
        detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        third = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(third.onset_confirmed_now)

    def test_recovery_edges_ignore_phase_precedence_and_clamp_to_response(self) -> None:
        detector = SafetyResponsePhaseDetector()
        pre = detector.update(
            control_step=0, simulation_time_s=0.00,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=True, bc_resumed=False,
        )
        self.assertTrue(pre.recovery_onset_now)
        self.assertEqual(pre.recovery_onset_step, -1)

        updates = []
        for step in (1, 2, 3):
            updates.append(detector.update(
                control_step=step, simulation_time_s=step / 100.0,
                cbf_constraint_active=True, intervention_norm_rad_s=0.06,
                smooth_tail_active=False, recovery_active=True,
                bc_resumed=False,
            ))
        confirmed = updates[-1]
        self.assertEqual(confirmed.phase, ResponsePhase.CBF_ACTIVE)
        self.assertEqual(confirmed.recovery_onset_step, 1)
        self.assertEqual(confirmed.recovery_onset_simulation_time_s, 0.01)

        tail = detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=False, intervention_norm_rad_s=0.02,
            smooth_tail_active=True, recovery_active=True, bc_resumed=False,
        )
        self.assertEqual(tail.phase, ResponsePhase.SMOOTH_TAIL)
        ended = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertTrue(ended.recovery_end_now)
        self.assertEqual(ended.recovery_end_step_exclusive, 5)
        self.assertEqual(ended.recovery_end_simulation_time_s, 0.05)


class _ArrayDataset:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, key: object) -> np.ndarray:
        if key != ():
            raise KeyError(key)
        return self.values

    def asstr(self) -> "_ArrayDataset":
        return self


class _ArrayGroup(dict[str, _ArrayDataset]):
    def __init__(self, values: dict[str, object], attrs: dict[str, object]) -> None:
        super().__init__(
            (name, _ArrayDataset(value)) for name, value in values.items()
        )
        self.attrs = attrs


class QueryHoldValidationTests(unittest.TestCase):
    @staticmethod
    def _validator() -> _Validator:
        validator = object.__new__(_Validator)
        validator.issues = []
        return validator

    @staticmethod
    def _group() -> _ArrayGroup:
        states = [
            TrialState.WAIT_FOR_TARGET_TASK_PHASE.value,
            TrialState.SHOW_HAND_CROSSING_CUE.value,
            TrialState.TRACK_SAFETY_RESPONSE_EPISODE.value,
            TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value,
            TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.RESUME_TASK_TO_COMPLETION_OR_TERMINAL.value,
        ]
        count = len(states)
        holds = np.asarray([False, False, False, False, True, True, False])
        return _ArrayGroup(
            {
                "control/trial_state": states,
                "control/control_mode": np.where(
                    holds, "protocol_hold", "nominal_bc"
                ),
                "task/phase_advance_enabled": np.where(holds, 0, 1),
                "actions/controller_task_action": np.zeros((count, 5)),
                "actions/controller_task_action_reason": np.where(
                    holds,
                    "current_ee_protocol_hold",
                    "frozen_bc_policy",
                ),
                "recovery/control_authority": np.zeros(count),
                "actions/action_pipeline_complete": np.ones(count),
                "cbf/intervention_available": np.ones(count),
                "safety/geometry_valid": np.ones(count),
                "recovery/stable_task_resumption": np.asarray(
                    [0, 0, 1, 1, 1, 1, 1]
                ),
            },
            {
                "response_onset_confirmed_step": 1,
                "stable_task_resumption_step": 2,
            },
        )

    def test_query_hold_requires_exact_zero_and_complete_cbf_evidence(self) -> None:
        group = self._group()
        validator = self._validator()
        validator._validate_fsm_rows(group, "/trials/t1")
        self.assertEqual(validator.issues, [])

        for field, bad_value in (
            ("actions/controller_task_action", (4, 0, 0.1)),
            ("recovery/control_authority", (4, 1)),
            ("cbf/intervention_available", (5, 0)),
            ("safety/geometry_valid", (5, 0)),
        ):
            with self.subTest(field=field):
                mutated = self._group()
                index_values = bad_value
                if len(index_values) == 3:
                    row, column, value = index_values
                    mutated[field].values[row, column] = value
                else:
                    row, value = index_values
                    mutated[field].values[row] = value
                validator = self._validator()
                validator._validate_fsm_rows(mutated, "/trials/t1")
                self.assertTrue(
                    any(issue.code == "trial.query_hold" for issue in validator.issues)
                )


class CrossingDirectionAuditTests(unittest.TestCase):
    @staticmethod
    def _monitor() -> CrossingPathMonitor:
        return CrossingPathMonitor(
            start_position_world=(0.0, 0.0, 0.0),
            end_position_world=(1.0, 0.0, 0.0),
            path_deviation_threshold_m=0.05,
        )

    def test_first_completed_traversal_preserves_its_signed_direction(self) -> None:
        forward = self._monitor()
        forward.observe(
            position_world=(0.0, 0.0, 0.0), simulation_time_s=0.0
        )
        forward.observe(
            position_world=(1.0, 0.0, 0.0), simulation_time_s=1.0
        )
        forward_metrics = forward.finalize()
        self.assertEqual(forward_metrics.first_completed_traversal_sign, 1)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "left_to_right",
                forward_metrics.first_completed_traversal_sign,
            ),
            "left_to_right",
        )

        reverse = self._monitor()
        reverse.observe(
            position_world=(1.0, 0.0, 0.0), simulation_time_s=0.0
        )
        reverse.observe(
            position_world=(0.0, 0.0, 0.0), simulation_time_s=1.0
        )
        reverse_metrics = reverse.finalize()
        self.assertEqual(reverse_metrics.first_completed_traversal_sign, -1)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "left_to_right",
                reverse_metrics.first_completed_traversal_sign,
            ),
            "right_to_left",
        )

    def test_incomplete_traversal_uses_explicit_not_observed_sentinel(self) -> None:
        incomplete = self._monitor().finalize()
        self.assertEqual(incomplete.first_completed_traversal_sign, 0)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "right_to_left", incomplete.first_completed_traversal_sign
            ),
            "not_observed",
        )
        self.assertIn("not_observed", ACTUAL_CROSSING_DIRECTIONS)


class FrozenConfigTests(unittest.TestCase):
    def test_default_config_matches_frozen_ac_and_feedback_contract(self) -> None:
        config = load_config()
        validate_config(config)
        self.assertEqual(
            config["study"]["available_modes"],
            {
                "minimal_pilot": {
                    "evaluated_trials": 16, "anchor_repeat_trials": 0,
                },
                "pilot_with_anchors": {
                    "evaluated_trials": 20, "anchor_repeat_trials": 4,
                },
            },
        )
        self.assertEqual(
            config["conditions"],
            {
                "A_reactive": {"objective_mode": "joint_nominal", "lambda_s": 0.0},
                "C_smooth": {"objective_mode": "smooth_intervention", "lambda_s": 4.0},
            },
        )
        self.assertFalse(config["feedback"]["haptics_enabled"])
        self.assertFalse(config["study"]["feedback_changes_schedule"])
        self.assertFalse(config["policy"]["online_updates"])
        self.assertFalse(config["shared_cbf"]["parameter_adaptation"])

    def test_haptics_condition_and_shared_safety_drift_are_rejected(self) -> None:
        for path, value, message in (
            (("feedback", "haptics_enabled"), True, "haptics_enabled"),
            (("conditions", "C_smooth", "lambda_s"), 2.0, "lambda_s"),
            (("shared_cbf", "safe_gap_m"), 0.04, "safe_gap_m"),
        ):
            with self.subTest(path=path):
                config = deepcopy(load_config())
                cursor = config
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

    def test_controller_mapping_key_and_value_drift_is_rejected(self) -> None:
        mutations = (
            (
                "missing realtime marker",
                lambda feedback: feedback["realtime_markers"]["crossing_left"].pop(
                    "realtime_safety_concern"
                ),
                "realtime_markers",
            ),
            (
                "swapped realtime marker",
                lambda feedback: feedback["realtime_markers"]["crossing_right"].update(
                    realtime_safety_concern="left.y"
                ),
                "realtime_markers",
            ),
            (
                "missing back chord",
                lambda feedback: feedback["questionnaire_navigation"].pop(
                    "back_chord"
                ),
                "questionnaire_navigation",
            ),
            (
                "changed navigation button",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    previous="left.y"
                ),
                "questionnaire_navigation",
            ),
            (
                "changed emergency chord",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_chord=["right.b", "right.a"]
                ),
                "questionnaire_navigation",
            ),
            (
                "changed emergency hold",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_hold_s=1.5
                ),
                "questionnaire_navigation",
            ),
            (
                "extra navigation key",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    fallback="right.a"
                ),
                "questionnaire_navigation",
            ),
        )
        for name, mutate, expected_path in mutations:
            with self.subTest(name=name):
                config = deepcopy(load_config())
                mutate(config["feedback"])
                with self.assertRaisesRegex(ValueError, expected_path):
                    validate_config(config)

    def test_static_semantics_rejects_mapping_and_emergency_drift(self) -> None:
        for field, mutate, expected_code in (
            (
                "realtime_markers",
                lambda feedback: feedback["realtime_markers"]["crossing_left"].update(
                    realtime_safety_concern="right.b"
                ),
                "config.marker_mapping",
            ),
            (
                "questionnaire_navigation",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_chord=["left.x", "left.y"]
                ),
                "config.questionnaire_navigation",
            ),
        ):
            with self.subTest(field=field):
                config = deepcopy(load_config())
                mutate(config["feedback"])
                validator = _Validator(
                    config_path=DEFAULT_CONFIG,
                    collection_path=None,
                    participant_id=None,
                    session_id=None,
                    seed=None,
                    mode=None,
                    allow_partial_path=False,
                )
                validator.config = config
                validator._validate_static_config_semantics()
                self.assertIn(expected_code, {issue.code for issue in validator.issues})

    def test_decision_context_rejects_missing_or_future_outcomes(self) -> None:
        context = {
            "task_phase": "transport",
            "grasp_state": "grasped",
            "attachment_state": "attached",
            "robot_joint_state": {},
            "ee_pose_velocity": {},
            "cube_pose": {},
            "goal_pose": {},
            "hand_pose_velocity": {},
            "surface_gap_m": 0.08,
            "ttc_s": 0.5,
            "closing_speed_m_s": 0.1,
            "active_candidate": {},
            "nominal_rmpflow_joint_command": [0.0] * 7,
            "bc_raw_action": [0.0] * 5,
            "condition_id": "A_reactive",
            "lambda_s": 0.0,
        }
        validate_decision_context(context)
        with self.assertRaisesRegex(ValueError, "future outcomes"):
            validate_decision_context({**context, "task_success": True})
        missing = dict(context)
        missing.pop("bc_raw_action")
        with self.assertRaisesRegex(ValueError, "bc_raw_action"):
            validate_decision_context(missing)


class ControllerFeedbackTests(unittest.TestCase):
    def test_controller_input_watchdog_distinguishes_health_and_dropout(self) -> None:
        watchdog = ControllerInputWatchdog()
        healthy = ControllerButtonSnapshot(
            left_connected=True,
            right_connected=True,
            left_x=False,
            left_y=True,
            right_a=False,
            right_b=True,
        )
        self.assertIs(watchdog.require_available(healthy), healthy)
        self.assertEqual(watchdog.unavailable_reasons(healthy), ())

        disconnected = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=True,
            right_a=False,
            right_b=False,
        )
        with self.assertRaises(ControllerInputUnavailable) as caught:
            watchdog.require_available(disconnected)
        self.assertEqual(
            caught.exception.reasons, ("left_controller_disconnected",)
        )

        unknown_button = ControllerButtonSnapshot(
            left_connected=True,
            right_connected=True,
            left_x=False,
            left_y=False,
            right_a=None,
            right_b=False,
        )
        with self.assertRaises(ControllerInputUnavailable) as caught:
            watchdog.require_available(unknown_button)
        self.assertEqual(caught.exception.reasons, ("right.a_state_unknown",))

    def test_controller_input_dropout_applies_verified_hold_before_abort(self) -> None:
        class Source:
            last_error = "right A unavailable"

            @staticmethod
            def read() -> ControllerButtonSnapshot:
                return ControllerButtonSnapshot(
                    left_connected=True,
                    right_connected=True,
                    left_x=False,
                    left_y=False,
                    right_a=None,
                    right_b=False,
                )

        robot = object()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback.apply_controlled_hold"
        ) as hold:
            with self.assertRaisesRegex(
                CollectionAbort, "controller_input_unavailable_pre_step"
            ):
                _read_required_controller_inputs(
                    source=Source(),
                    input_watchdog=ControllerInputWatchdog(),
                    robot=robot,
                    joint_count=9,
                    sample_phase="pre_step",
                )
        hold.assert_called_once_with(robot, joint_count=9)

    def test_realtime_mapping_uses_opposite_controller_and_separate_labels(self) -> None:
        config = load_config()["feedback"]["realtime_markers"]
        mapping = RealtimeButtonMapping.from_mapping({
            "left": config["crossing_left"],
            "right": config["crossing_right"],
        })
        self.assertEqual(
            tuple(button.key for button in mapping.for_crossing("left")),
            ("right.a", "right.b"),
        )
        self.assertEqual(
            tuple(button.key for button in mapping.for_crossing("right")),
            ("left.x", "left.y"),
        )
        marker_input = RealtimeMarkerInput(
            mapping=mapping, crossing_hand="left", id_factory=lambda: "marker-id",
        )
        records = marker_input.records_for_edges(
            ("right.a", "right.b"), _clock(1)
        )
        self.assertEqual(
            [record.marker_type for record in records],
            [REALTIME_SAFETY_CONCERN, REALTIME_BEHAVIOR_ANOMALY],
        )
        self.assertTrue(all(record.controller_hand == "right" for record in records))

    def test_realtime_mapping_does_not_fill_missing_bindings(self) -> None:
        config = load_config()["feedback"]["realtime_markers"]
        mapping = {
            "left": dict(config["crossing_left"]),
            "right": dict(config["crossing_right"]),
        }
        mapping["left"].pop(REALTIME_SAFETY_CONCERN)
        with self.assertRaisesRegex(ValueError, "exactly"):
            RealtimeButtonMapping.from_mapping(mapping)

    def test_configured_questionnaire_mapping_and_vocabulary_are_exact(self) -> None:
        feedback = load_config()["feedback"]
        mapping = QuestionnaireButtonMapping.from_mapping(
            feedback["questionnaire_navigation"]
        )
        self.assertEqual(mapping.previous.key, "left.x")
        self.assertEqual(mapping.next.key, "left.y")
        self.assertEqual(mapping.select.key, "right.a")
        self.assertEqual(mapping.submit.key, "right.b")
        self.assertEqual(mapping.back_keys, frozenset(("left.x", "left.y")))
        self.assertEqual(tuple(feedback["q1_responses"]), Q1_RESPONSE_CODES)
        self.assertEqual(tuple(feedback["modification_reasons"]), REJECTION_REASON_IDS)
        self.assertEqual(tuple(feedback["questions"])[1:], tuple(LIKERT_QUESTIONS_KO))

    def test_questionnaire_mapping_does_not_default_back_chord(self) -> None:
        navigation = dict(load_config()["feedback"]["questionnaire_navigation"])
        navigation.pop("back_chord")
        with self.assertRaisesRegex(ValueError, "back_chord"):
            QuestionnaireButtonMapping.from_mapping(navigation)

    def test_needs_modification_requires_at_least_one_reason(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0, id_factory=lambda: "query-id")
        questionnaire.start(_clock(0), context_id="trial-id")
        for index in range(6):
            questionnaire.handle_edges(("right.a",), _clock(1 + index * 2))
            questionnaire.handle_edges(("right.b",), _clock(2 + index * 2))
        self.assertEqual(questionnaire.state, "rejection_reasons")
        self.assertIsNone(questionnaire.handle_edges(("right.b",), _clock(13)))
        self.assertTrue(questionnaire.active)
        questionnaire.handle_edges(("right.a",), _clock(14))
        result = questionnaire.handle_edges(("right.b",), _clock(15))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "completed")
        self.assertEqual(result.response_disposition, "answered")
        self.assertEqual(result.q1_response, "needs_modification")
        self.assertEqual(result.rejection_reason_ids, ("too_close_or_late",))
        self.assertEqual(result.first_input_control_step, 1)
        self.assertEqual(len(result.answers), 7)
        self.assertTrue(
            all(answer.answer_status == "confirmed" for answer in result.answers)
        )
        reason_answer = result.answers[-1]
        self.assertEqual(reason_answer.question_id, "rejection_reasons")
        self.assertEqual(reason_answer.answer_status, "confirmed")
        self.assertEqual(reason_answer.prompt_shown_control_step, 12)
        self.assertEqual(reason_answer.first_input_control_step, 13)
        self.assertEqual(reason_answer.confirmed_control_step, 15)
        self.assertEqual(reason_answer.response_latency_ms, 300.0)
        self.assertEqual(reason_answer.input_device, "vr_controller")
        self.assertEqual(reason_answer.accidental_input_count, 1)

    def test_back_chord_is_audited_and_replaces_later_answers(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("right.a",), _clock(1))
        questionnaire.handle_edges(("right.b",), _clock(2))  # q1 -> q2
        questionnaire.handle_edges(("left.x", "left.y"), _clock(3))  # q2 -> q1
        self.assertEqual(questionnaire.state, "q1")
        questionnaire.handle_edges(("left.y",), _clock(4))  # cursor: acceptable
        questionnaire.handle_edges(("right.a",), _clock(5))  # select
        questionnaire.handle_edges(("right.b",), _clock(6))  # confirm
        result = None
        for index in range(5):
            questionnaire.handle_edges(("right.a",), _clock(7 + index * 2))
            result = questionnaire.handle_edges(
                ("right.b",), _clock(8 + index * 2)
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.q1_response, "acceptable_as_is")
        self.assertEqual(result.rejection_reason_ids, ())
        self.assertEqual(result.back_correction_count, 1)
        self.assertEqual(len(result.answers), 6)
        answers = {answer.question_id: answer for answer in result.answers}
        corrected_q1 = answers["q1_modification_needed"]
        self.assertEqual(corrected_q1.prompt_shown_control_step, 3)
        self.assertEqual(corrected_q1.first_input_control_step, 4)
        self.assertEqual(corrected_q1.confirmed_control_step, 6)
        self.assertEqual(corrected_q1.response_latency_ms, 300.0)
        self.assertEqual(corrected_q1.back_correction_count, 1)
        self.assertEqual(answers["q2"].prompt_shown_control_step, 6)

    def test_main_questions_require_select_then_confirm(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        self.assertIsNone(
            questionnaire.handle_edges(("right.b",), _clock(1))
        )
        self.assertEqual(questionnaire.state, "q1")
        self.assertIsNone(
            questionnaire.handle_edges(("right.a",), _clock(2))
        )
        self.assertEqual(questionnaire.state, "q1")
        questionnaire.handle_edges(("right.b",), _clock(3))
        self.assertEqual(questionnaire.state, "q2")

        q1 = questionnaire._answers["q1_modification_needed"]
        self.assertEqual(q1.prompt_shown_control_step, 0)
        self.assertEqual(q1.first_input_control_step, 1)
        self.assertEqual(q1.confirmed_control_step, 3)
        self.assertEqual(q1.response_latency_ms, 300.0)
        self.assertEqual(q1.input_device, "vr_controller")
        self.assertEqual(q1.back_correction_count, 0)
        self.assertEqual(q1.accidental_input_count, 1)

    def test_timeout_remains_missing_abstain(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=1.0)
        questionnaire.start(_clock(0))
        result = questionnaire.handle_edges((), _clock(10))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "timed_out")
        self.assertEqual(result.response_disposition, "missing_abstain")
        self.assertIsNone(result.q1_response)
        self.assertIn("q1_modification_needed", result.missing_question_ids)
        self.assertEqual(len(result.answers), 1)
        missing = result.answers[0]
        self.assertEqual(missing.question_id, "q1_modification_needed")
        self.assertEqual(missing.answer_status, "unconfirmed")
        self.assertIsNone(missing.value)
        self.assertEqual(missing.prompt_shown_control_step, 0)
        self.assertEqual(missing.first_input_control_step, -1)
        self.assertEqual(missing.confirmed_control_step, -1)
        self.assertEqual(missing.response_latency_ms, -1.0)

    def test_timeout_preserves_current_prompt_and_partial_first_input(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=1.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("left.y",), _clock(1))
        result = questionnaire.handle_edges((), _clock(10))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.response_disposition, "missing_abstain")
        self.assertEqual(len(result.answers), 1)
        partial = result.answers[0]
        self.assertEqual(partial.answer_status, "unconfirmed")
        self.assertEqual(partial.prompt_shown_control_step, 0)
        self.assertEqual(partial.first_input_control_step, 1)
        self.assertEqual(partial.confirmed_control_step, -1)
        self.assertIsNone(partial.value)

    def test_uncertain_completion_remains_an_explicit_abstain(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("left.y",), _clock(1))
        questionnaire.handle_edges(("left.y",), _clock(2))
        questionnaire.handle_edges(("right.a",), _clock(3))
        questionnaire.handle_edges(("right.b",), _clock(4))
        result = None
        for index in range(5):
            questionnaire.handle_edges(("right.a",), _clock(5 + index * 2))
            result = questionnaire.handle_edges(
                ("right.b",), _clock(6 + index * 2)
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "completed")
        self.assertEqual(result.q1_response, "uncertain")
        self.assertEqual(result.response_disposition, "uncertain_abstain")

    def test_per_question_audit_validator_rejects_clock_latency_drift(self) -> None:
        query = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "response_status": "timeout",
            "issued_simulation_time_s": 0.0,
            "issued_monotonic_ns": 1_000_000_000,
            "issued_unix_ns": 2_000_000_000,
            "issued_control_step": 0,
            "completed_simulation_time_s": 0.5,
            "completed_monotonic_ns": 1_500_000_000,
            "completed_unix_ns": 2_500_000_000,
            "completed_control_step": 5,
            "back_correction_count": 1,
            "accidental_input_count": 1,
        }
        answer = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "question_id": ONLINE_Q1_ID,
            "answer_status": "confirmed",
            "value_json": '"uncertain"',
            "prompt_shown_simulation_time_s": 0.1,
            "prompt_shown_monotonic_ns": 1_100_000_000,
            "prompt_shown_unix_ns": 2_100_000_000,
            "prompt_shown_control_step": 1,
            "first_input_simulation_time_s": 0.2,
            "first_input_monotonic_ns": 1_200_000_000,
            "first_input_unix_ns": 2_200_000_000,
            "first_input_control_step": 2,
            "confirmed_simulation_time_s": 0.4,
            "confirmed_monotonic_ns": 1_400_000_000,
            "confirmed_unix_ns": 2_400_000_000,
            "confirmed_control_step": 4,
            "response_latency_ms": 300.0,
            "input_device": "vr_controller",
            "back_correction_count": 1,
            "accidental_input_count": 1,
        }
        partial = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "question_id": "q2_perceived_danger",
            "answer_status": "unconfirmed",
            "value_json": "null",
            "prompt_shown_simulation_time_s": 0.4,
            "prompt_shown_monotonic_ns": 1_400_000_000,
            "prompt_shown_unix_ns": 2_400_000_000,
            "prompt_shown_control_step": 4,
            "first_input_simulation_time_s": -1.0,
            "first_input_monotonic_ns": 0,
            "first_input_unix_ns": 0,
            "first_input_control_step": -1,
            "confirmed_simulation_time_s": -1.0,
            "confirmed_monotonic_ns": 0,
            "confirmed_unix_ns": 0,
            "confirmed_control_step": -1,
            "response_latency_ms": -1.0,
            "input_device": "vr_controller",
            "back_correction_count": 0,
            "accidental_input_count": 0,
        }

        def validator_for_audit() -> _Validator:
            validator = _Validator(
                config_path=DEFAULT_CONFIG,
                collection_path=None,
                participant_id=None,
                session_id=None,
                seed=None,
                mode=None,
                allow_partial_path=False,
            )
            validator.file = {"trials": None}
            return validator

        valid = validator_for_audit()
        valid._validate_query_answer_audit([query], [answer, partial])
        self.assertEqual(valid.issues, [])

        invalid = validator_for_audit()
        invalid._validate_query_answer_audit(
            [query], [{**answer, "response_latency_ms": 301.0}, partial]
        )
        self.assertIn(
            "feedback.answer_latency", {issue.code for issue in invalid.issues}
        )

        missing_partial = validator_for_audit()
        missing_partial._validate_query_answer_audit([query], [answer])
        self.assertIn(
            "feedback.answer_cardinality",
            {issue.code for issue in missing_partial.issues},
        )

        fabricated_partial = validator_for_audit()
        fabricated_partial._validate_query_answer_audit(
            [query], [answer, {**partial, "value_json": '"provisional"'}]
        )
        self.assertIn(
            "feedback.answer_value",
            {issue.code for issue in fabricated_partial.issues},
        )

    def test_emergency_abort_requires_two_button_hold_and_resets_on_disconnect(self) -> None:
        pressed = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=True,
            right_a=True,
            right_b=True,
        )
        disconnected = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=False,
        )
        monitor = EmergencyAbortMonitor(
            buttons=("right.a", "right.b"), hold_s=2.0
        )
        self.assertFalse(monitor.update(pressed, monotonic_ns=0))
        self.assertFalse(monitor.update(pressed, monotonic_ns=1_999_999_999))
        self.assertFalse(monitor.update(disconnected, monotonic_ns=2_000_000_000))
        self.assertFalse(monitor.update(pressed, monotonic_ns=3_000_000_000))
        self.assertTrue(monitor.update(pressed, monotonic_ns=5_000_000_000))
        self.assertFalse(monitor.update(pressed, monotonic_ns=6_000_000_000))


if __name__ == "__main__":
    unittest.main()
