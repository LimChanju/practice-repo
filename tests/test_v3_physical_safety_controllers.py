import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from v3_chan.end_effector_safety_geometry import (
    EndEffectorSafetyResult,
    HandSafetyResult,
)
from v3_chan.physical_safety_controllers import (
    CBFConfig,
    DistalLinkVelocityCBF,
    project_velocity_qp,
)


def test_recorded_hand_tracking_requirement_is_opt_in_and_boolean():
    assert CBFConfig().require_valid_recorded_hand_tracking is False
    with pytest.raises(ValueError, match="must be boolean"):
        CBFConfig(require_valid_recorded_hand_tracking=1).validated()


def test_velocity_projection_satisfies_halfspace_and_joint_limits():
    projected, slack, feasible, before, after = project_velocity_qp(
        nominal_velocity=np.array([-1.0, 0.5]),
        constraint_matrix=np.array([[1.0, 0.0]]),
        lower_bounds=np.array([0.25]),
        velocity_lower=np.array([-2.0, -0.2]),
        velocity_upper=np.array([2.0, 0.2]),
    )

    np.testing.assert_allclose(projected, [0.25, 0.2], atol=1e-5)
    assert feasible
    assert np.isclose(slack, 0.0)
    assert before > 1.0
    assert after <= 1e-5


def test_velocity_projection_exposes_infeasible_constraints_with_slack():
    projected, slack, feasible, before, after = project_velocity_qp(
        nominal_velocity=np.array([0.0]),
        constraint_matrix=np.array([[1.0], [-1.0]]),
        lower_bounds=np.array([2.0, 2.0]),
        velocity_lower=np.array([-1.0]),
        velocity_upper=np.array([1.0]),
    )

    assert not feasible
    assert slack >= 1.99
    assert before >= 1.99
    assert after >= 1.99
    assert -1.0 <= projected[0] <= 1.0


def test_velocity_projection_does_not_false_stop_near_opposed_feasible_rows(
    monkeypatch,
):
    matrix = np.array(
        [
            [
                -0.13454057944982706,
                -0.3110663294493656,
                -0.286102766091242,
                -0.06758864778817839,
                0.8776393990485126,
                -0.14145906630524951,
                0.09193460114190675,
            ],
            [
                0.13377538022522695,
                0.3131102935811261,
                0.30464655718155265,
                0.06570133020854732,
                -0.8689556071556929,
                0.14784920228134876,
                -0.09998334002643329,
            ],
        ]
    )
    bounds = np.array([-0.6697118393717469, 0.6646923166264243])
    nominal = np.array(
        [
            -0.11564884740169701,
            0.536083934655756,
            -1.8727177198773508,
            1.1585578909728174,
            -0.6749370087669995,
            -0.9087573516217229,
            0.3118461306030129,
        ]
    )
    real_import = builtins.__import__

    def import_without_scipy_optimize(name, *args, **kwargs):
        if name == "scipy.optimize":
            raise ImportError("exercise deterministic dependency-free retry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy_optimize)

    projected, slack, feasible, _, after = project_velocity_qp(
        nominal_velocity=nominal,
        constraint_matrix=matrix,
        lower_bounds=bounds,
        velocity_lower=np.full(7, -2.0),
        velocity_upper=np.full(7, 2.0),
        max_iterations=80,
        tolerance=1e-5,
    )

    assert feasible
    assert slack == 0.0
    assert after <= 1e-5
    assert np.all(matrix @ projected >= bounds - 1e-5)


class _FakeArticulationView:
    body_names = ("panda_link0", "panda_hand")

    def get_jacobians(self):
        jacobian = np.zeros((1, 1, 6, 1), dtype=float)
        jacobian[0, 0, 0, 0] = 1.0
        return jacobian


class _FakeRobot:
    def __init__(self):
        self._articulation_view = _FakeArticulationView()
        self.dof_properties = {
            "maxVelocity": np.array([2.0]),
            "lower": np.array([-3.0]),
            "upper": np.array([3.0]),
        }

    def get_joint_positions(self):
        return np.array([0.0])


class _FakeSafetyGeometry:
    def closest_link_world_pose(self, hand_result):
        assert hand_result.closest_link == "panda_hand"
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), True


def _dynamic_hand(
    *, velocity=None, closing_speed=0.0, valid=True, closing_speed_valid=None
):
    return SimpleNamespace(
        hand_velocity_filtered_mps=np.zeros(3) if velocity is None else velocity,
        hand_velocity_valid=bool(valid),
        closing_speed_mps=closing_speed,
        closing_speed_valid=(
            bool(valid) if closing_speed_valid is None else bool(closing_speed_valid)
        ),
    )


def _filter_close_left_hand(
    *,
    arm_action=None,
    cbf=None,
    dynamic_left=None,
    human_valid_mask=None,
):
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = (
        SimpleNamespace(
            joint_indices=np.array([0]),
            joint_positions=np.array([-0.01]),
            joint_velocities=None,
        )
        if arm_action is None
        else arm_action
    )
    return (cbf or DistalLinkVelocityCBF()).filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand() if dynamic_left is None else dynamic_left,
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
        human_valid_mask=human_valid_mask,
    )


def _filter_intentionally_absent_human(
    *,
    arm_action=None,
    intentional_human_absence=True,
    human_valid_mask=None,
    observation=None,
    safety_result=None,
    fail_closed_on_invalid_active_hand=True,
):
    action = (
        SimpleNamespace(
            joint_indices=np.array([0]),
            joint_positions=np.array([-0.01]),
            joint_velocities=None,
        )
        if arm_action is None
        else arm_action
    )
    absent_safety = safety_result or EndEffectorSafetyResult(
        left=HandSafetyResult(hand="left", geometry_valid=False),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    absent_observation = observation or {
        "human_head_pos": np.zeros(3),
        "human_left_hand_pos": np.zeros(3),
        "human_right_hand_pos": np.zeros(3),
    }
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            require_valid_recorded_hand_tracking=True,
            fail_closed_on_invalid_active_hand=fail_closed_on_invalid_active_hand,
        )
    )
    return cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=absent_safety,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(valid=False),
            right=_dynamic_hand(valid=False),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation=absent_observation,
        physics_dt_s=0.1,
        human_valid_mask=(
            np.zeros(3, dtype=np.float32)
            if human_valid_mask is None
            else human_valid_mask
        ),
        intentional_human_absence=intentional_human_absence,
    )


def test_explicit_intentional_human_absence_keeps_nominal_action_and_cbf_inactive():
    original_joint_indices = np.array([0])
    original_joint_positions = np.array([-0.01])
    action = SimpleNamespace(
        joint_indices=original_joint_indices,
        joint_positions=original_joint_positions,
        joint_velocities=None,
    )
    filtered, diagnostics = _filter_intentionally_absent_human(arm_action=action)

    assert filtered is action
    assert filtered.joint_indices is original_joint_indices
    assert filtered.joint_positions is original_joint_positions
    assert filtered.joint_velocities is None
    np.testing.assert_array_equal(filtered.joint_positions, [-0.01])
    assert diagnostics.intentional_human_absence is True
    assert diagnostics.status == "inactive"
    assert diagnostics.active is False
    assert diagnostics.constraint_count == 0
    assert diagnostics.intervention_norm_radps == pytest.approx(0.0)
    assert diagnostics.feasible is True
    assert diagnostics.fallback_applied is False
    assert diagnostics.failure_reasons == ()


def test_all_zero_recorded_mask_without_absence_marker_still_fail_stops():
    filtered, diagnostics = _filter_intentionally_absent_human(
        intentional_human_absence=False,
        fail_closed_on_invalid_active_hand=False,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.intentional_human_absence is False
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert diagnostics.fallback_applied is True
    assert "left:recorded_tracking_mask_false" in diagnostics.failure_reasons
    assert "right:recorded_tracking_mask_false" in diagnostics.failure_reasons


@pytest.mark.parametrize(
    ("human_valid_mask", "observation", "safety_result", "expected_reason"),
    (
        (
            np.array([False, True, False]),
            None,
            None,
            "intentional_absence:tracking_mask_not_all_zero",
        ),
        (
            np.zeros(3),
            {
                "human_head_pos": np.zeros(3),
                "human_left_hand_pos": np.array([0.2, 0.0, 0.0]),
                "human_right_hand_pos": np.zeros(3),
            },
            None,
            "intentional_absence:human_left_hand_pos_present",
        ),
        (
            np.zeros(3),
            None,
            EndEffectorSafetyResult(
                left=HandSafetyResult(
                    hand="left",
                    geometry_valid=True,
                    surface_gap_m=0.2,
                    closest_link="panda_hand",
                ),
                right=HandSafetyResult(hand="right", geometry_valid=False),
            ),
            "intentional_absence:left_geometry_present",
        ),
    ),
)
def test_intentional_absence_marker_cannot_bypass_inconsistent_human_evidence(
    human_valid_mask,
    observation,
    safety_result,
    expected_reason,
):
    filtered, diagnostics = _filter_intentionally_absent_human(
        human_valid_mask=human_valid_mask,
        observation=observation,
        safety_result=safety_result,
        fail_closed_on_invalid_active_hand=False,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert diagnostics.fallback_applied is True
    assert expected_reason in diagnostics.failure_reasons


def test_intentional_absence_marker_must_be_exact_boolean():
    with pytest.raises(RuntimeError, match="must be an exact boolean"):
        _filter_intentionally_absent_human(intentional_human_absence="true")


def test_cbf_changes_nominal_action_away_from_close_hand():
    left = HandSafetyResult(
        hand="left",
        geometry_valid=True,
        surface_gap_m=0.02,
        closest_link="panda_hand",
        closest_surface_point_world_pos=(0.0, 0.0, 0.0),
        closest_surface_point_valid=True,
        near=True,
        distance_gate=1.0,
    )
    safety_result = EndEffectorSafetyResult(
        left=left,
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    dynamic_sample = SimpleNamespace(
        left=_dynamic_hand(),
        right=_dynamic_hand(),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            safe_gap_m=0.05,
            activation_gap_m=0.13,
            gamma_per_s=8.0,
            prediction_horizon_s=0.0,
        )
    )

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=dynamic_sample,
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    # h = 0.02 - 0.05, so qdot >= -gamma*h = 0.24 rad/s.
    assert filtered.joint_velocities[0] >= 0.24 - 1e-5
    assert filtered.joint_positions[0] > 0.0
    assert diagnostics.active
    assert diagnostics.feasible
    assert diagnostics.constraint_count == 1
    assert diagnostics.tracked_hand_count == 1
    assert diagnostics.valid_hand_count == 1
    assert diagnostics.intervention_norm_radps > 0.3
    assert diagnostics.max_constraint_violation_after <= 1e-5
    assert not diagnostics.fallback_applied
    assert diagnostics.solver_converged
    assert not diagnostics.infeasibility_proven
    assert not diagnostics.relaxed_solution_available
    assert not diagnostics.relaxed_solution_applied
    payload = diagnostics.as_dict()
    assert payload["max_constraint_violation_before_mps"] == (
        diagnostics.max_constraint_violation_before
    )
    assert payload["slack_mps"] == diagnostics.slack_radps


def test_active_cbf_preserves_exact_action_when_nominal_is_already_feasible():
    original_joint_indices = np.array([0])
    original_joint_positions = np.array([0.03])
    action = SimpleNamespace(
        joint_indices=original_joint_indices,
        joint_positions=original_joint_positions,
        joint_velocities=None,
    )

    filtered, diagnostics = _filter_close_left_hand(arm_action=action)

    assert filtered is action
    assert filtered.joint_indices is original_joint_indices
    assert filtered.joint_positions is original_joint_positions
    assert filtered.joint_velocities is None
    np.testing.assert_array_equal(filtered.joint_positions, [0.03])
    assert diagnostics.active
    assert diagnostics.constraint_count == 1
    assert diagnostics.intervention_available
    assert diagnostics.intervention_norm_radps == 0.0
    assert diagnostics.nominal_velocity_norm_radps == pytest.approx(0.3)
    assert diagnostics.filtered_velocity_norm_radps == pytest.approx(0.3)
    assert diagnostics.feasible
    assert diagnostics.solver_converged
    assert not diagnostics.fallback_applied
    assert diagnostics.failure_reasons == ()
    assert diagnostics.status == "active_nominal_feasible_passthrough"


def test_cbf_stays_inactive_when_hand_is_outside_activation_gap():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.20,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    original_joint_indices = np.array([0])
    original_joint_positions = np.array([19.0])
    original_joint_velocities = np.array([37.0])
    action = SimpleNamespace(
        joint_indices=original_joint_indices,
        joint_positions=original_joint_positions,
        joint_velocities=original_joint_velocities,
    )
    cbf = DistalLinkVelocityCBF()

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.25, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    assert filtered is action
    assert filtered.joint_indices is original_joint_indices
    assert filtered.joint_positions is original_joint_positions
    assert filtered.joint_velocities is original_joint_velocities
    np.testing.assert_array_equal(filtered.joint_positions, [19.0])
    np.testing.assert_array_equal(filtered.joint_velocities, [37.0])
    assert not diagnostics.active
    assert diagnostics.intervention_available
    assert diagnostics.constraint_count == 0
    assert diagnostics.intervention_norm_radps == 0.0
    assert diagnostics.nominal_velocity_norm_radps == 37.0
    assert diagnostics.filtered_velocity_norm_radps == 37.0
    assert diagnostics.feasible
    assert not diagnostics.fallback_applied
    assert diagnostics.failure_reasons == ()
    assert diagnostics.status == "inactive"
    assert diagnostics.projection_status == "inactive"
    assert diagnostics.solver_backend == "none"


def test_cbf_fail_closes_with_stop_when_active_hand_dynamics_are_invalid():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    cbf = DistalLinkVelocityCBF()

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(valid=False),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert "left:hand_velocity_invalid" in diagnostics.failure_reasons


@pytest.mark.parametrize(
    "velocity",
    (
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0]),
    ),
)
def test_cbf_fail_closes_when_valid_hand_velocity_has_invalid_payload(velocity):
    filtered, diagnostics = _filter_close_left_hand(
        dynamic_left=_dynamic_hand(velocity=velocity, valid=True)
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert "left:hand_velocity_payload_invalid" in diagnostics.failure_reasons


@pytest.mark.parametrize(
    "closing_speed",
    (np.nan, np.inf, -np.inf, np.array([0.1])),
)
def test_cbf_fail_closes_when_valid_closing_speed_has_invalid_payload(
    closing_speed,
):
    filtered, diagnostics = _filter_close_left_hand(
        dynamic_left=_dynamic_hand(
            closing_speed=closing_speed,
            valid=True,
            closing_speed_valid=True,
        )
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert "left:closing_speed_payload_invalid" in diagnostics.failure_reasons


def test_cbf_accepts_finite_negative_closing_speed_as_separating_motion():
    filtered, diagnostics = _filter_close_left_hand(
        dynamic_left=_dynamic_hand(
            closing_speed=-0.2,
            valid=True,
            closing_speed_valid=True,
        )
    )

    assert filtered.joint_velocities[0] >= 0.24 - 1e-5
    assert diagnostics.feasible
    assert not diagnostics.fallback_applied
    assert diagnostics.failure_reasons == ()


@pytest.mark.parametrize(
    ("human_valid_mask", "expected_reason"),
    (
        (None, "left:recorded_tracking_mask_missing"),
        (np.array([True, True]), "left:recorded_tracking_mask_invalid"),
        (np.array([True, np.nan, True]), "left:recorded_tracking_mask_invalid"),
        (np.array([True, True, False]), "right:recorded_tracking_mask_false"),
    ),
)
def test_required_recorded_hand_tracking_mask_fail_closes(
    human_valid_mask,
    expected_reason,
):
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            require_valid_recorded_hand_tracking=True,
            fail_closed_on_invalid_active_hand=False,
        )
    )

    filtered, diagnostics = _filter_close_left_hand(
        cbf=cbf,
        human_valid_mask=human_valid_mask,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert expected_reason in diagnostics.failure_reasons


def test_required_recorded_hand_tracking_accepts_head_left_right_valid_mask():
    cbf = DistalLinkVelocityCBF(CBFConfig(require_valid_recorded_hand_tracking=True))

    filtered, diagnostics = _filter_close_left_hand(
        cbf=cbf,
        human_valid_mask=np.array([False, True, True]),
    )

    assert filtered.joint_velocities[0] >= 0.24 - 1e-5
    assert diagnostics.feasible
    assert not diagnostics.fallback_applied
    assert diagnostics.failure_reasons == ()


def test_cbf_diagnostics_distinguish_applied_relaxation_from_solver_failure():
    cbf = DistalLinkVelocityCBF(CBFConfig(stop_on_infeasible=False))

    filtered, diagnostics = _filter_close_left_hand(
        cbf=cbf,
        dynamic_left=_dynamic_hand(velocity=np.array([5.0, 0.0, 0.0])),
    )

    assert np.isclose(filtered.joint_velocities[0], 2.0)
    assert not diagnostics.feasible
    assert diagnostics.solver_converged
    assert diagnostics.infeasibility_proven
    assert diagnostics.relaxed_solution_available
    assert diagnostics.relaxed_solution_applied
    assert not diagnostics.fallback_applied
    assert diagnostics.status == "solved_with_slack"
    assert diagnostics.projection_status == "infeasible_proven_relaxed"


def test_cbf_does_not_stop_for_closing_speed_only_collider_switch_gap():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )

    filtered, diagnostics = DistalLinkVelocityCBF().filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(valid=True, closing_speed_valid=False),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    assert filtered.joint_velocities[0] >= 0.24 - 1e-5
    assert diagnostics.feasible
    assert not diagnostics.fallback_applied
    assert diagnostics.failure_reasons == ()


def test_cbf_fail_closes_with_stop_when_active_dynamic_sample_is_missing():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )

    filtered, diagnostics = DistalLinkVelocityCBF().filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=None,
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert diagnostics.status == "fallback_stop_invalid_active_hand"
    assert "left:dynamic_sample_missing" in diagnostics.failure_reasons


def test_cbf_rejects_geometry_that_has_no_matching_tracked_hand_position():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )

    filtered, diagnostics = DistalLinkVelocityCBF().filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.zeros(3),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert "left:tracked_position_invalid" in diagnostics.failure_reasons


def test_cbf_fail_closes_with_stop_when_active_constraints_are_infeasible():
    left = HandSafetyResult(
        hand="left",
        geometry_valid=True,
        surface_gap_m=0.0,
        closest_link="panda_hand",
        closest_surface_point_world_pos=(0.0, 0.0, 0.0),
        closest_surface_point_valid=True,
    )
    right = HandSafetyResult(
        hand="right",
        geometry_valid=True,
        surface_gap_m=0.0,
        closest_link="panda_hand",
        closest_surface_point_world_pos=(0.0, 0.0, 0.0),
        closest_surface_point_valid=True,
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([0.01]),
        joint_velocities=None,
    )
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            safe_gap_m=0.05,
            activation_gap_m=0.13,
            gamma_per_s=8.0,
            prediction_horizon_s=0.0,
        )
    )

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=EndEffectorSafetyResult(left=left, right=right),
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.array([0.10, 0.0, 0.0]),
        },
        physics_dt_s=0.1,
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.0])
    np.testing.assert_allclose(filtered.joint_positions, [0.0])
    assert diagnostics.constraint_count == 2
    assert diagnostics.fallback_applied
    assert not diagnostics.feasible
    assert diagnostics.slack_radps > 0.0
    assert diagnostics.status == "fallback_stop_infeasible"
    assert diagnostics.relaxed_solution_available
    assert not diagnostics.relaxed_solution_applied


@pytest.mark.parametrize(
    ("config", "expected_message"),
    (
        (CBFConfig(objective_mode="unknown"), "objective_mode"),
        (CBFConfig(task_space_weight=0.0), "task_space_weight"),
        (
            CBFConfig(joint_regularization_epsilon=0.0),
            "joint_regularization_epsilon",
        ),
        (
            CBFConfig(correction_smoothness_weight=-1.0),
            "correction_smoothness_weight",
        ),
    ),
)
def test_cbf_objective_configuration_validation(config, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        config.validated()


def test_joint_nominal_objective_remains_the_default_and_ignores_new_weights():
    default_action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    explicit_action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    default_filtered, default_diagnostics = _filter_close_left_hand(
        arm_action=default_action,
        cbf=DistalLinkVelocityCBF(CBFConfig()),
    )
    explicit_filtered, explicit_diagnostics = _filter_close_left_hand(
        arm_action=explicit_action,
        cbf=DistalLinkVelocityCBF(
            CBFConfig(
                objective_mode="joint_nominal",
                task_space_weight=123.0,
                joint_regularization_epsilon=1e-4,
                correction_smoothness_weight=77.0,
            )
        ),
    )

    assert CBFConfig().objective_mode == "joint_nominal"
    np.testing.assert_array_equal(
        explicit_filtered.joint_positions, default_filtered.joint_positions
    )
    np.testing.assert_array_equal(
        explicit_filtered.joint_velocities, default_filtered.joint_velocities
    )
    assert explicit_diagnostics.objective_mode == "joint_nominal"
    assert (
        explicit_diagnostics.max_constraint_violation_after
        == default_diagnostics.max_constraint_violation_after
    )
    assert (
        explicit_diagnostics.intervention_norm_radps
        == default_diagnostics.intervention_norm_radps
    )


class _TwoJointArticulationView:
    body_names = ("panda_link0", "panda_hand")

    def get_jacobians(self):
        # The safety normal is +X, so the barrier row is [1, 1].  The large
        # second-joint Y component makes preserving task-space motion prefer
        # correcting joint 1, unlike the Euclidean joint-space projection.
        jacobian = np.zeros((1, 1, 6, 2), dtype=float)
        jacobian[0, 0, 0] = np.array([1.0, 1.0])
        jacobian[0, 0, 1] = np.array([0.0, 10.0])
        return jacobian


class _TwoJointRobot:
    def __init__(self):
        self._articulation_view = _TwoJointArticulationView()
        self.dof_properties = {
            "maxVelocity": np.full(2, 2.0),
            "lower": np.full(2, -3.0),
            "upper": np.full(2, 3.0),
        }

    def get_joint_positions(self):
        return np.zeros(2)


def _filter_two_joint_close_hand(objective_mode):
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            objective_mode=objective_mode,
            prediction_horizon_s=0.0,
            task_space_weight=1.0,
            joint_regularization_epsilon=0.01,
        )
    )
    action = SimpleNamespace(
        joint_indices=np.array([0, 1]),
        joint_positions=None,
        joint_velocities=np.array([-0.2, 0.0]),
    )
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    return cbf.filter_action(
        robot=_TwoJointRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )


def test_task_consistent_objective_can_differ_while_preserving_safe_set():
    joint_filtered, joint_diagnostics = _filter_two_joint_close_hand(
        "joint_nominal"
    )
    task_filtered, task_diagnostics = _filter_two_joint_close_hand(
        "task_consistent"
    )

    # Both variants enforce the unchanged barrier qdot_0 + qdot_1 >= 0.24.
    assert np.sum(joint_filtered.joint_velocities) >= 0.24 - 1e-5
    assert np.sum(task_filtered.joint_velocities) >= 0.24 - 1e-5
    assert joint_diagnostics.max_constraint_violation_after <= 1e-5
    assert task_diagnostics.max_constraint_violation_after <= 1e-5
    assert joint_diagnostics.feasible and task_diagnostics.feasible

    # A splits the Euclidean correction across the joints.  B avoids the
    # high task-space cost caused by joint 1's large Y Jacobian component.
    assert not np.allclose(
        task_filtered.joint_velocities,
        joint_filtered.joint_velocities,
        atol=1e-3,
    )
    assert abs(task_filtered.joint_velocities[1]) < abs(
        joint_filtered.joint_velocities[1]
    )
    assert task_diagnostics.objective_mode == "task_consistent"
    assert task_diagnostics.objective_solver_fallback is False
    assert task_diagnostics.solver_backend == "scipy_slsqp_task_consistent"
    assert len(task_diagnostics.task_velocity_nominal) == 4
    assert len(task_diagnostics.task_velocity_filtered) == 4


class _PhaseProgressArticulationView:
    body_names = ("panda_link0", "panda_hand")

    def get_jacobians(self):
        jacobian = np.zeros((1, 1, 6, 2), dtype=float)
        jacobian[0, 0, 0] = np.array([1.0, 1.0])
        jacobian[0, 0, 1] = np.array([0.0, -1.0])
        jacobian[0, 0, 2] = np.array([-1.0, 0.0])
        return jacobian


class _PhaseProgressRobot:
    def __init__(self):
        self._articulation_view = _PhaseProgressArticulationView()
        self.dof_properties = {
            "maxVelocity": np.full(2, 2.0),
            "lower": np.full(2, -3.0),
            "upper": np.full(2, 3.0),
        }

    def get_joint_positions(self):
        return np.zeros(2)


def _filter_phase_progress(
    *,
    event=5,
    attached=True,
    p_threshold=0.01,
    recovery=False,
    recovery_stage="inactive",
    place_spatially_ready=False,
    target_position_world_m=np.array([0.3, 1.0, 0.5]),
):
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            objective_mode="phase_progress",
            prediction_horizon_s=0.0,
            progress_retention_rho=0.9,
            progress_penalty_weight=200.0,
            progress_nominal_threshold_mps=p_threshold,
        )
    )
    action = SimpleNamespace(
        joint_indices=np.array([0, 1]),
        joint_positions=None,
        joint_velocities=np.array([-0.2, -0.1]),
    )
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    return cbf.filter_action(
        robot=_PhaseProgressRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
            "ee_pos": np.array([0.3, 0.0, 0.5]),
            "ee_to_cube": np.array([0.0, 1.0, 0.0]),
            "cube_to_place_target": np.array([0.0, 1.0, 0.0]),
            "has_grasped_cube": np.array([float(attached)]),
        },
        task_progress_context={
            "controller_event": event,
            "controller_t": 0.5,
            "target_position_world_m": target_position_world_m,
            "recovery_control_active": recovery,
            "recovery_stage": recovery_stage,
            "place_spatially_ready": place_spatially_ready,
        },
        physics_dt_s=0.1,
    )


def test_phase_progress_improves_retained_progress_without_changing_safe_set():
    filtered, diagnostics = _filter_phase_progress()

    assert np.sum(filtered.joint_velocities) >= 0.24 - 1e-5
    assert diagnostics.max_constraint_violation_after <= 1e-5
    assert diagnostics.task_progress_active
    assert diagnostics.task_progress_phase == "transport"
    assert diagnostics.task_progress_source == (
        "attached_cube_goal_ee_jacobian_proxy"
    )
    assert diagnostics.task_progress_A_mps < 0.0
    assert diagnostics.task_progress_filtered_mps > 0.08
    assert diagnostics.task_progress_filtered_mps <= (
        diagnostics.task_progress_nominal_mps + 1e-5
    )
    assert diagnostics.task_progress_shortfall_mps < (
        diagnostics.task_progress_A_shortfall_mps
    )
    assert diagnostics.solver_backend == "scipy_slsqp_phase_progress"


@pytest.mark.parametrize("event", [2, 3, 7, 8])
def test_phase_progress_uses_A_for_settle_grasp_and_release_events(event):
    filtered, diagnostics = _filter_phase_progress(event=event)

    np.testing.assert_allclose(filtered.joint_velocities, [0.07, 0.17], atol=1e-4)
    assert not diagnostics.task_progress_active
    assert diagnostics.solver_backend != "scipy_slsqp_phase_progress"


@pytest.mark.parametrize("event", [0, 1])
def test_phase_progress_enables_both_reach_events(event):
    _, diagnostics = _filter_phase_progress(event=event, attached=False)

    assert diagnostics.task_progress_active
    assert diagnostics.task_progress_phase == "approach"


def test_phase_progress_disables_transport_without_attachment():
    filtered, diagnostics = _filter_phase_progress(event=5, attached=False)

    np.testing.assert_allclose(filtered.joint_velocities, [0.07, 0.17], atol=1e-4)
    assert not diagnostics.task_progress_active
    assert diagnostics.task_progress_gate_reason == (
        "transport_without_attachment_uses_A_objective"
    )


def test_phase_progress_uses_actual_lift_target_only_when_attached():
    _, attached = _filter_phase_progress(event=4, attached=True)
    _, unattached = _filter_phase_progress(event=4, attached=False)

    assert attached.task_progress_active
    assert attached.task_progress_phase == "lift"
    assert attached.task_progress_source == "controller_target_direction"
    assert not unattached.task_progress_active
    assert unattached.task_progress_gate_reason == (
        "lift_without_attachment_uses_A_objective"
    )


@pytest.mark.parametrize(
    "target, expected_source",
    [
        (None, "world_up_fallback_missing_lift_target"),
        (
            np.array([0.3, 0.0, 0.5]),
            "world_up_fallback_degenerate_lift_target",
        ),
    ],
)
def test_phase_progress_lift_uses_world_up_only_as_target_fallback(
    target, expected_source
):
    _, diagnostics = _filter_phase_progress(
        event=4,
        attached=True,
        target_position_world_m=target,
    )

    assert diagnostics.task_progress_active
    assert diagnostics.task_progress_source == expected_source
    assert diagnostics.task_progress_direction_world == (0.0, 0.0, 1.0)


def test_phase_progress_disables_place_penalty_after_spatial_readiness():
    filtered, diagnostics = _filter_phase_progress(
        event=6, attached=True, place_spatially_ready=True
    )

    np.testing.assert_allclose(filtered.joint_velocities, [0.07, 0.17], atol=1e-4)
    assert not diagnostics.task_progress_active
    assert diagnostics.task_progress_gate_reason == (
        "place_spatially_ready_uses_A_objective"
    )


def test_phase_progress_recovery_stage_takes_precedence_over_aliased_event():
    _, diagnostics = _filter_phase_progress(
        event=6,
        attached=True,
        recovery=True,
        recovery_stage="place_lift",
        place_spatially_ready=True,
    )

    assert diagnostics.task_progress_active
    assert diagnostics.task_progress_phase == "recovery:place_lift"
    assert diagnostics.task_progress_source == "fixed_recovery_target_direction"


def test_phase_progress_small_nominal_gate_recovers_exact_A_solution():
    filtered, diagnostics = _filter_phase_progress(p_threshold=0.2)

    np.testing.assert_allclose(filtered.joint_velocities, [0.07, 0.17], atol=1e-4)
    assert not diagnostics.task_progress_active
    assert diagnostics.task_progress_gate_reason == (
        "nominal_progress_below_threshold"
    )


def _far_hand_safety_result():
    return EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.20,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )


def _filter_one_joint_with_safety_result(cbf, safety_result, nominal=-0.1):
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=None,
        joint_velocities=np.array([nominal], dtype=float),
    )
    return cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )


def test_smooth_intervention_memory_requires_commit_and_reset_clears_it():
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            objective_mode="smooth_intervention",
            correction_smoothness_weight=1.0,
            prediction_horizon_s=0.0,
        )
    )
    close_hand = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.02,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )

    _, uncommitted = _filter_one_joint_with_safety_result(cbf, close_hand)
    assert uncommitted.correction_radps[0] == pytest.approx(0.34, abs=1e-5)
    cbf.notify_action_committed(False)
    uncommitted_tail_action, uncommitted_tail = (
        _filter_one_joint_with_safety_result(cbf, _far_hand_safety_result())
    )
    np.testing.assert_array_equal(uncommitted_tail_action.joint_velocities, [-0.1])
    assert uncommitted_tail.status == "inactive"
    assert uncommitted_tail.intervention_norm_radps == 0.0

    _, committed = _filter_one_joint_with_safety_result(cbf, close_hand)
    cbf.notify_action_committed(True)
    tail_action, tail = _filter_one_joint_with_safety_result(
        cbf, _far_hand_safety_result()
    )
    previous = committed.correction_radps[0]
    expected_tail_correction = 0.5 * previous
    assert tail.status == "smooth_intervention_tail"
    assert tail.previous_correction_radps[0] == pytest.approx(previous, abs=1e-5)
    assert tail.correction_radps[0] == pytest.approx(
        expected_tail_correction, abs=1e-5
    )
    assert tail_action.joint_velocities[0] == pytest.approx(
        -0.1 + expected_tail_correction, abs=1e-5
    )
    assert abs(tail.correction_radps[0]) < abs(previous)

    cbf.reset()
    after_reset_action, after_reset = _filter_one_joint_with_safety_result(
        cbf, _far_hand_safety_result()
    )
    np.testing.assert_array_equal(after_reset_action.joint_velocities, [-0.1])
    assert after_reset.status == "inactive"
    assert after_reset.previous_correction_radps == (0.0,)
    assert after_reset.intervention_norm_radps == 0.0


def test_smooth_intervention_commit_marker_must_be_exact_boolean():
    cbf = DistalLinkVelocityCBF(
        CBFConfig(objective_mode="smooth_intervention")
    )
    with pytest.raises(ValueError, match="must be an exact boolean"):
        cbf.notify_action_committed(1)
