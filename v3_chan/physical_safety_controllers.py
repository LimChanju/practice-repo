"""Physical safety controllers shared by rollout and live-VR evaluation.

The learned task policy still selects an end-effector target.  RMPflow turns
that target into a nominal joint command.  The velocity-level CBF implemented
here projects that nominal command onto signed surface-gap constraints for the
tracked hands and the selected distal Panda links.

This module intentionally has no Isaac imports at module import time so the QP
projection and constraint construction can be unit tested without Isaac Sim.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


PHYSICAL_SAFETY_MODES = (
    "none",
    "rmpflow",
    "cbf",
    "rmpflow_cbf",
    "curobo",
    "curobo_cbf",
)

INTENTIONAL_HUMAN_ABSENCE_CONTRACT = (
    "explicit_true_all_zero_tracking_mask_no_human_positions_geometry_or_"
    "constraints_cbf_inactive_nominal_action_v1"
)

_NOMINAL_FEASIBLE_PASSTHROUGH_ATOL_RADPS = 1e-12


def mode_uses_rmpflow_obstacles(mode: str) -> bool:
    return str(mode) in {"rmpflow", "rmpflow_cbf"}


def mode_uses_cbf(mode: str) -> bool:
    return str(mode) in {"cbf", "rmpflow_cbf", "curobo_cbf"}


def mode_uses_curobo(mode: str) -> bool:
    return str(mode) in {"curobo", "curobo_cbf"}


@dataclass(frozen=True)
class CBFConfig:
    """Configuration for the velocity-level distal-link CBF filter."""

    safe_gap_m: float = 0.05
    activation_gap_m: float = 0.13
    gamma_per_s: float = 8.0
    prediction_horizon_s: float = 0.15
    max_prediction_buffer_m: float = 0.08
    max_joint_speed_rad_s: float = 2.0
    projection_iterations: int = 80
    projection_tolerance: float = 1e-5
    fail_closed_on_invalid_active_hand: bool = True
    stop_on_infeasible: bool = True
    require_valid_recorded_hand_tracking: bool = False

    def validated(self) -> "CBFConfig":
        values = (
            self.safe_gap_m,
            self.activation_gap_m,
            self.gamma_per_s,
            self.prediction_horizon_s,
            self.max_prediction_buffer_m,
            self.max_joint_speed_rad_s,
            self.projection_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("CBF configuration values must be finite")
        if self.safe_gap_m < 0.0:
            raise ValueError("safe_gap_m must be non-negative")
        if self.activation_gap_m < self.safe_gap_m:
            raise ValueError("activation_gap_m must be >= safe_gap_m")
        if self.gamma_per_s <= 0.0:
            raise ValueError("gamma_per_s must be positive")
        if self.prediction_horizon_s < 0.0:
            raise ValueError("prediction_horizon_s must be non-negative")
        if self.max_prediction_buffer_m < 0.0:
            raise ValueError("max_prediction_buffer_m must be non-negative")
        if self.max_joint_speed_rad_s <= 0.0:
            raise ValueError("max_joint_speed_rad_s must be positive")
        if int(self.projection_iterations) < 1:
            raise ValueError("projection_iterations must be positive")
        if self.projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive")
        if not isinstance(self.fail_closed_on_invalid_active_hand, bool):
            raise ValueError("fail_closed_on_invalid_active_hand must be boolean")
        if not isinstance(self.stop_on_infeasible, bool):
            raise ValueError("stop_on_infeasible must be boolean")
        if not isinstance(self.require_valid_recorded_hand_tracking, bool):
            raise ValueError("require_valid_recorded_hand_tracking must be boolean")
        return self


@dataclass(frozen=True)
class PhysicalSafetyDiagnostics:
    controller: str = "none"
    active: bool = False
    intervention_available: bool = False
    constraint_count: int = 0
    valid_hand_count: int = 0
    tracked_hand_count: int = 0
    intervention_norm_radps: float = 0.0
    nominal_velocity_norm_radps: float = 0.0
    filtered_velocity_norm_radps: float = 0.0
    max_constraint_violation_before: float = 0.0
    max_constraint_violation_after: float = 0.0
    slack_radps: float = 0.0
    min_predicted_gap_m: float = 10.0
    solve_time_ms: float = 0.0
    feasible: bool = True
    fallback_applied: bool = False
    failure_reasons: tuple[str, ...] = ()
    status: str = "inactive"
    solver_converged: bool = True
    infeasibility_proven: bool = False
    relaxed_solution_available: bool = False
    relaxed_solution_applied: bool = False
    solver_backend: str = "none"
    projection_status: str = "inactive"
    intentional_human_absence: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # These aliases correct legacy field names without breaking existing
        # artifacts. J [m/rad] @ qdot [rad/s] and its bound are in m/s, so
        # half-space violation and slack are not angular velocities.
        payload.update(
            {
                "max_constraint_violation_before_mps": float(
                    self.max_constraint_violation_before
                ),
                "max_constraint_violation_after_mps": float(
                    self.max_constraint_violation_after
                ),
                "slack_mps": float(self.slack_radps),
                "buffered_current_gap_m": float(self.min_predicted_gap_m),
                "legacy_unit_aliases": {
                    "slack_radps": "slack_mps",
                    "min_predicted_gap_m": "buffered_current_gap_m",
                },
            }
        )
        return payload


@dataclass(frozen=True)
class _CBFConstraint:
    hand: str
    jacobian_row: np.ndarray
    lower_bound_mps: float
    buffered_current_gap_m: float


@dataclass(frozen=True)
class _VelocityProjectionResult:
    velocity: np.ndarray
    slack_mps: float
    feasible: bool
    violation_before_mps: float
    violation_after_mps: float
    solver_converged: bool
    infeasibility_proven: bool
    relaxed_solution_available: bool
    solver_backend: str
    status: str


def project_velocity_qp(
    nominal_velocity: np.ndarray,
    constraint_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    *,
    max_iterations: int = 80,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, float, bool, float, float]:
    """Project a velocity onto box and half-space constraints.

    The solved convex problem is ``min 0.5 ||qdot-qdot_nom||^2`` subject to
    ``A qdot >= b`` and joint-velocity bounds.  Dykstra projections give the
    Euclidean projection without an external QP package.  If the constraints
    are infeasible under the velocity limits, a common non-negative slack is
    found by bisection and returned explicitly.
    """

    result = _project_velocity_qp_detailed(
        nominal_velocity,
        constraint_matrix,
        lower_bounds,
        velocity_lower,
        velocity_upper,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return (
        result.velocity,
        result.slack_mps,
        result.feasible,
        result.violation_before_mps,
        result.violation_after_mps,
    )


def _project_velocity_qp_detailed(
    nominal_velocity: np.ndarray,
    constraint_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> _VelocityProjectionResult:
    """Return a projection plus evidence about convergence/infeasibility."""

    nominal = np.asarray(nominal_velocity, dtype=float).reshape(-1)
    lower = np.asarray(velocity_lower, dtype=float).reshape(nominal.shape)
    upper = np.asarray(velocity_upper, dtype=float).reshape(nominal.shape)
    matrix = np.asarray(constraint_matrix, dtype=float)
    bounds = np.asarray(lower_bounds, dtype=float).reshape(-1)
    if matrix.size == 0:
        matrix = np.empty((0, nominal.size), dtype=float)
    else:
        matrix = matrix.reshape((-1, nominal.size))
    if bounds.shape != (matrix.shape[0],):
        raise ValueError("constraint bounds do not match constraint matrix")
    if np.any(lower > upper):
        raise ValueError("velocity lower bound exceeds upper bound")
    if not all(
        np.all(np.isfinite(value)) for value in (nominal, matrix, bounds, lower, upper)
    ):
        raise ValueError("QP inputs must be finite")

    clipped_nominal = np.clip(nominal, lower, upper)
    before = _max_halfspace_violation(matrix, bounds, clipped_nominal)
    if matrix.shape[0] == 0:
        return _VelocityProjectionResult(
            velocity=clipped_nominal,
            slack_mps=0.0,
            feasible=True,
            violation_before_mps=before,
            violation_after_mps=0.0,
            solver_converged=True,
            infeasibility_proven=False,
            relaxed_solution_available=False,
            solver_backend="box_clip",
            status="solved",
        )

    projected, violation, converged = _dykstra_project(
        nominal,
        matrix,
        bounds,
        lower,
        upper,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if violation <= tolerance and converged:
        return _VelocityProjectionResult(
            velocity=projected,
            slack_mps=0.0,
            feasible=True,
            violation_before_mps=before,
            violation_after_mps=violation,
            solver_converged=True,
            infeasibility_proven=False,
            relaxed_solution_available=False,
            solver_backend="dykstra",
            status="solved",
        )

    scipy_result = _scipy_projection_fallback(
        nominal,
        matrix,
        bounds,
        lower,
        upper,
        tolerance=tolerance,
        violation_before=before,
    )
    if scipy_result is not None:
        return scipy_result

    # Keep a deterministic dependency-free retry. This path is bounded and is
    # used only when scipy is unavailable or cannot certify the LP status.
    retry_iterations = min(50_000, max(20_000, int(max_iterations) * 250))
    projected, violation, converged = _dykstra_project(
        nominal,
        matrix,
        bounds,
        lower,
        upper,
        max_iterations=retry_iterations,
        tolerance=tolerance,
    )
    if violation <= tolerance:
        return _VelocityProjectionResult(
            velocity=projected,
            slack_mps=0.0,
            feasible=True,
            violation_before_mps=before,
            violation_after_mps=violation,
            solver_converged=bool(converged),
            infeasibility_proven=False,
            relaxed_solution_available=False,
            solver_backend="dykstra_retry",
            status=("solved_after_retry" if converged else "feasible_nonoptimal"),
        )

    # A positive common slack keeps the filter deterministic and exposes
    # infeasibility to the evaluator instead of silently returning bad values.
    slack_high = max(float(violation), tolerance)
    for _ in range(20):
        candidate, candidate_violation, _ = _dykstra_project(
            nominal,
            matrix,
            bounds - slack_high,
            lower,
            upper,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if candidate_violation <= tolerance:
            projected = candidate
            break
        slack_high *= 2.0
    else:
        return _VelocityProjectionResult(
            velocity=clipped_nominal,
            slack_mps=slack_high,
            feasible=False,
            violation_before_mps=before,
            violation_after_mps=violation,
            solver_converged=False,
            infeasibility_proven=False,
            relaxed_solution_available=False,
            solver_backend="dykstra_retry",
            status="solver_unresolved",
        )

    slack_low = 0.0
    for _ in range(24):
        slack_mid = 0.5 * (slack_low + slack_high)
        candidate, candidate_violation, _ = _dykstra_project(
            nominal,
            matrix,
            bounds - slack_mid,
            lower,
            upper,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if candidate_violation <= tolerance:
            slack_high = slack_mid
            projected = candidate
        else:
            slack_low = slack_mid

    original_violation = _max_halfspace_violation(matrix, bounds, projected)
    return _VelocityProjectionResult(
        velocity=projected,
        slack_mps=max(float(slack_high), original_violation),
        feasible=False,
        violation_before_mps=before,
        violation_after_mps=original_violation,
        solver_converged=False,
        infeasibility_proven=False,
        relaxed_solution_available=True,
        solver_backend="dykstra_retry",
        status="relaxed_unresolved",
    )


def _scipy_projection_fallback(
    nominal: np.ndarray,
    matrix: np.ndarray,
    bounds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
    violation_before: float,
) -> _VelocityProjectionResult | None:
    """Certify feasibility with HiGHS, then compute the closest safe velocity.

    Imports are lazy so the normal real-time path and installations without
    scipy retain the dependency-free Dykstra implementation.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
    except (ImportError, ModuleNotFoundError):
        return None

    box_nominal = np.clip(nominal, lower, upper)
    variable_bounds = list(zip(lower.tolist(), upper.tolist()))

    def run_feasibility_lp(*, presolve: bool):
        return linprog(
            np.zeros(nominal.size, dtype=float),
            A_ub=-matrix,
            b_ub=-bounds,
            bounds=variable_bounds,
            method="highs",
            options={"presolve": presolve},
        )

    try:
        feasibility = run_feasibility_lp(presolve=True)
        if int(feasibility.status) == 2:
            # Require an independent no-presolve result before treating an LP
            # status as proof; nearly opposed rows are numerically delicate.
            feasibility = run_feasibility_lp(presolve=False)
    except Exception:
        return None

    if int(feasibility.status) == 0 and feasibility.x is not None:
        feasible_seed = np.clip(np.asarray(feasibility.x, dtype=float), lower, upper)
        projected, converged = _slsqp_closest_velocity(
            nominal,
            matrix,
            bounds,
            lower,
            upper,
            feasible_seed,
            tolerance=tolerance,
            Bounds=Bounds,
            LinearConstraint=LinearConstraint,
            minimize=minimize,
        )
        violation = _max_halfspace_violation(matrix, bounds, projected)
        if violation > tolerance:
            # HiGHS' feasible point is still a safe deterministic fallback if
            # SLSQP fails to optimize the Euclidean projection.
            projected = feasible_seed
            violation = _max_halfspace_violation(matrix, bounds, projected)
            converged = False
        if violation <= tolerance:
            return _VelocityProjectionResult(
                velocity=projected,
                slack_mps=0.0,
                feasible=True,
                violation_before_mps=violation_before,
                violation_after_mps=violation,
                solver_converged=bool(converged),
                infeasibility_proven=False,
                relaxed_solution_available=False,
                solver_backend="scipy_highs_slsqp",
                status=("solved_after_retry" if converged else "feasible_nonoptimal"),
            )
        return None

    if int(feasibility.status) != 2:
        return None

    # HiGHS has certified that the original half-spaces and box are
    # inconsistent. Minimize one common non-negative slack and then project
    # onto that minimally relaxed feasible set.
    relaxed_objective = np.zeros(nominal.size + 1, dtype=float)
    relaxed_objective[-1] = 1.0
    relaxed_a_ub = np.column_stack((-matrix, -np.ones(matrix.shape[0])))
    relaxed_bounds = variable_bounds + [(0.0, None)]
    try:
        relaxed = linprog(
            relaxed_objective,
            A_ub=relaxed_a_ub,
            b_ub=-bounds,
            bounds=relaxed_bounds,
            method="highs",
            options={"presolve": True},
        )
    except Exception:
        violation = _max_halfspace_violation(matrix, bounds, box_nominal)
        return _VelocityProjectionResult(
            velocity=box_nominal,
            slack_mps=violation,
            feasible=False,
            violation_before_mps=violation_before,
            violation_after_mps=violation,
            solver_converged=False,
            infeasibility_proven=True,
            relaxed_solution_available=False,
            solver_backend="scipy_highs",
            status="infeasible_proven_relaxation_failed",
        )
    if int(relaxed.status) != 0 or relaxed.x is None:
        violation = _max_halfspace_violation(matrix, bounds, box_nominal)
        return _VelocityProjectionResult(
            velocity=box_nominal,
            slack_mps=violation,
            feasible=False,
            violation_before_mps=violation_before,
            violation_after_mps=violation,
            solver_converged=False,
            infeasibility_proven=True,
            relaxed_solution_available=False,
            solver_backend="scipy_highs",
            status="infeasible_proven_relaxation_failed",
        )

    slack = max(0.0, float(relaxed.x[-1]))
    relaxed_seed = np.clip(np.asarray(relaxed.x[:-1], dtype=float), lower, upper)
    relaxed_lower_bounds = bounds - slack
    projected, converged = _slsqp_closest_velocity(
        nominal,
        matrix,
        relaxed_lower_bounds,
        lower,
        upper,
        relaxed_seed,
        tolerance=tolerance,
        Bounds=Bounds,
        LinearConstraint=LinearConstraint,
        minimize=minimize,
    )
    relaxed_violation = _max_halfspace_violation(
        matrix, relaxed_lower_bounds, projected
    )
    if relaxed_violation > tolerance:
        projected = relaxed_seed
        converged = False
    original_violation = _max_halfspace_violation(matrix, bounds, projected)
    return _VelocityProjectionResult(
        velocity=projected,
        slack_mps=max(slack, original_violation),
        feasible=False,
        violation_before_mps=violation_before,
        violation_after_mps=original_violation,
        solver_converged=bool(converged),
        infeasibility_proven=True,
        relaxed_solution_available=True,
        solver_backend="scipy_highs_slsqp",
        status=(
            "infeasible_proven_relaxed"
            if converged
            else "infeasible_proven_relaxed_nonoptimal"
        ),
    )


def _slsqp_closest_velocity(
    nominal: np.ndarray,
    matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    initial: np.ndarray,
    *,
    tolerance: float,
    Bounds,
    LinearConstraint,
    minimize,
) -> tuple[np.ndarray, bool]:
    try:
        result = minimize(
            lambda value: 0.5 * float((value - nominal) @ (value - nominal)),
            initial,
            jac=lambda value: value - nominal,
            method="SLSQP",
            bounds=Bounds(velocity_lower, velocity_upper),
            constraints=LinearConstraint(matrix, lower_bounds, np.inf),
            options={
                "maxiter": 200,
                "ftol": min(1e-12, max(1e-15, tolerance * tolerance)),
                "disp": False,
            },
        )
    except Exception:
        return np.asarray(initial, dtype=float).copy(), False
    candidate = np.clip(
        np.asarray(result.x, dtype=float), velocity_lower, velocity_upper
    )
    finite = candidate.shape == nominal.shape and np.all(np.isfinite(candidate))
    return (candidate if finite else np.asarray(initial, dtype=float).copy()), bool(
        result.success and finite
    )


class DistalLinkVelocityCBF:
    """Filter an Isaac articulation action using distal-link CBF constraints."""

    def __init__(self, config: CBFConfig | None = None) -> None:
        self.config = (config or CBFConfig()).validated()
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="cbf")

    def reset(self) -> None:
        self.last_diagnostics = PhysicalSafetyDiagnostics(controller="cbf")

    def filter_action(
        self,
        *,
        robot,
        arm_action,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        physics_dt_s: float,
        human_valid_mask=None,
        intentional_human_absence: bool = False,
    ):
        started = time.perf_counter()
        try:
            filtered_action, diagnostics = self._filter_action(
                robot=robot,
                arm_action=arm_action,
                safety_result=safety_result,
                dynamic_sample=dynamic_sample,
                safety_geometry=safety_geometry,
                observation=observation,
                physics_dt_s=physics_dt_s,
                human_valid_mask=human_valid_mask,
                intentional_human_absence=intentional_human_absence,
            )
        except Exception as exc:
            diagnostics = PhysicalSafetyDiagnostics(
                controller="cbf",
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                feasible=False,
                solver_converged=False,
                projection_status="error",
                status=f"error:{type(exc).__name__}:{exc}",
            )
            self.last_diagnostics = diagnostics
            raise RuntimeError(
                f"CBF safety filter failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_diagnostics = diagnostics
        return filtered_action, diagnostics

    def _filter_action(
        self,
        *,
        robot,
        arm_action,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        physics_dt_s: float,
        human_valid_mask=None,
        intentional_human_absence: bool = False,
    ):
        started = time.perf_counter()
        if not isinstance(intentional_human_absence, (bool, np.bool_)):
            raise ValueError("intentional_human_absence must be an exact boolean")
        intentional_absence = bool(intentional_human_absence)
        dt_s = float(physics_dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")

        current_all = _to_numpy(robot.get_joint_positions()).reshape(-1)
        joint_indices = getattr(arm_action, "joint_indices", None)
        position_targets = getattr(arm_action, "joint_positions", None)
        velocity_targets = getattr(arm_action, "joint_velocities", None)
        if joint_indices is None:
            target_size = len(position_targets) if position_targets is not None else 0
            joint_indices = np.arange(target_size, dtype=int)
        else:
            joint_indices = _to_numpy(joint_indices).astype(int).reshape(-1)
        if joint_indices.size == 0:
            raise ValueError("arm action has no active joint indices")

        if velocity_targets is not None:
            nominal_velocity = _to_numpy(velocity_targets).astype(float).reshape(-1)
        elif position_targets is not None:
            positions = _to_numpy(position_targets).astype(float).reshape(-1)
            nominal_velocity = (positions - current_all[joint_indices]) / dt_s
        else:
            raise ValueError("arm action has no joint position or velocity targets")
        if nominal_velocity.shape != joint_indices.shape:
            raise ValueError("arm action target shape does not match joint indices")

        jacobians, body_names = _robot_jacobians_and_body_names(robot)
        tracked_hand_count, valid_hand_count, required_hands, failures = (
            self._active_hand_requirements(
                safety_result=safety_result,
                dynamic_sample=dynamic_sample,
                observation=observation,
                human_valid_mask=human_valid_mask,
                intentional_human_absence=intentional_absence,
            )
        )
        constraints = list(
            self._constraints(
                safety_result=safety_result,
                dynamic_sample=dynamic_sample,
                safety_geometry=safety_geometry,
                observation=observation,
                jacobians=jacobians,
                body_names=body_names,
                active_joint_indices=joint_indices,
            )
        )
        constructed_hands = {constraint.hand for constraint in constraints}
        for hand_name in required_hands:
            if hand_name not in constructed_hands:
                failures.append(f"{hand_name}:constraint_not_constructed")
        failures.extend(
            self._intentional_absence_failures(
                intentional_human_absence=intentional_absence,
                human_valid_mask=human_valid_mask,
                observation=observation,
                safety_result=safety_result,
                tracked_hand_count=tracked_hand_count,
                valid_hand_count=valid_hand_count,
                required_hands=required_hands,
                constructed_constraints=constraints,
            )
        )
        if not constraints and not failures:
            nominal_velocity_norm = float(np.linalg.norm(nominal_velocity))
            if not math.isfinite(nominal_velocity_norm):
                raise ValueError("QP inputs must be finite")
            diagnostics = PhysicalSafetyDiagnostics(
                controller="cbf",
                active=False,
                intervention_available=True,
                constraint_count=0,
                valid_hand_count=int(valid_hand_count),
                tracked_hand_count=int(tracked_hand_count),
                intervention_norm_radps=0.0,
                nominal_velocity_norm_radps=nominal_velocity_norm,
                filtered_velocity_norm_radps=nominal_velocity_norm,
                max_constraint_violation_before=0.0,
                max_constraint_violation_after=0.0,
                slack_radps=0.0,
                min_predicted_gap_m=10.0,
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                feasible=True,
                solver_converged=True,
                infeasibility_proven=False,
                relaxed_solution_available=False,
                relaxed_solution_applied=False,
                solver_backend="none",
                projection_status="inactive",
                fallback_applied=False,
                failure_reasons=(),
                status="inactive",
                intentional_human_absence=intentional_absence,
            )
            return arm_action, diagnostics

        speed_limit = _joint_speed_limits(robot, joint_indices, self.config)
        velocity_lower = -speed_limit
        velocity_upper = speed_limit
        matrix = np.asarray(
            [constraint.jacobian_row for constraint in constraints], dtype=float
        )
        bounds = np.asarray(
            [constraint.lower_bound_mps for constraint in constraints], dtype=float
        )
        projection = _project_velocity_qp_detailed(
            nominal_velocity,
            matrix,
            bounds,
            velocity_lower,
            velocity_upper,
            max_iterations=self.config.projection_iterations,
            tolerance=self.config.projection_tolerance,
        )
        filtered_velocity = projection.velocity.copy()
        slack = projection.slack_mps
        feasible = projection.feasible
        before = projection.violation_before_mps
        after = projection.violation_after_mps

        fallback_applied = False
        recorded_tracking_failure = any(
            ":recorded_tracking_" in reason for reason in failures
        )
        intentional_absence_failure = any(
            reason.startswith("intentional_absence:") for reason in failures
        )
        if recorded_tracking_failure or intentional_absence_failure or (
            failures and self.config.fail_closed_on_invalid_active_hand
        ):
            filtered_velocity = np.zeros_like(nominal_velocity)
            after = _max_halfspace_violation(matrix, bounds, filtered_velocity)
            feasible = False
            fallback_applied = True
            status = "fallback_stop_invalid_active_hand"
        elif not feasible and (
            self.config.stop_on_infeasible or not projection.relaxed_solution_available
        ):
            filtered_velocity = np.zeros_like(nominal_velocity)
            after = _max_halfspace_violation(matrix, bounds, filtered_velocity)
            fallback_applied = True
            status = "fallback_stop_infeasible"
        else:
            status = (
                "inactive"
                if not constraints
                else ("solved" if feasible else "solved_with_slack")
            )

        nominal_feasible_passthrough = bool(
            constraints
            and not failures
            and feasible
            and projection.solver_converged
            and not fallback_applied
            and np.allclose(
                filtered_velocity,
                nominal_velocity,
                rtol=0.0,
                atol=_NOMINAL_FEASIBLE_PASSTHROUGH_ATOL_RADPS,
            )
        )
        if nominal_feasible_passthrough:
            status = "active_nominal_feasible_passthrough"
            intervention = 0.0
            filtered_velocity_norm = float(np.linalg.norm(nominal_velocity))
        else:
            if position_targets is not None:
                safe_positions = current_all[joint_indices] + filtered_velocity * dt_s
                safe_positions = _clip_joint_positions(
                    robot, joint_indices, safe_positions
                )
                arm_action.joint_positions = safe_positions
            arm_action.joint_velocities = filtered_velocity
            intervention = float(np.linalg.norm(filtered_velocity - nominal_velocity))
            filtered_velocity_norm = float(np.linalg.norm(filtered_velocity))
        diagnostics = PhysicalSafetyDiagnostics(
            controller="cbf",
            active=bool(constraints),
            intervention_available=True,
            constraint_count=len(constraints),
            valid_hand_count=int(valid_hand_count),
            tracked_hand_count=int(tracked_hand_count),
            intervention_norm_radps=intervention,
            nominal_velocity_norm_radps=float(np.linalg.norm(nominal_velocity)),
            filtered_velocity_norm_radps=filtered_velocity_norm,
            max_constraint_violation_before=float(before),
            max_constraint_violation_after=float(after),
            slack_radps=float(slack),
            min_predicted_gap_m=min(
                (constraint.buffered_current_gap_m for constraint in constraints),
                default=10.0,
            ),
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            feasible=bool(feasible),
            solver_converged=bool(projection.solver_converged),
            infeasibility_proven=bool(projection.infeasibility_proven),
            relaxed_solution_available=bool(projection.relaxed_solution_available),
            relaxed_solution_applied=bool(
                not feasible
                and projection.relaxed_solution_available
                and not fallback_applied
            ),
            solver_backend=projection.solver_backend,
            projection_status=projection.status,
            fallback_applied=bool(fallback_applied),
            failure_reasons=tuple(sorted(set(failures))),
            status=status,
            intentional_human_absence=intentional_absence,
        )
        return arm_action, diagnostics

    def _active_hand_requirements(
        self,
        *,
        safety_result,
        dynamic_sample,
        observation: dict[str, np.ndarray],
        human_valid_mask=None,
        intentional_human_absence: bool = False,
    ) -> tuple[int, int, tuple[str, ...], list[str]]:
        """Audit every tracked hand that should produce an active constraint.

        Missing people are not failures. Once a finite hand position is
        present, however, invalid geometry is unknown safety state rather than
        evidence of clearance. Inside the activation band, a missing/invalid
        3-D hand velocity or a failed downstream constraint build triggers the
        controlled-stop path instead of silently becoming an inactive CBF.
        Closing speed is only an extra prediction-buffer signal: a collider
        identity switch may invalidate it for one row even while the 3-D hand
        velocity needed by the actual barrier constraint remains valid.
        """

        tracked = 0
        valid = 0
        required: list[str] = []
        failures = self._recorded_tracking_failures(
            human_valid_mask,
            intentional_human_absence=intentional_human_absence,
        )
        for hand_name in ("left", "right"):
            hand_result = getattr(safety_result, hand_name)
            hand_pos = _observation_vec3(observation, f"human_{hand_name}_hand_pos")
            if hand_pos is None:
                if bool(getattr(hand_result, "geometry_valid", False)):
                    failures.append(f"{hand_name}:tracked_position_invalid")
                continue
            tracked += 1
            if not bool(getattr(hand_result, "geometry_valid", False)):
                failures.append(f"{hand_name}:geometry_invalid")
                continue
            valid += 1
            gap_m = float(getattr(hand_result, "surface_gap_m", math.nan))
            if not math.isfinite(gap_m):
                failures.append(f"{hand_name}:gap_invalid")
                continue
            if gap_m > self.config.activation_gap_m:
                continue
            required.append(hand_name)
            dynamic_hand = getattr(dynamic_sample, hand_name, None)
            if dynamic_hand is None:
                failures.append(f"{hand_name}:dynamic_sample_missing")
                continue
            if not bool(getattr(dynamic_hand, "hand_velocity_valid", False)):
                failures.append(f"{hand_name}:hand_velocity_invalid")
                continue
            if _finite_hand_velocity(dynamic_hand) is None:
                failures.append(f"{hand_name}:hand_velocity_payload_invalid")
            if bool(getattr(dynamic_hand, "closing_speed_valid", False)) and (
                _finite_closing_speed(dynamic_hand) is None
            ):
                failures.append(f"{hand_name}:closing_speed_payload_invalid")
        return tracked, valid, tuple(required), failures

    def _recorded_tracking_failures(
        self,
        human_valid_mask,
        *,
        intentional_human_absence: bool = False,
    ) -> list[str]:
        """Validate a recorded ``[head, left, right]`` tracking mask."""

        if bool(intentional_human_absence):
            # A separate proof below distinguishes deliberate replay exhaustion
            # from a real recorded tracking dropout.
            return []
        if not self.config.require_valid_recorded_hand_tracking:
            return []
        if human_valid_mask is None:
            return [
                "left:recorded_tracking_mask_missing",
                "right:recorded_tracking_mask_missing",
            ]
        try:
            mask = _to_numpy(human_valid_mask)
        except Exception:
            mask = np.empty(0)
        if mask.shape != (3,) or mask.dtype.kind not in "buif":
            return [
                "left:recorded_tracking_mask_invalid",
                "right:recorded_tracking_mask_invalid",
            ]
        numeric_mask = mask.astype(float, copy=False)
        if not np.all(np.isfinite(numeric_mask)) or not np.all(
            (numeric_mask == 0.0) | (numeric_mask == 1.0)
        ):
            return [
                "left:recorded_tracking_mask_invalid",
                "right:recorded_tracking_mask_invalid",
            ]
        failures: list[str] = []
        for hand_name, mask_index in (("left", 1), ("right", 2)):
            if not bool(numeric_mask[mask_index]):
                failures.append(f"{hand_name}:recorded_tracking_mask_false")
        return failures

    @staticmethod
    def _intentional_absence_failures(
        *,
        intentional_human_absence: bool,
        human_valid_mask,
        observation: dict[str, np.ndarray],
        safety_result,
        tracked_hand_count: int,
        valid_hand_count: int,
        required_hands: tuple[str, ...],
        constructed_constraints: list[_CBFConstraint],
    ) -> list[str]:
        """Prove the narrow replay-exhaustion exception to tracking fail-stop."""

        if not intentional_human_absence:
            return []
        failures: list[str] = []
        try:
            mask = _to_numpy(human_valid_mask)
        except Exception:
            mask = np.empty(0)
        if mask.shape != (3,) or mask.dtype.kind not in "buif":
            failures.append("intentional_absence:tracking_mask_invalid")
        else:
            numeric_mask = mask.astype(float, copy=False)
            if not np.all(np.isfinite(numeric_mask)) or not np.array_equal(
                numeric_mask, np.zeros(3, dtype=float)
            ):
                failures.append("intentional_absence:tracking_mask_not_all_zero")

        for field in (
            "human_head_pos",
            "human_left_hand_pos",
            "human_right_hand_pos",
        ):
            if _observation_vec3(observation, field) is not None:
                failures.append(f"intentional_absence:{field}_present")

        hand_results = tuple(
            getattr(safety_result, hand_name, None)
            for hand_name in ("left", "right")
        )
        if any(result is None for result in hand_results):
            failures.append("intentional_absence:hand_geometry_result_missing")
        else:
            for hand_name, result in zip(("left", "right"), hand_results):
                if bool(getattr(result, "geometry_valid", False)):
                    failures.append(
                        f"intentional_absence:{hand_name}_geometry_present"
                    )
                if any(
                    bool(getattr(result, field, False))
                    for field in ("collision", "contact", "near_miss", "near")
                ):
                    failures.append(
                        f"intentional_absence:{hand_name}_safety_state_present"
                    )

        if int(tracked_hand_count) != 0:
            failures.append("intentional_absence:tracked_hand_count_nonzero")
        if int(valid_hand_count) != 0:
            failures.append("intentional_absence:valid_hand_count_nonzero")
        if tuple(required_hands):
            failures.append("intentional_absence:required_hands_present")
        if list(constructed_constraints):
            failures.append("intentional_absence:constraints_present")
        return failures

    def _constraints(
        self,
        *,
        safety_result,
        dynamic_sample,
        safety_geometry,
        observation: dict[str, np.ndarray],
        jacobians: np.ndarray,
        body_names: tuple[str, ...],
        active_joint_indices: np.ndarray,
    ) -> Iterable[_CBFConstraint]:
        for hand_name in ("left", "right"):
            hand_result = getattr(safety_result, hand_name)
            if not bool(hand_result.geometry_valid):
                continue
            gap_m = float(hand_result.surface_gap_m)
            if not math.isfinite(gap_m) or gap_m > self.config.activation_gap_m:
                continue
            hand_pos = _observation_vec3(observation, f"human_{hand_name}_hand_pos")
            if hand_pos is None:
                continue
            link_origin, _, origin_valid = safety_geometry.closest_link_world_pose(
                hand_result
            )
            if not origin_valid or link_origin is None:
                continue
            link_origin = np.asarray(link_origin, dtype=float).reshape(3)
            surface_point = None
            if bool(hand_result.closest_surface_point_valid):
                candidate = np.asarray(
                    hand_result.closest_surface_point_world_pos, dtype=float
                ).reshape(-1)
                if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                    surface_point = candidate[:3]
            if surface_point is None:
                surface_point = link_origin
            normal = surface_point - hand_pos
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-8:
                continue
            normal /= normal_norm

            body_jacobian = _body_jacobian(
                jacobians, body_names, str(hand_result.closest_link)
            )
            offset = surface_point - link_origin
            point_jacobian = body_jacobian[:3] - _skew(offset) @ body_jacobian[3:6]
            point_jacobian = point_jacobian[:, active_joint_indices]
            constraint_row = normal @ point_jacobian
            if float(np.linalg.norm(constraint_row)) <= 1e-9:
                continue

            # ``_active_hand_requirements`` already records a missing dynamic
            # sample as a fail-closed reason.  Do not raise here before that
            # reason can reach the controlled-stop path.
            hand_dynamic = getattr(dynamic_sample, hand_name, None)
            if hand_dynamic is None:
                continue
            if not bool(getattr(hand_dynamic, "hand_velocity_valid", False)):
                continue
            hand_velocity = _finite_hand_velocity(hand_dynamic)
            if hand_velocity is None:
                continue
            closing_speed = 0.0
            if bool(getattr(hand_dynamic, "closing_speed_valid", False)):
                candidate_closing_speed = _finite_closing_speed(hand_dynamic)
                if candidate_closing_speed is None:
                    continue
                closing_speed = max(0.0, candidate_closing_speed)
            prediction_buffer = min(
                self.config.max_prediction_buffer_m,
                self.config.prediction_horizon_s * closing_speed,
            )
            safe_gap = self.config.safe_gap_m + prediction_buffer
            barrier_value = gap_m - safe_gap
            lower_bound = float(
                normal @ hand_velocity - self.config.gamma_per_s * barrier_value
            )
            yield _CBFConstraint(
                hand=hand_name,
                jacobian_row=np.asarray(constraint_row, dtype=float),
                lower_bound_mps=lower_bound,
                buffered_current_gap_m=float(gap_m - prediction_buffer),
            )


def _dykstra_project(
    initial: np.ndarray,
    matrix: np.ndarray,
    bounds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    projections = 1 + matrix.shape[0]
    corrections = [np.zeros_like(initial) for _ in range(projections)]
    value = np.asarray(initial, dtype=float).copy()
    converged = False
    for _ in range(int(max_iterations)):
        previous = value.copy()
        shifted = value + corrections[0]
        value = np.clip(shifted, lower, upper)
        corrections[0] = shifted - value
        for index, (row, bound) in enumerate(zip(matrix, bounds), start=1):
            shifted = value + corrections[index]
            row_norm_sq = float(row @ row)
            if row_norm_sq <= 1e-16:
                value = shifted
            else:
                violation = float(bound - row @ shifted)
                value = (
                    shifted + (violation / row_norm_sq) * row
                    if violation > 0.0
                    else shifted
                )
            corrections[index] = shifted - value
        if (
            float(np.linalg.norm(value - previous, ord=np.inf)) <= tolerance
            and _max_halfspace_violation(matrix, bounds, value) <= tolerance
            and np.all(value >= lower - tolerance)
            and np.all(value <= upper + tolerance)
        ):
            converged = True
            break
    value = np.clip(value, lower, upper)
    return value, _max_halfspace_violation(matrix, bounds, value), converged


def _max_halfspace_violation(
    matrix: np.ndarray, bounds: np.ndarray, value: np.ndarray
) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    return float(max(0.0, np.max(bounds - matrix @ value)))


def _robot_jacobians_and_body_names(robot) -> tuple[np.ndarray, tuple[str, ...]]:
    view = getattr(robot, "_articulation_view", None)
    if view is None:
        raise RuntimeError("robot articulation view is unavailable")
    jacobians = _to_numpy(view.get_jacobians())
    if jacobians.ndim == 4 and jacobians.shape[0] == 1:
        jacobians = jacobians[0]
    if jacobians.ndim != 3 or jacobians.shape[1] != 6:
        raise RuntimeError(f"unexpected articulation Jacobian shape {jacobians.shape}")
    body_names = tuple(str(name) for name in view.body_names)
    return np.asarray(jacobians, dtype=float), body_names


def _body_jacobian(
    jacobians: np.ndarray, body_names: tuple[str, ...], link_name: str
) -> np.ndarray:
    if link_name not in body_names:
        raise RuntimeError(f"link {link_name!r} is absent from articulation metadata")
    body_index = body_names.index(link_name)
    if jacobians.shape[0] == len(body_names) - 1:
        jacobian_index = body_index - 1
    elif jacobians.shape[0] == len(body_names):
        jacobian_index = body_index
    else:
        raise RuntimeError(
            "cannot map body metadata to Jacobian rows: "
            f"bodies={len(body_names)} jacobians={jacobians.shape[0]}"
        )
    if jacobian_index < 0 or jacobian_index >= jacobians.shape[0]:
        raise RuntimeError(f"body {link_name!r} has no movable-link Jacobian")
    return jacobians[jacobian_index]


def _joint_speed_limits(
    robot, joint_indices: np.ndarray, config: CBFConfig
) -> np.ndarray:
    limits = np.full(joint_indices.shape, config.max_joint_speed_rad_s, dtype=float)
    properties = getattr(robot, "dof_properties", None)
    if properties is None:
        return limits
    try:
        candidate = np.asarray(properties["maxVelocity"], dtype=float)[joint_indices]
    except Exception:
        return limits
    valid = np.isfinite(candidate) & (candidate > 0.0)
    limits[valid] = np.minimum(limits[valid], candidate[valid])
    return limits


def _clip_joint_positions(
    robot, joint_indices: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    properties = getattr(robot, "dof_properties", None)
    if properties is None:
        return positions
    try:
        lower = np.asarray(properties["lower"], dtype=float)[joint_indices]
        upper = np.asarray(properties["upper"], dtype=float)[joint_indices]
    except Exception:
        return positions
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    result = positions.copy()
    result[valid] = np.clip(result[valid], lower[valid], upper[valid])
    return result


def _observation_vec3(
    observation: dict[str, np.ndarray], name: str
) -> np.ndarray | None:
    value = observation.get(name)
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if (
        array.size < 3
        or not np.all(np.isfinite(array[:3]))
        or float(np.linalg.norm(array[:3])) <= 1e-8
    ):
        return None
    return array[:3].copy()


def _finite_hand_velocity(dynamic_hand) -> np.ndarray | None:
    try:
        velocity = _to_numpy(
            getattr(dynamic_hand, "hand_velocity_filtered_mps")
        ).astype(float, copy=False)
    except (AttributeError, TypeError, ValueError):
        return None
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        return None
    return velocity.copy()


def _finite_closing_speed(dynamic_hand) -> float | None:
    try:
        value = _to_numpy(getattr(dynamic_hand, "closing_speed_mps")).astype(
            float, copy=False
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if value.shape != () or not math.isfinite(float(value)):
        return None
    return float(value)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
