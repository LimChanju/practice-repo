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
from typing import Any, Iterable, Mapping

import numpy as np


PHYSICAL_SAFETY_MODES = (
    "none",
    "rmpflow",
    "cbf",
    "rmpflow_cbf",
    "curobo",
    "curobo_cbf",
)

CBF_OBJECTIVE_MODES = (
    "joint_nominal",
    "task_consistent",
    "smooth_intervention",
    "phase_progress",
)

CBF_PHASE_PROGRESS_OBJECTIVE_SCHEMA = "phase_aware_one_sided_progress_v2"

INTENTIONAL_HUMAN_ABSENCE_CONTRACT = (
    "explicit_true_all_zero_tracking_mask_no_human_positions_geometry_or_"
    "constraints_cbf_inactive_nominal_action_v1"
)

_NOMINAL_FEASIBLE_PASSTHROUGH_ATOL_RADPS = 1e-12


def cbf_objective_schema(mode: str) -> str:
    return {
        "joint_nominal": "joint_nominal_projection_v1",
        "task_consistent": "task_consistent_quadratic_v1",
        "smooth_intervention": "smooth_intervention_reference_v1",
        "phase_progress": CBF_PHASE_PROGRESS_OBJECTIVE_SCHEMA,
    }.get(str(mode), "unknown")


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
    objective_mode: str = "joint_nominal"
    task_space_weight: float = 1.0
    task_yaw_length_scale_m_per_rad: float = 0.10
    joint_regularization_epsilon: float = 0.05
    correction_smoothness_weight: float = 1.0
    progress_retention_rho: float = 0.70
    progress_penalty_weight: float = 50.0
    progress_nominal_threshold_mps: float = 0.01
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
            self.task_space_weight,
            self.task_yaw_length_scale_m_per_rad,
            self.joint_regularization_epsilon,
            self.correction_smoothness_weight,
            self.progress_retention_rho,
            self.progress_penalty_weight,
            self.progress_nominal_threshold_mps,
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
        if self.objective_mode not in CBF_OBJECTIVE_MODES:
            raise ValueError(
                f"objective_mode must be one of {CBF_OBJECTIVE_MODES}"
            )
        if self.task_space_weight <= 0.0:
            raise ValueError("task_space_weight must be positive")
        if self.task_yaw_length_scale_m_per_rad <= 0.0:
            raise ValueError("task_yaw_length_scale_m_per_rad must be positive")
        if self.joint_regularization_epsilon <= 0.0:
            raise ValueError("joint_regularization_epsilon must be positive")
        if self.correction_smoothness_weight < 0.0:
            raise ValueError("correction_smoothness_weight must be non-negative")
        if not 0.0 <= self.progress_retention_rho <= 1.0:
            raise ValueError("progress_retention_rho must be in [0, 1]")
        if self.progress_penalty_weight < 0.0:
            raise ValueError("progress_penalty_weight must be non-negative")
        if self.progress_nominal_threshold_mps < 0.0:
            raise ValueError(
                "progress_nominal_threshold_mps must be non-negative"
            )
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
    objective_mode: str = "joint_nominal"
    objective_schema: str = "joint_nominal_projection_v1"
    objective_solver_fallback: bool = False
    task_space_weight: float = 0.0
    task_yaw_length_scale_m_per_rad: float = 0.0
    joint_regularization_epsilon: float = 0.0
    correction_smoothness_weight: float = 0.0
    progress_retention_rho: float = 0.0
    progress_penalty_weight: float = 0.0
    progress_nominal_threshold_mps: float = 0.0
    task_progress_active: bool = False
    task_progress_phase: str = "inactive"
    task_progress_source: str = "inactive"
    task_progress_gate_reason: str = "objective_not_phase_progress"
    task_progress_direction_world: tuple[float, ...] = ()
    task_progress_jacobian_row_m_per_rad: tuple[float, ...] = ()
    task_progress_nominal_mps: float = 0.0
    task_progress_A_mps: float = 0.0
    task_progress_retained_target_mps: float = 0.0
    task_progress_filtered_mps: float = 0.0
    task_progress_A_shortfall_mps: float = 0.0
    task_progress_shortfall_mps: float = 0.0
    task_progress_excess_over_nominal_mps: float = 0.0
    task_progress_base_objective: float = 0.0
    task_progress_penalty_objective: float = 0.0
    active_joint_indices: tuple[int, ...] = ()
    nominal_velocity_radps: tuple[float, ...] = ()
    objective_reference_velocity_radps: tuple[float, ...] = ()
    filtered_velocity_radps: tuple[float, ...] = ()
    previous_correction_radps: tuple[float, ...] = ()
    correction_radps: tuple[float, ...] = ()
    correction_rate_norm_radps2: float = 0.0
    task_velocity_nominal: tuple[float, ...] = ()
    task_velocity_filtered: tuple[float, ...] = ()
    task_velocity_error_norm: float = 0.0
    constraint_evidence: tuple[dict[str, Any], ...] = ()

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
    raw_surface_gap_m: float
    prediction_buffer_m: float
    effective_safe_gap_m: float
    barrier_value_m: float
    hand_velocity_world_mps: tuple[float, float, float]
    closing_speed_mps: float
    normal_world: tuple[float, float, float]
    closest_link: str
    closest_collider_path: str


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


@dataclass(frozen=True)
class _PhaseProgressSpecification:
    active: bool
    phase: str
    source: str
    gate_reason: str
    direction_world: tuple[float, ...] = ()
    jacobian_row_m_per_rad: tuple[float, ...] = ()

    @classmethod
    def inactive(
        cls,
        reason: str,
        *,
        phase: str = "inactive",
        source: str = "inactive",
    ) -> "_PhaseProgressSpecification":
        return cls(
            active=False,
            phase=str(phase),
            source=str(source),
            gate_reason=str(reason),
        )

    def disabled(self, reason: str) -> "_PhaseProgressSpecification":
        return _PhaseProgressSpecification(
            active=False,
            phase=self.phase,
            source=self.source,
            gate_reason=str(reason),
            direction_world=self.direction_world,
            jacobian_row_m_per_rad=self.jacobian_row_m_per_rad,
        )


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
        self._previous_correction: np.ndarray | None = None
        self._pending_correction: np.ndarray | None = None
        self.last_diagnostics = PhysicalSafetyDiagnostics(
            controller="cbf",
            objective_mode=self.config.objective_mode,
            objective_schema=cbf_objective_schema(self.config.objective_mode),
        )

    def reset(self) -> None:
        self._previous_correction = None
        self._pending_correction = None
        self.last_diagnostics = PhysicalSafetyDiagnostics(
            controller="cbf",
            objective_mode=self.config.objective_mode,
            objective_schema=cbf_objective_schema(self.config.objective_mode),
        )

    def notify_action_committed(self, arm_command_committed: bool) -> None:
        """Commit C's correction memory only when the arm command was applied.

        The legacy gripper merge sends a gripper-only action on open/close
        ticks.  A candidate CBF correction computed on such a tick must not
        become the temporal reference for the next solve.
        """

        if not isinstance(arm_command_committed, (bool, np.bool_)):
            raise ValueError("arm_command_committed must be an exact boolean")
        if bool(arm_command_committed):
            pending = getattr(self, "_pending_correction", None)
            if pending is not None:
                self._previous_correction = np.asarray(
                    pending, dtype=float
                ).copy()
        self._pending_correction = None

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
        task_progress_context: Mapping[str, Any] | None = None,
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
                task_progress_context=task_progress_context,
                human_valid_mask=human_valid_mask,
                intentional_human_absence=intentional_human_absence,
            )
        except Exception as exc:
            diagnostics = PhysicalSafetyDiagnostics(
                controller="cbf",
                objective_mode=self.config.objective_mode,
                objective_schema=cbf_objective_schema(
                    self.config.objective_mode
                ),
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
        task_progress_context: Mapping[str, Any] | None = None,
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
        previous_correction = self._previous_correction
        if (
            previous_correction is None
            or previous_correction.shape != nominal_velocity.shape
        ):
            previous_correction = np.zeros_like(nominal_velocity)
        if intentional_absence:
            previous_correction = np.zeros_like(nominal_velocity)
            self._previous_correction = previous_correction.copy()
        smooth_tail_requested = bool(
            self.config.objective_mode == "smooth_intervention"
            and not intentional_absence
            and np.linalg.norm(previous_correction) > 1e-8
        )
        if not constraints and not failures and not smooth_tail_requested:
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
                objective_mode=self.config.objective_mode,
                objective_schema=cbf_objective_schema(
                    self.config.objective_mode
                ),
                task_space_weight=float(self.config.task_space_weight),
                task_yaw_length_scale_m_per_rad=float(
                    self.config.task_yaw_length_scale_m_per_rad
                ),
                joint_regularization_epsilon=float(
                    self.config.joint_regularization_epsilon
                ),
                correction_smoothness_weight=float(
                    self.config.correction_smoothness_weight
                ),
                progress_retention_rho=float(
                    self.config.progress_retention_rho
                ),
                progress_penalty_weight=float(
                    self.config.progress_penalty_weight
                ),
                progress_nominal_threshold_mps=float(
                    self.config.progress_nominal_threshold_mps
                ),
                active_joint_indices=tuple(int(v) for v in joint_indices),
                nominal_velocity_radps=tuple(float(v) for v in nominal_velocity),
                objective_reference_velocity_radps=tuple(
                    float(v) for v in nominal_velocity
                ),
                filtered_velocity_radps=tuple(float(v) for v in nominal_velocity),
                previous_correction_radps=tuple(
                    float(v) for v in previous_correction
                ),
                correction_radps=tuple(0.0 for _ in nominal_velocity),
            )
            self._previous_correction = np.zeros_like(nominal_velocity)
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
        baseline_projection = _project_velocity_qp_detailed(
            nominal_velocity,
            matrix,
            bounds,
            velocity_lower,
            velocity_upper,
            max_iterations=self.config.projection_iterations,
            tolerance=self.config.projection_tolerance,
        )
        projection = baseline_projection
        objective_reference = nominal_velocity.copy()
        objective_solver_fallback = False
        task_jacobian = np.empty((0, nominal_velocity.size), dtype=float)
        task_velocity_nominal = np.empty(0, dtype=float)
        progress_specification = _PhaseProgressSpecification.inactive(
            "objective_not_phase_progress"
        )
        progress_nominal_mps = 0.0
        progress_A_mps = 0.0
        progress_retained_target_mps = 0.0
        if baseline_projection.feasible and not failures:
            if self.config.objective_mode == "task_consistent":
                task_jacobian = _task_consistency_jacobian(
                    jacobians,
                    body_names,
                    joint_indices,
                    yaw_length_scale_m_per_rad=(
                        self.config.task_yaw_length_scale_m_per_rad
                    ),
                )
                task_velocity_nominal = task_jacobian @ nominal_velocity
                hessian = (
                    float(self.config.task_space_weight)
                    * (task_jacobian.T @ task_jacobian)
                    + float(self.config.joint_regularization_epsilon)
                    * np.eye(nominal_velocity.size, dtype=float)
                )
                task_projection = _project_velocity_quadratic_detailed(
                    nominal_velocity=nominal_velocity,
                    hessian=hessian,
                    constraint_matrix=matrix,
                    lower_bounds=bounds,
                    velocity_lower=velocity_lower,
                    velocity_upper=velocity_upper,
                    feasible_seed=baseline_projection.velocity,
                    tolerance=self.config.projection_tolerance,
                    violation_before=baseline_projection.violation_before_mps,
                )
                if task_projection is None:
                    objective_solver_fallback = True
                else:
                    projection = task_projection
            elif self.config.objective_mode == "smooth_intervention":
                smoothness_weight = float(
                    self.config.correction_smoothness_weight
                )
                if smoothness_weight > 0.0:
                    objective_reference = nominal_velocity + (
                        smoothness_weight / (1.0 + smoothness_weight)
                    ) * previous_correction
                    projection = _project_velocity_qp_detailed(
                        objective_reference,
                        matrix,
                        bounds,
                        velocity_lower,
                        velocity_upper,
                        max_iterations=self.config.projection_iterations,
                        tolerance=self.config.projection_tolerance,
                    )
                    # This problem has the same feasible set as A.  If the
                    # alternate-reference solve cannot certify it, apply A's
                    # already-certified safe projection.
                    if not projection.feasible:
                        projection = baseline_projection
                        objective_solver_fallback = True
            elif self.config.objective_mode == "phase_progress":
                progress_specification = _phase_progress_specification(
                    jacobians=jacobians,
                    body_names=body_names,
                    active_joint_indices=joint_indices,
                    observation=observation,
                    context=task_progress_context,
                )
                if progress_specification.active:
                    progress_row = np.asarray(
                        progress_specification.jacobian_row_m_per_rad,
                        dtype=float,
                    )
                    progress_nominal_mps = float(
                        progress_row @ nominal_velocity
                    )
                    progress_A_mps = float(
                        progress_row @ baseline_projection.velocity
                    )
                    if progress_nominal_mps <= float(
                        self.config.progress_nominal_threshold_mps
                    ):
                        progress_specification = progress_specification.disabled(
                            "nominal_progress_below_threshold"
                        )
                    else:
                        progress_retained_target_mps = float(
                            self.config.progress_retention_rho
                        ) * progress_nominal_mps
                    if (
                        progress_specification.active
                        and float(self.config.progress_penalty_weight) <= 0.0
                    ):
                        progress_specification = progress_specification.disabled(
                            "zero_progress_penalty_weight"
                        )
                    elif progress_specification.active:
                        progress_projection = (
                            _project_velocity_phase_progress_detailed(
                                nominal_velocity=nominal_velocity,
                                progress_row=progress_row,
                                retained_target_mps=progress_retained_target_mps,
                                penalty_weight=float(
                                    self.config.progress_penalty_weight
                                ),
                                constraint_matrix=matrix,
                                lower_bounds=bounds,
                                velocity_lower=velocity_lower,
                                velocity_upper=velocity_upper,
                                feasible_seed=baseline_projection.velocity,
                                tolerance=self.config.projection_tolerance,
                                violation_before=(
                                    baseline_projection.violation_before_mps
                                ),
                            )
                        )
                        if progress_projection is None:
                            objective_solver_fallback = True
                            progress_specification = (
                                progress_specification.disabled(
                                    "phase_progress_solver_fallback_to_A"
                                )
                            )
                        else:
                            projection = progress_projection
        filtered_velocity = projection.velocity.copy()
        slack = projection.slack_mps
        feasible = projection.feasible
        # "before" always means the original nominal command, even when C
        # projects an alternate temporal reference.
        before = baseline_projection.violation_before_mps
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
            if (
                self.config.objective_mode == "smooth_intervention"
                and not constraints
                and not failures
                and not fallback_applied
                and intervention > 1e-8
            ):
                status = "smooth_intervention_tail"
        correction = filtered_velocity - nominal_velocity
        correction_rate_norm = float(
            np.linalg.norm(correction - previous_correction) / dt_s
        )
        self._pending_correction = correction.copy()
        if task_jacobian.size:
            task_velocity_filtered = task_jacobian @ filtered_velocity
            task_velocity_error_norm = float(
                np.linalg.norm(task_velocity_filtered - task_velocity_nominal)
            )
        else:
            task_velocity_filtered = np.empty(0, dtype=float)
            task_velocity_error_norm = 0.0
        progress_row = np.asarray(
            progress_specification.jacobian_row_m_per_rad, dtype=float
        )
        progress_filtered_mps = (
            float(progress_row @ filtered_velocity)
            if progress_row.size == filtered_velocity.size
            else 0.0
        )
        progress_shortfall_mps = (
            max(0.0, progress_retained_target_mps - progress_filtered_mps)
            if progress_specification.active
            else 0.0
        )
        progress_A_shortfall_mps = (
            max(0.0, progress_retained_target_mps - progress_A_mps)
            if progress_retained_target_mps > 0.0
            else 0.0
        )
        progress_delta = filtered_velocity - nominal_velocity
        progress_base_objective = (
            0.5 * float(progress_delta @ progress_delta)
            if progress_specification.active
            else 0.0
        )
        progress_penalty_objective = (
            float(self.config.progress_penalty_weight)
            * progress_shortfall_mps
            * progress_shortfall_mps
            if progress_specification.active
            else 0.0
        )
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
            objective_mode=self.config.objective_mode,
            objective_schema=cbf_objective_schema(self.config.objective_mode),
            objective_solver_fallback=bool(objective_solver_fallback),
            task_space_weight=float(self.config.task_space_weight),
            task_yaw_length_scale_m_per_rad=float(
                self.config.task_yaw_length_scale_m_per_rad
            ),
            joint_regularization_epsilon=float(
                self.config.joint_regularization_epsilon
            ),
            correction_smoothness_weight=float(
                self.config.correction_smoothness_weight
            ),
            progress_retention_rho=float(self.config.progress_retention_rho),
            progress_penalty_weight=float(self.config.progress_penalty_weight),
            progress_nominal_threshold_mps=float(
                self.config.progress_nominal_threshold_mps
            ),
            task_progress_active=bool(progress_specification.active),
            task_progress_phase=str(progress_specification.phase),
            task_progress_source=str(progress_specification.source),
            task_progress_gate_reason=str(progress_specification.gate_reason),
            task_progress_direction_world=tuple(
                float(value) for value in progress_specification.direction_world
            ),
            task_progress_jacobian_row_m_per_rad=tuple(
                float(value)
                for value in progress_specification.jacobian_row_m_per_rad
            ),
            task_progress_nominal_mps=float(progress_nominal_mps),
            task_progress_A_mps=float(progress_A_mps),
            task_progress_retained_target_mps=float(
                progress_retained_target_mps
            ),
            task_progress_filtered_mps=float(progress_filtered_mps),
            task_progress_A_shortfall_mps=float(
                progress_A_shortfall_mps
            ),
            task_progress_shortfall_mps=float(progress_shortfall_mps),
            task_progress_excess_over_nominal_mps=max(
                0.0, float(progress_filtered_mps - progress_nominal_mps)
            ),
            task_progress_base_objective=float(progress_base_objective),
            task_progress_penalty_objective=float(
                progress_penalty_objective
            ),
            active_joint_indices=tuple(int(v) for v in joint_indices),
            nominal_velocity_radps=tuple(float(v) for v in nominal_velocity),
            objective_reference_velocity_radps=tuple(
                float(v) for v in objective_reference
            ),
            filtered_velocity_radps=tuple(float(v) for v in filtered_velocity),
            previous_correction_radps=tuple(
                float(v) for v in previous_correction
            ),
            correction_radps=tuple(float(v) for v in correction),
            correction_rate_norm_radps2=correction_rate_norm,
            task_velocity_nominal=tuple(float(v) for v in task_velocity_nominal),
            task_velocity_filtered=tuple(float(v) for v in task_velocity_filtered),
            task_velocity_error_norm=task_velocity_error_norm,
            constraint_evidence=tuple(
                _constraint_evidence_payload(constraint, nominal_velocity, filtered_velocity)
                for constraint in constraints
            ),
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
                raw_surface_gap_m=gap_m,
                prediction_buffer_m=float(prediction_buffer),
                effective_safe_gap_m=float(safe_gap),
                barrier_value_m=float(barrier_value),
                hand_velocity_world_mps=tuple(float(v) for v in hand_velocity),
                closing_speed_mps=float(closing_speed),
                normal_world=tuple(float(v) for v in normal),
                closest_link=str(hand_result.closest_link),
                closest_collider_path=str(
                    getattr(hand_result, "closest_collider_path", "")
                ),
            )


def _task_consistency_jacobian(
    jacobians: np.ndarray,
    body_names: tuple[str, ...],
    active_joint_indices: np.ndarray,
    *,
    yaw_length_scale_m_per_rad: float,
) -> np.ndarray:
    """Return BC's controlled task dimensions: world XYZ plus world yaw.

    The frozen BC command is ``[dx, dy, dz, dyaw, gripper]``.  Gripper is a
    separate actuator, so the arm objective preserves the corresponding 4-D
    geometric velocity and deliberately does not invent roll/pitch targets.
    ``panda_hand`` is the movable body coincident with the Franka flange in
    the Isaac 4.5 asset and is the closest articulation Jacobian available to
    RMPFlow's configured gripper target frame.
    """

    hand_jacobian = _body_jacobian(jacobians, body_names, "panda_hand")
    hand_jacobian = hand_jacobian[:, active_joint_indices]
    return np.vstack(
        (
            hand_jacobian[:3],
            float(yaw_length_scale_m_per_rad) * hand_jacobian[5:6],
        )
    )


def _phase_progress_specification(
    *,
    jacobians: np.ndarray,
    body_names: tuple[str, ...],
    active_joint_indices: np.ndarray,
    observation: Mapping[str, np.ndarray],
    context: Mapping[str, Any] | None,
) -> _PhaseProgressSpecification:
    """Build B v2's one-dimensional, phase-aware progress direction.

    Recovery-stage context takes precedence over the aliased controller event
    and preserves progress toward the frozen bridge's actual target.  The
    objective is disabled in grasp-confirmation/release events.  For an
    attached cube, the Panda hand translational Jacobian is logged and used as
    an explicit rigid-attachment proxy rather than claiming a cube Jacobian.
    """

    if not isinstance(context, Mapping):
        return _PhaseProgressSpecification.inactive("missing_progress_context")
    if bool(context.get("recovery_control_active", False)):
        ee_position = _mapping_vec3(observation, "ee_pos")
        target_position = _mapping_vec3(context, "target_position_world_m")
        recovery_stage = str(context.get("recovery_stage", "unknown"))
        if ee_position is None or target_position is None:
            return _PhaseProgressSpecification.inactive(
                "missing_recovery_target",
                phase=f"recovery:{recovery_stage}",
                source="fixed_recovery_target_direction",
            )
        phase = f"recovery:{recovery_stage}"
        source = "fixed_recovery_target_direction"
        direction = target_position - ee_position
        event = -1
    else:
        try:
            event = int(context["controller_event"])
        except (KeyError, TypeError, ValueError):
            return _PhaseProgressSpecification.inactive("invalid_controller_event")

    if event == -1:
        pass
    elif event in (0, 1):
        phase = "approach"
        source = "ee_to_cube"
        direction = _mapping_vec3(observation, "ee_to_cube")
    elif event in (2, 3):
        return _PhaseProgressSpecification.inactive(
            "unconfirmed_grasp_uses_A_objective",
            phase="grasp_unconfirmed",
            source="none",
        )
    elif event == 4:
        phase = "lift"
        if not _observation_flag(observation, "has_grasped_cube"):
            return _PhaseProgressSpecification.inactive(
                "lift_without_attachment_uses_A_objective",
                phase=phase,
                source="none",
            )
        ee_position = _mapping_vec3(observation, "ee_pos")
        target_position = _mapping_vec3(context, "target_position_world_m")
        if ee_position is None or target_position is None:
            direction = np.asarray((0.0, 0.0, 1.0), dtype=float)
            source = "world_up_fallback_missing_lift_target"
        else:
            direction = target_position - ee_position
            if (
                not np.all(np.isfinite(direction))
                or float(np.linalg.norm(direction)) <= 1e-8
            ):
                direction = np.asarray((0.0, 0.0, 1.0), dtype=float)
                source = "world_up_fallback_degenerate_lift_target"
            else:
                source = "controller_target_direction"
    elif event in (5, 6):
        phase = "transport" if event == 5 else "place"
        if not _observation_flag(observation, "has_grasped_cube"):
            return _PhaseProgressSpecification.inactive(
                "transport_without_attachment_uses_A_objective",
                phase=phase,
                source="none",
            )
        if event == 6 and bool(context.get("place_spatially_ready", False)):
            return _PhaseProgressSpecification.inactive(
                "place_spatially_ready_uses_A_objective",
                phase=phase,
                source="none",
            )
        direction = _mapping_vec3(observation, "cube_to_place_target")
        source = "attached_cube_goal_ee_jacobian_proxy"
    else:
        return _PhaseProgressSpecification.inactive(
            "release_or_terminal_uses_A_objective",
            phase="release" if event >= 7 else "unknown",
            source="none",
        )

    if direction is None:
        return _PhaseProgressSpecification.inactive(
            "missing_phase_direction", phase=phase, source=source
        )
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-8:
        return _PhaseProgressSpecification.inactive(
            "degenerate_phase_direction", phase=phase, source=source
        )
    unit_direction = np.asarray(direction, dtype=float) / norm
    hand_jacobian = _body_jacobian(jacobians, body_names, "panda_hand")
    translational = np.asarray(
        hand_jacobian[:3, active_joint_indices], dtype=float
    )
    progress_row = unit_direction @ translational
    if (
        progress_row.shape != active_joint_indices.shape
        or not np.all(np.isfinite(progress_row))
        or float(np.linalg.norm(progress_row)) <= 1e-10
    ):
        return _PhaseProgressSpecification.inactive(
            "invalid_phase_progress_jacobian", phase=phase, source=source
        )
    return _PhaseProgressSpecification(
        active=True,
        phase=phase,
        source=source,
        gate_reason="active",
        direction_world=tuple(float(value) for value in unit_direction),
        jacobian_row_m_per_rad=tuple(float(value) for value in progress_row),
    )


def _mapping_vec3(
    mapping: Mapping[str, Any], name: str
) -> np.ndarray | None:
    value = mapping.get(name)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3].copy()


def _observation_flag(observation: Mapping[str, Any], name: str) -> bool:
    value = observation.get(name)
    if value is None:
        return False
    try:
        scalar = float(np.asarray(value, dtype=float).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return False
    return bool(math.isfinite(scalar) and scalar > 0.5)


def _project_velocity_quadratic_detailed(
    *,
    nominal_velocity: np.ndarray,
    hessian: np.ndarray,
    constraint_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    feasible_seed: np.ndarray,
    tolerance: float,
    violation_before: float,
) -> _VelocityProjectionResult | None:
    """Minimize a positive-definite quadratic over A's unchanged safe set."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except (ImportError, ModuleNotFoundError):
        return None

    nominal = np.asarray(nominal_velocity, dtype=float)
    matrix = np.asarray(constraint_matrix, dtype=float)
    bounds = np.asarray(lower_bounds, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    seed = np.asarray(feasible_seed, dtype=float)
    if hessian.shape != (nominal.size, nominal.size):
        raise ValueError("task-consistent Hessian has an invalid shape")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("task-consistent Hessian must be finite")

    def objective(value: np.ndarray) -> float:
        delta = value - nominal
        return 0.5 * float(delta @ hessian @ delta)

    def gradient(value: np.ndarray) -> np.ndarray:
        return hessian @ (value - nominal)

    try:
        result = minimize(
            objective,
            seed,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(velocity_lower, velocity_upper),
            constraints=LinearConstraint(matrix, bounds, np.inf),
            options={
                "maxiter": 200,
                "ftol": min(1e-12, max(1e-15, tolerance * tolerance)),
                "disp": False,
            },
        )
    except Exception:
        return None
    candidate = np.asarray(result.x, dtype=float)
    if candidate.shape != nominal.shape or not np.all(np.isfinite(candidate)):
        return None
    candidate = np.clip(candidate, velocity_lower, velocity_upper)
    violation_after = _max_halfspace_violation(matrix, bounds, candidate)
    if not bool(result.success) or violation_after > tolerance:
        return None
    return _VelocityProjectionResult(
        velocity=candidate,
        slack_mps=0.0,
        feasible=True,
        violation_before_mps=float(violation_before),
        violation_after_mps=float(violation_after),
        solver_converged=True,
        infeasibility_proven=False,
        relaxed_solution_available=False,
        solver_backend="scipy_slsqp_task_consistent",
        status="solved_task_consistent",
    )


def _project_velocity_phase_progress_detailed(
    *,
    nominal_velocity: np.ndarray,
    progress_row: np.ndarray,
    retained_target_mps: float,
    penalty_weight: float,
    constraint_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    feasible_seed: np.ndarray,
    tolerance: float,
    violation_before: float,
) -> _VelocityProjectionResult | None:
    """Minimize joint deviation plus a one-sided progress-shortfall penalty."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except (ImportError, ModuleNotFoundError):
        return None

    nominal = np.asarray(nominal_velocity, dtype=float)
    row = np.asarray(progress_row, dtype=float).reshape(-1)
    matrix = np.asarray(constraint_matrix, dtype=float)
    bounds = np.asarray(lower_bounds, dtype=float)
    seed = np.asarray(feasible_seed, dtype=float)
    retained_target = float(retained_target_mps)
    weight = float(penalty_weight)
    if row.shape != nominal.shape or not np.all(np.isfinite(row)):
        raise ValueError("phase-progress Jacobian row has an invalid shape")
    if not math.isfinite(retained_target) or not math.isfinite(weight) or weight < 0.0:
        raise ValueError("phase-progress objective parameters must be finite")

    def objective(value: np.ndarray) -> float:
        delta = value - nominal
        shortfall = max(0.0, retained_target - float(row @ value))
        return 0.5 * float(delta @ delta) + weight * shortfall * shortfall

    def gradient(value: np.ndarray) -> np.ndarray:
        shortfall = max(0.0, retained_target - float(row @ value))
        return (value - nominal) - (2.0 * weight * shortfall) * row

    try:
        result = minimize(
            objective,
            seed,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(velocity_lower, velocity_upper),
            constraints=LinearConstraint(matrix, bounds, np.inf),
            options={
                "maxiter": 200,
                "ftol": min(1e-12, max(1e-15, tolerance * tolerance)),
                "disp": False,
            },
        )
    except Exception:
        return None
    candidate = np.asarray(result.x, dtype=float)
    if candidate.shape != nominal.shape or not np.all(np.isfinite(candidate)):
        return None
    candidate = np.clip(candidate, velocity_lower, velocity_upper)
    violation_after = _max_halfspace_violation(matrix, bounds, candidate)
    if not bool(result.success) or violation_after > tolerance:
        return None
    return _VelocityProjectionResult(
        velocity=candidate,
        slack_mps=0.0,
        feasible=True,
        violation_before_mps=float(violation_before),
        violation_after_mps=float(violation_after),
        solver_converged=True,
        infeasibility_proven=False,
        relaxed_solution_available=False,
        solver_backend="scipy_slsqp_phase_progress",
        status="solved_phase_progress",
    )


def _constraint_evidence_payload(
    constraint: _CBFConstraint,
    nominal_velocity: np.ndarray,
    filtered_velocity: np.ndarray,
) -> dict[str, Any]:
    row = np.asarray(constraint.jacobian_row, dtype=float)
    nominal_lhs = float(row @ nominal_velocity)
    filtered_lhs = float(row @ filtered_velocity)
    return {
        "hand": constraint.hand,
        "closest_link": constraint.closest_link,
        "closest_collider_path": constraint.closest_collider_path,
        "raw_surface_gap_m": float(constraint.raw_surface_gap_m),
        "prediction_buffer_m": float(constraint.prediction_buffer_m),
        "effective_safe_gap_m": float(constraint.effective_safe_gap_m),
        "barrier_value_m": float(constraint.barrier_value_m),
        "buffered_current_gap_m": float(constraint.buffered_current_gap_m),
        "closing_speed_mps": float(constraint.closing_speed_mps),
        "hand_velocity_world_mps": list(constraint.hand_velocity_world_mps),
        "normal_world": list(constraint.normal_world),
        "jacobian_row_m_per_rad": row.tolist(),
        "lower_bound_mps": float(constraint.lower_bound_mps),
        "nominal_lhs_mps": nominal_lhs,
        "filtered_lhs_mps": filtered_lhs,
        "nominal_residual_mps": nominal_lhs - float(constraint.lower_bound_mps),
        "filtered_residual_mps": filtered_lhs - float(constraint.lower_bound_mps),
    }


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
