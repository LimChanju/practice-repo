"""Surface-referenced crossing corridor and read-only protocol geometry checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CrossingCorridor:
    start_world_m: np.ndarray
    end_world_m: np.ndarray
    center_world_m: np.ndarray
    tangent_world: np.ndarray
    outward_normal_world: np.ndarray
    planned_minimum_surface_gap_m: float
    closest_link: str
    closest_collider_path: str
    calibration_iterations: int
    geometry_query_semantics: str = "physx_protected_surface_to_hand_sphere"

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_world_m": self.start_world_m.tolist(),
            "end_world_m": self.end_world_m.tolist(),
            "center_world_m": self.center_world_m.tolist(),
            "tangent_world": self.tangent_world.tolist(),
            "outward_normal_world": self.outward_normal_world.tolist(),
            "planned_minimum_surface_gap_m": self.planned_minimum_surface_gap_m,
            "closest_link": self.closest_link,
            "closest_collider_path": self.closest_collider_path,
            "calibration_iterations": self.calibration_iterations,
            "geometry_query_semantics": self.geometry_query_semantics,
        }


NOMINAL_BC_SWEEP_SEMANTICS = (
    "bc_only_cue_latched_then_task_advancing_time_aligned_protected_surface_sweep_v1"
)


@dataclass(frozen=True)
class NominalCorridorCandidateAudit:
    """Geometry-query audit for one outward-normal corridor offset."""

    normal_offset_m: float
    geometry_valid: bool
    nominal_minimum_surface_gap_m: float | None
    nominal_minimum_simulation_time_s: float | None
    nominal_minimum_sample_index: int | None
    nominal_minimum_progress_fraction: float | None
    closest_link: str
    closest_collider_path: str
    crossing_start_surface_gap_m: float | None
    crossing_start_simulation_time_s: float | None
    crossing_start_sample_index: int | None
    crossing_start_closest_link: str
    crossing_start_closest_collider_path: str
    minimum_pre_cue_start_gap_m: float | None
    minimum_pre_cue_simulation_time_s: float | None
    minimum_pre_cue_sample_index: int | None
    pre_cue_sample_count: int
    crossing_sample_count: int
    invalid_geometry_query_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_offset_m": self.normal_offset_m,
            "geometry_valid": self.geometry_valid,
            "nominal_minimum_surface_gap_m": (
                self.nominal_minimum_surface_gap_m
            ),
            "nominal_minimum_simulation_time_s": (
                self.nominal_minimum_simulation_time_s
            ),
            "nominal_minimum_sample_index": self.nominal_minimum_sample_index,
            "nominal_minimum_progress_fraction": (
                self.nominal_minimum_progress_fraction
            ),
            "closest_link": self.closest_link,
            "closest_collider_path": self.closest_collider_path,
            "crossing_start_surface_gap_m": self.crossing_start_surface_gap_m,
            "crossing_start_simulation_time_s": (
                self.crossing_start_simulation_time_s
            ),
            "crossing_start_sample_index": self.crossing_start_sample_index,
            "crossing_start_closest_link": self.crossing_start_closest_link,
            "crossing_start_closest_collider_path": (
                self.crossing_start_closest_collider_path
            ),
            "minimum_pre_cue_start_gap_m": self.minimum_pre_cue_start_gap_m,
            "minimum_pre_cue_simulation_time_s": (
                self.minimum_pre_cue_simulation_time_s
            ),
            "minimum_pre_cue_sample_index": self.minimum_pre_cue_sample_index,
            "pre_cue_sample_count": self.pre_cue_sample_count,
            "crossing_sample_count": self.crossing_sample_count,
            "invalid_geometry_query_count": self.invalid_geometry_query_count,
        }


@dataclass(frozen=True)
class NominalCorridorSweepAudit:
    """Selected nominal-BC corridor and its time-aligned query provenance."""

    semantics: str
    phase: str
    severity: str
    hand: str
    crossing_direction: str
    speed_m_s: float
    cue_lead_s: float
    corridor_length_m: float
    target_gap_m: float
    allowed_gap_minimum_m: float
    allowed_gap_maximum_m: float
    selected_normal_offset_m: float
    nominal_minimum_surface_gap_m: float
    nominal_minimum_simulation_time_s: float
    nominal_minimum_sample_index: int
    nominal_minimum_progress_fraction: float
    closest_link: str
    closest_collider_path: str
    crossing_start_surface_gap_m: float
    crossing_start_query_simulation_time_s: float
    crossing_start_sample_index: int
    crossing_start_closest_link: str
    crossing_start_closest_collider_path: str
    minimum_pre_cue_start_gap_m: float | None
    required_minimum_pre_cue_start_gap_m: float | None
    minimum_pre_cue_simulation_time_s: float | None
    minimum_pre_cue_sample_index: int | None
    phase_start_simulation_time_s: float
    crossing_start_simulation_time_s: float
    crossing_end_simulation_time_s: float
    first_sample_simulation_time_s: float
    last_sample_simulation_time_s: float
    first_crossing_sample_simulation_time_s: float
    pre_cue_sample_count: int
    crossing_sample_count: int
    time_sample_count: int
    geometry_query_count: int
    candidate_count: int
    pre_cue_coverage_fraction: float
    crossing_coverage_fraction: float
    maximum_observed_sample_interval_s: float
    maximum_allowed_sample_interval_s: float
    geometry_valid: bool
    candidate_results: tuple[NominalCorridorCandidateAudit, ...]
    corridor: CrossingCorridor

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "phase": self.phase,
            "severity": self.severity,
            "hand": self.hand,
            "crossing_direction": self.crossing_direction,
            "speed_m_s": self.speed_m_s,
            "cue_lead_s": self.cue_lead_s,
            "corridor_length_m": self.corridor_length_m,
            "target_gap_m": self.target_gap_m,
            "allowed_gap_range_m": [
                self.allowed_gap_minimum_m,
                self.allowed_gap_maximum_m,
            ],
            "selected_normal_offset_m": self.selected_normal_offset_m,
            "nominal_minimum_surface_gap_m": (
                self.nominal_minimum_surface_gap_m
            ),
            "nominal_minimum_simulation_time_s": (
                self.nominal_minimum_simulation_time_s
            ),
            "nominal_minimum_sample_index": self.nominal_minimum_sample_index,
            "nominal_minimum_progress_fraction": (
                self.nominal_minimum_progress_fraction
            ),
            "closest_link": self.closest_link,
            "closest_collider_path": self.closest_collider_path,
            "crossing_start_surface_gap_m": self.crossing_start_surface_gap_m,
            "crossing_start_query_simulation_time_s": (
                self.crossing_start_query_simulation_time_s
            ),
            "crossing_start_sample_index": self.crossing_start_sample_index,
            "crossing_start_closest_link": self.crossing_start_closest_link,
            "crossing_start_closest_collider_path": (
                self.crossing_start_closest_collider_path
            ),
            "minimum_pre_cue_start_gap_m": self.minimum_pre_cue_start_gap_m,
            "required_minimum_pre_cue_start_gap_m": (
                self.required_minimum_pre_cue_start_gap_m
            ),
            "minimum_pre_cue_simulation_time_s": (
                self.minimum_pre_cue_simulation_time_s
            ),
            "minimum_pre_cue_sample_index": self.minimum_pre_cue_sample_index,
            "coverage": {
                "phase_start_simulation_time_s": (
                    self.phase_start_simulation_time_s
                ),
                "crossing_start_simulation_time_s": (
                    self.crossing_start_simulation_time_s
                ),
                "crossing_end_simulation_time_s": (
                    self.crossing_end_simulation_time_s
                ),
                "first_sample_simulation_time_s": (
                    self.first_sample_simulation_time_s
                ),
                "last_sample_simulation_time_s": (
                    self.last_sample_simulation_time_s
                ),
                "first_crossing_sample_simulation_time_s": (
                    self.first_crossing_sample_simulation_time_s
                ),
                "pre_cue_sample_count": self.pre_cue_sample_count,
                "crossing_sample_count": self.crossing_sample_count,
                "time_sample_count": self.time_sample_count,
                "geometry_query_count": self.geometry_query_count,
                "candidate_count": self.candidate_count,
                "pre_cue_coverage_fraction": self.pre_cue_coverage_fraction,
                "crossing_coverage_fraction": self.crossing_coverage_fraction,
                "maximum_observed_sample_interval_s": (
                    self.maximum_observed_sample_interval_s
                ),
                "maximum_allowed_sample_interval_s": (
                    self.maximum_allowed_sample_interval_s
                ),
            },
            "geometry_valid": self.geometry_valid,
            "candidate_results": [
                result.as_dict() for result in self.candidate_results
            ],
            "corridor": self.corridor.as_dict(),
        }


@dataclass
class _MutableCandidateSweep:
    normal_offset_m: float
    corridor: CrossingCorridor
    nominal_minimum_surface_gap_m: float = math.inf
    nominal_minimum_simulation_time_s: float | None = None
    nominal_minimum_sample_index: int | None = None
    nominal_minimum_progress_fraction: float | None = None
    closest_link: str = ""
    closest_collider_path: str = ""
    crossing_start_surface_gap_m: float | None = None
    crossing_start_simulation_time_s: float | None = None
    crossing_start_sample_index: int | None = None
    crossing_start_closest_link: str = ""
    crossing_start_closest_collider_path: str = ""
    minimum_pre_cue_start_gap_m: float = math.inf
    minimum_pre_cue_simulation_time_s: float | None = None
    minimum_pre_cue_sample_index: int | None = None
    pre_cue_sample_count: int = 0
    crossing_sample_count: int = 0
    invalid_geometry_query_count: int = 0

    @property
    def geometry_valid(self) -> bool:
        return self.invalid_geometry_query_count == 0

    def immutable_audit(self) -> NominalCorridorCandidateAudit:
        nominal_gap = (
            None
            if not math.isfinite(self.nominal_minimum_surface_gap_m)
            else float(self.nominal_minimum_surface_gap_m)
        )
        pre_cue_gap = (
            None
            if not math.isfinite(self.minimum_pre_cue_start_gap_m)
            else float(self.minimum_pre_cue_start_gap_m)
        )
        return NominalCorridorCandidateAudit(
            normal_offset_m=self.normal_offset_m,
            geometry_valid=self.geometry_valid,
            nominal_minimum_surface_gap_m=nominal_gap,
            nominal_minimum_simulation_time_s=(
                self.nominal_minimum_simulation_time_s
            ),
            nominal_minimum_sample_index=self.nominal_minimum_sample_index,
            nominal_minimum_progress_fraction=(
                self.nominal_minimum_progress_fraction
            ),
            closest_link=self.closest_link,
            closest_collider_path=self.closest_collider_path,
            crossing_start_surface_gap_m=self.crossing_start_surface_gap_m,
            crossing_start_simulation_time_s=(
                self.crossing_start_simulation_time_s
            ),
            crossing_start_sample_index=self.crossing_start_sample_index,
            crossing_start_closest_link=self.crossing_start_closest_link,
            crossing_start_closest_collider_path=(
                self.crossing_start_closest_collider_path
            ),
            minimum_pre_cue_start_gap_m=pre_cue_gap,
            minimum_pre_cue_simulation_time_s=(
                self.minimum_pre_cue_simulation_time_s
            ),
            minimum_pre_cue_sample_index=self.minimum_pre_cue_sample_index,
            pre_cue_sample_count=self.pre_cue_sample_count,
            crossing_sample_count=self.crossing_sample_count,
            invalid_geometry_query_count=self.invalid_geometry_query_count,
        )


class NominalBCCorridorSweep:
    """Accumulate a hypothetical crossing against an advancing BC-only run.

    ``observe`` must be called once per nominal BC control sample after the
    target phase fires.  It queries the *current* protected robot geometry at
    the hand position prescribed by each candidate corridor.  The cue-lead
    interval holds the hypothetical hand at the start marker and is audited
    separately; only samples at or after the START time determine severity.

    No actual tracked-hand position or actual protected gap is accepted by
    this API.  In particular, a threat corridor is selected solely from this
    time-aligned nominal-BC sweep, before a BC+CBF trial is executed.
    """

    def __init__(
        self,
        *,
        phase: str,
        base_corridor: CrossingCorridor,
        crossing_direction: str,
        speed_m_s: float,
        cue_lead_s: float,
        corridor_length_m: float,
        hand: str,
        candidate_normal_offsets_m: Sequence[float],
        phase_start_simulation_time_s: float,
        maximum_sample_interval_s: float,
    ) -> None:
        self.phase = str(phase)
        if not self.phase:
            raise ValueError("phase must be non-empty")
        self.hand = str(hand)
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        self.crossing_direction = str(crossing_direction)
        if self.crossing_direction not in {
            "left_to_right",
            "right_to_left",
        }:
            raise ValueError("unsupported crossing_direction")
        self.speed_m_s = _finite_positive(speed_m_s, "speed_m_s")
        self.corridor_length_m = _finite_positive(
            corridor_length_m, "corridor_length_m"
        )
        self.cue_lead_s = float(cue_lead_s)
        if not math.isfinite(self.cue_lead_s) or self.cue_lead_s < 0.0:
            raise ValueError("cue_lead_s must be finite and non-negative")
        self.phase_start_simulation_time_s = float(
            phase_start_simulation_time_s
        )
        if not math.isfinite(self.phase_start_simulation_time_s):
            raise ValueError("phase_start_simulation_time_s must be finite")
        self.maximum_sample_interval_s = _finite_positive(
            maximum_sample_interval_s, "maximum_sample_interval_s"
        )

        center = _point3(base_corridor.center_world_m, "base corridor center")
        normal = _unit_vector(
            base_corridor.outward_normal_world,
            "base corridor outward normal",
        )
        tangent = _unit_vector(
            base_corridor.tangent_world, "base corridor tangent"
        )
        tangent_component = float(np.dot(tangent, normal))
        if abs(tangent_component) > 1e-5:
            raise ValueError(
                "base corridor tangent and outward normal must be orthogonal"
            )
        offsets = tuple(
            sorted(float(value) for value in candidate_normal_offsets_m)
        )
        if not offsets:
            raise ValueError("candidate_normal_offsets_m must not be empty")
        if any(not math.isfinite(value) for value in offsets):
            raise ValueError("candidate normal offsets must be finite")
        if len(set(offsets)) != len(offsets):
            raise ValueError("candidate normal offsets must be unique")

        self._candidates = tuple(
            _MutableCandidateSweep(
                normal_offset_m=offset,
                corridor=_offset_corridor(
                    base_corridor,
                    center=center + normal * offset,
                    tangent=tangent,
                    normal=normal,
                    length=self.corridor_length_m,
                    direction=self.crossing_direction,
                ),
            )
            for offset in offsets
        )
        self._sample_times_s: list[float] = []
        self._first_crossing_sample_time_s: float | None = None
        self._crossing_sample_count = 0
        self._pre_cue_sample_count = 0
        self._end_sample_observed = False

    @property
    def crossing_start_simulation_time_s(self) -> float:
        return self.phase_start_simulation_time_s + self.cue_lead_s

    @property
    def crossing_end_simulation_time_s(self) -> float:
        return (
            self.crossing_start_simulation_time_s
            + self.corridor_length_m / self.speed_m_s
        )

    @property
    def time_sample_count(self) -> int:
        return len(self._sample_times_s)

    def observe(
        self, *, simulation_time_s: float, safety_geometry: Any
    ) -> bool:
        """Query every candidate at one current nominal-BC geometry state.

        Returns ``False`` only when the sweep already contains its first sample
        at or beyond the scheduled crossing end.  Earlier/duplicate time stamps
        are rejected because they cannot constitute a time-aligned rollout.
        """

        now = float(simulation_time_s)
        if not math.isfinite(now):
            raise ValueError("simulation_time_s must be finite")
        if self._end_sample_observed:
            return False
        if now < self.phase_start_simulation_time_s:
            raise ValueError("sample precedes phase_start_simulation_time_s")
        if self._sample_times_s and now <= self._sample_times_s[-1]:
            raise ValueError("nominal sweep sample times must strictly increase")

        elapsed_crossing_s = now - self.crossing_start_simulation_time_s
        is_crossing = elapsed_crossing_s >= 0.0
        progress = (
            float(
                np.clip(
                    elapsed_crossing_s
                    * self.speed_m_s
                    / self.corridor_length_m,
                    0.0,
                    1.0,
                )
            )
            if is_crossing
            else 0.0
        )
        sample_index = len(self._sample_times_s)
        self._sample_times_s.append(now)
        if is_crossing:
            self._crossing_sample_count += 1
            if self._first_crossing_sample_time_s is None:
                self._first_crossing_sample_time_s = now
        else:
            self._pre_cue_sample_count += 1

        for candidate in self._candidates:
            position = (
                candidate.corridor.start_world_m
                + progress
                * (
                    candidate.corridor.end_world_m
                    - candidate.corridor.start_world_m
                )
            )
            result = safety_geometry.evaluate_hand(self.hand, position)
            gap = float(getattr(result, "surface_gap_m", math.inf))
            valid = bool(getattr(result, "geometry_valid", False))
            link = str(getattr(result, "closest_link", ""))
            collider = str(getattr(result, "closest_collider_path", ""))
            if not valid or not math.isfinite(gap) or not link or not collider:
                candidate.invalid_geometry_query_count += 1
                continue
            if is_crossing:
                candidate.crossing_sample_count += 1
                if candidate.crossing_start_sample_index is None:
                    candidate.crossing_start_surface_gap_m = gap
                    candidate.crossing_start_simulation_time_s = now
                    candidate.crossing_start_sample_index = sample_index
                    candidate.crossing_start_closest_link = link
                    candidate.crossing_start_closest_collider_path = collider
                if gap < candidate.nominal_minimum_surface_gap_m:
                    candidate.nominal_minimum_surface_gap_m = gap
                    candidate.nominal_minimum_simulation_time_s = now
                    candidate.nominal_minimum_sample_index = sample_index
                    candidate.nominal_minimum_progress_fraction = progress
                    candidate.closest_link = link
                    candidate.closest_collider_path = collider
            else:
                candidate.pre_cue_sample_count += 1
                if gap < candidate.minimum_pre_cue_start_gap_m:
                    candidate.minimum_pre_cue_start_gap_m = gap
                    candidate.minimum_pre_cue_simulation_time_s = now
                    candidate.minimum_pre_cue_sample_index = sample_index

        if now >= self.crossing_end_simulation_time_s:
            self._end_sample_observed = True
        return True

    def select_severity(
        self,
        *,
        severity: str,
        target_gap_m: float,
        allowed_gap_range_m: Sequence[float],
        minimum_pre_cue_start_gap_m: float | None = None,
    ) -> NominalCorridorSweepAudit:
        """Select the valid candidate nearest the severity target.

        Selection is fail-closed: the sampled window must be complete and
        regular, and every geometry query for the selected candidate must have
        been valid.  The actual BC+CBF trial gap is intentionally not an input.
        """

        low, high = _gap_range(allowed_gap_range_m)
        target = float(target_gap_m)
        if not math.isfinite(target) or not low <= target <= high:
            raise ValueError("target_gap_m must lie inside allowed_gap_range_m")
        required_pre_cue_gap = (
            None
            if minimum_pre_cue_start_gap_m is None
            else float(minimum_pre_cue_start_gap_m)
        )
        if required_pre_cue_gap is not None and (
            not math.isfinite(required_pre_cue_gap)
            or self.cue_lead_s == 0.0
        ):
            raise ValueError(
                "minimum_pre_cue_start_gap_m requires a finite threshold and "
                "a positive cue_lead_s"
            )
        coverage = self._coverage()
        if not coverage["complete"]:
            raise RuntimeError(
                "nominal BC corridor sweep has incomplete temporal coverage: "
                f"{coverage}"
            )

        eligible = [
            candidate
            for candidate in self._candidates
            if candidate.geometry_valid
            and candidate.crossing_sample_count == self._crossing_sample_count
            and (
                self.cue_lead_s == 0.0
                or candidate.pre_cue_sample_count == self._pre_cue_sample_count
            )
            and (
                required_pre_cue_gap is None
                or candidate.minimum_pre_cue_start_gap_m
                >= required_pre_cue_gap
            )
            and low
            <= candidate.nominal_minimum_surface_gap_m
            <= high
        ]
        if not eligible:
            raise RuntimeError(
                "no geometry-valid nominal BC corridor candidate realizes "
                f"severity={severity!r} in [{low:.4f}, {high:.4f}] m"
            )
        nearest_error = min(
            abs(candidate.nominal_minimum_surface_gap_m - target)
            for candidate in eligible
        )
        nearest = [
            candidate
            for candidate in eligible
            if abs(candidate.nominal_minimum_surface_gap_m - target)
            <= nearest_error + 1e-12
        ]
        selected = min(nearest, key=lambda candidate: candidate.normal_offset_m)
        assert selected.nominal_minimum_simulation_time_s is not None
        assert selected.nominal_minimum_sample_index is not None
        assert selected.nominal_minimum_progress_fraction is not None
        assert selected.crossing_start_surface_gap_m is not None
        assert selected.crossing_start_simulation_time_s is not None
        assert selected.crossing_start_sample_index is not None
        selected_corridor = CrossingCorridor(
            start_world_m=selected.corridor.start_world_m.copy(),
            end_world_m=selected.corridor.end_world_m.copy(),
            center_world_m=selected.corridor.center_world_m.copy(),
            tangent_world=selected.corridor.tangent_world.copy(),
            outward_normal_world=(
                selected.corridor.outward_normal_world.copy()
            ),
            planned_minimum_surface_gap_m=float(
                selected.nominal_minimum_surface_gap_m
            ),
            closest_link=selected.closest_link,
            closest_collider_path=selected.closest_collider_path,
            calibration_iterations=(
                next(
                    index
                    for index, candidate in enumerate(self._candidates, start=1)
                    if candidate is selected
                )
            ),
            geometry_query_semantics=(
                "physx_protected_surface_to_hand_sphere"
            ),
        )
        candidate_results = tuple(
            candidate.immutable_audit() for candidate in self._candidates
        )
        first_crossing = self._first_crossing_sample_time_s
        assert first_crossing is not None
        return NominalCorridorSweepAudit(
            semantics=NOMINAL_BC_SWEEP_SEMANTICS,
            phase=self.phase,
            severity=str(severity),
            hand=self.hand,
            crossing_direction=self.crossing_direction,
            speed_m_s=self.speed_m_s,
            cue_lead_s=self.cue_lead_s,
            corridor_length_m=self.corridor_length_m,
            target_gap_m=target,
            allowed_gap_minimum_m=low,
            allowed_gap_maximum_m=high,
            selected_normal_offset_m=selected.normal_offset_m,
            nominal_minimum_surface_gap_m=float(
                selected.nominal_minimum_surface_gap_m
            ),
            nominal_minimum_simulation_time_s=(
                selected.nominal_minimum_simulation_time_s
            ),
            nominal_minimum_sample_index=(
                selected.nominal_minimum_sample_index
            ),
            nominal_minimum_progress_fraction=(
                selected.nominal_minimum_progress_fraction
            ),
            closest_link=selected.closest_link,
            closest_collider_path=selected.closest_collider_path,
            crossing_start_surface_gap_m=(
                selected.crossing_start_surface_gap_m
            ),
            crossing_start_query_simulation_time_s=(
                selected.crossing_start_simulation_time_s
            ),
            crossing_start_sample_index=(
                selected.crossing_start_sample_index
            ),
            crossing_start_closest_link=(
                selected.crossing_start_closest_link
            ),
            crossing_start_closest_collider_path=(
                selected.crossing_start_closest_collider_path
            ),
            minimum_pre_cue_start_gap_m=(
                None
                if self.cue_lead_s == 0.0
                else float(selected.minimum_pre_cue_start_gap_m)
            ),
            required_minimum_pre_cue_start_gap_m=required_pre_cue_gap,
            minimum_pre_cue_simulation_time_s=(
                selected.minimum_pre_cue_simulation_time_s
            ),
            minimum_pre_cue_sample_index=(
                selected.minimum_pre_cue_sample_index
            ),
            phase_start_simulation_time_s=(
                self.phase_start_simulation_time_s
            ),
            crossing_start_simulation_time_s=(
                self.crossing_start_simulation_time_s
            ),
            crossing_end_simulation_time_s=(
                self.crossing_end_simulation_time_s
            ),
            first_sample_simulation_time_s=self._sample_times_s[0],
            last_sample_simulation_time_s=self._sample_times_s[-1],
            first_crossing_sample_simulation_time_s=first_crossing,
            pre_cue_sample_count=self._pre_cue_sample_count,
            crossing_sample_count=self._crossing_sample_count,
            time_sample_count=len(self._sample_times_s),
            geometry_query_count=(
                len(self._sample_times_s) * len(self._candidates)
            ),
            candidate_count=len(self._candidates),
            pre_cue_coverage_fraction=float(
                coverage["pre_cue_coverage_fraction"]
            ),
            crossing_coverage_fraction=float(
                coverage["crossing_coverage_fraction"]
            ),
            maximum_observed_sample_interval_s=float(
                coverage["maximum_observed_sample_interval_s"]
            ),
            maximum_allowed_sample_interval_s=(
                self.maximum_sample_interval_s
            ),
            geometry_valid=True,
            candidate_results=candidate_results,
            corridor=selected_corridor,
        )

    def _coverage(self) -> dict[str, float | bool]:
        if not self._sample_times_s or self._first_crossing_sample_time_s is None:
            return {
                "complete": False,
                "pre_cue_coverage_fraction": 0.0,
                "crossing_coverage_fraction": 0.0,
                "maximum_observed_sample_interval_s": math.inf,
            }
        intervals = np.diff(np.asarray(self._sample_times_s, dtype=float))
        maximum_interval = float(np.max(intervals)) if intervals.size else 0.0
        first = self._sample_times_s[0]
        last = self._sample_times_s[-1]
        first_crossing = self._first_crossing_sample_time_s
        crossing_duration = self.corridor_length_m / self.speed_m_s
        crossing_coverage = float(
            np.clip(
                (
                    min(last, self.crossing_end_simulation_time_s)
                    - max(first_crossing, self.crossing_start_simulation_time_s)
                )
                / crossing_duration,
                0.0,
                1.0,
            )
        )
        if self.cue_lead_s == 0.0:
            pre_cue_coverage = 1.0
            pre_cue_present = True
        else:
            pre_cue_coverage = float(
                np.clip(
                    (
                        min(first_crossing, self.crossing_start_simulation_time_s)
                        - max(first, self.phase_start_simulation_time_s)
                    )
                    / self.cue_lead_s,
                    0.0,
                    1.0,
                )
            )
            pre_cue_present = self._pre_cue_sample_count > 0
        tolerance = self.maximum_sample_interval_s + 1e-12
        complete = bool(
            pre_cue_present
            and self._crossing_sample_count > 0
            and first - self.phase_start_simulation_time_s <= tolerance
            and first_crossing - self.crossing_start_simulation_time_s
            <= tolerance
            and last >= self.crossing_end_simulation_time_s
            and maximum_interval <= tolerance
        )
        return {
            "complete": complete,
            "pre_cue_coverage_fraction": pre_cue_coverage,
            "crossing_coverage_fraction": crossing_coverage,
            "maximum_observed_sample_interval_s": maximum_interval,
        }


def _offset_corridor(
    base: CrossingCorridor,
    *,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length: float,
    direction: str,
) -> CrossingCorridor:
    # Calibration's canonical +Y tangent traverses participant right-to-left.
    direction_sign = 1.0 if direction == "right_to_left" else -1.0
    travel_tangent = tangent * direction_sign
    return CrossingCorridor(
        start_world_m=center - travel_tangent * length * 0.5,
        end_world_m=center + travel_tangent * length * 0.5,
        center_world_m=center.copy(),
        tangent_world=travel_tangent,
        outward_normal_world=normal.copy(),
        planned_minimum_surface_gap_m=(
            float(base.planned_minimum_surface_gap_m)
        ),
        closest_link=str(base.closest_link),
        closest_collider_path=str(base.closest_collider_path),
        calibration_iterations=int(base.calibration_iterations),
        geometry_query_semantics=str(base.geometry_query_semantics),
    )


def _finite_positive(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _unit_vector(value: Sequence[float], label: str) -> np.ndarray:
    result = _point3(value, label)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{label} must have non-zero length")
    return result / norm


def _gap_range(values: Sequence[float]) -> tuple[float, float]:
    try:
        low, high = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("allowed_gap_range_m must contain two values") from error
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("allowed_gap_range_m must be finite and ordered")
    return low, high


def calibrate_surface_gap_corridor(
    safety_geometry: Any,
    *,
    hand: str,
    end_effector_position_world_m: Sequence[float],
    target_gap_m: float,
    allowed_gap_range_m: Sequence[float],
    corridor_length_m: float,
    samples: int = 41,
    maximum_iterations: int = 24,
) -> CrossingCorridor:
    """Calibrate a tangent line against the *actual* protected PhysX surface.

    A far probe obtains a surface point and an outward direction.  The center
    is then moved along that direction until the minimum queried gap along the
    entire planned line lies in the configured severity interval.  Failure is
    explicit; the collector never silently falls back to EE-center distance.
    """

    ee = _point3(end_effector_position_world_m, "end_effector_position_world_m")
    low, high = [float(value) for value in allowed_gap_range_m]
    target = float(target_gap_m)
    length = float(corridor_length_m)
    if not (0.0 <= low <= target <= high and length > 0.05):
        raise ValueError("invalid severity range/target/corridor length")
    hand_radius = float(getattr(safety_geometry.thresholds, "hand_radius_m", 0.035))

    probes = (
        np.asarray([0.0, 0.0, 0.55]),
        np.asarray([0.30, 0.0, 0.25]),
        np.asarray([-0.30, 0.0, 0.25]),
        np.asarray([0.0, 0.35, 0.25]),
        np.asarray([0.0, -0.35, 0.25]),
    )
    seed_result = None
    seed_pos = None
    for offset in probes:
        candidate = ee + offset
        result = safety_geometry.evaluate_hand(hand, candidate)
        if bool(getattr(result, "geometry_valid", False)) and bool(
            getattr(result, "closest_surface_point_valid", False)
        ):
            seed_result, seed_pos = result, candidate
            break
    if seed_result is None or seed_pos is None:
        raise RuntimeError("cannot calibrate crossing corridor: no protected surface query")
    surface = _point3(
        getattr(seed_result, "closest_surface_point_world_pos"),
        "closest_surface_point_world_pos",
    )
    normal = seed_pos - surface
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-6:
        normal = np.asarray([0.0, 0.0, 1.0])
    else:
        normal /= norm

    # Prefer a visually natural table-horizontal direction, projected onto the
    # local tangent plane.  Fall back to world X if the normal is parallel.
    tangent = np.asarray([0.0, 1.0, 0.0])
    tangent -= normal * float(np.dot(tangent, normal))
    if np.linalg.norm(tangent) <= 1e-5:
        tangent = np.asarray([1.0, 0.0, 0.0])
        tangent -= normal * float(np.dot(tangent, normal))
    tangent /= np.linalg.norm(tangent)

    center = surface + normal * (hand_radius + target)
    best: tuple[float, Any, np.ndarray] | None = None
    for iteration in range(1, maximum_iterations + 1):
        minimum, result = _line_minimum_gap(
            safety_geometry,
            hand=hand,
            center=center,
            tangent=tangent,
            length=length,
            samples=samples,
        )
        error = minimum - target
        if best is None or abs(error) < abs(best[0] - target):
            best = (minimum, result, center.copy())
        if low <= minimum <= high:
            reset = getattr(safety_geometry, "reset_link_origin_pose_cache", None)
            if callable(reset):
                reset()
            return _corridor_from_solution(
                center, tangent, normal, length, minimum, result, iteration
            )
        # Moving the hand center outward changes the hand-surface gap roughly
        # one-for-one.  Clamp the update so collider switches stay stable.
        center -= normal * float(np.clip(error, -0.06, 0.06))
        result_at_center = safety_geometry.evaluate_hand(hand, center)
        if bool(getattr(result_at_center, "closest_surface_point_valid", False)):
            new_surface = _point3(
                getattr(result_at_center, "closest_surface_point_world_pos"),
                "closest_surface_point_world_pos",
            )
            candidate_normal = center - new_surface
            if np.linalg.norm(candidate_normal) > 1e-6:
                normal = candidate_normal / np.linalg.norm(candidate_normal)
                tangent -= normal * float(np.dot(tangent, normal))
                if np.linalg.norm(tangent) > 1e-6:
                    tangent /= np.linalg.norm(tangent)
    reset = getattr(safety_geometry, "reset_link_origin_pose_cache", None)
    if callable(reset):
        reset()
    assert best is not None
    raise RuntimeError(
        "planned protected-surface corridor could not reach severity range: "
        f"target={target:.4f}, range=[{low:.4f},{high:.4f}], best={best[0]:.4f}"
    )


def _line_minimum_gap(
    geometry: Any,
    *,
    hand: str,
    center: np.ndarray,
    tangent: np.ndarray,
    length: float,
    samples: int,
) -> tuple[float, Any]:
    best_gap = math.inf
    best_result = None
    for alpha in np.linspace(-0.5, 0.5, max(5, int(samples))):
        point = center + tangent * length * float(alpha)
        result = geometry.evaluate_hand(hand, point)
        gap = float(getattr(result, "surface_gap_m", math.inf))
        if bool(getattr(result, "geometry_valid", False)) and gap < best_gap:
            best_gap = gap
            best_result = result
    if best_result is None or not math.isfinite(best_gap):
        raise RuntimeError("protected-surface query became invalid during calibration")
    return best_gap, best_result


def _corridor_from_solution(
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length: float,
    minimum: float,
    result: Any,
    iteration: int,
) -> CrossingCorridor:
    return CrossingCorridor(
        start_world_m=center - tangent * length * 0.5,
        end_world_m=center + tangent * length * 0.5,
        center_world_m=center.copy(),
        tangent_world=tangent.copy(),
        outward_normal_world=normal.copy(),
        planned_minimum_surface_gap_m=float(minimum),
        closest_link=str(getattr(result, "closest_link", "")),
        closest_collider_path=str(getattr(result, "closest_collider_path", "")),
        calibration_iterations=int(iteration),
    )


def _positive_radius(value: float, name: str) -> float:
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return radius


class CrossingCueVisuals:
    """Stage-native VR-visible corridor markers.  No haptic API is used."""

    def __init__(
        self,
        world: Any,
        *,
        sample_count: int = 11,
        corridor_radius_m: float = 0.012,
        start_radius_m: float = 0.035,
        end_radius_m: float = 0.035,
    ) -> None:
        from omni.isaac.core.objects import VisualSphere

        corridor_radius_m = _positive_radius(corridor_radius_m, "corridor_radius_m")
        start_radius_m = _positive_radius(start_radius_m, "start_radius_m")
        end_radius_m = _positive_radius(end_radius_m, "end_radius_m")

        self._hidden = np.asarray([0.0, 0.0, -20.0])
        self._active = False
        self._start_time_s = 0.0
        self._travel_duration_s = 1.0
        self._start = self._hidden.copy()
        self._end = self._hidden.copy()
        colors = {
            "corridor": np.asarray([0.1, 0.55, 1.0]),
            "start": np.asarray([0.1, 1.0, 0.2]),
            "end": np.asarray([1.0, 0.2, 0.2]),
            "guide": np.asarray([1.0, 0.9, 0.1]),
        }
        self._corridor = []
        for index in range(max(5, int(sample_count))):
            self._corridor.append(
                world.scene.add(
                    VisualSphere(
                        prim_path=f"/World/OnlineFeedback/Corridor/{index:02d}",
                        name=f"online_feedback_corridor_{index:02d}",
                        position=self._hidden.copy(),
                        radius=corridor_radius_m,
                        color=colors["corridor"],
                    )
                )
            )
        self._start_marker = world.scene.add(
            VisualSphere(
                prim_path="/World/OnlineFeedback/Start",
                name="online_feedback_start",
                position=self._hidden.copy(),
                radius=start_radius_m,
                color=colors["start"],
            )
        )
        self._end_marker = world.scene.add(
            VisualSphere(
                prim_path="/World/OnlineFeedback/End",
                name="online_feedback_end",
                position=self._hidden.copy(),
                radius=end_radius_m,
                color=colors["end"],
            )
        )
        self._guide = world.scene.add(
            VisualSphere(
                prim_path="/World/OnlineFeedback/Guide",
                name="online_feedback_guide",
                position=self._hidden.copy(),
                radius=max(0.012, corridor_radius_m * 0.55),
                color=colors["guide"],
            )
        )

    def show(self, corridor: CrossingCorridor, *, simulation_time_s: float, speed_m_s: float) -> None:
        self._start = corridor.start_world_m.copy()
        self._end = corridor.end_world_m.copy()
        length = float(np.linalg.norm(self._end - self._start))
        self._travel_duration_s = max(0.1, length / max(0.01, float(speed_m_s)))
        self._start_time_s = float(simulation_time_s)
        for index, marker in enumerate(self._corridor):
            alpha = index / max(1, len(self._corridor) - 1)
            marker.set_world_pose(position=(1.0 - alpha) * self._start + alpha * self._end)
        self._start_marker.set_world_pose(position=self._start)
        self._end_marker.set_world_pose(position=self._end)
        self._guide.set_world_pose(position=self._start)
        self._active = True

    def update(self, simulation_time_s: float) -> None:
        if not self._active:
            return
        elapsed = max(0.0, float(simulation_time_s) - self._start_time_s)
        alpha = min(1.0, elapsed / self._travel_duration_s)
        self._guide.set_world_pose(position=(1.0 - alpha) * self._start + alpha * self._end)

    def hide(self) -> None:
        for marker in (*self._corridor, self._start_marker, self._end_marker, self._guide):
            marker.set_world_pose(position=self._hidden.copy())
        self._active = False


class ProximalArmProtocolMonitor:
    """Conservative read-only AABB check for unprotected Panda links 0--5."""

    def __init__(self, *, robot_prim_path: str, proximal_link_tokens: Sequence[str], hand_radius_m: float) -> None:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._stage = omni.usd.get_context().get_stage()
        self._hand_radius_m = float(hand_radius_m)
        self._paths: list[str] = []
        root = self._stage.GetPrimAtPath(str(robot_prim_path))
        tokens = tuple(str(value) for value in proximal_link_tokens)
        if root.IsValid():
            for prim in Usd.PrimRange(root):
                path = str(prim.GetPath())
                if not any(f"/{token}" in path for token in tokens):
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision = UsdPhysics.CollisionAPI(prim)
                if collision and collision.GetCollisionEnabledAttr().Get() is not False:
                    self._paths.append(path)

    @property
    def collider_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def closest_surface_gap_m(self, point_world_m: Sequence[float]) -> tuple[float, str]:
        point = _point3(point_world_m, "point_world_m")
        cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [self._UsdGeom.Tokens.default_, self._UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        best_gap, best_path = math.inf, ""
        for path in self._paths:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            low = np.asarray(aligned.GetMin(), dtype=float)
            high = np.asarray(aligned.GetMax(), dtype=float)
            closest = np.clip(point, low, high)
            gap = float(np.linalg.norm(point - closest) - self._hand_radius_m)
            if gap < best_gap:
                best_gap, best_path = gap, path
        return best_gap, best_path


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 3-vector")
    return result.copy()
