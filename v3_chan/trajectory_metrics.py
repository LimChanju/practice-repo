from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CartesianMotionSample:
    dt_s: float = 0.0
    path_increment_m: float = 0.0
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    acceleration_mps2: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    jerk_mps3: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_valid: bool = False
    acceleration_valid: bool = False
    jerk_valid: bool = False

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_mps))

    @property
    def acceleration_norm_mps2(self) -> float:
        return float(np.linalg.norm(self.acceleration_mps2))

    @property
    def jerk_norm_mps3(self) -> float:
        return float(np.linalg.norm(self.jerk_mps3))


class CartesianMotionTracker:
    """Track raw finite-difference Cartesian motion on simulation timestamps."""

    def __init__(self, position_m, time_s: float) -> None:
        self._position = _vec3_or_none(position_m)
        self._time_s = float(time_s)
        self._velocity: np.ndarray | None = None
        self._acceleration: np.ndarray | None = None
        self._path_length_m = 0.0
        self._elapsed_s = 0.0
        self._speed_values: list[float] = []
        self._acceleration_values: list[float] = []
        self._jerk_values: list[float] = []
        self._integrated_squared_jerk_m2ps5 = 0.0

    def update(self, position_m, time_s: float) -> CartesianMotionSample:
        position = _vec3_or_none(position_m)
        timestamp = float(time_s)
        if (
            position is None
            or self._position is None
            or not np.isfinite(timestamp)
            or not np.isfinite(self._time_s)
        ):
            self._reset_derivatives(position, timestamp)
            return CartesianMotionSample()

        dt_s = timestamp - self._time_s
        if not np.isfinite(dt_s) or dt_s <= 1e-9:
            self._reset_derivatives(position, timestamp)
            return CartesianMotionSample()

        displacement = position - self._position
        path_increment_m = float(np.linalg.norm(displacement))
        velocity = displacement / dt_s
        acceleration = np.zeros(3, dtype=float)
        jerk = np.zeros(3, dtype=float)
        acceleration_valid = self._velocity is not None
        jerk_valid = acceleration_valid and self._acceleration is not None
        if acceleration_valid:
            acceleration = (velocity - self._velocity) / dt_s
        if jerk_valid:
            jerk = (acceleration - self._acceleration) / dt_s

        sample = CartesianMotionSample(
            dt_s=float(dt_s),
            path_increment_m=path_increment_m,
            velocity_mps=velocity,
            acceleration_mps2=acceleration,
            jerk_mps3=jerk,
            velocity_valid=True,
            acceleration_valid=acceleration_valid,
            jerk_valid=jerk_valid,
        )
        self._path_length_m += path_increment_m
        self._elapsed_s += dt_s
        self._speed_values.append(sample.speed_mps)
        if acceleration_valid:
            self._acceleration_values.append(sample.acceleration_norm_mps2)
        if jerk_valid:
            jerk_norm = sample.jerk_norm_mps3
            self._jerk_values.append(jerk_norm)
            self._integrated_squared_jerk_m2ps5 += jerk_norm**2 * dt_s

        self._position = position
        self._time_s = timestamp
        self._velocity = velocity
        self._acceleration = acceleration if acceleration_valid else None
        return sample

    def summary(self) -> dict[str, float | int]:
        speeds = np.asarray(self._speed_values, dtype=float)
        accelerations = np.asarray(self._acceleration_values, dtype=float)
        jerks = np.asarray(self._jerk_values, dtype=float)
        return {
            "ee_path_length_m": float(self._path_length_m),
            "ee_motion_duration_s": float(self._elapsed_s),
            "mean_ee_speed_mps": _mean_or_zero(speeds),
            "max_ee_speed_mps": _max_or_zero(speeds),
            "mean_ee_acceleration_mps2": _mean_or_zero(accelerations),
            "rms_ee_acceleration_mps2": _rms_or_zero(accelerations),
            "max_ee_acceleration_mps2": _max_or_zero(accelerations),
            "mean_ee_jerk_mps3": _mean_or_zero(jerks),
            "rms_ee_jerk_mps3": _rms_or_zero(jerks),
            "p95_ee_jerk_mps3": _percentile_or_zero(jerks, 95.0),
            "max_ee_jerk_mps3": _max_or_zero(jerks),
            "integrated_squared_ee_jerk_m2ps5": float(
                self._integrated_squared_jerk_m2ps5
            ),
            "ee_velocity_sample_count": int(speeds.size),
            "ee_acceleration_sample_count": int(accelerations.size),
            "ee_jerk_sample_count": int(jerks.size),
        }

    def _reset_derivatives(self, position: np.ndarray | None, timestamp: float) -> None:
        self._position = position
        self._time_s = timestamp
        self._velocity = None
        self._acceleration = None


def _vec3_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3].copy()


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _max_or_zero(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _rms_or_zero(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0
