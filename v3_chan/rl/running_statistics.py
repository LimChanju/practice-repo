from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """Numerically stable per-dimension running moments."""

    dimension: int
    std_floor: float = 1e-3

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if not np.isfinite(self.std_floor) or float(self.std_floor) <= 0.0:
            raise ValueError("std_floor must be finite and positive")
        self.count = 0
        self.mean = np.zeros(int(self.dimension), dtype=np.float64)
        self.m2 = np.zeros(int(self.dimension), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2 or batch.shape[1] != int(self.dimension):
            raise ValueError(
                f"values must have shape (N, {self.dimension}), got {batch.shape}"
            )
        if batch.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch)):
            raise ValueError("running statistics received non-finite values")

        batch_count = int(batch.shape[0])
        batch_mean = np.mean(batch, axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / combined_count)
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / combined_count)
        )
        self.count = combined_count

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 1:
            std = np.ones(int(self.dimension), dtype=np.float64)
        else:
            variance = np.maximum(self.m2 / self.count, 0.0)
            std = np.sqrt(variance)
        std = np.maximum(std, float(self.std_floor))
        return (
            self.mean.astype(np.float32).reshape(1, -1),
            std.astype(np.float32).reshape(1, -1),
        )
