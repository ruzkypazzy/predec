"""Base class and shared utilities for all bias detectors."""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np

from ..schema import DetectorResult, FlaggedExample, PreferencePair
from ..trajectory.logger import TrajectoryLogger


def bootstrap_ci(
    values: list[float] | np.ndarray,
    fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute bootstrap CI for a statistic. Returns (point_estimate, (lo, hi))."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(fn(arr))
    n = arr.size
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = fn(arr[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, (lo, hi)


class BaseDetector:
    """Subclass and implement `detect()`.

    Each detector receives a list of pairs and a trajectory logger. The
    logger is optional but recommended — the report needs trajectories.
    """

    name: str = "base"
    metric_description: str = ""
    uses_llm: bool = False
    # Bias-score threshold above which we consider the detector "flagged"
    flag_threshold: float = 0.60

    def __init__(self, trajectory: TrajectoryLogger | None = None, **kwargs: Any) -> None:
        self.trajectory = trajectory
        self.config = kwargs

    def detect(self, pairs: list[PreferencePair]) -> DetectorResult:
        raise NotImplementedError

    # --- helpers used by subclasses ---

    def _log(self, action: str, input: dict, output: dict, decision: str = "") -> None:
        if self.trajectory is not None:
            self.trajectory.record(
                actor=f"{self.name}-detector",
                action=action,
                input=input,
                output=output,
                decision=decision,
            )

    @staticmethod
    def _top_examples(
        pairs: list[PreferencePair],
        scores: list[float],
        n: int = 5,
    ) -> list[FlaggedExample]:
        """Return the n highest-scoring pairs as FlaggedExamples."""
        if not scores or not pairs:
            return []
        indexed = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
        out: list[FlaggedExample] = []
        for idx, s in indexed[:n]:
            p = pairs[idx]
            chosen_text = p.response_a if p.chosen == "a" else p.response_b
            rejected_text = p.response_b if p.chosen == "a" else p.response_a
            out.append(
                FlaggedExample(
                    pair_id=p.id,
                    evidence={"score": float(s), "prompt": p.prompt[:200]},
                    explanation=(
                        f"prompt: {p.prompt[:120]!r} | chosen excerpt: {chosen_text[:120]!r} "
                        f"| rejected excerpt: {rejected_text[:120]!r}"
                    ),
                )
            )
        return out
