"""Core data schemas for predec.

Everything in the package is built around `PreferencePair`. Detectors receive
lists of pairs and return `DetectorResult` objects. Reports serialize
`DetectorResult` to JSON / HTML.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


Choice = Literal["a", "b"]


@dataclass
class PreferencePair:
    """A single preference annotation.

    `chosen` indicates which response the human annotator picked. Positions
    (A vs B) are assumed to have been presented in some order; downstream
    detectors may swap them and track the original via `metadata`.
    """

    id: str
    prompt: str
    response_a: str
    response_b: str
    chosen: Choice
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreferencePair":
        return cls(
            id=str(d.get("id", "")),
            prompt=str(d.get("prompt", "")),
            response_a=str(d.get("response_a", "")),
            response_b=str(d.get("response_b", "")),
            chosen=d["chosen"],  # type: ignore[arg-type]
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class FlaggedExample:
    """A single example flagged by a detector, with evidence."""

    pair_id: str
    evidence: dict[str, Any]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorResult:
    """Output of one bias detector."""

    name: str
    score: float  # 0..1, higher = stronger bias signal
    confidence_interval: tuple[float, float]  # 95% bootstrap CI
    n_pairs_analyzed: int
    n_pairs_flagged: int
    metric_description: str
    threshold: float = 0.60  # the detector's flag threshold at detection time
    examples: list[FlaggedExample] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence_interval"] = list(self.confidence_interval)
        return d


@dataclass
class DatasetSummary:
    """Lightweight summary of the input dataset, included in the report."""

    name: str
    source: str
    n_pairs: int
    avg_prompt_tokens: float
    avg_response_tokens: float
    chosen_a_rate: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Top-level report: dataset summary + 4 detector results + overall."""

    dataset: DatasetSummary
    detectors: dict[str, DetectorResult]
    overall_bias_score: float
    overall_recommendation: str
    trajectory_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "detectors": {k: v.to_dict() for k, v in self.detectors.items()},
            "overall_bias_score": self.overall_bias_score,
            "overall_recommendation": self.overall_recommendation,
            "trajectory_log": self.trajectory_log,
        }
