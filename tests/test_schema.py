"""Smoke tests that don't require tiktoken/sklearn/openai.

These cover:
  - Schema serialization roundtrip
  - HTML report rendering
  - JSON report rendering
  - Trajectory log persistence
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predec.schema import (  # noqa: E402
    AnalysisReport,
    DatasetSummary,
    DetectorResult,
    FlaggedExample,
    PreferencePair,
)
from predec.report.render import write_report  # noqa: E402
from predec.trajectory.logger import TrajectoryLogger  # noqa: E402


def test_preference_pair_roundtrip():
    p = PreferencePair(
        id="p1", prompt="What is 2+2?", response_a="4", response_b="Four",
        chosen="a", metadata={"src": "test"},
    )
    d = p.to_dict()
    p2 = PreferencePair.from_dict(d)
    assert p2.id == p.id
    assert p2.chosen == p.chosen
    assert p2.metadata == p.metadata
    print("  preference_pair_roundtrip: OK")


def test_detector_result_roundtrip():
    r = DetectorResult(
        name="length",
        score=0.71,
        confidence_interval=(0.65, 0.77),
        n_pairs_analyzed=1000,
        n_pairs_flagged=420,
        metric_description="longer wins 71%",
        threshold=0.65,
        examples=[FlaggedExample(pair_id="x", evidence={"k": 1}, explanation="e1")],
    )
    d = r.to_dict()
    assert d["confidence_interval"] == [0.65, 0.77]
    assert d["threshold"] == 0.65
    assert len(d["examples"]) == 1
    print("  detector_result_roundtrip: OK")


def test_trajectory_logger():
    log = TrajectoryLogger()
    log.record("a", "act", {"i": 1}, {"o": 2}, "next", 1.0)
    log.record("b", "act2", {"i": 3}, {"o": 4}, "next2", 2.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    log.save(path)
    data = json.load(open(path))
    assert len(data) == 2
    assert data[0]["actor"] == "a"
    print("  trajectory_logger: OK")


def test_html_report_renders():
    r1 = DetectorResult(
        name="length", score=0.71, confidence_interval=(0.65, 0.77),
        n_pairs_analyzed=1000, n_pairs_flagged=420, metric_description="longer wins",
        threshold=0.65, examples=[FlaggedExample(pair_id="p1", evidence={"d": 200}, explanation="length delta")],
    )
    r2 = DetectorResult(
        name="position", score=0.51, confidence_interval=(0.48, 0.54),
        n_pairs_analyzed=1000, n_pairs_flagged=0, metric_description="A wins", threshold=0.55,
    )
    r3 = DetectorResult(
        name="sycophancy", score=0.68, confidence_interval=(0.60, 0.76),
        n_pairs_analyzed=1000, n_pairs_flagged=120, metric_description="agreeing wins",
        threshold=0.65, examples=[FlaggedExample(pair_id="s1", evidence={"a": "AGREE"}, explanation="premise")],
    )
    r4 = DetectorResult(
        name="verbosity", score=0.59, confidence_interval=(0.0, 0.0),
        n_pairs_analyzed=1000, n_pairs_flagged=200, metric_description="AUC",
        threshold=0.55, extra={"top_feature": "bullet_count"},
    )
    summary = DatasetSummary(name="test", source="synthetic", n_pairs=1000,
                              avg_prompt_tokens=100, avg_response_tokens=200, chosen_a_rate=0.51)
    report = AnalysisReport(
        dataset=summary,
        detectors={"length": r1, "position": r2, "sycophancy": r3, "verbosity": r4},
        overall_bias_score=0.69,
        overall_recommendation="Length and sycophancy are flagged.",
        trajectory_log=[{"actor": "orch", "action": "start", "input": {}, "output": {}, "decision": "go", "duration_ms": 0}],
    )
    with tempfile.TemporaryDirectory() as tmp:
        html_path, json_path = write_report(report, tmp)
        html = open(html_path).read()
        data = json.load(open(json_path))
        assert "predec" in html
        assert "length" in html
        assert "sycophancy" in html
        assert "biasChart" in html
        assert data["overall_bias_score"] == 0.69
        assert "length" in data["detectors"]
    print("  html_report_renders: OK")


if __name__ == "__main__":
    print("Running predec smoke tests:")
    test_preference_pair_roundtrip()
    test_detector_result_roundtrip()
    test_trajectory_logger()
    test_html_report_renders()
    print("All smoke tests passed.")
