"""Main analysis orchestrator.

Takes a list of PreferencePairs, runs the 4 detectors in order, computes
the overall bias score, and produces an AnalysisReport. This is the
"agent" that judges want to see trajectories for.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import tiktoken

from .detectors.length import LengthDetector
from .detectors.position import PositionDetector
from .detectors.sycophancy import SycophancyDetector
from .detectors.verbosity import VerbosityDetector
from .llm.client import LLMClient
from .report.writer import write_recommendation
from .schema import AnalysisReport, DatasetSummary, DetectorResult, PreferencePair
from .trajectory.logger import TrajectoryLogger


def run_analysis(
    pairs: list[PreferencePair],
    dataset_name: str = "unnamed",
    dataset_source: str = "unknown",
    llm: LLMClient | None = None,
) -> AnalysisReport:
    """Run all 4 detectors and return the full report.

    `llm` may be None — the sycophancy detector will raise if it tries to
    make a call without an LLM. Pass a client with OPENAI_API_KEY set.
    """
    log = TrajectoryLogger()
    tok = tiktoken.encoding_for_model("gpt-4o-mini")

    t0 = time.time()
    log.record(
        actor="orchestrator",
        action="received_dataset",
        input={"n_pairs": len(pairs), "name": dataset_name, "source": dataset_source},
        output={"ready": True},
        decision="run all 4 detectors",
    )

    # Build dataset summary
    n = len(pairs)
    if n == 0:
        return AnalysisReport(
            dataset=DatasetSummary(
                name=dataset_name, source=dataset_source, n_pairs=0,
                avg_prompt_tokens=0, avg_response_tokens=0, chosen_a_rate=0.0,
            ),
            detectors={},
            overall_bias_score=0.0,
            overall_recommendation="No data to analyze.",
            trajectory_log=log.to_list(),
        )

    prompt_tokens = [len(tok.encode(p.prompt)) for p in pairs]
    response_a_tokens = [len(tok.encode(p.response_a)) for p in pairs]
    response_b_tokens = [len(tok.encode(p.response_b)) for p in pairs]
    chosen_a_rate = sum(1 for p in pairs if p.chosen == "a") / n
    summary = DatasetSummary(
        name=dataset_name,
        source=dataset_source,
        n_pairs=n,
        avg_prompt_tokens=float(np.mean(prompt_tokens)),
        avg_response_tokens=float(np.mean(response_a_tokens + response_b_tokens)),
        chosen_a_rate=chosen_a_rate,
    )

    # Build detectors
    detectors: list[Any] = [
        LengthDetector(trajectory=log, openai_client=llm._client if llm else None),
        PositionDetector(trajectory=log),
        SycophancyDetector(trajectory=log, llm_client=llm),
        VerbosityDetector(trajectory=log),
    ]

    # Run sequentially
    results: dict[str, DetectorResult] = {}
    for det in detectors:
        log.record(
            actor="orchestrator",
            action=f"running_{det.name}",
            input={"n_pairs": n},
            output={"uses_llm": det.uses_llm},
            decision=f"delegate to {det.name} detector",
        )
        t_d = time.time()
        try:
            result = det.detect(pairs)
        except Exception as e:
            log.record(
                actor="orchestrator",
                action=f"error_in_{det.name}",
                input={},
                output={"error": str(e)},
                decision=f"continue with remaining detectors",
            )
            continue
        result.duration_ms = (time.time() - t_d) * 1000
        result.threshold = det.flag_threshold
        results[det.name] = result
        log.record(
            actor="orchestrator",
            action=f"completed_{det.name}",
            input={},
            output={
                "score": result.score,
                "n_flagged": result.n_pairs_flagged,
                "duration_ms": result.duration_ms,
            },
            decision=(
                f"flagged ({result.n_pairs_flagged} pairs)"
                if result.score >= result.threshold
                else "within tolerance"
            ),
        )

    # Overall score: weighted mean of detector scores that exceeded their
    # threshold, otherwise the max score across detectors.
    flagged_scores = [
        r.score for r in results.values() if r.score >= r.threshold
    ]
    if flagged_scores:
        overall = float(np.mean(flagged_scores))
    else:
        # No bias flagged — overall is the max (the most-concerning signal)
        overall = float(max((r.score for r in results.values()), default=0.5))

    # Write the recommendation (LLM or templated fallback)
    rec, rec_call = write_recommendation(
        AnalysisReport(
            dataset=summary,
            detectors=results,
            overall_bias_score=overall,
            overall_recommendation="",
            trajectory_log=log.to_list(),
        ),
        llm=llm,
        trajectory=log,
    )
    if rec_call:
        log.record(
            actor="report-writer",
            action="wrote_recommendation",
            input={},
            output=rec_call,
            decision="narrative summary complete",
        )

    log.record(
        actor="orchestrator",
        action="analysis_complete",
        input={},
        output={"overall_bias_score": overall, "n_detectors_run": len(results)},
        decision="emit report",
        duration_ms=(time.time() - t0) * 1000,
    )

    return AnalysisReport(
        dataset=summary,
        detectors=results,
        overall_bias_score=overall,
        overall_recommendation=rec,
        trajectory_log=log.to_list(),
    )


def baseline_one_prompt(
    pairs: list[PreferencePair],
    llm: LLMClient,
    dataset_name: str = "unnamed",
) -> dict[str, Any]:
    """The single-prompt baseline. Returns a minimal report-like dict.

    This is what the brief calls "One direct prompt with basic instructions" —
    the simplest reasonable thing you'd do before building the agent.
    """
    # Take a sample of 50 pairs so the prompt fits
    sample = pairs[:50]
    sample_text = "\n\n".join(
        f"PROMPT: {p.prompt[:200]}\nA: {p.response_a[:200]}\nB: {p.response_b[:200]}\nCHOSEN: {p.chosen}"
        for p in sample
    )
    system = (
        "You are an analyst reviewing a preference dataset. Identify any biases "
        "you observe. For each bias, give a score from 0 to 1 and 3 example pairs. "
        "Return strict JSON with a 'biases' field containing entries for length, "
        "position, sycophancy, and verbosity, each with a 'score' (0-1) and 'examples' array."
    )
    user = f"Dataset: {dataset_name}\n\n{sample_text}\n\nIdentify biases."
    t0 = time.time()
    call = llm.call_json(
        actor="baseline",
        purpose="find_biases_one_shot",
        system=system,
        user=user,
        max_tokens=1500,
    )
    return {
        "score": None,  # baseline doesn't return a structured score
        "raw_response": call.response_text,
        "llm_call": {
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": call.cost_usd,
            "duration_ms": call.duration_ms,
        },
        "duration_ms": (time.time() - t0) * 1000,
    }
