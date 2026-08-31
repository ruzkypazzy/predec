"""LLM-powered report writer.

Takes the 4 detector results + dataset summary and produces a single
human-readable recommendation paragraph. This is the second LLM call
in the pipeline (the first being the sycophancy judge).
"""

from __future__ import annotations

from typing import Any

from ..llm.client import LLMClient
from ..schema import AnalysisReport


SYSTEM_PROMPT = """You are a concise, technical analyst writing the executive summary of a bias
audit on an RLHF preference dataset. You receive structured detector outputs and
must write ONE short paragraph (3-5 sentences) summarizing:
  1. The most concerning bias detected (if any)
  2. Whether the dataset is safe to train a reward model on as-is
  3. A concrete next-step recommendation (reweight, reswap, filter, or accept)

Tone: direct, technical, no hedging. Output strict JSON only.
"""


def write_recommendation(
    report: AnalysisReport,
    llm: LLMClient | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (recommendation_text, llm_call_record). Falls back to a templated
    string if no LLM is available."""
    dets = report.detectors
    flagged = [
        (name, d)
        for name, d in dets.items()
        if d.score >= d.flag_threshold
    ]
    if not flagged:
        fallback = (
            f"No significant bias detected across the {len(dets)} detectors. "
            f"The dataset appears safe for reward model training as-is. "
            f"Re-run predec if the data source changes."
        )
        return fallback, {}

    # Sort flagged detectors by score
    flagged.sort(key=lambda t: t[1].score, reverse=True)
    worst_name, worst = flagged[0]

    if llm is None or not llm.available:
        # Templated fallback
        if worst_name == "position":
            fallback = (
                f"Position bias detected: A's win rate is {worst.score:.2f} (CI {worst.confidence_interval[0]:.2f}-{worst.confidence_interval[1]:.2f}), "
                f"significantly above 0.50. Recommendation: run \`predec debias --strategy reswap\` "
                f"to randomize A/B ordering, then re-run to confirm."
            )
        elif worst_name == "length":
            fallback = (
                f"Length bias detected: among semantically-equivalent pairs, the longer response wins "
                f"{worst.score:.2f} of the time. Recommendation: run \`predec debias --strategy reweight\` "
                f"to down-weight length-flagged examples, or filter the most extreme cases."
            )
        elif worst_name == "sycophancy":
            fallback = (
                f"Sycophancy bias detected: on prompts with a stated premise, the agreeing response "
                f"wins {worst.score:.2f} of the time. Recommendation: re-annotate these prompts with "
                f"rubric that explicitly rewards pushback, or filter them out."
            )
        else:
            fallback = (
                f"Verbosity bias detected: structural markers (bullets, hedges, preambles) predict "
                f"preferences with AUC {worst.score:.2f}. Recommendation: reweight or filter, and "
                f"audit annotator guidelines for explicit 'do not reward verbosity' rules."
            )
        return fallback, {}

    # LLM call
    payload = {
        "dataset": report.dataset.to_dict(),
        "detectors": {
            name: {
                "score": d.score,
                "ci": list(d.confidence_interval),
                "n_flagged": d.n_pairs_flagged,
                "metric": d.metric_description,
            }
            for name, d in dets.items()
        },
        "overall_bias_score": report.overall_bias_score,
    }
    user = f"Write the executive recommendation based on this bias audit:\n\n{json_dumps(payload)}"
    call = llm.call_json(
        actor="report-writer",
        purpose="write_recommendation",
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=400,
    )
    if call.response_json is None or "recommendation" not in call.response_json:
        return "Bias detected. See detector breakdown above for details.", call.to_dict() if hasattr(call, "to_dict") else {}
    return call.response_json["recommendation"], {
        "actor": call.actor,
        "purpose": call.purpose,
        "model": call.model,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "duration_ms": call.duration_ms,
    }


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
