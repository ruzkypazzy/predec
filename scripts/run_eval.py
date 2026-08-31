"""Run the eval: agentic pipeline vs single-prompt baseline.

Loads eval/eval_pairs.jsonl, runs both the orchestrator and the one-shot
baseline, and computes precision/recall/F1 for bias detection.

A bias is "detected" if:
  - Agentic: the corresponding detector's score >= its flag_threshold
  - Baseline: the baseline response text contains the bias name AND
    provides a numeric score > 0.5 for that bias

We also report per-detector precision/recall, and the overall F1.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Add src to path so we can import predec
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predec.llm.client import LLMClient  # noqa: E402
from predec.loaders import load_preference_dataset  # noqa: E402
from predec.orchestrator import baseline_one_prompt, run_analysis  # noqa: E402
from predec.schema import PreferencePair  # noqa: E402


def _baseline_parse_baseline_scores(text: str) -> dict[str, float]:
    """Parse the baseline's freeform response for any bias names + scores."""
    found: dict[str, float] = {}
    # Look for "length: 0.7" or "length bias: 0.71" etc.
    patterns = {
        "length": r"length[^:]*:\s*([0-9.]+)",
        "position": r"position[^:]*:\s*([0-9.]+)",
        "sycophancy": r"sycophanc?y[^:]*:\s*([0-9.]+)",
        "verbosity": r"verbosity[^:]*:\s*([0-9.]+)",
    }
    for name, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                found[name] = float(m.group(1))
            except ValueError:
                pass
    return found


def evaluate_agent(pairs: list[PreferencePair], llm: LLMClient) -> dict:
    """Run the agent and extract per-bias flags."""
    report = run_analysis(pairs, dataset_name="eval", dataset_source="synthetic", llm=llm)
    flags = {name: r.score >= r.threshold for name, r in report.detectors.items()}
    return {
        "flags": flags,
        "scores": {name: r.score for name, r in report.detectors.items()},
        "report": report,
        "overall_bias_score": report.overall_bias_score,
    }


def evaluate_baseline(pairs: list[PreferencePair], llm: LLMClient) -> dict:
    """Run the single-prompt baseline."""
    bl = baseline_one_prompt(pairs, llm, dataset_name="eval")
    parsed = _baseline_parse_baseline_scores(bl["raw_response"])
    flags = {name: (parsed.get(name, 0.0) > 0.5) for name in ("length", "position", "sycophancy", "verbosity")}
    return {
        "flags": flags,
        "scores": parsed,
        "raw_response": bl["raw_response"],
        "llm_call": bl["llm_call"],
    }


def compute_metrics(predicted: dict[str, bool], ground_truth: dict[str, int]) -> dict:
    """Compute precision/recall/F1 for the 4 bias types.

    `ground_truth` is a dict {bias_name: n_planted}. `predicted` is a dict
    {bias_name: bool (flagged?)}.
    """
    out = {}
    for name, n_planted in ground_truth.items():
        if name == "clean":
            continue
        tp = int(predicted.get(name, False) and n_planted > 0)
        fp = int(predicted.get(name, False) and n_planted == 0)
        fn = int((not predicted.get(name, False)) and n_planted > 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load eval set
    pairs = load_preference_dataset({"synthetic": "eval", "limit": args.limit})
    with open(ROOT / "eval" / "ground_truth.json") as f:
        gt = json.load(f)
    print(f"Loaded {len(pairs)} pairs; ground truth:")
    for k, v in gt["planted_biases"].items():
        print(f"  {k}: {v}")

    # Need an LLM for the sycophancy detector and the report writer
    llm = LLMClient()
    if not llm.available:
        print("WARNING: OPENAI_API_KEY is not set. The sycophancy detector will fail.")
        print("Set OPENAI_API_KEY in your environment to run the full eval.")
        return 1

    # Run agent
    print("\n=== AGENTIC PIPELINE ===")
    t0 = time.time()
    agent_result = evaluate_agent(pairs, llm)
    agent_time = time.time() - t0
    print(f"Ran in {agent_time:.1f}s; flags: {agent_result['flags']}")
    print(f"Scores: {agent_result['scores']}")

    # Run baseline
    print("\n=== SINGLE-PROMPT BASELINE ===")
    t0 = time.time()
    baseline_result = evaluate_baseline(pairs, llm)
    baseline_time = time.time() - t0
    print(f"Ran in {baseline_time:.1f}s; flags: {baseline_result['flags']}")
    print(f"Parsed scores: {baseline_result['scores']}")

    # Compute metrics
    print("\n=== METRICS ===")
    print(f"{'Bias':<12} {'Planted':<8} {'Agent F1':<10} {'Baseline F1':<14}")
    print("-" * 50)
    agent_metrics = compute_metrics(agent_result["flags"], gt["planted_biases"])
    baseline_metrics = compute_metrics(baseline_result["flags"], gt["planted_biases"])
    macro_f1_agent = 0.0
    macro_f1_baseline = 0.0
    n = 0
    for name in ("length", "position", "sycophancy", "verbosity"):
        a = agent_metrics[name]["f1"]
        b = baseline_metrics[name]["f1"]
        macro_f1_agent += a
        macro_f1_baseline += b
        n += 1
        print(f"{name:<12} {gt['planted_biases'][name]:<8} {a:<10.3f} {b:<14.3f}")
    macro_f1_agent /= n
    macro_f1_baseline /= n
    print(f"\n{'MACRO F1':<12} {'':<8} {macro_f1_agent:<10.3f} {macro_f1_baseline:<14.3f}")

    # Save results
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump({
            "n_pairs": len(pairs),
            "ground_truth": gt["planted_biases"],
            "agent": {
                "flags": agent_result["flags"],
                "scores": agent_result["scores"],
                "metrics": agent_metrics,
                "macro_f1": macro_f1_agent,
                "duration_s": agent_time,
                "llm_cost_usd": sum(c.cost_usd for c in llm.calls),
            },
            "baseline": {
                "flags": baseline_result["flags"],
                "scores": baseline_result["scores"],
                "metrics": baseline_metrics,
                "macro_f1": macro_f1_baseline,
                "duration_s": baseline_time,
            },
        }, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'eval_results.json'}")

    # Also write the report.html
    from predec.report.render import write_report
    html_path, json_path = write_report(agent_result["report"], str(out_dir / "report"))
    print(f"Wrote {html_path} and {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
