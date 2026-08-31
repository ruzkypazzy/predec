"""Debiasing strategies for preference datasets.

Three strategies, each addressing a different bias type:

  reweight  - Down-weight examples that exemplify a flagged bias. The new
              dataset has the same examples but a `weight` field that
              downstream reward model training can respect.

  reswap    - For position-biased datasets, swap A/B labels with 50%
              probability. This randomizes position assignment and washes
              out position bias.

  filter    - Drop the most biased examples (top 10% by bias score). The
              new dataset is smaller but cleaner.

Each strategy returns (new_pairs, metadata) where metadata describes what
was changed and the new bias estimates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tiktoken

from ..schema import DetectorResult, PreferencePair


Strategy = Literal["reweight", "reswap", "filter"]


@dataclass
class DebiasedResult:
    """Output of a debiasing pass."""

    strategy: Strategy
    pairs: list[PreferencePair]
    weights: list[float]  # parallel to pairs; 1.0 by default
    n_dropped: int
    n_reswapped: int
    notes: list[str]


def _reweight(
    pairs: list[PreferencePair],
    results: dict[str, DetectorResult],
    tokenizer,
) -> DebiasedResult:
    """Down-weight pairs that exemplify any flagged bias."""
    weights = [1.0] * len(pairs)
    notes: list[str] = []

    for det_name, det in results.items():
        if det.score < det.threshold:
            continue
        # Mark all pairs in this detector's flagged examples with lower weight
        flagged_ids = {ex.pair_id for ex in det.examples}
        for i, p in enumerate(pairs):
            if p.id in flagged_ids:
                weights[i] *= 0.5  # halve weight
        notes.append(
            f"reweighted {len(flagged_ids)} pairs flagged by {det_name} "
            f"(score={det.score:.2f}, weight × 0.5)"
        )

    return DebiasedResult(
        strategy="reweight",
        pairs=list(pairs),
        weights=weights,
        n_dropped=0,
        n_reswapped=0,
        notes=notes,
    )


def _reswap(
    pairs: list[PreferencePair],
    results: dict[str, DetectorResult],
    seed: int = 42,
) -> DebiasedResult:
    """For position bias: swap A/B labels on 50% of pairs at random."""
    pos = results.get("position")
    if pos is None or pos.score < pos.threshold:
        return DebiasedResult(
            strategy="reswap",
            pairs=list(pairs),
            weights=[1.0] * len(pairs),
            n_dropped=0,
            n_reswapped=0,
            notes=["no position bias detected; no reswap needed"],
        )

    rng = random.Random(seed)
    out: list[PreferencePair] = []
    reswapped = 0
    for p in pairs:
        if rng.random() < 0.5:
            out.append(
                PreferencePair(
                    id=p.id,
                    prompt=p.prompt,
                    response_a=p.response_b,
                    response_b=p.response_a,
                    chosen="b" if p.chosen == "a" else "a",
                    metadata={**p.metadata, "reswapped": True},
                )
            )
            reswapped += 1
        else:
            out.append(p)
    return DebiasedResult(
        strategy="reswap",
        pairs=out,
        weights=[1.0] * len(out),
        n_dropped=0,
        n_reswapped=reswapped,
        notes=[f"reswapped {reswapped} pairs to break position correlation"],
    )


def _filter(
    pairs: list[PreferencePair],
    results: dict[str, DetectorResult],
    top_pct: float = 0.10,
) -> DebiasedResult:
    """Drop the top `top_pct` most biased pairs."""
    if not pairs:
        return DebiasedResult(
            strategy="filter",
            pairs=[],
            weights=[],
            n_dropped=0,
            n_reswapped=0,
            notes=[],
        )

    # Per-pair "bias score" = number of detectors that flagged it as an example
    bias_count: dict[str, int] = {p.id: 0 for p in pairs}
    for det in results.values():
        if det.score < det.threshold:
            continue
        for ex in det.examples:
            if ex.pair_id in bias_count:
                bias_count[ex.pair_id] += 1

    n_to_drop = max(1, int(len(pairs) * top_pct))
    sorted_ids = sorted(bias_count.items(), key=lambda t: t[1], reverse=True)
    drop_ids = {pid for pid, _ in sorted_ids[:n_to_drop]}

    kept = [p for p in pairs if p.id not in drop_ids]
    return DebiasedResult(
        strategy="filter",
        pairs=kept,
        weights=[1.0] * len(kept),
        n_dropped=len(drop_ids),
        n_reswapped=0,
        notes=[f"dropped {len(drop_ids)} pairs ({top_pct*100:.0f}% of input) most flagged by detectors"],
    )


def debias(
    pairs: list[PreferencePair],
    results: dict[str, DetectorResult],
    strategy: Strategy = "reweight",
) -> DebiasedResult:
    """Apply the chosen debiasing strategy."""
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    if strategy == "reweight":
        return _reweight(pairs, results, tokenizer)
    if strategy == "reswap":
        return _reswap(pairs, results)
    if strategy == "filter":
        return _filter(pairs, results)
    raise ValueError(f"Unknown strategy: {strategy}")


def to_jsonl(
    pairs: list[PreferencePair],
    weights: list[float] | None = None,
    path: str | None = None,
) -> str:
    """Serialize the (optionally weighted) dataset to JSONL. Returns the string."""
    import json

    weights = weights or [1.0] * len(pairs)
    lines = []
    for p, w in zip(pairs, weights):
        d = p.to_dict()
        d["weight"] = w
        lines.append(json.dumps(d))
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text
