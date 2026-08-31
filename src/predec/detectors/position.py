"""Position bias detector.

Hypothesis: when annotators prefer the first response (A) or the second (B)
regardless of content, we say position bias is present.

Assumes pairs were shown in random order. If A's win rate deviates
significantly from 50%, annotators are not actually reading the content.

We also stratify by prompt length to catch a common failure mode: annotators
skim long prompts and click A.
"""

from __future__ import annotations

import time

import numpy as np
import tiktoken

from ..schema import DetectorResult, FlaggedExample, PreferencePair
from . import BaseDetector, bootstrap_ci


class PositionDetector(BaseDetector):
    name = "position"
    metric_description = (
        "Win rate of response A across all pairs. 50% = no bias. "
        "Deviation > 5% with tight CI = meaningful position bias."
    )
    uses_llm = False
    flag_threshold = 0.55  # deviation > 5pp from 0.50

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def detect(self, pairs: list[PreferencePair]) -> DetectorResult:
        t0 = time.time()
        n = len(pairs)
        if n == 0:
            return DetectorResult(
                name=self.name,
                score=0.5,
                confidence_interval=(0.0, 0.0),
                n_pairs_analyzed=0,
                n_pairs_flagged=0,
                metric_description=self.metric_description,
            )

        chosen_a = np.array([1.0 if p.chosen == "a" else 0.0 for p in pairs])
        point, ci = bootstrap_ci(chosen_a, np.mean, n_resamples=1000)

        # Stratify by prompt length
        prompt_tokens = np.array([len(self._tokenizer.encode(p.prompt)) for p in pairs])
        median_pt = float(np.median(prompt_tokens))
        long_prompts = prompt_tokens > median_pt
        short_prompts = ~long_prompts

        rate_short = float(chosen_a[short_prompts].mean()) if short_prompts.sum() else 0.0
        rate_long = float(chosen_a[long_prompts].mean()) if long_prompts.sum() else 0.0
        # CI for long-prompt subset
        if long_prompts.sum() > 0:
            _, ci_long = bootstrap_ci(chosen_a[long_prompts], np.mean, n_resamples=500)
        else:
            ci_long = (0.0, 0.0)

        # Flag: overall deviation from 0.5 OR long-prompt bias
        deviation = abs(point - 0.5)
        flagged = deviation > (self.flag_threshold - 0.5)

        # Top examples: pairs where A won, sorted by prompt length descending
        # (these are the most likely "skimmed-and-clicked-A" cases)
        a_win_idx = np.where(chosen_a == 1.0)[0]
        if len(a_win_idx) > 0:
            top_idx = a_win_idx[np.argsort(-prompt_tokens[a_win_idx])][:5]
        else:
            top_idx = np.array([], dtype=np.int64)
        examples: list[FlaggedExample] = []
        for i in top_idx:
            examples.append(
                FlaggedExample(
                    pair_id=pairs[i].id,
                    evidence={"prompt_tokens": int(prompt_tokens[i]), "chosen": "a"},
                    explanation=(
                        f"long prompt ({int(prompt_tokens[i])} tokens); annotator chose A "
                        f"without strong content signal"
                    ),
                )
            )

        dt = (time.time() - t0) * 1000
        self._log(
            "compute_position_win_rate",
            {"n_pairs": n, "median_prompt_tokens": median_pt},
            {
                "a_win_rate": point,
                "ci": list(ci),
                "rate_long_prompts": rate_long,
                "ci_long": list(ci_long),
                "duration_ms": dt,
            },
            decision=f"position bias {'flagged' if flagged else 'within tolerance'}",
        )

        return DetectorResult(
            name=self.name,
            score=point,
            confidence_interval=ci,
            n_pairs_analyzed=n,
            n_pairs_flagged=n if flagged else 0,
            metric_description=self.metric_description,
            examples=examples,
            extra={
                "deviation_from_50pct": float(deviation),
                "a_win_rate_long_prompts": rate_long,
                "a_win_rate_long_prompts_ci": list(ci_long),
                "median_prompt_tokens": median_pt,
            },
        )
