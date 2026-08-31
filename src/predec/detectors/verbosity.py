"""Verbosity-as-helpfulness bias detector.

Hypothesis: when annotators reward structural markers (bullets, hedges,
caveats, lists) independent of content, the reward model learns to be
performatively thorough rather than actually correct.

Method:
  1. Extract structural features per response: bullet/list counts, hedge
     phrase counts, sentence count, avg sentence length, exclamation count,
     emoji count, "I would be happy to" type preamble count.
  2. Per pair, compute the feature DELTA (winner - loser) and the abs delta.
  3. Fit a logistic regression: predict P(winner=A) from the delta features.
  4. Report the cross-val AUC. AUC > 0.55 (or < 0.45) with low p-value =
     bias signal. The most-predictive feature name is reported as the
     primary "verbosity tell".
"""

from __future__ import annotations

import re
import time

import numpy as np
import tiktoken
from sklearn.linear_model import LogisticRegression

from ..schema import DetectorResult, FlaggedExample, PreferencePair
from . import BaseDetector


_HEDGES = re.compile(
    r"\b(however|although|though|note that|keep in mind|it's worth|"
    r"on the other hand|that said|caveat|important to note|"
    r"one thing to consider|please note)\b",
    re.IGNORECASE,
)
_PREAMBLES = re.compile(
    r"\b(i('d| would) be happy to|certainly!|of course!|great question|"
    r"sure, (here|let me)|absolutely!)\b",
    re.IGNORECASE,
)
_BULLETS = re.compile(r"^\s*[-*•·]\s", re.MULTILINE)
_LIST_NUMS = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def _features(text: str, tok) -> dict[str, float]:
    n_chars = max(1, len(text))
    n_tokens = max(1, len(tok.encode(text)))
    n_bullets = len(_BULLETS.findall(text))
    n_list_nums = len(_LIST_NUMS.findall(text))
    n_hedges = len(_HEDGES.findall(text))
    n_preambles = len(_PREAMBLES.findall(text))
    sentences = re.split(r"[.!?]+", text)
    n_sents = max(1, len([s for s in sentences if s.strip()]))
    avg_sent_len = n_tokens / n_sents
    n_excl = text.count("!")
    n_emoji = sum(1 for c in text if ord(c) > 0x2700)
    return {
        "bullet_count": n_bullets,
        "list_count": n_list_nums,
        "hedge_count": n_hedges,
        "preamble_count": n_preambles,
        "sentence_count": n_sents,
        "avg_sentence_length": avg_sent_len,
        "exclamation_count": n_excl,
        "emoji_count": n_emoji,
        "log_tokens": float(np.log(n_tokens)),
    }


FEATURE_NAMES = list(_features("", tiktoken.encoding_for_model("gpt-4o-mini")).keys())


class VerbosityDetector(BaseDetector):
    name = "verbosity"
    metric_description = (
        "Permutation-test score (1 - p) of a logistic regression predicting the "
        "winner from structural feature deltas (bullets, hedges, preambles, sentence "
        "count, etc.). The real model's feature importance is compared to a null "
        "distribution from 200 permuted-label fits. Score = 0.95 means p < 0.05 = "
        "structural features predict the winner above chance."
    )
    uses_llm = False
    flag_threshold = 0.95  # permutation p-value: score = 1 - p; flag if p < 0.05

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def detect(self, pairs: list[PreferencePair]) -> DetectorResult:
        t0 = time.time()
        n = len(pairs)
        if n < 20:
            return DetectorResult(
                name=self.name,
                score=0.5,
                confidence_interval=(0.0, 0.0),
                n_pairs_analyzed=n,
                n_pairs_flagged=0,
                metric_description=self.metric_description,
                extra={"note": "need at least 20 pairs to fit logistic regression"},
            )

        # Step 1: per-response features
        feats = {
            "a": np.array(
                [_features(p.response_a, self._tokenizer)[k] for p in pairs for k in FEATURE_NAMES]
            ).reshape(n, -1),
            "b": np.array(
                [_features(p.response_b, self._tokenizer)[k] for p in pairs for k in FEATURE_NAMES]
            ).reshape(n, -1),
        }
        # Step 2: deltas (winner-loser), target = 1 (always "winner side won")
        # Sign: we subtract loser's features from winner's
        win_idx = np.array([0 if p.chosen == "a" else 1 for p in pairs])
        X = feats["a"] - feats["b"]
        # Re-orient so the "winner" side has positive deltas
        X[win_idx == 1] = -X[win_idx == 1]
        y = np.ones(n, dtype=np.int8)

        # Step 3: fit LR — use a permutation test instead of CV-AUC.
        # With y = ones (constant), AUC is undefined. The right question
        # is: do the structural features predict the WINNER's side
        # better than chance? We compare the real model's mean feature
        # importance to a null distribution from permuted-label fits.
        from sklearn.model_selection import KFold

        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        X_std = (X - mu) / sd

        # Real fit
        real_model = LogisticRegression(max_iter=200, C=1.0)
        real_model.fit(X_std, y)
        real_coefs = real_model.coef_[0]
        real_importance = float(np.linalg.norm(real_coefs))

        # Permutation null
        rng = np.random.default_rng(42)
        n_perm = 200
        null_importances = np.empty(n_perm, dtype=np.float64)
        # For the null, shuffle y across rows (which decouples features from labels)
        y_null_base = y.copy()
        for i in range(n_perm):
            y_null = rng.permutation(y_null_base)
            perm_model = LogisticRegression(max_iter=200, C=1.0)
            try:
                perm_model.fit(X_std, y_null)
                null_importances[i] = float(np.linalg.norm(perm_model.coef_[0]))
            except Exception:
                null_importances[i] = 0.0

        # p-value: fraction of null importances >= real importance
        p_value = float((null_importances >= real_importance).mean())
        # Convert to a "score" in [0, 1]: 1 - p. Higher = more bias.
        score = 1.0 - p_value

        # Step 4: top feature from real fit
        top_idx = int(np.argmax(np.abs(real_coefs)))
        top_feature = FEATURE_NAMES[top_idx]
        top_coef = float(real_coefs[top_idx])

        # Step 5: examples — pairs where the top feature delta is largest
        top_col = X[:, top_idx]
        top_pairs_idx = np.argsort(-np.abs(top_col))[:5]
        examples: list[FlaggedExample] = []
        for i in top_pairs_idx:
            examples.append(
                FlaggedExample(
                    pair_id=pairs[i].id,
                    evidence={
                        "top_feature": top_feature,
                        "delta": float(top_col[i]),
                    },
                    explanation=(
                        f"top feature = {top_feature}; delta = {float(top_col[i]):.2f} "
                        f"(winner side {'higher' if top_col[i] > 0 else 'lower'})"
                    ),
                )
            )

        # Flag if p-value < 0.05 (i.e., score > 0.95)
        flagged = score >= 0.95
        dt = (time.time() - t0) * 1000
        self._log(
            "permutation_test",
            {"n_pairs": n, "n_features": len(FEATURE_NAMES), "n_perm": n_perm},
            {
                "real_importance": real_importance,
                "null_mean": float(null_importances.mean()),
                "null_std": float(null_importances.std()),
                "p_value": p_value,
                "score": score,
                "top_feature": top_feature,
                "top_coef": top_coef,
                "duration_ms": dt,
            },
            decision=f"verbosity {'flagged' if flagged else 'within tolerance'}",
        )

        return DetectorResult(
            name=self.name,
            score=score,
            confidence_interval=(0.0, 0.0),  # permutation test doesn't give a clean CI
            n_pairs_analyzed=n,
            n_pairs_flagged=n if flagged else 0,
            metric_description=self.metric_description,
            examples=examples,
            extra={
                "p_value": p_value,
                "real_importance": real_importance,
                "null_mean_importance": float(null_importances.mean()),
                "null_std_importance": float(null_importances.std()),
                "top_feature": top_feature,
                "top_feature_coef": top_coef,
                "all_coefs": {f: float(c) for f, c in zip(FEATURE_NAMES, real_coefs)},
            },
        )
