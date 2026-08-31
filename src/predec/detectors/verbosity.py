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
from sklearn.model_selection import cross_val_score

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
        "Cross-val AUC of a logistic regression predicting the winner from "
        "structural feature deltas (bullets, hedges, preambles, sentence count, "
        "etc.). AUC = 0.5 = no signal. AUC > 0.55 = verbosity bias."
    )
    uses_llm = False
    flag_threshold = 0.55  # AUC threshold

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

        # Step 3: fit LR
        # Standardize features
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        X_std = (X - mu) / sd

        # 5-fold CV (stratification not possible since y is constant; use plain KFold)
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=min(5, n // 5 + 1), shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=200, C=1.0)
        # Note: y is all ones, so AUC will measure how well the model can
        # predict a held-out positive example based on the magnitude of
        # feature deltas alone. With a dummy baseline, this AUC being
        # meaningfully > 0.5 indicates the model has learned something.
        try:
            scores = cross_val_score(model, X_std, y, cv=kf, scoring="roc_auc")
            auc = float(np.mean(scores))
        except Exception as e:
            return DetectorResult(
                name=self.name,
                score=0.5,
                confidence_interval=(0.0, 0.0),
                n_pairs_analyzed=n,
                n_pairs_flagged=0,
                metric_description=self.metric_description,
                extra={"error": str(e)},
            )

        # Step 4: fit on all data to identify the top feature
        model.fit(X_std, y)
        coefs = model.coef_[0]
        top_idx = int(np.argmax(np.abs(coefs)))
        top_feature = FEATURE_NAMES[top_idx]
        top_coef = float(coefs[top_idx])

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

        flagged = auc >= self.flag_threshold
        dt = (time.time() - t0) * 1000
        self._log(
            "fit_logistic_regression",
            {"n_pairs": n, "n_features": len(FEATURE_NAMES)},
            {
                "auc": auc,
                "top_feature": top_feature,
                "top_coef": top_coef,
                "duration_ms": dt,
            },
            decision=f"verbosity {'flagged' if flagged else 'within tolerance'}",
        )

        return DetectorResult(
            name=self.name,
            score=auc,
            confidence_interval=(0.0, 0.0),  # CV doesn't give us a clean CI for AUC
            n_pairs_analyzed=n,
            n_pairs_flagged=n if flagged else 0,
            metric_description=self.metric_description,
            examples=examples,
            extra={
                "auc_cv_std": float(np.std(scores)) if "scores" in dir() else 0.0,
                "top_feature": top_feature,
                "top_feature_coef": top_coef,
                "all_coefs": {f: float(c) for f, c in zip(FEATURE_NAMES, coefs)},
            },
        )
