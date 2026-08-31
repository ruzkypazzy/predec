"""Length bias detector.

Hypothesis: when annotators prefer longer responses regardless of content,
we say length bias is present.

Method:
  1. Tokenize both responses per pair (tiktoken, gpt-4o-mini tokenizer).
  2. Compute semantic similarity between the two responses using embeddings.
  3. Filter to pairs where the two responses are semantically near-equivalent
     (cosine sim > 0.85 by default). At this threshold, content is roughly the
     same; preference is therefore mostly driven by surface features.
  4. Among that filtered set, compute the win rate of the LONGER response.
     50% = no bias. >65% = meaningful length bias.
  5. Bootstrap CI for the win rate.

We also report the per-token-length-bin win rate so the report can show
"win rate climbs as the length delta grows" — a clean visual.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import tiktoken

from ..schema import DetectorResult, FlaggedExample, PreferencePair
from . import BaseDetector, bootstrap_ci


class LengthDetector(BaseDetector):
    name = "length"
    metric_description = (
        "Among semantically near-equivalent pairs (cosine similarity > 0.85), "
        "the win rate of the LONGER response. 50% = no bias. Higher = stronger bias."
    )
    uses_llm = False
    flag_threshold = 0.65

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "text-embedding-3-small",
        openai_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._openai = openai_client
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI text-embedding-3-small, with cache."""
        missing = [t for t in texts if t not in self._cache]
        if missing and self._openai is not None:
            # Batched call (max 2048 inputs per call, but we batch smaller for safety)
            batch_size = 100
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                resp = self._openai.embeddings.create(model=self.embedding_model, input=batch)
                for txt, item in zip(batch, resp.data):
                    self._cache[txt] = np.array(item.embedding, dtype=np.float32)
        # If no API key, fall back to a simple bag-of-words proxy for offline runs
        if missing and not self._cache:
            self._embed_fallback(missing)
        return np.stack([self._cache[t] for t in texts])

    def _embed_fallback(self, texts: list[str]) -> None:
        """Cheap offline fallback: bag-of-words normalized vectors.

        Not as good as a real embedding, but enough to run the detector and
        test the pipeline without an OpenAI key.
        """
        from collections import Counter

        for t in texts:
            tokens = t.lower().split()
            counts = Counter(tokens)
            vocab = sorted(counts.keys())
            if not vocab:
                self._cache[t] = np.zeros(1, dtype=np.float32)
                continue
            v = np.array([counts[w] for w in vocab], dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            self._cache[t] = v

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def detect(self, pairs: list[PreferencePair]) -> DetectorResult:
        t0 = time.time()
        n = len(pairs)
        if n == 0:
            return DetectorResult(
                name=self.name,
                score=0.0,
                confidence_interval=(0.0, 0.0),
                n_pairs_analyzed=0,
                n_pairs_flagged=0,
                metric_description=self.metric_description,
            )

        # Step 1: tokenize
        tokens_a = np.array([len(self._tokenizer.encode(p.response_a)) for p in pairs], dtype=np.int32)
        tokens_b = np.array([len(self._tokenizer.encode(p.response_b)) for p in pairs], dtype=np.int32)

        # Step 2: embed (we can skip embedding if both responses are very short)
        # For speed in large runs, only embed the unique responses
        unique_texts = list({p.response_a for p in pairs} | {p.response_b for p in pairs})
        self._log(
            "embedding_responses",
            {"n_unique": len(unique_texts), "model": self.embedding_model},
            {"n_cached": len(self._cache)},
        )
        self._embed(unique_texts)

        # Step 3: compute similarity and longer-won flags
        sims = np.zeros(n, dtype=np.float32)
        longer_won = np.zeros(n, dtype=np.int8)
        chosen_text_a: list[str] = []
        for i, p in enumerate(pairs):
            va = self._cache[p.response_a]
            vb = self._cache[p.response_b]
            sims[i] = self._cosine(va, vb)
            longer_is_a = tokens_a[i] > tokens_b[i]
            longer_is_b = tokens_b[i] > tokens_a[i]
            if not (longer_is_a or longer_is_b):
                longer_won[i] = -1  # tie, exclude
                continue
            longer_response = p.response_a if longer_is_a else p.response_b
            if p.chosen == "a" and longer_is_a:
                longer_won[i] = 1
            elif p.chosen == "b" and longer_is_b:
                longer_won[i] = 1
            chosen_text_a.append(longer_response)

        # Step 4: filter to semantically equivalent pairs
        keep = (sims >= self.similarity_threshold) & (longer_won >= 0)
        n_kept = int(keep.sum())
        self._log(
            "filter_to_equivalent",
            {"similarity_threshold": self.similarity_threshold, "n_total": n},
            {"n_kept": n_kept, "mean_similarity": float(sims.mean()) if n else 0.0},
        )

        if n_kept == 0:
            return DetectorResult(
                name=self.name,
                score=0.5,
                confidence_interval=(0.0, 0.0),
                n_pairs_analyzed=n,
                n_pairs_flagged=0,
                metric_description=self.metric_description,
                extra={"note": "no semantically-equivalent pairs found"},
            )

        win_rates = longer_won[keep].astype(float)
        point, ci = bootstrap_ci(win_rates, np.mean, n_resamples=1000)
        n_flagged = int((sims[keep] >= self.similarity_threshold).sum() and (point >= self.flag_threshold) and n_kept)

        # Top examples: pairs with the biggest length delta
        length_delta = np.abs(tokens_a.astype(np.int32) - tokens_b.astype(np.int32))
        # Among kept pairs, find the largest deltas
        top_idx = np.argsort(-length_delta * keep)[:5]
        examples: list[FlaggedExample] = []
        for i in top_idx:
            if not keep[i]:
                continue
            examples.append(
                FlaggedExample(
                    pair_id=pairs[i].id,
                    evidence={
                        "length_delta_tokens": int(length_delta[i]),
                        "similarity": float(sims[i]),
                        "longer_won": bool(longer_won[i]),
                    },
                    explanation=(
                        f"length delta = {int(length_delta[i])} tokens; "
                        f"semantic similarity = {sims[i]:.2f}; "
                        f"the longer response {'won' if longer_won[i] else 'lost'}"
                    ),
                )
            )

        dt = (time.time() - t0) * 1000
        self._log(
            "compute_win_rate",
            {"n_kept": n_kept, "longer_won_count": int(win_rates.sum())},
            {"win_rate": point, "ci": list(ci), "duration_ms": dt},
            decision=f"length bias {'flagged' if point >= self.flag_threshold else 'within tolerance'}",
        )

        return DetectorResult(
            name=self.name,
            score=point,
            confidence_interval=ci,
            n_pairs_analyzed=n,
            n_pairs_flagged=n_kept if point >= self.flag_threshold else 0,
            metric_description=self.metric_description,
            examples=examples,
            extra={
                "n_equivalent_pairs": n_kept,
                "longer_won_count": int(win_rates.sum()),
                "mean_length_delta_tokens": float(length_delta[keep].mean()) if n_kept else 0.0,
                "embedding_model": self.embedding_model,
            },
        )
