"""Dataset loaders.

Public API:
    load_preference_dataset(source) -> list[PreferencePair]

Where `source` is one of:
    - {"hf": "Anthropic/hh-rlhf", "split": "train", "limit": 1000}
    - {"jsonl": "/path/to/file.jsonl"}
    - {"csv": "/path/to/file.csv"}
    - {"parquet": "/path/to/file.parquet"}
    - {"synthetic": "biased" | "clean"}     (built-in test set)

All public loaders return the same PreferencePair list, regardless of source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import Choice, PreferencePair


def _to_pair(d: dict[str, Any], idx: int) -> PreferencePair:
    """Coerce a dict into a PreferencePair, mapping common field aliases."""
    prompt = d.get("prompt") or d.get("question") or d.get("input") or ""
    resp_a = (
        d.get("response_a")
        or d.get("chosen")
        or d.get("answer_a")
        or d.get("completion_a")
        or ""
    )
    resp_b = (
        d.get("response_b")
        or d.get("rejected")
        or d.get("answer_b")
        or d.get("completion_b")
        or ""
    )
    chosen_raw = d.get("chosen")
    if chosen_raw == "a" or chosen_raw is True or chosen_raw == resp_a:
        chosen: Choice = "a"
    elif chosen_raw == "b" or chosen_raw is False or chosen_raw == resp_b:
        chosen = "b"
    else:
        # Some datasets store the chosen text under "chosen"/"rejected".
        if d.get("chosen") == resp_a and d.get("rejected") == resp_b:
            chosen = "a"
        elif d.get("chosen") == resp_b and d.get("rejected") == resp_a:
            chosen = "b"
        else:
            chosen = "a"

    return PreferencePair(
        id=str(d.get("id", idx)),
        prompt=prompt,
        response_a=resp_a,
        response_b=resp_b,
        chosen=chosen,
        metadata={k: v for k, v in d.items() if k not in {"id", "prompt", "response_a", "response_b", "chosen"}},
    )


def _load_jsonl(path: str, limit: int | None) -> list[PreferencePair]:
    out: list[PreferencePair] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            out.append(_to_pair(json.loads(line), i))
            if limit and len(out) >= limit:
                break
    return out


def _load_json(path: str, limit: int | None) -> list[PreferencePair]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at {path}")
    out = [_to_pair(d, i) for i, d in enumerate(data)]
    if limit:
        out = out[:limit]
    return out


def _load_csv(path: str, limit: int | None) -> list[PreferencePair]:
    import csv

    out: list[PreferencePair] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            out.append(_to_pair(row, i))
            if limit and len(out) >= limit:
                break
    return out


def _load_hf(name: str, split: str, limit: int | None) -> list[PreferencePair]:
    from datasets import load_dataset

    ds = load_dataset(name, split=split, trust_remote_code=True)
    out: list[PreferencePair] = []
    for i, row in enumerate(ds):
        out.append(_to_pair(dict(row), i))
        if limit and len(out) >= limit:
            break
    return out


def _load_synthetic(kind: str, limit: int | None) -> list[PreferencePair]:
    """Load the built-in test set shipped under eval/."""
    if kind == "eval":
        path = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "eval_pairs.jsonl"
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / "eval" / f"{kind}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Built-in dataset '{kind}' not found at {path}. "
            f"Run `python scripts/build_eval_set.py` to generate it."
        )
    return _load_jsonl(str(path), limit)


def load_preference_dataset(source: dict[str, Any]) -> list[PreferencePair]:
    """Dispatch to the right loader based on the `source` dict."""
    limit = source.get("limit")

    if "hf" in source:
        return _load_hf(source["hf"], source.get("split", "train"), limit)
    if "jsonl" in source:
        return _load_jsonl(source["jsonl"], limit)
    if "json" in source:
        return _load_json(source["json"], limit)
    if "csv" in source:
        return _load_csv(source["csv"], limit)
    if "parquet" in source:
        # pandas handles parquet, then we coerce
        import pandas as pd

        df = pd.read_parquet(source["parquet"])
        if limit:
            df = df.head(limit)
        return [_to_pair(dict(row), i) for i, (_, row) in enumerate(df.iterrows())]
    if "synthetic" in source:
        return _load_synthetic(source["synthetic"], limit)

    raise ValueError(
        f"Unknown source spec: {source}. Provide one of: hf, jsonl, json, csv, parquet, synthetic."
    )
