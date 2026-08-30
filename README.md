# predec

**Detect and quantify biases in RLHF preference datasets.**

`predec` is a small CLI that ingests any preference dataset (model response A vs. B with a human-chosen winner) and reports four well-known reward-modeling biases with statistical confidence:

- **Length bias** — annotators prefer longer responses regardless of content
- **Position bias** — annotators prefer response A (or B) regardless of content
- **Sycophancy bias** — annotators prefer responses that agree with a (often false) premise
- **Verbosity-as-helpfulness bias** — annotators reward structural markers (bullets, hedges, caveats) independent of content

The tool also emits a debiased version of the dataset (reweight, reswap, or filter strategies) and a self-contained HTML report.

## Install

```bash
git clone https://github.com/<OWNER>/predec.git
cd predec
pip install -e .
```

## Quick start

```bash
# Built-in: load a public HuggingFace dataset and run all 4 detectors
predec detect --dataset anthropic/hh-rlhf --limit 1000 --out runs/exp1

# Custom JSONL
predec detect --input data/my_pairs.jsonl --out runs/exp1

# Inspect the report
open runs/exp1/report.html
```

## The 4 detectors

| Detector | Method | LLM? |
|----------|--------|------|
| Length | token counts + embedding similarity, bootstrap CI | No |
| Position | win-rate + bootstrap CI, length-bucket stratification | No |
| Sycophancy | regex prompt filter → judge LLM on premise-agreement | **Yes** (1 call/1000 pairs) |
| Verbosity | structural feature extraction → logistic regression AUC | No |

**Total LLM cost:** ~$0.50 per 1000 pairs analyzed. Three of four detectors are purely statistical.

## Why this exists

Reward models trained on biased preference data quietly inherit those biases. `predec` quantifies them before training so you can decide whether to debias, retrain annotators, or regenerate data.

See `docs/methodology.md` for details on each detector and the math.

## Evaluation

Run the built-in eval against 200 hand-crafted pairs with known planted biases:

```bash
predec eval --out runs/eval
```

This reports F1 of the full agentic pipeline vs. a single-prompt baseline.

## License

MIT
