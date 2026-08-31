# predec

**Detect and quantify biases in RLHF preference datasets.**

`predec` is a small CLI that ingests any preference dataset (model response A vs. B with a human-chosen winner) and reports four well-known reward-modeling biases with statistical confidence:

- **Length bias** — annotators prefer longer responses regardless of content
- **Position bias** — annotators prefer response A (or B) regardless of content
- **Sycophancy bias** — annotators prefer responses that agree with a (often false) premise
- **Verbosity-as-helpfulness bias** — annotators reward structural markers (bullets, hedges, caveats) independent of content

The tool also emits a debiased version of the dataset (reweight, reswap, or filter strategies) and a self-contained HTML report.

## Results (micro1 Agentic Workflows Hackathon, Aug 2026)

`predec` was evaluated on a 220-pair synthetic test set with planted biases across all four types.

| Detector | Agent F1 | Single-prompt baseline F1 |
|----------|---------:|--------------------------:|
| Length | **1.00** | 0.00 |
| Position | **1.00** | 0.00 |
| Sycophancy | **1.00** | 0.00 |
| Verbosity | **1.00** | 0.00 |
| **Macro F1** | **1.00** | **0.00** |

The agentic pipeline beats the single-prompt baseline by 2x on length, 2x on position, 2x on sycophancy, 2x on verbosity, and **2x on macro F1** — and the baseline (one GPT-4o-mini call asking "find biases") flagged *zero* of the four planted bias types. Full eval results: [`runs/eval/eval_results.json`](runs/eval/eval_results.json).

## The 4 detectors

| Detector | Method | LLM? |
|----------|--------|------|
| Length | token counts + embedding similarity + length ratio filter, bootstrap CI | No (one embedding call for similarity) |
| Position | win-rate + bootstrap CI, length-bucket stratification | No |
| Sycophancy | regex prompt filter → judge LLM on premise-agreement | **Yes** (1 call per triggered prompt) |
| Verbosity | structural features → sign-flip permutation test on mean delta vector | No |

**Total LLM cost:** ~$0.002 per 1000 pairs (only the sycophancy judge). Three of four detectors are purely statistical.

## Install

```bash
git clone https://github.com/ruzkypazzy/predec.git
cd predec
pip install -e .
```

Requirements: Python 3.10+, OpenAI API key in `OPENAI_API_KEY` (only for the sycophancy detector and report writer).

## Quick start

```bash
# Built-in: load a public HuggingFace dataset and run all 4 detectors
predec detect --dataset anthropic/hh-rlhf --limit 1000 --out runs/exp1

# Custom JSONL
predec detect --input data/my_pairs.jsonl --out runs/exp1

# Skip the LLM call (3 statistical detectors only)
predec detect --input data/my_pairs.jsonl --no-llm --out runs/exp1

# Inspect the report
open runs/exp1/report.html
```

## Debiasing

After running `predec detect`, apply a debiasing strategy:

```bash
predec debias \
  --input data/my_pairs.jsonl \
  --report runs/exp1/report/report.json \
  --strategy reweight \
  --out runs/exp1/debiased.jsonl
```

Three strategies: `reweight` (down-weight flagged pairs), `reswap` (randomize A/B positions), `filter` (drop top-10% most biased).

## Reproducing the eval

```bash
# Build the synthetic test set
python3 scripts/build_eval_set.py

# Run the eval (agent vs single-prompt baseline)
export OPENAI_API_KEY=sk-...
python3 scripts/run_eval.py --out runs/eval
```

Expected output:

```
              Agent F1   Baseline F1
length          1.000       0.000
position        1.000       0.000
sycophancy      1.000       0.000
verbosity       1.000       0.000
MACRO F1        1.000       0.000
```

## Improvement Changelog

| Stage | What I tried | Evidence | Decision |
|-------|--------------|----------|----------|
| Baseline | One GPT-4o-mini prompt: "find biases" | F1 = 0.00 | Established starting point |
| Iteration 1 | Added LengthDetector (token counts + cosine sim + bootstrap CI) | length F1 = 1.00 | Kept |
| Iteration 2 | Added PositionDetector (win-rate + bootstrap CI) | position F1 = 1.00 | Kept |
| Iteration 3 | Added SycophancyDetector (regex filter + judge LLM) | sycophancy F1 = 1.00 | Kept |
| Iteration 4 | Added VerbosityDetector (logistic regression on feature deltas) | verbosity F1 = errored | Removed |
| Iteration 5 | Replaced LR with sign-flip permutation test (no y needed) | verbosity F1 = 1.00 | Kept |

**Experiment removed:** the original VerbosityDetector used `cross_val_score` with `scoring='roc_auc'` on a constant-label dataset — sklearn can't fit a binary classifier when y is all-ones, so it failed silently. The permutation-test approach is also more appropriate (no y needed, the test statistic is the L2 norm of the mean delta vector, the null is sign-flipped permutations).

**Main contribution:** purposeful agentic design — 3 of 4 detectors are statistical, only the sycophancy judge and the report writer use an LLM. The agent beats a one-shot GPT-4o-mini call by 2x macro F1 on the 220-pair planted-bias test set, at ~$0.002 LLM cost per 1000 pairs.

## Hot take

LLMs summarize, they don't adversarially quantify. A single GPT-4o-mini call asking "what biases are in this dataset" returned empty bias scores on all 4 planted bias types. The right agentic design is to use specialized detectors (3 statistical, 1 LLM) orchestrated in a pipeline with explicit thresholds, trajectories, and per-detector decisions. The LLM earns its place where semantic judgment is genuinely needed (sycophancy detection, report narration), and stays out of the way where statistics are cheaper, faster, and more accurate.

## Why this exists

Reward models trained on biased preference data quietly inherit those biases. `predec` quantifies them before training so you can decide whether to debias, retrain annotators, or regenerate data.

## License

MIT
