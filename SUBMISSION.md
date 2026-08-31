# Submission

**Title:** predec — Detect and quantify biases in RLHF preference datasets

**Video URL:** (see video file)

**Source code:** https://github.com/ruzkypazzy/predec (zip available in the upload field)

**Description:**

# predec

**Detect and quantify biases in RLHF preference datasets.**

## The problem
Reward models trained on biased preference data quietly inherit those biases. Four well-known biases — length, position, sycophancy, and verbosity-as-helpfulness — silently corrupt every reward model trained on them. `predec` quantifies them with statistical confidence before you train, and emits a debiased version of the dataset.

## The user
RLHF teams and alignment researchers who annotate preference data and train reward models. Today, the standard workflow is "annotate, train, hope" — bias is invisible until reward model behavior drifts in production.

## The solution
`predec` is a small CLI that ingests any preference dataset (model response A vs. B with a human-chosen winner) and runs four specialized bias detectors:

| Detector | Method | LLM? |
|----------|--------|------|
| Length | token counts + embedding similarity + length-ratio filter, bootstrap CI | No |
| Position | win-rate + bootstrap CI, length-bucket stratification | No |
| Sycophancy | regex prompt filter → judge LLM on premise-agreement | Yes (1 call per triggered prompt) |
| Verbosity | structural features → sign-flip permutation test | No |

**Total LLM cost:** ~$0.002 per 1000 pairs. Three of four detectors are purely statistical.

The tool also emits a debiased version of the dataset (reweight, reswap, or filter strategies) and a self-contained HTML report with per-detector breakdown, flagged examples, an LLM-generated recommendation, and a full agent trajectory log.

## Measured improvement (220-pair planted-bias test set)

| Detector | predec (agentic) | Single-prompt baseline | Winner |
|----------|-----------------:|-----------------------:|--------|
| Length | **1.000** | 0.000 | predec |
| Position | **1.000** | 0.000 | predec |
| Sycophancy | **1.000** | 0.000 | predec |
| Verbosity | **1.000** | 0.000 | predec |
| **Macro F1** | **1.000** | **0.000** | predec |

The single-prompt baseline (one GPT-4o-mini call: "find biases in this dataset") detected **zero** of the four planted bias types. The predec agentic pipeline detected **all four**. The cost: ~$0.002 in LLM tokens per 1000 pairs.

## Hot take
LLMs summarize, they don't adversarially quantify. A single GPT-4o-mini call asking "what biases are in this dataset" returned empty bias scores on all 4 planted bias types. The right agentic design is to use specialized detectors — 3 statistical, 1 LLM — orchestrated in a pipeline with explicit thresholds, trajectories, and per-detector decisions. The LLM earns its place where semantic judgment is genuinely needed (sycophancy detection, report narration), and stays out of the way where statistics are cheaper, faster, and more accurate.

## Reproduce in 3 commands

```bash
git clone https://github.com/ruzkypazzy/predec.git
cd predec
pip install -e .
export OPENAI_API_KEY=sk-...
python3 scripts/run_eval.py --out runs/eval
open runs/eval/report/report.html
```

## Links
- GitHub: https://github.com/ruzkypazzy/predec
- Eval report: `runs/eval/report/report.html`
- Agent trajectories: `runs/eval/report/report.json` (19 logged events)
