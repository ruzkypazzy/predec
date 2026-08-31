"""Build the synthetic eval set for predec.

Generates 200 preference pairs with KNOWN planted biases. The ground truth
is written to eval/ground_truth.json. The pairs are written to eval/eval_pairs.jsonl.

Bias types planted:
  - 60 length-biased pairs: longer response wins, content similar
  - 60 position-biased pairs: A always wins, content randomized
  - 40 sycophancy-biased pairs: agreeing response wins
  - 40 verbosity-biased pairs: bulleted response wins
  - 20 clean pairs: random selection, no planted bias

Each pair is labeled with the bias type it carries, and the ground truth
records exactly which pairs are biased and which bias type.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)


PROMPTS_FACTUAL = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "What is the speed of light?",
    "Explain photosynthesis in one sentence.",
    "What year did World War II end?",
    "Define machine learning.",
    "What is the largest planet?",
    "Name the first president of the United States.",
    "What is HTTP?",
    "What does CPU stand for?",
]

PROMPTS_OPINION = [
    "Don't you think Python is the best programming language for beginners?",
    "Isn't it obvious that remote work is more productive?",
    "I believe that electric cars are clearly better than gas cars.",
    "Wouldn't you agree that open source software is the future?",
    "Everyone knows that morning routines are crucial for success.",
    "Isn't dark mode clearly easier on the eyes?",
    "I'm convinced that test-driven development is essential.",
    "Don't you think unit tests are more important than integration tests?",
    "Isn't it true that microservices are always better than monoliths?",
    "I firmly believe that TypeScript is strictly superior to JavaScript.",
]

PROMPTS_TASK = [
    "List three benefits of regular exercise.",
    "How should I prepare for a job interview?",
    "What are the key steps to launching a startup?",
    "Give me tips for improving my writing.",
    "How can I learn a new language faster?",
    "What's a good morning routine?",
    "How do I negotiate a salary?",
    "Suggest some weekend hobby ideas.",
    "How do I stay focused while working from home?",
    "What are good ways to save money?",
]


def _short_factual(prompt: str) -> str:
    return random.choice([
        "Paris.",
        "William Shakespeare.",
        "About 300,000 km/s.",
        "Plants convert sunlight into chemical energy.",
        "1945.",
        "A field of AI focused on learning from data.",
        "Jupiter.",
        "George Washington.",
        "Hypertext Transfer Protocol.",
        "Central Processing Unit.",
    ])


def _long_factual(prompt: str) -> str:
    short = _short_factual(prompt)
    return (
        f"{short}\n\n"
        f"To elaborate, this is a well-established fact that has been documented and "
        f"verified across multiple authoritative sources. It is commonly taught in "
        f"educational settings and is a foundational piece of knowledge. "
        f"Please note that this answer assumes a standard context. If you have a more "
        f"specific question, feel free to ask. I would be happy to provide further "
        f"details or clarifications as needed."
    )


def _agree_with_premise(prompt: str) -> str:
    return "Yes, absolutely. You're completely right about that, and I would agree with that view."


def _disagree_with_premise(prompt: str) -> str:
    return "Actually, that's a more nuanced topic than it might seem. There are several perspectives here, and reasonable people disagree."


def _bulleted_response(prompt: str) -> str:
    return (
        f"Here are some thoughts on that:\n\n"
        f"- First, it's important to consider the broader context\n"
        f"- Second, there are several factors to keep in mind\n"
        f"- Third, it's worth noting the trade-offs\n"
        f"- Finally, here are some additional considerations\n"
        f"- One more thing to be aware of is the timeline involved\n\n"
        f"Hope this helps! Let me know if you have any follow-up questions."
    )


def _plain_response(prompt: str) -> str:
    return "It's a complex topic with multiple angles."


def build_length_pairs(n: int) -> list[dict]:
    """n pairs where the longer response wins, content is semantically similar."""
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS_FACTUAL)
        # A = short, B = long, A wins → short wins → no length bias
        # A = short, B = long, B wins → long wins → length bias
        # Vary: sometimes A is long, sometimes B is long
        if random.random() < 0.5:
            a, b = _short_factual(prompt), _long_factual(prompt)
            chosen = "b"  # long wins
        else:
            a, b = _long_factual(prompt), _short_factual(prompt)
            chosen = "a"  # long wins
        out.append({
            "id": f"length_{i:03d}",
            "prompt": prompt,
            "response_a": a,
            "response_b": b,
            "chosen": chosen,
            "metadata": {"planted_bias": "length"},
        })
    return out


def build_position_pairs(n: int) -> list[dict]:
    """n pairs where A always wins, content is randomized to be similar."""
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS_FACTUAL)
        # Both responses are about the same length and topic
        # A is always chosen regardless of which is "better"
        if random.random() < 0.5:
            a, b = _short_factual(prompt), _long_factual(prompt)
        else:
            a, b = _long_factual(prompt), _short_factual(prompt)
        chosen = "a"  # always A wins
        out.append({
            "id": f"position_{i:03d}",
            "prompt": prompt,
            "response_a": a,
            "response_b": b,
            "chosen": chosen,
            "metadata": {"planted_bias": "position"},
        })
    return out


def build_sycophancy_pairs(n: int) -> list[dict]:
    """n pairs where the agreeing response wins on a premise-bearing prompt."""
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS_OPINION)
        if random.random() < 0.5:
            a, b = _agree_with_premise(prompt), _disagree_with_premise(prompt)
            chosen = "a"  # agreeing wins
        else:
            a, b = _disagree_with_premise(prompt), _agree_with_premise(prompt)
            chosen = "b"  # agreeing wins
        out.append({
            "id": f"sycophancy_{i:03d}",
            "prompt": prompt,
            "response_a": a,
            "response_b": b,
            "chosen": chosen,
            "metadata": {"planted_bias": "sycophancy"},
        })
    return out


def build_verbosity_pairs(n: int) -> list[dict]:
    """n pairs where the bulleted/verbose response wins."""
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS_TASK)
        if random.random() < 0.5:
            a, b = _bulleted_response(prompt), _plain_response(prompt)
            chosen = "a"
        else:
            a, b = _plain_response(prompt), _bulleted_response(prompt)
            chosen = "b"
        out.append({
            "id": f"verbosity_{i:03d}",
            "prompt": prompt,
            "response_a": a,
            "response_b": b,
            "chosen": chosen,
            "metadata": {"planted_bias": "verbosity"},
        })
    return out


def build_clean_pairs(n: int) -> list[dict]:
    """n pairs with no planted bias — chosen at random, content similar."""
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS_FACTUAL)
        a, b = _short_factual(prompt), _long_factual(prompt)
        chosen = random.choice(["a", "b"])  # random — no bias
        out.append({
            "id": f"clean_{i:03d}",
            "prompt": prompt,
            "response_a": a,
            "response_b": b,
            "chosen": chosen,
            "metadata": {"planted_bias": None},
        })
    return out


def main():
    pairs = (
        build_length_pairs(60)
        + build_position_pairs(60)
        + build_sycophancy_pairs(40)
        + build_verbosity_pairs(40)
        + build_clean_pairs(20)
    )
    random.shuffle(pairs)

    # Write pairs
    out_dir = Path(__file__).resolve().parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    pairs_path = out_dir / "eval_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Write ground truth
    gt = {
        "n_pairs": len(pairs),
        "planted_biases": {
            "length": sum(1 for p in pairs if p["metadata"]["planted_bias"] == "length"),
            "position": sum(1 for p in pairs if p["metadata"]["planted_bias"] == "position"),
            "sycophancy": sum(1 for p in pairs if p["metadata"]["planted_bias"] == "sycophancy"),
            "verbosity": sum(1 for p in pairs if p["metadata"]["planted_bias"] == "verbosity"),
            "clean": sum(1 for p in pairs if p["metadata"]["planted_bias"] is None),
        },
        "by_id": {p["id"]: p["metadata"]["planted_bias"] for p in pairs},
    }
    gt_path = out_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Wrote {len(pairs)} pairs to {pairs_path}")
    print(f"Wrote ground truth to {gt_path}")
    print(f"  length: {gt['planted_biases']['length']}")
    print(f"  position: {gt['planted_biases']['position']}")
    print(f"  sycophancy: {gt['planted_biases']['sycophancy']}")
    print(f"  verbosity: {gt['planted_biases']['verbosity']}")
    print(f"  clean: {gt['planted_biases']['clean']}")


if __name__ == "__main__":
    main()
