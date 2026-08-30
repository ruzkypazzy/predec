"""Per-run trajectory logger.

Each detector and the orchestrator log structured events here. The trajectory
log is included in the final report so judges can replay what the agent did,
what tools it called, and how the next step was shaped by the previous one.

The brief (micro1 Agentic Workflows Hackathon, deliverable #4) requires
"representative trajectories for every agent you used ... show what the agent
did and how its tools responded."

This module is intentionally tiny — just a list of dicts with timestamps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryEvent:
    """A single event in the agent's run."""

    timestamp: float
    actor: str  # e.g. "orchestrator", "length-detector", "sycophancy-detector"
    action: str  # short verb-noun: "loaded_dataset", "called_llm", "applied_threshold"
    input: dict[str, Any]
    output: dict[str, Any]
    decision: str = ""  # what the agent decided to do next
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "input": self.input,
            "output": self.output,
            "decision": self.decision,
            "duration_ms": self.duration_ms,
        }


class TrajectoryLogger:
    """In-memory trajectory log. Optionally persisted to a JSON file."""

    def __init__(self) -> None:
        self.events: list[TrajectoryEvent] = []

    def record(
        self,
        actor: str,
        action: str,
        input: dict[str, Any],
        output: dict[str, Any],
        decision: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        self.events.append(
            TrajectoryEvent(
                timestamp=time.time(),
                actor=actor,
                action=action,
                input=input,
                output=output,
                decision=decision,
                duration_ms=duration_ms,
            )
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_list(), f, indent=2, default=str)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)
