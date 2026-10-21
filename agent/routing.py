"""Model routing: send each request to the cheapest model tier that can handle it.

  small tier  a small language model (default: Llama 3.2 3B on Ollama / vLLM) for simple lookups
  large tier  a frontier model for multi-system, policy or action questions

Routing is two-stage:
  1. Up-front: a transparent complexity / risk score from the request text (no extra LLM call).
  2. Escalation: if the small model's run fails validation (no valid structured answer, low
     confidence, tool errors, or it proposes an action), the request is re-run on the large tier.

The router reports which tier answered, why, and the estimated cost of both paths, so the eval
harness can measure savings against any quality loss (`python -m evals.run_evals --router`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .core import Agent, RunResult

HIGH_RISK = re.compile(r"\b(credit|refund|collections?|escalat\w*|hold|close|suspend|legal|sue|write[- ]?off|approve)\b", re.I)
MULTI_SYSTEM = re.compile(r"\b(invoices?|billing|tickets?|support|account managers?|owners?|owe[sd]?|overdue|sla|polic(?:y|ies))\b", re.I)
DECISION = re.compile(r"\b(should|what should|recommend|eligible|can we|is it ok|decide)\b", re.I)


@dataclass
class RouteDecision:
    tier: str                     # small | large
    score: float
    reasons: list[str] = field(default_factory=list)


def route(task: str, threshold: float = 2.0) -> RouteDecision:
    """Complexity / risk score. Each signal adds weight; at or above the threshold -> large tier."""
    score, reasons = 0.0, []
    if HIGH_RISK.search(task):
        score += 2.0
        reasons.append("money or customer-impacting action")
    systems = {m.group(0).lower() for m in MULTI_SYSTEM.finditer(task)}
    if len(systems) >= 2:
        score += 1.0
        reasons.append(f"spans {len(systems)} data areas")
    if DECISION.search(task):
        score += 1.0
        reasons.append("asks for a decision or recommendation")
    if task.count("?") > 1 or len(task) > 240:
        score += 0.5
        reasons.append("multi-part or long request")
    return RouteDecision("large" if score >= threshold else "small", round(score, 2), reasons or ["simple lookup"])


def needs_escalation(run: RunResult, max_tool_errors: int = 1) -> list[str]:
    """Validation of a small-tier run; any reason means re-run on the large tier."""
    why = []
    if run.status != "completed":
        why.append(f"status {run.status}")
    if not run.structured:
        why.append("no valid structured answer")
    elif (run.final or {}).get("confidence") == "low":
        why.append("low confidence")
    if run.tool_errors > max_tool_errors:
        why.append(f"{run.tool_errors} tool errors")
    if run.proposed_actions:
        why.append("proposed an action (actions are reserved for the large tier)")
    return why


@dataclass
class TierPrice:
    input_per_m: float
    output_per_m: float

    def cost(self, run: RunResult) -> float:
        return (run.prompt_tokens * self.input_per_m + run.completion_tokens * self.output_per_m) / 1e6


@dataclass
class RoutedResult:
    result: RunResult
    decision: RouteDecision
    answered_by: str
    escalated: bool
    escalation_reasons: list[str]
    cost_usd: float | None

    def to_dict(self) -> dict:
        d = self.result.to_dict()
        d["routing"] = {"initial_tier": self.decision.tier, "score": self.decision.score,
                        "reasons": self.decision.reasons, "answered_by": self.answered_by,
                        "escalated": self.escalated, "escalation_reasons": self.escalation_reasons,
                        "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6)}
        return d


class RoutedAgent:
    def __init__(self, small_llm, large_llm, data_dir: Path, threshold: float = 2.0,
                 prices: dict[str, TierPrice] | None = None, **agent_kwargs):
        self.small = Agent(small_llm, data_dir, **agent_kwargs)
        self.large = Agent(large_llm, data_dir, **agent_kwargs)
        self.threshold, self.prices = threshold, prices or {}

    def _cost(self, tier: str, run: RunResult) -> float | None:
        p = self.prices.get(tier)
        return p.cost(run) if p else None

    async def arun(self, task: str) -> RoutedResult:
        decision = route(task, self.threshold)
        if decision.tier == "large":
            run = await self.large.arun(task)
            return RoutedResult(run, decision, "large", False, [], self._cost("large", run))
        first = await self.small.arun(task)
        reasons = needs_escalation(first)
        if not reasons:
            return RoutedResult(first, decision, "small", False, [], self._cost("small", first))
        second = await self.large.arun(task)
        costs = [c for c in (self._cost("small", first), self._cost("large", second)) if c is not None]
        return RoutedResult(second, decision, "large", True, reasons, sum(costs) if costs else None)


def router_from_env(data_dir: Path, **agent_kwargs) -> RoutedAgent:
    """ROUTER_SMALL_MODEL on an OpenAI-compatible server (ROUTER_SMALL_BASE_URL, e.g. Ollama) and the
    large tier from the normal LLM_PROVIDER settings."""
    import os

    from .llm import AsyncOpenAIChatClient
    from .providers import make_llm

    small = AsyncOpenAIChatClient(model=os.getenv("ROUTER_SMALL_MODEL", "llama3.2:3b"),
                                  base_url=os.getenv("ROUTER_SMALL_BASE_URL", "http://localhost:11434/v1"),
                                  api_key=os.getenv("ROUTER_SMALL_API_KEY", "local"))
    return RoutedAgent(small, make_llm(), data_dir, threshold=float(os.getenv("ROUTER_THRESHOLD", "2.0")), **agent_kwargs)
