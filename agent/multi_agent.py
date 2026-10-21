"""Multi-agent system: a supervisor delegates to specialist agents.

  supervisor      plans, delegates sub-questions, combines findings, proposes the action
  billing_agent   invoices, amounts owed, credits and collections policy
  support_agent   tickets, priorities, SLA and escalation policy
  account_agent   customer profile, tier, region, account manager

Each specialist is a full agent restricted to its own tools (least privilege). Specialists can run
in-process or as remote agents over the A2A protocol (agent/a2a.py); the supervisor sees both the
same way. Findings are shared through a blackboard: every delegation carries the earlier findings,
so a later specialist can build on what an earlier one found. Independent delegations run in
parallel. Every hand-off is recorded for tracing and evaluation.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .core import Agent, RunResult
from .tools import Tool, ToolError, ToolRegistry, build_tools

SPECIALISTS = {
    "billing_agent": ({"resolve_entity", "query_billing", "search_policies"},
                      "Invoices, amounts owed, overdue status, service credits and collections policy."),
    "support_agent": ({"resolve_entity", "get_support_tickets", "search_policies"},
                      "Support tickets, priorities, SLAs and escalation policy."),
    "account_agent": ({"resolve_entity", "query_crm"},
                      "Customer profile, tier, region and account manager."),
}

SPECIALIST_PROMPT = """You are the {name} in a team of agents. {description}
Answer only the question you are given, using your tools. Do not propose actions; report facts.
Finish by calling final_answer with the facts you used.

{{ontology}}"""

SUPERVISOR_PROMPT = """You are the supervisor of a team of specialist agents for a B2B software company.
Break the user's request into sub-questions and delegate each to the right specialist with the ask_* tools.
Ask independent questions in the same turn so they run in parallel. Combine their findings, check them
against each other, and use propose_action for any recommended action (a human approves it).
Finish by calling final_answer citing the specialists' facts.

{ontology}"""


class Specialist(Protocol):
    name: str
    description: str

    async def ask(self, question: str) -> dict: ...


@dataclass
class LocalSpecialist:
    name: str
    description: str
    agent: Agent

    async def ask(self, question: str) -> dict:
        run = await self.agent.arun(question)
        final = run.final or {}
        return {"agent": self.name, "status": run.status, "answer": run.answer,
                "facts": final.get("facts_used", []), "confidence": final.get("confidence"),
                "tokens": run.total_tokens}


class DelegateArgs(BaseModel):
    model_config = {"extra": "forbid"}
    question: str = Field(..., min_length=5, description="A self-contained sub-question for this specialist.")


@dataclass
class Handoff:
    to: str
    question: str
    status: str
    latency_ms: int
    answer: str | None


@dataclass
class MultiAgentResult:
    result: RunResult
    handoffs: list[Handoff] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.result.to_dict()
        d["handoffs"] = [vars(h) for h in self.handoffs]
        return d


def make_specialists(llm_for: dict, data_dir: Path, **agent_kwargs) -> list[LocalSpecialist]:
    out = []
    for name, (tools, description) in SPECIALISTS.items():
        agent = Agent(llm_for[name], data_dir, allowed_tools=tools,
                      system_prompt=SPECIALIST_PROMPT.format(name=name, description=description), **agent_kwargs)
        out.append(LocalSpecialist(name, description, agent))
    return out


class Supervisor:
    def __init__(self, llm, specialists: list[Specialist], data_dir: Path, delegation_timeout_s: float = 120.0,
                 **agent_kwargs):
        self.specialists = {s.name: s for s in specialists}
        self.data_dir, self.timeout = Path(data_dir), delegation_timeout_s
        self.llm, self.agent_kwargs = llm, agent_kwargs

    def _tools_factory(self, handoffs: list[Handoff], blackboard: list[dict]):
        lock = threading.Lock()

        def factory(data_dir):
            base, ontology, proposed = build_tools(data_dir)
            tools = [base.tools["propose_action"]]
            for name, spec in self.specialists.items():
                def delegate(question: str, _spec=spec) -> dict:
                    with lock:
                        context = list(blackboard)
                    q = question if not context else (
                        question + "\n\nFindings so far from other agents:\n" +
                        "\n".join(f"- {f['agent']}: {f['answer']}" for f in context))
                    t0 = time.monotonic()
                    try:
                        out = asyncio.run(_spec.ask(q))  # runs in the tool worker thread with its own loop
                    except Exception as err:
                        raise ToolError(f"{_spec.name} failed: {err}") from None
                    with lock:
                        handoffs.append(Handoff(_spec.name, question, out.get("status", "?"),
                                                int((time.monotonic() - t0) * 1000), out.get("answer")))
                        if out.get("status") == "completed":
                            blackboard.append(out)
                    return out
                tools.append(Tool(f"ask_{name}", f"Delegate a sub-question to the {name}: {spec.description}",
                                  DelegateArgs, delegate))
            return ToolRegistry(tools, timeout_s=self.timeout), ontology, proposed
        return factory

    async def arun(self, task: str) -> MultiAgentResult:
        handoffs: list[Handoff] = []
        blackboard: list[dict] = []
        agent = Agent(self.llm, self.data_dir, tools_factory=self._tools_factory(handoffs, blackboard),
                      system_prompt=SUPERVISOR_PROMPT, **self.agent_kwargs)
        result = await agent.arun(task)
        return MultiAgentResult(result, handoffs)

    def run(self, task: str) -> MultiAgentResult:
        return asyncio.run(self.arun(task))
