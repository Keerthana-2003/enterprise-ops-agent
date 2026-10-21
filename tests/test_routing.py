from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from agent.llm import LLMResponse, ScriptedLLM, final, tool_call
from agent.routing import RoutedAgent, TierPrice, needs_escalation, route
from agent.seed import seed


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return seed(Path(tempfile.mkdtemp()) / "data")


@pytest.mark.parametrize("task,tier", [
    ("Who manages Tailspin Airlines?", "small"),
    ("What is the SLA for P2 tickets?", "small"),
    ("Fabrikam has open tickets and overdue invoices. What should we do?", "large"),
    ("Can we give Northwind a service credit?", "large"),
    ("Should we place Fabrikam on credit hold?", "large"),
])
def test_route_by_complexity_and_risk(task, tier):
    d = route(task)
    assert d.tier == tier and d.reasons


def _cheap_answer(tokens=200, **fields):
    r = final("Tailspin Airlines is managed by Marcus Lee.", customer_id="customer:5", confidence="high", **fields)
    r.prompt_tokens, r.completion_tokens = tokens, 20
    return r


def test_simple_question_stays_on_small_tier(data_dir):
    small = ScriptedLLM([tool_call("resolve_entity", {"mention": "Tailspin"}), _cheap_answer()])
    large = ScriptedLLM([])
    agent = RoutedAgent(small, large, data_dir, prices={"small": TierPrice(0.1, 0.1), "large": TierPrice(3.0, 15.0)})
    res = asyncio.run(agent.arun("Who manages Tailspin Airlines?"))
    assert res.answered_by == "small" and not res.escalated and large.calls == []
    assert res.to_dict()["routing"]["cost_usd"] == pytest.approx((200 * 0.1 + 20 * 0.1) / 1e6)


def test_low_confidence_small_answer_escalates(data_dir):
    small = ScriptedLLM([final("Not sure who manages it.", confidence="low")])
    large = ScriptedLLM([tool_call("resolve_entity", {"mention": "Tailspin"}), _cheap_answer()])
    res = asyncio.run(RoutedAgent(small, large, data_dir).arun("Who manages Tailspin Airlines?"))
    assert res.escalated and res.answered_by == "large" and "low confidence" in res.escalation_reasons
    assert "Marcus Lee" in res.result.answer


def test_small_tier_may_not_take_actions(data_dir):
    small = ScriptedLLM([tool_call("propose_action", {"action_type": "no_action", "customer_id": "customer:5",
                                                      "reason": "nothing needed"}),
                         _cheap_answer()])
    large = ScriptedLLM([_cheap_answer()])
    res = asyncio.run(RoutedAgent(small, large, data_dir).arun("Who manages Tailspin Airlines?"))
    assert res.escalated and any("action" in r for r in res.escalation_reasons)


def test_high_risk_goes_straight_to_large(data_dir):
    small, large = ScriptedLLM([]), ScriptedLLM([_cheap_answer()])
    res = asyncio.run(RoutedAgent(small, large, data_dir).arun("Can we give Northwind a service credit?"))
    assert res.answered_by == "large" and small.calls == [] and not res.escalated


def test_needs_escalation_reasons(data_dir):
    from agent.core import Agent

    run = Agent(ScriptedLLM([LLMResponse(content="plain text"), LLMResponse(content="plain text")]), data_dir).run("x")
    assert "no valid structured answer" in needs_escalation(run)
