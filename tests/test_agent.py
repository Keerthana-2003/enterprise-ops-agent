"""Offline tests: tools, ontology, structured output, parallel calls, budgets, streaming and
every failure-handling path in the agent loop. They use ScriptedLLM, so no API key is needed.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from agent.core import Agent
from agent.llm import LLMError, LLMResponse, ScriptedLLM, final, tool_call, tool_calls
from agent.seed import seed
from agent.tools import ToolError, build_tools
from evals.judge import JudgeScore, build_judge_prompt, parse_judge
from evals.run_evals import grade


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return seed(Path(tempfile.mkdtemp()) / "data")


# ------------------------------------------------------------------ tools & ontology

def test_resolve_entity_joins_identifiers_across_systems(data_dir):
    registry, _, _ = build_tools(data_dir)
    out = registry.call("resolve_entity", json.dumps({"mention": "Fabrikam"}))
    ids = out["best_match"]["identifiers"]
    assert ids == {"crm": 3, "billing": "ACCT-5520", "support_domain": "fabrikam.eu"}
    assert {"rel": "MANAGED_BY", "entity": "Priya Shah"} in out["relations"]


def test_resolve_entity_by_email_domain(data_dir):
    registry, _, _ = build_tools(data_dir)
    out = registry.call("resolve_entity", json.dumps({"mention": "tailspin.aero"}))
    assert out["best_match"]["name"] == "Tailspin Airlines"


def test_unknown_entity_is_a_recoverable_error(data_dir):
    registry, _, _ = build_tools(data_dir)
    with pytest.raises(ToolError, match="No entity matches"):
        registry.call("resolve_entity", json.dumps({"mention": "Globex Corporation"}))


@pytest.mark.parametrize("sql", [
    "DROP TABLE invoices",
    "SELECT * FROM invoices; DELETE FROM invoices",
    "UPDATE invoices SET status='paid'",
])
def test_sql_tools_are_read_only(data_dir, sql):
    registry, _, _ = build_tools(data_dir)
    with pytest.raises(ToolError, match="read-only"):
        registry.call("query_billing", json.dumps({"sql": sql}))


def test_billing_query_and_sql_error_message(data_dir):
    registry, _, _ = build_tools(data_dir)
    rows = registry.call("query_billing", json.dumps(
        {"sql": "SELECT invoice_id FROM invoices WHERE billing_code='ACCT-5520' AND status='overdue'"}))
    assert {r["invoice_id"] for r in rows} == {"INV-1004", "INV-1005"}
    with pytest.raises(ToolError, match="SQL error"):
        registry.call("query_billing", json.dumps({"sql": "SELECT nope FROM invoices"}))


def test_pydantic_argument_validation(data_dir):
    registry, _, _ = build_tools(data_dir)
    with pytest.raises(ToolError, match="valid JSON"):
        registry.call("query_crm", "{not json")
    with pytest.raises(ToolError, match="sql: Field required"):
        registry.call("query_crm", "{}")
    with pytest.raises(ToolError, match="status"):
        registry.call("get_support_tickets", json.dumps({"support_domain": "fabrikam.eu", "status": "pending"}))
    with pytest.raises(ToolError, match="Extra inputs are not permitted"):
        registry.call("search_policies", json.dumps({"query": "sla", "limit": 5}))
    with pytest.raises(ToolError, match="Unknown tool"):
        registry.call("delete_customer", "{}")


def test_tool_json_schemas_come_from_pydantic(data_dir):
    registry, _, _ = build_tools(data_dir)
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in registry.schemas()}
    assert schemas["propose_action"]["required"] == ["action_type", "customer_id", "reason"]
    assert "escalate_to_account_manager" in json.dumps(schemas["propose_action"])
    assert schemas["search_policies"]["additionalProperties"] is False


def test_policy_search_finds_escalation_rule(data_dir):
    registry, _, _ = build_tools(data_dir)
    hits = registry.call("search_policies", json.dumps({"query": "open P1 ticket and overdue invoice collections"}))
    assert hits[0]["source"] == "escalation_policy.md"


def test_propose_action_requires_canonical_id(data_dir):
    registry, _, actions = build_tools(data_dir)
    with pytest.raises(ToolError, match="customer_id"):
        registry.call("propose_action", json.dumps(
            {"action_type": "no_action", "customer_id": "Fabrikam", "reason": "testing"}))
    out = registry.call("propose_action", json.dumps(
        {"action_type": "escalate_to_account_manager", "customer_id": "customer:3", "reason": "P1 + overdue"}))
    assert out["status"] == "pending_human_approval" and len(actions) == 1


# ------------------------------------------------------------------ agent loop

def happy_script() -> list[LLMResponse]:
    return [
        tool_call("resolve_entity", {"mention": "Fabrikam"}),
        tool_calls([  # independent lookups requested in one turn -> run in parallel
            ("get_support_tickets", {"support_domain": "fabrikam.eu", "status": "open"}),
            ("query_billing", {"sql": "SELECT * FROM invoices WHERE billing_code='ACCT-5520'"}),
            ("search_policies", {"query": "P1 ticket overdue invoice"}),
        ]),
        tool_call("propose_action", {"action_type": "escalate_to_account_manager",
                                     "customer_id": "customer:3", "reason": "Open P1 T-504 with overdue invoices"}),
        final("Escalate Fabrikam to Priya Shah before any collections notice.",
              facts_used=["T-504 open P1", "INV-1004 overdue", "INV-1005 overdue", "escalation_policy.md"],
              recommended_action="escalate_to_account_manager", customer_id="customer:3", confidence="high"),
    ]


def test_happy_path_structured_output_and_parallel_calls(data_dir):
    run = Agent(ScriptedLLM(happy_script()), data_dir).run("Fabrikam has open tickets and unpaid invoices.")
    assert run.status == "completed" and run.structured
    assert run.tool_errors == 0
    assert run.max_parallel_tool_calls == 3
    assert run.final["recommended_action"] == "escalate_to_account_manager"
    assert [a["action_type"] for a in run.proposed_actions] == ["escalate_to_account_manager"]
    assert "Priya Shah" in run.answer


def test_parallel_tool_calls_run_concurrently(data_dir, monkeypatch):
    import agent.core as core

    real_build = core.build_tools

    def slow_build(d):
        registry, onto, actions = real_build(d)
        fn = registry.tools["search_policies"].fn
        registry.tools["search_policies"].fn = lambda query: (time.sleep(0.3), fn(query))[1]
        return registry, onto, actions

    Agent(ScriptedLLM([final("warm-up run")]), data_dir).run("warm up")  # exclude one-time import cost
    monkeypatch.setattr(core, "build_tools", slow_build)
    llm = ScriptedLLM([
        tool_calls([("search_policies", {"query": "sla"}), ("search_policies", {"query": "credit"}),
                    ("search_policies", {"query": "escalation"})]),
        final("Done reading policies."),
    ])
    t0 = time.monotonic()
    run = Agent(llm, data_dir).run("read policies")
    assert run.status == "completed"
    assert time.monotonic() - t0 < 0.75  # three 0.3s calls in parallel, not 0.9s in sequence


def test_invalid_final_answer_is_fed_back(data_dir):
    llm = ScriptedLLM([
        tool_call("final_answer", {"summary": "ok", "recommended_action": "delete_everything"}),
        final("Fabrikam is customer:3.", customer_id="customer:3"),
    ])
    run = Agent(llm, data_dir).run("Who is Fabrikam?")
    assert run.status == "completed" and run.structured and run.tool_errors == 1
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "Invalid final_answer" in tool_msgs[-1]["content"]


def test_plain_text_answer_is_nudged_then_accepted(data_dir):
    llm = ScriptedLLM([LLMResponse(content="Fabrikam is customer 3."),
                       LLMResponse(content="Fabrikam is customer 3.")])
    run = Agent(llm, data_dir).run("Who is Fabrikam?")
    assert "final_answer" in llm.calls[1][-1]["content"]
    assert run.status == "completed" and not run.structured and run.final["confidence"] == "low"


def test_agent_recovers_from_malformed_arguments(data_dir):
    llm = ScriptedLLM([
        tool_call("resolve_entity", "{mention: Fabrikam"),   # broken JSON
        tool_call("resolve_entity", {"mention": "Fabrikam"}),  # model corrects itself
        final("Fabrikam is customer:3."),
    ])
    run = Agent(llm, data_dir).run("Who is Fabrikam?")
    assert run.status == "completed" and run.tool_errors == 1
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "valid JSON" in tool_msgs[-1]["content"]


def test_llm_outage_ends_run_cleanly(data_dir):
    run = Agent(ScriptedLLM([LLMError("rate limited")]), data_dir).run("anything")
    assert run.status == "llm_error" and run.answer is None


def test_token_budget(data_dir):
    big = LLMResponse(content=None, prompt_tokens=900, completion_tokens=200,
                      tool_calls=tool_call("query_crm", {"sql": "SELECT id FROM customers"}).tool_calls)
    run = Agent(ScriptedLLM([big, big]), data_dir, max_total_tokens=2000).run("x")
    assert run.status == "token_budget_exceeded" and run.total_tokens == 2200


def test_cached_tokens_are_tracked(data_dir):
    script = happy_script()
    for r in script:
        r.prompt_tokens, r.cached_tokens = 1000, 768
    run = Agent(ScriptedLLM(script), data_dir).run("Fabrikam?")
    d = run.to_dict()
    assert d["cached_tokens"] == 4 * 768 and d["cache_hit_rate"] == 0.768


def test_loop_detection(data_dir):
    same = {"sql": "SELECT name FROM customers"}
    llm = ScriptedLLM([tool_call("query_crm", same) for _ in range(3)] + [final("Listed the customers.")])
    run = Agent(llm, data_dir).run("list customers")
    assert any(s.kind == "guard" for s in run.steps)
    assert run.status == "completed"


def test_tool_error_budget(data_dir):
    llm = ScriptedLLM([tool_call("query_crm", {"sql": f"DROP TABLE t{i}"}) for i in range(5)])
    run = Agent(llm, data_dir, max_tool_errors=3).run("break things")
    assert run.status == "tool_error_budget_exceeded" and run.tool_errors == 3


def test_max_steps_returns_best_effort_answer(data_dir):
    llm = ScriptedLLM([tool_call("query_crm", {"sql": f"SELECT id FROM customers WHERE id={i}"}) for i in range(3)]
                      + [LLMResponse(content="Partial answer; billing not verified.")])
    run = Agent(llm, data_dir, max_steps=3).run("long task")
    assert run.status == "max_steps"
    assert "Partial" in run.answer


def test_unexpected_tool_exception_does_not_crash(data_dir, monkeypatch):
    import agent.core as core

    real_build = core.build_tools

    def broken_build(d):
        registry, onto, actions = real_build(d)
        registry.tools["search_policies"].fn = lambda query: 1 / 0
        return registry, onto, actions

    monkeypatch.setattr(core, "build_tools", broken_build)
    llm = ScriptedLLM([tool_call("search_policies", {"query": "sla"}), final("No policy found.")])
    run = Agent(llm, data_dir).run("sla?")
    assert run.status == "completed" and "Internal tool failure" in run.steps[0].error


def test_stream_emits_events_in_order(data_dir):
    async def collect():
        return [ev async for ev in Agent(ScriptedLLM(happy_script()), data_dir).stream("Fabrikam?")]

    events = asyncio.run(collect())
    types = [e.type for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert types.count("tool_finished") == 5 and "final" in types
    assert types.index("final") < types.index("run_finished")
    assert events[-1].data["status"] == "completed"


# ------------------------------------------------------------------ evals

def test_grade_uses_structured_facts():
    task = {"expect_all": [["T-504"], ["Priya Shah"]], "expect_action": "escalate_to_account_manager"}
    run = {"status": "completed", "answer": "Escalate to Priya Shah.", "final": {"facts_used": ["T-504 open P1"]},
           "proposed_actions": [{"action_type": "escalate_to_account_manager"}]}
    assert grade(task, run) == (True, None)
    run["proposed_actions"] = []
    assert grade(task, run) == (False, "missing_expected_action")
    no_action_task = {"expect_all": [["x"]], "expect_action": None}
    assert grade(no_action_task, {"status": "completed", "answer": "x", "final": None,
                                  "proposed_actions": [{"action_type": "issue_service_credit"}]}) \
        == (False, "unwarranted_action")


def test_judge_parsing_and_prompt(data_dir):
    good = parse_judge('{"correctness": 5, "grounding": 4, "policy_compliance": 5, "safety": 5,'
                       ' "actionability": 4, "rationale": "Grounded."}')
    assert isinstance(good, JudgeScore) and good.mean == 4.6
    assert parse_judge('{"correctness": 9}') is None and parse_judge("not json") is None
    run = Agent(ScriptedLLM(happy_script()), data_dir).run("Fabrikam?").to_dict()
    payload = json.loads(build_judge_prompt("Fabrikam?", run)[1]["content"])
    assert "T-504" in payload["EVIDENCE"] and payload["PROPOSED_ACTIONS"][0]["action"] == "escalate_to_account_manager"


def test_retry_policy_fails_fast_on_auth_and_quota_errors():
    from agent.llm import _retryable

    class E(Exception):
        def __init__(self, status=None, code=None, type=None):
            self.status_code, self.code, self.type = status, code, type

    assert not _retryable(E(401)) and not _retryable(E(400)) and not _retryable(E(429, "insufficient_quota"))
    assert not _retryable(E(429, "credit_balance_exhausted", "insufficient_quota"))
    assert _retryable(E(429, "rate_limit_exceeded")) and _retryable(E(503)) and _retryable(E())
