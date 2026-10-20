"""LangGraph runtime tests: same scripted scenarios as the hand-written loop."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
from agent.graph import LangGraphAgent  # noqa: E402
from agent.llm import LLMError, LLMResponse, ScriptedLLM, final, tool_call, tool_calls  # noqa: E402
from agent.seed import seed  # noqa: E402


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return seed(Path(tempfile.mkdtemp()) / "data")


def test_graph_happy_path_with_parallel_tools(data_dir):
    llm = ScriptedLLM([
        tool_call("resolve_entity", {"mention": "Fabrikam"}),
        tool_calls([("get_support_tickets", {"support_domain": "fabrikam.eu", "status": "open"}),
                    ("query_billing", {"sql": "SELECT * FROM invoices WHERE billing_code='ACCT-5520'"})]),
        tool_call("propose_action", {"action_type": "escalate_to_account_manager",
                                     "customer_id": "customer:3", "reason": "Open P1 with overdue invoices"}),
        final("Escalate Fabrikam to Priya Shah.", recommended_action="escalate_to_account_manager",
              customer_id="customer:3", facts_used=["T-504", "INV-1004"]),
    ])
    run = LangGraphAgent(llm, data_dir).run("Fabrikam?")
    assert run.status == "completed" and run.structured
    assert run.max_parallel_tool_calls == 2 and run.llm_calls == 4
    assert [a["action_type"] for a in run.proposed_actions] == ["escalate_to_account_manager"]


def test_graph_recovers_from_invalid_final_answer(data_dir):
    llm = ScriptedLLM([tool_call("final_answer", {"summary": "x"}), final("Fabrikam is customer:3.")])
    run = LangGraphAgent(llm, data_dir).run("Who is Fabrikam?")
    assert run.status == "completed" and run.tool_errors == 1


def test_graph_llm_error(data_dir):
    run = LangGraphAgent(ScriptedLLM([LLMError("down")]), data_dir).run("x")
    assert run.status == "llm_error"


def test_graph_budgets(data_dir):
    bad = [tool_call("query_crm", {"sql": f"DROP TABLE t{i}"}) for i in range(5)]
    assert LangGraphAgent(ScriptedLLM(bad), data_dir, max_tool_errors=2).run("x").status == "tool_error_budget_exceeded"
    loop = [tool_call("query_crm", {"sql": f"SELECT id FROM customers WHERE id={i}"}) for i in range(10)]
    assert LangGraphAgent(ScriptedLLM(loop), data_dir, max_steps=3).run("x").status == "max_steps"
    big = LLMResponse(content=None, prompt_tokens=5000, completion_tokens=10,
                      tool_calls=tool_call("query_crm", {"sql": "SELECT id FROM customers"}).tool_calls)
    assert LangGraphAgent(ScriptedLLM([big]), data_dir, max_total_tokens=1000).run("x").status == "token_budget_exceeded"
