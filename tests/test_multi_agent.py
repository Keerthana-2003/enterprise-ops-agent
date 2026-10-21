"""Supervisor + specialist agents (in-process and over A2A)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent.llm import ScriptedLLM, final, tool_call, tool_calls
from agent.multi_agent import Supervisor, make_specialists
from agent.seed import seed


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return seed(Path(tempfile.mkdtemp()) / "data")


def scripted_team(data_dir):
    billing = ScriptedLLM([tool_call("resolve_entity", {"mention": "Fabrikam"}),
                           tool_call("query_billing", {"sql": "SELECT invoice_id, status FROM invoices WHERE billing_code='ACCT-5520'"}),
                           final("Fabrikam has overdue invoices INV-1004 and INV-1005.", facts_used=["INV-1004", "INV-1005"])])
    support = ScriptedLLM([tool_call("resolve_entity", {"mention": "Fabrikam"}),
                           tool_call("get_support_tickets", {"support_domain": "fabrikam.eu", "status": "open"}),
                           final("Fabrikam has an open P1 ticket T-504.", facts_used=["T-504"])])
    account = ScriptedLLM([tool_call("query_crm", {"sql": "SELECT account_manager FROM customers WHERE id=3"}),
                           final("Fabrikam is managed by Priya Shah.", facts_used=["crm customers.id=3"])])
    return billing, support, account, make_specialists(
        {"billing_agent": billing, "support_agent": support, "account_agent": account}, data_dir)


def test_supervisor_delegates_in_parallel_and_proposes_action(data_dir):
    billing, support, account, specialists = scripted_team(data_dir)
    sup_llm = ScriptedLLM([
        tool_calls([("ask_billing_agent", {"question": "What does Fabrikam owe and is anything overdue?"}),
                    ("ask_support_agent", {"question": "Does Fabrikam have open high-priority tickets?"})]),
        tool_call("ask_account_agent", {"question": "Who is Fabrikam's account manager?"}),
        tool_call("propose_action", {"action_type": "escalate_to_account_manager", "customer_id": "customer:3",
                                     "reason": "Open P1 T-504 with overdue invoices"}),
        final("Escalate Fabrikam to Priya Shah before any collections notice.",
              recommended_action="escalate_to_account_manager", customer_id="customer:3"),
    ])
    res = Supervisor(sup_llm, specialists, data_dir).run("Fabrikam has tickets and unpaid invoices. What should we do?")
    assert res.result.status == "completed" and res.result.max_parallel_tool_calls == 2
    assert {h.to for h in res.handoffs} == {"billing_agent", "support_agent", "account_agent"}
    assert all(h.status == "completed" for h in res.handoffs)
    assert [a["action_type"] for a in res.result.proposed_actions] == ["escalate_to_account_manager"]
    # blackboard: the account agent (asked last) received the earlier findings as context
    assert "Findings so far" in account.calls[0][1]["content"] and "INV-1004" in account.calls[0][1]["content"]
    # least privilege: specialists cannot see tools outside their role
    tool_names = {t for msgs in billing.calls for m in msgs if m["role"] == "assistant"
                  for t in [c["function"]["name"] for c in m.get("tool_calls") or []]}
    assert "propose_action" not in tool_names


def test_specialists_are_restricted_to_their_tools(data_dir):
    *_, specialists = scripted_team(data_dir)
    billing = {s.name: s for s in specialists}["billing_agent"]
    bad = ScriptedLLM([tool_call("get_support_tickets", {"support_domain": "fabrikam.eu"}), final("No access to tickets.")])
    billing.agent.llm = bad
    run = billing.agent.run("Show Fabrikam tickets")
    assert run.tool_errors == 1 and "Unknown tool" in run.steps[0].error


def test_specialist_over_a2a(data_dir):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    import asyncio

    from agent.a2a import RemoteSpecialist, agent_card, create_a2a_app

    *_, specialists = scripted_team(data_dir)
    billing = {s.name: s for s in specialists}["billing_agent"]
    app = create_a2a_app(billing, agent_card("billing_agent", billing.description, "http://billing"))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://billing") as client:
            remote = RemoteSpecialist("http://billing", client=client)
            card = await remote.load_card()
            out = await remote.ask("What does Fabrikam owe?")
            bad = await client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "tasks/cancel", "params": {}})
            return card, out, bad.json()

    card, out, bad = asyncio.run(scenario())
    assert card["name"] == "billing_agent" and card["skills"][0]["id"] == "billing_agent"
    assert out["status"] == "completed" and "INV-1004" in out["answer"] and out["agent"] == "billing_agent"
    assert bad["error"]["code"] == -32601
