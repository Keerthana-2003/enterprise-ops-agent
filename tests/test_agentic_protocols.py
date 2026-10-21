"""MCP server/client, Strands Agents tools and the Bedrock AgentCore entrypoint.
Run in the `agentic` CI job (requirements-agentic.txt); skipped where those SDKs are not installed."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from agent.core import Agent
from agent.llm import ScriptedLLM, final, tool_call
from agent.seed import seed


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return seed(Path(tempfile.mkdtemp()) / "data")


def test_mcp_server_and_agent_using_mcp_tools(data_dir):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from agent.mcp_tools import MCPToolset, create_mcp_server

    server = create_mcp_server(data_dir)

    async def scenario():
        async with create_connected_server_and_client_session(server) as session:
            names = sorted(t.name for t in (await session.list_tools()).tools)
            schema = (await session.read_resource("ontology://schema")).contents[0].text
            ok = await session.call_tool("query_billing", {"sql": "SELECT invoice_id FROM invoices LIMIT 2"})
            blocked = await session.call_tool("query_billing", {"sql": "DELETE FROM invoices"})

            toolset = MCPToolset(session, asyncio.get_running_loop())
            await toolset.load()
            llm = ScriptedLLM([tool_call("query_billing", {"sql": "SELECT invoice_id FROM invoices WHERE status='overdue'"}),
                               final("Overdue invoices found.", facts_used=["invoices"])])
            run = await Agent(llm, data_dir, tools_factory=toolset.registry_factory()).arun("Which invoices are overdue?")
            return names, schema, ok, blocked, run

    names, schema, ok, blocked, run = asyncio.run(scenario())
    assert {"resolve_entity", "query_crm", "query_billing", "get_support_tickets", "search_policies"} <= set(names)
    assert schema
    assert not ok.isError and len(json.loads(ok.content[0].text)) == 2
    assert blocked.isError and "read-only" in blocked.content[0].text
    assert run.status == "completed" and run.tool_errors == 0 and run.steps[0].name == "query_billing"


def test_strands_tools_wrap_the_registry(data_dir):
    pytest.importorskip("strands")
    from agent.strands_agent import invoke_registry_tool, strands_tools
    from agent.tools import build_tools

    registry, _, _ = build_tools(data_dir)
    tools = strands_tools(registry, {"query_billing", "propose_action"})
    assert sorted(t.tool_name for t in tools) == ["propose_action", "query_billing"]
    spec = next(t for t in tools if t.tool_name == "query_billing").tool_spec
    assert "sql" in spec["inputSchema"]["json"]["properties"]
    ok = invoke_registry_tool(registry, "query_billing", {"toolUseId": "t1", "input": {"sql": "SELECT 1 AS x"}})
    bad = invoke_registry_tool(registry, "query_billing", {"toolUseId": "t2", "input": {"sql": "DROP TABLE invoices"}})
    assert ok["status"] == "success" and json.loads(ok["content"][0]["text"]) == [{"x": 1}]
    assert bad["status"] == "error" and bad["toolUseId"] == "t2"


def test_strands_team_builds(data_dir, monkeypatch):
    pytest.importorskip("strands")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    from agent.strands_agent import build_team

    sup, specialists, _ = build_team(data_dir)
    assert set(specialists) == {"billing_agent", "support_agent", "account_agent"}
    assert {"ask_billing_agent", "ask_support_agent", "ask_account_agent", "propose_action"} <= set(sup.tool_names)
    assert "query_crm" not in specialists["billing_agent"].tool_names


def test_agentcore_entrypoint(data_dir, monkeypatch):
    pytest.importorskip("bedrock_agentcore")
    from starlette.testclient import TestClient

    import agent.agentcore_app as ac

    monkeypatch.setattr(ac, "DATA", data_dir)
    monkeypatch.setattr(ac, "_llm_factory", lambda role: ScriptedLLM([final(f"answer from {role}")]))
    client = TestClient(ac.app)
    assert client.get("/ping").status_code == 200
    r = client.post("/invocations", json={"prompt": "Is Fabrikam overdue?"})
    assert r.status_code == 200 and r.json()["status"] == "completed" and r.json()["answer"] == "answer from agent"
    r = client.post("/invocations", json={"prompt": "Plan for Fabrikam", "mode": "multi"})
    assert r.json()["answer"] == "answer from supervisor" and r.json()["handoffs"] == []
    assert "error" in client.post("/invocations", json={}).json()
