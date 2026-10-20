"""Bedrock Converse provider: format conversion and an agent run against a fake Bedrock client."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent.bedrock import BedrockConverseClient, from_converse, to_converse
from agent.core import Agent
from agent.seed import seed


def test_to_converse_maps_roles_tools_and_results():
    tools = [{"type": "function", "function": {"name": "query_crm", "description": "crm",
                                               "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}}}]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "query_crm", "arguments": '{"sql": "SELECT 1"}'}},
            {"id": "b", "type": "function", "function": {"name": "query_crm", "arguments": "{bad"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "[1]"},
        {"role": "tool", "tool_call_id": "b", "content": "err"},
        {"role": "user", "content": "now answer"},
    ]
    req = to_converse(messages, tools, prompt_cache=True)
    assert req["system"] == [{"text": "sys"}, {"cachePoint": {"type": "default"}}]
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]  # merged, alternating
    uses = req["messages"][1]["content"]
    assert uses[0]["toolUse"]["input"] == {"sql": "SELECT 1"}
    assert uses[1]["toolUse"]["input"] == {"_unparsed_arguments": "{bad"}
    last = req["messages"][2]["content"]
    assert [list(b)[0] for b in last] == ["toolResult", "toolResult", "text"]
    assert req["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]["sql"]


def test_from_converse_reads_text_tools_and_cache_usage():
    resp = {"output": {"message": {"role": "assistant", "content": [
        {"text": "Looking up."}, {"toolUse": {"toolUseId": "t1", "name": "resolve_entity", "input": {"mention": "x"}}}]}},
        "usage": {"inputTokens": 200, "outputTokens": 30, "cacheReadInputTokens": 800}}
    out = from_converse(resp)
    assert out.content == "Looking up." and out.tool_calls[0].name == "resolve_entity"
    assert json.loads(out.tool_calls[0].arguments) == {"mention": "x"}
    assert out.prompt_tokens == 1000 and out.cached_tokens == 800


class FakeBedrock:
    """Returns scripted Converse responses and records requests."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _tool_use(name, args, uid):
    return {"output": {"message": {"content": [{"toolUse": {"toolUseId": uid, "name": name, "input": args}}]}},
            "usage": {"inputTokens": 100, "outputTokens": 10}}


def test_agent_runs_on_bedrock_provider():
    data_dir = seed(Path(tempfile.mkdtemp()) / "data")
    fake = FakeBedrock([
        _tool_use("resolve_entity", {"mention": "Tailspin"}, "u1"),
        _tool_use("final_answer", {"summary": "Tailspin Airlines is managed by Marcus Lee.",
                                   "customer_id": "customer:5"}, "u2"),
    ])
    llm = BedrockConverseClient(model="test-model", client=fake, max_retries=0)
    run = Agent(llm, data_dir).run("Who manages Tailspin?")
    assert run.status == "completed" and run.structured and "Marcus Lee" in run.answer
    assert fake.requests[0]["modelId"] == "test-model" and "toolConfig" in fake.requests[1]
