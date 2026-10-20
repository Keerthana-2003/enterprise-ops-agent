"""OpenTelemetry spans for agent runs, LLM calls and tool calls (GenAI semantic conventions)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from agent.core import Agent  # noqa: E402
from agent.llm import ScriptedLLM, final, tool_call  # noqa: E402
from agent.seed import seed  # noqa: E402


def test_agent_emits_otel_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    data_dir = seed(Path(tempfile.mkdtemp()) / "data")
    llm = ScriptedLLM([tool_call("resolve_entity", {"mention": "Fabrikam"}), final("Fabrikam is customer:3.")])
    Agent(llm, data_dir).run("Who is Fabrikam?")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    run = spans["invoke_agent enterprise-ops-agent"]
    tool = spans["execute_tool resolve_entity"]
    chat = spans["chat scripted"]
    assert run.attributes["agent.status"] == "completed"
    assert tool.attributes["gen_ai.tool.name"] == "resolve_entity"
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert tool.parent.span_id == run.context.span_id  # tool spans nest under the run
