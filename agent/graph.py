"""The same agent built as a LangGraph state machine.

Graph:  START -> agent -> (tools -> agent)* -> END

  agent  node: calls the LLM with the conversation and tool schemas
  tools  node: runs the requested tools in parallel, validates final_answer (structured output)
  routing    : stop on a valid final answer, an LLM error, the token budget or the tool-error budget;
               the step budget is LangGraph's recursion limit

It reuses the tools, schemas, prompts and RunResult of the hand-written loop (agent/core.py), so
both runtimes can be compared on the same eval harness:  python -m evals.run_evals --runtime langgraph
"""
from __future__ import annotations

import asyncio
import json
import operator
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from .core import FINAL_TOOL, SYSTEM_PROMPT, Agent, RunResult, Step, _final_answer_schema
from .llm import LLMClient, LLMError
from .schemas import FinalAnswer
from .tools import ToolError, build_tools


class GraphState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    steps: Annotated[list[dict], operator.add]
    pending: list[dict]
    final: dict | None
    structured: bool
    status: str
    tool_errors: int
    llm_calls: int
    max_parallel: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


class LangGraphAgent:
    def __init__(self, llm: LLMClient, data_dir: Path, max_steps: int = 10, max_tool_errors: int = 4,
                 max_total_tokens: int = 60_000):
        self.llm = llm
        self.data_dir = Path(data_dir)
        self.max_steps = max_steps
        self.max_tool_errors = max_tool_errors
        self.max_total_tokens = max_total_tokens

    def _build(self, registry, tools: list[dict]):
        llm, max_tokens, max_errors = self.llm, self.max_total_tokens, self.max_tool_errors

        async def agent_node(state: GraphState) -> dict[str, Any]:
            try:
                resp = await llm.chat(state["messages"], tools)
            except LLMError as err:
                return {"status": "llm_error", "steps": [{"kind": "error", "error": str(err)}]}
            update: dict[str, Any] = {
                "llm_calls": state["llm_calls"] + 1,
                "prompt_tokens": state["prompt_tokens"] + resp.prompt_tokens,
                "completion_tokens": state["completion_tokens"] + resp.completion_tokens,
                "cached_tokens": state["cached_tokens"] + resp.cached_tokens,
            }
            if update["prompt_tokens"] + update["completion_tokens"] > max_tokens:
                return {**update, "status": "token_budget_exceeded", "pending": []}
            if not resp.tool_calls:  # unstructured answer: accept but flag it
                final = FinalAnswer.model_construct(
                    summary=(resp.content or "").strip(), facts_used=[], recommended_action="no_action",
                    customer_id=None, confidence="low", unverified=["answer was not structured"])
                return {**update, "final": final.model_dump(), "structured": False, "status": "completed",
                        "pending": [], "steps": [{"kind": "final", "name": FINAL_TOOL, "result": final.model_dump()}]}
            calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls]
            assistant = {"role": "assistant", "content": resp.content, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]}
            return {**update, "messages": [assistant], "pending": calls}

        async def run_one(call: dict) -> tuple[dict, dict, bool]:
            t0 = time.monotonic()
            try:
                out = await asyncio.to_thread(registry.call, call["name"], call["arguments"])
                content, step, failed = json.dumps(out, default=str), {"result": out}, False
            except ToolError as err:
                content, step, failed = json.dumps({"error": str(err)}), {"error": str(err)}, True
            except Exception as err:
                msg = f"Internal tool failure ({type(err).__name__}). Try another approach."
                content, step, failed = json.dumps({"error": msg}), {"error": msg}, True
            step.update(kind="tool", name=call["name"], arguments=call["arguments"],
                        latency_ms=int((time.monotonic() - t0) * 1000))
            return {"role": "tool", "tool_call_id": call["id"], "content": content}, step, failed

        async def tools_node(state: GraphState) -> dict[str, Any]:
            work = [c for c in state["pending"] if c["name"] != FINAL_TOOL]
            finals = [c for c in state["pending"] if c["name"] == FINAL_TOOL]
            results = await asyncio.gather(*(run_one(c) for c in work))
            messages = [r[0] for r in results]
            steps = [r[1] for r in results]
            errors = state["tool_errors"] + sum(r[2] for r in results)
            update: dict[str, Any] = {"pending": [], "max_parallel": max(state["max_parallel"], len(work))}
            if finals:
                final, error = Agent._parse_final(_as_tool_call(finals[0]))
                if final is not None:
                    steps.append({"kind": "final", "name": FINAL_TOOL, "result": final.model_dump()})
                    return {**update, "messages": messages, "steps": steps, "tool_errors": errors,
                            "final": final.model_dump(), "structured": True, "status": "completed"}
                errors += 1
                steps.append({"kind": "tool", "name": FINAL_TOOL, "arguments": finals[0]["arguments"], "error": error})
                messages += [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": error})}
                             for c in finals]
            status = "tool_error_budget_exceeded" if errors >= max_errors else state["status"]
            return {**update, "messages": messages, "steps": steps, "tool_errors": errors, "status": status}

        def after_agent(state: GraphState) -> str:
            return "tools" if state["status"] == "running" and state["pending"] else END

        def after_tools(state: GraphState) -> str:
            return "agent" if state["status"] == "running" else END

        graph = StateGraph(GraphState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
        return graph.compile()

    async def arun(self, task: str) -> RunResult:
        registry, ontology, proposed = build_tools(self.data_dir)
        app = self._build(registry, registry.schemas() + [_final_answer_schema()])
        state: GraphState = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT.format(ontology=ontology.schema_summary())},
                         {"role": "user", "content": task}],
            "steps": [], "pending": [], "final": None, "structured": False, "status": "running",
            "tool_errors": 0, "llm_calls": 0, "max_parallel": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
        }
        start = time.monotonic()
        try:
            # each step is an agent node plus a tools node
            out = await app.ainvoke(state, config={"recursion_limit": 2 * self.max_steps + 1})
        except GraphRecursionError:
            out = {**state, "status": "max_steps"}
        result = RunResult(run_id=uuid.uuid4().hex[:12], task=task, status=out["status"],
                           answer=(out["final"] or {}).get("summary"), final=out["final"],
                           structured=out["structured"], tool_errors=out["tool_errors"],
                           llm_calls=out["llm_calls"], max_parallel_tool_calls=out["max_parallel"],
                           prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                           cached_tokens=out["cached_tokens"])
        if result.status == "running":  # graph ended without a terminal status
            result.status = "max_steps"
        if result.status == "max_steps" and not result.answer:
            result.answer = "Step budget reached before a final answer. See the trace."
        result.steps = [Step(index=i, kind=s.get("kind", "tool"), name=s.get("name"), arguments=s.get("arguments"),
                             result=s.get("result"), error=s.get("error"), latency_ms=s.get("latency_ms", 0))
                        for i, s in enumerate(out["steps"])]
        result.proposed_actions = list(proposed)
        result.latency_ms = int((time.monotonic() - start) * 1000)
        return result

    def run(self, task: str) -> RunResult:
        return asyncio.run(self.arun(task))


def _as_tool_call(call: dict):
    from .llm import ToolCall

    return ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
