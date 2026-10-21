"""The async agent loop.

Production features:
  * structured output    -> the run ends only through the `final_answer` tool, validated by Pydantic
  * parallel tool calls  -> independent tool calls from one model turn run concurrently (asyncio.gather)
  * streaming            -> `stream()` yields typed events as they happen (served as SSE by the API)
  * cost control         -> per-run token budget; cached prompt tokens tracked to measure prompt caching
  * observability        -> Langfuse trace per run, generation per LLM call, span per tool call

Failure modes handled (each ends in a typed status or is fed back so the model can recover):
  * LLM outage / rate limit  -> client retries with backoff; if still failing, run ends "llm_error"
  * malformed / invalid args -> Pydantic error returned to the model so it can correct itself
  * tool exceptions          -> recoverable ToolError text fed back; unexpected errors caught too
  * invalid final answer     -> validation error fed back; plain-text answers are nudged once
  * repeated identical calls -> loop detected, model told to change approach
  * too many tool errors     -> run stops "tool_error_budget_exceeded"
  * token budget exhausted   -> run stops "token_budget_exceeded"
  * step budget exhausted    -> model asked for a best-effort answer, run ends "max_steps"
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from .llm import LLMClient, LLMError, ToolCall
from .schemas import FinalAnswer, StreamEvent
from .tools import ToolError, ToolRegistry, build_tools
from .telemetry import tracer as otel_tracer
from .tracing import Tracer

# Kept identical across runs and placed first, with the tool definitions, so providers can
# serve it from their prompt cache. Run-specific content only appears in later messages.
SYSTEM_PROMPT = """You are an operations agent for a B2B software company.
You answer questions and recommend actions using ONLY data returned by your tools.

Enterprise data is fragmented across systems that use different customer keys.
Always call resolve_entity first to get a customer's canonical id, CRM id, billing code and support domain,
then query each system with the matching key. Independent lookups can be requested together in one turn.
Check company policy with search_policies before recommending an action.
Use propose_action for any action; it only proposes, a human approves.

If a tool returns an error, read it and fix your call. Do not repeat a failing call unchanged.
Finish by calling final_answer: cite the facts you used (ticket ids, invoice ids, policy file),
the recommended action, and anything you could not verify.

{ontology}"""

FINAL_TOOL = "final_answer"


def _final_answer_schema() -> dict:
    params = FinalAnswer.model_json_schema()
    params.pop("title", None)
    params.pop("description", None)
    return {"type": "function", "function": {
        "name": FINAL_TOOL, "description": "Return the final structured answer. Call exactly once, last.",
        "parameters": params}}


@dataclass
class Step:
    index: int
    kind: str  # "tool" | "final" | "error" | "guard"
    name: str | None = None
    arguments: str | None = None
    result: Any = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class RunResult:
    run_id: str
    task: str
    status: str  # completed | max_steps | llm_error | tool_error_budget_exceeded | token_budget_exceeded
    answer: str | None
    final: dict | None = None  # validated FinalAnswer
    structured: bool = False   # False if the model never produced a valid final_answer
    steps: list[Step] = field(default_factory=list)
    proposed_actions: list[dict] = field(default_factory=list)
    tool_errors: int = 0
    llm_calls: int = 0
    max_parallel_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["cache_hit_rate"] = round(self.cached_tokens / self.prompt_tokens, 3) if self.prompt_tokens else 0.0
        return d


Emit = Callable[[StreamEvent], Awaitable[None]]


class Agent:
    def __init__(self, llm: LLMClient, data_dir: Path, max_steps: int = 10, max_tool_errors: int = 4,
                 max_total_tokens: int = 60_000, tracer: Tracer | None = None,
                 allowed_tools: set[str] | None = None, tools_factory=None, system_prompt: str | None = None):
        self.llm = llm
        self.data_dir = Path(data_dir)
        self.max_steps = max_steps
        self.max_tool_errors = max_tool_errors
        self.max_total_tokens = max_total_tokens
        self.tracer = tracer or Tracer()
        self.allowed_tools = allowed_tools          # restrict a specialist agent to a subset of tools
        self.tools_factory = tools_factory          # e.g. MCP-backed tools or supervisor delegation tools
        self.system_prompt = system_prompt

    # ---------------------------------------------------------------- public API

    def run(self, task: str) -> RunResult:
        """Synchronous convenience wrapper."""
        return asyncio.run(self.arun(task))

    async def arun(self, task: str) -> RunResult:
        async def ignore(_: StreamEvent) -> None: ...
        return await self._run(task, ignore)

    async def stream(self, task: str) -> AsyncIterator[StreamEvent]:
        """Yield events while the run progresses; the last event is `run_finished` with the full result."""
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        async def emit(ev: StreamEvent) -> None:
            await queue.put(ev)

        async def runner() -> None:
            try:
                await self._run(task, emit)
            finally:
                await queue.put(None)

        worker = asyncio.create_task(runner())
        try:
            while (ev := await queue.get()) is not None:
                yield ev
            await worker  # surface unexpected exceptions
        finally:
            if not worker.done():  # client disconnected mid-stream
                worker.cancel()

    # ---------------------------------------------------------------- loop

    async def _run(self, task: str, emit: Emit) -> RunResult:
        registry, ontology, proposed = (self.tools_factory or build_tools)(self.data_dir)  # fresh state per run
        if self.allowed_tools is not None:
            registry.tools = {k: v for k, v in registry.tools.items() if k in self.allowed_tools}
        tools = registry.schemas() + [_final_answer_schema()]
        result = RunResult(run_id=uuid.uuid4().hex[:12], task=task, status="running", answer=None)
        messages: list[dict] = [
            {"role": "system", "content": (self.system_prompt or SYSTEM_PROMPT).format(ontology=ontology.schema_summary())},
            {"role": "user", "content": task},
        ]
        seen_calls: dict[str, int] = {}
        nudged = False
        start = time.monotonic()
        trace = self.tracer.start_run("agent-run", input=task, metadata={"run_id": result.run_id,
                                                                          "model": getattr(self.llm, "model", "?")})
        model = getattr(self.llm, "model", "?")
        run_span = otel_tracer.start_span("invoke_agent enterprise-ops-agent", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": model, "agent.run_id": result.run_id})
        otel_token = otel_context.attach(otel_trace.set_span_in_context(run_span))
        await emit(StreamEvent(type="run_started", run_id=result.run_id, data={"task": task}))

        for i in range(self.max_steps):
            resp = await self._llm_call(messages, tools, result, trace, i, emit)
            if resp is None:
                result.status = "llm_error"
                break

            if result.total_tokens > self.max_total_tokens:
                result.status = "token_budget_exceeded"
                result.answer = f"Stopped: token budget of {self.max_total_tokens} exceeded. See the trace."
                break

            if not resp.tool_calls:  # model answered in plain text instead of structured output
                text = (resp.content or "").strip()
                if not nudged:
                    nudged = True
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": "Return your answer by calling final_answer."})
                    continue
                self._finish(result, FinalAnswer.model_construct(
                    summary=text, facts_used=[], recommended_action="no_action", customer_id=None,
                    confidence="low", unverified=["answer was not structured"]), structured=False, step=i)
                break

            messages.append({
                "role": "assistant", "content": resp.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments}} for tc in resp.tool_calls],
            })
            work = [tc for tc in resp.tool_calls if tc.name != FINAL_TOOL]
            finals = [tc for tc in resp.tool_calls if tc.name == FINAL_TOOL]
            result.max_parallel_tool_calls = max(result.max_parallel_tool_calls, len(work))

            outputs = await asyncio.gather(*(self._execute(registry, tc, i, seen_calls, result, trace, emit)
                                             for tc in work))
            for tc, content in zip(work, outputs):
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

            if finals:
                final, error = self._parse_final(finals[0])
                if final is not None:
                    self._finish(result, final, structured=True, step=i)
                    await emit(StreamEvent(type="final", run_id=result.run_id, step=i, data=result.final))
                    break
                result.tool_errors += 1
                result.steps.append(Step(i, "tool", FINAL_TOOL, finals[0].arguments, error=error))
                await emit(StreamEvent(type="tool_failed", run_id=result.run_id, step=i,
                                       data={"tool": FINAL_TOOL, "error": error}))
                for tc in finals:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": error})})

            if result.tool_errors >= self.max_tool_errors:
                result.status = "tool_error_budget_exceeded"
                result.answer = "Stopped: too many failed tool calls. Partial findings are in the trace."
                break
        else:
            result.status = "max_steps"
            result.answer = await self._best_effort(messages, result, trace, emit)

        result.proposed_actions = list(proposed)
        result.latency_ms = int((time.monotonic() - start) * 1000)
        trace.update(output={"status": result.status, "answer": result.answer, "final": result.final},
                     metadata={"tokens": result.total_tokens, "cached_tokens": result.cached_tokens,
                               "tool_errors": result.tool_errors})
        trace.end()
        run_span.set_attributes({"agent.status": result.status, "agent.tool_errors": result.tool_errors,
                                 "gen_ai.usage.input_tokens": result.prompt_tokens,
                                 "gen_ai.usage.output_tokens": result.completion_tokens,
                                 "agent.cached_tokens": result.cached_tokens})
        if result.status != "completed":
            run_span.set_status(Status(StatusCode.ERROR, result.status))
        run_span.end()
        otel_context.detach(otel_token)
        await asyncio.to_thread(self.tracer.flush)
        await emit(StreamEvent(type="run_finished", run_id=result.run_id, data=result.to_dict()))
        return result

    # ---------------------------------------------------------------- helpers

    async def _llm_call(self, messages, tools, result: RunResult, trace, i: int, emit: Emit):
        gen = trace.start_generation(name="llm", model=getattr(self.llm, "model", "?"), input=messages[-2:])
        t0 = time.monotonic()
        model = getattr(self.llm, "model", "?")
        with otel_tracer.start_as_current_span(f"chat {model}", attributes={
                "gen_ai.operation.name": "chat", "gen_ai.request.model": model}) as span:
            try:
                resp = await self.llm.chat(messages, tools)
            except LLMError as err:
                span.set_status(Status(StatusCode.ERROR, str(err)))
                result.steps.append(Step(i, "error", error=str(err)))
                gen.update(level="ERROR", status_message=str(err))
                gen.end()
                return None
            span.set_attributes({"gen_ai.usage.input_tokens": resp.prompt_tokens,
                                 "gen_ai.usage.output_tokens": resp.completion_tokens,
                                 "agent.cached_tokens": resp.cached_tokens,
                                 "agent.tool_calls": len(resp.tool_calls)})
        ms = int((time.monotonic() - t0) * 1000)
        result.llm_calls += 1
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        result.cached_tokens += resp.cached_tokens
        gen.update(output={"content": resp.content, "tool_calls": [tc.name for tc in resp.tool_calls]},
                   usage_details={"input": resp.prompt_tokens, "output": resp.completion_tokens,
                                  "cache_read_input_tokens": resp.cached_tokens})
        gen.end()
        await emit(StreamEvent(type="llm_call", run_id=result.run_id, step=i, data={
            "latency_ms": ms, "tool_calls": [tc.name for tc in resp.tool_calls],
            "prompt_tokens": resp.prompt_tokens, "cached_tokens": resp.cached_tokens}))
        return resp

    async def _execute(self, registry: ToolRegistry, tc: ToolCall, i: int, seen: dict,
                       result: RunResult, trace, emit: Emit) -> str:
        key = f"{tc.name}:{tc.arguments}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 2:
            msg = "Loop detected: you already made this exact call twice. Use a different tool or arguments, or answer."
            result.steps.append(Step(i, "guard", tc.name, tc.arguments, error=msg))
            result.tool_errors += 1
            await emit(StreamEvent(type="guard", run_id=result.run_id, step=i, data={"tool": tc.name, "error": msg}))
            return json.dumps({"error": msg})

        await emit(StreamEvent(type="tool_started", run_id=result.run_id, step=i,
                               data={"tool": tc.name, "arguments": tc.arguments}))
        span = trace.start_span(name=f"tool:{tc.name}", input=tc.arguments)
        t0 = time.monotonic()
        error: str | None = None
        with otel_tracer.start_as_current_span(f"execute_tool {tc.name}", attributes={
                "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tc.name,
                "gen_ai.tool.call.id": tc.id}) as otel_span:
            try:
                out = await asyncio.to_thread(registry.call, tc.name, tc.arguments)
            except ToolError as err:
                error = str(err)
            except Exception as err:  # never let a tool bug crash the run
                error = f"Internal tool failure ({type(err).__name__}). Try another approach."
            if error:
                otel_span.set_status(Status(StatusCode.ERROR, error))
        ms = int((time.monotonic() - t0) * 1000)

        if error is None:
            result.steps.append(Step(i, "tool", tc.name, tc.arguments, result=out, latency_ms=ms))
            span.update(output=out)
            span.end()
            await emit(StreamEvent(type="tool_finished", run_id=result.run_id, step=i,
                                   data={"tool": tc.name, "latency_ms": ms}))
            return json.dumps(out, default=str)

        result.tool_errors += 1
        result.steps.append(Step(i, "tool", tc.name, tc.arguments, error=error, latency_ms=ms))
        span.update(level="ERROR", status_message=error)
        span.end()
        await emit(StreamEvent(type="tool_failed", run_id=result.run_id, step=i,
                               data={"tool": tc.name, "error": error}))
        return json.dumps({"error": error})

    @staticmethod
    def _parse_final(tc: ToolCall) -> tuple[FinalAnswer | None, str | None]:
        try:
            return FinalAnswer.model_validate_json(tc.arguments or "{}"), None
        except ValidationError as err:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in err.errors())
            return None, f"Invalid final_answer: {problems}. Call final_answer again with valid fields."

    @staticmethod
    def _finish(result: RunResult, final: FinalAnswer, structured: bool, step: int) -> None:
        result.final = final.model_dump()
        result.structured = structured
        result.answer = final.summary
        result.status = "completed"
        result.steps.append(Step(step, "final", FINAL_TOOL, result=result.final))

    async def _best_effort(self, messages: list[dict], result: RunResult, trace, emit: Emit) -> str:
        messages.append({"role": "user", "content":
                         "Step budget reached. Give your best answer now from what you have, and say what is unverified."})
        resp = await self._llm_call(messages, [], result, trace, self.max_steps, emit)
        if resp is None:
            return "Step budget reached and the model was unavailable for a summary. See the trace."
        return (resp.content or "").strip()
