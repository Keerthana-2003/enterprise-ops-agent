"""Async LLM client layer.

`AsyncOpenAIChatClient` talks to any OpenAI-compatible chat API (OpenAI, Azure OpenAI, or a
local Llama 3 served by Ollama / vLLM via base_url) with timeouts and exponential backoff.
It reports cached prompt tokens so prompt-caching effectiveness can be measured.
`ScriptedLLM` replays canned responses so the agent loop can be tested offline.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the provider's prompt cache


class LLMError(Exception):
    """Raised when the model cannot be reached after all retries."""


class LLMClient(Protocol):
    model: str

    async def chat(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> LLMResponse: ...


class AsyncOpenAIChatClient:
    """OpenAI-compatible async client with timeouts and exponential backoff with jitter."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ):
        from openai import AsyncOpenAI  # imported lazily so tests run without the SDK

        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed-for-local"),
            timeout=timeout_s,
            max_retries=0,  # retries are handled here so they are visible in traces
        )
        self.max_retries = max_retries

    async def chat(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                calls = [
                    ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
                    for tc in (msg.tool_calls or [])
                ]
                usage = resp.usage
                details = getattr(usage, "prompt_tokens_details", None)
                return LLMResponse(
                    content=msg.content,
                    tool_calls=calls,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    cached_tokens=getattr(details, "cached_tokens", 0) or 0,
                )
            except Exception as err:  # network errors, rate limits, 5xx
                last_err = err
                if attempt < self.max_retries:
                    await asyncio.sleep(min(8.0, (2**attempt) + random.random()))
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")


class ScriptedLLM:
    """Deterministic fake model for tests: returns responses in order.

    Each item is either an LLMResponse or an Exception to raise (to simulate outages).
    """

    model = "scripted"

    def __init__(self, script: list[LLMResponse | Exception]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> LLMResponse:
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        if not self.script:
            return LLMResponse(content="(script exhausted)")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_call(name: str, args: dict[str, Any] | str, call_id: str | None = None) -> LLMResponse:
    """Helper for building a scripted response with one tool call."""
    return tool_calls([(name, args, call_id)])


def tool_calls(calls: list[tuple]) -> LLMResponse:
    """Helper for a scripted response with several (parallel) tool calls: [(name, args[, id]), ...]."""
    out = []
    for i, c in enumerate(calls):
        name, args = c[0], c[1]
        cid = c[2] if len(c) > 2 and c[2] else f"call_{name}_{i}"
        out.append(ToolCall(id=cid, name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
    return LLMResponse(content=None, tool_calls=out)


def final(summary: str, **fields: Any) -> LLMResponse:
    """Helper: a scripted structured final answer."""
    return tool_call("final_answer", {"summary": summary, **fields})
