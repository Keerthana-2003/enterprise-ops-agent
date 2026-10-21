"""AWS Bedrock provider using the Converse API (Claude, Llama, Mistral, Nova on Bedrock).

Converts the agent's OpenAI-style messages and tool schemas to Converse format and back, so the
same agent loop runs unchanged on OpenAI, a local Llama 3, or Bedrock. Optional prompt caching
adds a cache point after the stable system prompt (on models that support it).

Configure with:  LLM_PROVIDER=bedrock  BEDROCK_MODEL_ID=...  AWS_REGION=us-east-1
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

from .llm import LLMError, LLMResponse, ToolCall


def to_converse(messages: list[dict], tools: list[dict], prompt_cache: bool = False) -> dict:
    """Translate OpenAI chat format into Converse API request fields."""
    system: list[dict] = []
    converse: list[dict] = []

    def add(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if converse and converse[-1]["role"] == role:  # Converse requires alternating roles
            converse[-1]["content"].extend(blocks)
        else:
            converse.append({"role": role, "content": blocks})

    for m in messages:
        role, content = m["role"], m.get("content")
        if role == "system":
            system.append({"text": content})
        elif role == "user":
            add("user", [{"text": content}])
        elif role == "assistant":
            blocks = [{"text": content}] if content else []
            for tc in m.get("tool_calls") or []:
                raw = tc["function"]["arguments"] or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {"_unparsed_arguments": raw}
                blocks.append({"toolUse": {"toolUseId": tc["id"], "name": tc["function"]["name"],
                                           "input": args if isinstance(args, dict) else {"value": args}}})
            add("assistant", blocks)
        elif role == "tool":
            add("user", [{"toolResult": {"toolUseId": m["tool_call_id"], "content": [{"text": content}]}}])

    if prompt_cache and system:
        system.append({"cachePoint": {"type": "default"}})
    request: dict[str, Any] = {"system": system, "messages": converse}
    if tools:
        request["toolConfig"] = {"tools": [
            {"toolSpec": {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                          "inputSchema": {"json": t["function"]["parameters"]}}} for t in tools]}
    return request


def from_converse(response: dict) -> LLMResponse:
    """Translate a Converse API response into the agent's LLMResponse."""
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(b["text"] for b in blocks if "text" in b) or None
    calls = [ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"],
                      arguments=json.dumps(b["toolUse"].get("input", {})))
             for b in blocks if "toolUse" in b]
    usage = response.get("usage", {})
    return LLMResponse(content=text, tool_calls=calls,
                       prompt_tokens=usage.get("inputTokens", 0) + usage.get("cacheReadInputTokens", 0),
                       completion_tokens=usage.get("outputTokens", 0),
                       cached_tokens=usage.get("cacheReadInputTokens", 0))


class BedrockConverseClient:
    def __init__(self, model: str | None = None, region: str | None = None, max_retries: int = 3,
                 prompt_cache: bool | None = None, client: Any = None):
        self.model = model or os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
        self.max_retries = max_retries
        self.prompt_cache = prompt_cache if prompt_cache is not None else os.getenv("BEDROCK_PROMPT_CACHE") == "1"
        self._last_tools: list[dict] = []
        if client is None:
            import boto3  # imported lazily so tests run without the AWS SDK

            client = boto3.client("bedrock-runtime", region_name=region or os.getenv("AWS_REGION", "us-east-1"))
        self.client = client

    async def chat(self, messages: list[dict], tools: list[dict], json_mode: bool = False) -> LLMResponse:
        # Converse rejects a history containing toolUse blocks without a toolConfig,
        # so a follow-up call without tools reuses the last tool definitions.
        if tools:
            self._last_tools = tools
        if json_mode:
            messages = messages + [{"role": "user", "content": "Respond with a single JSON object only."}]
        request = to_converse(messages, tools or self._last_tools, self.prompt_cache)
        request["inferenceConfig"] = {"temperature": 0}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await asyncio.to_thread(self.client.converse, modelId=self.model, **request)
                return from_converse(resp)
            except Exception as err:  # throttling, timeouts, 5xx
                last_err = err
                if attempt < self.max_retries:
                    await asyncio.sleep(min(8.0, (2**attempt) + random.random()))
        raise LLMError(f"Bedrock call failed after {self.max_retries + 1} attempts: {last_err}")


def make_llm(model: str | None = None):
    """Kept for backwards compatibility; see agent.providers.make_llm."""
    from .providers import make_llm as _make_llm

    return _make_llm(model)
