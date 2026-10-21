"""Model gateway core: one controlled entry point between applications and model providers.

* authentication with per-team API keys (stored only as SHA-256 hashes)
* per-team model allow-lists, request rate limits (token bucket) and daily token budgets
* model aliases ("fast", "smart") mapped to provider models, so apps never hard-code a vendor
* fallback to a secondary model when the primary provider fails
* usage and cost accounting per team and model (for chargeback and inference-cost management)

Pure Python and provider-agnostic: upstream clients are anything with the agent's LLMClient
interface (OpenAI-compatible, Azure OpenAI, Vertex AI, Bedrock). gateway/app.py exposes it as an
OpenAI-compatible HTTP API.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm import LLMError, LLMResponse


class GatewayError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass
class ModelRoute:
    alias: str
    upstream: str                  # key into the gateway's client registry
    price_in: float = 0.0          # $ per 1M input tokens (input to cost accounting)
    price_out: float = 0.0
    fallback: str | None = None    # alias to try if this one fails


@dataclass
class Team:
    name: str
    models: set[str]
    rpm: int = 60
    daily_tokens: int = 1_000_000


class TokenBucket:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.rate, self.clock = per_minute, per_minute / 60.0, clock
        self.tokens, self.updated = float(per_minute), clock()

    def take(self) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    fallbacks: int = 0
    errors: int = 0


@dataclass
class Gateway:
    teams: dict[str, Team]                     # key hash -> team
    routes: dict[str, ModelRoute]              # alias -> route
    clients: dict[str, Any]                    # upstream name -> LLMClient
    clock: Callable[[], float] = time.monotonic
    day: Callable[[], str] = field(default=lambda: time.strftime("%Y-%m-%d"))
    usage: dict = field(default_factory=lambda: defaultdict(Usage))       # (team, alias) -> Usage
    _buckets: dict = field(default_factory=dict)
    _daily: dict = field(default_factory=lambda: defaultdict(int))        # (team, day) -> tokens

    def authenticate(self, api_key: str | None) -> Team:
        team = self.teams.get(hash_key(api_key or ""))
        if team is None:
            raise GatewayError(401, "Invalid API key")
        return team

    def _admit(self, team: Team, alias: str) -> None:
        if alias not in self.routes:
            raise GatewayError(404, f"Unknown model '{alias}'. Available: {sorted(self.routes)}")
        if alias not in team.models:
            raise GatewayError(403, f"Team '{team.name}' may not use model '{alias}'")
        bucket = self._buckets.setdefault(team.name, TokenBucket(team.rpm, self.clock))
        if not bucket.take():
            raise GatewayError(429, f"Rate limit of {team.rpm} requests/minute exceeded")
        if self._daily[(team.name, self.day())] >= team.daily_tokens:
            raise GatewayError(429, f"Daily token budget of {team.daily_tokens} exhausted")

    def _record(self, team: Team, route: ModelRoute, resp: LLMResponse) -> float:
        cost = (resp.prompt_tokens * route.price_in + resp.completion_tokens * route.price_out) / 1e6
        u = self.usage[(team.name, route.alias)]
        u.requests += 1
        u.prompt_tokens += resp.prompt_tokens
        u.completion_tokens += resp.completion_tokens
        u.cost_usd += cost
        self._daily[(team.name, self.day())] += resp.prompt_tokens + resp.completion_tokens
        return cost

    async def complete(self, api_key: str | None, alias: str, messages: list[dict], tools: list[dict] | None = None,
                       json_mode: bool = False) -> dict:
        team = self.authenticate(api_key)
        self._admit(team, alias)
        tried, current, last_err = [], alias, None
        while current and current not in tried:
            route = self.routes[current]
            tried.append(current)
            try:
                resp = await self.clients[route.upstream].chat(messages, tools or [], json_mode)
                cost = self._record(team, route, resp)
                if len(tried) > 1:
                    self.usage[(team.name, alias)].fallbacks += 1
                return to_openai_response(resp, served_by=current, requested=alias, cost=cost)
            except LLMError as err:
                self.usage[(team.name, current)].errors += 1
                last_err = err
                current = route.fallback
        raise GatewayError(502, f"All upstreams failed ({' -> '.join(tried)}): {last_err}")

    def usage_report(self) -> list[dict]:
        return [{"team": t, "model": m, **vars(u), "cost_usd": round(u.cost_usd, 6)} for (t, m), u in self.usage.items()]


def to_openai_response(resp: LLMResponse, served_by: str, requested: str, cost: float) -> dict:
    message: dict = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        message["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name, "arguments": tc.arguments}} for tc in resp.tool_calls]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()),
        "model": served_by,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if resp.tool_calls else "stop"}],
        "usage": {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                  "total_tokens": resp.prompt_tokens + resp.completion_tokens,
                  "prompt_tokens_details": {"cached_tokens": resp.cached_tokens}},
        "gateway": {"requested_model": requested, "served_by": served_by, "fallback": served_by != requested,
                    "cost_usd": round(cost, 8)},
    }
