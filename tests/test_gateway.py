"""Model gateway: auth, allow-lists, rate limits, budgets, fallback, cost accounting, HTTP API."""
from __future__ import annotations

import asyncio

import pytest

from agent.llm import LLMError, LLMResponse, ScriptedLLM, tool_call
from gateway.core import Gateway, GatewayError, ModelRoute, Team, TokenBucket, hash_key

KEY = "team-ops-secret"


def make_gateway(fast_script, smart_script, rpm=60, daily=1_000_000, clock=None):
    teams = {hash_key(KEY): Team("ops", {"fast", "smart"}, rpm=rpm, daily_tokens=daily)}
    routes = {"fast": ModelRoute("fast", "slm", 0.0, 0.0, fallback="smart"),
              "smart": ModelRoute("smart", "big", 3.0, 15.0)}
    kw = {"clock": clock} if clock else {}
    return Gateway(teams, routes, {"slm": ScriptedLLM(fast_script), "big": ScriptedLLM(smart_script)}, **kw)


def resp(text="ok", p=1000, c=100):
    return LLMResponse(content=text, prompt_tokens=p, completion_tokens=c)


def run(coro):
    return asyncio.run(coro)


def test_auth_and_allow_list():
    gw = make_gateway([resp()], [resp()])
    with pytest.raises(GatewayError) as e:
        run(gw.complete("wrong", "fast", []))
    assert e.value.status == 401
    with pytest.raises(GatewayError) as e:
        run(gw.complete(KEY, "gpt-9", []))
    assert e.value.status == 404


def test_openai_shape_cost_and_tool_calls():
    gw = make_gateway([], [tool_call("lookup", {"q": "x"})])
    gw.clients["big"].script[0].prompt_tokens, gw.clients["big"].script[0].completion_tokens = 2000, 100
    out = run(gw.complete(KEY, "smart", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]))
    assert out["object"] == "chat.completion" and out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert out["gateway"]["cost_usd"] == pytest.approx((2000 * 3 + 100 * 15) / 1e6)


def test_fallback_when_primary_fails():
    gw = make_gateway([LLMError("ollama down")], [resp("from smart")])
    out = run(gw.complete(KEY, "fast", [{"role": "user", "content": "hi"}]))
    assert out["model"] == "smart" and out["gateway"]["fallback"] and out["choices"][0]["message"]["content"] == "from smart"
    report = {(r["team"], r["model"]): r for r in gw.usage_report()}
    assert report[("ops", "fast")]["errors"] == 1 and report[("ops", "fast")]["fallbacks"] == 1
    assert report[("ops", "smart")]["requests"] == 1


def test_all_upstreams_failing_is_502():
    gw = make_gateway([LLMError("down")], [LLMError("down too")])
    with pytest.raises(GatewayError) as e:
        run(gw.complete(KEY, "fast", []))
    assert e.value.status == 502 and "fast -> smart" in e.value.message


def test_rate_limit_and_refill():
    now = [0.0]
    bucket = TokenBucket(2, clock=lambda: now[0])
    assert bucket.take() and bucket.take() and not bucket.take()
    now[0] += 30  # half a minute refills one request at 2/minute
    assert bucket.take() and not bucket.take()
    gw = make_gateway([resp(), resp(), resp()], [], rpm=2, clock=lambda: now[0])
    run(gw.complete(KEY, "fast", []))
    run(gw.complete(KEY, "fast", []))
    with pytest.raises(GatewayError) as e:
        run(gw.complete(KEY, "fast", []))
    assert e.value.status == 429


def test_daily_token_budget():
    gw = make_gateway([resp(p=900, c=100), resp()], [], daily=1000)
    run(gw.complete(KEY, "fast", []))
    with pytest.raises(GatewayError) as e:
        run(gw.complete(KEY, "fast", []))
    assert e.value.status == 429 and "budget" in e.value.message


def test_http_api():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gateway.app import create_app

    gw = make_gateway([resp("hello")], [])
    client = TestClient(create_app(gw, admin_key="admin"))
    auth = {"Authorization": f"Bearer {KEY}"}
    r = client.post("/v1/chat/completions", json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
                    headers=auth)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello"
    assert client.post("/v1/chat/completions", json={"model": "fast", "messages": []}).status_code == 401
    assert [m["id"] for m in client.get("/v1/models", headers=auth).json()["data"]] == ["fast", "smart"]
    assert client.get("/admin/usage").status_code == 401
    assert client.get("/admin/usage", headers={"X-Admin-Key": "admin"}).json()[0]["requests"] == 1
