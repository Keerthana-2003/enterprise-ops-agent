"""API tests: sync runs, SSE streaming, async runs with webhook delivery, and 404s."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402
from agent.llm import ScriptedLLM, final, tool_call  # noqa: E402
from agent.seed import seed  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    data_dir = seed(Path(tempfile.mkdtemp()) / "data")
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "_llm", ScriptedLLM([
        tool_call("resolve_entity", {"mention": "Fabrikam"}),
        final("Fabrikam is customer:3, managed by Priya Shah.", customer_id="customer:3"),
    ]))
    return TestClient(main.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "tracing" in body


def test_sync_run_and_lookup(client):
    result = client.post("/runs", json={"task": "Who is Fabrikam?"}).json()
    assert result["status"] == "completed" and result["structured"]
    assert client.get(f"/runs/{result['run_id']}").json()["answer"] == result["answer"]
    assert client.get("/runs/does-not-exist").status_code == 404


def test_stream_run_sends_sse_events(client):
    resp = client.post("/runs/stream", json={"task": "Who is Fabrikam?"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [line[len("event: "):] for line in resp.text.splitlines() if line.startswith("event: ")]
    assert events[0] == "run_started" and events[-1] == "run_finished"
    last = [line for line in resp.text.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(last[len("data: "):])["data"]["status"] == "completed"


def test_validation_rejects_bad_request(client):
    assert client.post("/runs", json={"task": "x"}).status_code == 422
    assert client.post("/runs/async", json={"task": "Who is Fabrikam?", "callback_url": "not-a-url"}).status_code == 422


def test_async_run_delivers_webhook(client, monkeypatch):
    delivered: list[tuple[str, dict]] = []

    async def fake_deliver(url, payload, attempts=4):
        delivered.append((url, payload))

    monkeypatch.setattr(main, "_deliver_webhook", fake_deliver)
    with client:  # keeps the app's event loop alive so the background job can finish
        resp = client.post("/runs/async", json={"task": "Who is Fabrikam?",
                                                "callback_url": "https://example.com/hook"})
        assert resp.status_code == 202
        request_id = resp.json()["request_id"]
        for _ in range(50):
            if delivered:
                break
            time.sleep(0.05)  # the job runs on the app's event loop thread
    assert delivered and delivered[0][0] == "https://example.com/hook"
    assert delivered[0][1]["request_id"] == request_id and delivered[0][1]["status"] == "completed"
