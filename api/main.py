"""FastAPI service exposing the agent.

Endpoints
  GET  /health                 liveness + model + whether tracing is on
  POST /runs                   run synchronously, return the full result
  POST /runs/stream            stream run events as Server-Sent Events (text/event-stream)
  POST /runs/async             accept the run (202), deliver the result to `callback_url` by webhook
  GET  /runs/{run_id}          fetch a stored result

All /runs endpoints require a Cognito bearer token when COGNITO_USER_POOL_ID is set.
LLM provider: LLM_PROVIDER=openai (default, any OpenAI-compatible endpoint) or bedrock.
Agent runtime for POST /runs and /runs/async: AGENT_RUNTIME=loop (default) or langgraph.

Run locally:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from agent.bedrock import make_llm
from agent.core import Agent
from agent.seed import DATA, seed
from agent.telemetry import configure_otel
from agent.tracing import Tracer
from api.auth import AuthError, bearer_token, verifier_from_env

log = logging.getLogger("api")
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", DATA))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
MAX_STORED_RUNS = 500

app = FastAPI(title="Enterprise Ops Agent", version="0.2.0")
_runs: "OrderedDict[str, dict]" = OrderedDict()
_llm = None
_tracer = Tracer()
_verifier = verifier_from_env()
configure_otel()


def require_user(authorization: str | None = Header(default=None)) -> dict | None:
    """Cognito JWT check; a no-op when no user pool is configured (local development)."""
    if _verifier is None:
        return None
    try:
        return _verifier.verify(bearer_token(authorization))
    except AuthError as err:
        raise HTTPException(status_code=401, detail=str(err), headers={"WWW-Authenticate": "Bearer"}) from None
_background: set[asyncio.Task] = set()


class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000,
                      examples=["Fabrikam has open tickets and overdue invoices. What should we do?"])
    max_steps: int = Field(10, ge=1, le=25)
    max_total_tokens: int = Field(60_000, ge=1_000, le=500_000)


class AsyncRunRequest(RunRequest):
    callback_url: HttpUrl


def make_agent(req: RunRequest, allow_graph: bool = True):
    global _llm
    if not (DATA_DIR / "ontology.json").exists():
        seed(DATA_DIR)
    if _llm is None:
        _llm = make_llm()
    if allow_graph and os.getenv("AGENT_RUNTIME", "loop") == "langgraph":
        from agent.graph import LangGraphAgent

        return LangGraphAgent(_llm, DATA_DIR, max_steps=req.max_steps, max_total_tokens=req.max_total_tokens)
    return Agent(_llm, DATA_DIR, max_steps=req.max_steps, max_total_tokens=req.max_total_tokens, tracer=_tracer)


def _store(result: dict) -> None:
    _runs[result["run_id"]] = result
    while len(_runs) > MAX_STORED_RUNS:
        _runs.popitem(last=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "openai"),
            "runtime": os.getenv("AGENT_RUNTIME", "loop"), "tracing": _tracer.enabled, "auth": _verifier is not None}


@app.post("/runs")
async def create_run(req: RunRequest, user: dict | None = Depends(require_user)) -> dict:
    result = (await make_agent(req).arun(req.task)).to_dict()
    _store(result)
    return result


@app.post("/runs/stream")
async def stream_run(req: RunRequest, user: dict | None = Depends(require_user)) -> StreamingResponse:
    agent = make_agent(req, allow_graph=False)  # streaming events come from the hand-written loop

    async def events():
        async for ev in agent.stream(req.task):
            if ev.type == "run_finished":
                _store(ev.data)
            yield f"event: {ev.type}\ndata: {ev.model_dump_json()}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _deliver_webhook(url: str, payload: dict, attempts: int = 4) -> None:
    body = json.dumps(payload, default=str).encode()
    headers = {"Content-Type": "application/json", "X-Delivery-Id": uuid.uuid4().hex}
    if WEBHOOK_SECRET:  # receivers verify the payload came from us
        headers["X-Signature-SHA256"] = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(attempts):
            try:
                resp = await client.post(url, content=body, headers=headers)
                if resp.status_code < 500:
                    return
            except httpx.HTTPError as err:
                log.warning("webhook attempt %d failed: %s", attempt + 1, err)
            await asyncio.sleep(2**attempt)
    log.error("webhook delivery to %s failed after %d attempts", url, attempts)


@app.post("/runs/async", status_code=202)
async def create_async_run(req: AsyncRunRequest, user: dict | None = Depends(require_user)) -> dict:
    agent = make_agent(req)
    ticket = uuid.uuid4().hex[:12]

    async def job() -> None:
        result = (await agent.arun(req.task)).to_dict()
        result["request_id"] = ticket
        _store(result)
        await _deliver_webhook(str(req.callback_url), result)

    task = asyncio.create_task(job())
    _background.add(task)
    task.add_done_callback(_background.discard)
    return {"request_id": ticket, "status": "accepted"}


@app.get("/runs/{run_id}")
def get_run(run_id: str, user: dict | None = Depends(require_user)) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]
