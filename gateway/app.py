"""OpenAI-compatible HTTP API for the model gateway.

  POST /v1/chat/completions   Authorization: Bearer <team key>; body: {model, messages, tools?, response_format?}
  GET  /v1/models             aliases the calling team may use
  GET  /admin/usage           usage and cost per team and model (X-Admin-Key header)

Configuration (GATEWAY_CONFIG, JSON): teams keyed by SHA-256 of their API key, model routes with
prices and fallbacks, and upstreams built with agent.providers. Example: gateway/config.example.json.
Any OpenAI SDK can use the gateway by setting base_url to http://<gateway>/v1.
"""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .core import Gateway, GatewayError, ModelRoute, Team


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    response_format: dict | None = None


def build_upstream(spec: dict):
    from agent.llm import AsyncOpenAIChatClient

    kind = spec["provider"]
    if kind == "openai-compatible":
        return AsyncOpenAIChatClient(model=spec["model"], base_url=spec.get("base_url"),
                                     api_key=os.getenv(spec.get("api_key_env", "OPENAI_API_KEY"), "local"))
    if kind == "azure":
        from agent.providers import azure_openai_client

        return azure_openai_client(spec["model"])
    if kind == "vertex":
        from agent.providers import VertexAIClient

        return VertexAIClient(spec["model"])
    if kind == "bedrock":
        from agent.bedrock import BedrockConverseClient

        return BedrockConverseClient(model=spec["model"])
    raise ValueError(f"unknown provider {kind}")


def gateway_from_config(cfg: dict, clients: dict | None = None) -> Gateway:
    teams = {h: Team(t["name"], set(t["models"]), t.get("rpm", 60), t.get("daily_tokens", 1_000_000))
             for h, t in cfg["teams"].items()}
    routes = {a: ModelRoute(a, r["upstream"], r.get("price_in", 0.0), r.get("price_out", 0.0), r.get("fallback"))
              for a, r in cfg["routes"].items()}
    clients = clients or {name: build_upstream(spec) for name, spec in cfg["upstreams"].items()}
    return Gateway(teams, routes, clients)


def create_app(gateway: Gateway, admin_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Model Gateway", version="0.1.0")

    def bearer(authorization: str | None) -> str | None:
        if authorization and authorization.lower().startswith("bearer "):
            return authorization.split(" ", 1)[1].strip()
        return None

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, authorization: str | None = Header(default=None)) -> dict:
        try:
            return await gateway.complete(bearer(authorization), req.model, req.messages, req.tools,
                                          json_mode=(req.response_format or {}).get("type") == "json_object")
        except GatewayError as err:
            raise HTTPException(err.status, err.message) from None

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict:
        try:
            team = gateway.authenticate(bearer(authorization))
        except GatewayError as err:
            raise HTTPException(err.status, err.message) from None
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in sorted(team.models)]}

    @app.get("/admin/usage")
    def usage(x_admin_key: str | None = Header(default=None)) -> list[dict]:
        if not admin_key or not x_admin_key or not hmac.compare_digest(admin_key, x_admin_key):
            raise HTTPException(401, "Admin key required")
        return gateway.usage_report()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "routes": sorted(gateway.routes)}

    return app


def app_from_env() -> FastAPI:
    cfg = json.loads(Path(os.environ["GATEWAY_CONFIG"]).read_text())
    return create_app(gateway_from_config(cfg), os.getenv("GATEWAY_ADMIN_KEY"))
