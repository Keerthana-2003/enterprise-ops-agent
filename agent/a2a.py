"""Agent-to-agent (A2A) protocol: publish a specialist agent as a remote agent and call it.

Server: an Agent Card at /.well-known/agent-card.json describing the agent and its skills, and a
JSON-RPC 2.0 endpoint handling `message/send`. The reply is an A2A Message with a text part (the
answer) and a data part (the structured final answer, facts and status).
Client: RemoteSpecialist reads the card and sends messages; the supervisor treats it exactly like
an in-process specialist, so agents can be split across services, teams or accounts.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PROTOCOL_VERSION = "0.3.0"


def agent_card(name: str, description: str, url: str, skills: list[dict] | None = None) -> dict:
    return {
        "name": name, "description": description, "url": url, "version": "1.0.0",
        "protocolVersion": PROTOCOL_VERSION, "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain", "application/json"],
        "skills": skills or [{"id": name, "name": name, "description": description, "tags": ["enterprise-ops"]}],
    }


def _error(rid, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def create_a2a_app(specialist, card: dict) -> FastAPI:
    """Expose any object with `async ask(question) -> dict` (e.g. LocalSpecialist) over A2A."""
    app = FastAPI(title=f"A2A: {card['name']}")

    @app.get("/.well-known/agent-card.json")
    def get_card() -> dict:
        return card

    @app.post("/")
    async def rpc(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return _error(None, -32700, "Parse error")
        rid = body.get("id")
        if body.get("jsonrpc") != "2.0" or "method" not in body:
            return _error(rid, -32600, "Invalid Request")
        if body["method"] != "message/send":
            return _error(rid, -32601, f"Method not found: {body['method']}")
        message = (body.get("params") or {}).get("message") or {}
        text = " ".join(p.get("text", "") for p in message.get("parts", []) if p.get("kind") == "text").strip()
        if not text:
            return _error(rid, -32602, "Invalid params: message needs a text part")
        out = await specialist.ask(text)
        reply = {"kind": "message", "role": "agent", "messageId": str(uuid.uuid4()),
                 "contextId": message.get("contextId") or str(uuid.uuid4()),
                 "parts": [{"kind": "text", "text": out.get("answer") or ""}, {"kind": "data", "data": out}]}
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": reply})

    return app


class RemoteSpecialist:
    """A2A client. `client` is an httpx.AsyncClient (tests pass one bound to an in-process app)."""

    def __init__(self, base_url: str, client=None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._client = client
        self._httpx = httpx
        self.name = self.description = None

    def _new_client(self):
        return self._client or self._httpx.AsyncClient(timeout=120)

    async def load_card(self) -> dict:
        c = self._new_client()
        r = await c.get(f"{self.base_url}/.well-known/agent-card.json")
        r.raise_for_status()
        card = r.json()
        self.name, self.description = card["name"], card["description"]
        return card

    async def ask(self, question: str) -> dict:
        c = self._new_client()
        payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
                   "params": {"message": {"kind": "message", "role": "user", "messageId": str(uuid.uuid4()),
                                          "parts": [{"kind": "text", "text": question}]}}}
        r = await c.post(f"{self.base_url}/", json=payload)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"A2A error {body['error']['code']}: {body['error']['message']}")
        parts = body["result"]["parts"]
        data = next((p["data"] for p in parts if p.get("kind") == "data"), {})
        text = next((p["text"] for p in parts if p.get("kind") == "text"), None)
        return {**data, "answer": data.get("answer", text), "agent": self.name or data.get("agent")}
