"""Tool registry and the enterprise tools the agent can call.

Every tool:
  * declares a JSON schema the LLM sees,
  * validates its arguments before running,
  * runs under a time budget,
  * returns a JSON-serialisable result or raises ToolError with a message the
    model can act on (the agent feeds that message back so the model can recover).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .ontology import Ontology
from .schemas import (PolicySearchArgs, ProposeActionArgs, ResolveEntityArgs, SqlArgs,
                      SupportTicketsArgs)


class ToolError(Exception):
    """A recoverable tool failure; the message is shown to the model."""


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    side_effect: bool = False  # tools with side effects only propose actions for human approval

    @property
    def parameters(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def validate(self, args: dict) -> dict:
        try:
            return self.args_model.model_validate(args).model_dump(exclude_none=True)
        except ValidationError as err:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in err.errors())
            raise ToolError(f"Invalid arguments for {self.name}: {problems}.") from None


class ToolRegistry:
    def __init__(self, tools: list[Tool], timeout_s: float = 10.0):
        self.tools = {t.name: t for t in tools}
        self.timeout_s = timeout_s

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, raw_args: str) -> Any:
        if name not in self.tools:
            raise ToolError(f"Unknown tool '{name}'. Available: {sorted(self.tools)}.")
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as err:
            raise ToolError(f"Arguments were not valid JSON ({err.msg}). Send a JSON object.") from None
        if not isinstance(args, dict):
            raise ToolError("Arguments must be a JSON object.")
        tool = self.tools[name]
        clean = tool.validate(args)
        start = time.monotonic()
        result = tool.fn(**clean)
        if time.monotonic() - start > self.timeout_s:
            raise ToolError(f"{name} exceeded its {self.timeout_s}s time budget; narrow the request.")
        return result


# ---------------------------------------------------------------- enterprise tools

_SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", re.IGNORECASE)


def _readonly_query(db: Path, sql: str, max_rows: int = 50) -> list[dict]:
    if not _SELECT_ONLY.match(sql) or _FORBIDDEN.search(sql) or ";" in sql.strip().rstrip(";"):
        raise ToolError("Only a single read-only SELECT statement is allowed.")
    uri = f"file:{db}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchmany(max_rows)
    except sqlite3.Error as err:
        raise ToolError(f"SQL error: {err}. Check table and column names.") from None
    return [dict(r) for r in rows]


class DocSearch:
    """TF-IDF search over policy documents, split into one passage per line."""

    def __init__(self, docs_dir: Path):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages: list[tuple[str, str]] = []
        for path in sorted(Path(docs_dir).glob("*.md")):
            for line in path.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    self.passages.append((path.name, line.strip()))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([p[1] for p in self.passages])

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        top = scores.argsort()[::-1][:k]
        return [{"source": self.passages[i][0], "text": self.passages[i][1], "score": round(float(scores[i]), 3)}
                for i in top if scores[i] > 0]


def build_tools(data_dir: Path) -> tuple[ToolRegistry, Ontology, list[dict]]:
    """Wire up all tools over a data directory. Returns (registry, ontology, proposed_actions log)."""
    data_dir = Path(data_dir)
    ontology = Ontology(data_dir / "ontology.json")
    docs = DocSearch(data_dir / "docs")
    vector_store = None
    if os.getenv("RETRIEVAL") == "bedrock-kb":  # Knowledge Bases for Amazon Bedrock (managed RAG)
        from .bedrock_services import KnowledgeBaseRetriever

        vector_store = KnowledgeBaseRetriever()
    elif os.getenv("DATABASE_URL"):  # PostgreSQL + pgvector retrieval instead of in-memory TF-IDF
        from .memory import store_from_env

        vector_store = store_from_env(os.environ["DATABASE_URL"], str(data_dir / "docs"))
    tickets = json.loads((data_dir / "tickets.json").read_text())
    proposed_actions: list[dict] = []
    known_domains = {e["identifiers"].get("support_domain") for e in ontology.entities.values()} - {None}

    def resolve_entity(mention: str, entity_type: str | None = None) -> dict:
        matches = ontology.resolve(mention, entity_type)
        if not matches:
            raise ToolError(f"No entity matches '{mention}'. Try the full company name or email domain.")
        best = matches[0]
        result = {"best_match": {k: best[k] for k in ("id", "type", "name", "identifiers", "attributes", "match_score")},
                  "relations": ontology.neighbors(best["id"])}
        if len(matches) > 1 and matches[1]["match_score"] >= best["match_score"] - 0.05:
            result["warning"] = "Ambiguous mention; other candidates: " + ", ".join(m["name"] for m in matches[1:])
        return result

    def query_crm(sql: str) -> list[dict]:
        return _readonly_query(data_dir / "crm.db", sql)

    def query_billing(sql: str) -> list[dict]:
        return _readonly_query(data_dir / "billing.db", sql)

    def get_support_tickets(support_domain: str, status: str | None = None) -> list[dict]:
        if support_domain not in known_domains:
            raise ToolError(f"Unknown support domain '{support_domain}'. Use resolve_entity to get the right domain.")
        return [t for t in tickets if t["domain"] == support_domain and (status is None or t["status"] == status)]

    def search_policies(query: str) -> list[dict]:
        hits = vector_store.search(query) if vector_store is not None else docs.search(query)
        if not hits:
            raise ToolError("No policy passages matched. Rephrase with policy terms like escalation, credit, SLA.")
        return hits

    def propose_action(action_type: str, customer_id: str, reason: str) -> dict:
        if customer_id not in ontology.entities:
            raise ToolError(f"Unknown customer_id '{customer_id}'. Use the canonical id from resolve_entity.")
        action = {"action_id": f"ACT-{uuid.uuid4().hex[:8]}", "action_type": action_type,
                  "customer_id": customer_id, "reason": reason, "status": "pending_human_approval"}
        proposed_actions.append(action)
        return action

    tools = [
        Tool("resolve_entity",
             "Resolve a company, person or product mention to its canonical entity, its identifiers in every "
             "source system (crm id, billing code, support domain), and its relationships. Call this first.",
             ResolveEntityArgs,
             resolve_entity),
        Tool("query_crm",
             "Read-only SQL over the CRM. Table customers(id INTEGER, name, tier, region, account_manager).",
             SqlArgs,
             query_crm),
        Tool("query_billing",
             "Read-only SQL over billing. Table invoices(invoice_id, billing_code, amount_usd, status, due_date). "
             "status is one of paid, open, overdue. billing_code comes from resolve_entity.",
             SqlArgs,
             query_billing),
        Tool("get_support_tickets",
             "List support tickets for a customer by support email domain (from resolve_entity).",
             SupportTicketsArgs,
             get_support_tickets),
        Tool("search_policies",
             "Search company policies (escalation, credits/refunds, SLA, collections).",
             PolicySearchArgs,
             search_policies),
        Tool("propose_action",
             "Propose a business action for human approval. Never executes it.",
             ProposeActionArgs,
             propose_action, side_effect=True),
    ]
    return ToolRegistry(tools), ontology, proposed_actions
