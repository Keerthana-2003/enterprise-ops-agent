"""PostgreSQL + pgvector retrieval and conversation memory. The database tests run in CI against a
pgvector/pgvector Postgres service; the embedder tests run anywhere."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from agent.memory import HashingEmbedder, chunk_documents, with_history
from agent.seed import seed


def test_hashing_embedder_is_normalised_and_semantic_enough():
    e = HashingEmbedder(256)
    v = e.embed(["overdue invoice escalation", "overdue invoice escalation", "uptime SLA for enterprise"])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)
    assert float(v[0] @ v[1]) == pytest.approx(1.0) and float(v[0] @ v[2]) < 0.5


def test_history_folding_keeps_recent_turns():
    hist = [{"role": "user", "content": "Who manages Fabrikam?"}, {"role": "assistant", "content": "Priya Shah."}]
    task = with_history("What about their invoices?", hist)
    assert task.startswith("Conversation so far:") and "Priya Shah." in task and task.endswith("their invoices?")
    assert with_history("hi there", []) == "hi there"


needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def store():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    pytest.importorskip("psycopg")
    from agent.memory import PgVectorStore

    s = PgVectorStore(os.environ["DATABASE_URL"], HashingEmbedder(), prefix=f"t{uuid.uuid4().hex[:6]}")
    yield s
    with s.conn.cursor() as cur:
        for t in ("messages", "conversations", "policy_chunks"):
            cur.execute(f"DROP TABLE IF EXISTS {s.p}_{t} CASCADE")


@needs_db
def test_vector_search_over_policies(store):
    data_dir = seed(Path(tempfile.mkdtemp()) / "data")
    chunks = chunk_documents(data_dir / "docs")
    assert store.upsert_chunks(chunks) == len(chunks)
    assert store.upsert_chunks(chunks) == 0  # idempotent ingestion
    store.migrate()  # migrations are re-runnable
    hits = store.search("open P1 ticket and overdue invoice: escalate or send collections notice?", k=3)
    assert hits[0]["source"] == "escalation_policy.md" and hits[0]["score"] > hits[-1]["score"]


@needs_db
def test_conversation_memory_order_limit_and_cascade(store):
    cid = store.start_conversation("user-1")
    for i in range(5):
        store.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m["content"] for m in store.history(cid)] == ["m0", "m1", "m2", "m3", "m4"]
    assert [m["content"] for m in store.history(cid, limit=2)] == ["m3", "m4"]
    store.delete_conversation(cid)
    assert store.history(cid) == []


@needs_db
def test_search_policies_tool_uses_pgvector(monkeypatch):
    from agent import memory
    from agent.tools import build_tools

    memory.store_from_env.cache_clear()
    data_dir = seed(Path(tempfile.mkdtemp()) / "data")
    registry, _, _ = build_tools(data_dir)
    hits = registry.call("search_policies", json.dumps({"query": "how much can service credits be capped at"}))
    assert hits[0]["source"] == "refund_policy.md"
