"""PostgreSQL + pgvector storage for the assistant.

* policy_chunks   document chunks with embeddings, searched by cosine distance (HNSW index)
* conversations / messages   multi-turn memory, so the assistant can follow up on earlier answers

Enabled when DATABASE_URL is set (e.g. postgresql://user:pass@host:5432/db with the `vector`
extension available). Embeddings come from OpenAI (`EMBEDDINGS=openai`) or, by default, a local
hashing embedder that needs no API key, so tests and offline demos behave the same way.
Vectors are passed as text and cast with ::vector, so only psycopg is required.
"""
from __future__ import annotations

import os
import re
import uuid
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Signed feature hashing of words and word bigrams, L2-normalised. Deterministic and offline."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, text in enumerate(texts):
            words = _TOKEN.findall(text.lower())
            for tok in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI

        self.client, self.model, self.dim = OpenAI(), model, dim

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.asarray([d.embedding for d in resp.data], np.float32)


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def chunk_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """One chunk per non-heading line (policies here are short); returns (source, text)."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                chunks.append((path.name, line.strip()))
    return chunks


class PgVectorStore:
    def __init__(self, dsn: str, embedder=None, prefix: str = "agent"):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.embedder = embedder or HashingEmbedder()
        self.p = prefix
        self.migrate()

    def migrate(self) -> None:
        p, d = self.p, self.embedder.dim
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {p}_policy_chunks (
                id BIGSERIAL PRIMARY KEY, source TEXT NOT NULL, chunk TEXT NOT NULL,
                embedding vector({d}) NOT NULL, UNIQUE (source, chunk))""")
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {p}_policy_chunks_hnsw
                ON {p}_policy_chunks USING hnsw (embedding vector_cosine_ops)""")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {p}_conversations (
                id UUID PRIMARY KEY, user_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {p}_messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id UUID NOT NULL REFERENCES {p}_conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp())""")
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {p}_messages_conv
                ON {p}_messages (conversation_id, created_at)""")

    # ------------------------------------------------------------------ retrieval
    def upsert_chunks(self, chunks: list[tuple[str, str]]) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.embed([c[1] for c in chunks])
        inserted = 0
        with self.conn.cursor() as cur:
            for (source, text), v in zip(chunks, vecs):
                cur.execute(f"""INSERT INTO {self.p}_policy_chunks (source, chunk, embedding)
                    VALUES (%s, %s, %s::vector) ON CONFLICT (source, chunk) DO NOTHING""", (source, text, _vec(v)))
                inserted += cur.rowcount
        return inserted

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embedder.embed_query(query) if hasattr(self.embedder, "embed_query") else self.embedder.embed([query])[0]
        v = _vec(q)
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT source, chunk, 1 - (embedding <=> %s::vector) AS score
                FROM {self.p}_policy_chunks ORDER BY embedding <=> %s::vector LIMIT %s""", (v, v, k))
            rows = cur.fetchall()
        return [{"source": s, "text": t, "score": round(float(sc), 3)} for s, t, sc in rows]

    # ------------------------------------------------------------------ conversation memory
    def start_conversation(self, user_id: str | None = None, conversation_id: str | None = None) -> str:
        cid = conversation_id or str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.p}_conversations (id, user_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                        (cid, user_id))
        return cid

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.p}_messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                        (conversation_id, role, content))

    def history(self, conversation_id: str, limit: int = 20) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT role, content FROM (
                SELECT role, content, created_at, id FROM {self.p}_messages WHERE conversation_id = %s
                ORDER BY created_at DESC, id DESC LIMIT %s) t ORDER BY created_at, id""", (conversation_id, limit))
            return [{"role": r, "content": c} for r, c in cur.fetchall()]

    def delete_conversation(self, conversation_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.p}_conversations WHERE id = %s", (conversation_id,))


def embedder_from_env():
    kind = os.getenv("EMBEDDINGS", "hashing")
    if kind == "openai":
        return OpenAIEmbedder()
    if kind.startswith("bedrock"):
        from .bedrock_services import BedrockEmbedder

        return BedrockEmbedder("cohere.embed-english-v3" if kind == "bedrock-cohere" else "amazon.titan-embed-text-v2:0")
    return HashingEmbedder()


@lru_cache(maxsize=4)
def store_from_env(dsn: str, docs_dir: str) -> PgVectorStore:
    """Shared store per database; policy documents are ingested once (idempotent)."""
    store = PgVectorStore(dsn, embedder_from_env())
    store.upsert_chunks(chunk_documents(Path(docs_dir)))
    return store


def with_history(task: str, history: list[dict], max_chars: int = 4000) -> str:
    """Fold earlier turns into the task so follow-up questions ("what about their invoices?") resolve."""
    if not history:
        return task
    lines, used = [], 0
    for m in reversed(history):
        line = f"{m['role']}: {m['content']}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "Conversation so far:\n" + "\n".join(reversed(lines)) + f"\n\nNew question: {task}"
