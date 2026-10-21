"""AWS Bedrock managed services used around the agent.

* BedrockEmbedder         Titan Text Embeddings v2 or Cohere Embed on Bedrock, for the pgvector store
                          (EMBEDDINGS=bedrock-titan | bedrock-cohere)
* KnowledgeBaseRetriever  Knowledge Bases for Amazon Bedrock as a managed RAG backend
                          (RETRIEVAL=bedrock-kb, BEDROCK_KB_ID)
* GuardrailChecker        Guardrails for Amazon Bedrock via ApplyGuardrail, to screen user input and final
                          answers independently of the model (BEDROCK_GUARDRAIL_ID / _VERSION). The same
                          guardrail is also attached to every Converse call in agent/bedrock.py.

Clients are injectable so everything is unit-tested without AWS (tests/test_bedrock_services.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


def _client(service: str, client: Any = None):
    if client is not None:
        return client
    import boto3

    return boto3.client(service, region_name=os.getenv("AWS_REGION", "us-east-1"))


class BedrockEmbedder:
    """Embeddings from Bedrock. Titan v2 supports 256/512/1024 dimensions with built-in normalisation;
    Cohere Embed v3 distinguishes documents from queries (input_type) and returns 1024 dimensions."""

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", dim: int = 512, client: Any = None):
        self.model_id, self.client = model_id, _client("bedrock-runtime", client)
        self.is_cohere = model_id.startswith("cohere.")
        self.dim = 1024 if self.is_cohere else dim

    def _invoke(self, body: dict) -> dict:
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body),
                                        contentType="application/json", accept="application/json")
        raw = resp["body"].read() if hasattr(resp["body"], "read") else resp["body"]
        return json.loads(raw)

    def embed(self, texts: list[str], input_type: str = "search_document") -> np.ndarray:
        if self.is_cohere:
            out = []
            for i in range(0, len(texts), 96):  # Cohere accepts up to 96 texts per call
                out += self._invoke({"texts": texts[i:i + 96], "input_type": input_type, "truncate": "END"})["embeddings"]
            vecs = np.asarray(out, np.float32)
            return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        return np.asarray([self._invoke({"inputText": t, "dimensions": self.dim, "normalize": True})["embedding"]
                           for t in texts], np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], input_type="search_query")[0]


class KnowledgeBaseRetriever:
    """Managed RAG: Bedrock handles chunking, embedding and the vector index; we only retrieve."""

    def __init__(self, knowledge_base_id: str | None = None, k: int = 3, client: Any = None):
        self.kb_id = knowledge_base_id or os.environ["BEDROCK_KB_ID"]
        self.k, self.client = k, _client("bedrock-agent-runtime", client)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        resp = self.client.retrieve(knowledgeBaseId=self.kb_id, retrievalQuery={"text": query},
                                    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k or self.k}})
        hits = []
        for r in resp.get("retrievalResults", []):
            loc = r.get("location", {})
            uri = (loc.get("s3Location") or {}).get("uri") or (loc.get("webLocation") or {}).get("url") or "kb"
            hits.append({"source": uri.rsplit("/", 1)[-1], "text": r["content"]["text"],
                         "score": round(float(r.get("score", 0.0)), 3)})
        return hits


@dataclass
class GuardrailResult:
    blocked: bool
    text: str              # original text, or the guardrail's replacement / masked text
    reasons: list[str]


class GuardrailChecker:
    def __init__(self, guardrail_id: str | None = None, version: str | None = None, client: Any = None):
        self.guardrail_id = guardrail_id or os.environ["BEDROCK_GUARDRAIL_ID"]
        self.version = version or os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
        self.client = _client("bedrock-runtime", client)

    def check(self, text: str, source: str = "INPUT") -> GuardrailResult:
        resp = self.client.apply_guardrail(guardrailIdentifier=self.guardrail_id, guardrailVersion=self.version,
                                           source=source, content=[{"text": {"text": text}}])
        if resp.get("action") != "GUARDRAIL_INTERVENED":
            return GuardrailResult(False, text, [])
        reasons = []
        for a in resp.get("assessments", []):
            for t in a.get("topicPolicy", {}).get("topics", []):
                reasons.append(f"denied topic: {t.get('name')}")
            for f in a.get("contentPolicy", {}).get("filters", []):
                reasons.append(f"content filter: {f.get('type')}")
            for e in a.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
                reasons.append(f"PII {e.get('action', '').lower()}: {e.get('type')}")
        out = resp.get("outputs") or [{"text": text}]
        masked_only = reasons and all(r.startswith("PII anonymized") for r in reasons)
        return GuardrailResult(not masked_only, out[0].get("text", text), reasons)


def guardrail_from_env() -> GuardrailChecker | None:
    return GuardrailChecker() if os.getenv("BEDROCK_GUARDRAIL_ID") else None
