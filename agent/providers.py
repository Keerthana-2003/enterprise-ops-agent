"""LLM provider selection for a cloud-agnostic deployment.

LLM_PROVIDER   model / deployment setting           credentials
openai         AGENT_MODEL (default gpt-4o-mini)      OPENAI_API_KEY (or OPENAI_BASE_URL for vLLM / Ollama)
azure          AZURE_OPENAI_DEPLOYMENT                AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
vertex         VERTEX_MODEL (e.g. google/gemini-2.0-flash-001)  Google ADC; GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION
bedrock        BEDROCK_MODEL_ID                       AWS credentials chain; AWS_REGION

Azure OpenAI and Vertex AI both speak the OpenAI chat-completions protocol (Vertex through its
OpenAI-compatible endpoint), so they reuse the same client, retries and tool calling; Bedrock uses
the Converse API (agent/bedrock.py). Switching cloud is a configuration change, not a code change.
"""
from __future__ import annotations

import os
import time

from .llm import AsyncOpenAIChatClient, LLMError

PROVIDERS = ("openai", "azure", "vertex", "bedrock")


def azure_openai_client(model: str | None = None, sdk=None) -> AsyncOpenAIChatClient:
    endpoint, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise LLMError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for LLM_PROVIDER=azure.")
    if sdk is None:
        import openai as sdk
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    client = sdk.AsyncAzureOpenAI(azure_endpoint=endpoint, api_key=key,
                                  api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                  timeout=30.0, max_retries=0)
    return AsyncOpenAIChatClient(model=deployment, client=client)


class VertexAIClient(AsyncOpenAIChatClient):
    """Gemini (and other Model Garden models) on Vertex AI via the OpenAI-compatible endpoint.
    Google access tokens expire after about an hour, so the token is refreshed before it lapses."""

    def __init__(self, model: str | None = None, credentials=None, sdk=None):
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise LLMError("Set GOOGLE_CLOUD_PROJECT (and Application Default Credentials) for LLM_PROVIDER=vertex.")
        if credentials is None:
            import google.auth

            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.credentials = credentials
        self._expires_at = 0.0
        if sdk is None:
            import openai as sdk
        base_url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
                    f"/locations/{location}/endpoints/openapi")
        client = sdk.AsyncOpenAI(base_url=base_url, api_key=self._token(), timeout=30.0, max_retries=0)
        super().__init__(model=model or os.getenv("VERTEX_MODEL", "google/gemini-2.0-flash-001"), client=client)

    def _token(self) -> str:
        if time.time() > self._expires_at - 300 or not getattr(self.credentials, "token", None):
            from google.auth.transport.requests import Request

            self.credentials.refresh(Request())
            expiry = getattr(self.credentials, "expiry", None)
            self._expires_at = expiry.timestamp() if expiry else time.time() + 3000
        return self.credentials.token

    async def chat(self, messages, tools, json_mode: bool = False):
        self.client.api_key = self._token()
        return await super().chat(messages, tools, json_mode)


def make_llm(model: str | None = None):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        return AsyncOpenAIChatClient(model=model)
    if provider == "azure":
        return azure_openai_client(model)
    if provider == "vertex":
        return VertexAIClient(model)
    if provider == "bedrock":
        from .bedrock import BedrockConverseClient

        return BedrockConverseClient(model=model)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of {PROVIDERS}.")
