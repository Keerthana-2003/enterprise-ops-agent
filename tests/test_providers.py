"""Provider selection for a cloud-agnostic deployment (no network: SDKs are replaced by fakes)."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from agent.llm import AsyncOpenAIChatClient, LLMError
from agent.providers import VertexAIClient, azure_openai_client, make_llm


class FakeSDK:
    def __init__(self):
        self.calls = []

    def AsyncAzureOpenAI(self, **kw):  # noqa: N802 - mirrors the SDK class name
        self.calls.append(("azure", kw))
        return SimpleNamespace(kind="azure", **kw)

    def AsyncOpenAI(self, **kw):  # noqa: N802
        self.calls.append(("openai", kw))
        return SimpleNamespace(kind="openai", **kw)


def test_azure_uses_endpoint_deployment_and_api_version(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-res.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt4o-prod")
    sdk = FakeSDK()
    llm = azure_openai_client(sdk=sdk)
    kind, kw = sdk.calls[0]
    assert kind == "azure" and kw["azure_endpoint"].endswith("azure.com") and kw["api_version"]
    assert isinstance(llm, AsyncOpenAIChatClient) and llm.model == "gpt4o-prod"


def test_azure_requires_configuration(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(LLMError, match="AZURE_OPENAI_ENDPOINT"):
        azure_openai_client(sdk=FakeSDK())


class FakeCreds:
    def __init__(self):
        self.token, self.expiry, self.refreshes = None, None, 0

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"tok{self.refreshes}"
        self.expiry = dt.datetime.now() + dt.timedelta(hours=1)


def test_vertex_builds_openai_compatible_endpoint_and_refreshes_token(monkeypatch):
    pytest.importorskip("google.auth")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "acme-ml")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")
    sdk, creds = FakeSDK(), FakeCreds()
    llm = VertexAIClient(credentials=creds, sdk=sdk)
    kind, kw = sdk.calls[0]
    assert kw["base_url"] == ("https://us-east5-aiplatform.googleapis.com/v1/projects/acme-ml/"
                              "locations/us-east5/endpoints/openapi")
    assert kw["api_key"] == "tok1" and llm.model.startswith("google/")
    assert llm._token() == "tok1" and creds.refreshes == 1  # cached while valid
    llm._expires_at = 0
    assert llm._token() == "tok2"  # refreshed once near expiry


def test_make_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "watson")
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        make_llm()
