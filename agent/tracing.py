"""Observability: Langfuse tracing with a no-op fallback.

Every run becomes a Langfuse trace with one generation per LLM call (model, tokens, cached
tokens, latency) and one span per tool call (arguments, result or error). Tracing is enabled
when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set and the `langfuse` package is installed.
Any tracing failure is swallowed: observability must never break a user-facing run.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **_: Any) -> None: ...
    def end(self, **_: Any) -> None: ...
    def start_span(self, **_: Any) -> "_NoopObservation": return self
    def start_generation(self, **_: Any) -> "_NoopObservation": return self


class _Safe:
    """Wraps a Langfuse observation so any SDK error is logged, not raised."""

    def __init__(self, obs: Any):
        self._obs = obs

    def _call(self, method: str, **kw: Any) -> Any:
        try:
            return getattr(self._obs, method)(**kw)
        except Exception as err:  # pragma: no cover - depends on SDK/network
            log.warning("tracing %s failed: %s", method, err)
            return None

    def update(self, **kw: Any) -> None: self._call("update", **kw)
    def end(self, **kw: Any) -> None: self._call("end", **kw)

    def start_span(self, **kw: Any) -> "_Safe | _NoopObservation":
        child = self._call("start_span", **kw)
        return _Safe(child) if child is not None else _NoopObservation()

    def start_generation(self, **kw: Any) -> "_Safe | _NoopObservation":
        child = self._call("start_generation", **kw)
        return _Safe(child) if child is not None else _NoopObservation()


class Tracer:
    def __init__(self) -> None:
        self._client = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client

                self._client = get_client()
            except Exception as err:  # package missing or misconfigured
                log.warning("Langfuse disabled: %s", err)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_run(self, name: str, input: Any, metadata: dict | None = None):
        if not self._client:
            return _NoopObservation()
        try:
            return _Safe(self._client.start_span(name=name, input=input, metadata=metadata or {}))
        except Exception as err:  # pragma: no cover
            log.warning("tracing start_run failed: %s", err)
            return _NoopObservation()

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception as err:  # pragma: no cover
                log.warning("tracing flush failed: %s", err)
