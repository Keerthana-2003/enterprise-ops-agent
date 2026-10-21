"""Prometheus metrics for the API: request rate, errors and latency per route, agent run outcomes,
latency and token usage. Scraped by Prometheus (k8s/monitoring) and read by deploy/rollout.py to
decide whether a canary is healthy enough to promote."""
from __future__ import annotations

import os

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Info, generate_latest

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["route"],
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
RUNS = Counter("agent_runs_total", "Agent runs by final status", ["status"])
RUN_LATENCY = Histogram("agent_run_duration_seconds", "Agent run latency",
                        buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120))
TOKENS = Counter("agent_tokens_total", "LLM tokens", ["kind"])
BUILD = Info("agent_build", "Build information")
BUILD.info({"version": os.getenv("APP_VERSION", "dev"), "track": os.getenv("DEPLOY_TRACK", "stable")})


def record_run(result: dict) -> None:
    RUNS.labels(status=result.get("status", "unknown")).inc()
    RUN_LATENCY.observe(result.get("latency_ms", 0) / 1000)
    TOKENS.labels(kind="prompt").inc(result.get("prompt_tokens", 0))
    TOKENS.labels(kind="completion").inc(result.get("completion_tokens", 0))
    TOKENS.labels(kind="cached").inc(result.get("cached_tokens", 0))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
