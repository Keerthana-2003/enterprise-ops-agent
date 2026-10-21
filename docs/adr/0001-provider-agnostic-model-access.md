# ADR-0001: Provider-agnostic model access through a gateway

**Status:** accepted

**Context.** Clients standardise on different clouds (Azure, AWS, Google Cloud) and some require
self-hosted models for data residency. Model prices and capabilities change every few months.

**Decision.** Applications call models only through one interface (`agent/providers.py`) and, for
shared use, through the model gateway (`gateway/`), which exposes an OpenAI-compatible API, maps
aliases such as `fast` and `smart` to provider models, enforces per-team keys, rate limits and
token budgets, falls back to a second provider on failure, and records usage and cost per team.

**Consequences.** Switching provider is configuration, not code. Central place for quotas,
chargeback and audit. Provider-specific features (for example Bedrock Guardrails) need an adapter;
the gateway adds a network hop (single-digit milliseconds, negligible against model latency).
