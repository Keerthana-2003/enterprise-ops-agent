# ADR-0003: Route requests between a small and a large model

**Status:** accepted

**Context.** Most requests are simple lookups; a minority need multi-step reasoning or touch money
and customer accounts. Sending everything to a frontier model wastes cost and adds latency.

**Decision.** Score each request for complexity and risk without an extra model call
(`agent/routing.py`). Low-score requests go to a small model; high-risk or multi-system requests go
straight to the large model. A small-model run that fails validation (no valid structured answer,
low confidence, tool errors, or any proposed action) is escalated to the large model.

**Consequences.** Lower average cost and latency, with quality protected by validation-based
escalation. Escalations cost both calls, so the threshold is tuned with the eval harness
(`--router`), which reports the share answered by each tier and the escalation rate. Actions are
reserved for the large tier by design.
