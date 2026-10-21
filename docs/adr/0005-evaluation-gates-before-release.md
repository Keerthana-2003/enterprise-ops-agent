# ADR-0005: Evaluation gates before every model, prompt or provider change

**Status:** accepted

**Context.** Changing a prompt, model or provider can silently change answer quality.

**Decision.** Every change runs the versioned evaluation set (`evals/`): deterministic checks,
an LLM-as-a-judge rubric (correctness, grounding, policy compliance, safety, actionability),
latency and cost. Releases go out as canaries (`deploy/rollout.py`) and are rolled back
automatically on error-rate or latency regressions.

**Consequences.** Quality, cost and latency trade-offs are visible before release. The evaluation
set must be maintained with the business as new question types appear.
