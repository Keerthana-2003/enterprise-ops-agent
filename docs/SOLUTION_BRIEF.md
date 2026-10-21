# Solution brief: AI Operations Agent

A client-facing proposal template for packaging this repository as a solution offering. Client
details are illustrative; numbers marked *assumption* are inputs to be replaced in a real proposal.

## 1. Client situation

A B2B software company's operations team answers questions such as "Can we give this customer a
credit?" or "Why is this account escalated?" by reading three systems by hand: CRM, billing and
support. The systems use different customer keys, so one answer takes 15 to 30 minutes, and
policies (escalation, credits, collections) are applied inconsistently.

**Goals:** cut time-to-answer, apply policy consistently, keep a human in control of every action,
and keep customer data inside the client's chosen cloud.

## 2. Proposed solution

A tool-calling LLM agent that resolves the customer across systems, gathers the facts, checks
policy and returns a cited, structured answer with a proposed action for human approval.

| Capability | Asset in this repo |
|---|---|
| Cross-system entity resolution | `agent/ontology.py` |
| Agent runtime (LangGraph or custom loop) | `agent/graph.py`, `agent/core.py` |
| Structured, validated answers | `agent/schemas.py` |
| Evaluation and acceptance gates | `evals/` (task success, LLM-as-a-judge rubric, latency, cost) |
| Production deployment | `k8s/`, `deploy/rollout.py`, `infra/`, `infra-cdk/` |
| Observability and incident response | `api/metrics.py`, `k8s/monitoring/`, `INCIDENT_RESPONSE.md` |

## 3. Architecture options (cloud-agnostic)

Only configuration changes between clouds (`LLM_PROVIDER`, connection settings).

| Layer | AWS | Azure | Google Cloud | On-premises |
|---|---|---|---|---|
| LLM | Bedrock | Azure OpenAI | Vertex AI | Llama 3 on vLLM / Ollama |
| Runtime | EKS or App Runner | AKS | GKE | Kubernetes / OpenShift |
| Secrets | Secrets Manager | Key Vault | Secret Manager | Vault / K8s Secrets |
| Identity | Cognito | Entra ID | Identity Platform | Keycloak |
| Observability | OTel to Datadog or CloudWatch | OTel to Azure Monitor | OTel to Cloud Trace | OTel to Prometheus / Grafana |

Recommendation: deploy in the cloud where the client's data already lives, which avoids moving
customer data and simplifies security review.

## 4. Implementation roadmap

| Phase | Duration (*assumption*) | Scope | Exit criteria |
|---|---|---|---|
| 0. Discovery | 2 weeks | Workshops, data access, top 20 questions, policy inventory, success metrics | Signed-off use cases and an evaluation set of 50+ real questions |
| 1. Prototype | 3 weeks | Connect read-only to the 3 systems, entity resolution, first agent, eval harness | At least 70% task success on the evaluation set, no unsafe actions |
| 2. Pilot | 4 weeks | 10 analysts, human approval flow, tracing, cost controls, feedback loop | At least 85% task success, judge score of 4 or more out of 5, p95 under 20 s, analysts' time-to-answer down by half |
| 3. Production | 4 weeks | Kubernetes, canary releases, alerts, runbook, SSO, security review | Security sign-off, SLOs met for 2 weeks, on-call handover |
| 4. Scale | ongoing | New tools and systems, more teams, provider and cost optimisation | Quarterly value review |

## 5. Team structure

| Role | Allocation (*assumption*) | Responsibilities |
|---|---|---|
| Solution architect | 50% | Architecture, client advisory, roadmap, risk |
| ML / LLM engineers | 2 x 100% | Agent, tools, evaluation, prompt and model choices |
| Data engineer | 50% (phases 0-2) | System connectors, entity resolution, data quality |
| DevOps / platform engineer | 50% (phases 2-3) | Kubernetes, CI/CD, observability, security controls |
| Engagement manager | 25% | Plan, status, stakeholders, change control |
| Client SMEs | 2-4 hours a week | Policies, evaluation set, acceptance testing |

## 6. Pricing and run cost

**Delivery fee:** sum of role allocations x weeks x blended rate per phase, with phases 0-1 as a
fixed fee and phases 2-3 as time and materials capped at an agreed amount. Rates are set by the
practice and are not included here.

**Run cost:** `deploy/cost_estimate.py` computes monthly LLM, infrastructure and observability cost
from volume and measured tokens per run (from the eval harness). With the illustrative defaults
(2,000 runs a day, 6,000 prompt and 400 completion tokens per run, 50% prompt-cache hits,
placeholder prices) it reports about $400 a month, or under a cent per answer. Replace the prices
with the client's contracted rates before quoting.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong or invented answers | Answers must cite tool results; LLM-as-a-judge grounding score gates every release |
| Unsafe actions | Read-only SQL; actions are proposals that need human approval |
| Data leaving the client's boundary | Provider in the client's cloud or self-hosted Llama 3; no training on client data |
| Cost overruns | Token budgets per run, prompt caching, cost dashboard and alerts |
| Provider outage | Provider switch by configuration; retries and fallbacks; incident runbook |
| Low adoption | Pilot with real analysts, weekly feedback, measure time-to-answer |

## 8. Sales enablement one-pager

**Elevator pitch:** "Your operations team spends 15 to 30 minutes per question piecing together
CRM, billing and support data. We deploy an AI agent in your own cloud that answers in seconds,
cites its sources, follows your policies and never acts without approval. A 2-week discovery and
3-week prototype show measured results on your own questions before you commit to production."

**Qualifying questions:** How many cross-system questions does the team handle each week? Which
systems hold the data, and do they have APIs or read replicas? Which cloud and LLM providers are
approved? Who signs off on policy and security?

**Good fit:** repeated questions across 2 or more systems, clear written policies, an approved
cloud. **Poor fit:** no programmatic data access, or a need for fully autonomous actions.

**Common objections:** "Will it make things up?" Every answer cites the records it used, and
releases are gated on a grounding score. "Is our data safe?" It runs in your cloud with read-only
access. "What does it cost to run?" Typically cents per answer; we model it with your volumes.
