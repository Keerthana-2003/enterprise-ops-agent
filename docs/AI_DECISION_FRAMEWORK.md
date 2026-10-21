# AI decision framework: when to use AI, which kind, and when not to

Use this in solution design sessions and architecture reviews before any model is chosen. Work
top to bottom and stop at the first option that meets the requirement: each step down adds cost,
latency, variability and governance work.

## 1. Decision ladder

| Step | Use when | Examples | Avoid when |
|---|---|---|---|
| **1. Rules, SQL, workflow automation** | Logic is known, stable and can be written down; exact answers are required; full auditability needed | Invoice > 30 days overdue → collections notice; routing by product line | Inputs are free text or the rule set keeps growing exceptions |
| **2. Search / reporting** | Users need to find or see existing information, not interpret it | Dashboards, keyword or faceted search over policies | Questions need synthesis across sources |
| **3. Traditional ML** | A measurable prediction from structured data with labelled history; consistent, cheap, explainable scoring at volume | Fraud scoring, churn, demand forecasting, anomaly detection | Little labelled data; output must be language |
| **4. Small language model (SLM)** | Language task with narrow scope: classification, extraction, short summaries, simple lookups; strict cost, latency or data-residency limits | Ticket triage, entity extraction, "who manages account X?" | Multi-step reasoning, high-stakes judgement |
| **5. LLM + RAG** | Answers must come from enterprise knowledge that changes; citations required | Policy Q&A, support answers from documentation | The knowledge fits in a rule or the answer must be exact arithmetic |
| **6. Agent (LLM + tools)** | The task needs several steps across systems and the path depends on what is found | Cross-system investigation, assembling a case file | A fixed workflow would do; actions are irreversible without human approval |
| **7. Fine-tuning** | Consistent format, tone or domain language that prompting and RAG cannot reach, with enough examples | Domain-specific extraction schema, house style | Knowledge changes often (use RAG instead) |

## 2. Scoring a use case

Score 1 (low) to 5 (high). High **value** and **tolerance for variability** favour AI; high
**risk** pushes toward human review, guardrails or a lower step on the ladder.

| Factor | Question |
|---|---|
| Business value | Hours saved, revenue or loss affected, customer impact? |
| Input structure | Structured fields (lower steps) or free text and documents (higher steps)? |
| Need for exactness | Must the answer be exactly right every time (rules), or is "usually right, reviewed" acceptable? |
| Knowledge volatility | Does the knowledge change weekly (RAG) or rarely (rules or fine-tuning)? |
| Risk of a wrong answer | Financial, legal, safety or reputational exposure? Decides human-in-the-loop and guardrails |
| Data sensitivity | PII or regulated data? Decides provider, region, self-hosting and masking |
| Volume and latency | Requests per day and acceptable response time? Decides model size and routing |
| Evaluation readiness | Can we build a labelled evaluation set? No evaluation set means no production AI |

## 3. Non-functional gates for any AI step

1. An evaluation set and acceptance thresholds agreed with the business before build.
2. Guardrails sized to risk: read-only tools, human approval for actions, content and PII filters.
3. Cost model per request and a monthly budget, with routing to the cheapest adequate model.
4. Observability: traces of every model and tool call, quality and cost dashboards, alerts.
5. A fallback path when the model or provider is unavailable (rules, queue for a human, second provider).

## 4. How this repository applies the framework

| Need | Ladder step chosen | Where |
|---|---|---|
| Detect overdue invoices, credit holds | Rules / SQL | billing queries and policies used as tools |
| Score transactions for fraud (sister project) | Traditional ML | fraud-ml-platform |
| Simple lookups ("who manages X?") | SLM via the router | `agent/routing.py` |
| Policy questions with citations | LLM + RAG | `search_policies` over pgvector or Bedrock Knowledge Bases |
| Cross-system "what should we do?" | Agent with human-approved actions | `agent/core.py`, `agent/graph.py` |
| Fine-tuning | Not used: policies change, so RAG is preferred | ADR-0002 |
