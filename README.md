# Enterprise Ops Agent

A production-style, tool-calling AI agent that automates an enterprise operations workflow over
**fragmented data**. Customer data lives in three systems that each use a different key. The agent
resolves entities across them through an **ontology layer**, gathers context with **parallel tool
calls**, checks company policy, and returns a **structured, validated answer** with a proposed
action for human approval. Runs **stream over SSE**, can be delivered by **webhook**, are **traced in Langfuse and
OpenTelemetry (exportable to Datadog)**, scored by an **LLM-as-a-judge eval harness**, and deploy to
**AWS (App Runner, CloudFront, Cognito, Bedrock) with Terraform or AWS CDK**. The agent runs on two
interchangeable runtimes, a hand-written async loop and a **LangGraph** state machine, and three
providers: **OpenAI**, local **Llama 3**, and **AWS Bedrock**.

```
                    ┌────────────── FastAPI (async) ──────────────┐
 client ── POST ──► │ /runs   /runs/stream (SSE)   /runs/async (webhook) │
                    └───────────────────────┬─────────────────────┘
                                            ▼
┌────────────────────────── Async agent loop (agent/core.py) ──────────────────────────┐
│ LLM (OpenAI or local Llama 3) ── tool calls ──► asyncio.gather ──► Tool registry      │
│      ▲                                           (parallel)         Pydantic-validated│
│      └──────── results / actionable errors ◄─────────────────────────────┘            │
│ Ends only via final_answer (Pydantic structured output)                               │
│ Guards: retries+backoff · loop detection · tool-error, step and token budgets         │
│ Cost: stable prompt prefix for prompt caching, cached-token tracking                  │
│ Observability: Langfuse trace / generation per LLM call / span per tool               │
└───────────────────────────────────────────────────────────────────────────────────────┘
     │              │                 │                  │                │
resolve_entity  query_crm /     get_support_      search_policies   propose_action
(ontology)      query_billing   tickets           (local doc search) (human approval)
                (read-only SQL)
```

## Production concerns and how they're handled

| Concern | Implementation |
|---|---|
| Tool calling with validated inputs | Each tool's JSON Schema is generated from a Pydantic model (`agent/schemas.py`); invalid arguments come back to the model as precise, fixable errors. Unknown fields are rejected. |
| Structured output | The run ends only through a `final_answer` tool validated against the `FinalAnswer` Pydantic model (summary, facts, action, confidence, unverified items). Invalid answers are fed back; plain-text answers are nudged once. |
| Parallel API orchestration | Independent tool calls from one model turn run concurrently with `asyncio.gather` (tested: 3 × 0.3 s calls finish in under 0.75 s). |
| Streaming | `POST /runs/stream` emits typed Server-Sent Events: `run_started`, `llm_call`, `tool_started`, `tool_finished`, `tool_failed`, `guard`, `final`, `run_finished`. |
| Webhooks | `POST /runs/async` returns 202, then POSTs the result to `callback_url` with retries and an HMAC-SHA256 signature. |
| Reliability | Async client with timeouts and exponential backoff with jitter; loop detection; tool-error, step and token budgets; every run ends in a typed status. |
| Latency and cost | System prompt and tool definitions are a stable prefix so providers can serve them from the prompt cache; cached tokens and cache hit rate are reported per run and per eval. |
| Observability | Langfuse trace per run, a generation per LLM call (model, tokens, cached tokens) and a span per tool call (input, output, errors). Tracing failures never break a run. |
| Evaluation | Deterministic checks plus an LLM-as-a-judge rubric (correctness, grounding, policy compliance, safety, actionability) with Pydantic-validated scores; versioned dataset; concurrent runs. |
| Fragmented data, context | CRM keys customers as `3`, billing as `ACCT-5520`, support as `fabrikam.eu`; `resolve_entity` maps any mention to all three plus relationships. |
| Safety | SQL tools are read-only. Actions are proposals that need human approval. |
| Infrastructure as code | Terraform (`infra/`) and an equivalent AWS CDK Python app (`infra-cdk/`): ECR, App Runner with health checks, CloudFront, Cognito, Secrets Manager, least-privilege IAM including Bedrock access. |
| Agent frameworks | The same agent also runs as a LangGraph `StateGraph` (`agent/graph.py`) with the same tools, budgets and structured output, so both runtimes are compared on one eval harness (`--runtime langgraph`). |
| Model portability | OpenAI-compatible endpoints (OpenAI, Azure, Llama 3 via Ollama or vLLM) and AWS Bedrock through the Converse API (`agent/bedrock.py`), with Bedrock prompt-cache points. |
| OpenTelemetry and Datadog | Spans for each run, LLM call and tool call using the GenAI semantic conventions (`gen_ai.*`), exported over OTLP; `deploy/docker-compose.datadog.yml` sends them to Datadog APM. |
| Authentication | Amazon Cognito JWTs verified against the user pool JWKS (signature, expiry, issuer, token use, app client) on every `/runs` endpoint. |
| Edge | CloudFront in front of App Runner: HTTPS, caching disabled for API responses, all methods allowed, SSE passed through. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m agent.seed            # sample enterprise data in data/
pytest -q                       # offline tests, no API key needed
cp .env.example .env            # add OPENAI_API_KEY (and Langfuse keys if you want traces)
set -a && source .env && set +a
uvicorn api.main:app --reload   # http://localhost:8000/docs
```

Synchronous run:

```bash
curl -s localhost:8000/runs -H 'content-type: application/json' \
  -d '{"task": "Fabrikam has open support tickets and unpaid invoices. What should we do?"}' | python -m json.tool
```

Streaming run (watch events arrive):

```bash
curl -N localhost:8000/runs/stream -H 'content-type: application/json' \
  -d '{"task": "Who manages Tailspin Airlines, and how much do they owe on open invoices?"}'
```

Async run with webhook (use https://webhook.site for a test URL):

```bash
curl -s localhost:8000/runs/async -H 'content-type: application/json' \
  -d '{"task": "Can we give Northwind Traders a service credit?", "callback_url": "https://webhook.site/<your-id>"}'
```

Local open-source model:

```bash
ollama pull llama3.1
OPENAI_BASE_URL=http://localhost:11434/v1 AGENT_MODEL=llama3.1 uvicorn api.main:app
```

Run on AWS Bedrock, or with the LangGraph runtime:

```bash
LLM_PROVIDER=bedrock BEDROCK_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0 uvicorn api.main:app
AGENT_RUNTIME=langgraph uvicorn api.main:app
```

Send traces to Datadog:

```bash
DD_API_KEY=... OPENAI_API_KEY=... docker compose -f deploy/docker-compose.datadog.yml up --build
```

## Evaluation

```bash
python -m evals.run_evals                         # agent + judge on gpt-4o-mini
python -m evals.run_evals --judge-model gpt-4o    # stronger judge
OPENAI_BASE_URL=http://localhost:11434/v1 AGENT_MODEL=llama3.1 python -m evals.run_evals --no-judge
python -m evals.run_evals --runtime langgraph               # same eval, LangGraph runtime
LLM_PROVIDER=bedrock python -m evals.run_evals              # agent and judge on Bedrock
```

Reported per model: task success rate, structured-output rate, judge mean and per-dimension
scores, failure categories, tool error rate, runs using parallel tool calls, average and p95
latency, average tokens, and prompt cache hit rate. Results are saved to
`evals/results/<model>.json` with the dataset version hash.


## Deploy to AWS (Terraform)

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars        # set openai_api_key
terraform init
terraform apply -target=aws_ecr_repository.agent    # 1. create the registry

REPO=$(terraform output -raw ecr_repository_url)    # 2. push the image
aws ecr get-login-password | docker login --username AWS --password-stdin ${REPO%/*}
docker build --platform linux/amd64 -t $REPO:latest .. && docker push $REPO:latest

terraform apply                                     # 3. App Runner, CloudFront, Cognito, secret, IAM
curl $(terraform output -raw cloudfront_url)/health
```

Create a user and call the API with a Cognito token:

```bash
POOL=$(terraform output -raw cognito_user_pool_id); CLIENT=$(terraform output -raw cognito_client_id)
aws cognito-idp admin-create-user --user-pool-id $POOL --username you@example.com --message-action SUPPRESS
aws cognito-idp admin-set-user-password --user-pool-id $POOL --username you@example.com --password 'Str0ng!Passw0rd' --permanent
TOKEN=$(aws cognito-idp initiate-auth --client-id $CLIENT --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=you@example.com,PASSWORD='Str0ng!Passw0rd' --query AuthenticationResult.AccessToken --output text)
curl -s $(terraform output -raw cloudfront_url)/runs -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"task": "Is Wide World Importers in good standing?"}'
```

The same stack in AWS CDK (Python):

```bash
cd infra-cdk && pip install -r requirements.txt
npx aws-cdk@2 bootstrap && npx aws-cdk@2 deploy
```

`terraform destroy` removes everything. Note the OpenAI key is stored in Terraform state; use a
remote encrypted backend for anything beyond a demo.

## Kubernetes and progressive delivery

```bash
kubectl apply -k k8s/base                                  # Deployment (4 replicas), Service, HPA, PDB
kubectl -n ops-agent create secret generic agent-secrets --from-literal=openai-api-key=$OPENAI_API_KEY
python deploy/rollout.py canary --image <registry>/ops-agent:v2      # one-box -> 25% -> 50% -> 100%
python deploy/rollout.py bluegreen --image <registry>/ops-agent:v2   # deploy idle colour, smoke test, switch
kubectl apply -f k8s/monitoring/                            # ServiceMonitor + alert rules (Prometheus Operator)
```

- **Canary with gradual dial-up:** a single canary pod (one-box) joins the stable pods behind the
  same Service, then its share grows step by step. After each bake period the canary's error rate
  and p95 latency (read from each pod's `/metrics`) are compared with the stable baseline, and the
  rollout promotes or rolls back automatically (`deploy/rollout.py`, unit-tested decision logic).
- **Blue/green:** the idle colour is deployed and smoke-tested before the Service selector switches;
  the old colour stays up for an instant rollback.
- **Resilience:** readiness and liveness probes, resource requests and limits, HPA autoscaling,
  PodDisruptionBudget, zero-downtime rolling updates, non-root containers.
- **Alarms and incident response:** Prometheus metrics (`/metrics`), alert rules for error rate,
  p95 latency, LLM provider errors, unavailable pods and crash loops, each linked to a section of
  [`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md).
- **Tested in CI:** every push creates a kind cluster, deploys the stack, runs a full canary dial-up
  and promotion and a blue/green switch, and validates the alert rules with `promtool`.

## PostgreSQL + pgvector (retrieval and conversation memory)

Set `DATABASE_URL` to a PostgreSQL database with the `vector` extension and the agent switches
policy search from in-memory TF-IDF to pgvector cosine search (HNSW index), and `POST /runs` with a
`conversation_id` keeps multi-turn memory, so follow-ups like "what about their invoices?" resolve
against earlier turns (`GET /conversations/{id}` returns the history). Schema and migrations live in
`agent/memory.py`; embeddings come from OpenAI (`EMBEDDINGS=openai`) or an offline hashing embedder.
CI runs these tests against a real `pgvector/pgvector` Postgres container.

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

## Multi-cloud LLM providers

`LLM_PROVIDER` selects the model platform without code changes: `openai` (or any OpenAI-compatible
server such as vLLM or Ollama), `azure` (Azure OpenAI), `vertex` (Gemini on Google Cloud Vertex AI
through its OpenAI-compatible endpoint, with automatic token refresh) or `bedrock` (AWS Bedrock
Converse API). See `agent/providers.py`.

## Packaging as a client solution

[`docs/SOLUTION_BRIEF.md`](docs/SOLUTION_BRIEF.md) turns this repository into a proposal: client
goals, cloud-by-cloud architecture options, a phased implementation roadmap with exit criteria,
team structure, pricing approach, risks and a sales enablement one-pager. `deploy/cost_estimate.py`
models monthly run cost from volume and the token counts measured by the eval harness.

## Project layout

```
agent/
  core.py       async agent loop, structured output, parallel calls, budgets, streaming events, OTel spans
  graph.py      the same agent as a LangGraph state machine
  bedrock.py    AWS Bedrock Converse provider and provider selection
  telemetry.py  OpenTelemetry setup with OTLP export (Datadog Agent or any collector)
  llm.py        async OpenAI-compatible client with retries and cached-token reporting; ScriptedLLM for tests
  schemas.py    Pydantic models: tool arguments, FinalAnswer, stream events
  tools.py      tool registry and enterprise tools
  tracing.py    Langfuse tracing with a safe no-op fallback
  ontology.py   entity resolution and relationships across systems
  seed.py       sample fragmented enterprise data and policies
api/main.py     FastAPI: /health, /runs, /runs/stream (SSE), /runs/async (webhook), /runs/{id}
api/auth.py     Cognito JWT verification
evals/          versioned task set, deterministic grader, LLM-as-a-judge, harness
infra/          Terraform: ECR, App Runner, CloudFront, Cognito, Secrets Manager, IAM
infra-cdk/      the same stack in AWS CDK (Python)
k8s/            Kubernetes manifests: base, canary, blue/green, monitoring and alerts
deploy/         rollout.py: canary dial-up and blue/green with automatic rollback
deploy/         Docker Compose with the Datadog Agent for OpenTelemetry traces
tests/          offline tests for the loop, tools, evals and API
```

CI (GitHub Actions) runs lint, the full test suite (agent loop, LangGraph runtime, Bedrock
provider, Cognito auth, OpenTelemetry spans, API), `terraform validate`, `cdk synth`, a Docker
build and a `/health` smoke test on every push.
