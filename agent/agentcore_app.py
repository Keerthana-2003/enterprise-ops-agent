"""Amazon Bedrock AgentCore Runtime entrypoint.

AgentCore Runtime hosts the agent as a serverless, session-isolated container (POST /invocations,
GET /ping on port 8080, arm64). Build with Dockerfile.agentcore, push to ECR and create the runtime
(`agentcore configure` / `agentcore launch`, or the AWS::BedrockAgentCore::Runtime resource).

Payload: {"prompt": "...", "mode": "single" | "multi"}. "multi" runs the supervisor and specialist
agents. Responses include the structured answer, proposed actions (for human approval), the
specialist hand-offs, token use and the run id for tracing.
"""
from __future__ import annotations

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .core import Agent
from .multi_agent import Supervisor, make_specialists, SPECIALISTS
from .seed import DATA, seed

app = BedrockAgentCoreApp()
_llm_factory = None  # tests inject a factory; production uses LLM_PROVIDER (bedrock on AgentCore)


def _llm(role: str):
    if _llm_factory:
        return _llm_factory(role)
    from .providers import make_llm

    return make_llm(os.getenv(f"MODEL_{role.upper()}") or None)


def _data_dir():
    if not (DATA / "crm.db").exists():
        seed(DATA)
    return DATA


@app.entrypoint
async def invoke(payload: dict) -> dict:
    prompt = (payload or {}).get("prompt", "").strip()
    if not prompt:
        return {"error": "payload needs a non-empty 'prompt'"}
    data = _data_dir()
    if payload.get("mode", "single") == "multi":
        specialists = make_specialists({n: _llm(n) for n in SPECIALISTS}, data)
        res = await Supervisor(_llm("supervisor"), specialists, data).arun(prompt)
        return res.to_dict()
    run = await Agent(_llm("agent"), data).arun(prompt)
    return run.to_dict()


if __name__ == "__main__":
    app.run()
