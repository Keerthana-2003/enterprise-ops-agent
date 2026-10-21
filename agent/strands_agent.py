"""The same multi-agent team built with Strands Agents (AWS's open-source agent SDK).

Each enterprise tool becomes a Strands tool with the agent's own JSON schema and guards; each
specialist is a Strands Agent limited to its tools; the supervisor uses the specialists as tools
("agents as tools"). Models: Amazon Bedrock by default (BedrockModel), or any model a Strands
provider supports. Run:  python -m agent.strands_agent "Fabrikam has unpaid invoices..."
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .multi_agent import SPECIALIST_PROMPT, SPECIALISTS, SUPERVISOR_PROMPT
from .tools import ToolError, ToolRegistry, build_tools


def invoke_registry_tool(registry: ToolRegistry, name: str, tool_use: dict) -> dict:
    """Run one tool call and return a Strands ToolResult; tool errors go back to the model."""
    try:
        out = registry.call(name, json.dumps(tool_use.get("input") or {}))
        return {"toolUseId": tool_use["toolUseId"], "status": "success",
                "content": [{"text": json.dumps(out, default=str)}]}
    except ToolError as err:
        return {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": str(err)}]}


def strands_tools(registry: ToolRegistry, names: set[str] | None = None) -> list:
    from strands.tools.tools import PythonAgentTool

    out = []
    for name, t in registry.tools.items():
        if names is not None and name not in names:
            continue
        spec = {"name": name, "description": t.description, "inputSchema": {"json": t.parameters}}
        out.append(PythonAgentTool(name, spec, lambda tool_use, _n=name, **_: invoke_registry_tool(registry, _n, tool_use)))
    return out


def default_model():
    from strands.models import BedrockModel

    return BedrockModel(model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
                        region_name=os.getenv("AWS_REGION", "us-east-1"))


def build_team(data_dir: Path, model=None):
    """Return (supervisor, specialists, proposed_actions)."""
    from strands import Agent, tool

    model = model or default_model()
    registry, ontology, proposed = build_tools(Path(data_dir))
    schema = ontology.schema_summary()
    specialists = {}
    for name, (allowed, description) in SPECIALISTS.items():
        specialists[name] = Agent(model=model, name=name, tools=strands_tools(registry, allowed),
                                  system_prompt=SPECIALIST_PROMPT.format(name=name, description=description)
                                  .format(ontology=schema), callback_handler=None)

    def as_tool(name: str, description: str):
        @tool(name=f"ask_{name}", description=f"Delegate a sub-question to the {name}: {description}")
        def ask(question: str) -> str:
            """Ask the specialist a self-contained question.

            Args:
                question: the sub-question for this specialist
            """
            return str(specialists[name](question))
        return ask

    supervisor_tools = [as_tool(n, d) for n, (_, d) in SPECIALISTS.items()] + strands_tools(registry, {"propose_action"})
    supervisor = Agent(model=model, name="supervisor", tools=supervisor_tools,
                       system_prompt=SUPERVISOR_PROMPT.format(ontology=schema).replace(
                           "Finish by calling final_answer citing the specialists' facts.",
                           "Finish with a short answer citing the specialists' facts."))
    return supervisor, specialists, proposed


if __name__ == "__main__":
    import sys

    from .seed import DATA, seed

    if not (DATA / "crm.db").exists():
        seed(DATA)
    sup, _, actions = build_team(DATA)
    print(sup(" ".join(sys.argv[1:]) or "Fabrikam has open tickets and unpaid invoices. What should we do?"))
    print("Proposed actions:", json.dumps(actions, indent=2))
