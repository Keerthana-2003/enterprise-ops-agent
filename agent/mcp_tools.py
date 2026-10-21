"""Model Context Protocol (MCP): expose the enterprise tools as an MCP server, and use MCP tools in the agent.

Server
  `python -m agent.mcp_tools` serves resolve_entity, query_crm, query_billing, get_support_tickets,
  search_policies and propose_action over stdio, with the same JSON schemas, validation and
  read-only guards the agent uses. Any MCP client (Claude Desktop, Strands, LangGraph, Bedrock
  AgentCore Gateway, another team's agent) can then use them. The data model is published as an
  MCP resource (ontology://schema).
Client
  MCPToolset connects to any MCP server and turns its tools into agent tools, so the agent can mix
  local tools with tools other teams publish over MCP.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .tools import Tool, ToolError, ToolRegistry, build_tools


# ---------------------------------------------------------------- server

def create_mcp_server(data_dir: Path, allowed: set[str] | None = None):
    import mcp.types as types
    from mcp.server.lowlevel import Server

    registry, ontology, proposed = build_tools(Path(data_dir))
    tools = {n: t for n, t in registry.tools.items() if allowed is None or n in allowed}
    server = Server("enterprise-ops-tools")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name=t.name, description=t.description, inputSchema=t.parameters) for t in tools.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name not in tools:
            raise ValueError(f"Unknown tool '{name}'")
        try:
            result = await asyncio.to_thread(registry.call, name, json.dumps(arguments or {}))
        except ToolError as err:
            raise ValueError(str(err)) from None
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [types.Resource(uri="ontology://schema", name="ontology", mimeType="text/plain",
                               description="Entities, systems and join keys the tools operate on")]

    @server.read_resource()
    async def read_resource(uri) -> str:
        if str(uri) != "ontology://schema":
            raise ValueError(f"Unknown resource {uri}")
        return ontology.schema_summary()

    server.proposed_actions = proposed  # inspected by tests / the approval UI
    return server


async def serve_stdio(data_dir: Path) -> None:
    from mcp.server.stdio import stdio_server

    server = create_mcp_server(data_dir)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


# ---------------------------------------------------------------- client

class _AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


@dataclass
class MCPTool(Tool):
    input_schema: dict = field(default_factory=dict)

    @property
    def parameters(self) -> dict:
        return self.input_schema

    def validate(self, args: dict) -> dict:
        missing = [k for k in self.input_schema.get("required", []) if k not in args]
        if missing:
            raise ToolError(f"Invalid arguments for {self.name}: missing {missing}.")
        return args  # the MCP server validates fully and returns errors the model can act on


class MCPToolset:
    """Wrap an initialized mcp.ClientSession. Tool calls run on the agent's worker threads and are
    scheduled onto the event loop that owns the session."""

    def __init__(self, session, loop: asyncio.AbstractEventLoop, prefix: str = ""):
        self.session, self.loop, self.prefix = session, loop, prefix
        self._tools: list[MCPTool] = []

    async def load(self) -> list[MCPTool]:
        listed = await self.session.list_tools()
        self._tools = [self._wrap(t) for t in listed.tools]
        return self._tools

    def _wrap(self, t) -> MCPTool:
        def call(**kwargs):
            fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(t.name, kwargs), self.loop)
            res = fut.result(timeout=60)
            text = "\n".join(getattr(c, "text", "") for c in res.content)
            if res.isError:
                raise ToolError(text or f"{t.name} failed")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return MCPTool(self.prefix + t.name, t.description or "", _AnyArgs, call, input_schema=t.inputSchema)

    def registry_factory(self, include_local: set[str] = frozenset()):
        """A tools_factory for Agent: the MCP tools plus any chosen local tools."""
        def factory(data_dir):
            local, ontology, proposed = build_tools(data_dir)
            tools = [t for n, t in local.tools.items() if n in include_local] + list(self._tools)
            return ToolRegistry(tools), ontology, proposed
        return factory


if __name__ == "__main__":
    from .seed import DATA, seed

    if not (DATA / "crm.db").exists():
        seed(DATA)
    asyncio.run(serve_stdio(DATA))
