"""Pydantic schemas: tool arguments, the agent's structured final answer, and stream events.

Each tool's JSON Schema (what the LLM sees) is generated from its Pydantic model, so the
schema the model is given and the validation applied to its arguments can never drift apart.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown arguments are rejected, not ignored


class ResolveEntityArgs(_Args):
    mention: str = Field(..., min_length=1, description="Name, partial name or email domain.")
    entity_type: Literal["Customer", "AccountManager", "Product"] | None = None


class SqlArgs(_Args):
    sql: str = Field(..., min_length=6, description="A single read-only SELECT statement.")


class SupportTicketsArgs(_Args):
    support_domain: str = Field(..., description="Customer email domain from resolve_entity.")
    status: Literal["open", "closed"] | None = None


class PolicySearchArgs(_Args):
    query: str = Field(..., min_length=2)


ActionType = Literal[
    "escalate_to_account_manager", "issue_service_credit", "send_collections_notice",
    "place_credit_hold", "no_action",
]


class ProposeActionArgs(_Args):
    action_type: ActionType
    customer_id: str = Field(..., pattern=r"^customer:\d+$", description="Canonical id, e.g. customer:3")
    reason: str = Field(..., min_length=5)


class FinalAnswer(_Args):
    """Structured output the agent must return through the `final_answer` tool."""

    summary: str = Field(..., min_length=5, description="Direct answer to the user's question.")
    facts_used: list[str] = Field(default_factory=list,
                                  description="Evidence cited, e.g. 'T-504 open P1', 'INV-1004 overdue', 'escalation_policy.md'.")
    recommended_action: ActionType = "no_action"
    customer_id: str | None = Field(None, description="Canonical customer id, if one customer is involved.")
    confidence: Literal["high", "medium", "low"] = "medium"
    unverified: list[str] = Field(default_factory=list, description="Anything the agent could not confirm.")


class StreamEvent(BaseModel):
    """One Server-Sent Event emitted while an agent run is in progress."""

    type: Literal["run_started", "llm_call", "tool_started", "tool_finished", "tool_failed",
                  "guard", "final", "run_finished"]
    run_id: str
    step: int | None = None
    data: dict = Field(default_factory=dict)
