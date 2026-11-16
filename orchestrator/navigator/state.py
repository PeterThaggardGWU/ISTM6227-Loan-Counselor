"""The graph's state. The case file is stored as JSON-ready dicts so checkpoints
and LangGraph Studio can show it; nodes convert to and from mrn_shared models."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class NavState(TypedDict, total=False):
    session_id: str
    profile_input: dict[str, Any]      # stand-in for the Intake conversation
    case: dict[str, Any]               # mrn_shared.CaseFile
    to_run: list[str]                  # agents to run this round
    lender_match: Optional[dict]       # this round's results (None = not rerun)
    risk: Optional[dict]
    policy: Optional[dict]
    draft: Optional[dict]              # Explainer's current draft report
    critic: Optional[dict]             # {"passed": bool, "fixes": [...]}
    revisions: int
    reply: Optional[dict]              # the borrower's answer after a report
    reply_error: Optional[str]         # why the last what-if could not be applied
