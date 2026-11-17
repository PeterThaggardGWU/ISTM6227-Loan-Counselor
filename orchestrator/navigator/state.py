"""The graph's state. The case file is stored as JSON-ready dicts so checkpoints
and LangGraph Studio can show it; nodes convert to and from mrn_shared models."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class NavState(TypedDict, total=False):
    session_id: str
    # Intake
    profile_input: dict[str, Any]      # profile so far (structured answers merged in)
    intake_text: str                   # optional first message in plain language
    intake_text_done: bool
    intake_turns: int
    intake_error: Optional[str]
    intake_done: bool
    # Case file and this round
    case: dict[str, Any]               # mrn_shared.CaseFile
    to_run: list[str]                  # specialists to run this round
    lender_match: Optional[dict]       # this round's results (None = not rerun)
    risk: Optional[dict]
    policy: Optional[dict]
    draft: Optional[dict]              # Explainer's current draft report
    critic: Optional[dict]             # {"passed": bool, "fixes": [...]}
    revisions: int
    llm_flags: list[str]               # LLM problems this round (refusals, fallbacks)
    escalation: list[str]              # why a counselor must review this report
    # Human replies
    reply: Optional[dict]              # the borrower's answer after a report
    reply_error: Optional[str]         # why the last reply could not be used
    counselor_error: Optional[str]
    counselor_done: bool
