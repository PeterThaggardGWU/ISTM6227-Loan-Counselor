"""The Intake node: one interview turn per run.

Each run either finishes the profile (builds the baseline scenario) or asks
up to 3 questions with a LangGraph interrupt, merges the answer and returns
to itself (graph edge intake -> intake). One turn per run means a resume
never repeats an LLM call that already happened.

Answers can be:
- a nested dict of fields:  {"purchase": {"target_state": "MD"}}
- free text:                "about 120k a year, and it's a condo townhouse"
  (or {"text": "..."}), which the LLM turns into fields.
After MAX_INTAKE_TURNS the questions go to a counselor instead (E-I15).
"""
from __future__ import annotations

import uuid

from langgraph.types import interrupt

from mrn_shared import CaseFile, Scenario, derive

from .. import config
from ..core.case import dump
from ..core.routing import AGENTS
from ..llm import LLMUnavailable
from .checks import all_problems
from .extractor import extract_profile
from .questions import next_questions


def merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _understand(answer, raw: dict, asked: list[str]) -> tuple[dict, str | None]:
    """Merge one answer into the raw profile. Returns (raw, error message or None)."""
    if isinstance(answer, dict) and "text" not in answer:
        return merge(raw, answer), None
    text = answer.get("text") if isinstance(answer, dict) else str(answer or "")
    if not text.strip():
        return raw, "I didn't get an answer. Please answer the questions above."
    try:
        return merge(raw, extract_profile(text, raw, asked)), None
    except LLMUnavailable:
        return raw, ("I can't read free-text answers right now. Please answer field by field, "
                     "for example {\"purchase\": {\"target_state\": \"MD\"}}.")


def intake(state):
    raw = dict(state.get("profile_input") or {})
    turns = state.get("intake_turns", 0)

    # First message in plain language (optional): read it once, then loop.
    if state.get("intake_text") and not state.get("intake_text_done"):
        raw, _ = _understand({"text": state["intake_text"]}, raw, [])
        return {"profile_input": raw, "intake_text_done": True, "intake_done": False}

    profile, problems = all_problems(raw)
    if not problems:
        case = CaseFile(session_id=state.get("session_id") or uuid.uuid4().hex[:8], status="running")
        case.scenarios.append(Scenario(id=0, label="Baseline", kind="baseline", rerun=list(AGENTS),
                                       profile=profile, derived=derive(profile, config.ASSUMED_RATE_PCT)))
        case.log("intake_complete", turns=turns)
        case.log("ran_agents", scenario_id=0, agents=list(AGENTS))
        return {"case": dump(case), "intake_done": True, "to_run": list(AGENTS),
                "lender_match": None, "risk": None, "policy": None, "draft": None, "critic": None,
                "revisions": 0, "reply": None, "reply_error": None, "llm_flags": []}

    questions = next_questions(problems)
    handoff = turns >= config.MAX_INTAKE_TURNS
    answer = interrupt({
        "type": "intake_questions",
        "role": "counselor" if handoff else "borrower",
        "message": ("Intake reached its question limit; a counselor please complete these fields."
                    if handoff else "A few more details before we can look at lenders and risk."),
        "questions": questions,
        "problems": problems,
        "error": state.get("intake_error"),
        "how_to_reply": "Free text, or a nested dict of fields, e.g. {'purchase': {'target_state': 'MD'}}",
    })
    raw, error = _understand(answer, raw, [q["field"] for q in questions])
    return {"profile_input": raw, "intake_turns": turns + 1, "intake_error": error, "intake_done": False}
