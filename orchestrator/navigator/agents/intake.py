"""Intake (STAND-IN): validates a structured profile and asks for anything missing.

The real Intake agent will interview the borrower in plain language (LLM with
structured output) and produce the same BorrowerProfile.
"""
from __future__ import annotations

import uuid

from langgraph.types import interrupt
from pydantic import ValidationError

from mrn_shared import BorrowerProfile, CaseFile, Scenario, derive

from .. import config
from ..routing import AGENTS
from ._common import dump


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def intake(state):
    raw = dict(state.get("profile_input") or {})
    while True:
        try:
            profile = BorrowerProfile.model_validate(raw)
            break
        except ValidationError as e:
            problems = [{"field": ".".join(map(str, err["loc"])), "problem": err["msg"]} for err in e.errors()]
            answers = interrupt({
                "type": "intake_questions",
                "message": "Intake needs a few more details before the agents can run.",
                "problems": problems,
                "how_to_reply": "Resume with a nested dict of answers, e.g. "
                                "{'purchase': {'target_state': 'MD'}}",
            })
            raw = _merge(raw, answers or {})

    case = CaseFile(session_id=state.get("session_id") or uuid.uuid4().hex[:8], status="running")
    case.scenarios.append(Scenario(
        id=0, label="Baseline", rerun=list(AGENTS),
        profile=profile, derived=derive(profile, config.ASSUMED_RATE_PCT),
    ))
    return {"case": dump(case), "to_run": list(AGENTS), "lender_match": None, "risk": None,
            "policy": None, "draft": None, "critic": None, "revisions": 0, "reply": None,
            "reply_error": None}
