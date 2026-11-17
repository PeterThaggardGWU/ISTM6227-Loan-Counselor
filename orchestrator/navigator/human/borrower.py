"""Borrower nodes: publish the report, wait for a reply, apply a what-if.

Replies to await_borrower:
    {"action": "what_if", "changes": {"purchase.down_payment": 90000}, "label": "..."}
    {"action": "what_if", "text": "what if I put 20% down?"}          # LLM parses it
    {"action": "what_if", ..., "from_scenario": 1}                    # branch from an earlier scenario (E-W9)
    {"action": "done"}   {"action": "give_up"}
Add "by": "counselor" when a counselor makes the change.
"""
from __future__ import annotations

from langgraph.types import interrupt
from pydantic import ValidationError

from mrn_shared import Report, Scenario, apply_changes, derive

from .. import config
from ..core.case import dump, load_case
from ..core.escalation import reasons_to_escalate
from ..core.routing import EDITABLE_PATHS, agents_to_rerun
from ..core.stopping import stalled
from ..explainer.template import signal_label
from ..intake.extractor import changes_dict, parse_what_if
from ..llm import LLMUnavailable

ACTIONS = ("what_if", "done", "give_up")
HOW_TO_REPLY = {
    "what_if": {"action": "what_if", "label": "20% down, car paid off",
                "changes": {"purchase.down_payment": 90000, "finances.remove_debt": "car loan"}},
    "what_if_text": {"action": "what_if", "text": "What if I put 20% down and paid off my car?"},
    "done": {"action": "done"},
    "give_up": {"action": "give_up"},
}


def publish_report(state):
    """Finalize the draft; escalate to a counselor when a rule says so (D15)."""
    case = load_case(state)
    scen = case.current()
    critic = state.get("critic") or {}
    report = Report.model_validate(state["draft"])
    report.revisions = state.get("revisions", 0)
    report.status = "passed" if critic.get("passed") else "needs_review"
    scen.report = report
    reasons = reasons_to_escalate(scen, report, state.get("llm_flags"))
    if reasons:
        case.status = "needs_review"
        case.log("escalated", scenario_id=scen.id, reasons=reasons)
    else:
        case.status = "awaiting_borrower"
    return {"case": dump(case), "escalation": reasons}


def await_borrower(state):
    case = load_case(state)
    scen = case.current()
    wrap_up = stalled(case)
    reply = interrupt({
        "type": "report",
        "role": "borrower",
        "scenario_id": scen.id,
        "label": scen.label,
        "approval_signal": scen.report.approval_signal.value,
        "sustainability_signal": scen.report.sustainability_signal.value,
        "report_markdown": scen.report.markdown,
        "counselor_reviewed": scen.report.status == "approved",
        "suggest_wrap_up": wrap_up,
        "message": ("Your last few changes didn't move either signal much. Want to wrap up with an "
                    "action plan? Reply done, or keep trying what-ifs.") if wrap_up else None,
        "error": state.get("reply_error"),
        "how_to_reply": HOW_TO_REPLY,
    })
    if not isinstance(reply, dict) or reply.get("action") not in ACTIONS:
        return {"reply": {"action": "invalid", "raw": reply},
                "reply_error": "I didn't understand that reply. Choose a what-if, done or give up."}
    return {"reply": reply, "reply_error": None}


def _reject(case, why: str):
    case.log("what_if_rejected", reason=why)
    return {"case": dump(case), "reply_error": why}


def apply_what_if(state):
    case = load_case(state)
    reply = state["reply"]
    try:
        parent = case.get(reply["from_scenario"]) if "from_scenario" in reply else case.current()
    except StopIteration:
        return _reject(case, f"There's no scenario {reply['from_scenario']}.")

    changes, label = dict(reply.get("changes") or {}), reply.get("label")
    if not changes and reply.get("text"):
        try:
            draft = parse_what_if(reply["text"], parent.profile)
        except LLMUnavailable:
            return _reject(case, "I can't read free-text what-ifs right now. Please send the changes as fields, "
                                 "for example {\"purchase.down_payment\": 90000}.")
        if draft.unparsed:
            return _reject(case, f"I couldn't turn \"{draft.unparsed}\" into a number. Can you say it as an "
                                 "amount, for example a new income or down payment?")
        changes, label = changes_dict(draft), label or draft.label

    if not changes:
        return _reject(case, "A what-if needs at least one change.")
    not_editable = sorted(set(changes) - EDITABLE_PATHS)
    if not_editable:
        return _reject(case, f"These can't be changed in a what-if: {', '.join(not_editable)}.")
    try:
        profile = apply_changes(parent.profile, changes)
    except (KeyError, ValueError, ValidationError) as e:
        msg = e.errors()[0]["msg"].removeprefix("Value error, ") if isinstance(e, ValidationError) else str(e).strip("'\"")
        return _reject(case, f"Couldn't apply that what-if: {msg}")
    if profile == parent.profile:
        return _reject(case, "That's the same as the current scenario, so nothing would change.")

    to_run = agents_to_rerun(changes)
    new_id = max(s.id for s in case.scenarios) + 1
    scen = Scenario(
        id=new_id, label=label or f"What-if {new_id}", kind="what_if",
        origin="counselor" if reply.get("by") == "counselor" else "borrower",
        parent_id=parent.id, changes=changes, rerun=to_run, profile=profile,
        derived=derive(profile, config.ASSUMED_RATE_PCT),
        # results of agents that don't need to rerun carry over from the parent
        lender_match=None if "lender_match" in to_run else parent.lender_match,
        risk=None if "risk" in to_run else parent.risk,
        policy=None if "policy" in to_run else parent.policy,
    )
    case.scenarios.append(scen)
    case.status = "running"
    case.log("ran_agents", scenario_id=new_id, agents=to_run, changes=changes)
    return {"case": dump(case), "to_run": to_run, "lender_match": None, "risk": None, "policy": None,
            "draft": None, "critic": None, "revisions": 0, "reply_error": None, "llm_flags": [],
            "escalation": []}


def report_summary(scen) -> str:
    r = scen.report
    return (f"{scen.label}: approval {signal_label(r.approval_signal)}, "
            f"sustainability {signal_label(r.sustainability_signal)}") if r else scen.label
