"""The human side of the loop: publish the report, wait for the borrower, apply what-ifs."""
from __future__ import annotations

from langgraph.types import interrupt
from pydantic import ValidationError

from mrn_shared import Report, Scenario, apply_changes, derive

from .. import config
from ..routing import agents_to_rerun
from ._common import dump, load_case

HOW_TO_REPLY = {
    "what_if": {"action": "what_if", "label": "20% down, car paid off",
                "changes": {"purchase.down_payment": 90000, "finances.remove_debt": "car loan"}},
    "done": {"action": "done"},
    "give_up": {"action": "give_up"},
}


def publish_report(state):
    case = load_case(state)
    scen = case.current()
    report = Report.model_validate(state["draft"])
    report.critic_notes = (state.get("critic") or {}).get("fixes", [])
    report.revisions = state.get("revisions", 0)
    scen.report = report
    case.status = "awaiting_borrower"
    return {"case": dump(case)}


def await_borrower(state):
    case = load_case(state)
    scen = case.current()
    reply = interrupt({
        "type": "report",
        "scenario_id": scen.id,
        "label": scen.label,
        "approval_signal": scen.report.approval_signal.value,
        "sustainability_signal": scen.report.sustainability_signal.value,
        "report_markdown": scen.report.markdown,
        "error": state.get("reply_error"),
        "how_to_reply": HOW_TO_REPLY,
    })
    if not isinstance(reply, dict) or reply.get("action") not in ("what_if", "done", "give_up"):
        reply = {"action": "invalid", "raw": reply}
    return {"reply": reply, "reply_error": None}


def apply_what_if(state):
    case = load_case(state)
    parent = case.current()
    reply = state["reply"]
    changes = reply.get("changes") or {}
    try:
        if not changes:
            raise ValueError("A what-if needs at least one change.")
        profile = apply_changes(parent.profile, changes)
        derived = derive(profile, config.ASSUMED_RATE_PCT)
        if derived.cash_after_down_payment < 0:
            raise ValueError("The down payment is larger than the savings entered.")
    except (KeyError, ValueError, ValidationError) as e:
        return {"reply_error": f"Couldn't apply that what-if: {e}"}

    to_run = agents_to_rerun(changes)
    scen = Scenario(
        id=parent.id + 1, label=reply.get("label") or f"What-if {parent.id + 1}",
        parent_id=parent.id, changes=changes, rerun=to_run, profile=profile, derived=derived,
        # results of agents that don't need to rerun carry over from the parent
        lender_match=None if "lender_match" in to_run else parent.lender_match,
        risk=None if "risk" in to_run else parent.risk,
        policy=None if "policy" in to_run else parent.policy,
    )
    case.scenarios.append(scen)
    case.status = "running"
    return {"case": dump(case), "to_run": to_run, "lender_match": None, "risk": None,
            "policy": None, "draft": None, "critic": None, "revisions": 0, "reply_error": None}


def finish(state):
    case = load_case(state)
    action = (state.get("reply") or {}).get("action")
    case.status = "gave_up" if action == "give_up" else "done"
    return {"case": dump(case)}
