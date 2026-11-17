"""Counselor nodes: review an escalated report, and sign off at the end.

counselor_review replies:
    {"action": "approve"}
    {"action": "annotate", "note": "..."}
    {"action": "edit", "markdown": "...", "note": "..."}     # original kept in original_markdown
counselor_signoff replies:
    {"action": "approve", "note": "..."}
    {"action": "edit_plan", "steps": ["...", "..."], "note": "..."}
"""
from __future__ import annotations

from langgraph.types import interrupt

from mrn_shared import Note

from ..core.case import dump, load_case
from .borrower import report_summary

REVIEW_ACTIONS = ("approve", "annotate", "edit")
SIGNOFF_ACTIONS = ("approve", "edit_plan")


def counselor_review(state):
    case = load_case(state)
    scen = case.current()
    reply = interrupt({
        "type": "counselor_review",
        "role": "counselor",
        "scenario_id": scen.id,
        "reasons": state.get("escalation") or [],
        "report_markdown": scen.report.markdown,
        "critic_notes": scen.report.critic_notes,
        "error": state.get("counselor_error"),
        "how_to_reply": {"approve": {"action": "approve"},
                         "annotate": {"action": "annotate", "note": "..."},
                         "edit": {"action": "edit", "markdown": "...", "note": "..."}},
    })
    action = reply.get("action") if isinstance(reply, dict) else None
    if action not in REVIEW_ACTIONS or (action == "edit" and not reply.get("markdown")):
        return {"counselor_error": "Reply approve, annotate (with a note) or edit (with markdown).",
                "counselor_done": False}

    if action == "edit":
        scen.report.original_markdown = scen.report.markdown
        scen.report.markdown = reply["markdown"]
    if reply.get("note"):
        case.counselor_notes.append(Note(scenario_id=scen.id, text=reply["note"]))
    scen.report.status = "approved"
    case.status = "awaiting_borrower"
    case.log("counselor_action", scenario_id=scen.id, action=action)
    return {"case": dump(case), "counselor_error": None, "counselor_done": True, "escalation": []}


def counselor_signoff(state):
    case = load_case(state)
    final = case.current()
    reply = interrupt({
        "type": "counselor_signoff",
        "role": "counselor",
        "ended_reason": case.ended_reason,
        "action_plan": case.action_plan,
        "scenarios": [report_summary(s) for s in case.scenarios],
        "final_report_markdown": final.report.markdown if final.report else None,
        "error": state.get("counselor_error"),
        "how_to_reply": {"approve": {"action": "approve", "note": "optional"},
                         "edit_plan": {"action": "edit_plan", "steps": ["..."], "note": "optional"}},
    })
    action = reply.get("action") if isinstance(reply, dict) else None
    if action not in SIGNOFF_ACTIONS or (action == "edit_plan" and not reply.get("steps")):
        return {"counselor_error": "Reply approve, or edit_plan with a list of steps.", "counselor_done": False}
    if action == "edit_plan":
        case.action_plan = [str(s) for s in reply["steps"]]
    if reply.get("note"):
        case.counselor_notes.append(Note(text=reply["note"]))
    case.status = "signed_off"
    case.log("counselor_action", action=f"signoff:{action}")
    return {"case": dump(case), "counselor_error": None, "counselor_done": True}
