"""The action_plan node: record why the session ended and write the plan."""
from __future__ import annotations

from ..core.case import dump, load_case
from ..core.stopping import max_rounds_reached, stalled
from ..llm import LLMUnavailable
from .action_plan import llm_steps, rule_steps


def ended_reason(state, case) -> str:
    action = (state.get("reply") or {}).get("action")
    if action == "give_up":
        return "gave_up"
    if action == "done":
        return "no_progress" if stalled(case) else "done"
    if max_rounds_reached(case):
        return "max_rounds"
    return "done"


def action_plan(state):
    case = load_case(state)
    case.ended_reason = ended_reason(state, case)
    case.status = "gave_up" if case.ended_reason == "gave_up" else "done"
    steps = rule_steps(case)
    try:
        steps = llm_steps(steps)
    except LLMUnavailable:
        pass
    case.action_plan = steps
    case.log("stopped", reason=case.ended_reason, scenarios=len(case.scenarios))
    return {"case": dump(case)}
