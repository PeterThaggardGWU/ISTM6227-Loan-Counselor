"""The Critic node: rules first; if they pass and an LLM wrote the draft, an LLM review.

Writes {"critic": {"passed", "fixes"}} and bumps "revisions" on a failure.
Each pass is appended to the draft's critic_rounds for evaluation.
"""
from __future__ import annotations

from mrn_shared import CriticRound, Report

from ..core.case import dump, load_case
from ..llm import LLMUnavailable
from .reviewer import llm_review
from .rules import review


def critic(state):
    case = load_case(state)
    scen = case.current()
    report = Report.model_validate(state["draft"])
    flags = list(state.get("llm_flags") or [])

    fixes = review(report, list(scen.policy.citations) if scen.policy else [])
    reviewer = "rules"
    if not fixes and report.writer == "llm":
        try:
            fixes = llm_review(report)
            reviewer = "rules+llm"
        except LLMUnavailable as e:
            if str(e) != "disabled":
                flags.append(f"Critic LLM review unavailable ({e}); rules only")

    passed = not fixes
    report.critic_rounds.append(CriticRound(passed=passed, fixes=fixes, reviewer=reviewer))
    report.critic_notes = fixes
    if not passed:
        case.log("critic_failed", scenario_id=scen.id, fixes=fixes)
    return {"case": dump(case), "draft": dump(report), "llm_flags": flags,
            "critic": {"passed": passed, "fixes": fixes},
            "revisions": state.get("revisions", 0) + (0 if passed else 1)}
