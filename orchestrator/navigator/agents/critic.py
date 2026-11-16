"""Critic (STAND-IN): rule checks on the Explainer's draft before anyone sees it.

The real Critic adds an LLM review (claims match the case file, every claim is
cited, no steering). These rules stay as a cheap first pass either way.
"""
from __future__ import annotations

import re

from mrn_shared import Report

from ._common import load_case

DECISION_LANGUAGE = re.compile(
    r"\b(you\s+(are|will\s+be|would\s+be)\s+(approved|denied|rejected)|guarantee[ds]?)\b", re.I)


def review(report: Report, policy_citation_count: int) -> list[str]:
    fixes = []
    if DECISION_LANGUAGE.search(report.markdown):
        fixes.append("Remove approve/deny or guarantee language; the tool never makes a lending decision.")
    if policy_citation_count and not report.citations:
        fixes.append("Cite the guideline passages the Policy agent found.")
    if "not a lending decision" not in report.markdown:
        fixes.append("Add the 'not a lending decision' disclaimer.")
    return fixes


def critic(state):
    case = load_case(state)
    scen = case.current()
    report = Report.model_validate(state["draft"])
    n_cites = len(scen.policy.citations) if scen.policy else 0
    fixes = review(report, n_cites)
    return {"critic": {"passed": not fixes, "fixes": fixes},
            "revisions": state.get("revisions", 0) + (1 if fixes else 0)}
