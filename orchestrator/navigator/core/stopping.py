"""Stopping rules for the borrower loop (design.md §7.7, E-W10, E-W12).

The session ends when the borrower says done or gives up, or after MAX_ROUNDS
borrower scenarios. When rounds stop changing anything material, the borrower
is offered a wrap-up instead of the session silently continuing or ending.
Every stop goes to the action plan.
"""
from __future__ import annotations

from typing import Optional

from mrn_shared import CaseFile, Scenario

from .. import config


def borrower_rounds(case: CaseFile) -> int:
    """Scenarios that count toward MAX_ROUNDS (suggested scenarios don't)."""
    return sum(1 for s in case.scenarios if s.kind != "suggested")


def max_rounds_reached(case: CaseFile) -> bool:
    return borrower_rounds(case) >= config.MAX_ROUNDS


def _best_odds(s: Scenario) -> Optional[float]:
    return s.lender_match.best_approval_probability if s.lender_match else None


def material_change(before: Scenario, after: Scenario) -> bool:
    """A round matters if either signal changed or approval odds moved by MATERIAL_ODDS_CHANGE or more."""
    if not (before.report and after.report):
        return True
    if (before.report.approval_signal != after.report.approval_signal
            or before.report.sustainability_signal != after.report.sustainability_signal):
        return True
    a, b = _best_odds(before), _best_odds(after)
    return a is not None and b is not None and abs(b - a) >= config.MATERIAL_ODDS_CHANGE


def stalled(case: CaseFile) -> bool:
    """True when the last NO_PROGRESS_ROUNDS what-ifs each changed nothing material."""
    rounds = [s for s in case.scenarios if s.kind != "suggested"]
    n = config.NO_PROGRESS_ROUNDS
    if len(rounds) < n + 1:
        return False
    recent = rounds[-(n + 1):]
    return not any(material_change(a, b) for a, b in zip(recent, recent[1:]))
