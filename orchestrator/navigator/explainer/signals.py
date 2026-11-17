"""The two readiness signals, computed by rule (design.md §8). The LLM never picks a color.

approval:        mean approval odds of the top 3 lenders with enough history
                 (>= MIN_LENDER_APPLICATIONS), thresholds APPROVAL_GREEN/YELLOW.
                 Capped at yellow when the credit score is below 620.
sustainability:  Risk tier (low/moderate/elevated -> green/yellow/red),
                 raised to at least yellow when DTI is above 50%.
Each comes with the rule in plain words, printed next to the color.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mrn_shared import LenderMatchResult, RiskResult, Scenario, Signal

from .. import config

TOP_N = 3
_ORDER = {Signal.GREEN: 0, Signal.YELLOW: 1, Signal.RED: 2}


@dataclass(frozen=True)
class SignalResult:
    signal: Signal
    value: Optional[float]   # approval: mean odds; sustainability: Risk probability if any
    rule: str
    notes: tuple[str, ...] = ()


def approval_odds(lm: Optional[LenderMatchResult]) -> tuple[Optional[float], int]:
    """(mean odds of the top lenders with enough history, how many lenders were averaged)."""
    if not lm or lm.source == "unavailable" or not lm.lenders:
        return None, 0
    known = [l for l in lm.lenders if l.historical_applications is not None]
    pool = [l for l in known if l.historical_applications >= config.MIN_LENDER_APPLICATIONS] if known else lm.lenders
    top = sorted(pool, key=lambda l: l.approval_probability, reverse=True)[:TOP_N]
    if not top:
        return None, 0
    return sum(l.approval_probability for l in top) / len(top), len(top)


def approval_signal(scen: Scenario) -> SignalResult:
    rule = (f"Average approval odds of the top {TOP_N} lenders in your area: green at "
            f"{config.APPROVAL_GREEN:.0%} or more, yellow from {config.APPROVAL_YELLOW:.0%}, red below.")
    odds, _ = approval_odds(scen.lender_match)
    if odds is None:
        why = scen.lender_match.note if scen.lender_match and scen.lender_match.note else "Lender Match didn't return results."
        return SignalResult(Signal.NOT_ASSESSED, None, rule, (why,))
    sig = (Signal.GREEN if odds >= config.APPROVAL_GREEN
           else Signal.YELLOW if odds >= config.APPROVAL_YELLOW else Signal.RED)
    notes = ["This estimate doesn't include your credit score, which lenders weigh heavily."]
    score = scen.profile.borrower.credit_score
    if score is not None and score < 620 and sig == Signal.GREEN:
        sig = Signal.YELLOW
        notes.append("Capped at yellow: most conventional loan programs need a credit score of 620 or more.")
    return SignalResult(sig, odds, rule, tuple(notes))


def sustainability_signal(scen: Scenario) -> SignalResult:
    rule = "Risk model tier for serious delinquency in the first two years: low green, moderate yellow, elevated red."
    rk: Optional[RiskResult] = scen.risk
    dti = scen.derived.dti_pct
    if not rk or not rk.assessed or rk.tier is None:
        why = rk.note if rk and rk.note else "Risk didn't return a result."
        return SignalResult(Signal.NOT_ASSESSED, None, rule, (why,))
    sig = {"low": Signal.GREEN, "moderate": Signal.YELLOW, "elevated": Signal.RED}[rk.tier]
    notes = []
    if dti is not None and dti > 50 and _ORDER[sig] < _ORDER[Signal.YELLOW]:
        sig = Signal.YELLOW
        notes.append("Raised to yellow: debt-to-income above 50% is hard to sustain whatever the model says.")
    return SignalResult(sig, rk.probability, rule, tuple(notes))
