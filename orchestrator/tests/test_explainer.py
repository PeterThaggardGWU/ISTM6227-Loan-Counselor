"""Signals are computed by rule; facts cover every number; comparisons need like sources."""
import copy

from mrn_shared import (BorrowerProfile, LenderMatchResult, LenderOption, RiskResult, Scenario, Signal, derive)
from navigator.explainer.facts import build_facts
from navigator.explainer.signals import approval_odds, approval_signal, sustainability_signal
from sample_profiles import TOWNHOUSE_MONTGOMERY_PUD


def scen(credit=715, income=None, lenders=None, risk=None):
    d = copy.deepcopy(TOWNHOUSE_MONTGOMERY_PUD)
    d["borrower"]["credit_score"] = credit
    if income:
        d["finances"]["annual_income"] = income
    prof = BorrowerProfile.model_validate(d)
    return Scenario(id=0, label="Baseline", profile=prof, derived=derive(prof, 6.5),
                    lender_match=lenders, risk=risk)


def lm(*probs, apps=None):
    return LenderMatchResult(source="service", lenders=[
        LenderOption(lei=f"L{i}", name=f"Lender {i}", approval_probability=p,
                     historical_applications=(apps[i] if apps else None)) for i, p in enumerate(probs)])


def test_approval_uses_mean_of_top_three_not_the_max():                  # D8 / B8
    odds, n = approval_odds(lm(0.99, 0.70, 0.65, 0.60))
    assert n == 3 and round(odds, 3) == 0.78


def test_lenders_with_little_history_are_left_out():
    odds, _ = approval_odds(lm(0.99, 0.70, 0.65, 0.60, apps=[10, 500, 500, 500]))
    assert round(odds, 3) == 0.65


def test_low_credit_caps_approval_at_yellow():                           # E-M7
    assert approval_signal(scen(credit=600, lenders=lm(0.95, 0.94, 0.93))).signal == Signal.YELLOW


def test_dti_above_50_raises_sustainability_to_yellow():
    s = scen(income=55000, risk=RiskResult(source="service", assessed=True, tier="low"))
    assert s.derived.dti_pct > 50
    assert sustainability_signal(s).signal == Signal.YELLOW


def test_unavailable_lender_match_is_not_assessed():
    s = scen(lenders=LenderMatchResult(source="unavailable", note="down"))
    assert approval_signal(s).signal == Signal.NOT_ASSESSED


def test_no_comparison_between_different_sources():                      # E-O6
    parent = scen(lenders=LenderMatchResult(source="stub", lenders=lm(0.9).lenders))
    parent.id, child = 0, scen(lenders=lm(0.8))
    child.id, child.parent_id = 1, 0
    facts = build_facts(child, parent, approval_signal(child), sustainability_signal(child))
    assert "previous_approval_odds" not in facts
