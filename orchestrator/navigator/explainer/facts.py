"""The facts sheet: every value a report may state, already formatted.

The LLM writer gets only these values, and the Critic checks that every dollar
amount and percentage in the report appears here (critic/rules.py, rule 3).
Comparisons with the parent scenario are included only when both results came
from the same kind of source, so stand-in or missing numbers are never compared
with real ones (design.md E-O6).
"""
from __future__ import annotations

from typing import Optional

from mrn_shared import Scenario

from .. import config

from .signals import SignalResult, approval_odds


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}f}%"


def odds(p: float) -> str:
    return f"{p:.0%}"


def _comparable(a, b) -> bool:
    return a is not None and b is not None and a.source == b.source and a.source != "unavailable"


def build_facts(scen: Scenario, parent: Optional[Scenario],
                approval: SignalResult, sustain: SignalResult) -> dict[str, str]:
    d, p, f, b = scen.derived, scen.profile.purchase, scen.profile.finances, scen.profile.borrower
    facts: dict[str, str] = {
        "scenario": scen.label,
        "approval_signal": approval.signal.value,
        "sustainability_signal": sustain.signal.value,
        "purchase_price": money(p.purchase_price),
        "down_payment": money(p.down_payment),
        "down_payment_pct": pct(d.down_payment_pct, 1),
        "loan_amount": money(d.loan_amount),
        "ltv": pct(d.ltv_pct, 1),
        "annual_income": money(f.annual_income),
        "monthly_income": money(d.monthly_income),
        "monthly_debts_total": money(d.monthly_debt_payments),
        "savings": money(f.savings),
        "cash_after_down_payment": money(d.cash_after_down_payment),
        "county": p.target_county_name or p.target_county_fips or p.target_state,
        "property_type": p.property_type.value.replace("_", " "),
        "loan_type": p.loan_type,
        # thresholds printed in the signal rules
        "approval_green_threshold": odds(config.APPROVAL_GREEN),
        "approval_yellow_threshold": odds(config.APPROVAL_YELLOW),
        "dti_sustainability_limit": "50%",
    }
    for i, debt in enumerate(f.monthly_debts, 1):
        facts[f"debt_{i}"] = f"{debt.name}: {money(debt.monthly_payment)} a month"
    if b.credit_score is not None:
        facts["credit_score"] = str(b.credit_score)
    if d.dti_pct is not None:
        facts["dti"] = pct(d.dti_pct, 1)
        facts["assumed_rate"] = pct(d.assumed_rate_pct, 2)
        facts["est_principal_interest"] = money(d.est_monthly_principal_interest)
    if approval.value is not None:
        facts["approval_odds"] = odds(approval.value)
    lm = scen.lender_match
    if lm and lm.lenders:
        for i, l in enumerate(sorted(lm.lenders, key=lambda l: l.approval_probability, reverse=True), 1):
            facts[f"lender_{i}"] = f"{l.name or 'Lender (LEI ' + l.lei + ')'}: {odds(l.approval_probability)}"
    rk = scen.risk
    if rk and rk.assessed:
        facts["risk_tier"] = rk.tier or ""
        if rk.top_factors:
            facts["risk_factors"] = ", ".join(rk.top_factors)

    if parent is not None:
        pd = parent.derived
        facts["previous_scenario"] = parent.label
        facts["previous_ltv"] = pct(pd.ltv_pct, 1)
        facts["previous_loan_amount"] = money(pd.loan_amount)
        if pd.dti_pct is not None and d.dti_pct is not None:
            facts["previous_dti"] = pct(pd.dti_pct, 1)
        if _comparable(parent.lender_match, lm):
            prev, _ = approval_odds(parent.lender_match)
            if prev is not None:
                facts["previous_approval_odds"] = odds(prev)
        if parent.report:
            facts["previous_approval_signal"] = parent.report.approval_signal.value
            facts["previous_sustainability_signal"] = parent.report.sustainability_signal.value
        if _comparable(parent.risk, rk) and parent.risk.assessed:
            facts["previous_risk_tier"] = parent.risk.tier or ""
        for path, value in scen.changes.items():
            facts[f"change:{path}"] = money(value) if isinstance(value, (int, float)) and value > 1000 else str(value)
    return facts
