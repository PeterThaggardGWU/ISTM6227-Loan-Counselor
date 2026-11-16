"""Figures derived from a BorrowerProfile. Plain arithmetic; no LLM, no model."""
from __future__ import annotations

from typing import Optional

from .schema import BorrowerProfile, Derived


def monthly_payment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    """Standard fully amortizing principal-and-interest payment."""
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / term_months
    return principal * r / (1 - (1 + r) ** -term_months)


def derive(profile: BorrowerProfile, assumed_rate_pct: Optional[float] = None) -> Derived:
    p, f = profile.purchase, profile.finances
    loan = max(p.purchase_price - p.down_payment, 0.0)
    monthly_income = f.annual_income / 12
    debts = sum(d.monthly_payment for d in f.monthly_debts)

    pi = dti = None
    if assumed_rate_pct is not None:
        pi = monthly_payment(loan, assumed_rate_pct, p.loan_term_months)
        dti = round(100 * (debts + pi) / monthly_income, 1) if monthly_income > 0 else None

    return Derived(
        loan_amount=round(loan, 2),
        ltv_pct=round(100 * loan / p.purchase_price, 2),
        down_payment_pct=round(100 * p.down_payment / p.purchase_price, 2),
        monthly_income=round(monthly_income, 2),
        monthly_debt_payments=round(debts, 2),
        assumed_rate_pct=assumed_rate_pct,
        est_monthly_principal_interest=None if pi is None else round(pi, 2),
        dti_pct=dti,
        cash_after_down_payment=round(f.savings - p.down_payment, 2),
    )
