"""Risk (STAND-IN): placeholder rules until the origination-time model exists.

The real agent will call the Risk service (risk/, model trained in MRI) and
return its probability, tier and top factors. These rules are NOT a model.
"""
from __future__ import annotations

from mrn_shared import RiskResult

from ._common import current, dump


def risk(state):
    scen = current(state)
    p, d, b = scen.profile.purchase, scen.derived, scen.profile.borrower
    if p.loan_type != "conventional":
        result = RiskResult(source="stub", assessed=False,
                            note="Risk covers conventional loans only (FHA not assessed).")
    elif b.credit_score is None:
        result = RiskResult(source="stub", assessed=False, note="Needs a credit score.")
    else:
        dti = d.dti_pct if d.dti_pct is not None else 0
        factors = []
        if b.credit_score < 680: factors.append("credit score")
        if dti > 43: factors.append("debt-to-income ratio")
        if d.ltv_pct > 90: factors.append("loan-to-value ratio (small down payment)")
        tier = "elevated" if len(factors) >= 2 else "moderate" if factors else "low"
        result = RiskResult(source="stub", assessed=True, tier=tier, top_factors=factors,
                            note="STAND-IN rules, not the Risk model")
    return {"risk": dump(result)}
