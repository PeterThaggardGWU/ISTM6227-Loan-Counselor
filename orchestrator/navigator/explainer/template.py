"""Report layout, and the rule-based narrative used when the LLM is off.

compose() builds every report, whoever wrote the narrative:

    ## Your readiness: <scenario>
    **Can you get approved?** <SIGNAL> ...          <- code (signals.py)
    **Can you sustain the payments?** <SIGNAL> ...  <- code
    <narrative>                                     <- LLM (writer.py) or template_narrative()
    ### Your numbers / Lenders to consider / Guidelines that apply   <- code (facts)
    disclaimer + what-if offer                      <- code

The template version is also the "pipeline baseline" for evaluation (design.md §11.3).
"""
from __future__ import annotations

from typing import Optional

from mrn_shared import Citation, Scenario, Signal

from .facts import odds
from .signals import SignalResult

DISCLAIMER = ("_This is an estimate to help you prepare. It is not a lending decision: "
              "only a lender can approve or deny a loan._")
WHAT_IF_OFFER = ("Want to try a what-if? For example: a larger down payment, paying off a debt, "
                 "a different county or property type.")
LENDER_ORDER_NOTE = ("Lenders are listed by the model's estimated approval odds. This is not a "
                     "recommendation of any lender; compare offers from several.")


CHANGE_LABELS = {
    "purchase.purchase_price": "Purchase price", "purchase.down_payment": "Down payment",
    "purchase.target_state": "State", "purchase.target_county_fips": "County",
    "purchase.target_county_name": "County", "purchase.property_type": "Property type",
    "purchase.loan_type": "Loan type", "purchase.loan_term_months": "Loan term (months)",
    "finances.annual_income": "Yearly income", "finances.monthly_debts": "Monthly debts",
    "finances.remove_debt": "Paid off", "finances.savings": "Savings",
    "borrower.credit_score": "Credit score", "borrower.num_borrowers": "Number of borrowers",
    "borrower.first_time_buyer": "First-time buyer",
}


def signal_label(sig: Signal) -> str:
    return "NOT ASSESSED" if sig == Signal.NOT_ASSESSED else sig.value.upper()


def signal_lines(approval: SignalResult, sustain: SignalResult, facts: dict) -> list[str]:
    a = f"**Can you get approved?** {signal_label(approval.signal)}"
    if approval.value is not None:
        a += f": average estimated approval odds among the top lenders in your area are {facts['approval_odds']}."
    s = f"**Can you sustain the payments?** {signal_label(sustain.signal)}"
    if sustain.signal != Signal.NOT_ASSESSED:
        s += f": {facts.get('risk_tier', '')} risk of early delinquency."
    lines = [a, *[f"- {n}" for n in approval.notes], f"  _Rule: {approval.rule}_", "",
             s, *[f"- {n}" for n in sustain.notes], f"  _Rule: {sustain.rule}_"]
    return lines


def template_narrative(facts: dict) -> list[str]:
    out = []
    if "risk_factors" in facts:
        out.append(f"The main factors behind the payment-risk estimate are: {facts['risk_factors']}.")
    if "previous_scenario" in facts:
        out += ["", f"### What changed since \"{facts['previous_scenario']}\""]
        out += [f"- {CHANGE_LABELS.get(k.removeprefix('change:'), k.removeprefix('change:'))}: {v}"
                for k, v in facts.items() if k.startswith("change:")]
        if "previous_approval_odds" in facts and "approval_odds" in facts:
            out.append(f"- Approval odds: {facts['previous_approval_odds']} → {facts['approval_odds']}")
        if "previous_risk_tier" in facts and facts.get("previous_risk_tier") != facts.get("risk_tier"):
            out.append(f"- Risk tier: {facts['previous_risk_tier']} → {facts.get('risk_tier', 'not assessed')}")
        if "previous_dti" in facts and "dti" in facts:
            out.append(f"- Estimated debt-to-income: {facts['previous_dti']} → {facts['dti']}")
        out.append(f"- Loan-to-value: {facts['previous_ltv']} → {facts['ltv']}")
    return out


def compose(scen: Scenario, approval: SignalResult, sustain: SignalResult, facts: dict,
            narrative: list[str], citations: list[Citation]) -> str:
    lines = [f"## Your readiness: {scen.label}", "", *signal_lines(approval, sustain, facts)]
    if narrative:
        lines += ["", *narrative]

    lines += ["", "### Your numbers",
              f"- Loan amount {facts['loan_amount']}; down payment {facts['down_payment_pct']} ({facts['down_payment']})",
              f"- Loan-to-value {facts['ltv']}"]
    if "dti" in facts:
        lines.append(f"- Estimated debt-to-income {facts['dti']} (principal and interest at an assumed "
                     f"{facts['assumed_rate']} rate; taxes, insurance and HOA not included yet)")
    lines.append(f"- Cash left after the down payment: {facts['cash_after_down_payment']}")
    if scen.derived.cash_after_down_payment <= 0:
        lines.append("- **Note:** this leaves nothing for closing costs or cash reserves, which lenders usually expect.")

    lm = scen.lender_match
    if lm and lm.lenders:
        lines += ["", "### Lenders to consider", "| Lender | Est. approval odds |", "|---|---|"]
        for l in sorted(lm.lenders, key=lambda l: l.approval_probability, reverse=True):
            lines.append(f"| {l.name or 'Lender (LEI ' + l.lei + ')'} | {odds(l.approval_probability)} |")
        lines += ["", f"_{LENDER_ORDER_NOTE}_"]
        if lm.note:
            lines.append(f"_{lm.note}_")

    if citations:
        lines += ["", "### Guidelines that apply"]
        lines += [f"{i}. {c.excerpt} ({c.source}{', ' + c.section if c.section else ''})"
                  for i, c in enumerate(citations, 1)]
    elif scen.policy and scen.policy.source == "unavailable":
        lines += ["", "_Guideline references are unavailable for this report._"]

    lines += ["", DISCLAIMER, "", WHAT_IF_OFFER]
    return "\n".join(lines)
