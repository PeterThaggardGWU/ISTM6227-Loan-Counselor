"""Explainer (STAND-IN): writes the report from the case file with a template.

The real agent will be an LLM writing plain English from the same inputs, and
will address the Critic's fixes. This version is deterministic, so the loop can
be tested without LLM calls.
"""
from __future__ import annotations

from mrn_shared import LenderMatchResult, PolicyResult, Report, RiskResult, Signal

from .. import config
from ._common import dump, load_case

DISCLAIMER = ("_This is an estimate to help you prepare. It is not a lending decision: "
              "only a lender can approve or deny a loan._")


def approval_signal(lm: LenderMatchResult | None) -> Signal:
    p = lm.best_approval_probability if lm else None
    if p is None:
        return Signal.NOT_ASSESSED
    return Signal.GREEN if p >= config.APPROVAL_GREEN else Signal.YELLOW if p >= config.APPROVAL_YELLOW else Signal.RED


def sustainability_signal(risk: RiskResult | None) -> Signal:
    if not risk or not risk.assessed:
        return Signal.NOT_ASSESSED
    return {"low": Signal.GREEN, "moderate": Signal.YELLOW, "elevated": Signal.RED}[risk.tier]


def _attach_results(state, case):
    """Store this round's results on the scenario; carried-over results stay as they are."""
    scen = case.current()
    for key, model in (("lender_match", LenderMatchResult), ("risk", RiskResult), ("policy", PolicyResult)):
        if key in scen.rerun and state.get(key):
            setattr(scen, key, model.model_validate(state[key]))
    return scen


def _money(x):
    return f"${x:,.0f}"


def write_report(scen, parent) -> Report:
    d, lm, rk, pol = scen.derived, scen.lender_match, scen.risk, scen.policy
    a_sig, s_sig = approval_signal(lm), sustainability_signal(rk)
    citations = list(pol.citations) if pol else []
    lines = [f"## Your readiness: {scen.label}", ""]

    best = lm.best_approval_probability if lm else None
    lines.append(f"**Can you get approved?** {a_sig.value.upper()}"
                 + (f": the best estimated approval odds among lenders active in your area are {best:.0%}." if best is not None else "."))
    if rk and rk.assessed:
        why = f" Main factors: {', '.join(rk.top_factors)}." if rk.top_factors else ""
        lines.append(f"**Can you sustain the payments?** {s_sig.value.upper()}: {rk.tier} risk of early delinquency.{why}")
    else:
        lines.append(f"**Can you sustain the payments?** NOT ASSESSED. {rk.note if rk else ''}".rstrip())

    if parent is not None:
        lines += ["", f"### What changed since \"{parent.label}\""]
        for path, value in scen.changes.items():
            lines.append(f"- `{path}` → {value}")
        if parent.lender_match and lm and parent.lender_match.best_approval_probability is not None and best is not None:
            lines.append(f"- Best approval odds: {parent.lender_match.best_approval_probability:.0%} → {best:.0%}")
        if parent.risk and rk and parent.risk.tier != rk.tier:
            lines.append(f"- Risk tier: {parent.risk.tier or 'not assessed'} → {rk.tier or 'not assessed'}")
        if parent.derived.dti_pct is not None and d.dti_pct is not None:
            lines.append(f"- Estimated DTI: {parent.derived.dti_pct}% → {d.dti_pct}%")
        lines.append(f"- Loan-to-value: {parent.derived.ltv_pct}% → {d.ltv_pct}%")

    lines += ["", "### Your numbers",
              f"- Loan amount {_money(d.loan_amount)}; down payment {d.down_payment_pct}% ({_money(scen.profile.purchase.down_payment)})",
              f"- Loan-to-value {d.ltv_pct}%"]
    if d.dti_pct is not None:
        lines.append(f"- Estimated debt-to-income {d.dti_pct}% (principal and interest at an assumed "
                     f"{d.assumed_rate_pct}% rate; taxes, insurance and HOA not included yet)")
    lines.append(f"- Cash left after the down payment: {_money(d.cash_after_down_payment)}")
    if d.cash_after_down_payment <= 0:
        lines.append("- **Note:** this leaves nothing for closing costs or cash reserves, which lenders usually expect.")

    if lm and lm.lenders:
        lines += ["", "### Lenders to consider", "| Lender | Est. approval odds |", "|---|---|"]
        lines += [f"| {l.name or l.lei} | {l.approval_probability:.0%} |" for l in lm.lenders]
        if lm.note:
            lines.append(f"\n_{lm.note}_")

    if citations:
        lines += ["", "### Guidelines that apply"]
        lines += [f"{i}. {c.excerpt} ({c.source}{', ' + c.section if c.section else ''})"
                  for i, c in enumerate(citations, 1)]

    lines += ["", DISCLAIMER, "",
              "Want to try a what-if? For example: a larger down payment, paying off a debt, "
              "a different county or property type."]
    return Report(approval_signal=a_sig, sustainability_signal=s_sig,
                  markdown="\n".join(lines), citations=citations)


def explainer(state):
    case = load_case(state)
    scen = _attach_results(state, case)
    parent = case.get(scen.parent_id) if scen.parent_id is not None else None
    report = write_report(scen, parent)
    return {"case": dump(case), "draft": dump(report)}
