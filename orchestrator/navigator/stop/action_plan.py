"""3 to 5 next steps from the strongest scenario, plus a counselor handoff line.

rule_steps()  -> deterministic steps (also the fallback)
llm_steps()   -> the LLM rewrites them in plain, warm English; the result must
                 pass the Critic's text rules, or the rule steps are used.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from mrn_shared import CaseFile, Scenario, Signal

from ..critic.rules import check_text
from ..explainer.facts import money
from ..llm import LLMUnavailable, structured

HANDOFF = "Review this plan with your housing counselor before you apply to any lender."
_RANK = {Signal.GREEN: 0, Signal.YELLOW: 1, Signal.RED: 2, Signal.NOT_ASSESSED: 3}


def best_scenario(case: CaseFile) -> Scenario:
    """Strongest signals first, then higher approval odds; earlier scenario wins a tie."""
    def key(s: Scenario):
        r = s.report
        sig = (_RANK[r.approval_signal] + _RANK[r.sustainability_signal]) if r else 99
        odds = s.lender_match.best_approval_probability if s.lender_match and s.lender_match.best_approval_probability else 0
        return (sig, -odds, s.id)
    reported = [s for s in case.scenarios if s.report] or case.scenarios
    return min(reported, key=key)


def rule_steps(case: CaseFile) -> list[str]:
    best = best_scenario(case)
    p, f, b, d = best.profile.purchase, best.profile.finances, best.profile.borrower, best.derived
    steps: list[str] = []
    if best.id != 0:
        steps.append(f"Your strongest result was the \"{best.label}\" scenario. Use it as your target.")
    if b.credit_score is None:
        steps.append("Get your free credit reports at AnnualCreditReport.com and check them for errors; "
                     "lenders will look at your score closely.")
    elif b.credit_score < 620:
        steps.append("Work on your credit score first: most conventional loan programs need 620 or more.")
    if d.dti_pct is not None and d.dti_pct > 43 and f.monthly_debts:
        biggest = max(f.monthly_debts, key=lambda x: x.monthly_payment)
        steps.append(f"Lower your monthly debts. Your {biggest.name} ({money(biggest.monthly_payment)} a month) "
                     "is the biggest one.")
    if d.ltv_pct > 80:
        steps.append("Plan for mortgage insurance, since your down payment is under 20%, or keep saving toward 20%.")
    if d.cash_after_down_payment < 0.03 * p.purchase_price:
        steps.append(f"Set aside money for closing costs (often about 3% of the price, {money(0.03 * p.purchase_price)}) "
                     "and a few months of payments as reserves.")
    if b.first_time_buyer:
        steps.append("Take a HUD-approved homebuyer education course; many first-time buyer programs require one.")
    steps = steps[:5] or ["Keep your finances steady until you apply: avoid new debt and large unexplained deposits."]
    return [*steps, HANDOFF]


class PlanDraft(BaseModel):
    steps: list[str] = Field(description="The same steps, rewritten; same order, same numbers")


SYSTEM = """Rewrite these next steps for a first-time homebuyer in plain, warm English (8th-grade level).
Keep every step, in order, with its numbers exactly as given. One or two sentences each.
Never say the borrower will be approved, denied or qualify; never recommend a lender."""


def llm_steps(steps: list[str]) -> list[str]:
    draft = structured(PlanDraft, system=SYSTEM, user=json.dumps({"steps": steps}), effort="low")
    out = [s.strip() for s in draft.steps if s.strip()]
    if len(out) != len(steps) or check_text(" ".join(out)):
        raise LLMUnavailable("rewrite changed the steps or failed the Critic rules")
    if out[-1] != HANDOFF:
        out[-1] = HANDOFF
    return out
