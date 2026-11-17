"""LLM narrative: the explanation paragraphs between the signals and the numbers.

The model sees only the facts sheet, the signal rules and the citations; never
the borrower's raw text (so injected instructions can't reach it, E-S3). It
must answer every Critic fix. Raises LLMUnavailable on any failure; the node
then uses template_narrative().
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from mrn_shared import Citation

from ..llm import structured


class NarrativeDraft(BaseModel):
    paragraphs: list[str] = Field(
        description="2 to 5 short paragraphs or markdown bullet lists. No headings for signals.")


SYSTEM = """You are the Explainer for a HUD-approved housing counseling tool. You write the
explanation part of a readiness report for a first-time homebuyer.

Write in plain English at about an 8th-grade reading level. Be warm and direct, never shaming.

The report around your text already shows the two signals with their rules, the numbers table,
the lender list, the guideline citations and the disclaimer. Your job: explain what the signals
mean for this borrower and what drives them; if there is a previous scenario, explain what changed
and why; point to the guideline passages by number, e.g. "(guideline 2)".

Hard rules:
- Use only numbers that appear in the facts. Copy them exactly as written there. Never compute new ones.
- Never say the borrower is or will be approved, denied, qualified or pre-approved, and never guarantee anything.
- Never recommend one lender over another, and never suggest a neighborhood for any reason except cost.
- Never mention race, ethnicity, sex, age, marital or family status, religion, national origin or disability.
- If a signal is not assessed, say so plainly and give the reason from the facts.
- Address every Critic fix you are given."""


def write_narrative(facts: dict, rules: dict, citations: list[Citation], fixes: list[str]) -> list[str]:
    user = json.dumps({
        "facts": facts,
        "signal_rules": rules,
        "guidelines": [{"number": i, "source": c.source, "section": c.section, "excerpt": c.excerpt}
                       for i, c in enumerate(citations, 1)],
        "critic_fixes_to_address": fixes,
    }, indent=1)
    draft = structured(NarrativeDraft, system=SYSTEM, user=user, effort="medium")
    return [p for para in draft.paragraphs for p in (para.strip(), "") if para.strip()][:-1] or []
