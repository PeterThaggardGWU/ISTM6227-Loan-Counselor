"""LLM second opinion on a draft that passed the rules.

A different model (CRITIC_MODEL) and a different prompt from the Explainer, so
the writer isn't grading its own work (D6). Looks for what regexes miss:
claims a citation doesn't support, paraphrased decisions ("you're a lock"),
steering, a discouraging tone, unexplained jargon.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from mrn_shared import Report

from .. import config
from ..llm import structured


class Issue(BaseModel):
    kind: Literal["unsupported_claim", "decision_language", "steering", "discouraging", "jargon"]
    quote: str = Field(description="The exact words from the report")
    fix: str = Field(description="One sentence telling the writer what to change")


class Verdict(BaseModel):
    issues: list[Issue] = Field(default_factory=list)


SYSTEM = """You review a mortgage-readiness report before a first-time homebuyer sees it.
Report only real problems of these kinds:
- unsupported_claim: a statement the cited guideline or the facts don't support
- decision_language: anything that says or implies the borrower is approved, denied, qualified or certain to get a loan
- steering: favoring or discouraging a specific lender, or a neighborhood for any reason other than cost
- discouraging: shaming or needlessly negative wording
- jargon: a mortgage term used without a plain explanation (DTI, LTV, PMI, PUD...)
Return an empty list when the report is fine. Don't invent issues."""


def llm_review(report: Report) -> list[str]:
    user = json.dumps({"report_markdown": report.markdown, "facts": report.facts,
                       "citations": [c.model_dump() for c in report.citations]}, indent=1)
    verdict = structured(Verdict, system=SYSTEM, user=user, model=config.CRITIC_MODEL, effort="low")
    return [f"[{i.kind}] \"{i.quote}\": {i.fix}" for i in verdict.issues]
