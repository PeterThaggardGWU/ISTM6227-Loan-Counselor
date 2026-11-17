"""When a counselor must see a report before the borrower does (design.md §7.8, D15).

Returns plain-English reasons; an empty list means no escalation.
"""
from __future__ import annotations

from mrn_shared import Report, Scenario, Signal

from .specialists import specialist_status


def reasons_to_escalate(scen: Scenario, report: Report, llm_flags: list[str] | None = None) -> list[str]:
    reasons = []
    if report.status == "needs_review":
        reasons.append("The report still failed the Critic after the revision limit.")
    if report.sustainability_signal == Signal.RED:
        reasons.append("Sustainability is red: elevated risk of early delinquency.")
    if report.approval_signal == Signal.RED and report.sustainability_signal == Signal.NOT_ASSESSED:
        reasons.append("Approval is red and payment risk could not be assessed.")
    status = specialist_status(scen)
    if all(v == "unavailable" for v in (status["lender_match"], status["risk"])):
        reasons.append("Lender Match and Risk were both unavailable; the report has no signals.")
    for flag in llm_flags or []:
        reasons.append(f"LLM issue: {flag}")
    return reasons
