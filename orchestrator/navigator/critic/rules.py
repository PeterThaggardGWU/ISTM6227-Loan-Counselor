"""Rule checks on a draft report. Cheap and deterministic; they run first.

    review(report, policy_citations) -> list of fixes ([] = pass)

1. No decision or guarantee language (approved, denied, qualify, pre-approved, guarantee)
2. The not-a-lending-decision disclaimer is present
3. Every dollar amount and percentage matches a value on the facts sheet
4. Every citation is one the Policy agent returned, word for word
5. The signal lines match the report's signals
6. No protected-trait words
7. No steering language toward or away from a lender or neighborhood

check_text() runs rules 1, 6 and 7 on any text (the action plan uses it too).
"""
from __future__ import annotations

import re

from mrn_shared import Citation, Report

from ..explainer.template import DISCLAIMER, signal_label

DECISION = re.compile(
    r"\b(you\s*(?:'re|'ll|are|will|would|should)\s+(?:be\s+|get\s+)?(?:approved|denied|rejected|declined)"
    r"|(?:pre-?approved|pre-?qualified)"
    r"|(?:you|they)\s+(?:qualify|qualifies|qualified)"
    r"|guarantee[ds]?"
    r"|(?:a\s+)?lock\s+for)\b", re.I)
PROTECTED = re.compile(
    r"\b(race|racial|ethnicity|ethnic|religio(?:n|us)|national\s+origin|gender|sex|pregnan\w*|"
    r"disab(?:led|ility)|marital|married|single\s+(?:mom|mother|dad|father|parent)|familial|"
    r"your\s+age|immigrant|citizenship)\b", re.I)
STEERING = re.compile(
    r"\b(we\s+recommend|you\s+should\s+(?:go\s+with|choose|pick|use|apply\s+(?:to|with))|"
    r"(?:the\s+)?best\s+lender|avoid\s+(?:this|that|these)\s+lenders?|better\s+neighborhood|"
    r"(?:safer|nicer)\s+(?:area|neighborhood))\b", re.I)

# $1,234 / $1,234.56 and 12% / 12.5%
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def _numbers(text: str) -> set[float]:
    nums = {float(m.replace(",", "")) for m in MONEY.findall(text)}
    nums |= {float(m) for m in PERCENT.findall(text)}
    return nums


def _matches(n: float, allowed: set[float]) -> bool:
    return any(abs(n - a) <= max(0.51, abs(a) * 0.005) for a in allowed)


def check_text(text: str) -> list[str]:
    fixes = []
    if m := DECISION.search(text):
        fixes.append(f"Remove decision or guarantee language (\"{m.group(0)}\"): the tool never "
                     "approves, denies or qualifies anyone.")
    if m := PROTECTED.search(text):
        fixes.append(f"Remove the reference to a protected characteristic (\"{m.group(0)}\").")
    if m := STEERING.search(text):
        fixes.append(f"Remove steering language (\"{m.group(0)}\"); list options neutrally.")
    return fixes


def review(report: Report, policy_citations: list[Citation]) -> list[str]:
    text = report.markdown
    fixes = check_text(text)

    if DISCLAIMER not in text:                                                      # rule 2
        fixes.append("Add the 'not a lending decision' disclaimer exactly as written.")

    allowed = set()                                                                 # rule 3
    for v in report.facts.values():
        allowed |= _numbers(v)
    for c in report.citations:
        allowed |= _numbers(c.excerpt)
    text_wo_quotes = text
    for c in report.citations:
        text_wo_quotes = text_wo_quotes.replace(c.excerpt, "")
    bad = sorted(n for n in _numbers(text_wo_quotes) if not _matches(n, allowed))
    if report.facts and bad:
        fixes.append("These numbers aren't in the case file: "
                     + ", ".join(f"{n:g}" for n in bad[:5]) + ". Use only values from the facts.")

    known = {(c.source, c.excerpt) for c in policy_citations}                       # rule 4
    invented = [c for c in report.citations if (c.source, c.excerpt) not in known]
    if invented:
        fixes.append(f"{len(invented)} citation(s) don't match the Policy agent's passages; cite only those.")

    for question, sig in (("Can you get approved?", report.approval_signal),        # rule 5
                          ("Can you sustain the payments?", report.sustainability_signal)):
        if f"**{question}** {signal_label(sig)}" not in text:
            fixes.append(f"The '{question}' line must show {signal_label(sig)}.")
    return fixes
