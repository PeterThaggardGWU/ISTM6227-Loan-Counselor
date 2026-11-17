"""Turn profile problems into at most MAX_QUESTIONS plain-English questions.

Deterministic on purpose: the same problems always give the same questions,
so the interview is testable. The LLM's job in Intake is understanding the
answers (extractor.py), not phrasing the questions.
"""
from __future__ import annotations

MAX_QUESTIONS = 3

QUESTIONS = {
    "finances.annual_income": "What is your total yearly income before taxes (all borrowers together)?",
    "finances.savings": "How much do you have saved for the down payment, closing costs and reserves?",
    "finances.monthly_debts": "What monthly debt payments do you have (car loan, student loans, credit cards)?",
    "purchase.target_state": "Which state do you want to buy in (DC, MD or VA)?",
    "purchase.target_county_name": "Which county or city do you want to buy in?",
    "purchase.target_county_fips": "Which county do you want to buy in? It needs to be in the DC metro area.",
    "purchase.purchase_price": "About how much do you expect the home to cost?",
    "purchase.down_payment": "How much do you plan to put down?",
    "purchase.property_type": ("Is the townhouse part of a planned community where you own the land "
                               "(PUD), or is it a condo unit? Your listing or agent can tell you."),
    "purchase.loan_type": "Are you looking at a conventional loan or an FHA loan?",
    "borrower.credit_score": "Do you know your credit score? A range is fine, and you can skip this.",
    "confirm.high_dti": ("Your monthly debts plus the estimated payment come to more than half your income. "
                         "Are the income and debt numbers right?"),
    "confirm.combined_finances": "Do the income and debts you gave include every borrower on the loan?",
    "confirm.unusual_values": "The income or price looks unusual for this area. Can you confirm both numbers?",
}

OUT_OF_AREA = ("The Navigator covers the DC metro area only (DC, nearby Maryland and Northern Virginia). "
               "Which county in that area are you considering? A counselor can also refer you elsewhere.")


def question_for(problem: dict) -> str:
    if problem.get("kind") == "out_of_area":
        return OUT_OF_AREA
    q = QUESTIONS.get(problem["field"])
    if q:
        return q
    return f"Please check {problem['field'].replace('_', ' ')}: {problem['problem']}."


def next_questions(problems: list[dict]) -> list[dict]:
    """Missing fields first, then invalid ones, then confirmations; at most MAX_QUESTIONS."""
    order = {"missing": 0, "out_of_area": 1, "invalid": 2, "confirm": 3}
    seen, out = set(), []
    for p in sorted(problems, key=lambda p: order.get(p.get("kind"), 9)):
        if p["field"] in seen:
            continue
        seen.add(p["field"])
        out.append({"field": p["field"], "question": question_for(p)})
        if len(out) == MAX_QUESTIONS:
            break
    return out
