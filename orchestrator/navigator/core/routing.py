"""Which agents a what-if change affects ("rerun only what changed").

Keys are dotted profile paths (see mrn_shared.schema). A change to a field not
listed here reruns every agent, which is always safe. The Explainer and Critic
run every round regardless.
"""
from __future__ import annotations

from typing import Iterable

AGENTS = ("lender_match", "risk", "policy")
ALL = frozenset(AGENTS)

AFFECTS: dict[str, frozenset[str]] = {
    "purchase.purchase_price":     ALL,                              # LTV: mortgage insurance rules
    "purchase.down_payment":       ALL,                              # LTV: mortgage insurance rules
    "purchase.target_state":       ALL,
    "purchase.target_county_fips": ALL,
    "purchase.target_county_name": ALL,
    "purchase.property_type":      frozenset({"risk", "policy"}),   # e.g. condo project rules
    "purchase.loan_type":          ALL,
    "purchase.loan_term_months":   frozenset({"lender_match", "risk"}),
    "finances.annual_income":      frozenset({"lender_match", "risk"}),
    "finances.monthly_debts":      frozenset({"lender_match", "risk"}),
    "finances.remove_debt":        frozenset({"lender_match", "risk"}),
    "finances.savings":            frozenset(),                      # Explainer only (reserves)
    "borrower.credit_score":       frozenset({"risk"}),              # HMDA data has no scores
    "borrower.num_borrowers":      frozenset({"risk"}),
    "borrower.first_time_buyer":   frozenset({"risk", "policy"}),
}


# What a borrower (or counselor) may change in a what-if. Anything else, including
# where the borrower lives now and any field that could carry a protected trait,
# is rejected before a scenario is created (design.md E-W5, §10.1).
EDITABLE_PATHS: frozenset[str] = frozenset(AFFECTS)


def agents_to_rerun(changed_paths: Iterable[str]) -> list[str]:
    needed: set[str] = set()
    for path in changed_paths:
        needed |= AFFECTS.get(path, ALL)
    return [a for a in AGENTS if a in needed]   # stable order
