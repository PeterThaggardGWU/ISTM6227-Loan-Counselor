"""Problems with a profile that field validation alone can't catch.

Two kinds of problem, both returned as dicts the question writer understands:
- blocking: the borrower must change an answer (e.g. a county outside the service area)
- confirm:  the value is allowed but unusual; the borrower confirms it once
            (answers are stored under profile_input["confirm"][<key>] = True)
"""
from __future__ import annotations

from pydantic import ValidationError

from mrn_shared import BorrowerProfile, PropertyType, derive

from .. import config


def schema_problems(raw: dict) -> tuple[BorrowerProfile | None, list[dict]]:
    """Validate against the shared schema; return the profile or the field problems."""
    try:
        return BorrowerProfile.model_validate(raw), []
    except ValidationError as e:
        problems = []
        for err in e.errors():
            field = ".".join(str(p) for p in err["loc"])
            if err["type"] == "value_error" and field in ("", "purchase", "finances", "borrower"):
                field = _field_from_message(err["msg"])     # cross-field rule: point at the field to fix
            problems.append({"field": field, "problem": err["msg"].removeprefix("Value error, "),
                             "kind": "missing" if err["type"] == "missing" else "invalid"})
        return None, problems


def _field_from_message(msg: str) -> str:
    """Cross-field validators have no location; map their message to the field to ask about."""
    m = msg.lower()
    if "county" in m:
        return "purchase.target_county_fips"
    if "savings" in m:
        return "finances.savings"
    if "down payment" in m:
        return "purchase.down_payment"
    return "profile"


def consistency_problems(profile: BorrowerProfile, raw: dict) -> list[dict]:
    confirmed = raw.get("confirm") or {}
    p, f, b = profile.purchase, profile.finances, profile.borrower
    problems: list[dict] = []

    if p.property_type == PropertyType.TOWNHOUSE:
        problems.append({"field": "purchase.property_type", "kind": "invalid",
                         "problem": "townhouse ownership type unknown (PUD or condo)"})
    if config.SERVICE_AREA_ENFORCED and p.target_county_fips and p.target_county_fips not in config.DC_METRO_FIPS:
        problems.append({"field": "purchase.target_county_fips", "kind": "out_of_area",
                         "problem": "outside the DC metro service area"})
    if p.target_county_fips is None and p.target_county_name is None:
        problems.append({"field": "purchase.target_county_name", "kind": "missing",
                         "problem": "target county missing"})

    d = derive(profile, config.ASSUMED_RATE_PCT)
    if d.dti_pct is not None and d.dti_pct > 50 and not confirmed.get("high_dti"):
        problems.append({"field": "confirm.high_dti", "kind": "confirm",
                         "problem": f"estimated debt-to-income {d.dti_pct}% is above 50%"})
    if b.num_borrowers > 1 and not confirmed.get("combined_finances"):
        problems.append({"field": "confirm.combined_finances", "kind": "confirm",
                         "problem": "co-borrowers: income and debts must be combined"})
    if (f.annual_income > 2_000_000 or p.purchase_price < 75_000) and not confirmed.get("unusual_values"):
        problems.append({"field": "confirm.unusual_values", "kind": "confirm",
                         "problem": "income or price is unusual for this area"})
    return problems


def all_problems(raw: dict) -> tuple[BorrowerProfile | None, list[dict]]:
    profile, problems = schema_problems(raw)
    if profile is None:
        return None, problems
    return profile, consistency_problems(profile, raw)
