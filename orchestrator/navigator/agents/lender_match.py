"""Lender Match: ranks lenders and their approval odds for the current scenario.

Calls the Lender Match service (lender_match/main.py, POST /recommend) when
LENDER_MATCH_URL is set; otherwise, or if the service can't be reached, returns
STAND-IN results so the loop keeps running. A later version can reach the same
service through its MCP server (langchain-mcp-adapters) when an LLM needs to
choose the tool calls itself.
"""
from __future__ import annotations

import httpx

from mrn_shared import LenderMatchResult, LenderOption

from .. import config
from ._common import current, dump


def _payload(scen) -> dict:
    p, d = scen.profile.purchase, scen.derived
    body = {
        "state_code": p.target_state.upper(),
        "loan_amount": d.loan_amount,
        "income": scen.profile.finances.annual_income,
        "property_value": p.purchase_price,
        "loan_purpose": 1,                                   # purchase
        "loan_type": 2 if p.loan_type == "fha" else 1,
        "lien_status": 1,
        "occupancy_type": 1,                                 # principal residence
        "top_k": 5,
        "model_variant": "B",                                # never uses protected traits
    }
    if p.target_county_fips:
        body["county_code"] = p.target_county_fips
    elif p.target_county_name:
        body["county_name"] = p.target_county_name
    if d.dti_pct is not None:
        body["debt_to_income_ratio"] = min(d.dti_pct, 100.0)
    return body


def _from_service(scen) -> LenderMatchResult:
    r = httpx.post(f"{config.LENDER_MATCH_URL}/recommend", json=_payload(scen), timeout=30)
    r.raise_for_status()
    recs = r.json().get("recommendations", [])
    lenders = [LenderOption(
        lei=x["lei"], name=x.get("lender_name"),
        approval_probability=x["approval_probability"],
        historical_approval_rate=x.get("historical_approval_rate"),
        historical_applications=x.get("historical_application_count"),
    ) for x in recs]
    return LenderMatchResult(
        source="service", lenders=lenders,
        best_approval_probability=max((l.approval_probability for l in lenders), default=None))


def _stand_in(scen, note: str) -> LenderMatchResult:
    """STAND-IN numbers: a simple formula, not the model."""
    d = scen.derived
    base = 0.92 - 0.004 * max(d.ltv_pct - 80, 0) - 0.006 * max((d.dti_pct or 36) - 36, 0)
    lenders = [LenderOption(lei=f"STUB{i:016d}", name=f"Example Lender {chr(64 + i)}",
                            approval_probability=round(max(base - 0.03 * (i - 1), 0.05), 3))
               for i in range(1, 6)]
    return LenderMatchResult(source="stub", lenders=lenders,
                             best_approval_probability=lenders[0].approval_probability, note=note)


def lender_match(state):
    scen = current(state)
    if config.LENDER_MATCH_URL:
        try:
            result = _from_service(scen)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            result = _stand_in(scen, f"Lender Match service unavailable ({type(e).__name__}); stand-in results")
    else:
        result = _stand_in(scen, "LENDER_MATCH_URL not set; stand-in results")
    return {"lender_match": dump(result)}
