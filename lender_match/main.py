"""
main.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — FastAPI Inference Service
ISTM6218 Group Project

Exposes the trained LightGBM models as a REST API.  Designed to run on
localhost:8000 for development / demonstration and on AWS ECS for production.

Endpoints:
  GET  /health              — liveness check, reports loaded artifact status
  POST /recommend           — main recommendation endpoint
  GET  /lenders/{state}     — list lenders active in a state (for exploration)
  GET  /docs                — auto-generated Swagger UI (FastAPI built-in)

Startup sequence:
  1. ModelStore.load() — loads all artifacts into memory (~10-30s)
  2. Uvicorn begins accepting requests

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  python main.py   (runs uvicorn directly)
────────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    BorrowerProfile,
    HealthResponse,
    LenderResult,
    RecommendResponse,
)
from model_loader import ModelStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global model store ────────────────────────────────────────────────────────
# Instantiated once at startup, shared across all requests.
store = ModelStore()


# ── Lifespan — load artifacts before accepting requests ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts at startup; nothing to clean up at shutdown."""
    log.info("Starting up — loading model artifacts …")
    t0 = time.time()
    try:
        store.load()
        log.info(f"Startup complete in {time.time() - t0:.1f}s — ready for requests.")
    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="The Loan Arranger",
    description=(
        "AI-powered mortgage lender recommendation service. "
        "Given a borrower's financial profile and target location, returns a "
        "ranked list of lenders most likely to approve their application, "
        "based on historical HMDA approval patterns (2019–2024)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the MCP server (running on the same machine) to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def profile_to_dict(profile: BorrowerProfile) -> dict:
    """
    Convert a BorrowerProfile to the flat dict that ModelStore.score() expects.
    Field names match the HMDA column names used during training.
    """
    return {
        # Continuous features
        "loan_amount":          profile.loan_amount,
        "income":               profile.income,
        "property_value":       profile.property_value,
        "debt_to_income_ratio": profile.debt_to_income_ratio,
        # Integer categoricals
        "loan_purpose":         int(profile.loan_purpose),
        "loan_type":            int(profile.loan_type),
        "lien_status":          int(profile.lien_status),
        "occupancy_type":       int(profile.occupancy_type),
        # Demographics (only used if model_variant == "A")
        "derived_race":         profile.derived_race,
        "derived_ethnicity":    profile.derived_ethnicity,
        "derived_sex":          profile.derived_sex,
        "applicant_age":        profile.applicant_age,
    }


def build_request_summary(profile: BorrowerProfile) -> dict:
    """Build a human-readable summary of the request for the response payload."""
    summary = {
        "state":          profile.state_code,
        "county":         profile.county_code or "not specified (state-level filter)",
        "loan_amount":    f"${profile.loan_amount:,.0f}",
        "income":         f"${profile.income:,.0f}",
        "loan_purpose":   profile.loan_purpose.name.replace("_", " ").title(),
        "loan_type":      profile.loan_type.name.replace("_", " ").title(),
        "model_variant":  profile.model_variant,
    }
    if profile.property_value:
        summary["property_value"] = f"${profile.property_value:,.0f}"
    if profile.debt_to_income_ratio:
        summary["dti_ratio"] = f"{profile.debt_to_income_ratio:.1f}%"
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Returns the current status of the service and which model artifacts
    are loaded.  The MCP server calls this on startup to confirm the
    inference service is ready before registering the tool with Claude.
    """
    h = store.health_dict()
    return HealthResponse(**h)


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get lender recommendations for a borrower profile",
    tags=["Recommendations"],
)
async def recommend(profile: BorrowerProfile) -> RecommendResponse:
    """
    Core recommendation endpoint.

    Accepts a borrower profile (financial info + location), scores all
    lenders active in the target geography using the trained LightGBM model,
    and returns a ranked list of the top-K lenders by predicted approval
    probability.

    **Model variants:**
    - `model_variant="B"` (default): ECOA-compliant model, does not use
      race, ethnicity, sex, or age as inputs.
    - `model_variant="A"`: Includes demographic features. For academic /
      analysis purposes only.

    **Geography filtering:**
    Lenders are filtered to those with at least `min_applications` historical
    applications in the specified state (and county, if provided).  If no
    county-level lenders are found, the service automatically falls back to
    state-level filtering.
    """
    if not store.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service is still loading model artifacts. Retry in a few seconds."
        )

    t0 = time.time()
    log.info(
        f"POST /recommend  state={profile.state_code}  "
        f"county={profile.county_code}  variant={profile.model_variant}  "
        f"loan=${profile.loan_amount:,.0f}  income=${profile.income:,.0f}"
    )

    # Resolve county_code from county_name or city if not directly provided
    county_code = profile.county_code
    if not county_code and (profile.county_name or profile.city):
        county_code = store.resolve_county_code(
            state_code  = profile.state_code,
            county_name = profile.county_name,
            city        = profile.city,
        )
        if county_code:
            log.info(f"  Resolved geography input -> county_code={county_code}")
        else:
            log.info(
                f"  Could not resolve county from "
                f"county_name={profile.county_name!r} city={profile.city!r} — "
                f"falling back to state-level filter"
            )

    # Convert profile to feature dict
    profile_dict = profile_to_dict(profile)

    try:
        results, n_evaluated = store.score(
            profile_dict    = profile_dict,
            state_code      = profile.state_code,
            county_code     = county_code,
            model_variant   = profile.model_variant,
            top_k           = profile.top_k,
            min_applications= profile.min_applications,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error.")

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No lenders found in {profile.state_code}"
                + (f" county {profile.county_code}" if profile.county_code else "")
                + f" with at least {profile.min_applications} historical applications. "
                "Try reducing min_applications or broadening the geography."
            )
        )

    elapsed = time.time() - t0
    log.info(
        f"  → {len(results)} recommendations from {n_evaluated} candidates "
        f"in {elapsed*1000:.0f}ms"
    )

    return RecommendResponse(
        recommendations   = [LenderResult(**r) for r in results],
        request_summary   = build_request_summary(profile),
        model_variant_used= profile.model_variant,
        lenders_evaluated = n_evaluated,
        geography_filter  = {
            "state":  profile.state_code,
            "county": profile.county_code,
        },
    )


@app.get(
    "/lenders/{state_code}",
    summary="List lenders active in a state",
    tags=["Exploration"],
)
async def list_lenders(
    state_code: str,
    county_code: Optional[str] = Query(
        default=None,
        description="Filter to a specific county (5-digit FIPS code)"
    ),
    min_applications: int = Query(
        default=100,
        ge=1,
        description="Minimum historical applications"
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of lenders to return"
    ),
) -> JSONResponse:
    """
    Returns a list of lenders with historical activity in the specified
    geography, sorted by number of applications descending.  Useful for
    exploring which lenders are active in a market before making a
    recommendation request.
    """
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Service is still loading.")

    candidates = store.get_lenders_in_geography(
        state_code.upper(), county_code, min_applications
    )

    if len(candidates) == 0:
        return JSONResponse(content={
            "state":   state_code.upper(),
            "county":  county_code,
            "lenders": [],
            "count":   0,
        })

    top = (
        candidates
        .sort_values("n_applications", ascending=False)
        .head(limit)
    )

    lenders = [
        {
            "lei":            row["lei"],
            "n_applications": int(row["n_applications"]),
            "approval_rate":  round(float(row["approval_rate"]), 4),
        }
        for _, row in top.iterrows()
    ]

    return JSONResponse(content={
        "state":   state_code.upper(),
        "county":  county_code,
        "lenders": lenders,
        "count":   len(lenders),
    })


@app.get(
    "/",
    include_in_schema=False,
)
async def root():
    return {
        "service":  "The Loan Arranger",
        "version":  "1.0.0",
        "status":   "ok" if store.is_loaded else "loading",
        "docs":     "http://localhost:8000/docs",
        "health":   "http://localhost:8000/health",
        "recommend":"http://localhost:8000/recommend",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — run directly with: python main.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
