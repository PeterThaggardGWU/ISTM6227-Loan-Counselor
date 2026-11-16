"""
schemas.py
Loan Arranger - FastAPI Request / Response Schemas
ISTM6218 Group Project
"""

from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class LoanPurpose(IntEnum):
    PURCHASE          = 1
    HOME_IMPROVEMENT  = 2
    REFINANCE_CASHOUT = 31
    REFINANCE_NOCASH  = 32
    OTHER             = 4


class LoanType(IntEnum):
    CONVENTIONAL = 1
    FHA          = 2
    VA           = 3
    USDA         = 4


class LienStatus(IntEnum):
    FIRST_LIEN  = 1
    SUBORDINATE = 2


class OccupancyType(IntEnum):
    PRINCIPAL_RESIDENCE = 1
    INVESTMENT          = 2
    SECOND_HOME         = 3


class BorrowerProfile(BaseModel):
    """
    Input profile submitted by the borrower (or extracted from natural
    language by Claude and passed through the MCP server).
    Required fields are the minimum necessary to produce a recommendation.
    Optional fields improve accuracy when provided.
    """

    # Geography
    state_code: str = Field(
        ..., min_length=2, max_length=2,
        description="Two-letter state code e.g. 'TX'",
        examples=["TX", "VA", "CA"],
    )
    county_code: Optional[str] = Field(
        default=None,
        description="5-digit FIPS county code e.g. '48029' for Bexar County TX. "
                    "If omitted, results are filtered by state only. "
                    "Provide this OR county_name/city — not required if those are given.",
        examples=["48029", "51013"],
    )
    county_name: Optional[str] = Field(
        default=None,
        description="County name e.g. 'Bexar County' or 'Bexar'. "
                    "Resolved to a FIPS code automatically. "
                    "Use instead of county_code if the FIPS is not known.",
        examples=["Bexar County", "Fairfax", "Los Angeles County"],
    )
    city: Optional[str] = Field(
        default=None,
        description="City name e.g. 'San Antonio'. "
                    "Resolved to the containing county automatically. "
                    "Use when the borrower provides a city rather than a county name.",
        examples=["San Antonio", "Arlington", "Denver"],
    )

    # Loan characteristics
    loan_amount: float = Field(
        ..., gt=0,
        description="Requested loan amount in dollars",
        examples=[350000, 500000],
    )
    loan_purpose: LoanPurpose = Field(
        default=LoanPurpose.PURCHASE,
        description="Purpose of the loan",
    )
    loan_type: LoanType = Field(
        default=LoanType.CONVENTIONAL,
        description="Loan type (conventional, FHA, VA, USDA)",
    )
    lien_status: LienStatus = Field(
        default=LienStatus.FIRST_LIEN,
        description="Lien position",
    )
    occupancy_type: OccupancyType = Field(
        default=OccupancyType.PRINCIPAL_RESIDENCE,
        description="Intended occupancy",
    )

    # Borrower financials (required)
    income: float = Field(
        ..., gt=0,
        description="Annual gross income in dollars",
        examples=[120000, 85000],
    )

    # Borrower financials (optional but recommended)
    property_value: Optional[float] = Field(
        default=None, gt=0,
        description="Estimated property value in dollars",
        examples=[425000, 600000],
    )
    debt_to_income_ratio: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Debt-to-income ratio as a percentage e.g. 35.0",
        examples=[35.0, 42.0],
    )

    # Demographics (optional, Model A only, user opt-in)
    derived_race: Optional[str] = Field(
        default=None,
        description="Self-reported race. Used only with model_variant='A'. "
                    "Ignored by the default compliant Model B.",
        examples=["White", "Black or African American", "Asian"],
    )
    derived_ethnicity: Optional[str] = Field(
        default=None,
        description="Self-reported ethnicity (Model A only).",
        examples=["Hispanic or Latino", "Not Hispanic or Latino"],
    )
    derived_sex: Optional[str] = Field(
        default=None,
        description="Self-reported sex (Model A only).",
        examples=["Male", "Female", "Joint"],
    )
    applicant_age: Optional[str] = Field(
        default=None,
        description="Age bracket (Model A only).",
        examples=["25-34", "35-44", "45-54"],
    )

    # Request options
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of lender recommendations to return",
    )
    model_variant: str = Field(
        default="B",
        pattern="^[AaBb]$",
        description="'B' = default ECOA-compliant model. "
                    "'A' = includes demographics, academic use only.",
    )
    min_applications: int = Field(
        default=50, ge=1,
        description="Minimum historical applications a lender must have "
                    "in the target geography to appear in results.",
    )

    @field_validator("state_code")
    @classmethod
    def uppercase_state(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("model_variant")
    @classmethod
    def uppercase_variant(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def demographics_silently_ignored_for_model_b(self) -> "BorrowerProfile":
        """Demographics are silently ignored when model_variant is B."""
        return self


class LenderResult(BaseModel):
    """A single lender recommendation with predicted approval probability."""

    rank: int = Field(
        description="Rank in the list (1 = most likely to approve)"
    )
    lei: str = Field(
        description="Legal Entity Identifier of the lender"
    )
    approval_probability: float = Field(
        description="Predicted probability of approval (0.0-1.0)",
        ge=0.0, le=1.0,
    )
    historical_approval_rate: Optional[float] = Field(
        default=None,
        description="Historical approval rate for this lender in the "
                    "target geography (from training data)",
        ge=0.0, le=1.0,
    )
    historical_application_count: Optional[int] = Field(
        default=None,
        description="Total applications from this lender in the target "
                    "geography across all training years",
    )
    lender_name: Optional[str] = Field(
        default=None,
        description="Human-readable lender name if available in metadata",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level based on data volume: "
                    "'high' (1000+ apps), 'medium' (200-999), 'low' (50-199)",
    )


class RecommendResponse(BaseModel):
    """
    Full recommendation response from the /recommend endpoint.
    Rich enough for Claude to compose a useful natural language narrative.
    """
    recommendations: list[LenderResult] = Field(
        description="Ranked lender recommendations, most likely first"
    )
    request_summary: dict = Field(
        description="Echo of key request parameters used for scoring"
    )
    model_variant_used: str = Field(
        description="Model variant that produced these results ('A' or 'B')"
    )
    lenders_evaluated: int = Field(
        description="Total lenders evaluated before returning top_k"
    )
    geography_filter: dict = Field(
        description="Geographic filter applied: state and county if provided"
    )
    compliance_note: str = Field(
        default=(
            "Recommendations are based on historical HMDA approval patterns "
            "and do not guarantee approval. Model B (default) does not use "
            "race, ethnicity, sex, or other protected characteristics."
        ),
    )


class HealthResponse(BaseModel):
    status: str
    model_a_loaded: bool
    model_b_loaded: bool
    embeddings_loaded: bool
    lender_count: int
    training_years: Optional[list[int]] = None
