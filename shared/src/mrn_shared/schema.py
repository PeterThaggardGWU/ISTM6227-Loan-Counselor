"""
Case-file schema for Mortgage Readiness Navigator — DRAFT v0.1.

Every component reads and writes these structures:
  - Intake fills a BorrowerProfile.
  - Each round of the loop is a Scenario: one version of the profile, the figures
    derived from it, and the results each agent computed from it.
  - The CaseFile holds every scenario, so the Explainer can compare rounds and the
    Orchestrator can rerun only the agents a change affects.

Fields marked PROVISIONAL depend on the Risk model, which is still being designed
(an origination-time model trained in MRI; see risk/README.md).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

__all__ = [
    "PropertyType", "Debt", "Borrower", "Finances", "Purchase", "BorrowerProfile",
    "Derived", "LenderOption", "LenderMatchResult", "RiskResult", "Citation",
    "PolicyResult", "Signal", "Report", "Scenario", "CaseFile",
]


# ── Borrower profile (what Intake collects) ───────────────────────────────────

class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"   # detached house
    TOWNHOUSE = "townhouse"           # Fannie Mae codes it as PUD or condo; Intake must ask which
    CONDO = "condo"


class Debt(BaseModel):
    name: str = Field(description="What the debt is, in the borrower's words, e.g. 'car loan'")
    monthly_payment: float = Field(ge=0, description="Required monthly payment, dollars")
    balance: Optional[float] = Field(default=None, ge=0, description="Amount still owed, dollars")


class Borrower(BaseModel):
    num_borrowers: int = Field(default=1, ge=1, le=4)
    credit_score: Optional[int] = Field(
        default=None, ge=300, le=850,
        description="Lowest score among the borrowers. PROVISIONAL: a Risk input; Lender Match "
                    "does not use it (public HMDA data has no credit scores).")
    first_time_buyer: bool = True
    current_state: Optional[str] = Field(default=None, min_length=2, max_length=2,
                                         description="Where the borrower lives now")
    current_county_fips: Optional[str] = Field(default=None, pattern=r"^\d{5}$")


class Finances(BaseModel):
    annual_income: float = Field(ge=0, description="Gross yearly income of all borrowers, dollars")
    monthly_debts: list[Debt] = Field(default_factory=list,
                                      description="Recurring debts that will remain after closing")
    savings: float = Field(default=0, ge=0,
                           description="Cash available for down payment, closing costs and reserves")


class Purchase(BaseModel):
    target_state: str = Field(min_length=2, max_length=2, description="Two-letter state code")
    target_county_fips: Optional[str] = Field(default=None, pattern=r"^\d{5}$")
    target_county_name: Optional[str] = None
    property_type: PropertyType = PropertyType.SINGLE_FAMILY
    purchase_price: float = Field(gt=0, description="Expected price, dollars")
    down_payment: float = Field(ge=0, description="Dollars")
    loan_type: Literal["conventional", "fha"] = Field(
        default="conventional",
        description="Risk assesses conventional loans only (Decision 2); Lender Match handles both.")
    loan_term_months: int = Field(default=360, ge=60, le=480)
    occupancy: Literal["primary"] = "primary"   # scope: buyers purchasing a home to live in
    units: int = Field(default=1, ge=1, le=4)


class BorrowerProfile(BaseModel):
    borrower: Borrower = Field(default_factory=Borrower)
    finances: Finances
    purchase: Purchase


class Derived(BaseModel):
    """Figures computed from a profile by mrn_shared.derive(); never entered by hand."""
    loan_amount: float
    ltv_pct: float
    down_payment_pct: float
    monthly_income: float
    monthly_debt_payments: float
    assumed_rate_pct: Optional[float] = Field(
        default=None, description="Rate used for the payment estimate; None = no estimate")
    est_monthly_principal_interest: Optional[float] = Field(
        default=None, description="Principal and interest only; taxes, insurance, HOA and MI not included yet")
    dti_pct: Optional[float] = Field(
        default=None, description="(debts + estimated P&I) / monthly income; None without a rate")
    cash_after_down_payment: float


# ── Agent results ─────────────────────────────────────────────────────────────

Source = Literal["service", "stub"]


class LenderOption(BaseModel):
    lei: str
    name: Optional[str] = None
    approval_probability: float
    historical_approval_rate: Optional[float] = None
    historical_applications: Optional[int] = None


class LenderMatchResult(BaseModel):
    source: Source
    lenders: list[LenderOption] = Field(default_factory=list)
    best_approval_probability: Optional[float] = None
    note: Optional[str] = None


class RiskResult(BaseModel):
    """PROVISIONAL until the origination-time Risk model exists."""
    source: Source
    assessed: bool
    probability: Optional[float] = None
    tier: Optional[Literal["low", "moderate", "elevated"]] = None
    top_factors: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class Citation(BaseModel):
    source: str = Field(description="e.g. 'Fannie Mae Selling Guide'")
    section: Optional[str] = None
    excerpt: str


class PolicyResult(BaseModel):
    source: Source
    citations: list[Citation] = Field(default_factory=list)


# ── Report and case file ──────────────────────────────────────────────────────

class Signal(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    NOT_ASSESSED = "not_assessed"


class Report(BaseModel):
    approval_signal: Signal = Field(description="'Can you get approved?' — from Lender Match")
    sustainability_signal: Signal = Field(description="'Can you sustain the payments?' — from Risk")
    markdown: str
    citations: list[Citation] = Field(default_factory=list)
    critic_notes: list[str] = Field(default_factory=list)
    revisions: int = 0


class Scenario(BaseModel):
    id: int
    label: str
    parent_id: Optional[int] = None
    changes: dict[str, Any] = Field(default_factory=dict,
                                    description="Dotted field path -> new value, relative to the parent")
    rerun: list[str] = Field(default_factory=list, description="Agents rerun for this scenario")
    profile: BorrowerProfile
    derived: Derived
    lender_match: Optional[LenderMatchResult] = None
    risk: Optional[RiskResult] = None
    policy: Optional[PolicyResult] = None
    report: Optional[Report] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseFile(BaseModel):
    session_id: str
    status: Literal["intake", "running", "awaiting_borrower", "done", "gave_up"] = "intake"
    scenarios: list[Scenario] = Field(default_factory=list)

    def current(self) -> Scenario:
        return self.scenarios[-1]

    def get(self, scenario_id: int) -> Scenario:
        return next(s for s in self.scenarios if s.id == scenario_id)
