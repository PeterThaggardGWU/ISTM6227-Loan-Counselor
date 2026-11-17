"""
Case-file schema for Mortgage Readiness Navigator — DRAFT v0.2 (see SCHEMA_VERSION).

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

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.2"

__all__ = [
    "SCHEMA_VERSION", "STATE_FIPS", "PropertyType", "Debt", "Borrower", "Finances", "Purchase", "BorrowerProfile",
    "Derived", "LenderOption", "LenderMatchResult", "RiskResult", "Citation",
    "PolicyResult", "Signal", "Report", "CriticRound", "Scenario", "Event", "Note", "CaseFile",
]

# Two-digit state FIPS codes; a county FIPS starts with its state's code.
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}


# ── Borrower profile (what Intake collects) ───────────────────────────────────

class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"   # detached house
    TOWNHOUSE = "townhouse"           # ambiguous: Fannie Mae codes it as PUD or condo; Intake asks which
    TOWNHOUSE_PUD = "townhouse_pud"   # townhouse in a planned unit development
    TOWNHOUSE_CONDO = "townhouse_condo"  # townhouse owned as a condo unit (project review applies)
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
    annual_income: float = Field(gt=0, description="Gross yearly income of all borrowers, dollars")
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

    @model_validator(mode="after")
    def _consistent(self):
        self.target_state = self.target_state.upper()
        if self.down_payment > self.purchase_price:
            raise ValueError("The down payment can't be larger than the purchase price.")
        if self.target_county_fips:
            state_code = STATE_FIPS.get(self.target_state)
            if state_code and not self.target_county_fips.startswith(state_code):
                raise ValueError(f"County {self.target_county_fips} is not in {self.target_state}; "
                                 f"pick a county in {self.target_state}.")
        return self


class BorrowerProfile(BaseModel):
    borrower: Borrower = Field(default_factory=Borrower)
    finances: Finances
    purchase: Purchase

    @model_validator(mode="after")
    def _down_payment_within_savings(self):
        if self.purchase.down_payment > self.finances.savings:
            raise ValueError("The down payment is larger than the savings entered.")
        return self


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

Source = Literal["service", "stub", "unavailable"]   # unavailable = the agent failed or timed out


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


class CriticRound(BaseModel):
    """One Critic pass over one draft; kept for evaluation."""
    passed: bool
    fixes: list[str] = Field(default_factory=list)
    reviewer: Literal["rules", "rules+llm"] = "rules"


class Report(BaseModel):
    approval_signal: Signal = Field(description="'Can you get approved?' — from Lender Match")
    sustainability_signal: Signal = Field(description="'Can you sustain the payments?' — from Risk")
    markdown: str
    citations: list[Citation] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict,
                                  description="Every value the report may state (explainer/facts.py)")
    writer: Literal["template", "llm"] = "template"
    status: Literal["draft", "passed", "needs_review", "approved"] = "draft"
    critic_notes: list[str] = Field(default_factory=list)
    critic_rounds: list[CriticRound] = Field(default_factory=list)
    revisions: int = 0
    original_markdown: Optional[str] = Field(default=None, description="Set when a counselor edits the report")


class Scenario(BaseModel):
    id: int
    label: str
    kind: Literal["baseline", "what_if", "suggested"] = "baseline"
    origin: Literal["borrower", "counselor", "system"] = "borrower"
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


class Event(BaseModel):
    """One entry in the case's append-only audit log."""
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: str = Field(description="e.g. intake_complete, ran_agents, agent_unavailable, critic_failed, "
                                  "escalated, counselor_action, what_if_rejected, stopped")
    detail: dict[str, Any] = Field(default_factory=dict)


class Note(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scenario_id: Optional[int] = None
    text: str


EndedReason = Literal["done", "gave_up", "max_rounds", "no_progress", "intake_handoff"]


class CaseFile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    status: Literal["intake", "running", "awaiting_borrower", "needs_review",
                    "done", "gave_up", "signed_off"] = "intake"
    ended_reason: Optional[EndedReason] = None
    scenarios: list[Scenario] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    counselor_notes: list[Note] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)

    def log(self, kind: str, **detail) -> None:
        self.events.append(Event(kind=kind, detail=detail))

    def current(self) -> Scenario:
        return self.scenarios[-1]

    def get(self, scenario_id: int) -> Scenario:
        return next(s for s in self.scenarios if s.id == scenario_id)
