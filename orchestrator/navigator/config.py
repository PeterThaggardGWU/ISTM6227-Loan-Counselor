"""Settings, read from the environment (orchestrator/.env when present)."""
from __future__ import annotations

import os

try:  # python-dotenv is optional; langgraph dev also loads .env itself
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


# Where the Lender Match service runs. Empty = use the stand-in agent.
LENDER_MATCH_URL = os.environ.get("LENDER_MATCH_URL", "").strip().rstrip("/")

# Interest rate for the payment and DTI estimate. PLACEHOLDER: replace with a
# current market rate (e.g. Freddie Mac PMMS). Empty = no payment estimate.
ASSUMED_RATE_PCT = _float("ASSUMED_RATE_PCT", None)

# Loop limits (stopping rules from the design review)
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "6"))        # scenarios per session
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "2"))  # Explainer-Critic rounds per report (design §7.6)

# Traffic-light thresholds for "Can you get approved?" (best lender approval odds).
# PLACEHOLDERS for the team to set.
APPROVAL_GREEN = _float("APPROVAL_GREEN", 0.80)
APPROVAL_YELLOW = _float("APPROVAL_YELLOW", 0.60)

# Minimum historical applications for a lender to count toward the approval signal (D8).
MIN_LENDER_APPLICATIONS = int(os.environ.get("MIN_LENDER_APPLICATIONS", "200"))

# Escalation and stopping rules (design.md §7.7, §7.8)
MAX_INTAKE_TURNS = int(os.environ.get("MAX_INTAKE_TURNS", "8"))      # then a counselor takes over
NO_PROGRESS_ROUNDS = int(os.environ.get("NO_PROGRESS_ROUNDS", "2"))  # unchanged rounds before offering to wrap up
MATERIAL_ODDS_CHANGE = _float("MATERIAL_ODDS_CHANGE", 0.02)          # 2 points of approval odds
SPECIALIST_TIMEOUT_S = _float("SPECIALIST_TIMEOUT_S", 30.0)          # per specialist call (join timeout)

# LLM (Claude). NAVIGATOR_LLM: "on", "off", or "auto" (on when ANTHROPIC_API_KEY is set).
# Tests force "off"; every LLM agent then uses its rule-based fallback.
NAVIGATOR_LLM = os.environ.get("NAVIGATOR_LLM", "auto").strip().lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-5-5")            # Intake, Explainer, action plan
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "claude-sonnet-5-5")    # different model reviews the work (D6)
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 60.0)


def llm_enabled() -> bool:
    if NAVIGATOR_LLM == "on":
        return True
    if NAVIGATOR_LLM == "off":
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

# Service area (D12): DC metro county FIPS codes. SERVICE_AREA=any turns the check off.
DC_METRO_FIPS = frozenset({
    "11001",                                   # District of Columbia
    "24031", "24033", "24021", "24017",        # MD: Montgomery, Prince George's, Frederick, Charles
    "51013", "51059", "51107", "51153",        # VA: Arlington, Fairfax, Loudoun, Prince William
    "51510", "51610", "51600", "51683", "51685",  # VA cities: Alexandria, Falls Church, Fairfax, Manassas, Manassas Park
})
SERVICE_AREA_ENFORCED = os.environ.get("SERVICE_AREA", "dc_metro").strip().lower() != "any"
