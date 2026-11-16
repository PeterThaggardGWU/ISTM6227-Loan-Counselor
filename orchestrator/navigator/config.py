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
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "3"))  # Explainer-Critic rounds per report

# Traffic-light thresholds for "Can you get approved?" (best lender approval odds).
# PLACEHOLDERS for the team to set.
APPROVAL_GREEN = _float("APPROVAL_GREEN", 0.80)
APPROVAL_YELLOW = _float("APPROVAL_YELLOW", 0.60)
