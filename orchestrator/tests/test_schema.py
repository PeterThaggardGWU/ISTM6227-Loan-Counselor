"""Shared schema v0.2 validators (bugs B3, B4, B5)."""
import copy

import pytest
from pydantic import ValidationError

from mrn_shared import BorrowerProfile, apply_changes
from sample_profiles import TOWNHOUSE_MONTGOMERY_PUD as P


def profile(**purchase):
    d = copy.deepcopy(P)
    d["purchase"].update(purchase)
    return d


def test_down_payment_above_price_is_rejected():            # B5 / E-I2
    with pytest.raises(ValidationError, match="larger than the purchase price"):
        BorrowerProfile.model_validate(profile(purchase_price=50000, down_payment=60000))


def test_zero_income_is_rejected():                        # B3 / E-I4
    d = copy.deepcopy(P)
    d["finances"]["annual_income"] = 0
    with pytest.raises(ValidationError):
        BorrowerProfile.model_validate(d)


def test_state_change_with_stale_county_is_rejected():     # B4 / E-W2
    base = BorrowerProfile.model_validate(P)
    with pytest.raises(ValidationError, match="not in VA"):
        apply_changes(base, {"purchase.target_state": "VA"})


def test_down_payment_above_savings_is_rejected_at_intake_too():
    with pytest.raises(ValidationError, match="savings"):
        BorrowerProfile.model_validate(profile(down_payment=95000))
