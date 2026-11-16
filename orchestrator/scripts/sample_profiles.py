"""Fictional borrowers for demos and tests."""

TOWNHOUSE_MONTGOMERY = {
    "borrower": {"num_borrowers": 1, "credit_score": 715, "first_time_buyer": True,
                 "current_state": "DC"},
    "finances": {"annual_income": 120000, "savings": 90000,
                 "monthly_debts": [{"name": "car loan", "monthly_payment": 776, "balance": 18000},
                                   {"name": "student loan", "monthly_payment": 250}]},
    "purchase": {"target_state": "MD", "target_county_fips": "24031",
                 "target_county_name": "Montgomery County", "property_type": "townhouse",
                 "purchase_price": 450000, "down_payment": 67500},
}
