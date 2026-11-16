# shared — the case-file schema

The borrower profile, scenario and case-file definitions that every component
uses (`mrn_shared`). Draft v0.1: fields marked PROVISIONAL depend on the Risk
model, which is still being designed.

- `schema.py` — the data structures (Pydantic models)
- `derive.py` — loan amount, LTV, DTI and other figures computed from a profile
- `changes.py` — applies what-if changes ("purchase.down_payment": 80000,
  "finances.remove_debt": "car loan") and re-validates the result

Install it into a component's virtual environment from that component's folder:

```bash
pip install -e ../shared
```
