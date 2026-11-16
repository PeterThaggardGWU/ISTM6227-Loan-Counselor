# Risk (planned)

Estimates how likely a new conventional loan is to become seriously delinquent
in its early years, for a borrower who has not taken out the loan yet.

**Status: not built.** The Orchestrator uses a stand-in
(`orchestrator/navigator/agents/risk.py`) until this service exists.

## Plan

- **Model:** an *origination-time* model trained in the MRI project (WSL), as a
  new experiment that reuses MRI's pipeline. MRI's existing models score loans
  already being repaid (at least 6 months old) and use payment history, so they
  can't score a prospective borrower.
  - Observation point: each loan's first month in the Fannie Mae data
  - Inputs: MRI's origination features (credit score, DTI, LTV/CLTV, rate,
    loan amount, purpose, occupancy, property type, first-time buyer, borrowers,
    units, state) plus state unemployment and the rate gap to the market at
    origination; channel and origination year dropped
  - Label: 90+ days delinquent or a foreclosure-track exit within the first
    24 months (horizon to confirm with an event-rate study)
  - Conventional loans only (FHA is out of scope for Risk)
- **Service (this folder):** the exported model in LightGBM's text format, the
  feature list and reason-code settings, a small FastAPI service and an MCP
  server, mirroring `lender_match/`. No training data lives here.
- **Environment:** its own `.venv`, pinned to the library versions the model
  was trained with.
