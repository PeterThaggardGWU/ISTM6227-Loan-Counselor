# Policy (planned)

Retrieves the guideline passages that apply to a borrower's situation, so the
Explainer can cite them.

**Status: sources only.** The Orchestrator uses a stand-in
(`orchestrator/navigator/agents/policy.py`).

- `sources/` — the documents to index: the Fannie Mae Selling Guide and
  Servicing Guide (Regulation B and HMDA rules still to add)
- Next: chunk and index the sources in a vector store, and replace the stand-in
  with retrieval that returns `Citation` objects (`mrn_shared.schema`)
