# Mortgage Readiness Navigator

A multi-agent assistant that helps first-time homebuyers and their housing
counselors answer two questions before anyone applies for a mortgage:
**Can this borrower get approved?** and **Can they sustain the payments?**

An Orchestrator agent runs a loop with the borrower. Intake interviews the
borrower and builds a structured profile. Lender Match, Risk and Policy then
work in parallel, and an Explainer writes a plain-English report that a
Critic checks before anyone sees it. The borrower can then play "what-if"
(more down payment, a paid-off car loan, a different county); the
Orchestrator reruns only the agents each change affects and reports what
moved. The loop ends when the borrower is done, with a counselor signing off
on the final report.

ISTM 6227, Fall 2026 · Peter Thaggard, Kushi Khandoji, Kritika Berry,
Alekya Kowta

---

## Components

| Folder | What it is | Status |
|---|---|---|
| [`lender_match/`](lender_match/README.md) | HMDA-trained lender-recommendation service and its MCP server, formerly **Loan Arranger** | Working |
| [`orchestrator/`](orchestrator/README.md) | The Orchestrator loop (LangGraph) and its agents: Intake, Explainer, Critic, the borrower and counselor steps, and the action plan; it calls Lender Match, Risk and Policy | Loop, escalation and stopping rules done; LLM agents with rule-based fallbacks; Risk and Policy are still stand-ins |
| [`shared/`](shared/README.md) | The borrower-profile and case-file schema that every component uses | Draft v0.2 |
| [`risk/`](risk/README.md) | Early-delinquency model for a new conventional loan, trained in the MRI project, behind an MCP server | Planned |
| [`policy/`](policy/README.md) | Guideline sources for retrieval (RAG): the Selling Guide, Servicing Guide, Regulation B and HMDA rules | Sources only |
| `docs/` | Proposal and design documents | — |

## One virtual environment per program

Each component that runs as its own program has its own virtual
environment and its own `requirements.txt` inside its folder:

- `lender_match/.venv` holds PyTorch, LightGBM 4.6.0 and scikit-learn 1.8.0.
  The pickled models load reliably only with these exact versions.
- `risk/.venv` will pin whatever versions the MRI model was trained with.
- `orchestrator/.venv` holds the agent framework and the LLM SDK, which
  every agent shares because they all run in the Orchestrator's process.

The components talk to each other over MCP and HTTP, not Python imports, so
separate environments cost nothing. `shared/` will be installed into each
environment with `pip install -e ../shared`. Never commit a `.venv` folder.

## Getting started

```bash
git clone https://github.com/PeterThaggardGWU/ISTM6227-Loan-Counselor.git
cd ISTM6227-Loan-Counselor
```

Then follow the README in each component's folder:

1. [`lender_match/README.md`](lender_match/README.md) sets up its environment
   and starts the service on <http://localhost:8000>.
2. [`orchestrator/README.md`](orchestrator/README.md) sets up the Orchestrator,
   runs a demo session in the terminal, and opens the graph in LangGraph Studio.

## Large files

Raw data and large intermediate files stay out of Git; see `.gitignore`.
Each component's README explains how to download or rebuild what it needs.
Every pattern in `.gitignore` that contains a `/` is matched from the
repository root, so a new component's ignore rules must start with its
folder name (for example, `risk/data/*`).
