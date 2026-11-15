# Loan Arranger

An AI-assisted tool that points a mortgage borrower toward the lenders most
likely to approve their application. It was trained on six years of public
HMDA (Home Mortgage Disclosure Act) data, 2019–2024: about 76 million
applications from roughly 6,100 lenders.

Give it a borrower profile (location, loan amount, income, loan type and so
on) and it returns a ranked list of lenders. Each lender comes with a
predicted approval probability and its historical approval rate in that
area. It runs as a REST service, and an MCP adapter lets Claude Desktop call
it in conversation.

---

## How it works

The model has two stages:

1. **Embedding network** (`02_train_embeddings.py`). A PyTorch neural
   network learns a compact numeric profile ("embedding") for every lender
   (64 numbers), census tract (32) and county (16) from historical approval
   patterns. Only the learned profiles are kept; the network itself is not
   used to make predictions.
2. **LightGBM model** (`03_train_lgbm.py`). A gradient-boosted tree model
   takes those profiles plus the borrower's own features and predicts the
   probability of approval. This is the model the service runs.

There are two versions of the LightGBM model:

| Variant | Uses applicant race, ethnicity, sex, age? | Purpose |
|---|---|---|
| **B** (default) | No | Recommendations; intended to align with ECOA / Fair Housing Act limits |
| **A** | Yes | Academic and fairness analysis only |

At request time, the service finds the lenders active in the borrower's
state (and county, if one is given), scores the borrower against each
lender, and returns the top results.

```
Claude Desktop ──(MCP, stdio)──> mcp_server/server.py ──(HTTP)──> main.py (FastAPI)
                                                                     │
                                                         model_loader.py loads
                                                         models/ + data/ files
```

---

## Repository layout

| Path | What it is |
|---|---|
| `main.py` | FastAPI service: `/health`, `/recommend`, `/lenders/{state}`, `/docs` |
| `model_loader.py` | Loads the models and lookup files at startup; holds the scoring logic |
| `schemas.py` | Request and response definitions (`BorrowerProfile`, `LenderResult`, …) |
| `mcp_server/server.py` | Thin MCP adapter that forwards Claude Desktop tool calls to the service |
| `payloads/san-antonio.json` | Sample `/recommend` request |
| `01_preprocess.py` | Training step 1: cleans one year of raw HMDA data into parquet |
| `01b_merge.py` | Training step 2: merges all years and builds the shared lookup indexes |
| `build_lender_names.py` | Builds `data/lender_names.parquet` (lender ID → lender name) |
| `build_county_lookup.py` | Builds `data/county_fips.parquet` (county name → FIPS code) |
| `build_lender_meta.py` | Builds `data/lender_meta.parquet` (approval rate and active counties per lender per state) |
| `02_train_embeddings.py` | Training step 3: embedding network |
| `03_train_lgbm.py` | Training step 4: LightGBM models A and B |
| `compute_fairness.py` | Disparate-impact analysis of Model A |
| `list_columns.py` | Tiny helper that prints the columns of `hmda_all_clean.parquet` |
| `models/` | Trained models and training reports (see below) |
| `data/` | Lookup files the service needs; raw and intermediate data are **not** in Git |
| `training_log.json`, `training-notes-log.docx`, `GPU-in-action.png` | Notes and records from the original training run |

### What's in `models/`

| File | Role | Used by the service? |
|---|---|---|
| `lgbm_model_a.pkl`, `lgbm_model_b.pkl` | The trained LightGBM models (joblib pickles) | **Yes** |
| `embedding_weights.pt` | Learned lender, tract and county embeddings | **Yes** |
| `lgbm_training_summary.json` | Metrics, parameters and feature list | No |
| `lgbm_feature_importance_a/b.parquet` | Feature-importance tables | No |
| `fairness_report.json` / `.txt` | Output of `compute_fairness.py` | No |

Not in Git, because the service doesn't need them (each run of the training
scripts regenerates them): `embedding_network.pt` (full training checkpoint)
and `lgbm_model_a/b.txt` (the same models in LightGBM's native text format).

---

## Quick start: run the service

You don't need the raw data or a GPU to run the service. Everything it loads
is in the repository (about 53 MB).

**Requires Python 3.11.** The commands below are for Windows PowerShell; on
macOS or Linux use `source .venv/bin/activate` and `python3`.

```powershell
git clone <repo-url> loan-arranger
cd loan-arranger
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` asks for PyTorch, and a plain `pip install` gets the CPU
build, which is all the service needs. You only need the CUDA build for
retraining (see the note at the top of `requirements.txt`).

Start the service from the repository root:

```powershell
python main.py
```

Startup can take up to half a minute while the models load. Then:

- Interactive API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>. This should report both
  models and the embeddings as loaded, with 6,103 lenders and training years
  2019–2024.

### Try a recommendation

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/recommend `
  -ContentType 'application/json' -InFile payloads\san-antonio.json | ConvertTo-Json -Depth 5
```

or, with curl:

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d @payloads/san-antonio.json
```

**Key request fields** (full definitions are in `schemas.py` and at `/docs`):

| Field | Notes |
|---|---|
| `state_code` | Required, two-letter state, e.g. `"VA"` |
| `county_code` | Optional five-digit county FIPS code, e.g. `"51107"` (Loudoun). Alternatively `county_name` or `city` |
| `loan_amount`, `income`, `property_value`, `debt_to_income_ratio` | Borrower financials |
| `loan_purpose`, `loan_type`, `lien_status`, `occupancy_type` | HMDA numeric codes, e.g. `1` = home purchase, conventional, first lien, principal residence |
| `top_k` | Number of lenders to return |
| `model_variant` | `"B"` (default) or `"A"` |
| `min_applications` | Skip lenders with fewer historical applications than this |

Each recommendation includes the lender's LEI and name, its predicted
`approval_probability`, its `historical_approval_rate` and application count
in that state, and a `confidence` label. If no lenders match the requested
county, the service falls back to the whole state.

`GET /lenders/{state_code}` lists the lenders active in a state, with
optional `county_code`, `min_applications` (default 100) and `limit`
(default 50) query parameters.

---

## Using it from Claude Desktop (MCP)

With the service running, add this to `claude_desktop_config.json` (adjust
the paths) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "loan-arranger": {
      "command": "C:/path/to/loan-arranger/.venv/Scripts/python.exe",
      "args": ["C:/path/to/loan-arranger/mcp_server/server.py"],
      "env": { "LOAN_ARRANGER_SERVICE_URL": "http://localhost:8000" }
    }
  }
}
```

It exposes three tools: `get_lender_recommendations`, `get_lenders_in_area`
and `check_service_health`. The adapter contains no model logic; it only
forwards each call to the service.

---

## Rebuilding from raw data

You only need this to change the features or retrain the models. Be warned
that it's heavy: the raw files total about 40 GB, the merged training set has
76 million rows, and the original run used a high-memory Windows workstation
with an NVIDIA RTX 5090. On that machine one epoch of the embedding network
took about an hour, and training LightGBM Model B took about 55 minutes.

### 1. Download the raw HMDA files

Get the national loan-level LAR files from the FFIEC HMDA Platform's data
publications page (<https://ffiec.cfpb.gov/data-publication/>). The original
run used the **One Year National Loan-Level Dataset** for 2019–2023 and the
**Snapshot National Loan-Level Dataset** for 2024, in the pipe-delimited
format. Put them in per-year folders:

```
data/2019/2019_public_lar_one_year_psv.txt
data/2020/2020_public_lar_one_year_psv.txt
data/2021/2021_public_lar_one_year_pipe.txt
data/2022/2022_public_lar_one_year_pipe.txt
data/2023/2023_public_lar_one_year_pipe.txt
data/2024/2024_public_lar_pipe.txt
```

The two small source files for the lookup tables are already in the
repository: `data/lei-catalogs/{year}.json` (HMDA filer names by year) and
`data/national_county.txt` (Census Bureau county FIPS codes).

### 2. Run the pipeline in order

```powershell
# Preprocess each year. Always pass --year; it defaults to 2023 and names the output files.
python 01_preprocess.py --input data\2019\2019_public_lar_one_year_psv.txt --year 2019
python 01_preprocess.py --input data\2020\2020_public_lar_one_year_psv.txt --year 2020
python 01_preprocess.py --input data\2021\2021_public_lar_one_year_pipe.txt --year 2021
python 01_preprocess.py --input data\2022\2022_public_lar_one_year_pipe.txt --year 2022
python 01_preprocess.py --input data\2023\2023_public_lar_one_year_pipe.txt --year 2023
python 01_preprocess.py --input data\2024\2024_public_lar_pipe.txt          --year 2024

python 01b_merge.py               # -> hmda_all_*.parquet, unified indexes, feature_stats, merge_manifest.json
python build_lender_names.py      # -> data/lender_names.parquet
python build_county_lookup.py     # -> data/county_fips.parquet
python build_lender_meta.py       # -> data/lender_meta.parquet (re-run after every merge)
python 02_train_embeddings.py     # -> models/embedding_weights.pt, embedding_network.pt, training_log.json
python 03_train_lgbm.py           # -> models/lgbm_model_a/b.pkl (+ .txt), feature importance, summary
python compute_fairness.py        # -> models/fairness_report.json / .txt
```

Each script takes `--help` for its options. `03_train_lgbm.py` defaults to
`--device gpu`; pass `--device cpu` on a machine without a CUDA GPU.
`01_preprocess.py` uses NVIDIA's cuDF library when installed (Linux/WSL2
only) and otherwise falls back to pandas.

---

## Results from the original run

| Model | Validation AUC | Average precision | Brier score |
|---|---|---|---|
| Embedding network (epoch 1, best) | 0.875 | — | — |
| LightGBM Model B | 0.898 | 0.969 | 0.117 |

Model A's metrics were not saved (see Known issues).

---

## Library versions

The `.pkl` models are Python pickles. They load reliably only with the same
LightGBM and scikit-learn versions that saved them, and `requirements.txt`
lists only minimum versions. If loading fails or warns, install these exact
versions, which are the ones the models were trained and served with:

| Package | Version |
|---|---|
| Python | 3.11.9 |
| lightgbm | 4.6.0 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| pandas | 3.0.2 |
| numpy | 2.4.3 |
| pyarrow | 23.0.1 |
| torch | 2.12.0 nightly (CUDA 12.8 build) |
| fastapi / uvicorn / pydantic | 0.135.3 / 0.44.0 / 2.12.5 |
| mcp | 1.27.0 |

---

## Known issues

- **The fairness report doesn't tell us anything yet.** In
  `models/fairness_report.txt` every group in every dimension shows an
  approval rate, true-positive rate and false-positive rate of 1.000. At the
  0.5 threshold, Model A predicted approval for every validation
  application, so every disparate-impact ratio comes out 1.0 by default. The
  "all groups above 0.8" checkmarks are therefore not evidence of fairness.
  Next step: look at the spread of predicted probabilities and pick a
  threshold that matches the actual approval rate.
- **Model B's embeddings were learned with demographics.** According to
  `training_log.json`, the single embedding network was trained with
  `use_demographics: true`, and both models use its lender, tract and county
  profiles. If Model B must be entirely free of protected characteristics,
  retrain the embeddings with `02_train_embeddings.py --model-b` before
  training Model B.
- **`lgbm_training_summary.json` covers Model B only.** Model A's metrics
  were probably overwritten when Model B was trained separately.
- `training_log.json` sits at the repository root, but
  `02_train_embeddings.py` writes it to `models/`. The copy at the root was
  moved there by hand.
- `preprocess_output.txt` isn't preprocessing output. It's a captured
  PowerShell message about cuDF falling back to pandas and can be deleted.
