"""
01_preprocess.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Phase 1 Preprocessing
ISTM6218 Group Project

Reads the raw HMDA LAR CSV downloaded from:
  https://ffiec.cfpb.gov/data-publication/modified-lar

Produces two parquet files in data/:
  hmda_{year}_clean.parquet        — all retained rows, all features (Model A + B)
  hmda_{year}_model_b.parquet      — same rows, protected attributes dropped (Model B)

Also writes:
  data/lei_{year}_index.parquet     — LEI → integer index mapping (for embeddings)
  data/census_tract_{year}_index.parquet — census_tract → integer index mapping
  data/county_code_{year}_index.parquet  — county_code  → integer index mapping
  data/feature_stats_{year}.parquet — mean/std/min/max for continuous features (used
                                at inference time for normalisation)

Usage:
  python 01_preprocess.py --input data/hmda_raw.csv [--year 2023]

Requirements (install in your loan-arranger conda env):
  pip install cudf-cu13 --extra-index-url https://pypi.anaconda.org/rapidsai/simple
  pip install pandas pyarrow tqdm
  (cuDF is used when available; falls back to pandas automatically)
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── cuDF / pandas fallback ────────────────────────────────────────────────────
try:
    import cudf as pd
    import cudf.pandas  # noqa: F401  (enables cudf.pandas accelerator mode)
    GPU_MODE = True
    logging.info("cuDF detected — running in GPU mode.")
except ImportError:
    import pandas as pd
    GPU_MODE = False
    logging.warning(
        "cuDF not found — falling back to CPU pandas.  "
        "Install cudf-cu13 for GPU acceleration."
    )



# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR  = REPO_ROOT / "data"

# ── HMDA column selection ─────────────────────────────────────────────────────
# Full LAR has 99 fields; we keep only what the model needs.
# See: https://ffiec.cfpb.gov/documentation/publications/modified-lar/modified-lar-schema

TARGET_COL = "action_taken"

# Lender identity → embedding
LENDER_COLS = ["lei"]

# Geography → embeddings
GEO_COLS = ["state_code", "county_code", "census_tract"]

# Continuous borrower / loan features
CONTINUOUS_COLS = [
    "loan_amount",          # dollars
    "income",               # thousands of dollars (HMDA reports this way)
    "property_value",       # dollars  (or 'Exempt')
    "loan_to_value_ratio",  # percent  (or 'Exempt')
    "debt_to_income_ratio", # percent range string → midpoint (see below)
    "combined_loan_to_value_ratio",  # percent (or 'Exempt')
    "interest_rate",        # percent  (or 'Exempt')
    "total_loan_costs",     # dollars  (or 'Exempt' / 'NA')
]

# Categorical borrower / loan features (low cardinality — label-encoded)
CATEGORICAL_COLS = [
    "loan_purpose",         # 1=purchase 2=improvement 31=refi-cashout 32=refi-no-cashout 4=other
    "loan_type",            # 1=conv 2=FHA 3=VA 4=USDA
    "lien_status",          # 1=first 2=subordinate
    "occupancy_type",       # 1=principal 2=investment 3=second
    "derived_loan_product_type",  # e.g. "Conventional:First Lien"
]

# Protected / demographic attributes (Model A only)
DEMOGRAPHIC_COLS = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",         # string bins e.g. "25-34" — kept as-is for now
    "co_applicant_age",
]

ALL_KEEP_COLS = (
    [TARGET_COL]
    + LENDER_COLS
    + GEO_COLS
    + CONTINUOUS_COLS
    + CATEGORICAL_COLS
    + DEMOGRAPHIC_COLS
)

# ── action_taken label mapping ────────────────────────────────────────────────
# 1 = Loan originated              → APPROVED  (1)
# 2 = Application approved but not accepted → APPROVED (lender said yes)
# 3 = Application denied           → DENIED    (0)
# 4 = Application withdrawn        → EXCLUDE
# 5 = File closed for incompleteness → EXCLUDE
# 6 = Purchased loan               → EXCLUDE (not an application decision)
# 7 = Preapproval request denied   → DENIED   (0)
# 8 = Preapproval approved not accepted → APPROVED (1)
APPROVED_CODES = {1, 2, 8}
DENIED_CODES   = {3, 7}
# Anything else is excluded from training

# ── HMDA sentinel values to treat as NaN ─────────────────────────────────────
HMDA_NA_STRINGS = {"Exempt", "NA", "na", "N/A", "", " "}

# ── DTI ratio bin midpoints ───────────────────────────────────────────────────
# HMDA reports DTI as strings like "<20%", "20%-<30%", "50%-60%", ">60%"
DTI_MIDPOINT = {
    "<20%":      10.0,
    "20%-<30%":  25.0,
    "30%-<36%":  33.0,
    "36":        36.0,   # reported as exactly "36" in some years
    "37":        37.0,
    "38":        38.0,
    "39":        39.0,
    "40":        40.0,
    "41":        41.0,
    "42":        42.0,
    "43":        43.0,
    "44":        44.0,
    "45":        45.0,
    "46":        46.0,
    "47":        47.0,
    "48":        48.0,
    "49":        49.0,
    "50%-60%":   55.0,
    ">60%":      65.0,
    "50%":       50.0,
    "60%":       60.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(csv_path: Path, year: int) -> "pd.DataFrame":
    """
    Load the raw HMDA CSV/PSV.  HMDA files are large (~3–5 GB for a single year);
    we stream-read only the columns we need to minimize memory pressure.

    Automatically detects delimiter based on file extension:
      - .csv or similar: comma-separated
      - .psv or .pipe or similar: pipe-separated (|)
    """
    log.info(f"Loading {csv_path} (this may take a minute for large files) …")
    t0 = time.time()

    # Detect delimiter from file extension
    filename_lower = csv_path.name.lower()
    if "psv" in filename_lower or "pipe" in filename_lower:
        delimiter = "|"
    else:
        delimiter = ","

    log.info(f"Using delimiter: {repr(delimiter)}")

    # Only read the columns we actually need — saves ~80% of I/O on a 99-col file
    needed = list(set(ALL_KEEP_COLS))

    if GPU_MODE:
        df = pd.read_csv(
            str(csv_path),
            sep=delimiter,
            usecols=lambda c: c in needed,
            dtype=str,          # read everything as string first; we cast below
            na_values=list(HMDA_NA_STRINGS),
        )
    else:
        df = pd.read_csv(
            csv_path,
            sep=delimiter,
            usecols=lambda c: c in needed,
            dtype=str,
            na_values=list(HMDA_NA_STRINGS),
            low_memory=False,
        )

    elapsed = time.time() - t0
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols in {elapsed:.1f}s")

    # Add a year column for provenance / multi-year stacking
    df["hmda_year"] = year

    return df


def filter_action_taken(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Keep only rows with a clear approval or denial outcome.
    Converts action_taken to a binary integer label: 1=approved, 0=denied.
    """
    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df.dropna(subset=["action_taken"])
    df["action_taken"] = df["action_taken"].astype(int)

    approved_mask = df["action_taken"].isin(list(APPROVED_CODES))
    denied_mask   = df["action_taken"].isin(list(DENIED_CODES))
    df = df[approved_mask | denied_mask].copy()

    df["label"] = (df["action_taken"].isin(list(APPROVED_CODES))).astype(int)
    df = df.drop(columns=["action_taken"])

    n_approved = int(df["label"].sum())
    n_denied   = len(df) - n_approved
    log.info(
        f"After action_taken filter: {len(df):,} rows  "
        f"(approved={n_approved:,}  denied={n_denied:,}  "
        f"ratio={n_approved/len(df):.2%})"
    )
    return df


def clean_continuous(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Cast continuous columns to float32, replacing HMDA sentinel strings with NaN.
    DTI ratio bins are converted to numeric midpoints first.
    """
    # DTI special handling
    if "debt_to_income_ratio" in df.columns:
        df["debt_to_income_ratio"] = (
            df["debt_to_income_ratio"]
            .map(DTI_MIDPOINT)
            .astype("float32")
        )

    numeric_cols = [c for c in CONTINUOUS_COLS if c != "debt_to_income_ratio"
                    and c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def clean_categoricals(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Cast low-cardinality categorical columns to int8 where numeric,
    or to pandas category dtype where string.
    """
    int_cats = ["loan_purpose", "loan_type", "lien_status", "occupancy_type"]
    for col in int_cats:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")

    if "derived_loan_product_type" in df.columns:
        df["derived_loan_product_type"] = df["derived_loan_product_type"].astype("category")

    return df


def clean_demographics(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Normalise demographic string fields.
    Values like 'Race Not Available', 'Sex Not Available' etc. are kept as-is;
    the embedding layer will learn a representation for 'unknown'.
    applicant_age is kept as a string bin (e.g. '25-34') — the embedding
    network will handle it as a categorical.
    """
    str_demo_cols = ["derived_race", "derived_ethnicity", "derived_sex",
                     "applicant_age", "co_applicant_age"]
    for col in str_demo_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip().astype("category")

    return df


def build_embedding_indexes(df: "pd.DataFrame") -> dict:
    """
    Build integer index mappings for the three embedding columns:
      lei (lender)  →  lender_idx
      census_tract  →  tract_idx
      county_code   →  county_idx

    Index 0 is reserved as an <UNKNOWN> / padding token.
    Returns a dict of {col_name: DataFrame(original_value, idx)}.
    """
    indexes = {}
    mapping_defs = [
        ("lei",          "lender_idx"),
        ("census_tract", "tract_idx"),
        ("county_code",  "county_idx"),
    ]
    for src_col, idx_col in mapping_defs:
        if src_col not in df.columns:
            log.warning(f"Column '{src_col}' not found — skipping index.")
            continue

        unique_vals = df[src_col].dropna().unique()
        # Sort for determinism
        if GPU_MODE:
            unique_vals = sorted(unique_vals.to_pandas().tolist())
        else:
            unique_vals = sorted(unique_vals.tolist())

        # 0 = <UNKNOWN>, actual values start at 1
        idx_map = {v: i + 1 for i, v in enumerate(unique_vals)}

        if GPU_MODE:
            import cudf as _cudf
            idx_df = _cudf.DataFrame({
                src_col: list(idx_map.keys()),
                idx_col: list(idx_map.values()),
            })
        else:
            idx_df = pd.DataFrame({
                src_col: list(idx_map.keys()),
                idx_col: list(idx_map.values()),
            })

        indexes[src_col] = (idx_col, idx_df)
        log.info(f"  {src_col}: {len(unique_vals):,} unique values  "
                 f"→ embedding vocab size = {len(unique_vals) + 1:,} (+ 1 unknown)")

    return indexes


def apply_embedding_indexes(df: "pd.DataFrame", indexes: dict) -> "pd.DataFrame":
    """
    Join the integer indexes onto the main dataframe.
    """
    for src_col, (idx_col, idx_df) in indexes.items():
        df = df.merge(idx_df, on=src_col, how="left")
        df[idx_col] = df[idx_col].fillna(0).astype("int32")

    return df


def compute_feature_stats(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Compute mean / std / min / max for continuous features.
    These are saved as a reference table and loaded at inference time
    to normalise inputs consistently with training.
    """
    stat_cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    stats_rows = []
    for col in stat_cols:
        series = df[col].dropna()
        if GPU_MODE:
            series = series.to_pandas()
        stats_rows.append({
            "feature": col,
            "mean":    float(series.mean()),
            "std":     float(series.std()),
            "min":     float(series.min()),
            "max":     float(series.max()),
            "p01":     float(np.percentile(series, 1)),
            "p99":     float(np.percentile(series, 99)),
            "n_valid": int(series.notna().sum()),
            "n_null":  int(df[col].isna().sum()),
        })

    if GPU_MODE:
        import pandas as _pd
        return _pd.DataFrame(stats_rows)
    else:
        return pd.DataFrame(stats_rows)


def write_parquet(df: "pd.DataFrame", path: Path, label: str) -> None:
    """Write a dataframe to parquet, handling cuDF → arrow conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if GPU_MODE:
        # cuDF has native parquet writer
        df.to_parquet(str(path), index=False, compression="snappy")
    else:
        df.to_parquet(path, index=False, compression="snappy", engine="pyarrow")
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"Wrote {label}: {path.name}  ({len(df):,} rows, {size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(csv_path: Path, year: int, output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 1: Loading raw data")
    log.info("=" * 70)
    df = load_raw(csv_path, year)
    log.info(f"Loaded dataframe shape: {df.shape}")

    # ── 2. Drop rows with no lender or no geography ───────────────────────────
    log.info("=" * 70)
    log.info("STEP 2: Filtering for required geography/lender fields")
    log.info("=" * 70)
    critical_cols = ["lei", "census_tract", "county_code", "state_code"]
    before = len(df)
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    log.info(f"Dropped {before - len(df):,} rows missing lender/geography fields.")
    log.info(f"Remaining: {len(df):,} rows")

    # ── 3. Filter to actionable outcomes and create binary label ──────────────
    log.info("=" * 70)
    log.info("STEP 3: Filtering for actionable outcomes (approved/denied)")
    log.info("=" * 70)
    df = filter_action_taken(df)

    # ── 4. Clean continuous features ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 4: Cleaning continuous features")
    log.info("=" * 70)
    df = clean_continuous(df)
    log.info(f"After continuous cleaning: {df.shape}")

    # ── 5. Clean categorical features ─────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 5: Cleaning categorical features")
    log.info("=" * 70)
    df = clean_categoricals(df)
    log.info(f"After categorical cleaning: {df.shape}")

    # ── 6. Clean demographic features ─────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 6: Cleaning demographic features")
    log.info("=" * 70)
    df = clean_demographics(df)
    log.info(f"After demographic cleaning: {df.shape}")

    # ── 7. Build and save embedding indexes ───────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 7: Building embedding indexes")
    log.info("=" * 70)
    indexes = build_embedding_indexes(df)

    for src_col, (idx_col, idx_df) in indexes.items():
        idx_path = output_dir / f"{src_col.replace('_', '')}_{year}_index.parquet"
        write_parquet(idx_df, idx_path, f"{src_col} index")
    log.info(f"Built and saved {len(indexes)} embedding indexes")

    df = apply_embedding_indexes(df, indexes)
    log.info(f"After applying embedding indexes: {df.shape}")

    # ── 8. Compute and save feature statistics ────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 8: Computing feature statistics")
    log.info("=" * 70)
    stats_df = compute_feature_stats(df)
    stats_path = output_dir / f"feature_stats_{year}.parquet"
    if GPU_MODE:
        stats_df.to_parquet(str(stats_path), index=False)
    else:
        stats_df.to_parquet(stats_path, index=False)
    log.info(f"Feature statistics:\n{stats_df.to_string(index=False)}")

    # ── 9. Write Model A parquet (all features including demographics) ─────────
    log.info("=" * 70)
    log.info("STEP 9: Writing Model A parquet (all features)")
    log.info("=" * 70)
    model_a_path = output_dir / f"hmda_{year}_clean.parquet"
    write_parquet(df, model_a_path, "Model A (all features)")

    # ── 10. Write Model B parquet (protected attributes removed) ──────────────
    log.info("=" * 70)
    log.info("STEP 10: Writing Model B parquet (no demographics)")
    log.info("=" * 70)
    drop_for_b = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    df_b = df.drop(columns=drop_for_b)
    model_b_path = output_dir / f"hmda_{year}_model_b.parquet"
    write_parquet(df_b, model_b_path, "Model B (no demographics)")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    total = time.time() - t_start
    log.info(
        f"\n{'─'*60}\n"
        f"  Preprocessing complete in {total:.1f}s\n"
        f"  Mode:          {'GPU (cuDF)' if GPU_MODE else 'CPU (pandas)'}\n"
        f"  Final rows:    {len(df):,}\n"
        f"  Model A cols:  {len(df.columns)}\n"
        f"  Model B cols:  {len(df_b.columns)}\n"
        f"  Output dir:    {output_dir}\n"
        f"{'─'*60}"
    )

    # Print final column list for reference
    log.info(f"Model A columns: {sorted(df.columns.tolist())}")
    log.info(f"Dropped for B:   {drop_for_b}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess HMDA LAR CSV → cleaned parquet for model training"
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to raw HMDA CSV (e.g. data/hmda_2023_nationwide_all-records_labels.csv)"
    )
    p.add_argument(
        "--year", "-y",
        type=int,
        default=2023,
        help="HMDA data year (default: 2023); used for output filenames"
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for parquet files (default: {DATA_DIR})"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        run(csv_path=args.input, year=args.year, output_dir=args.output)
        log.info("SUCCESS: Preprocessing completed without errors.")
    except Exception as e:
        log.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)
