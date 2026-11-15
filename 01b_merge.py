"""
01b_merge.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Multi-Year HMDA Merge
ISTM6218 Group Project

Consolidates the per-year parquet files produced by 01_preprocess.py into a
single unified set of files ready for model training.

Inputs (from data/ directory, one set per year):
  hmda_{year}_clean.parquet           — Model A feature data
  hmda_{year}_model_b.parquet         — Model B feature data (no demographics)
  lei_{year}_index.parquet            — lender LEI → integer ID
  censustract_{year}_index.parquet    — census tract → integer ID
  countycode_{year}_index.parquet     — county code  → integer ID

  Note: un-stamped fallbacks (e.g. lei_index.parquet) are also accepted for
  any files produced before the year-stamp convention was adopted.

Outputs (single unified files, written to data/):
  hmda_all_clean.parquet           — Model A, all years, re-indexed
  hmda_all_model_b.parquet         — Model B, all years, re-indexed
  lei_index.parquet                — unified LEI vocabulary (overwrites)
  censustract_index.parquet        — unified tract vocabulary (overwrites)
  countycode_index.parquet         — unified county vocabulary (overwrites)
  feature_stats.parquet            — pooled statistics across all years
  merge_manifest.json              — audit log: years merged, row counts, vocab sizes

Usage:
  # Merge all years found automatically in data/
  python 01b_merge.py

  # Merge specific years only
  python 01b_merge.py --years 2021 2022 2023

  # Use a different data directory
  python 01b_merge.py --data-dir /path/to/data --years 2022 2023
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"

# ── Embedding index column definitions ───────────────────────────────────────
# Each entry: (source_col_in_data, idx_col_in_data, index_filename_stem)
INDEX_DEFS = [
    ("lei",          "lender_idx", "lei_index"),
    ("census_tract", "tract_idx",  "censustract_index"),
    ("county_code",  "county_idx", "countycode_index"),
]

# Continuous columns for pooled stats (must match 01_preprocess.py)
CONTINUOUS_COLS = [
    "loan_amount",
    "income",
    "property_value",
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "interest_rate",
    "total_loan_costs",
]


# ─────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_years(data_dir: Path) -> list[int]:
    """
    Scan data_dir for hmda_{year}_clean.parquet files and return sorted years.
    """
    pattern = re.compile(r"hmda_(\d{4})_clean\.parquet$")
    years = []
    for f in data_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def year_files(data_dir: Path, year: int) -> dict:
    """
    Return a dict of expected file paths for a given year.
    Raises FileNotFoundError if the Model A clean file is missing.
    """
    files = {
        "clean":   data_dir / f"hmda_{year}_clean.parquet",
        "model_b": data_dir / f"hmda_{year}_model_b.parquet",
        "lei":     data_dir / f"lei_index.parquet",
        "tract":   data_dir / f"censustract_index.parquet",
        "county":  data_dir / f"countycode_index.parquet",
        "stats":   data_dir / f"feature_stats.parquet",
    }
    if not files["clean"].exists():
        raise FileNotFoundError(
            f"Expected file not found: {files['clean']}\n"
            f"Run 01_preprocess.py --year {year} first."
        )
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Index merging
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_index(
    data_dir: Path,
    years: list[int],
    src_col: str,
    idx_col: str,
    filename_stem: str,
) -> pd.DataFrame:
    """
    Load per-year index files for one categorical dimension, take the union
    of all unique source values, assign new sequential integer IDs starting
    at 1 (0 is reserved for <UNKNOWN>), and return the unified DataFrame.

    The per-year index files are expected to follow the naming convention
    produced by 01_preprocess.py.  If a year's index file is missing we log
    a warning and skip it — the values from that year will map to 0 (unknown)
    in the final dataset, which the embedding layer handles gracefully.
    """
    all_values: set = set()

    for year in years:
        # Naming convention from 01_preprocess.py:
        #   {prefix}_{year}_index.parquet  e.g. lei_2023_index.parquet
        # filename_stem is e.g. "lei_index", so strip "_index" to get prefix.
        stem_prefix = filename_stem.replace("_index", "")
        candidates = [
            data_dir / f"{stem_prefix}_{year}_index.parquet",   # lei_2023_index.parquet
            data_dir / f"{filename_stem}_{year}.parquet",        # lei_index_2023.parquet (fallback)
            data_dir / f"{filename_stem}.parquet",               # lei_index.parquet (fallback)
        ]
        idx_path = next((p for p in candidates if p.exists()), None)
        if idx_path is None:
            log.warning(
                f"  [{src_col}] No index file found for year {year} "
                f"(tried: {[str(p) for p in candidates]}) — skipping."
            )
            continue

        df_idx = pd.read_parquet(idx_path, columns=[src_col])
        values = set(df_idx[src_col].dropna().unique().tolist())
        all_values.update(values)
        log.info(
            f"  [{src_col}] year={year}: {len(values):,} unique values "
            f"(running total: {len(all_values):,})"
        )

    # Also sweep the actual data files to catch any values that appear in the
    # data but were somehow absent from the index files (shouldn't happen, but
    # defensive programming costs nothing here).
    for year in years:
        clean_path = data_dir / f"hmda_{year}_clean.parquet"
        if not clean_path.exists():
            continue
        if src_col not in pd.read_parquet(clean_path, columns=[src_col]).columns:
            continue
        df_data = pd.read_parquet(clean_path, columns=[src_col])
        extra = set(df_data[src_col].dropna().unique().tolist()) - all_values
        if extra:
            log.info(
                f"  [{src_col}] year={year}: found {len(extra):,} values in "
                f"data not in any index file — adding to vocabulary."
            )
            all_values.update(extra)

    # Assign new IDs (0 = <UNKNOWN>, values start at 1)
    sorted_values = sorted(all_values)
    unified_df = pd.DataFrame({
        src_col: sorted_values,
        idx_col: range(1, len(sorted_values) + 1),
    })

    log.info(
        f"  [{src_col}] Unified vocabulary: {len(sorted_values):,} values  "
        f"(embedding vocab size = {len(sorted_values) + 1:,} incl. <UNKNOWN>)"
    )
    return unified_df


# ─────────────────────────────────────────────────────────────────────────────
# Data merging
# ─────────────────────────────────────────────────────────────────────────────

def reindex_dataframe(
    df: pd.DataFrame,
    unified_indexes: dict,   # {src_col: unified_index_df}
) -> pd.DataFrame:
    """
    Drop the old per-year integer index columns and join the unified ones.
    Rows whose source value isn't in the unified index map to 0 (unknown).
    """
    for src_col, (idx_col, unified_df) in unified_indexes.items():
        if src_col not in df.columns:
            continue
        # Drop the stale index column produced by 01_preprocess.py
        if idx_col in df.columns:
            df = df.drop(columns=[idx_col])
        # Join unified index
        df = df.merge(unified_df[[src_col, idx_col]], on=src_col, how="left")
        df[idx_col] = df[idx_col].fillna(0).astype("int32")
    return df


def merge_data_files(
    data_dir: Path,
    years: list[int],
    suffix: str,              # "_clean" or "_model_b"
    unified_indexes: dict,
) -> pd.DataFrame:
    """
    Concatenate per-year data parquets and re-apply the unified embedding
    indexes.  Adds a 'hmda_year' column if not already present.
    """
    frames = []
    for year in years:
        path = data_dir / f"hmda_{year}{suffix}.parquet"
        if not path.exists():
            log.warning(f"  Data file not found, skipping: {path.name}")
            continue
        df = pd.read_parquet(path)
        if "hmda_year" not in df.columns:
            df["hmda_year"] = year
        frames.append(df)
        log.info(f"  Loaded {path.name}: {len(df):,} rows")

    if not frames:
        raise RuntimeError(f"No data files found for suffix='{suffix}'")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"  Combined ({suffix}): {len(combined):,} rows before re-indexing")

    combined = reindex_dataframe(combined, unified_indexes)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Feature statistics — pooled across years
# ─────────────────────────────────────────────────────────────────────────────

def pool_feature_stats(data_dir: Path, years: list[int]) -> pd.DataFrame:
    """
    Compute pooled mean/std/min/max across all years directly from the
    combined clean data rather than trying to algebraically combine per-year
    stats files.  This guarantees correctness regardless of year sample sizes.

    We read only the continuous columns to keep memory usage down.
    """
    log.info("Computing pooled feature statistics from combined data …")
    frames = []
    for year in years:
        path = data_dir / f"hmda_{year}_clean.parquet"
        if not path.exists():
            continue
        avail_cols = [c for c in CONTINUOUS_COLS
                      if c in pd.read_parquet(path, columns=[]).columns
                      or True]  # read_parquet with empty columns list returns schema
        try:
            df = pd.read_parquet(path, columns=CONTINUOUS_COLS)
        except Exception:
            # Some columns may be missing in older HMDA vintages
            schema_cols = pd.read_parquet(path).columns.tolist()
            read_cols = [c for c in CONTINUOUS_COLS if c in schema_cols]
            df = pd.read_parquet(path, columns=read_cols)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for col in CONTINUOUS_COLS:
        if col not in combined.columns:
            continue
        series = combined[col].dropna().astype(float)
        rows.append({
            "feature": col,
            "mean":    float(series.mean()),
            "std":     float(series.std()),
            "min":     float(series.min()),
            "max":     float(series.max()),
            "p01":     float(np.percentile(series, 1)),
            "p99":     float(np.percentile(series, 99)),
            "n_valid": int(series.notna().sum()),
            "n_null":  int(combined[col].isna().sum()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path, years: list[int]) -> None:
    t_start = time.time()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "data_dir": str(data_dir),
        "row_counts": {},
        "vocab_sizes": {},
    }

    log.info(f"Merging {len(years)} year(s): {years}")
    log.info(f"Data directory: {data_dir}")

    # ── 1. Build unified embedding indexes ───────────────────────────────────
    log.info("\n── Step 1: Building unified embedding indexes ──────────────────")
    unified_indexes = {}   # {src_col: (idx_col, unified_df)}

    for src_col, idx_col, filename_stem in INDEX_DEFS:
        log.info(f"Processing [{src_col}] …")
        unified_df = build_unified_index(
            data_dir, years, src_col, idx_col, filename_stem
        )
        unified_indexes[src_col] = (idx_col, unified_df)
        manifest["vocab_sizes"][src_col] = len(unified_df) + 1  # +1 for unknown

        # Overwrite the index file with the unified version
        out_path = data_dir / f"{filename_stem}.parquet"
        unified_df.to_parquet(out_path, index=False, compression="snappy")
        log.info(f"  Written: {out_path.name}")

    # ── 2. Merge Model A data ─────────────────────────────────────────────────
    log.info("\n── Step 2: Merging Model A data (all features) ─────────────────")
    df_a = merge_data_files(data_dir, years, "_clean", unified_indexes)
    out_a = data_dir / "hmda_all_clean.parquet"
    df_a.to_parquet(out_a, index=False, compression="snappy")
    size_a = out_a.stat().st_size / 1_048_576
    log.info(f"Written: {out_a.name}  ({len(df_a):,} rows, {size_a:.1f} MB)")
    manifest["row_counts"]["model_a"] = len(df_a)
    manifest["model_a_cols"] = sorted(df_a.columns.tolist())

    # ── 3. Merge Model B data ─────────────────────────────────────────────────
    log.info("\n── Step 3: Merging Model B data (no demographics) ──────────────")
    df_b = merge_data_files(data_dir, years, "_model_b", unified_indexes)
    out_b = data_dir / "hmda_all_model_b.parquet"
    df_b.to_parquet(out_b, index=False, compression="snappy")
    size_b = out_b.stat().st_size / 1_048_576
    log.info(f"Written: {out_b.name}  ({len(df_b):,} rows, {size_b:.1f} MB)")
    manifest["row_counts"]["model_b"] = len(df_b)

    # Sanity check — both files should have the same number of rows
    if len(df_a) != len(df_b):
        log.warning(
            f"Row count mismatch: Model A={len(df_a):,}  Model B={len(df_b):,}. "
            f"This is unexpected — check for missing per-year model_b files."
        )

    # ── 4. Pooled feature statistics ─────────────────────────────────────────
    log.info("\n── Step 4: Computing pooled feature statistics ──────────────────")
    stats_df = pool_feature_stats(data_dir, years)
    stats_path = data_dir / "feature_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)
    log.info(f"Written: {stats_path.name}")
    log.info(f"\n{stats_df.to_string(index=False)}")

    # ── 5. Write manifest ─────────────────────────────────────────────────────
    manifest["elapsed_seconds"] = round(time.time() - t_start, 1)
    manifest_path = data_dir / "merge_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"\nWritten: {manifest_path.name}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info(
        f"\n{'─'*60}\n"
        f"  Merge complete in {elapsed:.1f}s\n"
        f"  Years merged:    {years}\n"
        f"  Model A rows:    {len(df_a):,}\n"
        f"  Model B rows:    {len(df_b):,}\n"
        f"  Lender vocab:    {manifest['vocab_sizes'].get('lei', '?'):,}\n"
        f"  Tract vocab:     {manifest['vocab_sizes'].get('census_tract', '?'):,}\n"
        f"  County vocab:    {manifest['vocab_sizes'].get('county_code', '?'):,}\n"
        f"\n"
        f"  Ready for:  python 02_train_embeddings.py\n"
        f"{'─'*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Merge per-year HMDA parquet files from 01_preprocess.py into "
            "unified training-ready files with consistent embedding indexes."
        )
    )
    p.add_argument(
        "--years", "-y",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Years to merge (e.g. --years 2021 2022 2023). "
            "If omitted, all years found in --data-dir are used."
        )
    )
    p.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing per-year parquet files (default: {DATA_DIR})"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.data_dir.exists():
        log.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    years = args.years
    if years is None:
        years = discover_years(args.data_dir)
        if not years:
            log.error(
                f"No hmda_{{year}}_clean.parquet files found in {args.data_dir}. "
                f"Run 01_preprocess.py first."
            )
            sys.exit(1)
        log.info(f"Auto-discovered years: {years}")
    else:
        # Validate that the requested years actually exist
        missing = [
            y for y in years
            if not (args.data_dir / f"hmda_{y}_clean.parquet").exists()
        ]
        if missing:
            log.error(
                f"Missing data files for year(s): {missing}. "
                f"Run 01_preprocess.py for each missing year first."
            )
            sys.exit(1)

    run(data_dir=args.data_dir, years=years)