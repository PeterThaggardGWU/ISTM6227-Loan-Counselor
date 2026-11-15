"""
build_lender_meta.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Lender Metadata Summary Builder
ISTM6218 Group Project

Pre-computes the lender metadata table that model_loader.py used to build at
every startup from the 1.65 GB data/hmda_all_clean.parquet, and saves it as a
small parquet so the service (and teammates) don't need the full training file.

One row per (lei, state_code):
  lei              str
  state_code       str
  n_applications   int     count of applications with a label
  approval_rate    float   mean of label, rounded to 4 places
  active_counties  list    county codes where the lender had applications
                           (model_loader converts this back to a set on load)

The input is read in row-group batches and reduced as it goes, so this runs
in a few GB of RAM instead of loading 76M rows at once.

Input:   data/hmda_all_clean.parquet   (columns: lei, state_code, county_code, label)
Output:  data/lender_meta.parquet

Usage:
  python build_lender_meta.py
  python build_lender_meta.py --input data/hmda_all_clean.parquet --output data/lender_meta.parquet
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
COLS = ["lei", "state_code", "county_code", "label"]
KEYS = ["lei", "state_code", "county_code"]
REDUCE_EVERY = 8   # batches between intermediate reductions (bounds memory)


def _reduce(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine partial (lei, state, county) tallies into one table."""
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(KEYS, dropna=False, as_index=False)[["n", "approved"]].sum()


def build(input_path: Path, output_path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(input_path)
    log.info(f"Reading {input_path}  ({pf.metadata.num_rows:,} rows, "
             f"{pf.metadata.num_row_groups} row groups)")

    # Pass 1: tally applications and approvals per (lei, state, county).
    # dropna=False keeps rows with a missing county so they still count toward
    # the lender's state-level totals, matching the original loader logic.
    running: list[pd.DataFrame] = []
    for i, batch in enumerate(pf.iter_batches(columns=COLS), start=1):
        df = batch.to_pandas()
        df = df[df["label"].notna()]
        part = (
            df.groupby(KEYS, dropna=False, as_index=False)
              .agg(n=("label", "size"), approved=("label", "sum"))
        )
        running.append(part)
        if len(running) >= REDUCE_EVERY:
            running = [_reduce(running)]
            log.info(f"  processed {i} batches — {len(running[0]):,} lender/county rows so far")

    tally = _reduce(running)

    # The original groupby on (lei, state_code) dropped rows missing either key.
    tally = tally[tally["lei"].notna() & tally["state_code"].notna()]

    # Pass 2: roll up to (lei, state_code).
    meta = (
        tally.groupby(["lei", "state_code"], as_index=False)
             .agg(n_applications=("n", "sum"), approved=("approved", "sum"))
    )
    meta["approval_rate"] = (meta["approved"] / meta["n_applications"]).round(4)
    meta = meta.drop(columns="approved")

    counties = (
        tally[tally["county_code"].notna()]
        .groupby(["lei", "state_code"])["county_code"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"county_code": "active_counties"})
    )
    meta = meta.merge(counties, on=["lei", "state_code"], how="left")
    meta["active_counties"] = meta["active_counties"].apply(
        lambda c: c if isinstance(c, list) else []
    )
    meta["n_applications"] = meta["n_applications"].astype("int64")

    meta = meta[["lei", "state_code", "n_applications", "approval_rate", "active_counties"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"Saved {output_path}  ({len(meta):,} lender/state rows, {size_mb:.1f} MB)")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=Path, default=REPO_ROOT / "data" / "hmda_all_clean.parquet")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "lender_meta.parquet")
    args = p.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
