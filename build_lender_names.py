"""
build_lender_names.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — LEI to Lender Name Catalog Builder
ISTM6218 Group Project

Reads per-year institution JSON files from data/lei-catalogs/{year}.json,
merges them keeping only the most recent year's name for each LEI, and
writes the result to data/lender_names.parquet for fast startup loading.

Input files:  data/lei-catalogs/2019.json ... 2024.json
              Each file: {"institutions": [{"lei": "...", "name": "...", "period": "YYYY"}, ...]}

Output:       data/lender_names.parquet
              Columns: lei (str), name (str), year (int)

Usage:
  python build_lender_names.py
  python build_lender_names.py --catalog-dir data/lei-catalogs --output data/lender_names.parquet
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).resolve().parent
CATALOG_DIR = REPO_ROOT / "data" / "lei-catalogs"
OUTPUT_PATH = REPO_ROOT / "data" / "lender_names.parquet"


def run(catalog_dir: Path, output_path: Path) -> None:

    # ── Discover year files ───────────────────────────────────────────────────
    year_files = sorted(catalog_dir.glob("*.json"))
    if not year_files:
        log.error(f"No JSON files found in {catalog_dir}")
        raise SystemExit(1)

    log.info(f"Found {len(year_files)} catalog file(s): "
             f"{[f.stem for f in year_files]}")

    # ── Load all years ────────────────────────────────────────────────────────
    frames = []
    for path in year_files:
        year = int(path.stem)
        log.info(f"  Loading {path.name} ...")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        institutions = data.get("institutions", [])
        if not institutions:
            log.warning(f"  {path.name}: no 'institutions' key or empty list — skipping.")
            continue

        df = pd.DataFrame(institutions)

        # Normalise column names — CFPB files use 'lei', 'name', 'period'
        df = df.rename(columns={"period": "year"})

        # Keep only the columns we need
        keep = [c for c in ["lei", "name", "year"] if c in df.columns]
        df = df[keep].copy()

        # Coerce year to int (period field is a string like "2019")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(year).astype(int)

        # Drop rows with missing LEI or name
        before = len(df)
        df = df.dropna(subset=["lei", "name"])
        df = df[df["lei"].str.strip() != ""]
        df = df[df["name"].str.strip() != ""]
        dropped = before - len(df)
        if dropped:
            log.info(f"    Dropped {dropped:,} rows with missing LEI or name")

        frames.append(df)
        log.info(f"    {len(df):,} institutions loaded from {path.name}")

    if not frames:
        log.error("No institution records loaded from any file.")
        raise SystemExit(1)

    # ── Merge — keep most recent name per LEI ─────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"\nCombined: {len(combined):,} rows across all years "
             f"({combined['lei'].nunique():,} unique LEIs)")

    # Sort by year descending then deduplicate — first occurrence = most recent
    combined = combined.sort_values("year", ascending=False)
    deduped  = combined.drop_duplicates(subset=["lei"], keep="first").reset_index(drop=True)

    log.info(f"After deduplication (most recent year wins): {len(deduped):,} unique LEIs")

    # Year distribution of the winning names
    year_dist = deduped["year"].value_counts().sort_index()
    log.info("  Year distribution of winning name assignments:")
    for yr, cnt in year_dist.items():
        log.info(f"    {yr}: {cnt:,} LEIs")

    # ── Write parquet ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(output_path, index=False, compression="snappy")
    size_kb = output_path.stat().st_size / 1024
    log.info(f"\nWritten: {output_path}  ({len(deduped):,} rows, {size_kb:.1f} KB)")
    log.info("Ready for: model_loader.py _load_lender_names()")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-year LEI→name catalogs into a single parquet"
    )
    p.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR,
                   help=f"Directory containing {{year}}.json files (default: {CATALOG_DIR})")
    p.add_argument("--output",      type=Path, default=OUTPUT_PATH,
                   help=f"Output parquet path (default: {OUTPUT_PATH})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.catalog_dir.exists():
        log.error(f"Catalog directory not found: {args.catalog_dir}")
        raise SystemExit(1)
    run(args.catalog_dir, args.output)
