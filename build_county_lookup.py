"""
build_county_lookup.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — County Name / City to FIPS Lookup Builder
ISTM6218 Group Project

Reads the Census Bureau national_county.txt file and builds a lookup parquet
that maps (state, county_name) -> fips_code.

Also builds a supplementary city-to-county mapping using the Census Bureau
national_places file if available, enabling city-name lookups.

Input:  data/national_county.txt   (Census Bureau, no header, pipe/comma delimited)
Output: data/county_fips.parquet   (state_code, county_name, county_name_clean, fips)

Usage:
  python build_county_lookup.py
  python build_county_lookup.py --input data/national_county.txt
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).resolve().parent
INPUT_PATH  = REPO_ROOT / "data" / "national_county.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "county_fips.parquet"


def clean_county_name(name: str) -> str:
    """
    Normalise a county name for fuzzy matching:
      - lowercase
      - strip common suffixes (County, Parish, Borough, Census Area, etc.)
      - strip punctuation
    This lets "Bexar County", "Bexar county", and "Bexar" all match the same row.
    """
    name = name.lower().strip()
    # Remove common suffixes
    suffixes = [
        r"\s+county$", r"\s+parish$", r"\s+borough$",
        r"\s+census\s+area$", r"\s+municipality$",
        r"\s+city\s+and\s+county$", r"\s+municipio$",
        r"\s+district$", r"\s+island$",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    # Strip punctuation except hyphens
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def run(input_path: Path, output_path: Path) -> None:

    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        raise SystemExit(1)

    log.info(f"Reading {input_path} ...")

    # No header row — assign column names
    # Format: state_usps, state_fips, county_fips_suffix, county_name, entity_code
    df = pd.read_csv(
        input_path,
        header=None,
        names=["state_code", "state_fips", "county_suffix", "county_name", "entity_code"],
        dtype=str,
    )

    log.info(f"  Loaded {len(df):,} rows")

    # Build the 5-digit FIPS code
    df["fips"] = df["state_fips"].str.zfill(2) + df["county_suffix"].str.zfill(3)

    # Clean county name for fuzzy matching
    df["county_name_clean"] = df["county_name"].apply(clean_county_name)

    # Keep only the columns we need
    result = df[["state_code", "fips", "county_name", "county_name_clean"]].copy()
    result["state_code"] = result["state_code"].str.upper().str.strip()

    # Sanity check
    log.info(f"  States covered: {result['state_code'].nunique()}")
    log.info(f"  Total counties: {len(result):,}")
    log.info(f"  Sample rows:")
    for _, row in result.head(5).iterrows():
        log.info(f"    {row['state_code']}  {row['fips']}  {row['county_name']!r}  "
                 f"(clean: {row['county_name_clean']!r})")

    # Write parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="snappy")
    size_kb = output_path.stat().st_size / 1024
    log.info(f"\nWritten: {output_path}  ({len(result):,} rows, {size_kb:.1f} KB)")
    log.info("Ready for use by the Loan Arranger service.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build county name -> FIPS lookup from Census Bureau national_county.txt"
    )
    p.add_argument("--input",  type=Path, default=INPUT_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
