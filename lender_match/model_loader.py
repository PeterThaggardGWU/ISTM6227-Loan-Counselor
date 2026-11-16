"""
model_loader.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger - Model Artifact Loader
ISTM6218 Group Project

Loads and holds in memory all artifacts needed for inference:
  - LightGBM models (A and B)
  - Embedding weight matrices (lender, tract, county)
  - Lender index (LEI -> integer ID)
  - Lender metadata (historical approval rates, application counts, names)
  - Feature statistics (mean/std for normalisation)
  - Merge manifest (training year range)

Designed to be instantiated once at FastAPI startup and shared across all
requests via FastAPI's dependency injection / app.state pattern.
────────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

# ── Feature columns (must match 03_train_lgbm.py) ────────────────────────────
CONTINUOUS_COLS = [
    "loan_amount",
    "income",
    "property_value",
    "debt_to_income_ratio",
]

INT_CAT_COLS = [
    "loan_purpose",
    "loan_type",
    "lien_status",
    "occupancy_type",
]

DEMOGRAPHIC_COLS = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",
    "co_applicant_age",
]

EMB_NAMES = ["lender", "tract", "county"]


def _confidence_label(n: int) -> str:
    if n >= 1000:
        return "high"
    if n >= 200:
        return "medium"
    return "low"


class ModelStore:
    """
    Holds all inference artifacts in memory.
    Call load() once at startup; then use score() per request.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        data_dir:   Path = DATA_DIR,
    ):
        self.models_dir = models_dir
        self.data_dir   = data_dir

        # Set after load()
        self.model_a = None
        self.model_b = None
        self.embeddings: dict       = {}   # {name: np.ndarray [vocab, dim]}
        self.feature_stats: dict    = {}   # {col: {mean, std}}
        self.lender_index: pd.DataFrame = None   # LEI -> lender_idx
        self.lender_meta:  pd.DataFrame = None   # LEI -> approval_rate, n_apps, etc.
        self.lender_names: dict         = {}     # LEI -> human-readable name
        self.county_lookup: pd.DataFrame = None  # (state, county_name_clean) -> fips
        self.training_years: list[int]  = []
        self._loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all artifacts from disk. Call once at startup."""
        log.info("ModelStore: loading artifacts ...")

        self._load_models()
        self._load_embeddings()
        self._load_feature_stats()
        self._load_lender_index()
        self._load_lender_names()
        self._load_county_lookup()
        self._build_lender_metadata()
        self._load_manifest()

        self._loaded = True
        log.info(
            f"ModelStore ready — "
            f"model_a={'yes' if self.model_a else 'no'}  "
            f"model_b={'yes' if self.model_b else 'no'}  "
            f"lenders={len(self.lender_index):,}  "
            f"years={self.training_years}"
        )

    def _load_models(self) -> None:
        for variant, attr in [("a", "model_a"), ("b", "model_b")]:
            path = self.models_dir / f"lgbm_model_{variant}.pkl"
            if path.exists():
                setattr(self, attr, joblib.load(path))
                log.info(f"  Loaded {path.name}")
            else:
                log.warning(f"  {path.name} not found — Model {variant.upper()} unavailable")

    def _load_embeddings(self) -> None:
        path = self.models_dir / "embedding_weights.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding weights not found: {path}\n"
                "Run 02_train_embeddings.py first."
            )
        raw = torch.load(path, map_location="cpu", weights_only = False)
        # Ensure numpy arrays (may already be numpy from export)
        for name, weights in raw.items():
            if isinstance(weights, torch.Tensor):
                self.embeddings[name] = weights.numpy()
            else:
                self.embeddings[name] = np.array(weights)
            log.info(f"  Embeddings [{name}]: shape {self.embeddings[name].shape}")

    def _load_feature_stats(self) -> None:
        path = self.data_dir / "feature_stats.parquet"
        if not path.exists():
            raise FileNotFoundError(f"feature_stats.parquet not found: {path}")
        df = pd.read_parquet(path).set_index("feature")
        self.feature_stats = {
            col: {
                "mean": float(df.loc[col, "mean"]),
                "std":  float(df.loc[col, "std"]),
            }
            for col in df.index
        }
        log.info(f"  Feature stats: {list(self.feature_stats.keys())}")

    def _load_lender_index(self) -> None:
        path = self.data_dir / "lei_index.parquet"
        if not path.exists():
            raise FileNotFoundError(f"lei_index.parquet not found: {path}")
        self.lender_index = pd.read_parquet(path)
        log.info(f"  Lender index: {len(self.lender_index):,} LEIs")

    def _load_lender_names(self) -> None:
        """
        Load the LEI -> lender name mapping built by build_lender_names.py.
        Falls back gracefully if the file is not present.
        """
        path = self.data_dir / "lender_names.parquet"
        if not path.exists():
            log.warning(
                "lender_names.parquet not found — lender_name will be null in responses. "
                "Run build_lender_names.py to generate it."
            )
            self.lender_names = {}
            return
        df = pd.read_parquet(path, columns=["lei", "name"])
        self.lender_names = dict(zip(df["lei"], df["name"]))
        log.info(f"  Lender names: {len(self.lender_names):,} LEIs")

    def _load_county_lookup(self) -> None:
        """Load the county name -> FIPS lookup built by build_county_lookup.py."""
        path = self.data_dir / "county_fips.parquet"
        if not path.exists():
            log.warning(
                "county_fips.parquet not found — city/county_name resolution unavailable. "
                "Run build_county_lookup.py to generate it."
            )
            self.county_lookup = None
            return
        self.county_lookup = pd.read_parquet(path)
        log.info(f"  County lookup: {len(self.county_lookup):,} entries")

    def resolve_county_code(
        self,
        state_code: str,
        county_name: Optional[str] = None,
        city: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve a county name or city name to a 5-digit FIPS code.

        Matching strategy (in order):
          1. Exact match on cleaned county name within the given state
          2. Prefix match (e.g. "Bexar" matches "Bexar County")
          3. For city input: check if the city name matches any county name
             (works for independent cities like "St. Louis City")

        Returns the FIPS code string if found, None otherwise.
        """
        if self.county_lookup is None:
            return None

        import re as _re

        def clean(s: str) -> str:
            s = s.lower().strip()
            for suffix in [r"\s+county$", r"\s+parish$", r"\s+borough$",
                           r"\s+census\s+area$", r"\s+municipality$",
                           r"\s+city\s+and\s+county$", r"\s+municipio$"]:
                s = _re.sub(suffix, "", s, flags=_re.IGNORECASE)
            s = _re.sub(r"[^\w\s\-]", "", s)
            return _re.sub(r"\s+", " ", s).strip()

        state_mask = self.county_lookup["state_code"] == state_code.upper()
        state_df   = self.county_lookup[state_mask]

        if len(state_df) == 0:
            log.warning(f"  resolve_county_code: no counties found for state {state_code}")
            return None

        # Try county_name first, then city as fallback
        for raw_input in filter(None, [county_name, city]):
            cleaned = clean(raw_input)

            # 1. Exact match
            exact = state_df[state_df["county_name_clean"] == cleaned]
            if len(exact) == 1:
                fips = exact["fips"].iloc[0]
                log.info(f"  Resolved {raw_input!r} -> {fips} (exact match)")
                return fips
            if len(exact) > 1:
                fips = exact["fips"].iloc[0]
                log.info(f"  Resolved {raw_input!r} -> {fips} (first of {len(exact)} exact matches)")
                return fips

            # 2. Prefix match
            prefix = state_df[state_df["county_name_clean"].str.startswith(cleaned)]
            if len(prefix) == 1:
                fips = prefix["fips"].iloc[0]
                log.info(f"  Resolved {raw_input!r} -> {fips} (prefix match)")
                return fips

            # 3. Substring match (county name contains the input)
            substr = state_df[state_df["county_name_clean"].str.contains(cleaned, regex=False)]
            if len(substr) == 1:
                fips = substr["fips"].iloc[0]
                log.info(f"  Resolved {raw_input!r} -> {fips} (substring match)")
                return fips

            log.info(f"  resolve_county_code: no match for {raw_input!r} in {state_code}")

        return None

    def _build_lender_metadata(self) -> None:
        """
        Build a lender metadata table aggregated at the (lei, state_code) level,
        ensuring exactly one row per lender per state — no duplicate LEIs.

        County codes are retained as a set per lender so that
        get_lenders_in_geography() can still filter by county presence.

        Prefers the pre-computed data/lender_meta.parquet (built by
        build_lender_meta.py, ~2 MB); falls back to aggregating the full
        hmda_all_clean.parquet if that file is absent.
        """
        summary_path = self.data_dir / "lender_meta.parquet"
        if summary_path.exists():
            meta = pd.read_parquet(summary_path)
            meta["active_counties"] = meta["active_counties"].apply(set)  # parquet stores lists
            self.lender_meta = meta
            log.info(f"  Lender metadata: {len(meta):,} (lender, state) rows from {summary_path.name}")
            return

        clean_path = self.data_dir / "hmda_all_clean.parquet"
        if not clean_path.exists():
            log.warning("hmda_all_clean.parquet not found — lender metadata unavailable")
            self.lender_meta = pd.DataFrame(
                columns=["lei", "state_code", "active_counties",
                         "approval_rate", "n_applications"]
            )
            return

        log.info("  Building lender metadata from training data ...")
        cols = ["lei", "state_code", "county_code", "label"]
        df = pd.read_parquet(clean_path, columns=cols)

        # One row per (lender, state) — eliminates duplicate LEI issue
        meta = (
            df.groupby(["lei", "state_code"])
            .agg(
                n_applications=("label", "count"),
                approval_rate=("label", "mean"),
            )
            .reset_index()
        )
        meta["approval_rate"] = meta["approval_rate"].round(4)

        # Track which counties each lender is active in (for county filtering)
        county_sets = (
            df.groupby(["lei", "state_code"])["county_code"]
            .apply(set)
            .reset_index()
            .rename(columns={"county_code": "active_counties"})
        )
        meta = meta.merge(county_sets, on=["lei", "state_code"], how="left")

        self.lender_meta = meta
        log.info(
            f"  Lender metadata: {len(meta):,} unique (lender, state) combinations"
        )

    def _load_manifest(self) -> None:
        path = self.data_dir / "merge_manifest.json"
        if path.exists():
            with open(path) as f:
                manifest = json.load(f)
            self.training_years = manifest.get("years", [])
        else:
            self.training_years = []

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def get_lenders_in_geography(
        self,
        state_code: str,
        county_code: Optional[str],
        min_applications: int,
    ) -> pd.DataFrame:
        """
        Return lender metadata rows for the requested geography.
        Each row is unique per LEI — no duplicates.

        If county_code is provided, only lenders that have historically
        operated in that county are returned (checked via active_counties set).
        Falls back to state-level if no county results are found.
        """
        if self.lender_meta is None or len(self.lender_meta) == 0:
            return pd.DataFrame()

        mask = self.lender_meta["state_code"] == state_code.upper()
        mask &= self.lender_meta["n_applications"] >= min_applications

        if county_code:
            county_str = str(county_code)
            county_mask = self.lender_meta["active_counties"].apply(
                lambda counties: county_str in counties
                if isinstance(counties, set) else False
            )
            mask &= county_mask

        return self.lender_meta[mask].copy()

    def build_feature_row(
        self,
        lei: str,
        county_code: Optional[str],
        profile_dict: dict,
        use_demographics: bool,
    ) -> np.ndarray:
        """
        Build one feature row for a given lender + borrower profile.
        Returns a 1-D numpy array matching the feature order used in training.
        """
        features = []

        # ── Embedding lookups ─────────────────────────────────────────────────
        # Lender embedding
        lender_idx = self._lookup_idx("lei", "lender_idx", lei)
        features.extend(self._get_embedding("lender", lender_idx))

        # Tract embedding — use county-level proxy (tract unknown at inference time)
        # We average all tract embeddings for lenders in this county as a proxy.
        # At inference time we don't know the exact census tract so this is a
        # reasonable geographic prior.
        tract_emb = self._mean_tract_embedding_for_county(county_code)
        features.extend(tract_emb)

        # County embedding
        county_idx = self._lookup_idx("county_code", "county_idx", county_code or "")
        features.extend(self._get_embedding("county", county_idx))

        # ── Normalised continuous features ────────────────────────────────────
        for col in CONTINUOUS_COLS:
            val = profile_dict.get(col)
            stats = self.feature_stats.get(col, {"mean": 0.0, "std": 1.0})
            mean = stats["mean"]
            std  = max(stats["std"], 1e-8)
            if val is None:
                features.append(0.0)   # fill with 0 (= normalised mean)
            else:
                features.append((float(val) - mean) / std)

        # ── Integer categorical features ──────────────────────────────────────
        for col in INT_CAT_COLS:
            features.append(float(profile_dict.get(col, 0)))

        # ── Demographic features (Model A only) ───────────────────────────────
        if use_demographics:
            for col in DEMOGRAPHIC_COLS:
                val = profile_dict.get(col)
                # Encode as integer category code — 0 for unknown/missing
                features.append(0.0 if val is None else 1.0)

        # ── Year feature — use most recent training year as inference proxy ───
        year = max(self.training_years) if self.training_years else 2024
        features.append(float(year))

        return np.array(features, dtype=np.float32)

    def score(
        self,
        profile_dict: dict,
        state_code: str,
        county_code: Optional[str],
        model_variant: str,
        top_k: int,
        min_applications: int,
    ) -> tuple[list[dict], int]:
        """
        Main inference method.

        Returns (ranked_results, n_evaluated) where ranked_results is a list
        of dicts ready to be converted to LenderResult objects.
        """
        if not self._loaded:
            raise RuntimeError("ModelStore.load() has not been called.")

        model = self.model_a if model_variant == "A" else self.model_b
        if model is None:
            raise ValueError(
                f"Model {model_variant} is not loaded. "
                f"Run 03_train_lgbm.py first."
            )

        use_demographics = (model_variant == "A")

        # Get candidate lenders in the requested geography
        candidates = self.get_lenders_in_geography(
            state_code, county_code, min_applications
        )

        if len(candidates) == 0:
            # Fall back to state-level if no county-level results
            if county_code:
                log.info(
                    f"  No lenders found for county {county_code} — "
                    f"falling back to state {state_code}"
                )
                candidates = self.get_lenders_in_geography(
                    state_code, None, min_applications
                )

        if len(candidates) == 0:
            return [], 0

        n_evaluated = len(candidates)

        # Build feature matrix: one row per candidate lender
        rows = []
        for _, row in candidates.iterrows():
            feature_row = self.build_feature_row(
                lei=row["lei"],
                county_code=county_code or row.get("county_code"),
                profile_dict=profile_dict,
                use_demographics=use_demographics,
            )
            rows.append(feature_row)

        X = np.vstack(rows)

        # Score all candidates at once
        probs = model.predict_proba(X)[:, 1]
        candidates = candidates.copy()
        candidates["approval_probability"] = probs

        # Sort by predicted probability descending
        candidates = candidates.sort_values(
            "approval_probability", ascending=False
        ).head(top_k).reset_index(drop=True)

        # Build result dicts
        results = []
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            results.append({
                "rank":                       rank,
                "lei":                        row["lei"],
                "approval_probability":       round(float(row["approval_probability"]), 4),
                "historical_approval_rate":   round(float(row["approval_rate"]), 4),
                "historical_application_count": int(row["n_applications"]),
                "lender_name":                self.lender_names.get(row["lei"]),
                "confidence":                 _confidence_label(int(row["n_applications"])),
            })

        return results, n_evaluated

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _lookup_idx(self, src_col: str, idx_col: str, value: str) -> int:
        """Look up an integer index from the lender index table."""
        if self.lender_index is None:
            return 0
        if src_col not in self.lender_index.columns:
            return 0
        match = self.lender_index[self.lender_index[src_col] == value]
        if len(match) == 0:
            return 0   # unknown token
        return int(match[idx_col].iloc[0])

    def _get_embedding(self, name: str, idx: int) -> np.ndarray:
        """Return the embedding vector for a given name and index."""
        W = self.embeddings.get(name)
        if W is None:
            return np.zeros(64, dtype=np.float32)
        idx = max(0, min(idx, W.shape[0] - 1))
        return W[idx]

    def _mean_tract_embedding_for_county(
        self, county_code: Optional[str]
    ) -> np.ndarray:
        """
        At inference time we don't know the census tract. Return the mean
        of all tract embeddings as a geographic prior — a reasonable proxy
        that stays within the embedding space.
        """
        W = self.embeddings.get("tract")
        if W is None:
            return np.zeros(32, dtype=np.float32)
        # Mean over all non-zero rows (skip the unknown token at index 0)
        return W[1:].mean(axis=0).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def health_dict(self) -> dict:
        return {
            "status":           "ok" if self._loaded else "loading",
            "model_a_loaded":   self.model_a is not None,
            "model_b_loaded":   self.model_b is not None,
            "embeddings_loaded": len(self.embeddings) > 0,
            "lender_count":     len(self.lender_index) if self.lender_index is not None else 0,
            "training_years":   self.training_years,
        }
