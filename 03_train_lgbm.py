"""
03_train_lgbm.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Phase 2: LightGBM Gradient Boosted Tree Training
ISTM6218 Group Project

Loads the pre-trained entity embedding weights from Phase 1, looks up the
embedding vector for each row's lender, census tract, and county, concatenates
those vectors with the continuous and categorical borrower features, and trains
a LightGBM gradient boosted tree model on the enriched feature set.

This is the production model — the one that will be loaded by the FastAPI
inference service and called at recommendation time.

Two variants are trained sequentially:
  Model A — includes demographic features (for academic / fairness analysis)
  Model B — excludes protected characteristics (FHA / ECOA compliant)

Inputs:
  data/hmda_all_clean.parquet       — Model A training data
  data/hmda_all_model_b.parquet     — Model B training data
  data/feature_stats.parquet        — feature normalization statistics
  models/embedding_weights.pt       — lender / tract / county embeddings
  models/training_log.json          — embedding metadata (vocab sizes, dims)

Outputs:
  models/lgbm_model_a.pkl           — fitted Model A pipeline
  models/lgbm_model_b.pkl           — fitted Model B pipeline  (default)
  models/lgbm_model_a.txt           — LightGBM text format (portable)
  models/lgbm_model_b.txt           — LightGBM text format (portable)
  models/lgbm_feature_importance.parquet  — SHAP feature importance
  models/lgbm_training_summary.json — metrics, parameters, feature list

Usage:
  python 03_train_lgbm.py
  python 03_train_lgbm.py --no-model-a   # skip Model A, train B only
  python 03_train_lgbm.py --no-model-b   # skip Model B, train A only
  python 03_train_lgbm.py --n-estimators 2000 --learning-rate 0.05

Key hyperparameters:
  --n-estimators      Number of boosting rounds     (default: 1000)
  --learning-rate     Step size shrinkage            (default: 0.05)
  --num-leaves        Max leaves per tree            (default: 127)
  --min-child-samples Min samples per leaf           (default: 100)
  --subsample         Row subsampling fraction       (default: 0.8)
  --colsample-bytree  Feature subsampling fraction   (default: 0.8)
  --val-split         Validation fraction            (default: 0.1)
  --early-stopping    Early stopping rounds          (default: 50)
  --device            'gpu' or 'cpu'                 (default: 'gpu')
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

# ── Feature columns (must match 02_train_embeddings.py, minus leaky cols) ─────
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

# Columns that encode post-decision information — never used as features
LEAKY_COLS = [
    "interest_rate",
    "total_loan_costs",
    "combined_loan_to_value_ratio",
]

TARGET_COL = "label"

# Embedding index columns in the data
EMB_IDX_COLS = {
    "lender": "lender_idx",
    "tract":  "tract_idx",
    "county": "county_idx",
}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_weights(models_dir: Path) -> dict:
    """
    Load the embedding weight matrices saved by 02_train_embeddings.py.
    Returns dict of {name: np.ndarray shape [vocab, dim]}.
    """
    weights_path = models_dir / "embedding_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Embedding weights not found: {weights_path}\n"
            f"Run 02_train_embeddings.py first."
        )
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    log.info("Loaded embedding weights:")
    for name, arr in weights.items():
        log.info(f"  [{name}] shape: {arr.shape}")
    return weights


def lookup_embeddings(
    df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:
    """
    For each embedding dimension, look up the embedding vector for every row
    using the integer index column, and add those as new columns.

    A lender with lender_idx=0 (unknown) gets a zero vector.
    Returns the dataframe with new columns emb_lender_0 ... emb_lender_63 etc.
    """
    emb_frames = []

    for name, idx_col in EMB_IDX_COLS.items():
        if idx_col not in df.columns:
            log.warning(f"  Index column '{idx_col}' not found — skipping {name} embeddings.")
            continue

        W = weights[name]          # shape [vocab, dim]
        vocab_size, dim = W.shape

        # Clamp indices to valid range (defensive — unknown → 0)
        idx = df[idx_col].fillna(0).astype(int).clip(0, vocab_size - 1).values

        # Vectorised lookup: result shape [n_rows, dim]
        emb_matrix = W[idx]

        col_names = [f"emb_{name}_{i}" for i in range(dim)]
        emb_frames.append(
            pd.DataFrame(emb_matrix, columns=col_names, index=df.index)
        )
        log.info(f"  [{name}] looked up {len(df):,} rows → {dim} embedding columns")

    if emb_frames:
        df = pd.concat([df] + emb_frames, axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    feature_stats: dict,
    use_demographics: bool,
    weights: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble the full feature matrix:
      1. Look up embedding vectors for lender / tract / county
      2. Normalise continuous features
      3. Pass integer categoricals through as-is
      4. Label-encode demographic strings (Model A only)

    Returns (feature_df, feature_names).
    """
    # Drop leaky columns if present
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])

    # ── Embedding lookup ──────────────────────────────────────────────────────
    log.info("Looking up embedding vectors …")
    df = lookup_embeddings(df, weights)

    feature_cols = []

    # ── Embedding columns ─────────────────────────────────────────────────────
    for name in weights:
        emb_cols = [c for c in df.columns if c.startswith(f"emb_{name}_")]
        feature_cols.extend(emb_cols)

    # ── Normalised continuous features ────────────────────────────────────────
    for col in CONTINUOUS_COLS:
        if col not in df.columns:
            log.warning(f"  Continuous column '{col}' not found — skipping.")
            continue
        stats = feature_stats.get(col, {"mean": 0.0, "std": 1.0})
        mean = stats["mean"]
        std  = max(stats["std"], 1e-8)
        df[f"norm_{col}"] = ((df[col].fillna(mean).astype(np.float32) - mean) / std)
        feature_cols.append(f"norm_{col}")

    # ── Integer categorical features ──────────────────────────────────────────
    for col in INT_CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)
        feature_cols.append(col)

    # ── Demographic features (Model A only) ───────────────────────────────────
    if use_demographics:
        for col in DEMOGRAPHIC_COLS:
            if col not in df.columns:
                continue
            enc_col = f"enc_{col}"
            if hasattr(df[col], "cat"):
                df[enc_col] = df[col].cat.codes.astype(np.float32)
            else:
                df[enc_col] = pd.Categorical(df[col]).codes.astype(np.float32)
            feature_cols.append(enc_col)

    # ── hmda_year as a feature (captures temporal drift) ─────────────────────
    if "hmda_year" in df.columns:
        df["hmda_year"] = df["hmda_year"].astype(np.float32)
        feature_cols.append("hmda_year")

    log.info(f"Total features: {len(feature_cols)}")
    return df[feature_cols], feature_cols


def load_feature_stats(data_dir: Path) -> dict:
    path = data_dir / "feature_stats.parquet"
    if not path.exists():
        raise FileNotFoundError(f"feature_stats.parquet not found in {data_dir}")
    df = pd.read_parquet(path).set_index("feature")
    return {
        col: {"mean": float(df.loc[col, "mean"]), "std": float(df.loc[col, "std"])}
        for col in df.index
    }


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM training
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    args: argparse.Namespace,
    model_label: str,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with early stopping on validation AUC.
    """
    # Class weight for imbalanced labels
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info(f"  [{model_label}] scale_pos_weight: {scale_pos_weight:.3f}")

    params = dict(
        n_estimators      = args.n_estimators,
        learning_rate     = args.learning_rate,
        num_leaves        = args.num_leaves,
        min_child_samples = args.min_child_samples,
        subsample         = args.subsample,
        subsample_freq    = 1,
        colsample_bytree  = args.colsample_bytree,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        scale_pos_weight  = scale_pos_weight,
        objective         = "binary",
        metric            = "auc",
        device            = args.device,
        n_jobs            = -1,
        random_state      = 42,
        verbose           = -1,
    )

    model = lgb.LGBMClassifier(**params)

    log.info(f"  [{model_label}] Training LightGBM "
             f"(n_estimators={args.n_estimators}, lr={args.learning_rate}, "
             f"device={args.device}) …")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    log.info(f"  [{model_label}] Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    split_label: str,
    model_label: str,
) -> dict:
    """Compute and log a full set of evaluation metrics."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_roc":          round(float(roc_auc_score(y, probs)), 4),
        "avg_precision":    round(float(average_precision_score(y, probs)), 4),
        "brier_score":      round(float(brier_score_loss(y, probs)), 4),
    }

    log.info(f"\n  [{model_label}] {split_label} metrics:")
    log.info(f"    AUC-ROC:           {metrics['auc_roc']:.4f}")
    log.info(f"    Avg Precision:     {metrics['avg_precision']:.4f}")
    log.info(f"    Brier Score:       {metrics['brier_score']:.4f}")
    log.info(f"\n{classification_report(y, preds, target_names=['Denied','Approved'])}")

    return metrics


def compute_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    model_label: str,
) -> pd.DataFrame:
    """Extract LightGBM's built-in split-based feature importance."""
    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
        "model":      model_label,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    log.info(f"\n  [{model_label}] Top 20 features by importance:")
    log.info(df_imp.head(20).to_string(index=False))

    return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# Fairness metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_disparate_impact(
    probs: np.ndarray,
    df_val: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """
    Compute disparate impact ratio for race and ethnicity groups.
    Disparate Impact = approval_rate(minority) / approval_rate(majority)
    ECOA threshold: ratio < 0.8 warrants scrutiny.
    """
    results = {}
    preds = (probs >= threshold).astype(int)

    for col in ["derived_race", "derived_ethnicity"]:
        if col not in df_val.columns:
            continue
        group_rates = {}
        for group, mask in df_val.groupby(col).groups.items():
            group_preds = preds[mask]
            if len(group_preds) < 100:   # too small to be meaningful
                continue
            group_rates[str(group)] = float(group_preds.mean())

        if not group_rates:
            continue

        max_rate = max(group_rates.values())
        di_ratios = {
            grp: round(rate / max_rate, 4)
            for grp, rate in group_rates.items()
        }
        results[col] = {
            "approval_rates": {k: round(v, 4) for k, v in group_rates.items()},
            "disparate_impact_ratios": di_ratios,
            "groups_below_0.8": [g for g, r in di_ratios.items() if r < 0.8],
        }
        log.info(f"\n  Disparate Impact [{col}]:")
        for grp, ratio in sorted(di_ratios.items(), key=lambda x: x[1]):
            flag = "  ⚠ < 0.8" if ratio < 0.8 else ""
            log.info(f"    {grp:<45} DI={ratio:.4f}{flag}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(
    data_path: Path,
    weights: dict,
    feature_stats: dict,
    use_demographics: bool,
    model_label: str,
    args: argparse.Namespace,
    summary: dict,
) -> None:
    """Full training pipeline for one model variant (A or B)."""
    t0 = time.time()

    log.info(f"\n{'═'*60}")
    log.info(f"  Training {model_label}  "
             f"({'with' if use_demographics else 'without'} demographics)")
    log.info(f"{'═'*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading {data_path.name} …")
    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df):,} rows")

    # ── Build feature matrix ──────────────────────────────────────────────────
    X, feature_names = build_feature_matrix(
        df, feature_stats, use_demographics, weights)
    y = df[TARGET_COL].astype(int).values
    del df

    log.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # ── Train / val split ─────────────────────────────────────────────────────
    log.info(f"Splitting {args.val_split:.0%} validation …")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_split,
        random_state=42,
        stratify=y,
    )
    log.info(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
    del X

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_lgbm(X_train, y_train, X_val, y_val, args, model_label)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_metrics = evaluate_model(model, X_train, y_train, "TRAIN", model_label)
    val_metrics   = evaluate_model(model, X_val,   y_val,   "VAL",   model_label)

    # ── Fairness metrics (Model A only — needs demographic columns) ───────────
    fairness = {}
    # Compute val probabilities (used for fairness metrics and potential future use)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Fairness metrics (Model A only)
    fairness = {}
    if use_demographics:
        log.info("\n  Computing disparate impact metrics …")
        df_all = pd.read_parquet(data_path, columns=DEMOGRAPHIC_COLS + [TARGET_COL])
        _, df_val_demo = train_test_split(
            df_all, test_size=args.val_split, random_state=42,
            stratify=df_all[TARGET_COL])
        df_val_demo = df_val_demo.reset_index(drop=True)
        fairness = compute_disparate_impact(val_probs, df_val_demo)

    # ── Feature importance ────────────────────────────────────────────────────
    imp_df = compute_feature_importance(model, feature_names, model_label)

    # ── Persist model ─────────────────────────────────────────────────────────
    suffix = "a" if use_demographics else "b"

    # joblib pickle — includes full sklearn-compatible interface
    pkl_path = MODELS_DIR / f"lgbm_model_{suffix}.pkl"
    joblib.dump(model, pkl_path)
    log.info(f"\n  Saved: {pkl_path.name}  ({pkl_path.stat().st_size/1e6:.1f} MB)")

    # LightGBM native text format — portable, can be loaded without joblib
    txt_path = MODELS_DIR / f"lgbm_model_{suffix}.txt"
    model.booster_.save_model(str(txt_path))
    log.info(f"  Saved: {txt_path.name}")

    # Feature importance
    imp_path = MODELS_DIR / f"lgbm_feature_importance_{suffix}.parquet"
    imp_df.to_parquet(imp_path, index=False)
    log.info(f"  Saved: {imp_path.name}")

    elapsed = time.time() - t0
    summary[model_label] = {
        "data_file":       data_path.name,
        "n_rows":          len(X_train) + len(X_val),
        "n_features":      len(feature_names),
        "best_iteration":  model.best_iteration_,
        "train_metrics":   train_metrics,
        "val_metrics":     val_metrics,
        "fairness":        fairness,
        "elapsed_seconds": round(elapsed, 1),
        "feature_names":   feature_names,
    }

    log.info(
        f"\n  {model_label} complete in {elapsed/60:.1f} min  "
        f"| val AUC: {val_metrics['auc_roc']:.4f}  "
        f"| best iter: {model.best_iteration_}"
    )


def run(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summary = {}

    # ── Load shared resources ─────────────────────────────────────────────────
    log.info("Loading embedding weights …")
    weights = load_embedding_weights(MODELS_DIR)

    log.info("Loading feature statistics …")
    feature_stats = load_feature_stats(args.data_dir)

    # ── Model A ───────────────────────────────────────────────────────────────
    if not args.no_model_a:
        data_a = args.data_dir / "hmda_all_clean.parquet"
        if not data_a.exists():
            log.error(f"Model A data not found: {data_a}")
            sys.exit(1)
        train_one_model(
            data_path        = data_a,
            weights          = weights,
            feature_stats    = feature_stats,
            use_demographics = True,
            model_label      = "Model_A",
            args             = args,
            summary          = summary,
        )

    # ── Model B ───────────────────────────────────────────────────────────────
    if not args.no_model_b:
        data_b = args.data_dir / "hmda_all_model_b.parquet"
        if not data_b.exists():
            log.error(f"Model B data not found: {data_b}")
            sys.exit(1)
        train_one_model(
            data_path        = data_b,
            weights          = weights,
            feature_stats    = feature_stats,
            use_demographics = False,
            model_label      = "Model_B",
            args             = args,
            summary          = summary,
        )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = MODELS_DIR / "lgbm_training_summary.json"
    # feature_names lists can be long — keep them but truncate for readability
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved: {summary_path.name}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.time() - t_start
    log.info(f"\n{'─'*60}")
    log.info(f"  All models trained in {total/60:.1f} min")
    for label, s in summary.items():
        log.info(
            f"  {label}:  val_auc={s['val_metrics']['auc_roc']:.4f}  "
            f"best_iter={s['best_iteration']}  "
            f"features={s['n_features']}"
        )
    log.info(f"\n  Ready for:  python main.py")
    log.info(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LightGBM models on HMDA data with pre-trained embeddings"
    )
    p.add_argument("--data-dir",         type=Path,  default=DATA_DIR)
    p.add_argument("--no-model-a",       action="store_true",
                   help="Skip Model A (with demographics)")
    p.add_argument("--no-model-b",       action="store_true",
                   help="Skip Model B (no demographics)")
    p.add_argument("--n-estimators",     type=int,   default=1000)
    p.add_argument("--learning-rate",    type=float, default=0.05)
    p.add_argument("--num-leaves",       type=int,   default=127)
    p.add_argument("--min-child-samples",type=int,   default=100)
    p.add_argument("--subsample",        type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--val-split",        type=float, default=0.1)
    p.add_argument("--early-stopping",   type=int,   default=50)
    p.add_argument("--device",           type=str,   default="gpu",
                   choices=["gpu", "cpu"],
                   help="LightGBM device (default: gpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_model_a and args.no_model_b:
        log.error("Both --no-model-a and --no-model-b set — nothing to train.")
        sys.exit(1)

    if not args.data_dir.exists():
        log.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    run(args)