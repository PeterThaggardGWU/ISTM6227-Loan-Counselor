"""
compute_fairness.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Standalone Fairness Analysis
ISTM6218 Group Project

Loads the saved Model A LightGBM artifact and the training data, rebuilds
the validation split (using the same random_state=42 as training so the
split is identical), and computes fairness metrics without retraining.

Outputs:
  models/fairness_report.json     — full disparate impact results
  models/fairness_report.txt      — human-readable summary

Usage:
  python compute_fairness.py
  python compute_fairness.py --threshold 0.5
  python compute_fairness.py --val-split 0.1  (must match training)
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parent
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

DEMOGRAPHIC_COLS = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",
]
TARGET_COL = "label"


def compute_disparate_impact(
    probs: np.ndarray,
    df_val: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Compute disparate impact ratio for each demographic group column.

    Disparate Impact = approval_rate(group) / approval_rate(highest_group)
    ECOA / fair lending threshold: ratio < 0.8 warrants scrutiny.

    Also computes equalized odds (TPR and FPR per group) for comparison
    between Model A and Model B — a key academic finding for the project.
    """
    preds  = (probs >= threshold).astype(int)
    labels = df_val[TARGET_COL].values
    results = {}

    for col in DEMOGRAPHIC_COLS:
        if col not in df_val.columns:
            continue

        group_stats = {}
        for group_name, group_idx in df_val.groupby(col).groups.items():
            if len(group_idx) < 100:
                continue   # too small to be meaningful

            g_probs  = probs[group_idx]
            g_preds  = preds[group_idx]
            g_labels = labels[group_idx]

            n_total    = len(g_labels)
            approval_r = float(g_preds.mean())

            # Equalized odds components
            pos_mask = g_labels == 1
            neg_mask = g_labels == 0
            tpr = float(g_preds[pos_mask].mean())  if pos_mask.sum() > 0 else None
            fpr = float(g_preds[neg_mask].mean())  if neg_mask.sum() > 0 else None
            avg_prob = float(g_probs.mean())

            group_stats[str(group_name)] = {
                "n":               n_total,
                "approval_rate":   round(approval_r, 4),
                "avg_probability": round(avg_prob, 4),
                "tpr":             round(tpr, 4) if tpr is not None else None,
                "fpr":             round(fpr, 4) if fpr is not None else None,
            }

        if not group_stats:
            continue

        # Disparate impact ratios relative to the highest approval rate group
        max_rate   = max(s["approval_rate"] for s in group_stats.values())
        below_0_8  = []
        for grp, stats in group_stats.items():
            di = stats["approval_rate"] / max_rate if max_rate > 0 else 1.0
            stats["disparate_impact_ratio"] = round(di, 4)
            if di < 0.8:
                below_0_8.append(grp)

        results[col] = {
            "groups":             group_stats,
            "groups_below_0.8":   below_0_8,
            "threshold_used":     threshold,
        }

        # Log summary
        log.info(f"\n  Disparate Impact — {col}:")
        sorted_groups = sorted(
            group_stats.items(),
            key=lambda x: x[1]["approval_rate"],
            reverse=True,
        )
        for grp, stats in sorted_groups:
            flag = "  ⚠  < 0.8 threshold" if stats["disparate_impact_ratio"] < 0.8 else ""
            log.info(
                f"    {grp:<45}  "
                f"approval={stats['approval_rate']:.3f}  "
                f"DI={stats['disparate_impact_ratio']:.3f}  "
                f"n={stats['n']:,}{flag}"
            )
        if below_0_8:
            log.info(f"  ⚠  Groups below 0.8 threshold: {below_0_8}")
        else:
            log.info(f"  ✓  All groups above 0.8 disparate impact threshold")

    return results


def write_text_report(results: dict, output_path: Path) -> None:
    """Write a human-readable text summary for inclusion in the project report."""
    lines = [
        "Loan Arranger — Fairness Analysis Report",
        "Model A (includes demographic features)",
        "=" * 60,
        "",
    ]

    for col, data in results.items():
        lines.append(f"Dimension: {col}")
        lines.append("-" * 40)
        sorted_groups = sorted(
            data["groups"].items(),
            key=lambda x: x[1]["approval_rate"],
            reverse=True,
        )
        lines.append(
            f"  {'Group':<45}  {'Approval':>8}  {'DI Ratio':>8}  "
            f"{'TPR':>6}  {'FPR':>6}  {'N':>8}"
        )
        lines.append(f"  {'-'*90}")
        for grp, stats in sorted_groups:
            flag = " ⚠" if stats["disparate_impact_ratio"] < 0.8 else ""
            tpr = f"{stats['tpr']:.3f}" if stats["tpr"] is not None else "  N/A"
            fpr = f"{stats['fpr']:.3f}" if stats["fpr"] is not None else "  N/A"
            lines.append(
                f"  {grp:<45}  "
                f"{stats['approval_rate']:>8.3f}  "
                f"{stats['disparate_impact_ratio']:>8.3f}  "
                f"{tpr:>6}  {fpr:>6}  "
                f"{stats['n']:>8,}{flag}"
            )
        if data["groups_below_0.8"]:
            lines.append(f"\n  ⚠  Groups below 0.8 DI threshold: {data['groups_below_0.8']}")
        else:
            lines.append(f"\n  ✓  All groups above 0.8 DI threshold")
        lines.append("")

    lines += [
        "Notes:",
        "  DI Ratio = group approval rate / highest group approval rate",
        "  ECOA / fair lending scrutiny threshold: DI Ratio < 0.8",
        "  TPR = True Positive Rate (approved applicants correctly predicted approved)",
        "  FPR = False Positive Rate (denied applicants incorrectly predicted approved)",
        "  Model B excludes all demographic features and is the default deployment model.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Text report written: {output_path}")


def run(args: argparse.Namespace) -> None:

    # ── Load Model A ──────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "lgbm_model_a.pkl"
    if not model_path.exists():
        log.error(f"Model A not found: {model_path}\nRun 03_train_lgbm.py first.")
        raise SystemExit(1)

    log.info(f"Loading {model_path.name} ...")
    model = joblib.load(model_path)
    log.info(f"  Best iteration: {model.best_iteration_}")

    # ── Load validation data ──────────────────────────────────────────────────
    # We need the same feature matrix that was used during training PLUS
    # the demographic columns for grouping. Load both together then split.
    log.info("Loading feature list from feature importance parquet ...")
    imp_path = MODELS_DIR / "lgbm_feature_importance_a.parquet"
    if not imp_path.exists():
        log.error(f"Feature importance file not found: {imp_path}")
        raise SystemExit(1)

    feature_names = pd.read_parquet(imp_path)["feature"].tolist()
    log.info(f"  Features: {len(feature_names)}")

    # Load the full Model A parquet — we need features + demographics + label
    data_path = DATA_DIR / "hmda_all_clean.parquet"
    log.info(f"Loading {data_path.name} ...")

    # Load only the columns we need to keep memory manageable
    demo_and_label = DEMOGRAPHIC_COLS + [TARGET_COL]
    df_full = pd.read_parquet(data_path)

    # Reproduce the exact same validation split as training
    log.info(f"Reproducing val split (val_split={args.val_split}, random_state=42) ...")
    df_train, df_val = train_test_split(
        df_full,
        test_size=args.val_split,
        random_state=42,
        stratify=df_full[TARGET_COL],
    )
    df_val = df_val.reset_index(drop=True)
    log.info(f"  Val rows: {len(df_val):,}")
    del df_train, df_full

    # ── Build feature matrix for val set ─────────────────────────────────────
    # Keep only the feature columns that exist in the data
    available_features = [f for f in feature_names if f in df_val.columns]
    missing = set(feature_names) - set(available_features)
    if missing:
        log.warning(f"  Missing features (will be filled with 0): {missing}")

    X_val = df_val[available_features].copy()
    # Fill any missing feature columns with zeros
    for col in missing:
        X_val[col] = 0.0
    X_val = X_val[feature_names]   # ensure correct column order

    # ── Score ─────────────────────────────────────────────────────────────────
    log.info("Scoring validation set ...")
    probs = model.predict_proba(X_val)[:, 1]
    log.info(f"  Scored {len(probs):,} rows  (threshold={args.threshold})")

    # ── Compute fairness metrics ───────────────────────────────────────────────
    log.info("\nComputing disparate impact metrics ...")
    fairness_results = compute_disparate_impact(
        probs=probs,
        df_val=df_val[demo_and_label],
        threshold=args.threshold,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = MODELS_DIR / "fairness_report.json"
    with open(json_path, "w") as f:
        json.dump(fairness_results, f, indent=2)
    log.info(f"\nSaved: {json_path.name}")

    txt_path = MODELS_DIR / "fairness_report.txt"
    write_text_report(fairness_results, txt_path)

    log.info(
        f"\n{'─'*60}\n"
        f"  Fairness analysis complete\n"
        f"  Threshold:    {args.threshold}\n"
        f"  Val rows:     {len(probs):,}\n"
        f"  JSON report:  {json_path}\n"
        f"  Text report:  {txt_path}\n"
        f"{'─'*60}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute fairness / disparate impact metrics for Model A "
                    "without retraining"
    )
    p.add_argument("--val-split",  type=float, default=0.1,
                   help="Validation fraction — must match the value used in "
                        "03_train_lgbm.py (default: 0.1)")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Classification threshold for approval/denial (default: 0.5)")
    p.add_argument("--data-dir",   type=Path,  default=DATA_DIR)
    p.add_argument("--models-dir", type=Path,  default=MODELS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR   = args.data_dir
    MODELS_DIR = args.models_dir
    run(args)