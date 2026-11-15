"""
02_train_embeddings.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — Phase 1: Entity Embedding Network Training
ISTM6218 Group Project

Trains a feed-forward neural network on the merged HMDA dataset to learn
dense vector representations (embeddings) for high-cardinality categorical
variables: lender (LEI), census tract, and county.

The network is not the final production model — its purpose is to produce
meaningful embedding weight matrices that capture lender and geographic
approval behavior patterns from millions of historical applications.  Those
weights are exported and consumed by 03_train_lgbm.py as enriched input
features for the gradient-boosted model.

Inputs (from data/):
  hmda_all_clean.parquet      — merged Model A data (all features)
  lei_index.parquet           — LEI → lender_idx mapping  (for vocab size)
  censustract_index.parquet   — tract → tract_idx mapping (for vocab size)
  countycode_index.parquet    — county → county_idx mapping (for vocab size)
  feature_stats.parquet       — mean/std for continuous feature normalisation

Outputs (to models/):
  embedding_weights.pt        — dict of learned embedding matrices:
                                  { "lender":  Tensor [vocab, 64],
                                    "tract":   Tensor [vocab, 32],
                                    "county":  Tensor [vocab, 16] }
  embedding_network.pt        — full model checkpoint (resume training / debug)
  training_log.json           — loss / AUC history per epoch

Usage:
  python 02_train_embeddings.py
  python 02_train_embeddings.py --epochs 20 --batch-size 8192
  python 02_train_embeddings.py --model-a          # default
  python 02_train_embeddings.py --model-b          # no demographics

Key hyperparameters (override via CLI):
  --epochs          Training epochs          (default: 10)
  --batch-size      Rows per mini-batch      (default: 8192)
  --lr              Learning rate            (default: 1e-3)
  --val-split       Validation fraction      (default: 0.1)
  --emb-lender      Lender embedding dim     (default: 64)
  --emb-tract       Census tract emb dim     (default: 32)
  --emb-county      County embedding dim     (default: 16)
  --hidden          Hidden layer sizes       (default: 256 128 64)
  --dropout         Dropout rate             (default: 0.3)
  --patience        Early stopping patience  (default: 3)
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

# ── Feature definitions ───────────────────────────────────────────────────────
# Integer index columns fed into embedding layers
EMBEDDING_COLS = [
    ("lender_idx", "lender"),   # (column_in_data, embedding_name)
    ("tract_idx",  "tract"),
    ("county_idx", "county"),
]

# Continuous columns fed directly into the dense layers (after normalisation)
CONTINUOUS_COLS = [
    "loan_amount",
    "income",
    "property_value",
    #"loan_to_value_ratio",
    "debt_to_income_ratio",
    #"combined_loan_to_value_ratio",
    #"interest_rate",
]

# Low-cardinality integer categoricals — treated as continuous after casting
# (they have meaningful ordinal structure, e.g. loan_type 1=conv, 2=FHA ...)
INT_CAT_COLS = [
    "loan_purpose",
    "loan_type",
    "lien_status",
    "occupancy_type",
]

# Demographic columns present in Model A but not Model B
DEMOGRAPHIC_COLS = [
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",
    "co_applicant_age",
]

TARGET_COL = "label"


# ─────────────────────────────────────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingNetwork(nn.Module):
    """
    Feed-forward network with entity embeddings for categorical variables.

    Architecture:
      [lender_emb | tract_emb | county_emb | continuous_features]
          → BatchNorm
          → Dense(hidden[0]) → ReLU → Dropout
          → Dense(hidden[1]) → ReLU → Dropout
          → ...
          → Dense(1) → Sigmoid
    """

    def __init__(
        self,
        vocab_sizes: dict,        # {"lender": N, "tract": N, "county": N}
        emb_dims: dict,           # {"lender": 64, "tract": 32, "county": 16}
        n_continuous: int,
        hidden_sizes: list[int],
        dropout: float,
    ):
        super().__init__()

        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=vocab_sizes[name],
                embedding_dim=emb_dims[name],
                padding_idx=0,    # index 0 = <UNKNOWN>, always maps to zero vector
            )
            for name in vocab_sizes
        })

        # Total input width = sum of all embedding dims + continuous features
        total_emb_dim = sum(emb_dims.values())
        input_dim = total_emb_dim + n_continuous

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_features = input_dim
        for out_features in hidden_sizes:
            layers += [
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.BatchNorm1d(out_features),
                nn.Dropout(p=dropout),
            ]
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.dense = nn.Sequential(*layers)

    def forward(self, cat_inputs: dict, cont_input: torch.Tensor) -> torch.Tensor:
        """
        cat_inputs:  dict of {name: LongTensor[batch]}
        cont_input:  FloatTensor[batch, n_continuous]
        returns:     FloatTensor[batch] — predicted approval probability
        """
        emb_parts = [
            self.embedding_layers[name](cat_inputs[name])
            for name in self.embedding_layers
        ]
        x = torch.cat(emb_parts + [cont_input], dim=1)
        x = self.input_bn(x)
        x = self.dense(x)
        return torch.sigmoid(x).squeeze(1)

    def get_embedding_weights(self) -> dict:
        """Return a plain dict of {name: numpy array} for export."""
        return {
            name: layer.weight.detach().cpu().numpy()
            for name, layer in self.embedding_layers.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab_sizes(data_dir: Path, emb_dims: dict) -> dict:
    """
    Read the unified index parquets to get the vocabulary size for each
    embedding dimension.  vocab_size = max(idx) + 1 to include the 0 token.
    """
    index_files = {
        "lender": data_dir / "lei_index.parquet",
        "tract":  data_dir / "censustract_index.parquet",
        "county": data_dir / "countycode_index.parquet",
    }
    vocab_sizes = {}
    for name, path in index_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Index file not found: {path}\n"
                f"Run 01b_merge.py first."
            )
        df = pd.read_parquet(path)
        # The idx column is the second column (first is the source value)
        idx_col = df.columns[1]
        vocab_sizes[name] = int(df[idx_col].max()) + 1  # +1 for 0-index unknown
        log.info(f"  Vocab [{name}]: {vocab_sizes[name]:,}  "
                 f"(emb_dim={emb_dims[name]})")
    return vocab_sizes


def load_feature_stats(data_dir: Path) -> dict:
    """Load mean/std for continuous feature normalisation."""
    path = data_dir / "feature_stats.parquet"
    if not path.exists():
        raise FileNotFoundError(f"feature_stats.parquet not found in {data_dir}")
    df = pd.read_parquet(path).set_index("feature")
    return {
        col: {"mean": float(df.loc[col, "mean"]), "std": float(df.loc[col, "std"])}
        for col in df.index
    }


def prepare_tensors(
    df: pd.DataFrame,
    feature_stats: dict,
    use_demographics: bool,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Convert a DataFrame into the tensors the network expects.

    Returns:
      cat_tensors   — dict of {emb_name: LongTensor[N]}
      cont_tensor   — FloatTensor[N, n_continuous]
      label_tensor  — FloatTensor[N]
    """
    # ── Categorical (embedding) inputs ────────────────────────────────────────
    cat_tensors = {}
    col_map = {
        "lender": "lender_idx",
        "tract":  "tract_idx",
        "county": "county_idx",
    }
    for name, col in col_map.items():
        values = df[col].fillna(0).astype(int).values
        cat_tensors[name] = torch.tensor(values, dtype=torch.long, device=device)

    # ── Continuous inputs ──────────────────────────────────────────────────────
    cont_cols = []

    # Normalised continuous features
    for col in CONTINUOUS_COLS:
        if col not in df.columns:
            log.warning(f"  Continuous column '{col}' not found — filling with 0.")
            cont_cols.append(np.zeros(len(df), dtype=np.float32))
            continue
        stats = feature_stats.get(col, {"mean": 0.0, "std": 1.0})
        mean, std = stats["mean"], max(stats["std"], 1e-8)
        vals = df[col].fillna(mean).astype(np.float32).values
        cont_cols.append((vals - mean) / std)

    # Integer categoricals as continuous (already ordinal-ish)
    for col in INT_CAT_COLS:
        if col in df.columns:
            vals = df[col].fillna(0).astype(np.float32).values
            cont_cols.append(vals)

    # Demographic columns (Model A only) — label-encoded as integers
    if use_demographics:
        for col in DEMOGRAPHIC_COLS:
            if col not in df.columns:
                continue
            # Convert category codes to float
            if hasattr(df[col], "cat"):
                vals = df[col].cat.codes.astype(np.float32).values
            else:
                vals = pd.Categorical(df[col]).codes.astype(np.float32)
            cont_cols.append(vals)

    cont_array = np.stack(cont_cols, axis=1).astype(np.float32)
    cont_tensor = torch.tensor(cont_array, dtype=torch.float32, device=device)

    # ── Labels ─────────────────────────────────────────────────────────────────
    labels = df[TARGET_COL].astype(np.float32).values
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

    return cat_tensors, cont_tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: EmbeddingNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    emb_names: list[str],
) -> tuple[float, float]:
    """One full pass over training data.  Returns (avg_loss, auc)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        # Batch layout: [lender_idx, tract_idx, county_idx, cont_features..., label]
        n_emb = len(emb_names)
        cat_inputs = {name: batch[i].to(device) for i, name in enumerate(emb_names)}
        cont = batch[n_emb].to(device)
        labels = batch[n_emb + 1].to(device)

        optimizer.zero_grad()
        preds = model(cat_inputs, cont)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(
    model: EmbeddingNetwork,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    emb_names: list[str],
) -> tuple[float, float]:
    """Evaluate on validation set.  Returns (avg_loss, auc)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        n_emb = len(emb_names)
        cat_inputs = {name: batch[i].to(device) for i, name in enumerate(emb_names)}
        cont = batch[n_emb].to(device)
        labels = batch[n_emb + 1].to(device)

        preds = model(cat_inputs, cont)
        loss = criterion(preds, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


def make_dataloader(
    cat_tensors: dict,
    cont_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    emb_names: list[str],
) -> DataLoader:
    """Pack tensors into a TensorDataset and wrap in a DataLoader."""
    tensors = [cat_tensors[name] for name in emb_names]
    tensors += [cont_tensor, label_tensor]
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,   # tensors already on GPU
        num_workers=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"GPU: {gpu_name}  (CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available — training on CPU (will be slow).")

    # ── Embedding dimensions ──────────────────────────────────────────────────
    emb_dims = {
        "lender": args.emb_lender,
        "tract":  args.emb_tract,
        "county": args.emb_county,
    }
    emb_names = list(emb_dims.keys())

    # ── Load vocab sizes from unified index files ─────────────────────────────
    log.info("Loading vocabulary sizes …")
    vocab_sizes = load_vocab_sizes(args.data_dir, emb_dims)

    # ── Load feature stats for normalisation ──────────────────────────────────
    log.info("Loading feature statistics …")
    feature_stats = load_feature_stats(args.data_dir)

    # ── Load training data ────────────────────────────────────────────────────
    data_file = (
        args.data_dir / "hmda_all_clean.parquet"
        if not args.model_b
        else args.data_dir / "hmda_all_model_b.parquet"
    )
    if not data_file.exists():
        log.error(f"Data file not found: {data_file}\nRun 01b_merge.py first.")
        sys.exit(1)

    log.info(f"Loading {data_file.name} …")
    t0 = time.time()
    df = pd.read_parquet(data_file)
    log.info(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    LEAKY_COLS = ["interest_rate", "total_loan_costs", "combined_loan_to_value_ratio"]
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])
    log.info(f"Dropped leaky post-decision columns: {LEAKY_COLS}")

    use_demographics = not args.model_b and any(
        c in df.columns for c in DEMOGRAPHIC_COLS
    )

    # ── Train / validation split ──────────────────────────────────────────────
    log.info(f"Splitting {args.val_split:.0%} validation …")
    df_train, df_val = train_test_split(
        df, test_size=args.val_split, random_state=42,
        stratify=df[TARGET_COL]   # preserve class balance
    )
    log.info(f"  Train: {len(df_train):,}  Val: {len(df_val):,}")
    del df  # free memory

    # ── Build tensors ─────────────────────────────────────────────────────────
    log.info("Building tensors …")
    cat_tr, cont_tr, lbl_tr = prepare_tensors(
        df_train, feature_stats, use_demographics, device)
    cat_va, cont_va, lbl_va = prepare_tensors(
        df_val,   feature_stats, use_demographics, device)
    del df_train, df_val

    n_continuous = cont_tr.shape[1]
    log.info(f"  Continuous features: {n_continuous}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = make_dataloader(
        cat_tr, cont_tr, lbl_tr, args.batch_size, shuffle=True,  emb_names=emb_names)
    val_loader   = make_dataloader(
        cat_va, cont_va, lbl_va, args.batch_size, shuffle=False, emb_names=emb_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmbeddingNetwork(
        vocab_sizes=vocab_sizes,
        emb_dims=emb_dims,
        n_continuous=n_continuous,
        hidden_sizes=args.hidden,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")
    log.info(f"\n{model}\n")

    # ── Optimiser and loss ────────────────────────────────────────────────────
    # Positive class weight to handle approval/denial imbalance
    n_pos = int(lbl_tr.sum().item())
    n_neg = len(lbl_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log.info(f"Class balance — pos: {n_pos:,}  neg: {n_neg:,}  "
             f"pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCELoss()   # labels already in [0,1]; sigmoid in forward()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*60}")
    log.info(f"Training for up to {args.epochs} epochs  "
             f"(early stopping patience={args.patience})")
    log.info(f"{'─'*60}")

    best_val_auc   = 0.0
    best_epoch     = 0
    patience_count = 0
    training_log   = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, emb_names)
        val_loss, val_auc = eval_epoch(
            model, val_loader, criterion, device, emb_names)

        scheduler.step(val_auc)
        elapsed = time.time() - t_ep

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_auc={train_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
            f"({elapsed:.0f}s)"
        )

        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_auc":  round(train_auc, 6),
            "val_loss":   round(val_loss, 6),
            "val_auc":    round(val_auc, 6),
            "elapsed_s":  round(elapsed, 1),
        })

        # Save best model checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_count = 0
            ckpt_path = MODELS_DIR / "embedding_network.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc":     val_auc,
                "vocab_sizes": vocab_sizes,
                "emb_dims":    emb_dims,
                "n_continuous": n_continuous,
                "hidden_sizes": args.hidden,
                "dropout":     args.dropout,
                "use_demographics": use_demographics,
            }, ckpt_path)
            log.info(f"  ✓ New best — checkpoint saved (val_auc={val_auc:.4f})")
        else:
            patience_count += 1
            log.info(f"  No improvement ({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    log.info(f"\nBest epoch: {best_epoch}  Best val AUC: {best_val_auc:.4f}")

    # ── Export embedding weights ───────────────────────────────────────────────
    log.info("\nExporting embedding weights …")

    # Reload best checkpoint
    ckpt = torch.load(MODELS_DIR / "embedding_network.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    emb_weights = model.get_embedding_weights()

    weights_path = MODELS_DIR / "embedding_weights.pt"
    torch.save(emb_weights, weights_path)

    for name, arr in emb_weights.items():
        log.info(f"  [{name}] shape: {arr.shape}  "
                 f"(vocab={arr.shape[0]}, dim={arr.shape[1]})")
    log.info(f"Saved: {weights_path}")

    # ── Save training log ──────────────────────────────────────────────────────
    log_path = MODELS_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "best_epoch":   best_epoch,
            "best_val_auc": best_val_auc,
            "emb_dims":     emb_dims,
            "vocab_sizes":  vocab_sizes,
            "n_continuous": n_continuous,
            "hidden_sizes": args.hidden,
            "dropout":      args.dropout,
            "use_demographics": use_demographics,
            "history":      training_log,
        }, f, indent=2)
    log.info(f"Saved: {log_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.time() - t_start
    log.info(
        f"\n{'─'*60}\n"
        f"  Training complete in {total/60:.1f} min\n"
        f"  Device:        {device} ({gpu_name if device.type=='cuda' else 'CPU'})\n"
        f"  Best val AUC:  {best_val_auc:.4f}  (epoch {best_epoch})\n"
        f"  Weights:       {weights_path}\n"
        f"  Ready for:     python 03_train_lgbm.py\n"
        f"{'─'*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train entity embedding network on HMDA data (Phase 1)"
    )
    p.add_argument("--data-dir",   type=Path,  default=DATA_DIR)
    p.add_argument("--model-b",    action="store_true",
                   help="Use Model B data (no demographics)")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=8192)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val-split",  type=float, default=0.1)
    p.add_argument("--emb-lender", type=int,   default=64)
    p.add_argument("--emb-tract",  type=int,   default=32)
    p.add_argument("--emb-county", type=int,   default=16)
    p.add_argument("--hidden",     type=int,   nargs="+", default=[256, 128, 64],
                   help="Hidden layer sizes e.g. --hidden 256 128 64")
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=3,
                   help="Early stopping patience (epochs without improvement)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.data_dir.exists():
        log.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    run(args)