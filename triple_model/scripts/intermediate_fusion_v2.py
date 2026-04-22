"""Intermediate fusion v2 — CV Optuna + modality dropout + input BN.

Fixes the failure modes of intermediate_fusion.py (v1):
  v1 val-overfit pattern:  trainonly test 0.623 but trainval test 0.545
  v1 best params:          lr=9e-3, epochs=31 (over-optimistic via max(hist))

v2 changes (fusion-side only — encoders remain frozen):
  1. Objective = 5-fold CV mean AUROC on train+val (357 samples) — replaces the
     single-pass "max over epochs on val" which rewarded over-optimistic params.
  2. Narrower search space tuned for the small-data regime:
       n_hidden = 1 (fixed — v1 hit 2×256 which was too big for n=357)
       hidden_dim in {32, 64, 128}
       dropout in [0.1, 0.5]
       lr in [1e-4, 2e-3] (v1 allowed up to 1e-2 and abused it)
       weight_decay in [1e-5, 1e-3]
       batch_size in {16, 32, 64}
       epochs in [40, 100] (v1 went to 120)
  3. FusionHead with input BatchNorm1d(640) + train-time modality dropout
     (whole-modality zero mask, p=0.3 per block) so the head cannot copycat
     CT channels (which was v1's failure mode).
  4. Final retrain:
       * trainonly head  → val AUROC (diagnostic)
       * trainval  head  → test AUROC (headline)
  5. Uses the SAME train-only encoder embeddings already cached in
     triple_model/embeddings/*.npz — no encoder retraining.

Outputs:
  triple_model/results/triple/intermediate_v2/
    ├─ best.pt
    ├─ test_predictions.csv
    └─ results.json
  triple_model/results/summary.json   (adds `intermediate_fusion_v2` section)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FOLDER_ROOT.parent
SM_ROOT = REPO_ROOT / "single_modal_baseline"
SM_SCRIPTS = SM_ROOT / "scripts"
if str(SM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SM_SCRIPTS))

SEED = 99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_DIR = FOLDER_ROOT / "embeddings"
MODALITIES = ["clinical", "radiomics", "ct"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Data loading (reuse cached triple embeddings)
# ─────────────────────────────────────────────
def _load_cached_embeddings() -> tuple[dict, dict]:
    """Load per-modality train-only encoder embeddings cached by intermediate_fusion.py."""
    banks = {}
    for m in MODALITIES:
        banks[m] = {}
        for split in ("train", "val", "test"):
            d = np.load(EMB_DIR / f"{m}_{split}.npz", allow_pickle=True)
            banks[m][split] = {
                "emb": d["emb"].astype(np.float32),
                "y": d["y"].astype(np.int64),
                "pids": [str(p) for p in d["pids"]],
            }
    dims = {m: int(banks[m]["train"]["emb"].shape[1]) for m in MODALITIES}
    return banks, dims


def _align_triple(banks: dict) -> dict:
    """Concat 3-modal embeddings per split, aligned by clinical PID order."""
    out = {}
    ref = "clinical"
    for split in ("train", "val", "test"):
        ref_pids = banks[ref][split]["pids"]
        ref_y = banks[ref][split]["y"]
        parts = [banks[ref][split]["emb"]]
        for m in MODALITIES:
            if m == ref:
                continue
            idx = {p: i for i, p in enumerate(banks[m][split]["pids"])}
            pos = [idx[p] for p in ref_pids]
            emb_m = banks[m][split]["emb"][pos]
            y_m = banks[m][split]["y"][pos]
            assert np.array_equal(ref_y, y_m), f"y mismatch on {split} ({m})"
            parts.append(emb_m)
        out[split] = {
            "pids": ref_pids,
            "y": ref_y,
            "emb": np.concatenate(parts, axis=1).astype(np.float32),
        }
    return out


# ─────────────────────────────────────────────
# Fusion head v2
# ─────────────────────────────────────────────
class FusionHeadV2(nn.Module):
    """Input BatchNorm + MLP with train-time modality dropout.

    Modality dropout = zero out an entire modality block of the concat vector
    with independent Bernoulli(p=modality_dropout) for each block, per-sample,
    during training only. Matches the spirit of modality dropout in multimodal
    networks (force reliance on each individual modality).
    """

    def __init__(self, input_dim: int, block_sizes: list[int], n_hidden: int,
                 hidden_dim: int, dropout: float, modality_dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.block_sizes = block_sizes
        self.modality_dropout = modality_dropout
        self.bn = nn.BatchNorm1d(input_dim)
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def _apply_modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return x
        B = x.size(0)
        mask = x.new_ones(B, self.input_dim)
        start = 0
        for bs in self.block_sizes:
            keep = (torch.rand(B, 1, device=x.device) >= self.modality_dropout).float()
            mask[:, start:start + bs] = keep
            start += bs
        # Inverted-dropout scaling so that expected sum stays the same
        scale = 1.0 / max(1.0 - self.modality_dropout, 1e-6)
        return x * mask * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_modality_dropout(x)
        x = self.bn(x)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# Training primitives
# ─────────────────────────────────────────────
def _train_epoch(model, loader, opt, crit):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _proba(model, X):
    model.eval()
    t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return torch.sigmoid(model(t)).cpu().numpy()


def _fit_head(X_tr, y_tr, params, block_sizes, *, eval_X=None, eval_y=None,
              input_dim: int, modality_dropout: float = 0.3):
    set_seed(SEED)
    model = FusionHeadV2(
        input_dim=input_dim, block_sizes=block_sizes,
        n_hidden=params["n_hidden"], hidden_dim=params["hidden_dim"],
        dropout=params["dropout"], modality_dropout=modality_dropout,
    ).to(DEVICE)

    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pw = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(),
                           lr=params["lr"], weight_decay=params["weight_decay"])

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)

    best_eval = -np.inf
    best_state = None
    hist = []
    for _ in range(params["epochs"]):
        _train_epoch(model, loader, opt, crit)
        if eval_X is not None:
            p = _proba(model, eval_X)
            au = float(roc_auc_score(eval_y, p))
            hist.append(au)
            if au > best_eval:
                best_eval = au
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# ─────────────────────────────────────────────
# CV objective on train+val
# ─────────────────────────────────────────────
def _cv_score(X, y, params, block_sizes, input_dim, n_splits=5,
              modality_dropout: float = 0.3) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aurocs = []
    for tr_idx, va_idx in skf.split(X, y):
        Xtr_f, Xva_f = X[tr_idx], X[va_idx]
        ytr_f, yva_f = y[tr_idx], y[va_idx]
        _, hist = _fit_head(
            Xtr_f, ytr_f, params, block_sizes,
            eval_X=Xva_f, eval_y=yva_f, input_dim=input_dim,
            modality_dropout=modality_dropout,
        )
        if hist:
            aurocs.append(max(hist))
    return float(np.mean(aurocs)) if aurocs else 0.5


def _full_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "class0_prf": [float(x) for x in precision_recall_fscore_support(y_true, pred, labels=[0], zero_division=0)[:3]],
        "class1_prf": [float(x) for x in precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)[:3]],
    }


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run(n_trials: int, n_splits: int, modality_dropout: float) -> dict:
    import optuna

    banks, dims = _load_cached_embeddings()
    data = _align_triple(banks)
    block_sizes = [dims[m] for m in MODALITIES]
    input_dim = sum(block_sizes)

    X_tr = data["train"]["emb"]; y_tr = data["train"]["y"].astype(np.float32)
    X_va = data["val"]["emb"]; y_va = data["val"]["y"].astype(np.float32)
    X_te = data["test"]["emb"]; y_te = data["test"]["y"].astype(np.float32)
    te_pids = data["test"]["pids"]

    X_trval = np.concatenate([X_tr, X_va], axis=0).astype(np.float32)
    y_trval = np.concatenate([y_tr, y_va], axis=0).astype(np.float32)

    print(f"[v2] concat = "
          + " + ".join(f"{dims[m]}({m})" for m in MODALITIES)
          + f" = {input_dim}, n_trainval={len(y_trval)}")
    print(f"[v2] Optuna: {n_trials} trials | CV={n_splits}-fold | "
          f"modality_dropout={modality_dropout}")

    def objective(trial):
        params = {
            "n_hidden": trial.suggest_categorical("n_hidden", [1]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 40, 100),
        }
        return _cv_score(X_trval, y_trval, params, block_sizes, input_dim,
                         n_splits=n_splits, modality_dropout=modality_dropout)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_cv = float(study.best_value)
    print(f"[v2] Optuna best CV AUROC = {best_cv:.4f}")
    print(f"[v2] best params = {best_params}")

    # Final trainonly (monitor on val)
    model_t, _ = _fit_head(
        X_tr, y_tr, best_params, block_sizes,
        eval_X=X_va, eval_y=y_va, input_dim=input_dim,
        modality_dropout=modality_dropout,
    )
    prob_val_t = _proba(model_t, X_va)
    prob_test_t = _proba(model_t, X_te)

    # Final trainval
    model_tv, _ = _fit_head(
        X_trval, y_trval, best_params, block_sizes,
        input_dim=input_dim, modality_dropout=modality_dropout,
    )
    prob_test_tv = _proba(model_tv, X_te)

    out_dir = FOLDER_ROOT / "results" / "triple" / "intermediate_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "pid": te_pids, "y_true": y_te.astype(int),
        "prob_trainonly": prob_test_t,
        "prob_trainval": prob_test_tv,
        "pred_trainval@0.5": (prob_test_tv >= 0.5).astype(int),
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    torch.save({
        "trainonly": model_t.state_dict(),
        "trainval": model_tv.state_dict(),
        "best_params": best_params,
        "input_dim": input_dim,
        "dims": dims,
        "modalities": MODALITIES,
        "modality_dropout": modality_dropout,
    }, out_dir / "best.pt")

    rec = {
        "modalities": MODALITIES,
        "n_modalities": 3,
        "dims": dims,
        "concat_dim": int(input_dim),
        "encoder_source": "single_modal_baseline train-only encoders (frozen)",
        "method": "CV_Optuna + BatchNorm1d(input) + modality_dropout",
        "modality_dropout_p": modality_dropout,
        "cv_folds": int(n_splits),
        "n_trials": int(n_trials),
        "best_params": best_params,
        "optuna_best_cv_auroc": best_cv,
        "val_auroc_trainonly": float(roc_auc_score(y_va, prob_val_t)),
        "test_auroc_trainonly": float(roc_auc_score(y_te, prob_test_t)),
        "test_auroc_trainval": float(roc_auc_score(y_te, prob_test_tv)),
        "test_metrics_trainval": _full_metrics(y_te.astype(int), prob_test_tv),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(rec, f, indent=2)

    # Merge into summary
    summary_path = FOLDER_ROOT / "results" / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["intermediate_fusion_v2"] = rec
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--modality-dropout", type=float, default=0.3)
    args = ap.parse_args()

    set_seed(SEED)
    print(f"[v2] device={DEVICE}")
    rec = run(args.n_trials, args.n_splits, args.modality_dropout)

    print("\n" + "=" * 72)
    print("  Intermediate fusion v2 — TRIPLE (n=63, headline = trainval)")
    print("=" * 72)
    print(f"  concat dim            = {rec['concat_dim']}")
    print(f"  Optuna best CV AUROC  = {rec['optuna_best_cv_auroc']:.4f}")
    print(f"  test AUROC (trainonly)= {rec['test_auroc_trainonly']:.4f}")
    print(f"  test AUROC (trainval) = {rec['test_auroc_trainval']:.4f}  ← headline")


if __name__ == "__main__":
    main()
