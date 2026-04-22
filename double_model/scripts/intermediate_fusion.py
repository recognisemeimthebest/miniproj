"""Intermediate (embedding-level) fusion for all 3 modality pairs.

Protocol
--------
1. Reconstruct clinical / radiomics encoders from single_modal_baseline/results/*/best.pt
   using the TRAIN-ONLY state dicts (encoder has never seen val/test → Optuna is safe).
2. Forward train/val/test raw features through these frozen encoders to obtain
   embeddings (clinical 32-dim, radiomics 256-dim). CT embeddings (256-dim) are
   already available in single_modal_baseline/results/ct/features/*.npz.
3. For each pair (clin_rad, clin_ct, rad_ct):
     concat(emb_A, emb_B) → FusionHead (MLP) → logit
   - Optuna (N_TRIALS) over head hyperparameters, maximise val AUROC.
   - Final retrain with best params:
       * trainonly head  → val & test AUROC
       * trainval  head  → test AUROC (headline, matches single_modal_baseline protocol)
   - Save head weights, results.json, test predictions.
4. Merge into double_model/results/summary.json alongside late fusion numbers.

Encoders are ALWAYS frozen (train-only state). The fusion experiment adds only the
FusionHead's parameters, so improvements reflect head capacity + cross-modal
interaction, not encoder retraining.
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FOLDER_ROOT.parent
SM_ROOT = REPO_ROOT / "single_modal_baseline"
SM_SCRIPTS = SM_ROOT / "scripts"
if str(SM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SM_SCRIPTS))

from common_split import load_split_pids  # noqa: E402
from eval_clinical import ClinicalBranch, load_clinical_frame  # noqa: E402
from eval_radiomics import RadiomicsMLP, load_radiomics_frame  # noqa: E402

SEED = 99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "label_2yr"
LEAKAGE = ["Survival.time", "deadstatus.event"]

EMB_DIR = FOLDER_ROOT / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "clin_rad": ("clinical", "radiomics"),
    "clin_ct":  ("clinical", "ct"),
    "rad_ct":   ("radiomics", "ct"),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────
def _scale(X: np.ndarray, mean, scale) -> np.ndarray:
    return ((X - np.asarray(mean)) / np.asarray(scale)).astype(np.float32)


@torch.no_grad()
def _encode(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    emb = model(t, return_embedding=True)
    return emb.cpu().numpy().astype(np.float32)


def extract_clinical_embeddings() -> dict:
    ckpt = torch.load(SM_ROOT / "results" / "clinical" / "best.pt", map_location=DEVICE, weights_only=False)
    bp = ckpt["best_params"]
    model = ClinicalBranch(
        input_dim=ckpt["input_dim"],
        hidden_dim=bp["hidden_dim"],
        embed_dim=bp["embed_dim"],
        dropout=bp["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["trainonly"])

    df, feat_cols = load_clinical_frame()
    assert feat_cols == ckpt["feature_cols"], "clinical feature col mismatch"

    pids = load_split_pids()
    mean = ckpt["scaler_trainonly_mean"]
    scale = ckpt["scaler_trainonly_scale"]
    out = {}
    for split, split_pids in pids.items():
        sub = df.loc[split_pids]
        X = sub[feat_cols].values.astype(np.float32)
        y = sub[TARGET].values.astype(np.int64)
        Xs = _scale(X, mean, scale)
        emb = _encode(model, Xs)
        out[split] = {"emb": emb, "y": y, "pids": split_pids}
    return out


def extract_radiomics_embeddings() -> dict:
    ckpt = torch.load(SM_ROOT / "results" / "radiomics" / "best.pt", map_location=DEVICE, weights_only=False)
    bp = ckpt["best_params"]
    model = RadiomicsMLP(
        input_dim=ckpt["input_dim"],
        hidden_dim=bp["hidden_dim"],
        embed_dim=bp["embed_dim"],
        dropout=bp["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["trainonly"])

    df, feat_cols = load_radiomics_frame()
    assert feat_cols == ckpt["feature_cols"], "radiomics feature col mismatch"

    pids = load_split_pids()
    mean = ckpt["scaler_trainonly_mean"]
    scale = ckpt["scaler_trainonly_scale"]
    out = {}
    for split, split_pids in pids.items():
        sub = df.loc[split_pids]
        X = sub[feat_cols].values.astype(np.float32)
        y = sub[TARGET].values.astype(np.int64)
        Xs = _scale(X, mean, scale)
        emb = _encode(model, Xs)
        out[split] = {"emb": emb, "y": y, "pids": split_pids}
    return out


def extract_ct_embeddings() -> dict:
    feat_dir = SM_ROOT / "results" / "ct" / "features"
    out = {}
    for split in ("train", "val", "test"):
        d = np.load(feat_dir / f"{split}.npz", allow_pickle=True)
        out[split] = {
            "emb": d["X"].astype(np.float32) if "X" in d else d["features"].astype(np.float32),
            "y": d["y"].astype(np.int64),
            "pids": [str(p) for p in d["pids"]],
        }
    return out


def align_emb(a: dict, b: dict) -> dict:
    """Return same-split pair tensors aligned by PID ordering of `a`."""
    out = {}
    for split in ("train", "val", "test"):
        pa = a[split]["pids"]
        pb = b[split]["pids"]
        idx_map_b = {p: i for i, p in enumerate(pb)}
        idx = [idx_map_b[p] for p in pa]
        emb_a = a[split]["emb"]
        emb_b = b[split]["emb"][idx]
        y_a = a[split]["y"]
        y_b = b[split]["y"][idx]
        assert np.array_equal(y_a, y_b), f"y mismatch on {split}"
        out[split] = {
            "pids": pa,
            "y": y_a,
            "emb": np.concatenate([emb_a, emb_b], axis=1).astype(np.float32),
        }
    return out


# ─────────────────────────────────────────────
# Fusion head + training loop
# ─────────────────────────────────────────────
class FusionHead(nn.Module):
    def __init__(self, input_dim: int, n_hidden: int, hidden_dim: int, dropout: float):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def _fit_head(X_tr, y_tr, params, *, eval_X=None, eval_y=None, input_dim: int):
    set_seed(SEED)
    model = FusionHead(
        input_dim=input_dim,
        n_hidden=params["n_hidden"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(DEVICE)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pw = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)

    history = {"eval_auroc": []}
    best_eval = -np.inf
    best_state = None
    for _ in range(params["epochs"]):
        _train_epoch(model, loader, opt, crit)
        if eval_X is not None and eval_y is not None:
            prob = _proba(model, eval_X)
            ea = float(roc_auc_score(eval_y, prob))
            history["eval_auroc"].append(ea)
            if ea > best_eval:
                best_eval = ea
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────
# Per-pair Optuna + final retrain
# ─────────────────────────────────────────────
def _full_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "class0_prf": [float(x) for x in precision_recall_fscore_support(y_true, pred, labels=[0], zero_division=0)[:3]],
        "class1_prf": [float(x) for x in precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)[:3]],
    }


def run_pair(pair_key: str, a_name: str, b_name: str, pair_data: dict, n_trials: int) -> dict:
    import optuna

    X_tr = pair_data["train"]["emb"]
    y_tr = pair_data["train"]["y"].astype(np.float32)
    X_va = pair_data["val"]["emb"]
    y_va = pair_data["val"]["y"].astype(np.float32)
    X_te = pair_data["test"]["emb"]
    y_te = pair_data["test"]["y"].astype(np.float32)
    te_pids = pair_data["test"]["pids"]
    input_dim = X_tr.shape[1]

    dim_a = pair_data["_dims"][0]
    dim_b = pair_data["_dims"][1]
    print(f"\n[{pair_key}] input_dim = {dim_a}({a_name}) + {dim_b}({b_name}) = {input_dim}")
    print(f"[{pair_key}] Optuna: {n_trials} trials")

    def objective(trial):
        params = {
            "n_hidden": trial.suggest_categorical("n_hidden", [1, 2]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 30, 120),
        }
        _, hist = _fit_head(X_tr, y_tr, params, eval_X=X_va, eval_y=y_va, input_dim=input_dim)
        return float(max(hist["eval_auroc"])) if hist["eval_auroc"] else 0.5

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_val = float(study.best_value)
    print(f"[{pair_key}] Optuna best val AUROC = {best_val:.4f}")
    print(f"[{pair_key}] best params = {best_params}")

    # ── Final trainonly head
    model_t, hist_t = _fit_head(X_tr, y_tr, best_params, eval_X=X_va, eval_y=y_va, input_dim=input_dim)
    prob_val_t = _proba(model_t, X_va)
    prob_test_t = _proba(model_t, X_te)

    # ── Final trainval head (train + val concat)
    X_trval = np.concatenate([X_tr, X_va], axis=0)
    y_trval = np.concatenate([y_tr, y_va], axis=0)
    model_tv, _ = _fit_head(X_trval, y_trval, best_params, input_dim=input_dim)
    prob_test_tv = _proba(model_tv, X_te)

    out_dir = FOLDER_ROOT / "results" / pair_key / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "pid": te_pids,
        "y_true": y_te.astype(int),
        "prob_trainonly": prob_test_t,
        "prob_trainval": prob_test_tv,
        "pred_trainval@0.5": (prob_test_tv >= 0.5).astype(int),
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    torch.save({
        "trainonly": model_t.state_dict(),
        "trainval": model_tv.state_dict(),
        "best_params": best_params,
        "input_dim": input_dim,
        "dim_a": dim_a, "dim_b": dim_b,
        "modalities": [a_name, b_name],
    }, out_dir / "best.pt")

    rec = {
        "modalities": [a_name, b_name],
        "n_modalities": 2,
        "dim_a": int(dim_a), "dim_b": int(dim_b), "concat_dim": int(input_dim),
        "encoder_source": "single_modal_baseline train-only encoders (frozen)",
        "best_params": best_params,
        "optuna_best_val_auroc": best_val,
        "val_auroc_trainonly": float(roc_auc_score(y_va, prob_val_t)),
        "test_auroc_trainonly": float(roc_auc_score(y_te, prob_test_t)),
        "test_auroc_trainval": float(roc_auc_score(y_te, prob_test_tv)),
        "test_metrics_trainval": _full_metrics(y_te.astype(int), prob_test_tv),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(rec, f, indent=2)
    return rec


# ─────────────────────────────────────────────
# Summary merge
# ─────────────────────────────────────────────
def merge_summary(inter: dict) -> Path:
    summary_path = FOLDER_ROOT / "results" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["intermediate_fusion"] = {
        "method": "frozen_concat_mlp_head",
        "encoder_source": "single_modal_baseline train-only encoders (frozen)",
        "head_optuna_trials_per_pair": inter.pop("_n_trials"),
        "threshold": 0.5,
        "pairs": inter,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(n_trials: int) -> None:
    set_seed(SEED)
    print(f"[intermediate] device={DEVICE}")

    print("\n[embeddings] extracting clinical …")
    clin = extract_clinical_embeddings()
    print(f"  clinical emb dim = {clin['train']['emb'].shape[1]}")
    print("[embeddings] extracting radiomics …")
    rad = extract_radiomics_embeddings()
    print(f"  radiomics emb dim = {rad['train']['emb'].shape[1]}")
    print("[embeddings] loading CT features …")
    ct = extract_ct_embeddings()
    print(f"  ct emb dim = {ct['train']['emb'].shape[1]}")

    # Save cached embeddings (small — handy for reproducibility checks)
    for name, bank in [("clinical", clin), ("radiomics", rad), ("ct", ct)]:
        for split in ("train", "val", "test"):
            np.savez(EMB_DIR / f"{name}_{split}.npz",
                     emb=bank[split]["emb"], y=bank[split]["y"],
                     pids=np.array(bank[split]["pids"], dtype=object))

    modal_banks = {"clinical": clin, "radiomics": rad, "ct": ct}

    inter_results = {}
    for pair_key, (a, b) in PAIRS.items():
        pair_data = align_emb(modal_banks[a], modal_banks[b])
        pair_data["_dims"] = (modal_banks[a]["train"]["emb"].shape[1],
                               modal_banks[b]["train"]["emb"].shape[1])
        rec = run_pair(pair_key, a, b, pair_data, n_trials=n_trials)
        inter_results[pair_key] = rec

    inter_results["_n_trials"] = n_trials
    summary_path = merge_summary(inter_results)

    print("\n" + "=" * 72)
    print("  Intermediate fusion (frozen encoders + MLP head) on LJW seed99 test (n=63)")
    print("=" * 72)
    hdr = f"{'Pair':<10} {'concat':>7} | {'Optuna val':>11} {'test trainonly':>15} {'test trainval':>15}"
    print(hdr); print("-" * len(hdr))
    for k in PAIRS:
        r = inter_results[k]
        print(f"{k:<10} {r['concat_dim']:>7} | "
              f"{r['optuna_best_val_auroc']:>11.4f} "
              f"{r['test_auroc_trainonly']:>15.4f} "
              f"{r['test_auroc_trainval']:>15.4f}")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    main(n_trials=ap.parse_args().n_trials)
