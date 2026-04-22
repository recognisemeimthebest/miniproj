"""Radiomics single-modal evaluation on the LJW seed99 partition.

Uses the 15 LASSO-selected features from origin/feature/LJY
(features/final_features.csv) and a small MLP classifier.

Pipeline matches eval_clinical.py:
1. Subset 15 features to the 293/64/63 seed99 partition.
2. Optuna (N_TRIALS, default 100) on (train, val) — maximise val AUROC.
3. Retrain with the best params:
     - trainonly : train on 293, report test AUROC
     - trainval  : train on 293+64, report test AUROC (headline)
4. Save everything under experiments/single_modal/radiomics/.

The architecture follows LJY's RadiomicsMLP: Linear -> ReLU -> Dropout ->
Linear -> ReLU -> Linear(1). We train a single logit (BCEWithLogitsLoss)
so the same pos_weight-based class balancing used in eval_clinical applies here.
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.single_modal.common_split import load_split_pids  # noqa: E402

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
RADIOMICS_CSV = REPO_ROOT / "data" / "radiomics" / "final_features.csv"
CLINICAL_CSV = REPO_ROOT / "preprocessing" / "output_clinical" / "lung1_clinical_encoded.csv"
OUT_DIR = REPO_ROOT / "experiments" / "single_modal" / "radiomics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label_2yr"
SEED = 99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_radiomics_frame() -> tuple[pd.DataFrame, list[str]]:
    rad = pd.read_csv(RADIOMICS_CSV)
    rad = rad.rename(columns={"patient_id": "PatientID"}).set_index("PatientID")
    feat_cols = list(rad.columns)
    cln = pd.read_csv(CLINICAL_CSV, index_col="PatientID")[[TARGET]].dropna()
    merged = rad.join(cln, how="inner")
    merged[feat_cols] = merged[feat_cols].fillna(0).astype(np.float32)
    return merged, feat_cols


def build_split_arrays(df: pd.DataFrame, feat_cols: list[str]):
    pids = load_split_pids()
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, split_pids in pids.items():
        sub = df.loc[split_pids]
        X = sub[feat_cols].values.astype(np.float32)
        y = sub[TARGET].values.astype(np.float32)
        arrays[split] = (X, y, split_pids)
    return arrays


class RadiomicsDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ─────────────────────────────────────────────
# Model — LJY RadiomicsMLP adapted for logit output
# ─────────────────────────────────────────────
class RadiomicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        emb = self.encoder(x)
        if return_embedding:
            return emb
        return self.classifier(emb).squeeze(-1)


# ─────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────
def _train_loop(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = model(xb)
    return torch.sigmoid(logits).cpu().numpy()


def _auroc(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, _predict_proba(model, X)))


def _fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    params: dict,
    *,
    eval_X: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
    input_dim: int,
) -> tuple[nn.Module, dict]:
    set_seed(SEED)
    model = RadiomicsMLP(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        embed_dim=params["embed_dim"],
        dropout=params["dropout"],
    ).to(DEVICE)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pw = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loader = DataLoader(
        RadiomicsDS(X_tr, y_tr), batch_size=params["batch_size"], shuffle=True
    )

    history = {"train_loss": [], "eval_auroc": []}
    best_eval = -np.inf
    best_state = None
    for epoch in range(params["epochs"]):
        loss = _train_loop(model, loader, optimizer, criterion)
        history["train_loss"].append(loss)
        if eval_X is not None and eval_y is not None:
            ea = _auroc(model, eval_X, eval_y)
            history["eval_auroc"].append(ea)
            if ea > best_eval:
                best_eval = ea
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────
# Optuna
# ─────────────────────────────────────────────
def run_optuna(arrays, feat_cols, n_trials: int):
    import optuna

    X_tr_raw, y_tr, _ = arrays["train"]
    X_va_raw, y_va, _ = arrays["val"]
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_va = scaler.transform(X_va_raw).astype(np.float32)
    input_dim = X_tr.shape[1]

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "embed_dim": trial.suggest_categorical("embed_dim", [16, 32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 30, 150),
        }
        _, hist = _fit(
            X_tr, y_tr, params, eval_X=X_va, eval_y=y_va, input_dim=input_dim
        )
        return float(max(hist["eval_auroc"])) if hist["eval_auroc"] else 0.5

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ─────────────────────────────────────────────
# Final retrain + eval
# ─────────────────────────────────────────────
def _full_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    auroc = float(roc_auc_score(y_true, prob))
    acc = float(accuracy_score(y_true, pred))
    cm = confusion_matrix(y_true, pred).tolist()
    p0, r0, f0, _ = precision_recall_fscore_support(y_true, pred, labels=[0], zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
    return {
        "auroc": auroc,
        "accuracy": acc,
        "confusion_matrix": cm,
        "class0_prf": [float(p0[0]), float(r0[0]), float(f0[0])],
        "class1_prf": [float(p1[0]), float(r1[0]), float(f1[0])],
    }


def final_retrain_and_eval(arrays, feat_cols, best_params):
    X_tr_raw, y_tr, _ = arrays["train"]
    X_va_raw, y_va, _ = arrays["val"]
    X_te_raw, y_te, te_pids = arrays["test"]

    scaler_t = StandardScaler().fit(X_tr_raw)
    X_tr = scaler_t.transform(X_tr_raw).astype(np.float32)
    X_va = scaler_t.transform(X_va_raw).astype(np.float32)
    X_te = scaler_t.transform(X_te_raw).astype(np.float32)
    model_t, hist_t = _fit(
        X_tr, y_tr, best_params, eval_X=X_va, eval_y=y_va, input_dim=X_tr.shape[1]
    )
    prob_val_t = _predict_proba(model_t, X_va)
    prob_test_t = _predict_proba(model_t, X_te)
    met_val_t = _full_metrics(y_va, prob_val_t)
    met_test_t = _full_metrics(y_te, prob_test_t)

    X_trval_raw = np.concatenate([X_tr_raw, X_va_raw], axis=0)
    y_trval = np.concatenate([y_tr, y_va], axis=0)
    scaler_tv = StandardScaler().fit(X_trval_raw)
    X_trval = scaler_tv.transform(X_trval_raw).astype(np.float32)
    X_te2 = scaler_tv.transform(X_te_raw).astype(np.float32)
    model_tv, _ = _fit(
        X_trval, y_trval, best_params, input_dim=X_trval.shape[1]
    )
    prob_test_tv = _predict_proba(model_tv, X_te2)
    met_test_tv = _full_metrics(y_te, prob_test_tv)

    torch.save(
        {
            "trainonly": model_t.state_dict(),
            "trainval": model_tv.state_dict(),
            "scaler_trainonly_mean": scaler_t.mean_.tolist(),
            "scaler_trainonly_scale": scaler_t.scale_.tolist(),
            "scaler_trainval_mean": scaler_tv.mean_.tolist(),
            "scaler_trainval_scale": scaler_tv.scale_.tolist(),
            "feature_cols": feat_cols,
            "input_dim": X_tr.shape[1],
            "best_params": best_params,
        },
        OUT_DIR / "best.pt",
    )

    pred_df = pd.DataFrame(
        {
            "patient_id": te_pids,
            "y_true": y_te.astype(int),
            "prob_trainonly": prob_test_t,
            "prob_trainval": prob_test_tv,
        }
    )
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    return {
        "val_trainonly": met_val_t,
        "test_trainonly": met_test_t,
        "test_trainval": met_test_tv,
        "epochs_used_trainonly_best": int(np.argmax(hist_t["eval_auroc"]) + 1),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(n_trials: int) -> None:
    set_seed(SEED)
    print(f"[radiomics] device={DEVICE} csv={RADIOMICS_CSV}")
    df, feat_cols = load_radiomics_frame()
    print(f"[radiomics] n_all={len(df)} feat_cols={len(feat_cols)}")

    arrays = build_split_arrays(df, feat_cols)
    for k, (X, y, _) in arrays.items():
        print(f"  {k}: X={X.shape}  pos={int(y.sum())}  neg={int((y==0).sum())}")

    print(f"\n[radiomics] Optuna: {n_trials} trials")
    study = run_optuna(arrays, feat_cols, n_trials=n_trials)
    best_params = study.best_params
    best_val_auroc = float(study.best_value)
    print(f"[radiomics] best val AUROC (Optuna): {best_val_auroc:.4f}")
    print(f"[radiomics] best params: {best_params}")

    print("\n[radiomics] Retrain + test evaluation...")
    final = final_retrain_and_eval(arrays, feat_cols, best_params)

    results = {
        "seed": SEED,
        "n_trials": n_trials,
        "split": {"train": 293, "val": 64, "test": 63},
        "feature_cols": feat_cols,
        "input_dim": len(feat_cols),
        "best_params": best_params,
        "optuna_best_val_auroc": best_val_auroc,
        **final,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n─── RADIOMICS RESULTS ───")
    print(f"  best params        : {best_params}")
    print(f"  Optuna val AUROC   : {best_val_auroc:.4f}")
    print(f"  val AUROC (train)  : {final['val_trainonly']['auroc']:.4f}")
    print(f"  test AUROC (train) : {final['test_trainonly']['auroc']:.4f}")
    print(f"  test AUROC (tr+val): {final['test_trainval']['auroc']:.4f}  ← headline")
    print(f"  saved: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    args = ap.parse_args()
    main(n_trials=args.n_trials)
