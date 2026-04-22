"""Late fusion (prob-level) for all 3 modality pairs — equal weights, 2 averaging rules.

Inputs  : single_modal_baseline/results/*/test_predictions*.csv  (train+val retrain probs)
Outputs : double_model/results/{clin_rad,clin_ct,rad_ct}/predictions_{arithmetic,logit}.csv
          double_model/results/summary.json

Design notes (so triple fusion can reuse the same schema later):
  - Source probabilities are the "train+val retrained" test predictions so the headline
    metric is directly comparable to single_modal_baseline (CT 0.6337 reference).
  - Equal weights (w=0.5 for each modality) — no val set tuning, honest baseline.
  - Two averaging rules recorded side by side:
        arithmetic : p_fused = (p_A + p_B) / 2
        logit      : p_fused = sigmoid( (logit(p_A) + logit(p_B)) / 2 )
  - Test threshold = 0.5 (same as single_modal_baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FOLDER_ROOT.parent
SM_RESULTS = REPO_ROOT / "single_modal_baseline" / "results"

MODAL_PROB_CSV = {
    "clinical":  (SM_RESULTS / "clinical"  / "test_predictions.csv",                "patient_id", "y_true", "prob_trainval"),
    "radiomics": (SM_RESULTS / "radiomics" / "test_predictions.csv",                "patient_id", "y_true", "prob_trainval"),
    "ct":        (SM_RESULTS / "ct" / "optuna_mlp" / "test_predictions_optuna_mlp.csv", "PatientID",  "label",  "prob_class1"),
}

PAIRS = {
    "clin_rad": ("clinical",  "radiomics"),
    "clin_ct":  ("clinical",  "ct"),
    "rad_ct":   ("radiomics", "ct"),
}


def _load_probs(modal: str) -> pd.DataFrame:
    path, pid_col, y_col, p_col = MODAL_PROB_CSV[modal]
    df = pd.read_csv(path)[[pid_col, y_col, p_col]].rename(
        columns={pid_col: "pid", y_col: "y_true", p_col: f"prob_{modal}"}
    )
    return df


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fuse_pair(a: str, b: str) -> dict:
    da = _load_probs(a)
    db = _load_probs(b)
    m = da.merge(db[["pid", f"prob_{b}"]], on="pid", how="inner")
    assert len(m) == len(da) == len(db), f"PID mismatch for ({a}, {b})"

    pa = m[f"prob_{a}"].values
    pb = m[f"prob_{b}"].values
    y = m["y_true"].values.astype(int)

    p_arith = (pa + pb) / 2.0
    p_logit = _sigmoid((_logit(pa) + _logit(pb)) / 2.0)

    out_dir = FOLDER_ROOT / "results" / f"{a[:4]}_{b[:4]}".replace("clin", "clin").replace("radi", "rad")
    # explicit naming for clarity
    pair_key = {("clinical","radiomics"):"clin_rad",("clinical","ct"):"clin_ct",("radiomics","ct"):"rad_ct"}[(a,b)]
    out_dir = FOLDER_ROOT / "results" / pair_key
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "pid": m["pid"], "y_true": y,
        f"prob_{a}": pa, f"prob_{b}": pb,
        "prob_fused_arithmetic": p_arith,
        "pred_arithmetic@0.5": (p_arith >= 0.5).astype(int),
    }).to_csv(out_dir / "predictions_arithmetic.csv", index=False)

    pd.DataFrame({
        "pid": m["pid"], "y_true": y,
        f"prob_{a}": pa, f"prob_{b}": pb,
        "prob_fused_logit": p_logit,
        "pred_logit@0.5": (p_logit >= 0.5).astype(int),
    }).to_csv(out_dir / "predictions_logit.csv", index=False)

    return {
        "modalities": [a, b],
        "n_modalities": 2,
        "weights": {a: 0.5, b: 0.5},
        "source_probs": "train+val retrain",
        "arithmetic_mean": {"test_auroc": float(roc_auc_score(y, p_arith))},
        "logit_mean":      {"test_auroc": float(roc_auc_score(y, p_logit))},
    }


def _load_single_modal_reference() -> dict:
    """Pull each modality's trainval test AUROC from single_modal_baseline/results/summary.json."""
    sm = json.load(open(SM_RESULTS / "summary.json"))
    return {m: sm["modalities"][m]["test_auroc_trainval"] for m in ("clinical", "radiomics", "ct")}


def main() -> None:
    summary = {
        "split": {"train": 293, "val": 64, "test": 63, "source": "shared 70/15/15 label-stratified split"},
        "method": "late_fusion_equal_weight",
        "source_probs": "single_modal_baseline train+val retrain (headline)",
        "averaging_rules": ["arithmetic", "logit"],
        "threshold": 0.5,
        "pairs": {},
        "single_modal_reference_test_auroc_trainval": _load_single_modal_reference(),
    }

    for key, (a, b) in PAIRS.items():
        summary["pairs"][key] = fuse_pair(a, b)

    out_path = FOLDER_ROOT / "results" / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("  Late fusion (equal weights) on shared test set (n=63)")
    print("=" * 72)
    hdr = f"{'Pair':<10} {'modalities':<24} | {'AUROC arith':>12} {'AUROC logit':>12}"
    print(hdr)
    print("-" * len(hdr))
    for key, rec in summary["pairs"].items():
        print(
            f"{key:<10} {'+'.join(rec['modalities']):<24} | "
            f"{rec['arithmetic_mean']['test_auroc']:>12.4f} "
            f"{rec['logit_mean']['test_auroc']:>12.4f}"
        )
    print("-" * len(hdr))
    print("Single-modal reference (train+val retrain test AUROC):")
    for k, v in summary["single_modal_reference_test_auroc_trainval"].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
