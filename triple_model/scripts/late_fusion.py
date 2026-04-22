"""Late fusion (prob-level) for the triple (clinical + radiomics + ct) — equal weights, 2 rules.

Inputs  : single_modal_baseline/results/*/test_predictions*.csv (train+val retrain probs)
Outputs : triple_model/results/triple/predictions_{arithmetic,logit}.csv
          triple_model/results/summary.json

Design notes (consistent with double_model/scripts/late_fusion.py):
  - Source probabilities = "train+val retrained" test predictions (matches single-modal headline).
  - Equal weights (w=1/3 for each modality) — no val tuning, honest baseline.
  - Two averaging rules recorded side by side:
        arithmetic : p_fused = (p_clin + p_rad + p_ct) / 3
        logit      : p_fused = sigmoid( (logit(p_clin) + logit(p_rad) + logit(p_ct)) / 3 )
  - Test threshold = 0.5.
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


def fuse_triple() -> dict:
    dc = _load_probs("clinical")
    dr = _load_probs("radiomics")
    dt = _load_probs("ct")
    m = dc.merge(dr[["pid", "prob_radiomics"]], on="pid", how="inner") \
          .merge(dt[["pid", "prob_ct"]], on="pid", how="inner")
    assert len(m) == len(dc) == len(dr) == len(dt), "PID mismatch across modalities"

    pc = m["prob_clinical"].values
    pr = m["prob_radiomics"].values
    pt = m["prob_ct"].values
    y = m["y_true"].values.astype(int)

    p_arith = (pc + pr + pt) / 3.0
    p_logit = _sigmoid((_logit(pc) + _logit(pr) + _logit(pt)) / 3.0)

    out_dir = FOLDER_ROOT / "results" / "triple"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "pid": m["pid"], "y_true": y,
        "prob_clinical": pc, "prob_radiomics": pr, "prob_ct": pt,
        "prob_fused_arithmetic": p_arith,
        "pred_arithmetic@0.5": (p_arith >= 0.5).astype(int),
    }).to_csv(out_dir / "predictions_arithmetic.csv", index=False)

    pd.DataFrame({
        "pid": m["pid"], "y_true": y,
        "prob_clinical": pc, "prob_radiomics": pr, "prob_ct": pt,
        "prob_fused_logit": p_logit,
        "pred_logit@0.5": (p_logit >= 0.5).astype(int),
    }).to_csv(out_dir / "predictions_logit.csv", index=False)

    return {
        "modalities": ["clinical", "radiomics", "ct"],
        "n_modalities": 3,
        "weights": {"clinical": 1 / 3, "radiomics": 1 / 3, "ct": 1 / 3},
        "source_probs": "train+val retrain",
        "arithmetic_mean": {"test_auroc": float(roc_auc_score(y, p_arith))},
        "logit_mean":      {"test_auroc": float(roc_auc_score(y, p_logit))},
    }


def _load_single_modal_reference() -> dict:
    sm = json.load(open(SM_RESULTS / "summary.json"))
    return {m: sm["modalities"][m]["test_auroc_trainval"] for m in ("clinical", "radiomics", "ct")}


def _load_double_reference() -> dict:
    dbl_path = REPO_ROOT / "double_model" / "results" / "summary.json"
    if not dbl_path.exists():
        return {}
    d = json.load(open(dbl_path))
    out = {}
    for pair, rec in d.get("pairs", {}).items():
        out[pair] = {
            "late_arith": rec["arithmetic_mean"]["test_auroc"],
            "late_logit": rec["logit_mean"]["test_auroc"],
        }
    for pair, rec in d.get("intermediate_fusion", {}).get("pairs", {}).items():
        out.setdefault(pair, {})["intermediate"] = rec["test_auroc_trainval"]
    return out


def main() -> None:
    summary = {
        "split": {"train": 293, "val": 64, "test": 63, "source": "shared 70/15/15 label-stratified split"},
        "method": "late_fusion_equal_weight",
        "source_probs": "single_modal_baseline train+val retrain (headline)",
        "averaging_rules": ["arithmetic", "logit"],
        "threshold": 0.5,
        "triple": fuse_triple(),
        "single_modal_reference_test_auroc_trainval": _load_single_modal_reference(),
        "double_reference": _load_double_reference(),
    }

    out_path = FOLDER_ROOT / "results" / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("  Late fusion (equal weights) on shared test set (n=63) — TRIPLE")
    print("=" * 72)
    print(f"  arithmetic mean AUROC = {summary['triple']['arithmetic_mean']['test_auroc']:.4f}")
    print(f"  logit      mean AUROC = {summary['triple']['logit_mean']['test_auroc']:.4f}")
    print("\nSingle-modal reference (train+val retrain test AUROC):")
    for k, v in summary["single_modal_reference_test_auroc_trainval"].items():
        print(f"  {k:<10} {v:.4f}")
    print("\nDouble-model reference (test AUROC):")
    for pair, rec in summary["double_reference"].items():
        parts = " | ".join(f"{k}={v:.4f}" for k, v in rec.items())
        print(f"  {pair:<10} {parts}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
