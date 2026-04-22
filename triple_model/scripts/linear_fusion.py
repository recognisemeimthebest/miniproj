"""Linear fusion on 640-dim concat — LogisticRegressionCV + ElasticNet.

Motivation
----------
intermediate_fusion.py trainval collapsed to 0.545 because a 640-dim concat → MLP
head is high capacity for n=357 train+val.  A *linear* head (≈640 params) is a
natural low-capacity baseline that keeps the concat representation but drops the
non-linear head that was overfitting.  This is still fusion-side only — the
train-only encoder embeddings cached in triple_model/embeddings/ are reused
verbatim.

Two models (both scikit-learn, fit on scaled 640-dim concat):
  1. LogisticRegressionCV (L2) — CV picks C on train+val.
  2. ElasticNet-like LR via SGDClassifier(loss='log_loss', penalty='elasticnet')
     — grid over (alpha, l1_ratio) on val AUROC, then trainval retrain.

Protocol (matches single/double/triple headline):
  - trainonly fit (+ val AUROC)       = diagnostic
  - trainval  fit (+ test AUROC)      = headline

Outputs:
  triple_model/results/triple/linear/
    ├─ test_predictions.csv  (prob_trainonly, prob_trainval for each model)
    ├─ results.json
  triple_model/results/summary.json adds `linear_fusion` section.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FOLDER_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = FOLDER_ROOT / "embeddings"
OUT_DIR = FOLDER_ROOT / "results" / "triple" / "linear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 99
MODALITIES = ["clinical", "radiomics", "ct"]


def _load() -> dict:
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
            assert np.array_equal(ref_y, banks[m][split]["y"][pos]), f"y mismatch ({split},{m})"
            parts.append(banks[m][split]["emb"][pos])
        out[split] = {
            "pids": ref_pids,
            "y": ref_y.astype(int),
            "X": np.concatenate(parts, axis=1).astype(np.float32),
        }
    return out


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
# Model 1: LogisticRegressionCV (L2)
# ─────────────────────────────────────────────
def fit_logreg_l2(X_tr, y_tr, X_va, y_va, X_trval, y_trval, X_te, y_te):
    """LogisticRegressionCV picks C via 5-fold CV internally."""
    scaler_t = StandardScaler().fit(X_tr)
    Xtr_s = scaler_t.transform(X_tr); Xva_s = scaler_t.transform(X_va); Xte_s_t = scaler_t.transform(X_te)
    Cs = np.logspace(-4, 2, 13)
    clf_t = LogisticRegressionCV(
        Cs=Cs, cv=5, penalty="l2", solver="lbfgs",
        class_weight="balanced", scoring="roc_auc",
        max_iter=2000, random_state=SEED, n_jobs=-1,
    ).fit(Xtr_s, y_tr)
    prob_val_t = clf_t.predict_proba(Xva_s)[:, 1]
    prob_test_t = clf_t.predict_proba(Xte_s_t)[:, 1]

    scaler_tv = StandardScaler().fit(X_trval)
    Xtv_s = scaler_tv.transform(X_trval); Xte_s_tv = scaler_tv.transform(X_te)
    clf_tv = LogisticRegressionCV(
        Cs=Cs, cv=5, penalty="l2", solver="lbfgs",
        class_weight="balanced", scoring="roc_auc",
        max_iter=2000, random_state=SEED, n_jobs=-1,
    ).fit(Xtv_s, y_trval)
    prob_test_tv = clf_tv.predict_proba(Xte_s_tv)[:, 1]

    return {
        "C_trainonly": float(clf_t.C_[0]),
        "C_trainval": float(clf_tv.C_[0]),
        "val_auroc_trainonly": float(roc_auc_score(y_va, prob_val_t)),
        "test_auroc_trainonly": float(roc_auc_score(y_te, prob_test_t)),
        "test_auroc_trainval": float(roc_auc_score(y_te, prob_test_tv)),
        "test_metrics_trainval": _full_metrics(y_te, prob_test_tv),
        "prob_test_trainonly": prob_test_t,
        "prob_test_trainval": prob_test_tv,
    }


# ─────────────────────────────────────────────
# Model 2: ElasticNet logistic via SGDClassifier
# ─────────────────────────────────────────────
def fit_elasticnet(X_tr, y_tr, X_va, y_va, X_trval, y_trval, X_te, y_te):
    """SGD logistic with elastic-net penalty; grid over (alpha, l1_ratio)."""
    scaler_t = StandardScaler().fit(X_tr)
    Xtr_s = scaler_t.transform(X_tr); Xva_s = scaler_t.transform(X_va); Xte_s_t = scaler_t.transform(X_te)

    alphas = np.logspace(-5, -1, 9)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best = {"val_auroc": -np.inf, "alpha": None, "l1_ratio": None}
    for a in alphas:
        for r in l1_ratios:
            clf = SGDClassifier(
                loss="log_loss", penalty="elasticnet",
                alpha=a, l1_ratio=r, class_weight="balanced",
                max_iter=2000, tol=1e-4, random_state=SEED,
            ).fit(Xtr_s, y_tr)
            prob = _sgd_prob(clf, Xva_s)
            auc = float(roc_auc_score(y_va, prob))
            if auc > best["val_auroc"]:
                best = {"val_auroc": auc, "alpha": float(a), "l1_ratio": float(r)}

    # refit trainonly with best params → test
    clf_t = SGDClassifier(
        loss="log_loss", penalty="elasticnet",
        alpha=best["alpha"], l1_ratio=best["l1_ratio"],
        class_weight="balanced", max_iter=2000, tol=1e-4, random_state=SEED,
    ).fit(Xtr_s, y_tr)
    prob_test_t = _sgd_prob(clf_t, Xte_s_t)

    # trainval refit
    scaler_tv = StandardScaler().fit(X_trval)
    Xtv_s = scaler_tv.transform(X_trval); Xte_s_tv = scaler_tv.transform(X_te)
    clf_tv = SGDClassifier(
        loss="log_loss", penalty="elasticnet",
        alpha=best["alpha"], l1_ratio=best["l1_ratio"],
        class_weight="balanced", max_iter=2000, tol=1e-4, random_state=SEED,
    ).fit(Xtv_s, y_trval)
    prob_test_tv = _sgd_prob(clf_tv, Xte_s_tv)

    return {
        "best_alpha": best["alpha"],
        "best_l1_ratio": best["l1_ratio"],
        "val_auroc_trainonly": best["val_auroc"],
        "test_auroc_trainonly": float(roc_auc_score(y_te, prob_test_t)),
        "test_auroc_trainval": float(roc_auc_score(y_te, prob_test_tv)),
        "test_metrics_trainval": _full_metrics(y_te, prob_test_tv),
        "prob_test_trainonly": prob_test_t,
        "prob_test_trainval": prob_test_tv,
    }


def _sgd_prob(clf, X):
    """SGDClassifier with log_loss — use decision_function → sigmoid (predict_proba also available)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    from scipy.special import expit
    return expit(clf.decision_function(X))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    data = _load()
    X_tr, y_tr = data["train"]["X"], data["train"]["y"]
    X_va, y_va = data["val"]["X"], data["val"]["y"]
    X_te, y_te = data["test"]["X"], data["test"]["y"]
    te_pids = data["test"]["pids"]
    X_trval = np.concatenate([X_tr, X_va], axis=0)
    y_trval = np.concatenate([y_tr, y_va], axis=0)

    print(f"[linear] concat dim={X_tr.shape[1]} | "
          f"n_train={len(y_tr)}, n_val={len(y_va)}, n_trainval={len(y_trval)}, n_test={len(y_te)}")

    print("[linear] fitting LogisticRegressionCV (L2) …")
    logreg = fit_logreg_l2(X_tr, y_tr, X_va, y_va, X_trval, y_trval, X_te, y_te)
    print(f"  C(trainonly)={logreg['C_trainonly']:.4g}, C(trainval)={logreg['C_trainval']:.4g}")
    print(f"  val(trainonly)={logreg['val_auroc_trainonly']:.4f}, "
          f"test(trainonly)={logreg['test_auroc_trainonly']:.4f}, "
          f"test(trainval)={logreg['test_auroc_trainval']:.4f}")

    print("[linear] fitting ElasticNet logistic (SGD) …")
    enet = fit_elasticnet(X_tr, y_tr, X_va, y_va, X_trval, y_trval, X_te, y_te)
    print(f"  alpha={enet['best_alpha']:.4g}, l1_ratio={enet['best_l1_ratio']}")
    print(f"  val(trainonly)={enet['val_auroc_trainonly']:.4f}, "
          f"test(trainonly)={enet['test_auroc_trainonly']:.4f}, "
          f"test(trainval)={enet['test_auroc_trainval']:.4f}")

    # Save test predictions (both models)
    pd.DataFrame({
        "pid": te_pids,
        "y_true": y_te.astype(int),
        "prob_trainonly_logreg_l2": logreg["prob_test_trainonly"],
        "prob_trainval_logreg_l2": logreg["prob_test_trainval"],
        "prob_trainonly_elasticnet": enet["prob_test_trainonly"],
        "prob_trainval_elasticnet": enet["prob_test_trainval"],
        "pred_trainval_logreg_l2@0.5": (logreg["prob_test_trainval"] >= 0.5).astype(int),
        "pred_trainval_elasticnet@0.5": (enet["prob_test_trainval"] >= 0.5).astype(int),
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    rec = {
        "modalities": MODALITIES,
        "concat_dim": int(X_tr.shape[1]),
        "encoder_source": "single_modal_baseline train-only encoders (frozen)",
        "method": "Linear head on 640-dim concat (L2 + ElasticNet)",
        "threshold": 0.5,
        "logreg_l2": {k: v for k, v in logreg.items() if not k.startswith("prob_")},
        "elasticnet": {k: v for k, v in enet.items() if not k.startswith("prob_")},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(rec, f, indent=2)

    # Merge into summary
    summary_path = FOLDER_ROOT / "results" / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["linear_fusion"] = rec
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("  Linear fusion on 640-dim concat — TRIPLE (n=63, headline = trainval)")
    print("=" * 72)
    print(f"  LogisticRegressionCV(L2):  test AUROC (trainval) = {logreg['test_auroc_trainval']:.4f}")
    print(f"  ElasticNet (SGD):          test AUROC (trainval) = {enet['test_auroc_trainval']:.4f}")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
