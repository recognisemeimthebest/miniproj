"""Quick seed survey: how do clinical / radiomics / CT perform on each of
LJW's Option-C seeds {7, 99, 123}?

For speed we reuse seed99's Optuna-best hyperparameters for clinical and
radiomics across all seeds (no re-tuning per seed). CT numbers are read
directly from each seed's existing 3D CNN+TTA `test_predictions.csv`.

Output: one table per seed and a combined summary showing which seed yields
the most balanced "clinical / radiomics / CT" AUROC triple.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
SM_SCRIPTS = REPO / "single_modal_baseline" / "scripts"
if str(SM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SM_SCRIPTS))

from eval_clinical import (ClinicalBranch, load_clinical_frame, _fit as fit_clin,
                           _predict_proba as proba_clin)
from eval_radiomics import (RadiomicsMLP, load_radiomics_frame, _fit as fit_rad,
                            _predict_proba as proba_rad)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "label_2yr"

CLIN_RES = json.loads((REPO / "single_modal_baseline" / "results" / "clinical" / "results.json").read_text())
RAD_RES  = json.loads((REPO / "single_modal_baseline" / "results" / "radiomics" / "results.json").read_text())
CLIN_BP = CLIN_RES["best_params"]
RAD_BP  = RAD_RES["best_params"]


def load_seed_pids(seed: int) -> dict[str, list[str]]:
    feat_dir = REPO / "experiments" / f"m1_hosny_iter7b_seed{seed}" / "features"
    out = {}
    for split in ("train", "val", "test"):
        d = np.load(feat_dir / f"{split}.npz", allow_pickle=True)
        out[split] = [str(p) for p in d["pids"]]
    return out


def eval_modal(df: pd.DataFrame, feat_cols: list[str], pids: dict, *, kind: str) -> dict:
    """Retrain MLP with seed99 best_params on the given seed's split, report test AUROCs."""
    arr = {}
    for k in ("train", "val", "test"):
        sub = df.loc[pids[k]]
        arr[k] = (
            sub[feat_cols].values.astype(np.float32),
            sub[TARGET].values.astype(np.float32),
        )

    (X_tr_raw, y_tr) = arr["train"]
    (X_va_raw, y_va) = arr["val"]
    (X_te_raw, y_te) = arr["test"]

    bp = CLIN_BP if kind == "clinical" else RAD_BP
    fit = fit_clin if kind == "clinical" else fit_rad
    proba = proba_clin if kind == "clinical" else proba_rad

    # trainonly
    sc = StandardScaler().fit(X_tr_raw)
    X_tr = sc.transform(X_tr_raw).astype(np.float32)
    X_va = sc.transform(X_va_raw).astype(np.float32)
    X_te = sc.transform(X_te_raw).astype(np.float32)
    model_t, _ = fit(X_tr, y_tr, bp, eval_X=X_va, eval_y=y_va, input_dim=X_tr.shape[1])
    p_val_t = proba(model_t, X_va)
    p_te_t = proba(model_t, X_te)

    # trainval
    X_trval_raw = np.concatenate([X_tr_raw, X_va_raw], axis=0)
    y_trval = np.concatenate([y_tr, y_va], axis=0)
    sc2 = StandardScaler().fit(X_trval_raw)
    X_trval = sc2.transform(X_trval_raw).astype(np.float32)
    X_te2 = sc2.transform(X_te_raw).astype(np.float32)
    model_tv, _ = fit(X_trval, y_trval, bp, input_dim=X_trval.shape[1])
    p_te_tv = proba(model_tv, X_te2)

    return {
        "val_trainonly": float(roc_auc_score(y_va, p_val_t)),
        "test_trainonly": float(roc_auc_score(y_te, p_te_t)),
        "test_trainval": float(roc_auc_score(y_te, p_te_tv)),
        "n_pos_test": int(y_te.sum()), "n_test": int(len(y_te)),
    }


def ct_auroc(seed: int) -> float:
    p = REPO / "experiments" / f"m1_hosny_iter7b_seed{seed}" / "test_predictions.csv"
    df = pd.read_csv(p)
    # columns: PatientID, label, prob_class1, pred@0.5
    return float(roc_auc_score(df["label"], df["prob_class1"])), len(df)


def main() -> None:
    seeds = [7, 99, 123]
    clin_df, clin_cols = load_clinical_frame()
    rad_df, rad_cols = load_radiomics_frame()

    table = []
    for s in seeds:
        print(f"\n=== seed {s} ===")
        pids = load_seed_pids(s)
        print(f"  split sizes: train={len(pids['train'])} val={len(pids['val'])} test={len(pids['test'])}")

        cln = eval_modal(clin_df, clin_cols, pids, kind="clinical")
        rad = eval_modal(rad_df, rad_cols, pids, kind="radiomics")
        ct_auc, ct_n = ct_auroc(s)
        print(f"  clinical   val_tr-only={cln['val_trainonly']:.4f} | test_tr-only={cln['test_trainonly']:.4f} | test_trval={cln['test_trainval']:.4f}")
        print(f"  radiomics  val_tr-only={rad['val_trainonly']:.4f} | test_tr-only={rad['test_trainonly']:.4f} | test_trval={rad['test_trainval']:.4f}")
        print(f"  ct (3DCNN+TTA)          | test={ct_auc:.4f}  (n={ct_n}, pos={cln['n_pos_test']}/{cln['n_test']})")

        table.append({
            "seed": s,
            "clin_val":  cln['val_trainonly'],
            "clin_test_tr_only": cln['test_trainonly'],
            "clin_test_tr+val": cln['test_trainval'],
            "rad_val":   rad['val_trainonly'],
            "rad_test_tr_only": rad['test_trainonly'],
            "rad_test_tr+val": rad['test_trainval'],
            "ct_test_3dcnn_tta": ct_auc,
            "n_pos_test": cln['n_pos_test'],
        })

    print("\n" + "=" * 96)
    print("  Summary — test AUROC across 3 seeds")
    print("  (clinical/radiomics reuse seed99 best_params, CT = 3D CNN + TTA)")
    print("=" * 96)
    hdr = (f"{'seed':>4} | {'clin test (tr+val)':>18} {'rad test (tr+val)':>18} {'CT test (3DCNN+TTA)':>21} | "
           f"{'max-min range':>14}")
    print(hdr); print("-" * len(hdr))
    for r in table:
        vals = [r['clin_test_tr+val'], r['rad_test_tr+val'], r['ct_test_3dcnn_tta']]
        print(f"{r['seed']:>4} | {r['clin_test_tr+val']:>18.4f} {r['rad_test_tr+val']:>18.4f} {r['ct_test_3dcnn_tta']:>21.4f} | "
              f"{max(vals)-min(vals):>14.4f}")

    out = REPO / "seed_survey.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
