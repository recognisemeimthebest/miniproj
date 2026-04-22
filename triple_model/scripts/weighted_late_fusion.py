"""Weighted late fusion for the triple (clinical + radiomics + ct).

Fusion-side tuning only — no single-modal retraining.

Pipeline:
  1) Reconstruct trainonly single-modal encoders + heads (clinical MLP, radiomics MLP,
     ct MLP head) and forward the VAL set through them to obtain honest val probs
     (trainonly models never saw val → no leakage).
  2) Grid-search w = (w_clin, w_rad, w_ct) on the simplex (step=0.05, 231 combos)
     selecting the combination that maximises val AUROC. Two averaging rules are
     tuned independently:
       arithmetic : p = w_c*p_c + w_r*p_r + w_t*p_t
       logit      : p = sigmoid( w_c*logit(p_c) + w_r*logit(p_r) + w_t*logit(p_t) )
  3) Apply the chosen w to the TRAINVAL single-modal test probabilities (already
     saved in single_modal_baseline) to get the headline test AUROC — matches the
     single-modal / double_model protocol.

Outputs:
  triple_model/results/triple/weighted_late/
    ├─ grid_val_arithmetic.csv       (all 231 grid points for diagnostics)
    ├─ grid_val_logit.csv
    ├─ predictions_arithmetic.csv    (test probs with best w)
    ├─ predictions_logit.csv         (test probs with best w)
    └─ results.json                   (best w for each rule + val/test AUROC)
  triple_model/results/summary.json   (adds `weighted_late_fusion` section)
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FOLDER_ROOT.parent
SM_ROOT = REPO_ROOT / "single_modal_baseline"
SM_SCRIPTS = SM_ROOT / "scripts"
if str(SM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SM_SCRIPTS))

from common_split import load_split_pids  # noqa: E402
from eval_clinical import ClinicalBranch, load_clinical_frame  # noqa: E402
from eval_radiomics import RadiomicsMLP, load_radiomics_frame  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "label_2yr"

OUT_DIR = FOLDER_ROOT / "results" / "triple" / "weighted_late"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Val probs using TRAINONLY single-modal models
# ─────────────────────────────────────────────
def _scale(X, mean, scale):
    return ((X - np.asarray(mean)) / np.asarray(scale)).astype(np.float32)


@torch.no_grad()
def _forward_prob(model, X):
    model.eval()
    t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = model(t)
    return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def val_probs_clinical() -> tuple[np.ndarray, np.ndarray, list]:
    ckpt = torch.load(SM_ROOT / "results" / "clinical" / "best.pt",
                      map_location=DEVICE, weights_only=False)
    bp = ckpt["best_params"]
    model = ClinicalBranch(
        input_dim=ckpt["input_dim"], hidden_dim=bp["hidden_dim"],
        embed_dim=bp["embed_dim"], dropout=bp["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["trainonly"])
    df, feat_cols = load_clinical_frame()
    pids = load_split_pids()["val"]
    sub = df.loc[pids]
    X = _scale(sub[feat_cols].values.astype(np.float32),
               ckpt["scaler_trainonly_mean"], ckpt["scaler_trainonly_scale"])
    y = sub[TARGET].values.astype(int)
    return _forward_prob(model, X), y, list(pids)


def val_probs_radiomics() -> tuple[np.ndarray, np.ndarray, list]:
    ckpt = torch.load(SM_ROOT / "results" / "radiomics" / "best.pt",
                      map_location=DEVICE, weights_only=False)
    bp = ckpt["best_params"]
    model = RadiomicsMLP(
        input_dim=ckpt["input_dim"], hidden_dim=bp["hidden_dim"],
        embed_dim=bp["embed_dim"], dropout=bp["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["trainonly"])
    df, feat_cols = load_radiomics_frame()
    pids = load_split_pids()["val"]
    sub = df.loc[pids]
    X = _scale(sub[feat_cols].values.astype(np.float32),
               ckpt["scaler_trainonly_mean"], ckpt["scaler_trainonly_scale"])
    y = sub[TARGET].values.astype(int)
    return _forward_prob(model, X), y, list(pids)


def val_probs_ct() -> tuple[np.ndarray, np.ndarray, list]:
    """CT MLP head — retrain trainonly head from saved best_params, predict val.

    single_modal_baseline/results/ct/optuna_mlp/best_optuna_mlp.pt stores only the
    TRAINVAL combined model. For honest val probs (Optuna used val for selection),
    we retrain a TRAINONLY head here from the saved best_params on features/train.
    This adds ~seconds and matches the single_modal Optuna protocol exactly.
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    ct_dir = SM_ROOT / "results" / "ct" / "optuna_mlp"
    summary = json.load(open(ct_dir / "optuna_mlp_summary.json"))
    bp = summary["best_params"]
    seed = summary.get("seed", 123)

    class MLPHead(nn.Module):
        def __init__(self, in_dim, n_hidden, hidden_dim, dropout, use_bn, num_classes=2):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(d, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d = hidden_dim
            layers.append(nn.Linear(d, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Load frozen 256-dim features
    feat_dir = SM_ROOT / "results" / "ct" / "features"
    tr = np.load(feat_dir / "train.npz", allow_pickle=True)
    va = np.load(feat_dir / "val.npz", allow_pickle=True)
    Xtr = tr["X"].astype(np.float32); ytr = tr["y"].astype(int)
    Xv = va["X"].astype(np.float32); yv = va["y"].astype(int)
    pid_v = [str(p) for p in va["pids"]]

    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xv_s = sc.transform(Xv).astype(np.float32)

    # Retrain trainonly head
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = MLPHead(
        in_dim=Xtr_s.shape[1], n_hidden=bp["n_hidden"],
        hidden_dim=bp["hidden_dim"], dropout=bp["dropout"],
        use_bn=bp["use_bn"],
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    if bp["class_weight"] == "balanced":
        n_pos = float((ytr == 1).sum()); n_neg = float((ytr == 0).sum())
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=DEVICE)
        w = w * 2 / w.sum()
    else:
        w = None
    crit = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=bp["label_smoothing"])

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xv_t = torch.tensor(Xv_s, dtype=torch.float32, device=DEVICE)

    bs = bp["batch_size"]; n = Xtr_t.shape[0]
    best_val = -1.0; best_state = None; stale = 0; patience = 15
    rng = np.random.default_rng(seed)
    for _ in range(bp["epochs"]):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            logits = model(Xtr_t[j])
            loss = crit(logits, ytr_t[j])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
        au = float(roc_auc_score(yv, p))
        if au > best_val:
            best_val = au
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy().astype(np.float64)
    print(f"    [ct] retrain trainonly head → val AUROC = {best_val:.4f} "
          f"(baseline reported = {summary.get('final_val_auroc_trainonly', float('nan')):.4f})")
    return prob_val, yv, pid_v


# ─────────────────────────────────────────────
# Test probs = existing single_modal_baseline trainval probs
# ─────────────────────────────────────────────
def _load_test_probs():
    dc = pd.read_csv(SM_ROOT / "results" / "clinical" / "test_predictions.csv")[
        ["patient_id", "y_true", "prob_trainval"]
    ].rename(columns={"patient_id": "pid", "prob_trainval": "p_c"})
    dr = pd.read_csv(SM_ROOT / "results" / "radiomics" / "test_predictions.csv")[
        ["patient_id", "y_true", "prob_trainval"]
    ].rename(columns={"patient_id": "pid", "prob_trainval": "p_r"})
    dt = pd.read_csv(SM_ROOT / "results" / "ct" / "optuna_mlp" / "test_predictions_optuna_mlp.csv")[
        ["PatientID", "label", "prob_class1"]
    ].rename(columns={"PatientID": "pid", "label": "y_true", "prob_class1": "p_t"})
    m = dc.merge(dr[["pid", "p_r"]], on="pid").merge(dt[["pid", "p_t"]], on="pid")
    return m


# ─────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────
def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _grid(step: float = 0.05):
    """Return all (w_c, w_r, w_t) on the simplex with given step size."""
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


def _align_val(pc, yc, pidc, pr, yr, pidr, pt, yt, pidt):
    """Align val predictions from 3 modalities by PID (clinical order as reference)."""
    idx_r = {p: i for i, p in enumerate(pidr)}
    idx_t = {p: i for i, p in enumerate(pidt)}
    order = [(idx_r[p], idx_t[p]) for p in pidc]
    pr_a = np.array([pr[i] for i, _ in order])
    yr_a = np.array([yr[i] for i, _ in order])
    pt_a = np.array([pt[i] for _, i in order])
    yt_a = np.array([yt[i] for _, i in order])
    assert np.array_equal(yc, yr_a) and np.array_equal(yc, yt_a), "val y mismatch across modalities"
    return pc, pr_a, pt_a, yc


def grid_search(pc, pr, pt, y, step=0.05):
    rows_arith = []
    rows_logit = []
    lc, lr, lt = _logit(pc), _logit(pr), _logit(pt)
    for w in _grid(step):
        wc, wr, wt = w
        p_a = wc * pc + wr * pr + wt * pt
        p_l = _sigmoid(wc * lc + wr * lr + wt * lt)
        rows_arith.append({"w_c": wc, "w_r": wr, "w_t": wt,
                           "val_auroc": float(roc_auc_score(y, p_a))})
        rows_logit.append({"w_c": wc, "w_r": wr, "w_t": wt,
                           "val_auroc": float(roc_auc_score(y, p_l))})
    ga = pd.DataFrame(rows_arith).sort_values("val_auroc", ascending=False).reset_index(drop=True)
    gl = pd.DataFrame(rows_logit).sort_values("val_auroc", ascending=False).reset_index(drop=True)
    return ga, gl


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    print(f"[weighted-late] device={DEVICE}")

    # 1) honest val probs
    print("[val] clinical …")
    pc_v, yc_v, pid_c = val_probs_clinical()
    print("[val] radiomics …")
    pr_v, yr_v, pid_r = val_probs_radiomics()
    print("[val] ct …")
    pt_v, yt_v, pid_t = val_probs_ct()
    pc_v, pr_v, pt_v, y_v = _align_val(pc_v, yc_v, pid_c, pr_v, yr_v, pid_r, pt_v, yt_v, pid_t)
    print(f"  val n={len(y_v)}  class1={int(y_v.sum())}")

    # Single-modal val AUROC (sanity)
    single_val = {
        "clinical":  float(roc_auc_score(y_v, pc_v)),
        "radiomics": float(roc_auc_score(y_v, pr_v)),
        "ct":        float(roc_auc_score(y_v, pt_v)),
    }
    print(f"  single-modal val AUROC: "
          + " | ".join(f"{k}={v:.4f}" for k, v in single_val.items()))

    # 2) grid search
    grid_a, grid_l = grid_search(pc_v, pr_v, pt_v, y_v, step=0.05)
    grid_a.to_csv(OUT_DIR / "grid_val_arithmetic.csv", index=False)
    grid_l.to_csv(OUT_DIR / "grid_val_logit.csv", index=False)

    best_a = grid_a.iloc[0]
    best_l = grid_l.iloc[0]
    print(f"\n[best arith ] w=({best_a.w_c:.2f},{best_a.w_r:.2f},{best_a.w_t:.2f}) val AUROC={best_a.val_auroc:.4f}")
    print(f"[best logit ] w=({best_l.w_c:.2f},{best_l.w_r:.2f},{best_l.w_t:.2f}) val AUROC={best_l.val_auroc:.4f}")

    # 3) apply best w to TRAINVAL test probs
    test = _load_test_probs()
    pc_t = test["p_c"].values; pr_t = test["p_r"].values; pt_t = test["p_t"].values
    y_t = test["y_true"].values.astype(int)
    pid_test = test["pid"].tolist()

    def apply(w_c, w_r, w_t, rule):
        if rule == "arithmetic":
            return w_c * pc_t + w_r * pr_t + w_t * pt_t
        lc, lr, lt = _logit(pc_t), _logit(pr_t), _logit(pt_t)
        return _sigmoid(w_c * lc + w_r * lr + w_t * lt)

    p_test_a = apply(best_a.w_c, best_a.w_r, best_a.w_t, "arithmetic")
    p_test_l = apply(best_l.w_c, best_l.w_r, best_l.w_t, "logit")

    pd.DataFrame({
        "pid": pid_test, "y_true": y_t,
        "prob_clinical": pc_t, "prob_radiomics": pr_t, "prob_ct": pt_t,
        "prob_fused_arithmetic": p_test_a,
        "pred_arithmetic@0.5": (p_test_a >= 0.5).astype(int),
    }).to_csv(OUT_DIR / "predictions_arithmetic.csv", index=False)

    pd.DataFrame({
        "pid": pid_test, "y_true": y_t,
        "prob_clinical": pc_t, "prob_radiomics": pr_t, "prob_ct": pt_t,
        "prob_fused_logit": p_test_l,
        "pred_logit@0.5": (p_test_l >= 0.5).astype(int),
    }).to_csv(OUT_DIR / "predictions_logit.csv", index=False)

    rec = {
        "modalities": ["clinical", "radiomics", "ct"],
        "method": "weighted_late_fusion_simplex_grid",
        "grid_step": 0.05, "grid_size": len(grid_a),
        "val_n": int(len(y_v)),
        "single_modal_val_auroc": single_val,
        "arithmetic": {
            "w": {"clinical": float(best_a.w_c), "radiomics": float(best_a.w_r), "ct": float(best_a.w_t)},
            "val_auroc": float(best_a.val_auroc),
            "test_auroc_trainval": float(roc_auc_score(y_t, p_test_a)),
        },
        "logit": {
            "w": {"clinical": float(best_l.w_c), "radiomics": float(best_l.w_r), "ct": float(best_l.w_t)},
            "val_auroc": float(best_l.val_auroc),
            "test_auroc_trainval": float(roc_auc_score(y_t, p_test_l)),
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(rec, f, indent=2)

    # 4) merge into summary.json
    summary_path = FOLDER_ROOT / "results" / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["weighted_late_fusion"] = rec
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("  Weighted late fusion — TRIPLE (n=63, w picked on val n=64)")
    print("=" * 72)
    print(f"  arithmetic  test AUROC = {rec['arithmetic']['test_auroc_trainval']:.4f}  "
          f"(w={rec['arithmetic']['w']})")
    print(f"  logit       test AUROC = {rec['logit']['test_auroc_trainval']:.4f}  "
          f"(w={rec['logit']['w']})")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
