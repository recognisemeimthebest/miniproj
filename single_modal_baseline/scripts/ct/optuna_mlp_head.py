"""Optuna-tuned PyTorch MLP head on frozen Hosny CNN features.

Hosny 2018 구조 유지:
    Hosny CNN backbone (frozen, from best.pt)
        → 256-dim GAP feature (pre-computed, features/*.npz)
        → MLP head (Optuna-searched)
        → 2-class logits

최종 산출물: combined CNN+MLP `.pt` 파일 (submission-ready)

Search space (MLP head):
    - n_hidden: {0, 1, 2}  (0 = linear only)
    - hidden_dim: {32, 64, 128, 256}
    - dropout: [0.0, 0.6]
    - use_bn: {True, False}
    - lr: [1e-4, 1e-2] log
    - weight_decay: [1e-6, 1e-2] log
    - epochs: [30, 200]
    - batch_size: {16, 32, 64}
    - label_smoothing: [0.0, 0.2]
    - class_weight: {None, "balanced"}

사용 (repo root에서, 로컬 single_modal_baseline artifacts 에 대해):
    PYTHONPATH=. python single_modal_baseline/scripts/ct/optuna_mlp_head.py \
        --ckpt single_modal_baseline/results/ct/best.pt \
        --features single_modal_baseline/results/ct/features \
        --n-trials 200 --n-jobs 4 \
        --out single_modal_baseline/results/ct/optuna_mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_hosny_cnn


# ---------------------------------------------------------------------------
# MLP head module
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """Configurable MLP head. Linear(0 hidden) / 1-hidden / 2-hidden."""

    def __init__(self, in_dim: int, n_hidden: int, hidden_dim: int,
                 dropout: float, use_bn: bool, num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Combined model (frozen CNN + trainable MLP head) for final .pt
# ---------------------------------------------------------------------------
class HosnyWithMLPHead(nn.Module):
    """CNN features (frozen) + MLP head → logits. For submission."""

    def __init__(self, cnn: nn.Module, head: MLPHead,
                 scaler_mean: np.ndarray | None, scaler_scale: np.ndarray | None):
        super().__init__()
        # Reuse Hosny CNN blocks (features, gap, flatten), drop its own head.
        self.features = cnn.features
        self.gap = cnn.gap
        self.flatten = cnn.flatten
        self.head = head
        # StandardScaler params as buffers (applied after flatten if provided)
        if scaler_mean is not None:
            self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer("scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32))
            self.use_scaler = True
        else:
            self.use_scaler = False

    def forward(self, x):
        f = self.features(x)
        f = self.gap(f)
        f = self.flatten(f)  # (B, 256)
        if self.use_scaler:
            f = (f - self.scaler_mean) / self.scaler_scale
        return self.head(f)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------
def load_features(fdir: Path):
    out = {}
    for split in ("train", "val", "test"):
        d = np.load(fdir / f"{split}.npz", allow_pickle=True)
        out[split] = {"pids": d["pids"], "X": d["X"], "y": d["y"]}
    return out


# ---------------------------------------------------------------------------
# Train one MLP given hyperparams — returns val AUROC
# ---------------------------------------------------------------------------
def train_eval_mlp(Xtr, ytr, Xv, yv, hp, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLPHead(
        in_dim=Xtr.shape[1],
        n_hidden=hp["n_hidden"],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
        use_bn=hp["use_bn"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    if hp["class_weight"] == "balanced":
        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32, device=device)
        w = w * 2 / w.sum()  # normalize
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=hp["label_smoothing"])

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xv_t = torch.tensor(Xv, dtype=torch.float32, device=device)

    n = Xtr_t.shape[0]
    bs = hp["batch_size"]
    best_val = -1.0
    best_state = None
    patience = 15
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(hp["epochs"]):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            logits = model(Xtr_t[j])
            loss = crit(logits, ytr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
        auroc = roc_auc_score(yv, p)
        if auroc > best_val:
            best_val = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Restore best
    model.load_state_dict(best_state)
    return best_val, model


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _sample_hp(trial: optuna.trial.Trial):
    return {
        "n_hidden": trial.suggest_categorical("n_hidden", [0, 1, 2]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "use_bn": trial.suggest_categorical("use_bn", [False, True]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 30, 200),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _objective(data, device, use_scaler, base_seed):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xv = data["val"]["X"];   yv = data["val"]["y"]
    if use_scaler:
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr); Xv_s = sc.transform(Xv)
    else:
        Xtr_s, Xv_s = Xtr, Xv

    def obj(trial: optuna.trial.Trial) -> float:
        hp = _sample_hp(trial)
        val_auroc, _ = train_eval_mlp(Xtr_s, ytr, Xv_s, yv, hp, device, base_seed)
        return val_auroc
    return obj


# ---------------------------------------------------------------------------
# Final retrain (train+val) + save combined .pt
# ---------------------------------------------------------------------------
def final_retrain_and_save(best_params, data, device, args, cnn_state_dict):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xv = data["val"]["X"];   yv = data["val"]["y"]
    Xt = data["test"]["X"];  yt = data["test"]["y"]

    if args.scaler:
        sc_trainonly = StandardScaler().fit(Xtr)
        sc_tv = StandardScaler().fit(np.vstack([Xtr, Xv]))
        Xtr_s = sc_trainonly.transform(Xtr)
        Xv_s = sc_trainonly.transform(Xv)
        Xt_s_to = sc_trainonly.transform(Xt)
        Xtv_s = sc_tv.transform(np.vstack([Xtr, Xv]))
        Xt_s_tv = sc_tv.transform(Xt)
    else:
        sc_trainonly = sc_tv = None
        Xtr_s = Xtr; Xv_s = Xv; Xt_s_to = Xt
        Xtv_s = np.vstack([Xtr, Xv]); Xt_s_tv = Xt

    hp = {
        "n_hidden": best_params["n_hidden"],
        "hidden_dim": best_params["hidden_dim"],
        "dropout": best_params["dropout"],
        "use_bn": best_params["use_bn"],
        "lr": best_params["lr"],
        "weight_decay": best_params["weight_decay"],
        "epochs": best_params["epochs"],
        "batch_size": best_params["batch_size"],
        "label_smoothing": best_params["label_smoothing"],
        "class_weight": best_params["class_weight"],
    }

    # 1) train-only fit (for reporting val AUROC and test AUROC)
    val_auroc, mlp_trainonly = train_eval_mlp(Xtr_s, ytr, Xv_s, yv, hp, device, args.seed)
    mlp_trainonly.eval()
    with torch.no_grad():
        xt = torch.tensor(Xt_s_to, dtype=torch.float32, device=device)
        p_to = torch.softmax(mlp_trainonly(xt), dim=1)[:, 1].cpu().numpy()
    test_auroc_trainonly = roc_auc_score(yt, p_to)

    # 2) train+val fit (final, for .pt submission)
    ytv = np.concatenate([ytr, yv])
    # no val → need a pseudo-val; just train full epochs without early stop
    hp_tv = dict(hp); hp_tv["epochs"] = hp["epochs"]  # same budget
    # train+val: use a fraction of train+val as holdout for best-state pick
    # but simpler: just train full epochs without early stop
    torch.manual_seed(args.seed)
    model_tv = MLPHead(
        in_dim=Xtv_s.shape[1], n_hidden=hp_tv["n_hidden"],
        hidden_dim=hp_tv["hidden_dim"], dropout=hp_tv["dropout"],
        use_bn=hp_tv["use_bn"],
    ).to(device)
    opt = torch.optim.AdamW(model_tv.parameters(), lr=hp_tv["lr"],
                            weight_decay=hp_tv["weight_decay"])
    if hp_tv["class_weight"] == "balanced":
        n_pos = float((ytv == 1).sum()); n_neg = float((ytv == 0).sum())
        w = torch.tensor([1.0/n_neg, 1.0/n_pos], dtype=torch.float32, device=device)
        w = w * 2 / w.sum()
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=hp_tv["label_smoothing"])

    Xtv_t = torch.tensor(Xtv_s, dtype=torch.float32, device=device)
    ytv_t = torch.tensor(ytv, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    n_tv = Xtv_t.shape[0]
    for epoch in range(hp_tv["epochs"]):
        model_tv.train()
        idx = rng.permutation(n_tv)
        for i in range(0, n_tv, hp_tv["batch_size"]):
            j = idx[i:i + hp_tv["batch_size"]]
            logits = model_tv(Xtv_t[j]); loss = crit(logits, ytv_t[j])
            opt.zero_grad(); loss.backward(); opt.step()

    model_tv.eval()
    with torch.no_grad():
        xt = torch.tensor(Xt_s_tv, dtype=torch.float32, device=device)
        p_tv = torch.softmax(model_tv(xt), dim=1)[:, 1].cpu().numpy()
    test_auroc_tv = roc_auc_score(yt, p_tv)

    # Combined model (train+val version) — for submission
    cnn = build_hosny_cnn(num_classes=2, dropout_fc=args.dropout_fc).to(device)
    cnn.load_state_dict(cnn_state_dict)
    for p in cnn.parameters():
        p.requires_grad_(False)
    cnn.eval()
    combined = HosnyWithMLPHead(
        cnn, model_tv,
        scaler_mean=sc_tv.mean_ if args.scaler else None,
        scaler_scale=sc_tv.scale_ if args.scaler else None,
    ).to(device)
    combined.eval()

    return {
        "val_auroc_trainonly_fit": float(val_auroc),
        "test_auroc_trainonly_fit": float(test_auroc_trainonly),
        "test_auroc_trainval_fit": float(test_auroc_tv),
        "test_prob_trainval_fit": p_tv,
        "combined_model": combined,
        "sc_tv_mean": sc_tv.mean_.tolist() if args.scaler else None,
        "sc_tv_scale": sc_tv.scale_.tolist() if args.scaler else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Hosny CNN best.pt")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--scaler", action="store_true", default=True)
    ap.add_argument("--no-scaler", dest="scaler", action="store_false")
    ap.add_argument("--dropout-fc", type=float, default=0.4,
                    help="Hosny CNN build용 (head shape 맞추기, frozen이라 실제 영향 없음)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} ckpt={args.ckpt} seed={args.seed} n_jobs={args.n_jobs}", flush=True)

    data = load_features(args.features)
    print(f"[data] train={len(data['train']['y'])} val={len(data['val']['y'])} "
          f"test={len(data['test']['y'])} dim={data['train']['X'].shape[1]}", flush=True)

    # Load CNN state for combined model later
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cnn_sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _objective(data, device, use_scaler=args.scaler, base_seed=args.seed),
        n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=False,
    )
    print(f"\n[study] best val AUROC = {study.best_value:.4f}")
    print(f"[study] best params = {study.best_params}", flush=True)

    # Final retrain + combined .pt
    final = final_retrain_and_save(study.best_params, data, device, args, cnn_sd)
    print(f"\n[final] val AUROC (train-only fit) = {final['val_auroc_trainonly_fit']:.4f}")
    print(f"[final] test AUROC (train-only fit) = {final['test_auroc_trainonly_fit']:.4f}")
    print(f"[final] test AUROC (train+val fit)  = {final['test_auroc_trainval_fit']:.4f}")

    # Save combined .pt (submission artifact)
    combined_pt = args.out / "best_optuna_mlp.pt"
    torch.save({
        "model_state_dict": final["combined_model"].state_dict(),
        "optuna_params": study.best_params,
        "val_auroc": final["val_auroc_trainonly_fit"],
        "test_auroc_trainonly": final["test_auroc_trainonly_fit"],
        "test_auroc_trainval": final["test_auroc_trainval_fit"],
        "args": vars(args) | {
            "ckpt": str(args.ckpt), "features": str(args.features), "out": str(args.out),
        },
        "scaler_mean": final["sc_tv_mean"],
        "scaler_scale": final["sc_tv_scale"],
    }, combined_pt)
    print(f"[save] combined model → {combined_pt}", flush=True)

    # Summary JSON
    summary = {
        "ckpt": str(args.ckpt),
        "features_dir": str(args.features),
        "seed": args.seed,
        "n_trials": args.n_trials,
        "scaler": args.scaler,
        "best_val_auroc": float(study.best_value),
        "best_params": study.best_params,
        "final_val_auroc_trainonly": final["val_auroc_trainonly_fit"],
        "final_test_auroc_trainonly": final["test_auroc_trainonly_fit"],
        "final_test_auroc_trainval": final["test_auroc_trainval_fit"],
    }
    (args.out / "optuna_mlp_summary.json").write_text(json.dumps(summary, indent=2))

    # Test predictions CSV
    import pandas as pd
    pd.DataFrame({
        "PatientID": data["test"]["pids"],
        "label": data["test"]["y"].astype(int),
        "prob_class1": final["test_prob_trainval_fit"],
        "pred@0.5": (final["test_prob_trainval_fit"] >= 0.5).astype(int),
    }).to_csv(args.out / "test_predictions_optuna_mlp.csv", index=False)
    print(f"[save] test predictions → {args.out/'test_predictions_optuna_mlp.csv'}", flush=True)


if __name__ == "__main__":
    main()
