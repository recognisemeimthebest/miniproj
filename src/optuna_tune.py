"""Optuna 하이퍼파라미터 튜닝 — CT 256-dim feature 기반 classical classifier.

Objective: val AUROC 최대화. 4가지 classifier family 탐색.
    - LogReg (L1/L2, C, class_weight)
    - XGBoost (n_estimators, max_depth, lr, subsample, ...)
    - RandomForest (n_estimators, max_depth, min_samples, ...)
    - SVM (kernel, C, gamma)

Final: best trial로 test AUROC 평가 + summary JSON 저장.

사용:
    python src/optuna_tune.py \
        --features experiments/m1_hosny_iter7b_seed99/features \
        --n-trials 200 \
        --out experiments/m1_hosny_iter7b_seed99/optuna
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_features(fdir: Path):
    out = {}
    for split in ("train", "val", "test"):
        d = np.load(fdir / f"{split}.npz", allow_pickle=True)
        out[split] = {"pids": d["pids"], "X": d["X"], "y": d["y"]}
        print(f"  {split}: n={len(d['y'])} X={d['X'].shape} "
              f"class1={int((d['y']==1).sum())}", flush=True)
    return out


def _make_classifier(trial: optuna.trial.Trial):
    """Trial이 고르는 classifier + hyperparameters."""
    clf_name = trial.suggest_categorical(
        "clf", ["logreg", "xgboost", "rf", "svm"]
    )
    if clf_name == "logreg":
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        C = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        cw = trial.suggest_categorical("lr_class_weight", [None, "balanced"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        return clf_name, LogisticRegression(
            penalty=penalty, C=C, class_weight=cw, solver=solver, max_iter=2000,
            random_state=42,
        )
    if clf_name == "xgboost":
        params = dict(
            n_estimators=trial.suggest_int("xgb_n_estimators", 50, 600),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            learning_rate=trial.suggest_float("xgb_lr", 1e-3, 3e-1, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.3, 1.0),
            min_child_weight=trial.suggest_int("xgb_min_child", 1, 10),
            reg_alpha=trial.suggest_float("xgb_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("xgb_lambda", 1e-4, 10, log=True),
            random_state=42, eval_metric="auc", tree_method="hist",
            n_jobs=1,
        )
        return clf_name, XGBClassifier(**params)
    if clf_name == "rf":
        params = dict(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 500),
            max_depth=trial.suggest_int("rf_max_depth", 2, 20),
            min_samples_split=trial.suggest_int("rf_min_split", 2, 20),
            min_samples_leaf=trial.suggest_int("rf_min_leaf", 1, 10),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
            class_weight=trial.suggest_categorical("rf_cw", [None, "balanced"]),
            random_state=42, n_jobs=1,
        )
        return clf_name, RandomForestClassifier(**params)
    if clf_name == "svm":
        kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        C = trial.suggest_float("svm_C", 1e-3, 1e2, log=True)
        gamma = trial.suggest_categorical("svm_gamma", ["scale", "auto"])
        cw = trial.suggest_categorical("svm_cw", [None, "balanced"])
        return clf_name, SVC(
            kernel=kernel, C=C, gamma=gamma, probability=True,
            class_weight=cw, random_state=42,
        )
    raise ValueError(clf_name)


def _objective_factory(data, use_scaler: bool):
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xv,  yv  = data["val"]["X"],   data["val"]["y"]

    def objective(trial: optuna.trial.Trial) -> float:
        _, clf = _make_classifier(trial)
        if use_scaler:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xv_s  = sc.transform(Xv)
        else:
            Xtr_s, Xv_s = Xtr, Xv
        clf.fit(Xtr_s, ytr)
        p = clf.predict_proba(Xv_s)[:, 1]
        return roc_auc_score(yv, p)
    return objective


def _eval_on_test(trial_params: dict, data, use_scaler: bool):
    """Best trial 파라미터로 train+val 합쳐 재학습 후 test 평가."""
    class _FakeTrial:
        def __init__(self, p): self.p = p
        def suggest_categorical(self, k, _): return self.p[k]
        def suggest_float(self, k, *_, **__): return self.p[k]
        def suggest_int(self, k, *_, **__): return self.p[k]
    clf_name, clf = _make_classifier(_FakeTrial(trial_params))

    # train-only fit (val 포함 재학습 버전도 기록)
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xv,  yv  = data["val"]["X"],   data["val"]["y"]
    Xt,  yt  = data["test"]["X"],  data["test"]["y"]

    if use_scaler:
        sc1 = StandardScaler().fit(Xtr)
        clf.fit(sc1.transform(Xtr), ytr)
        val_p = clf.predict_proba(sc1.transform(Xv))[:, 1]
        test_p_trainonly = clf.predict_proba(sc1.transform(Xt))[:, 1]
    else:
        clf.fit(Xtr, ytr)
        val_p = clf.predict_proba(Xv)[:, 1]
        test_p_trainonly = clf.predict_proba(Xt)[:, 1]

    val_auroc = roc_auc_score(yv, val_p)
    test_auroc_trainonly = roc_auc_score(yt, test_p_trainonly)

    # train+val 합쳐 재학습
    _, clf2 = _make_classifier(_FakeTrial(trial_params))
    X_tv = np.vstack([Xtr, Xv]); y_tv = np.concatenate([ytr, yv])
    if use_scaler:
        sc2 = StandardScaler().fit(X_tv)
        clf2.fit(sc2.transform(X_tv), y_tv)
        test_p_tv = clf2.predict_proba(sc2.transform(Xt))[:, 1]
    else:
        clf2.fit(X_tv, y_tv)
        test_p_tv = clf2.predict_proba(Xt)[:, 1]
    test_auroc_tv = roc_auc_score(yt, test_p_tv)

    return clf_name, val_auroc, test_auroc_trainonly, test_auroc_tv, test_p_trainonly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaler", action="store_true", default=True,
                    help="StandardScaler 적용 (CNN GAP feature는 이미 ReLU 출력이라 양수 skew).")
    ap.add_argument("--no-scaler", dest="scaler", action="store_false")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Optuna 병렬 trial 수. >1이면 joblib threading으로 여러 trial 동시 실행.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[env] features={args.features} scaler={args.scaler} n_jobs={args.n_jobs}", flush=True)
    data = load_features(args.features)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _objective_factory(data, use_scaler=args.scaler),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=False,
    )

    print(f"\n[study] best val AUROC = {study.best_value:.4f}")
    print(f"[study] best params = {study.best_params}")

    clf_name, val_auroc, test_auroc_trainonly, test_auroc_tv, test_p = _eval_on_test(
        study.best_params, data, use_scaler=args.scaler
    )

    print(f"\n[final] clf = {clf_name}")
    print(f"[final] val  AUROC = {val_auroc:.4f}")
    print(f"[final] test AUROC (train only)    = {test_auroc_trainonly:.4f}")
    print(f"[final] test AUROC (train+val fit) = {test_auroc_tv:.4f}")

    # per-classifier best val 분포
    per_clf_best = {}
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        name = t.params.get("clf", "?")
        per_clf_best[name] = max(per_clf_best.get(name, -1), t.value)
    print(f"\n[per-clf best val AUROC]")
    for k, v in sorted(per_clf_best.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s}: {v:.4f}")

    summary = {
        "features_dir": str(args.features),
        "n_trials": args.n_trials,
        "scaler": args.scaler,
        "best_clf": clf_name,
        "best_params": study.best_params,
        "best_val_auroc": float(study.best_value),
        "final_val_auroc": float(val_auroc),
        "final_test_auroc_trainonly": float(test_auroc_trainonly),
        "final_test_auroc_trainval_fit": float(test_auroc_tv),
        "per_clf_best_val_auroc": per_clf_best,
    }
    out_json = args.out / "optuna_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}", flush=True)

    # test prediction CSV
    import pandas as pd
    pd.DataFrame({
        "PatientID": data["test"]["pids"],
        "label": data["test"]["y"].astype(int),
        "prob_class1": test_p,
        "pred@0.5": (test_p >= 0.5).astype(int),
    }).to_csv(args.out / "test_predictions_optuna.csv", index=False)
    print(f"[save] {args.out / 'test_predictions_optuna.csv'}", flush=True)


if __name__ == "__main__":
    main()
