"""Triple-fusion plots — single / double refs + all triple tuning variants.

Reads:
  - triple_model/results/summary.json                 (all numbers)
  - triple_model/results/triple/predictions_*.csv     (late arith / late logit)
  - triple_model/results/triple/intermediate/test_predictions.csv           (v1)
  - triple_model/results/triple/intermediate_v2/test_predictions.csv        (v2)
  - triple_model/results/triple/linear/test_predictions.csv                 (linear L2 + ENet)
  - single_modal_baseline/results/{clinical,radiomics,ct}/…                 (single refs)
  - double_model/results/*.csv / summary.json                               (double refs)

Outputs (triple_model/figures/):
  - comparison_bar.png       : single (3) | double best-per-pair (3) | triple baseline (3)
  - fusion_tuning_bar.png    : all 8 triple methods side-by-side (baseline 3 + tuned 5)
  - roc_curves.png           : single dashed + double dotted + top triple methods solid
  - confusion_matrices.png   : 1×3 CM (late logit, intermediate v2, linear L2) @thr=0.5
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

rcParams["axes.unicode_minus"] = False

FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FOLDER_ROOT.parent
RESULTS = FOLDER_ROOT / "results"
FIG_DIR = FOLDER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SM_RESULTS = REPO_ROOT / "single_modal_baseline" / "results"
DBL_ROOT = REPO_ROOT / "double_model"

MODAL_COLOR = {"clinical": "#3498db", "radiomics": "#e67e22", "ct": "#2ecc71"}
PAIR_COLOR = {"clin_rad": "#9b59b6", "clin_ct": "#16a085", "rad_ct": "#d35400"}
TRIPLE_BASE = "#c0392b"
TRIPLE_TUNED = "#8e44ad"
PAIR_LABEL = {"clin_rad": "Clin+Rad", "clin_ct": "Clin+CT", "rad_ct": "Rad+CT"}


def _load_summary() -> dict:
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Probability loaders
# ─────────────────────────────────────────────
def _triple_late_probs(rule: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(RESULTS / "triple" / f"predictions_{rule}.csv")
    return df["y_true"].values.astype(int), df[f"prob_fused_{rule}"].values.astype(float)


def _triple_intermediate_probs() -> tuple[np.ndarray, np.ndarray] | None:
    path = RESULTS / "triple" / "intermediate" / "test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def _triple_intermediate_v2_probs() -> tuple[np.ndarray, np.ndarray] | None:
    path = RESULTS / "triple" / "intermediate_v2" / "test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def _triple_linear_probs(model: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = RESULTS / "triple" / "linear" / "test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = f"prob_trainval_{model}"
    if col not in df.columns:
        return None
    return df["y_true"].values.astype(int), df[col].values.astype(float)


def _single_probs(modal: str) -> tuple[np.ndarray, np.ndarray]:
    if modal == "ct":
        df = pd.read_csv(SM_RESULTS / "ct" / "optuna_mlp" / "test_predictions_optuna_mlp.csv")
        return df["label"].values.astype(int), df["prob_class1"].values.astype(float)
    df = pd.read_csv(SM_RESULTS / modal / "test_predictions.csv")
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def _double_probs(pair: str, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    if kind == "intermediate":
        path = DBL_ROOT / "results" / pair / "intermediate" / "test_predictions.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)
    if kind == "late_arith":
        path = DBL_ROOT / "results" / pair / "predictions_arithmetic.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df["y_true"].values.astype(int), df["prob_fused_arithmetic"].values.astype(float)
    if kind == "late_logit":
        path = DBL_ROOT / "results" / pair / "predictions_logit.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df["y_true"].values.astype(int), df["prob_fused_logit"].values.astype(float)
    raise KeyError(kind)


def _double_best_kind(summary: dict, pair: str) -> tuple[str, float]:
    dbl = summary.get("double_reference", {}).get(pair, {})
    if not dbl:
        return ("late_arith", np.nan)
    best = max(dbl.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -np.inf))
    return best


# ─────────────────────────────────────────────
# Triple method registry (baseline + tuned)
# ─────────────────────────────────────────────
def _triple_methods(summary: dict) -> list[dict]:
    """Return ordered list of triple methods with (name, label, test_auroc, color, hatch)."""
    trp = summary["triple"]
    inter_v1 = summary.get("intermediate_fusion", {}).get("triple", {})
    wlf = summary.get("weighted_late_fusion", {})
    inter_v2 = summary.get("intermediate_fusion_v2", {})
    lin = summary.get("linear_fusion", {})

    methods = [
        ("late_arith",   "Late arith\n(equal 1/3)",   trp["arithmetic_mean"]["test_auroc"],    TRIPLE_BASE, ""),
        ("late_logit",   "Late logit\n(equal 1/3)",   trp["logit_mean"]["test_auroc"],         TRIPLE_BASE, "//"),
        ("inter_v1",     "Intermediate\n(v1 MLP)",   inter_v1.get("test_auroc_trainval"),     TRIPLE_BASE, "xx"),
        ("wlf_arith",    "Weighted late\narith",      (wlf.get("arithmetic") or {}).get("test_auroc_trainval"), TRIPLE_TUNED, ""),
        ("wlf_logit",    "Weighted late\nlogit",      (wlf.get("logit") or {}).get("test_auroc_trainval"),      TRIPLE_TUNED, "//"),
        ("inter_v2",     "Intermediate\nv2 (CV+BN+MD)", inter_v2.get("test_auroc_trainval"),    TRIPLE_TUNED, "xx"),
        ("linear_l2",    "Linear\nL2-LogReg",         (lin.get("logreg_l2") or {}).get("test_auroc_trainval"),  TRIPLE_TUNED, ".."),
        ("linear_enet",  "Linear\nElasticNet",        (lin.get("elasticnet") or {}).get("test_auroc_trainval"),  TRIPLE_TUNED, "\\\\"),
    ]
    return [dict(zip(("key", "label", "val", "color", "hatch"), m)) for m in methods]


# ─────────────────────────────────────────────
# 1. Comparison bar — single | double | triple (top-3)
# ─────────────────────────────────────────────
def plot_bar(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    sm = summary["single_modal_reference_test_auroc_trainval"]
    bars = []
    pos = 0
    width = 0.7

    for m in ["clinical", "radiomics", "ct"]:
        bars.append((pos, sm[m], MODAL_COLOR[m], "", f"{m}\n(single)"))
        pos += 1
    pos += 0.6

    for pair in ["clin_rad", "clin_ct", "rad_ct"]:
        best_kind, best_val = _double_best_kind(summary, pair)
        bars.append((pos, best_val, PAIR_COLOR[pair], "",
                     f"{PAIR_LABEL[pair]}\n({best_kind})"))
        pos += 1
    pos += 0.6

    trp_methods = _triple_methods(summary)
    scored = [(m["val"], m) for m in trp_methods if m["val"] is not None and not np.isnan(m["val"])]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    top3 = [m for _, m in scored[:3]]
    for m in top3:
        bars.append((pos, m["val"], m["color"], m["hatch"], f"Triple\n{m['label'].replace(chr(10),' ')}"))
        pos += 1

    for x, v, c, h, _ in bars:
        ax.bar(x, v, width, color=c, hatch=h, edgecolor="white", alpha=0.95)
        if not (v is None or (isinstance(v, float) and np.isnan(v))):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    ax.set_xticks([b[0] for b in bars])
    ax.set_xticklabels([b[4] for b in bars], fontsize=8.5)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.axhline(sm["ct"], color=MODAL_COLOR["ct"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"CT single ({sm['ct']:.3f})")
    ax.set_title("Single → Double → Triple (top-3 methods) on shared test split (n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "comparison_bar.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 1b. Fusion tuning bar — all 8 triple methods
# ─────────────────────────────────────────────
def plot_fusion_tuning_bar(summary: dict) -> Path:
    methods = _triple_methods(summary)
    sm = summary["single_modal_reference_test_auroc_trainval"]
    dbl_best = max(
        (_double_best_kind(summary, p)[1] for p in ["clin_rad", "clin_ct", "rad_ct"]),
        default=np.nan,
    )

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    xs = np.arange(len(methods))
    for i, m in enumerate(methods):
        v = m["val"]
        ax.bar(i, v if v is not None else 0, 0.75, color=m["color"], hatch=m["hatch"],
               edgecolor="white", alpha=0.95)
        if v is not None and not np.isnan(v):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in methods], fontsize=9)
    ax.set_ylabel("Test AUROC (trainval retrain)", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.axhline(sm["ct"], color=MODAL_COLOR["ct"], linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"CT single ({sm['ct']:.3f})")
    ax.axhline(dbl_best, color="#9b59b6", linestyle=":", linewidth=1.4,
               alpha=0.85, label=f"Best double ({dbl_best:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_title("Triple fusion — 8 tuning variants (baseline 3 + tuned 5, n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Separator between baseline and tuned groups (after first 3)
    ax.axvline(2.5, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.text(1, 0.78, "Baseline", ha="center", fontsize=10, alpha=0.7, style="italic")
    ax.text(5.5, 0.78, "Fusion-side tuned", ha="center", fontsize=10, alpha=0.7, style="italic")

    out = FIG_DIR / "fusion_tuning_bar.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 2. ROC curves
# ─────────────────────────────────────────────
def plot_roc(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    for m in ["clinical", "radiomics", "ct"]:
        y, p = _single_probs(m)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, "--", color=MODAL_COLOR[m], linewidth=1.2, alpha=0.6,
                label=f"{m} single (AUROC={auc:.3f})")

    for pair in ["clin_rad", "clin_ct", "rad_ct"]:
        best_kind, _ = _double_best_kind(summary, pair)
        d = _double_probs(pair, best_kind)
        if d is None:
            continue
        y, p = d
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, ":", color=PAIR_COLOR[pair], linewidth=1.5, alpha=0.8,
                label=f"{PAIR_LABEL[pair]} {best_kind} (AUROC={auc:.3f})")

    # Top triple methods (solid)
    triple_overlays = [
        ("Triple late logit", _triple_late_probs("logit"), TRIPLE_BASE, 1.6),
        ("Triple intermediate v2", _triple_intermediate_v2_probs(), "#16a085", 2.0),
        ("Triple linear L2", _triple_linear_probs("logreg_l2"), "black", 2.4),
    ]
    for label, probs, color, lw in triple_overlays:
        if probs is None:
            continue
        y, p = probs
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, "-", color=color, linewidth=lw,
                label=f"{label} (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC — Triple fusion (top 3) vs single / double (n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "roc_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 3. Confusion matrices — top triple methods
# ─────────────────────────────────────────────
def plot_confusion(threshold: float = 0.5) -> Path:
    panels = [("Triple late logit", _triple_late_probs("logit"))]
    v2 = _triple_intermediate_v2_probs()
    if v2 is not None:
        panels.append(("Triple intermediate v2", v2))
    lin = _triple_linear_probs("logreg_l2")
    if lin is not None:
        panels.append(("Triple linear L2", lin))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (title, (y, p)) in zip(axes, panels):
        pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        auc = roc_auc_score(y, p)
        ax.imshow(cm, cmap="Reds", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["<2yr", ">=2yr"]); ax.set_yticklabels(["<2yr", ">=2yr"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title}\nAUROC={auc:.3f} (thr={threshold})", fontsize=11)
    fig.suptitle("Triple fusion test confusion matrices (trainval retrain, n=63)",
                 fontsize=12, fontweight="bold", y=1.02)
    out = FIG_DIR / "confusion_matrices.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    summary = _load_summary()
    p1 = plot_bar(summary)
    p1b = plot_fusion_tuning_bar(summary)
    p2 = plot_roc(summary)
    p3 = plot_confusion()
    for p in (p1, p1b, p2, p3):
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
