"""Late-fusion comparison plots (3 pairs × 2 averaging rules).

Reads double_model/results/summary.json + per-pair predictions CSVs.
Also overlays single-modal baselines for direct comparison.

Outputs (double_model/figures/):
  - comparison_bar.png     : 3 pair × 2 rule bars + single-modal reference lines
  - roc_curves.png         : 3 pair ROC (arithmetic) + 3 single-modal (dashed)
  - confusion_matrices.png : 1x3 grid of pair confusion matrices (arithmetic, threshold=0.5)
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

PAIR_ORDER = ["clin_rad", "clin_ct", "rad_ct"]
PAIR_LABEL = {"clin_rad": "Clin+Rad", "clin_ct": "Clin+CT", "rad_ct": "Rad+CT"}
PAIR_COLOR = {"clin_rad": "#9b59b6", "clin_ct": "#16a085", "rad_ct": "#d35400"}

MODAL_REF_COLOR = {"clinical": "#3498db", "radiomics": "#e67e22", "ct": "#2ecc71"}


def _load_summary() -> dict:
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


def _load_pair_probs(pair: str, rule: str = "arithmetic") -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(RESULTS / pair / f"predictions_{rule}.csv")
    prob_col = f"prob_fused_{rule}"
    return df["y_true"].values.astype(int), df[prob_col].values.astype(float)


def _load_single_probs(modal: str) -> tuple[np.ndarray, np.ndarray]:
    if modal == "ct":
        df = pd.read_csv(SM_RESULTS / "ct" / "optuna_mlp" / "test_predictions_optuna_mlp.csv")
        return df["label"].values.astype(int), df["prob_class1"].values.astype(float)
    df = pd.read_csv(SM_RESULTS / modal / "test_predictions.csv")
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def plot_bar(summary: dict) -> Path:
    pair_arith = [summary["pairs"][p]["arithmetic_mean"]["test_auroc"] for p in PAIR_ORDER]
    pair_logit = [summary["pairs"][p]["logit_mean"]["test_auroc"] for p in PAIR_ORDER]
    single = summary["single_modal_reference_test_auroc_trainval"]

    x = np.arange(len(PAIR_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, pair_arith, width, label="Arithmetic mean",
                color=[PAIR_COLOR[p] for p in PAIR_ORDER], edgecolor="white")
    b2 = ax.bar(x + width/2, pair_logit, width, label="Logit mean",
                color=[PAIR_COLOR[p] for p in PAIR_ORDER], edgecolor="white",
                alpha=0.55, hatch="//")
    for bars, vals in [(b1, pair_arith), (b2, pair_logit)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    for modal, val in single.items():
        ax.axhline(val, color=MODAL_REF_COLOR[modal], linestyle="--", linewidth=1.2,
                   label=f"{modal} single ({val:.3f})", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")

    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABEL[p] for p in PAIR_ORDER], fontsize=11)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.set_title(
        "Late fusion (equal weights) on LJW seed99 test (n=63)\n"
        "source probs = train+val retrain",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "comparison_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_roc() -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    for modal in ["clinical", "radiomics", "ct"]:
        y, p = _load_single_probs(modal)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, "--", color=MODAL_REF_COLOR[modal], linewidth=1.3, alpha=0.7,
                label=f"{modal} single (AUROC={auc:.3f})")
    for pair in PAIR_ORDER:
        y, p = _load_pair_probs(pair, "arithmetic")
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=PAIR_COLOR[pair], linewidth=2.2,
                label=f"{PAIR_LABEL[pair]} arith (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC — pair late fusion vs single modalities (n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "roc_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def plot_confusion(threshold: float = 0.5) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, pair in zip(axes, PAIR_ORDER):
        y, p = _load_pair_probs(pair, "arithmetic")
        pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        auc = roc_auc_score(y, p)
        ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["<2yr", ">=2yr"]); ax.set_yticklabels(["<2yr", ">=2yr"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{PAIR_LABEL[pair]} (arith)\nAUROC={auc:.3f} (thr={threshold})",
                     fontsize=11)
    fig.suptitle(
        "Test confusion matrices — pair late fusion, arithmetic mean (n=63)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    out = FIG_DIR / "confusion_matrices.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    summary = _load_summary()
    p1 = plot_bar(summary)
    p2 = plot_roc()
    p3 = plot_confusion()
    for p in (p1, p2, p3):
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
