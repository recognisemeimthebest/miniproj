"""Single-modal comparison plots.

Reads the three modalities' test predictions + summary and writes:
  - comparison_bar.png    : grouped bar of val / test_trainonly / test_trainval
  - roc_curves.png        : test ROC overlay (trainval headline)
  - confusion_matrices.png: 1x3 confusion matrices (trainval headline, threshold=0.5)

Output directory is resolved relative to this file (single_modal_baseline/figures/).
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
SM_DIR = FOLDER_ROOT / "results"
CT_MLP_DIR = FOLDER_ROOT / "results" / "ct" / "optuna_mlp"
FIG_DIR = FOLDER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODAL_ORDER = ["clinical", "radiomics", "ct"]
COLORS = {"clinical": "#3498db", "radiomics": "#e67e22", "ct": "#2ecc71"}


def _load_summary() -> dict:
    with open(SM_DIR / "summary.json") as f:
        return json.load(f)["modalities"]



def _load_trainval_probs(modal: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, prob) for the trainval-retrained model on the test set."""
    if modal == "ct":
        df = pd.read_csv(CT_MLP_DIR / "test_predictions_optuna_mlp.csv")
        return df["label"].values.astype(int), df["prob_class1"].values.astype(float)
    df = pd.read_csv(SM_DIR / modal / "test_predictions.csv")
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


# ─────────────────────────────────────────────
# 1. Grouped bar chart
# ─────────────────────────────────────────────
def plot_comparison_bar(summary: dict) -> Path:
    labels = ["Val AUROC\n(Optuna best)", "Test AUROC\n(train only)", "Test AUROC\n(train+val, headline)"]
    keys = ["optuna_best_val_auroc", "test_auroc_trainonly", "test_auroc_trainval"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, modal in enumerate(MODAL_ORDER):
        m = summary[modal]
        vals = [m[k] for k in keys]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=f"{modal.capitalize()} (input={m['input_dim']})",
            color=COLORS[modal],
            edgecolor="white",
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.set_title(
        "Single-modal AUROC on shared split (test n=63)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "comparison_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 2. ROC curves
# ─────────────────────────────────────────────
def plot_roc_curves() -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    for modal in MODAL_ORDER:
        y, p = _load_trainval_probs(modal)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(
            fpr,
            tpr,
            color=COLORS[modal],
            linewidth=2.0,
            label=f"{modal.capitalize()} (AUROC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "Test ROC — single modalities (train+val retrain, n=63)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    out = FIG_DIR / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 3. Confusion matrices
# ─────────────────────────────────────────────
def plot_confusion_matrices(threshold: float = 0.5) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, modal in zip(axes, MODAL_ORDER):
        y, p = _load_trainval_probs(modal)
        pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        auc = roc_auc_score(y, p)
        ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["<2yr", "≥2yr"])
        ax.set_yticklabels(["<2yr", "≥2yr"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{modal.capitalize()}\nAUROC={auc:.3f} (threshold={threshold})",
            fontsize=11,
        )

    fig.suptitle(
        "Test confusion matrices — single modalities (train+val retrain, n=63)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    out = FIG_DIR / "confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    summary = _load_summary()
    p1 = plot_comparison_bar(summary)
    p2 = plot_roc_curves()
    p3 = plot_confusion_matrices()
    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p3}")


if __name__ == "__main__":
    main()
