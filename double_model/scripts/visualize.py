"""Double-model comparison plots — late (arith/logit) vs intermediate fusion.

Reads double_model/results/summary.json (both `pairs` late section and
`intermediate_fusion.pairs`) + per-pair prediction CSVs.

Outputs (double_model/figures/):
  - comparison_bar.png     : 3 pair × 3 method (late arith, late logit, intermediate)
                             + single-modal baselines drawn as horizontal reference lines
  - roc_curves.png         : per pair, best method ROC (intermediate if available else
                             late arith) + single-modal dashed overlay
  - confusion_matrices.png : 1x3 pair CM using intermediate predictions (trainval, thr=0.5)
                             — falls back to late arithmetic if intermediate absent
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

METHOD_STYLE = {
    "late_arith": {"hatch": "",   "alpha": 1.0,  "label": "Late arith"},
    "late_logit": {"hatch": "//", "alpha": 0.55, "label": "Late logit"},
    "intermediate": {"hatch": "xx", "alpha": 0.85, "label": "Intermediate (frozen concat + MLP)"},
}


def _load_summary() -> dict:
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


def _late_probs(pair: str, rule: str = "arithmetic") -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(RESULTS / pair / f"predictions_{rule}.csv")
    prob_col = f"prob_fused_{rule}"
    return df["y_true"].values.astype(int), df[prob_col].values.astype(float)


def _intermediate_probs(pair: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = RESULTS / pair / "intermediate" / "test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def _single_probs(modal: str) -> tuple[np.ndarray, np.ndarray]:
    if modal == "ct":
        df = pd.read_csv(SM_RESULTS / "ct" / "optuna_mlp" / "test_predictions_optuna_mlp.csv")
        return df["label"].values.astype(int), df["prob_class1"].values.astype(float)
    df = pd.read_csv(SM_RESULTS / modal / "test_predictions.csv")
    return df["y_true"].values.astype(int), df["prob_trainval"].values.astype(float)


def _val(summary: dict, pair: str, method: str) -> float:
    if method == "late_arith":
        return summary["pairs"][pair]["arithmetic_mean"]["test_auroc"]
    if method == "late_logit":
        return summary["pairs"][pair]["logit_mean"]["test_auroc"]
    if method == "intermediate":
        return summary.get("intermediate_fusion", {}).get("pairs", {}).get(pair, {}).get("test_auroc_trainval", np.nan)
    raise KeyError(method)


# ─────────────────────────────────────────────
# 1. Grouped bar chart
# ─────────────────────────────────────────────
def plot_bar(summary: dict) -> Path:
    methods = ["late_arith", "late_logit", "intermediate"]
    x = np.arange(len(PAIR_ORDER))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 6))

    for i, method in enumerate(methods):
        vals = [_val(summary, p, method) for p in PAIR_ORDER]
        style = METHOD_STYLE[method]
        bars = ax.bar(
            x + (i - 1) * width, vals, width,
            label=style["label"],
            color=[PAIR_COLOR[p] for p in PAIR_ORDER],
            edgecolor="white",
            alpha=style["alpha"],
            hatch=style["hatch"],
        )
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    single = summary["single_modal_reference_test_auroc_trainval"]
    for modal, val in single.items():
        ax.axhline(val, color=MODAL_REF_COLOR[modal], linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"{modal} single ({val:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")

    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABEL[p] for p in PAIR_ORDER], fontsize=11)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.set_title(
        "Double-model fusion on shared test split (n=63) — late vs intermediate",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "comparison_bar.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 2. ROC curves
# ─────────────────────────────────────────────
def plot_roc() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for modal in ["clinical", "radiomics", "ct"]:
        y, p = _single_probs(modal)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, "--", color=MODAL_REF_COLOR[modal], linewidth=1.3, alpha=0.7,
                label=f"{modal} single (AUROC={auc:.3f})")
    for pair in PAIR_ORDER:
        inter = _intermediate_probs(pair)
        if inter is not None:
            y, p = inter
            tag = "inter"
        else:
            y, p = _late_probs(pair, "arithmetic")
            tag = "late arith"
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=PAIR_COLOR[pair], linewidth=2.2,
                label=f"{PAIR_LABEL[pair]} {tag} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC — best pair fusion vs single modalities (n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "roc_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# 3. Confusion matrices
# ─────────────────────────────────────────────
def plot_confusion(threshold: float = 0.5) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, pair in zip(axes, PAIR_ORDER):
        inter = _intermediate_probs(pair)
        if inter is not None:
            y, p = inter; tag = "intermediate"
        else:
            y, p = _late_probs(pair, "arithmetic"); tag = "late arith"
        pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        auc = roc_auc_score(y, p)
        ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["<2yr", ">=2yr"]); ax.set_yticklabels(["<2yr", ">=2yr"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{PAIR_LABEL[pair]} ({tag})\nAUROC={auc:.3f} (thr={threshold})",
                     fontsize=11)
    fig.suptitle(
        "Test confusion matrices — best pair fusion (trainval retrain, n=63)",
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
