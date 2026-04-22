"""Triple-fusion plots — late (arith/logit) + intermediate vs single + double references.

Reads:
  - triple_model/results/summary.json                    (late + intermediate numbers)
  - triple_model/results/triple/predictions_*.csv        (late fused probs)
  - triple_model/results/triple/intermediate/test_predictions.csv  (intermediate probs)
  - single_modal_baseline/results/{clinical,radiomics,ct}/*         (single-modal refs)
  - double_model/results/*.csv / summary.json                        (double refs)

Outputs (triple_model/figures/):
  - comparison_bar.png    : single (3 bars) | double best-per-pair (3 bars) | triple (3 bars: late arith / late logit / inter)
  - roc_curves.png        : triple (intermediate + late arith) + single-modal dashed overlay + best double per pair
  - confusion_matrices.png: 1x2 pair CM (late arith, intermediate) at threshold=0.5
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
TRIPLE_COLOR = "#c0392b"
PAIR_LABEL = {"clin_rad": "Clin+Rad", "clin_ct": "Clin+CT", "rad_ct": "Rad+CT"}


def _load_summary() -> dict:
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


def _triple_late_probs(rule: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(RESULTS / "triple" / f"predictions_{rule}.csv")
    return df["y_true"].values.astype(int), df[f"prob_fused_{rule}"].values.astype(float)


def _triple_intermediate_probs() -> tuple[np.ndarray, np.ndarray] | None:
    path = RESULTS / "triple" / "intermediate" / "test_predictions.csv"
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
    """Return (method_tag, auroc) for the best-scoring method per pair."""
    dbl = summary.get("double_reference", {}).get(pair, {})
    if not dbl:
        return ("late_arith", np.nan)
    best = max(dbl.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -np.inf))
    return best


# ─────────────────────────────────────────────
# 1. Comparison bar
# ─────────────────────────────────────────────
def plot_bar(summary: dict) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))

    sm = summary["single_modal_reference_test_auroc_trainval"]
    dbl = summary.get("double_reference", {})
    trp = summary["triple"]
    inter = summary.get("intermediate_fusion", {}).get("triple", {})

    bars = []
    pos = 0
    width = 0.7

    # Singles
    for m in ["clinical", "radiomics", "ct"]:
        bars.append((pos, sm[m], MODAL_COLOR[m], "", f"{m}\n(single)"))
        pos += 1
    pos += 0.6  # gap

    # Doubles (best method per pair)
    for pair in ["clin_rad", "clin_ct", "rad_ct"]:
        best_kind, best_val = _double_best_kind(summary, pair)
        bars.append((pos, best_val, PAIR_COLOR[pair], "",
                     f"{PAIR_LABEL[pair]}\n(double-{best_kind})"))
        pos += 1
    pos += 0.6  # gap

    # Triple: late arith / late logit / intermediate
    bars.append((pos, trp["arithmetic_mean"]["test_auroc"], TRIPLE_COLOR, "",
                 "Triple\n(late arith)"))
    pos += 1
    bars.append((pos, trp["logit_mean"]["test_auroc"], TRIPLE_COLOR, "//",
                 "Triple\n(late logit)"))
    pos += 1
    if inter:
        bars.append((pos, inter["test_auroc_trainval"], TRIPLE_COLOR, "xx",
                     "Triple\n(intermediate)"))
        pos += 1

    for x, v, c, h, _ in bars:
        ax.bar(x, v, width, color=c, hatch=h, edgecolor="white", alpha=0.95)
        if not (v is None or (isinstance(v, float) and np.isnan(v))):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    ax.set_xticks([b[0] for b in bars])
    ax.set_xticklabels([b[4] for b in bars], fontsize=9)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_ylim(0.4, 0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.axhline(sm["ct"], color=MODAL_COLOR["ct"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"CT single ({sm['ct']:.3f})")
    ax.set_title("Single → Double → Triple fusion on shared test split (n=63)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "comparison_bar.png"
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

    # Best double per pair
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

    # Triple
    y_a, p_a = _triple_late_probs("arithmetic")
    fpr, tpr, _ = roc_curve(y_a, p_a)
    auc = roc_auc_score(y_a, p_a)
    ax.plot(fpr, tpr, "-", color=TRIPLE_COLOR, linewidth=2.2,
            label=f"Triple late arith (AUROC={auc:.3f})")

    inter = _triple_intermediate_probs()
    if inter is not None:
        y, p = inter
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, "-", color="black", linewidth=2.2,
                label=f"Triple intermediate (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC — Triple fusion vs single / double (n=63)",
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
    panels = [("Triple late arith", _triple_late_probs("arithmetic"))]
    inter = _triple_intermediate_probs()
    if inter is not None:
        panels.append(("Triple intermediate", inter))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.2))
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
    p2 = plot_roc(summary)
    p3 = plot_confusion()
    for p in (p1, p2, p3):
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
