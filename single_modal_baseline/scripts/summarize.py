"""Aggregate single-modal results (clinical / radiomics / CT) into one summary.

- Clinical + radiomics results are read from their respective results.json files.
- CT numbers come from results/ct/optuna_mlp/optuna_mlp_summary.json (local copy
  of the LJW seed99 MLP-head Optuna search).

The headline metric is the "train+val" retrain test AUROC (matches the CT MLP-head
'final_test_auroc_trainval' used as the reference point: 0.6337).
"""
from __future__ import annotations

import json
from pathlib import Path

FOLDER_ROOT = Path(__file__).resolve().parents[1]
SM_DIR = FOLDER_ROOT / "results"
CT_SUM = FOLDER_ROOT / "results" / "ct" / "optuna_mlp" / "optuna_mlp_summary.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    cln = _load(SM_DIR / "clinical" / "results.json")
    rad = _load(SM_DIR / "radiomics" / "results.json")
    ct = _load(CT_SUM)

    summary = {
        "split": {"train": 293, "val": 64, "test": 63, "source": "LJW seed99"},
        "modalities": {
            "clinical": {
                "input_dim": cln["input_dim"],
                "best_params": cln["best_params"],
                "optuna_best_val_auroc": cln["optuna_best_val_auroc"],
                "val_auroc_trainonly": cln["val_trainonly"]["auroc"],
                "test_auroc_trainonly": cln["test_trainonly"]["auroc"],
                "test_auroc_trainval": cln["test_trainval"]["auroc"],
            },
            "radiomics": {
                "input_dim": rad["input_dim"],
                "best_params": rad["best_params"],
                "optuna_best_val_auroc": rad["optuna_best_val_auroc"],
                "val_auroc_trainonly": rad["val_trainonly"]["auroc"],
                "test_auroc_trainonly": rad["test_trainonly"]["auroc"],
                "test_auroc_trainval": rad["test_trainval"]["auroc"],
            },
            "ct": {
                "input_dim": 256,
                "best_params": ct["best_params"],
                "optuna_best_val_auroc": ct["best_val_auroc"],
                "val_auroc_trainonly": ct["final_val_auroc_trainonly"],
                "test_auroc_trainonly": ct["final_test_auroc_trainonly"],
                "test_auroc_trainval": ct["final_test_auroc_trainval"],
                "note": "Feature input = 256-dim from LJW seed99 3D CNN encoder (best.pt)",
            },
        },
    }

    out_path = SM_DIR / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 68)
    print("  Single-modal comparison on LJW seed99 split (train=293, val=64, test=63)")
    print("=" * 68)
    hdr = f"{'Modality':<10} {'input':>6} | {'val(OptMax)':>12} {'test trainonly':>15} {'test trainval':>15}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("clinical", "radiomics", "ct"):
        m = summary["modalities"][name]
        print(
            f"{name:<10} {m['input_dim']:>6} | "
            f"{m['optuna_best_val_auroc']:>12.4f} "
            f"{m['test_auroc_trainonly']:>15.4f} "
            f"{m['test_auroc_trainval']:>15.4f}"
        )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
