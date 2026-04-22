"""Shared train/val/test PatientID split for single-modal comparison.

All three modalities (clinical / radiomics / CT) are evaluated on the same
293/64/63 partition originally produced by the LJW seed99 3D-CNN pipeline.
The partition is read from results/ct/features/*.npz (local copies inside
this single_modal_baseline folder, so teammates can run the scripts
without pulling any other directory of the repo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FOLDER_ROOT = Path(__file__).resolve().parents[1]
SEED99_FEATURES = FOLDER_ROOT / "results" / "ct" / "features"


def load_split_pids() -> dict[str, list[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} patient ID lists."""
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        d = np.load(SEED99_FEATURES / f"{split}.npz", allow_pickle=True)
        out[split] = [str(p) for p in d["pids"]]
    return out


def load_split_labels() -> dict[str, np.ndarray]:
    """Return {'train': y, 'val': y, 'test': y} label arrays (int)."""
    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        d = np.load(SEED99_FEATURES / f"{split}.npz", allow_pickle=True)
        out[split] = d["y"].astype(int)
    return out


if __name__ == "__main__":
    pids = load_split_pids()
    ys = load_split_labels()
    for k in ("train", "val", "test"):
        print(f"{k}: n={len(pids[k])}  pos={int(ys[k].sum())}  neg={int((ys[k]==0).sum())}")
