"""Hosny CNN 256-dim GAP feature extraction.

iter 7b checkpoint 로드 → forward 중간 GAP 출력 (B, 256) 저장.
TTA (L-R flip) 평균 선택 가능.

사용:
    python src/extract_features.py \
        --ckpt experiments/m1_hosny_iter7b_seed99/best.pt \
        --seed 99 \
        --out experiments/m1_hosny_iter7b_seed99/features \
        --tta
출력:
    <out>/train.npz, val.npz, test.npz
    each: {"pids": (N,), "X": (N, 256), "y": (N,)}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import PersistentDataset, list_data_collate
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_preprocess
from src.data.dataset import load_manifest, split_manifest
from src.models import build_hosny_cnn


def _to_data_list(df):
    return [
        {
            "image": str(r["image_path"]),
            "mask": str(r["mask_path"]),
            "label": int(r["label"]),
            "patient_id": r["patient_id"],
        }
        for _, r in df.iterrows()
    ]


@torch.no_grad()
def extract(model, loader, device, tta: bool):
    model.eval()
    pids, labels, feats = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        # forward features → gap → flatten (head 직전까지)
        f = model.features(img)
        f = model.gap(f)
        f = model.flatten(f)  # (B, 256)
        if tta:
            fx = model.features(torch.flip(img, dims=[2]))
            fx = model.gap(fx)
            fx = model.flatten(fx)
            f = (f + fx) / 2
        label = batch["label"]
        if isinstance(label, torch.Tensor):
            label = label.cpu().tolist()
        pids.extend(batch["patient_id"])
        labels.extend(label)
        feats.append(f.float().cpu().numpy())
    return np.array(pids), np.array(labels, dtype=np.int64), np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/preproc_80"))
    ap.add_argument("--roi-size", type=int, nargs=3, default=[80, 80, 80])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--dropout-fc", type=float, default=0.4,
                    help="ckpt 학습시 쓴 값 (head shape 맞추기용, extract에 영향 없음)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} ckpt={args.ckpt}", flush=True)

    # data
    manifest = load_manifest()
    splits = split_manifest(manifest, seed=args.seed)
    for name, df in splits.items():
        p = int((df["label"] == 1).sum()); n = int((df["label"] == 0).sum())
        print(f"  {name}: n={len(df)} class1={p} class0={n}", flush=True)

    tf = build_preprocess(training=False, roi_size=tuple(args.roi_size))

    # model
    model = build_hosny_cnn(num_classes=2, dropout_fc=args.dropout_fc).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"[ckpt] loaded epoch={state.get('epoch')} val_auroc={state.get('val_auroc'):.4f}",
              flush=True)
    else:
        model.load_state_dict(state)

    for split_name in ("train", "val", "test"):
        ds = PersistentDataset(
            data=_to_data_list(splits[split_name]),
            transform=tf,
            cache_dir=str(args.cache_dir / split_name),
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False, collate_fn=list_data_collate,
        )
        pids, y, X = extract(model, loader, device, tta=args.tta)
        out_path = args.out / f"{split_name}.npz"
        np.savez(out_path, pids=pids, X=X, y=y)
        print(f"[save] {split_name}: N={len(y)} X={X.shape} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
