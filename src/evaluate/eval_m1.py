"""M1 챔피언 체크포인트 test set 평가.

사용:
    python src/evaluate/eval_m1.py --ckpt experiments/m1_run1/best.pt

출력:
  - val AUROC (held-out 64 샘플)
  - test AUROC (held-out 63 샘플)
  - test accuracy / confusion matrix / per-class precision & recall
  - experiments/m1_run1/test_predictions.csv (환자별 prob/label)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import PersistentDataset, list_data_collate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import build_preprocess
from src.data.dataset import load_manifest, split_manifest
from src.models import build_nsclc_classifier, build_hosny_cnn


def _to_data_list(df):
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "image": str(r["image_path"]),
            "mask": str(r["mask_path"]),
            "label": int(r["label"]),
            "patient_id": r["patient_id"],
        })
    return rows


@torch.no_grad()
def run_eval(model, loader, device, amp: bool, tta: bool):
    model.eval()
    pids, labels_all, probs_all = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        label = batch["label"]
        if isinstance(label, torch.Tensor):
            label = label.cpu().tolist()
        ac = torch.amp.autocast("cuda", dtype=torch.float16) if amp else None
        if ac:
            with ac:
                logits = model(img)
                if tta:
                    logits = (logits + model(torch.flip(img, dims=[2]))) / 2
        else:
            logits = model(img)
            if tta:
                logits = (logits + model(torch.flip(img, dims=[2]))) / 2
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
        pids.extend(batch["patient_id"])
        labels_all.extend(label)
        probs_all.extend(probs)
    return pids, np.array(labels_all), np.array(probs_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/m1_run1/best.pt"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/preproc"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--tta", action="store_true", help="L-R flip TTA")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="기본은 ckpt 폴더. 환자별 prediction CSV 저장 위치.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="이진 예측 threshold (정확도/혼동행렬용)")
    ap.add_argument("--arch", type=str, default="resnet18",
                    choices=["resnet18", "hosny"],
                    help="모델 아키텍처. train.py 와 일치시켜야 함.")
    ap.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 64],
                    metavar=("H", "W", "D"),
                    help="Crop 크기. Hosny 재현은 80 80 80.")
    args = ap.parse_args()

    if args.out is None:
        args.out = args.ckpt.parent
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} ckpt={args.ckpt}", flush=True)

    # 1. data
    manifest = load_manifest()
    splits = split_manifest(manifest, seed=args.seed)
    for name, df in splits.items():
        p = int((df["label"] == 1).sum()); n = int((df["label"] == 0).sum())
        print(f"  {name}: n={len(df)} class1={p} class0={n}", flush=True)

    val_tf = build_preprocess(training=False, roi_size=tuple(args.roi_size))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    val_ds = PersistentDataset(
        data=_to_data_list(splits["val"]),
        transform=val_tf,
        cache_dir=str(args.cache_dir / "val"),
    )
    test_ds = PersistentDataset(
        data=_to_data_list(splits["test"]),
        transform=val_tf,
        cache_dir=str(args.cache_dir / "test"),
    )

    nw = args.num_workers
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=False,
                            collate_fn=list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=False,
                             collate_fn=list_data_collate)

    # 2. model + checkpoint
    if args.arch == "hosny":
        model = build_hosny_cnn(num_classes=2, dropout_fc=args.dropout).to(device)
    else:
        model = build_nsclc_classifier(num_classes=2, pretrained=False, dropout=args.dropout).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    # checkpoint 구조 (train.py L329): {"epoch", "model_state_dict", "val_auroc", "args", "ema"}
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"[ckpt] loaded epoch={state.get('epoch')} val_auroc={state.get('val_auroc'):.4f}",
              flush=True)
    else:
        model.load_state_dict(state)
        print(f"[ckpt] loaded raw state_dict", flush=True)

    # 3. val (재현 확인)
    _, val_y, val_p = run_eval(model, val_loader, device, amp=args.amp, tta=args.tta)
    val_auroc = roc_auc_score(val_y, val_p)
    print(f"\n[val ] n={len(val_y)} AUROC={val_auroc:.4f}", flush=True)

    # 4. test (진짜 held-out)
    pids, test_y, test_p = run_eval(model, test_loader, device, amp=args.amp, tta=args.tta)
    test_auroc = roc_auc_score(test_y, test_p)
    print(f"[test] n={len(test_y)} AUROC={test_auroc:.4f}", flush=True)

    # binary metrics
    pred = (test_p >= args.threshold).astype(int)
    acc = accuracy_score(test_y, pred)
    cm = confusion_matrix(test_y, pred, labels=[0, 1])
    pr, rc, f1, _ = precision_recall_fscore_support(test_y, pred, labels=[0, 1], zero_division=0)

    print(f"\n[test binary @ threshold={args.threshold}]")
    print(f"  accuracy        = {acc:.4f}")
    print(f"  confusion matrix (rows=true 0/1, cols=pred 0/1):")
    print(f"    [[{cm[0,0]:3d} {cm[0,1]:3d}]")
    print(f"     [{cm[1,0]:3d} {cm[1,1]:3d}]]")
    print(f"  class 0 (died <=2y):     precision={pr[0]:.3f}  recall={rc[0]:.3f}  f1={f1[0]:.3f}")
    print(f"  class 1 (survived >2y):  precision={pr[1]:.3f}  recall={rc[1]:.3f}  f1={f1[1]:.3f}")

    # 5. save
    csv_path = args.out / "test_predictions.csv"
    pd.DataFrame({
        "PatientID": pids, "label": test_y.astype(int),
        "prob_class1": test_p, "pred@0.5": pred.astype(int),
    }).to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}", flush=True)

    summary = {
        "ckpt": str(args.ckpt),
        "val_auroc": float(val_auroc), "val_n": int(len(val_y)),
        "test_auroc": float(test_auroc), "test_n": int(len(test_y)),
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "class0_precision_recall_f1": [float(pr[0]), float(rc[0]), float(f1[0])],
        "class1_precision_recall_f1": [float(pr[1]), float(rc[1]), float(f1[1])],
        "tta": args.tta, "threshold": args.threshold,
    }
    (args.out / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {args.out / 'test_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
