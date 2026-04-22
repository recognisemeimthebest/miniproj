"""M1 CT-only 2년 생존 분류 학습 스크립트 (MedicalNet ResNet-18 fine-tune).

기획서 §3.4 M1 1차 파이프라인. 과적합 방지:
  - pretrained backbone (MedicalNet 23 dataset)
  - class-weighted CE + label_smoothing
  - WeightedRandomSampler (class balance per batch)
  - AdamW + weight_decay, differential LR (backbone / head)
  - CosineAnnealingLR
  - gradient clipping + accumulation
  - early stopping on val AUROC
  - optional: warmup freeze, EMA, TTA

사용:
    python src/train.py --epochs 50 --batch-size 4 --num-workers 2 --out experiments/m1_run1
    python src/train.py --smoke                        # 2 epoch × 작은 샘플 스모크 테스트
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.data import PersistentDataset, list_data_collate
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_preprocess
from src.data.dataset import load_manifest, split_manifest
from src.models import build_nsclc_classifier, build_hosny_cnn
from src.optim import SAM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/m1_run1"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/preproc"),
                    help="PersistentDataset 디스크 캐시 디렉토리. 첫 실행만 캐시 생성.")
    ap.add_argument("--amp", action="store_true", help="Mixed precision (fp16 + GradScaler)")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 autocast (GradScaler 불필요). SAM과 호환성 좋음.")
    ap.add_argument("--sam", action="store_true",
                    help="Sharpness-Aware Minimization (Foret 2021). 1.7x slower per step.")
    ap.add_argument("--sam-rho", type=float, default=0.05,
                    help="SAM neighborhood 반경. 논문 default 0.05, small-data 0.05-0.1 권장.")
    ap.add_argument("--no-class-weight", action="store_true",
                    help="CE loss의 class weight 비활성. WeightedRandomSampler와 이중 보정 방지.")
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="첫 N epoch 동안 backbone freeze (head만 학습)")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="0이면 EMA 비활성. 일반적으로 0.999 권장.")
    ap.add_argument("--tta", action="store_true", help="Val/test 시 L-R flip TTA")
    # augmentation
    ap.add_argument("--rot-deg", type=float, default=5.0)
    ap.add_argument("--scale-pct", type=float, default=5.0)
    ap.add_argument("--affine-prob", type=float, default=0.5)
    ap.add_argument("--flip-prob", type=float, default=0.5)
    ap.add_argument("--intensity-prob", type=float, default=0.3)
    ap.add_argument("--intensity-scale", type=float, default=0.05)
    ap.add_argument("--intensity-shift", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epoch × train 8 / val 4 샘플로 파이프라인 빠르게 검증")
    # --- Hosny 2018 재현용 플래그 ---
    ap.add_argument("--arch", type=str, default="resnet18",
                    choices=["resnet18", "hosny"],
                    help="모델 아키텍처. resnet18=MedicalNet pretrained, hosny=shallow 4-block 3D CNN.")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="pretrained weight 로드 안 함 (arch=hosny는 항상 from-scratch).")
    ap.add_argument("--light-aug", action="store_true",
                    help="Hosny 2018 방식: flip만 augmentation (affine/intensity 끔).")
    ap.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 64],
                    metavar=("H", "W", "D"),
                    help="Crop 크기. Hosny 재현은 80 80 80 권장.")
    return ap.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_data_list(df):
    """manifest DataFrame → MONAI Dataset이 먹는 dict 리스트."""
    return [
        {
            "image": str(r["image_path"]),
            "mask": str(r["mask_path"]),
            "label": int(r["label"]),
            "patient_id": r["patient_id"],
        }
        for _, r in df.iterrows()
    ]


class EMA:
    """Exponential Moving Average of float parameters. Int buffers (e.g. BN num_batches_tracked)
    는 shadow에 단순 복사."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self._backup: dict | None = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_shadow(self, model: nn.Module):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        assert self._backup is not None
        model.load_state_dict(self._backup)
        self._backup = None


def build_loaders(splits, args):
    if args.smoke:
        splits = {k: v.head(8 if k == "train" else 4) for k, v in splits.items()}

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    roi = tuple(args.roi_size)
    train_tf = build_preprocess(
        training=True,
        roi_size=roi,
        rot_deg=args.rot_deg, scale_pct=args.scale_pct,
        affine_prob=args.affine_prob, flip_prob=args.flip_prob,
        intensity_prob=args.intensity_prob,
        intensity_scale=args.intensity_scale, intensity_shift=args.intensity_shift,
        light_aug=args.light_aug,
    )
    val_tf = build_preprocess(training=False, roi_size=roi)

    # PersistentDataset: 결정적 transform(Load/Spacing/Crop/Scale)을 디스크에 캐시.
    # 첫 실행 후 재실행은 거의 즉시 로드 → 하이퍼파라미터 스윕 빠름.
    train_ds = PersistentDataset(
        data=_to_data_list(splits["train"]),
        transform=train_tf,
        cache_dir=str(args.cache_dir / "train"),
    )
    val_ds = PersistentDataset(
        data=_to_data_list(splits["val"]),
        transform=val_tf,
        cache_dir=str(args.cache_dir / "val"),
    )

    # WeightedRandomSampler: 매 배치에서 class 균형
    labels = splits["train"]["label"].values
    class_counts = np.bincount(labels, minlength=2)
    per_sample_w = np.array([1.0 / class_counts[l] for l in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(per_sample_w, num_samples=len(labels), replacement=True)

    nw = args.num_workers
    # persistent_workers=False: warmup→unfreeze 전환 시 워커가 stale shared-memory를
    # 들고있어 "DataLoader worker exited unexpectedly"로 죽는 버그 회피 (Windows).
    # 캐시는 이미 디스크에 있으므로 에폭마다 워커 재생성해도 오버헤드 작음.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=nw, pin_memory=True, persistent_workers=False,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=False,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader, class_counts


def _label_from_batch(batch, device):
    lbl = batch["label"]
    if isinstance(lbl, torch.Tensor):
        return lbl.to(device, non_blocking=True)
    return torch.tensor(lbl, dtype=torch.long, device=device)


def evaluate(model, loader, loss_fn, device, amp: bool, tta: bool, bf16: bool = False):
    model.eval()
    labels_all, probs_all, losses = [], [], []
    # autocast dtype 결정: bf16 우선, 아니면 fp16(amp), 아니면 없음
    if bf16:
        ac = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    elif amp:
        ac = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        ac = None
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            label = _label_from_batch(batch, device)
            if ac is not None:
                with ac:
                    logits = model(img)
                    if tta:
                        logits = (logits + model(torch.flip(img, dims=[2]))) / 2
                    loss = loss_fn(logits, label)
            else:
                logits = model(img)
                if tta:
                    logits = (logits + model(torch.flip(img, dims=[2]))) / 2
                loss = loss_fn(logits, label)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            losses.append(loss.item())
            labels_all.extend(label.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
    try:
        auroc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auroc = float("nan")
    return float(np.mean(losses)), auroc


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}", flush=True)
    print(f"[cfg] warmup={args.warmup_epochs} ema={args.ema_decay} tta={args.tta} "
          f"accum={args.accum_steps} backbone_lr={args.backbone_lr} head_lr={args.head_lr} "
          f"wd={args.weight_decay} dropout={args.dropout} ls={args.label_smoothing} "
          f"rot={args.rot_deg}° scale={args.scale_pct}% affine_p={args.affine_prob}",
          flush=True)

    # 1. data
    manifest = load_manifest()
    splits = split_manifest(manifest, seed=args.seed)
    for name, df in splits.items():
        p = int((df["label"] == 1).sum()); n = int((df["label"] == 0).sum())
        print(f"  {name}: n={len(df)} class1={p} class0={n}", flush=True)

    train_loader, val_loader, class_counts = build_loaders(splits, args)

    # 2. model
    if args.arch == "hosny":
        # Hosny 2018 shallow 3D CNN: always from scratch (LIDC pretrain 없음)
        model = build_hosny_cnn(num_classes=2, dropout_fc=args.dropout).to(device)
        print(f"[model] arch=hosny (shallow 3D CNN, ~1M params, from-scratch)", flush=True)
    else:
        pretrained = not args.no_pretrained
        model = build_nsclc_classifier(num_classes=2, pretrained=pretrained, dropout=args.dropout).to(device)
        print(f"[model] arch=resnet18 pretrained={pretrained}", flush=True)

    # 3. loss
    # P5 (중복 보정 제거): WeightedRandomSampler가 이미 batch-level class balance를 맞추므로
    # class-weighted CE를 더하면 minority class에 over-correction. --no-class-weight로 끔.
    if args.no_class_weight:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[loss] class_weight=OFF (sampler only) label_smoothing={args.label_smoothing}",
              flush=True)
    else:
        total = int(class_counts.sum())
        class_w = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])],
                               dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)
        print(f"[loss] class_weight={class_w.tolist()} label_smoothing={args.label_smoothing}",
              flush=True)

    # 4. optimizer — differential LR (pretrained 보호)
    # SAM을 쓰면 AdamW를 base로 감싸는 wrapper. 학습 시간 1.7x, 체크포인트 크기 동일.
    # Hosny CNN은 pretrain 없으므로 단일 LR(head_lr) 사용.
    if args.arch == "hosny":
        param_groups = [{"params": model.parameters(), "lr": args.head_lr}]
    else:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ]
    if args.sam:
        optimizer = SAM(param_groups, torch.optim.AdamW,
                        rho=args.sam_rho, weight_decay=args.weight_decay)
        print(f"[opt] SAM(rho={args.sam_rho}) over AdamW", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    # scheduler는 base_optimizer (SAM일 때) 또는 optimizer 자체에 연결 — 둘 다 param_groups 공유
    sched_target = optimizer.base_optimizer if args.sam else optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sched_target, T_max=args.epochs)

    # 5. EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # 6. train loop
    epochs = 2 if args.smoke else args.epochs
    best_auroc = 0.0
    best_epoch = -1
    patience_counter = 0
    history = []
    # AMP mode: fp16+GradScaler | bf16 autocast (no scaler) | fp32
    # SAM은 GradScaler와 결합 시 first_step의 perturbation에 scale 섞여 미묘한 bug 발생.
    # → SAM일 땐 bf16 autocast로 자동 전환 권장 (bf16은 fp32 range이라 scaler 불필요).
    use_fp16 = args.amp and not args.sam
    use_bf16 = args.bf16 or (args.sam and args.amp)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    if use_fp16:
        print("[amp] fp16 + GradScaler", flush=True)
    elif use_bf16:
        print("[amp] bf16 autocast (no scaler)", flush=True)
    else:
        print("[amp] off (fp32)", flush=True)
    backbone_frozen = False

    for epoch in range(1, epochs + 1):
        # progressive unfreezing — Hosny CNN은 backbone 없으므로 skip
        if args.warmup_epochs > 0 and args.arch != "hosny":
            if epoch == 1 and not backbone_frozen:
                model.freeze_backbone()
                backbone_frozen = True
                print(f"[warmup] backbone frozen (epoch 1-{args.warmup_epochs})", flush=True)
            elif epoch == args.warmup_epochs + 1 and backbone_frozen:
                model.unfreeze_backbone()
                backbone_frozen = False
                print(f"[warmup] backbone unfrozen at epoch {epoch}", flush=True)

        t0 = time.time()
        # ── train ──
        model.train()
        train_losses = []
        optimizer.zero_grad()

        def _forward_loss(img, label):
            """autocast를 골라 forward+loss 계산. args.accum_steps로 나눔 (accum 지원)."""
            if use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(img)
                    loss = loss_fn(logits, label) / args.accum_steps
            elif use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(img)
                    loss = loss_fn(logits, label) / args.accum_steps
            else:
                logits = model(img)
                loss = loss_fn(logits, label) / args.accum_steps
            return loss

        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            label = _label_from_batch(batch, device)

            # ── SAM 경로 (2-step) ──
            if args.sam:
                # 1st: ascent to worst-case nearby point
                loss = _forward_loss(img, label)
                loss.backward()
                train_losses.append(loss.item() * args.accum_steps)
                # accum_steps > 1이면 first_step을 accum 경계에서만 (실험상 accum=1만 쓸 예정)
                if (step + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.first_step(zero_grad=True)
                    # 2nd: 재계산 후 실제 update
                    loss2 = _forward_loss(img, label)
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.second_step(zero_grad=True)
                    if ema is not None:
                        ema.update(model)
                continue

            # ── 일반 경로 (single step) ──
            loss = _forward_loss(img, label)
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_losses.append(loss.item() * args.accum_steps)

            if (step + 1) % args.accum_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

        # ── val (EMA weights 사용 시 적용 후 복원) ──
        if ema is not None:
            ema.apply_shadow(model)
        vl, vauc = evaluate(model, val_loader, loss_fn, device, use_fp16, args.tta, bf16=use_bf16)
        if ema is not None:
            ema.restore(model)

        scheduler.step()

        tr = float(np.mean(train_losses))
        dt = time.time() - t0
        row = {"epoch": epoch, "train_loss": tr, "val_loss": vl, "val_auroc": vauc, "seconds": dt}
        history.append(row)
        print(f"[epoch {epoch:02d}/{epochs}] train_loss={tr:.4f} val_loss={vl:.4f} "
              f"val_auroc={vauc:.4f} ({dt:.1f}s)", flush=True)

        # checkpoint
        if vauc > best_auroc:
            best_auroc = vauc
            best_epoch = epoch
            state = copy.deepcopy(ema.shadow) if ema is not None else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "val_auroc": vauc,
                "args": vars(args),
                "ema": ema is not None,
            }, args.out / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[early stop] no val AUROC improvement for {args.patience} epochs", flush=True)
                break

    # save history
    with (args.out / "history.json").open("w") as f:
        json.dump({"history": history,
                   "best_epoch": best_epoch, "best_val_auroc": best_auroc,
                   "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
                  f, indent=2)
    print(f"[done] best val AUROC = {best_auroc:.4f} at epoch {best_epoch}. "
          f"Checkpoint: {args.out/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
