"""Grad-CAM (3D) on Hosny CNN — M2 CT branch explainer.

기획서 §3.7.1 구현. 단일 환자 스모크 테스트부터 Test n=63 전수까지 공통 사용.

- 대상 모델: Hosny 2018 4-block 3D CNN (`single_modal_baseline/results/ct/best.pt`)
- Target layer: `model.features[3].act` (마지막 block ReLU 직후)
- Score: `target_class` logit (default 1 = 2년 생존)
- 출력: (80, 80, 80) NIfTI heatmap + axial MIP 3-pane PNG

사용 (repo root에서):
    # 스모크 (첫 test PID 자동 선택)
    PYTHONPATH=. python src/explain/gradcam_ct.py --smoke

    # 특정 PID
    PYTHONPATH=. python src/explain/gradcam_ct.py --pid LUNG1-355

    # Test 전수 (n=63)
    PYTHONPATH=. python src/explain/gradcam_ct.py --all-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    XGradCAM,
    LayerCAM,
    EigenCAM,
    EigenGradCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
}

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from src.data import build_preprocess, load_manifest
from src.models import build_hosny_cnn


CKPT_DEFAULT = Path("single_modal_baseline/results/ct/best.pt")
SPLIT_TEST_NPZ = Path("single_modal_baseline/results/ct/features/test.npz")
OUT_NII_DEFAULT = Path("results/gradcam/test")
OUT_PNG_DEFAULT = Path("figures/gradcam_overlays")


def load_model(ckpt: Path, device: torch.device, dropout_fc: float = 0.4):
    model = build_hosny_cnn(num_classes=2, dropout_fc=dropout_fc).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch")
        v = state.get("val_auroc")
        print(f"[ckpt] {ckpt.name} epoch={epoch} val_auroc={v:.4f}" if v else f"[ckpt] {ckpt.name}", flush=True)
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def get_test_pids() -> list[str]:
    d = np.load(SPLIT_TEST_NPZ, allow_pickle=True)
    return [str(p) for p in d["pids"]]


def load_sample(manifest, pid: str, device: torch.device) -> dict:
    row = manifest[manifest["patient_id"] == pid].iloc[0]
    data = {
        "image": str(row["image_path"]),
        "mask": str(row["mask_path"]),
        "label": int(row["label"]),
        "patient_id": row["patient_id"],
    }
    tf = build_preprocess(training=False, roi_size=(80, 80, 80))
    sample = tf(data)
    # mask is float 0/1 after Spacingd; binarize for safety
    mask_t = sample["mask"]
    if torch.is_tensor(mask_t):
        mask_np = (mask_t.detach().cpu().numpy() > 0.5)
    else:
        mask_np = (np.asarray(mask_t) > 0.5)
    if mask_np.ndim == 4:  # (1, D, H, W) -> (D, H, W)
        mask_np = mask_np[0]
    return {
        "image": sample["image"].unsqueeze(0).to(device),
        "mask": mask_np.astype(bool),   # (80, 80, 80)
        "label": int(sample["label"]) if not torch.is_tensor(sample["label"]) else int(sample["label"].item()),
        "pid": pid,
    }


def compute_metrics(cam_iou: np.ndarray, cam_peak: np.ndarray, gtv_mask: np.ndarray) -> dict:
    """IoU@25%, IoU@50%, Pointing-game per patient.

    Uses `cam_iou` for IoU thresholds and `cam_peak` for argmax (so that
    plateau-inducing post-proc like p99-clip doesn't corrupt PG via tie break).
    """
    out = {"gtv_voxels": int(gtv_mask.sum())}
    if out["gtv_voxels"] == 0:
        out.update({"iou_25": float("nan"), "iou_50": float("nan"), "pointing_game": float("nan")})
        return out

    for k in (25, 50):
        thr = np.percentile(cam_iou, 100 - k)
        cam_bin = cam_iou >= thr
        inter = np.logical_and(cam_bin, gtv_mask).sum()
        union = np.logical_or(cam_bin, gtv_mask).sum()
        out[f"iou_{k}"] = float(inter / union) if union > 0 else 0.0

    peak_idx = np.unravel_index(int(np.argmax(cam_peak)), cam_peak.shape)
    out["pointing_game"] = float(bool(gtv_mask[peak_idx]))
    out["peak_idx"] = list(map(int, peak_idx))
    return out


def run_gradcam(
    model,
    img: torch.Tensor,
    target_class: int,
    cam_method: str = "gradcam",
    target_block: int = 3,
    post: str = "relu_minmax",
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cam_post, cam_pre_post). Both are upsampled to input spatial size."""
    target_layers = [model.features[target_block].act]  # ReLU pre-pool
    cam_cls = CAM_METHODS[cam_method]
    cam = cam_cls(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    heatmap = cam(input_tensor=img, targets=targets)  # numpy (B, ...)
    cam_np = np.asarray(heatmap)[0]
    if cam_np.shape != tuple(img.shape[2:]):
        t = torch.from_numpy(cam_np).float()[None, None]
        t = F.interpolate(t, size=tuple(img.shape[2:]), mode="trilinear", align_corners=False)
        cam_np = t.squeeze().numpy()
    cam_pre_post = cam_np.copy()  # already ReLU+minmax'd by pytorch-grad-cam

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        cam_np = gaussian_filter(cam_np, sigma=smooth_sigma)

    if post == "p75p99":
        lo = np.percentile(cam_np, 75)
        hi = np.percentile(cam_np, 99)
        cam_np = np.clip(cam_np, lo, hi)
        if hi > lo:
            cam_np = (cam_np - lo) / (hi - lo)
        else:
            cam_np = np.zeros_like(cam_np)
    elif post == "relu_minmax":
        lo, hi = cam_np.min(), cam_np.max()
        if hi > lo:
            cam_np = (cam_np - lo) / (hi - lo)
    return cam_np, cam_pre_post


def save_nifti(cam_np: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(cam_np.astype(np.float32), affine=np.eye(4))
    nib.save(nii, str(out_path))


def save_overlay_png(
    ct_np: np.ndarray,
    cam_np: np.ndarray,
    pid: str,
    label: int,
    prob_survivor: float,
    target_class: int,
    out_path: Path,
    gtv_mask: np.ndarray | None = None,
) -> None:
    """3 orthogonal slices through GTV centroid (or volume center if no mask).
    CAM shown only above p50 with value-proportional alpha, GTV contour overlaid."""
    # Slice through CAM peak so the brightest activation is always visible
    zc, yc, xc = np.unravel_index(int(np.argmax(cam_np)), cam_np.shape)

    thr = np.percentile(cam_np, 75)  # top 25% only — visual focus on high-signal
    cam_disp = np.where(cam_np > thr, cam_np, np.nan)

    views = [
        ("Axial", ct_np[:, :, xc].T, cam_disp[:, :, xc].T,
         (gtv_mask[:, :, xc].T if gtv_mask is not None else None)),
        ("Coronal", ct_np[:, yc, :].T, cam_disp[:, yc, :].T,
         (gtv_mask[:, yc, :].T if gtv_mask is not None else None)),
        ("Sagittal", ct_np[zc, :, :].T, cam_disp[zc, :, :].T,
         (gtv_mask[zc, :, :].T if gtv_mask is not None else None)),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, ct_sl, cam_sl, gtv_sl) in zip(axes, views):
        ax.imshow(ct_sl, cmap="gray", origin="lower")
        ax.imshow(cam_sl, cmap="hot", origin="lower", alpha=0.6, vmin=thr, vmax=cam_np.max())
        if gtv_sl is not None and gtv_sl.any():
            ax.contour(gtv_sl.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(
        f"PID={pid}  label={label}  P(surv)={prob_survivor:.3f}  class={target_class}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_one(
    model,
    manifest,
    pid: str,
    device: torch.device,
    target_class: int,
    out_nii_dir: Path,
    out_png_dir: Path,
    cam_method: str = "gradcam",
    target_block: int = 3,
    post: str = "relu_minmax",
    smooth_sigma: float = 0.0,
    lung_mask: bool = False,
) -> dict:
    s = load_sample(manifest, pid, device)
    # probability
    with torch.no_grad():
        logits = model(s["image"])
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    cam_np, cam_pre = run_gradcam(
        model, s["image"], target_class,
        cam_method=cam_method, target_block=target_block,
        post=post, smooth_sigma=smooth_sigma,
    )
    ct_np = s["image"][0, 0].detach().cpu().numpy()

    if lung_mask:
        # CT is min-max to [0,1] from HU [-1000, 400]. HU > -500 → norm > 0.357.
        tissue = (ct_np > 0.357).astype(np.float32)
        cam_np = cam_np * tissue
        cam_pre = cam_pre * tissue
        lo, hi = cam_np.min(), cam_np.max()
        if hi > lo:
            cam_np = (cam_np - lo) / (hi - lo)

    metrics = compute_metrics(cam_iou=cam_np, cam_peak=cam_pre, gtv_mask=s["mask"])

    nii_path = out_nii_dir / f"{pid}_gradcam.nii.gz"
    png_path = out_png_dir / f"{pid}.png"
    save_nifti(cam_np, nii_path)
    save_overlay_png(
        ct_np=ct_np,
        cam_np=cam_np,
        pid=pid,
        label=s["label"],
        prob_survivor=float(probs[1]),
        target_class=target_class,
        out_path=png_path,
        gtv_mask=s["mask"],
    )
    return {
        "pid": pid,
        "label": s["label"],
        "prob_class0": float(probs[0]),
        "prob_class1": float(probs[1]),
        "cam_min": float(cam_np.min()),
        "cam_max": float(cam_np.max()),
        "gtv_voxels": metrics.get("gtv_voxels"),
        "iou_25": metrics.get("iou_25"),
        "iou_50": metrics.get("iou_50"),
        "pointing_game": metrics.get("pointing_game"),
        "nii": str(nii_path),
        "png": str(png_path),
    }


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="첫 test PID 1명만 처리")
    mode.add_argument("--pid", type=str, help="처리할 단일 PID (e.g. LUNG1-355)")
    mode.add_argument("--all-test", action="store_true", help="Test 전수 n=63")

    ap.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--target-class", type=int, default=1, choices=[0, 1])
    ap.add_argument("--out-nii", type=Path, default=OUT_NII_DEFAULT)
    ap.add_argument("--out-png", type=Path, default=OUT_PNG_DEFAULT)
    ap.add_argument("--dropout-fc", type=float, default=0.4)
    ap.add_argument("--cam-method", type=str, default="gradcam", choices=list(CAM_METHODS.keys()))
    ap.add_argument("--target-block", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--post", type=str, default="relu_minmax", choices=["relu_minmax", "p75p99"])
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian σ (voxels) applied before post")
    ap.add_argument("--lung-mask", action="store_true", help="Zero CAM voxels outside tissue (CT HU > −500)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[env] device={device} ckpt={args.ckpt} cam={args.cam_method} block={args.target_block} post={args.post}",
        flush=True,
    )

    model = load_model(args.ckpt, device, dropout_fc=args.dropout_fc)
    manifest = load_manifest()

    # PID 결정
    if args.all_test:
        pids = get_test_pids()
        print(f"[mode] all-test: n={len(pids)}", flush=True)
    elif args.pid:
        pids = [args.pid]
    else:  # default smoke
        pids = [get_test_pids()[0]]
        print(f"[mode] smoke: first test PID = {pids[0]}", flush=True)

    results = []
    for pid in pids:
        try:
            r = process_one(
                model=model,
                manifest=manifest,
                pid=pid,
                device=device,
                target_class=args.target_class,
                out_nii_dir=args.out_nii,
                out_png_dir=args.out_png,
                cam_method=args.cam_method,
                target_block=args.target_block,
                post=args.post,
                smooth_sigma=args.smooth_sigma,
                lung_mask=args.lung_mask,
            )
            results.append(r)
            print(
                f"[done] {pid}  label={r['label']} P(surv)={r['prob_class1']:.3f} "
                f"IoU@25={r['iou_25']:.3f} IoU@50={r['iou_50']:.3f} PG={r['pointing_game']:.0f}",
                flush=True,
            )
        except Exception as e:
            print(f"[fail] {pid}: {e}", flush=True)

    print(f"[summary] processed={len(results)}/{len(pids)}", flush=True)

    # all-test 모드: CSV + 요약 통계 + 히스토그램 저장
    if args.all_test and results:
        import csv
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 전체 환자 CSV
        iou_csv = results_dir / "gradcam_iou.csv"
        fieldnames = [
            "pid", "label", "prob_class0", "prob_class1",
            "gtv_voxels", "iou_25", "iou_50", "pointing_game",
        ]
        with iou_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"[save] per-patient CSV → {iou_csv}", flush=True)

        # 요약 통계 (NaN 제외)
        iou25 = np.array([r["iou_25"] for r in results], dtype=float)
        iou50 = np.array([r["iou_50"] for r in results], dtype=float)
        pg = np.array([r["pointing_game"] for r in results], dtype=float)
        valid25 = iou25[~np.isnan(iou25)]
        valid50 = iou50[~np.isnan(iou50)]
        validpg = pg[~np.isnan(pg)]

        summary = (
            f"Grad-CAM quantitative evaluation (Test n={len(results)})\n"
            f"  IoU@25%       mean={valid25.mean():.3f}  std={valid25.std():.3f}  n={len(valid25)}\n"
            f"  IoU@50%       mean={valid50.mean():.3f}  std={valid50.std():.3f}  n={len(valid50)}\n"
            f"  Pointing-game rate={validpg.mean():.3f}  n_hit={int(validpg.sum())}/{len(validpg)}\n"
        )
        summary_path = results_dir / "gradcam_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        print(f"[save] summary → {summary_path}\n{summary}", flush=True)

        # 히스토그램 PNG
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(valid25, bins=20, color="steelblue", edgecolor="black")
        axes[0].set_title(f"IoU@25%  (mean={valid25.mean():.3f})")
        axes[0].set_xlabel("IoU"); axes[0].set_ylabel("count")
        axes[1].hist(valid50, bins=20, color="orange", edgecolor="black")
        axes[1].set_title(f"IoU@50%  (mean={valid50.mean():.3f})")
        axes[1].set_xlabel("IoU")
        axes[2].bar([0, 1], [(validpg == 0).sum(), (validpg == 1).sum()], color=["lightcoral", "mediumseagreen"])
        axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["miss", "hit"])
        axes[2].set_title(f"Pointing-game  (rate={validpg.mean():.3f})")
        plt.tight_layout()
        hist_path = Path("figures/gradcam_iou_hist.png")
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] histogram → {hist_path}", flush=True)


if __name__ == "__main__":
    main()
