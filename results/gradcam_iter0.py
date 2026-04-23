"""Grad-CAM iter0 reproducible runner — frozen snapshot for review.

Config (Project Plan §3.7.1, evaluator-guided tuning Iter 0):
    CAM method     = GradCAM
    Target layer   = model.features[3].act   (last ReLU pre-pool, 10^3)
    Post-proc      = ReLU + min-max to [0, 1]
    Smoothing      = none
    Lung mask      = off
    Target class   = 1 (2-year survivor)

Quantitative result on seed=123 CT checkpoint (test AUROC 0.601, n=63):
    IoU@25 = 0.149   IoU@50 = 0.282   PG = 0.127
    (Random baseline: IoU@25=0.193, IoU@50=0.315, PG=0.459 — iter0 below random.)

Iter0 is kept on the repo because visually the CAM is the most
GTV-concentrated single-blob of the 5-iter loop (deepest layer, lowest
spatial noise), even though its PG and IoU are worst quantitatively. See
docs/xai_interpretation.md §"Depth-peak trade-off" for why.

Usage (repo root):
    # demo on first test PID
    PYTHONPATH=. python results/gradcam_iter0.py --smoke
    # single PID
    PYTHONPATH=. python results/gradcam_iter0.py --pid LUNG1-015
    # full test batch (n=63) -> figures/gradcam_overlays_iter0/
    PYTHONPATH=. python results/gradcam_iter0.py --all-test

This is a thin wrapper that hardcodes the iter0 flags and delegates all
logic to src/explain/gradcam_ct.py. The canonical implementation lives there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.explain import gradcam_ct  # noqa: E402


# ---- iter0 frozen config --------------------------------------------------
ITER0_CONFIG = dict(
    cam_method="gradcam",
    target_block=3,
    post="relu_minmax",
    smooth_sigma=0.0,
    lung_mask=False,
    target_class=1,
)
OUT_PNG_DIR = Path("figures/gradcam_overlays_iter0")
OUT_NII_DIR = Path("results/gradcam/test_iter0")
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Grad-CAM iter0 (GradCAM, features[3].act, relu_minmax) — frozen runner."
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="first test PID only")
    mode.add_argument("--pid", type=str, help="single PID (e.g. LUNG1-015)")
    mode.add_argument("--all-test", action="store_true", help="full test batch n=63")
    args = ap.parse_args()

    # Re-dispatch through gradcam_ct.main() by injecting iter0 argv
    argv = [sys.argv[0]]
    if args.all_test:
        argv.append("--all-test")
    elif args.pid:
        argv += ["--pid", args.pid]
    else:
        argv.append("--smoke")
    argv += [
        "--cam-method", ITER0_CONFIG["cam_method"],
        "--target-block", str(ITER0_CONFIG["target_block"]),
        "--post", ITER0_CONFIG["post"],
        "--smooth-sigma", str(ITER0_CONFIG["smooth_sigma"]),
        "--target-class", str(ITER0_CONFIG["target_class"]),
        "--out-png", str(OUT_PNG_DIR),
        "--out-nii", str(OUT_NII_DIR),
    ]
    # iter0 does NOT use --lung-mask (that's iter5)

    sys.argv = argv
    print(f"[iter0] config={ITER0_CONFIG}", flush=True)
    print(f"[iter0] out-png={OUT_PNG_DIR}  out-nii={OUT_NII_DIR}", flush=True)
    gradcam_ct.main()


if __name__ == "__main__":
    main()
