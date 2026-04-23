"""각 test 환자의 CT NIfTI → Plotly 3D volume HTML.

Output: web/patients3d/<PID>.html (interactive 3D volume + GTV overlay)
줌/회전/팬 모두 브라우저에서 가능.

Usage:
    cd /home/team4/miniproj_ljw/web
    python build_3d_viewers.py --smoke      # 1명 (LUNG1-355)
    python build_3d_viewers.py              # test 63명 전체
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_fill_holes, label, zoom
from skimage.measure import marching_cubes


def extract_lung_mask(hu):
    body = hu > -500
    filled = np.zeros_like(body)
    for k in range(body.shape[2]):
        filled[:, :, k] = binary_fill_holes(body[:, :, k])
    lung = (hu < -500) & filled
    lab, n = label(lung, structure=np.ones((3, 3, 3)))
    if n == 0:
        return lung
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.isin(lab, np.argsort(sizes)[-2:])


def refine_gtv(hu_raw, gtv_raw, spacing_mm):
    """RTSTRUCT 합본에서 실제 GTV만 heuristic으로 추출.

    규칙: (a) 5~200cc 크기 (b) 폐 내부 비율 ≥ 30% (c) 위 조건 만족 중 가장 큰 것.
    어떤 component도 조건 만족 못하면 → 5cc 이상 중 가장 폐 내부 비율 높은 것.
    """
    gtv_bool = gtv_raw > 0
    if gtv_bool.sum() == 0:
        return gtv_bool
    lung = extract_lung_mask(hu_raw)
    lab, n = label(gtv_bool, structure=np.ones((3, 3, 3)))
    if n == 0:
        return gtv_bool
    voxel_cc = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0

    cand = []  # (id, vol_cc, in_lung_ratio)
    for i in range(1, n + 1):
        comp = (lab == i)
        vol = comp.sum() * voxel_cc
        in_lung_ratio = (comp & lung).sum() / max(comp.sum(), 1)
        cand.append((i, vol, float(in_lung_ratio)))

    # 우선순위 단계적 탐색 (더 큰 body/폐 합본이 선택되지 않도록 상한 200cc 고수)
    # 1순위: 5~200cc & 폐 내부 ≥ 0.3 중 가장 큰 것
    primary = [c for c in cand if 5 <= c[1] <= 200 and c[2] >= 0.3]
    if primary:
        best = max(primary, key=lambda c: c[1])
    else:
        # 2순위: 5~200cc 중 폐 내부 비율 가장 높은 것 (흉벽/폐외 GTV 포함)
        sec = [c for c in cand if 5 <= c[1] <= 200]
        if sec:
            best = max(sec, key=lambda c: c[1])  # 그 중 가장 큰 것이 GTV일 가능성
        else:
            # 3순위: 2~200cc 범위 내 가장 큰 것 (폐 합본 제외)
            tert = [c for c in cand if 2 <= c[1] <= 200]
            if tert:
                best = max(tert, key=lambda c: c[1])
            else:
                # 4순위: 0.5cc 이상 중 폐 내부 비율 가장 높은 것 (아주 작은 소결절 포함)
                quat = [c for c in cand if 0.5 <= c[1] <= 200 and c[2] >= 0.2]
                if quat:
                    best = max(quat, key=lambda c: c[2])
                else:
                    return gtv_bool
    return lab == best[0]


ROOT = Path(__file__).resolve().parent
EMB = ROOT / "embeddings"
NIFTI_ROOT = Path("/home/team4/LJW/data/processed/ct_nifti")   # fallback
RT_ROOT = ROOT / "rt_data"                                     # DICOM 재변환본 (우선)
OUT_DIR = ROOT / "patients3d"
OUT_DIR.mkdir(exist_ok=True)


def load_and_downsample(pid: str, target: int = 80):
    """CT + DICOM 원본 GTV-1 로드 → (target, target, target) + mm spacing.

    우선: rt_data/<PID>/image.nii.gz + mask_GTV-1.nii.gz (DICOM 재변환본, 깨끗)
    대체: NIfTI_ROOT (heuristic refine)
    """
    rt_ct = RT_ROOT / pid / "image.nii.gz"
    rt_gtv = RT_ROOT / pid / "mask_GTV-1.nii.gz"
    if rt_ct.exists() and rt_gtv.exists():
        ct_img = nib.load(rt_ct)
        ct_hu = ct_img.get_fdata().astype(np.float32)
        gtv_clean = (nib.load(rt_gtv).get_fdata() > 0)
        spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float32)
    else:
        ct_img = nib.load(NIFTI_ROOT / pid / "ct.nii.gz")
        ct_hu = ct_img.get_fdata().astype(np.float32)
        spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float32)
        gt_path = NIFTI_ROOT / pid / "gtv_mask.nii.gz"
        gtv_clean = refine_gtv(ct_hu, nib.load(gt_path).get_fdata(), spacing) if gt_path.exists() else None

    ct_norm = (np.clip(ct_hu, -1000, 400) + 1000) / 1400.0
    zooms = [target / s for s in ct_norm.shape]
    ct_ds = zoom(ct_norm, zooms, order=1, prefilter=False)
    gt_ds = None
    if gtv_clean is not None:
        gt_ds = (zoom(gtv_clean.astype(np.float32), zooms, order=0, prefilter=False) > 0.5).astype(np.uint8)

    ds_spacing = spacing * (np.array(ct_norm.shape, np.float32) / target)
    return ct_ds, gt_ds, tuple(float(x) for x in ds_spacing)


def mesh_from_volume(vol, level, step=1, spacing=(1, 1, 1)):
    """volume → Plotly Mesh3d (verts in mm, with voxel spacing applied)."""
    v, f, _, _ = marching_cubes(vol, level=level, step_size=step, spacing=spacing)
    return v[:, 0], v[:, 1], v[:, 2], f[:, 0], f[:, 1], f[:, 2]


def build_fig(pid: str, target: int = 80) -> go.Figure:
    ct, gt, spacing = load_and_downsample(pid, target)

    fig = go.Figure()

    # Body mesh (semi-transparent skin) — mm 좌표
    try:
        bx, by, bz, bi, bj, bk = mesh_from_volume(ct, level=0.35, step=2, spacing=spacing)
        fig.add_trace(go.Mesh3d(
            x=bx, y=by, z=bz, i=bi, j=bj, k=bk,
            opacity=0.14, color="#c0c0c0",
            flatshading=True, showscale=False, hoverinfo="skip",
            name="body",
        ))
    except Exception:
        pass

    # GTV mesh (tumor, red solid) — same mm coordinate system
    if gt is not None and gt.sum() > 10:
        try:
            gx, gy, gz, gi, gj, gk = mesh_from_volume(gt.astype(np.float32), level=0.5, step=1, spacing=spacing)
            fig.add_trace(go.Mesh3d(
                x=gx, y=gy, z=gz, i=gi, j=gj, k=gk,
                opacity=0.95, color="#ff3b3b",
                flatshading=False, showscale=False, hoverinfo="skip",
                name="GTV",
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(
            text=f"{pid} · 3D Volume (drag=rotate · scroll=zoom · shift-drag=pan)",
            font=dict(color="#7fdbca", size=12), x=0.02, y=0.98,
        ),
        paper_bgcolor="#000", plot_bgcolor="#000", font_color="#e6edf3",
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            bgcolor="#000",
            xaxis=dict(visible=False, showbackground=False, autorange="reversed"),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        showlegend=False,
    )
    return fig


def save_html(pid: str, fig: go.Figure) -> Path:
    out = OUT_DIR / f"{pid}.html"
    fig.write_html(
        out,
        include_plotlyjs="cdn",   # CDN 사용 — 오프라인이면 'inline'
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return out


def list_test_pids():
    z = np.load(EMB / "clinical_test.npz", allow_pickle=True)
    return [str(p) for p in z["pids"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="LUNG1-355 only")
    ap.add_argument("--pid", type=str, help="single PID")
    ap.add_argument("--size", type=int, default=80, help="downsampled cube size")
    args = ap.parse_args()

    if args.pid:
        pids = [args.pid]
    elif args.smoke:
        pids = ["LUNG1-355"]
    else:
        pids = list_test_pids()

    print(f"[3d] generating {len(pids)} viewer(s) at {args.size}³")
    for i, pid in enumerate(pids, 1):
        try:
            fig = build_fig(pid, target=args.size)
            out = save_html(pid, fig)
            print(f"  [{i:>3}/{len(pids)}] {pid}  →  {out.name}  ({out.stat().st_size/1024:.0f} KB)")
        except Exception as e:
            print(f"  [{i:>3}/{len(pids)}] {pid}  FAILED: {e}")


if __name__ == "__main__":
    main()
