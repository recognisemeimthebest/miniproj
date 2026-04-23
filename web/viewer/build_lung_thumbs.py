"""왼쪽 사이드바용 미니 3D — 반투명 폐(파랑) + 종양(빨강) 실루엣.

Output: web/patients3d_lung/<PID>.html
각도는 정면 약간 기울임으로 고정 + 마우스 회전 가능.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_fill_holes, label, zoom
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parent
EMB = ROOT / "embeddings"
NIFTI_ROOT = Path("/home/team4/LJW/data/processed/ct_nifti")
RT_ROOT = ROOT / "rt_data"
OUT_DIR = ROOT / "patients3d_lung"
OUT_DIR.mkdir(exist_ok=True)


def extract_lung_mask(hu):
    """HU 기반 단순 폐 분할 (공기 ∩ body_filled)."""
    body = hu > -500
    # axial 슬라이스별로 구멍 채우기
    filled = np.zeros_like(body)
    for k in range(body.shape[2]):
        filled[:, :, k] = binary_fill_holes(body[:, :, k])
    lung = (hu < -500) & filled
    lab, n = label(lung, structure=np.ones((3, 3, 3)))
    if n == 0:
        return lung
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    top2 = np.argsort(sizes)[-2:]
    return np.isin(lab, top2)


def refine_gtv(hu, gtv_raw, spacing_mm, lung):
    """RTSTRUCT 합본에서 실제 GTV만 heuristic으로 추출."""
    gtv_bool = gtv_raw > 0
    if gtv_bool.sum() == 0:
        return gtv_bool
    lab, n = label(gtv_bool, structure=np.ones((3, 3, 3)))
    if n == 0:
        return gtv_bool
    voxel_cc = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    cand = []
    for i in range(1, n + 1):
        comp = (lab == i)
        vol = comp.sum() * voxel_cc
        in_lung_ratio = (comp & lung).sum() / max(comp.sum(), 1)
        cand.append((i, vol, float(in_lung_ratio)))
    primary = [c for c in cand if 5 <= c[1] <= 200 and c[2] >= 0.3]
    if primary:
        best = max(primary, key=lambda c: c[1])
    else:
        sec = [c for c in cand if 5 <= c[1] <= 200]
        if sec:
            best = max(sec, key=lambda c: c[1])
        else:
            tert = [c for c in cand if 2 <= c[1] <= 200]
            if tert:
                best = max(tert, key=lambda c: c[1])
            else:
                quat = [c for c in cand if 0.5 <= c[1] <= 200 and c[2] >= 0.2]
                if quat:
                    best = max(quat, key=lambda c: c[2])
                else:
                    return gtv_bool
    return lab == best[0]


def load_masks(pid: str, target: int = 72):
    """우선 DICOM 재변환본 사용."""
    rt_ct = RT_ROOT / pid / "image.nii.gz"
    rt_gtv = RT_ROOT / pid / "mask_GTV-1.nii.gz"
    if rt_ct.exists() and rt_gtv.exists():
        img = nib.load(rt_ct)
        ct = img.get_fdata().astype(np.float32)
        spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
        gtv_clean = (nib.load(rt_gtv).get_fdata() > 0)
    else:
        img = nib.load(NIFTI_ROOT / pid / "ct.nii.gz")
        ct = img.get_fdata().astype(np.float32)
        spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
        gtv_path = NIFTI_ROOT / pid / "gtv_mask.nii.gz"
        gtv_raw = nib.load(gtv_path).get_fdata() if gtv_path.exists() else None
        lung_pre = extract_lung_mask(ct)
        gtv_clean = refine_gtv(ct, gtv_raw, spacing, lung_pre) if gtv_raw is not None else None
    lung_raw = extract_lung_mask(ct)

    zooms_f = [target / s for s in ct.shape]
    lung_ds = (zoom(lung_raw.astype(np.float32), zooms_f, order=1, prefilter=False) > 0.5)
    gtv_ds = None
    if gtv_clean is not None:
        gtv_ds = (zoom(gtv_clean.astype(np.float32), zooms_f, order=0, prefilter=False) > 0.5)

    ds_spacing = spacing * (np.array(ct.shape, np.float32) / target)
    return lung_ds, gtv_ds, tuple(float(x) for x in ds_spacing)


def mesh_from(vol, level=0.5, step=2, spacing=(1, 1, 1)):
    v, f, _, _ = marching_cubes(vol.astype(np.float32), level=level,
                                step_size=step, spacing=spacing)
    return v[:, 0], v[:, 1], v[:, 2], f[:, 0], f[:, 1], f[:, 2]


def build_fig(pid, target=72):
    lung, gtv, spacing = load_masks(pid, target)
    fig = go.Figure()

    if lung.sum() > 100:
        x, y, z, i, j, k = mesh_from(lung, 0.5, 2, spacing)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=0.22, color="#6ab7ff",
            flatshading=True, showscale=False, hoverinfo="skip",
            name="lung",
        ))

    if gtv is not None and gtv.sum() > 8:
        x, y, z, i, j, k = mesh_from(gtv, 0.5, 1, spacing)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=0.95, color="#ff3b3b",
            flatshading=False, showscale=False, hoverinfo="skip",
            name="GTV",
        ))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(visible=False, showbackground=False, autorange="reversed"),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=-1.35, z=0.55),
                        up=dict(x=0, y=0, z=1)),
        ),
        showlegend=False,
    )
    return fig


def save_html(pid, fig):
    out = OUT_DIR / f"{pid}.html"
    fig.write_html(
        out, include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "responsive": True, "displayModeBar": False},
    )
    return out


def list_pids():
    z = np.load(EMB / "clinical_test.npz", allow_pickle=True)
    return [str(p) for p in z["pids"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pid", type=str)
    ap.add_argument("--size", type=int, default=72)
    args = ap.parse_args()

    pids = [args.pid] if args.pid else (["LUNG1-355"] if args.smoke else list_pids())
    print(f"[lung-mini] generating {len(pids)} thumbs at {args.size}³")
    for i, pid in enumerate(pids, 1):
        try:
            fig = build_fig(pid, args.size)
            out = save_html(pid, fig)
            print(f"  [{i:>3}/{len(pids)}] {pid}  →  {out.name}  ({out.stat().st_size/1024:.0f} KB)")
        except Exception as e:
            print(f"  [{i:>3}/{len(pids)}] {pid}  FAILED: {e}")


if __name__ == "__main__":
    main()
