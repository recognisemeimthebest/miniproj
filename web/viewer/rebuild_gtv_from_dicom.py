"""DICOM RTSTRUCT에서 'GTV-1'만 정확히 추출 → web/rt_data/<PID>/

각 환자별 생성물:
  mask_GTV-1.nii.gz  (binary GTV 마스크, CT와 동일한 voxel grid)
  image.nii.gz       (DICOM 원본 CT 재조립)
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import numpy as np
import pydicom
if not hasattr(pydicom, "read_file"):
    pydicom.read_file = pydicom.dcmread  # pydicom 3.x compat
from dcmrtstruct2nii import dcmrtstruct2nii

ROOT = Path(__file__).resolve().parent
DICOM_ROOT = Path("/home/team4/miniproj/metadata/nsclc_radiomics")
OUT_ROOT = ROOT / "rt_data"
OUT_ROOT.mkdir(exist_ok=True)


def find_ct_and_rt(pid_dir: Path):
    ct_dir, rt_file = None, None
    for p in pid_dir.rglob("*.dcm"):
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        m = getattr(d, "Modality", "?")
        if m == "CT" and ct_dir is None:
            ct_dir = p.parent
        elif m == "RTSTRUCT":
            rt_file = p
    return ct_dir, rt_file


def find_gtv_name(rt_file):
    """RTSTRUCT에서 'GTV-1' 또는 유사한 이름 선택."""
    d = pydicom.dcmread(rt_file, force=True)
    names = [r.ROIName for r in d.StructureSetROISequence]
    # 우선순위: 'GTV-1' > 'gtv-1' > 'GTV' > 첫 번째 GTV-* > 첫 번째
    for cand in names:
        if cand == "GTV-1": return cand
    for cand in names:
        if cand.lower() == "gtv-1": return cand
    for cand in names:
        if cand.lower() == "gtv": return cand
    for cand in names:
        if cand.lower().startswith("gtv"): return cand
    return None


def convert_one(pid: str) -> str:
    out = OUT_ROOT / pid
    mask_out = out / "mask_GTV-1.nii.gz"
    if mask_out.exists() and mask_out.stat().st_size > 0:
        return "cached"

    pid_dir = DICOM_ROOT / pid
    if not pid_dir.exists():
        return f"no-dicom"
    ct_dir, rt_file = find_ct_and_rt(pid_dir)
    if not ct_dir or not rt_file:
        return f"missing CT({bool(ct_dir)}) or RT({bool(rt_file)})"
    gtv_name = find_gtv_name(rt_file)
    if not gtv_name:
        return "no GTV ROI"

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    try:
        dcmrtstruct2nii(
            rtstruct_file=str(rt_file),
            dicom_file=str(ct_dir),
            output_path=str(out),
            structures=[gtv_name],
            convert_original_dicom=True,
            mask_background_value=0, mask_foreground_value=1,
        )
    except Exception as e:
        return f"FAIL: {type(e).__name__} {e}"

    # normalize output names
    for f in list(out.iterdir()):
        if f.name.startswith("mask_") and f.name != "mask_GTV-1.nii.gz":
            f.rename(out / "mask_GTV-1.nii.gz")
    return f"ok (GTV='{gtv_name}')"


def list_pids():
    z = np.load(ROOT / "embeddings" / "clinical_test.npz", allow_pickle=True)
    return [str(p) for p in z["pids"]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=str)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    pids = [args.pid] if args.pid else (["LUNG1-355"] if args.smoke else list_pids())
    print(f"[rebuild] {len(pids)} patients → {OUT_ROOT}")
    for i, pid in enumerate(pids, 1):
        msg = convert_one(pid)
        print(f"  [{i:>3}/{len(pids)}] {pid}: {msg}")


if __name__ == "__main__":
    main()
