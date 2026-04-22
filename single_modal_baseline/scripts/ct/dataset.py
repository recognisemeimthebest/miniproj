"""NSCLC-Radiomics Lung1 Dataset (CT + GTV mask + 2년 생존 레이블).

NIfTI(`ct.nii.gz`, `gtv_mask.nii.gz`) + 임상 CSV를 조인해서
PyTorch Dataset + train/val/test split을 제공.

기획서 §2.3 레이블 정의:
    Survival.time >= 730 & dead == 0  → Class 1 (2년 이상 생존 확정)
    Survival.time >= 730 & dead == 1  → Class 1 (2년 이후 사망 = 2년 시점엔 생존)
    Survival.time <  730 & dead == 1  → Class 0 (2년 이내 사망 확정)
    Survival.time <  730 & dead == 0  → 제외 (조기 censored, outcome unknown)

사용 예:
    from src.data import build_preprocess
    from src.data.dataset import load_manifest, split_manifest, NSCLCDataset

    manifest = load_manifest()                         # 420명
    splits = split_manifest(manifest, seed=42)         # train/val/test dict
    ds = NSCLCDataset(splits["train"],
                      transform=build_preprocess(training=True))
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# NIfTI 파일이 모여있는 루트 디렉토리.
# 이 밑에 환자 ID 별 서브디렉토리 {PatientID}/{ct.nii.gz, gtv_mask.nii.gz} 구조.
# scripts/dicom_to_nifti.py가 원본 DICOM → NIfTI 변환하면서 이 레이아웃으로 저장함.
NIFTI_ROOT_DEFAULT = Path("data/processed/ct_nifti")

# TCIA NSCLC-Radiomics Lung1 공식 임상 CSV (버전 3, 2019년 10월자).
# 컬럼: PatientID, Survival.time (일수), deadstatus.event (0=censored, 1=death),
#       Overall.Stage, age, gender, clinical.T.Stage, clinical.N.Stage, ...
LABEL_CSV_DEFAULT = Path("metadata/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")

# 2년 = 730일. M1 태스크 endpoint의 컷오프.
TWO_YEARS_DAYS = 730


def load_manifest(
    nifti_root: Path | str = NIFTI_ROOT_DEFAULT,
    label_csv: Path | str = LABEL_CSV_DEFAULT,
) -> pd.DataFrame:
    """임상 CSV + 디스크 NIfTI 교집합으로 manifest 생성.

    - `Survival.time < 730 & dead == 0` (조기 censored) 환자는 제외
    - NIfTI 파일이 디스크에 없는 환자도 제외

    Returns:
        pd.DataFrame with columns:
            patient_id, image_path, mask_path, label (0/1),
            stage, survival_time, dead
    """
    nifti_root = Path(nifti_root)
    df = pd.read_csv(label_csv)  # 원본 임상 CSV (422명)

    # ──────────────────────────────────────────────────────────────────
    # [필터 1] 조기 censored 환자 제외
    # ──────────────────────────────────────────────────────────────────
    # "Survival.time<730 & dead=0" 조합은 "2년 안 됐는데 추적이 끊긴 환자".
    # 2년 시점 생사를 모르기 때문에 레이블이 unknown → 학습 데이터에서 제외.
    # (Survival.time<730 & dead=1 은 유지: 2년 안에 사망 = Class 0으로 확정)
    keep = ~((df["Survival.time"] < TWO_YEARS_DAYS) & (df["deadstatus.event"] == 0))
    df = df.loc[keep].copy()

    # ──────────────────────────────────────────────────────────────────
    # [레이블 부여] 2년 생존 여부 → 0/1 이진 분류
    # ──────────────────────────────────────────────────────────────────
    # Survival.time >= 730  → Class 1 (2년 시점 생존)
    # Survival.time <  730  → Class 0 (2년 시점 사망)  ※ 위 필터 통과한 건 모두 dead=1
    df["label"] = (df["Survival.time"] >= TWO_YEARS_DAYS).astype(int)

    # ──────────────────────────────────────────────────────────────────
    # [경로 매핑] 환자 ID → 디스크상 NIfTI 파일 경로
    # ──────────────────────────────────────────────────────────────────
    # 예: "LUNG1-001" → data/processed/ct_nifti/LUNG1-001/ct.nii.gz
    #                   data/processed/ct_nifti/LUNG1-001/gtv_mask.nii.gz
    df["image_path"] = df["PatientID"].apply(lambda pid: nifti_root / pid / "ct.nii.gz")
    df["mask_path"] = df["PatientID"].apply(lambda pid: nifti_root / pid / "gtv_mask.nii.gz")

    # ──────────────────────────────────────────────────────────────────
    # [필터 2] 실제로 NIfTI가 디스크에 존재하는 환자만 유지
    # ──────────────────────────────────────────────────────────────────
    # DICOM→NIfTI 변환 실패, RT-Struct 누락, 원본 데이터 손상 등으로
    # 일부 환자는 변환본이 없을 수 있음 → 그런 환자 제외.
    # 필터 1 + 필터 2 통과 → 최종 420명.
    have_both = df["image_path"].apply(Path.exists) & df["mask_path"].apply(Path.exists)
    df = df.loc[have_both]

    # 필요한 컬럼만 뽑고 이름도 파이썬 스타일(snake_case)로 통일.
    out = df.rename(
        columns={
            "PatientID": "patient_id",
            "Survival.time": "survival_time",
            "deadstatus.event": "dead",
            "Overall.Stage": "stage",
        }
    )[["patient_id", "image_path", "mask_path", "label", "stage", "survival_time", "dead"]]
    return out.reset_index(drop=True)


def split_manifest(
    manifest: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify: str = "label",
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split.

    기본 stratify는 label (2 classes) — n=420에서 label×stage (최대 8 strata)는
    일부 cell의 샘플 수가 너무 적어 split이 깨질 수 있어 label only가 안전.

    Args:
        stratify: "label" (기본) 또는 "label_stage" (과잉 분할, 주의).

    Returns:
        {"train": df, "val": df, "test": df}, 각 df는 index reset됨.
    """
    # ──────────────────────────────────────────────────────────────────
    # [stratify 기준] split 하면서 각 set의 class 비율을 원본과 동일하게 유지
    # ──────────────────────────────────────────────────────────────────
    # 기본은 label(2-class)만 — 이게 가장 안정적.
    # label_stage는 2×4=8 strata로 쪼개지는데, n=420에선 일부 cell이 2-3명뿐이라
    # train_test_split이 "너무 적어 stratify 불가" 에러를 낼 위험이 있음.
    if stratify == "label_stage":
        strata = manifest["label"].astype(str) + "_" + manifest["stage"].astype(str)
    else:
        strata = manifest["label"]

    # ──────────────────────────────────────────────────────────────────
    # 2-stage split (sklearn에 3-way split 직접 지원 없음)
    # ──────────────────────────────────────────────────────────────────
    # 1단계: 전체에서 test 15% 분리 (나머지 85%는 train+val)
    rest_idx, test_idx = train_test_split(
        manifest.index, test_size=test_frac, stratify=strata, random_state=seed
    )
    # 2단계: 나머지 85%에서 val 분리.
    #   전체 대비 15%를 원한다면, 나머지(85%) 대비 비율은 15/85 ≈ 0.176
    rest_strata = strata.loc[rest_idx]
    val_size = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_size, stratify=rest_strata, random_state=seed
    )
    # 결과 예 (seed=42, n=420): train=293 / val=64 / test=63
    #   train: class1=118 (40%), class0=175 (60%)
    #   val:   class1=26  (41%), class0=38  (59%)
    #   test:  class1=25  (40%), class0=38  (60%)   ← 비율이 보존됨
    return {
        "train": manifest.loc[train_idx].reset_index(drop=True),
        "val": manifest.loc[val_idx].reset_index(drop=True),
        "test": manifest.loc[test_idx].reset_index(drop=True),
    }


class NSCLCDataset(Dataset):
    """CT + GTV mask + 2년 생존 레이블 PyTorch Dataset.

    __getitem__ 리턴:
        {
            "image": Tensor(1, H, W, D) float32 ∈ [0, 1],
            "mask":  Tensor(1, H, W, D) float32 ∈ {0, 1},
            "label": Tensor() int64 (0 or 1),
            "patient_id": str,
        }

    transform은 MONAI Compose (build_preprocess 리턴값) 전달.
    """

    def __init__(self, manifest: pd.DataFrame, transform: Optional[Callable] = None):
        # manifest: split_manifest()가 돌려준 train/val/test dict 중 하나.
        # transform: build_preprocess()가 돌려준 MONAI Compose 파이프라인.
        self.rows = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        # MONAI 컨벤션: transform은 dict를 받아 dict를 리턴.
        # 여기선 경로 string만 담아 넘기고, LoadImaged가 안에서 실제 파일을 읽음.
        sample = {"image": str(row["image_path"]), "mask": str(row["mask_path"])}
        if self.transform is not None:
            sample = self.transform(sample)  # 파일 로드 + 전처리 7단계 + (train이면) augmentation
        # label과 patient_id는 파이프라인 안 거치고 manifest에서 직접 꺼내옴.
        # (transform 안에서 처리해도 되지만 여기서 하는 게 명시적이라 디버깅 편함)
        return {
            "image": sample["image"],           # Tensor(1, 128, 128, 64) float32 ∈ [0,1]
            "mask": sample["mask"],             # Tensor(1, 128, 128, 64) float32 ∈ {0,1}
            "label": torch.tensor(int(row["label"]), dtype=torch.long),  # 0 or 1
            "patient_id": row["patient_id"],    # str (e.g. "LUNG1-001") — 디버깅/추적용
        }


__all__ = [
    "NSCLCDataset",
    "load_manifest",
    "split_manifest",
    "NIFTI_ROOT_DEFAULT",
    "LABEL_CSV_DEFAULT",
    "TWO_YEARS_DAYS",
]
