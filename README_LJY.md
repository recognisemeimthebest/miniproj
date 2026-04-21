# 라디오믹스 특징 추출 및 전처리 - LJY

## 개요
LUNG1 데이터셋을 이용한 라디오믹스 특징 추출 및 MLP Fusion 전처리

## 환경
- Python 3.9
- pyradiomics
- scikit-learn
- torch
- optuna

## 데이터
- 환자 수: 421명 (LUNG1-128 마스크 없음으로 제외)
- 입력: CT 영상 (`ct.nii.gz`) + 종양 마스크 (`gtv_mask.nii.gz`)
- 목표 변수: 2년 이내 생존여부 (`label_2yr`)
- 생존(1): 169명 / 사망(0): 251명

## 처리 과정

### 1. 특징 추출 (`extract.py`)
- pyradiomics로 851개 특징 추출
- 강도 / 모양 / 질감 / 웨이블릿 포함
- 설정 파일: `params.yaml`

### 2. 특징 선택 (`filter_features.py` → `lasso.py`)
- 상관 필터 (>0.90 제거): 851개 → 269개
- LASSO (CV=5): 269개 → 15개
- 결과: `features/final_features.csv`

### 3. 성능 평가 (`evaluate.py`)
- 256dim 임베딩 기준 모델별 성능

| 모델 | Val AUROC | Test AUROC | Val-Test 차이 | Recall | 통과 여부 |
|---|---|---|---|---|---|
| LR (balanced) | 0.5747 | 0.5465 | 0.0282 | 0.5294 | ⚠️ 미달 |
| RF (balanced) | 0.5706 | 0.5382 | 0.0324 | 0.4118 | ⚠️ 미달 |
| **GBM** | **0.5259** | **0.6112** | **0.0853** | **0.4412** | **✅ 통과** |
| LR (C=1) | 0.5159 | 0.5329 | 0.0171 | 0.5000 | ⚠️ 미달 |
| RF (depth=3) | 0.5676 | 0.5318 | 0.0359 | 0.4118 | ⚠️ 미달 |

- 최종 선택 모델: **GBM** (Test AUROC 0.6112)
- 평가 기준: Test AUROC > 0.55 / Val-Test < 0.15 / Recall > 0.1

### 4. Optuna 최적 dim 탐색 (`optuna_dim.py`)
- 100회 탐색으로 최적 하이퍼파라미터 결정

| 항목 | 최적값 |
|---|---|
| embed_dim | 256 |
| hidden_dim | 128 |
| dropout | 0.1409 |
| lr | 0.003147 |
| epochs | 84 |
| Val AUROC | 0.6400 |

### 5. MLP 256dim 변환 (`dim_radiomics.py`)
- 15개 특징 → MLP → 256dim 임베딩
- 결과: `features/radiomics_embed_256.npy`
- Fusion 총 차원: CT(256) + 라디오믹스(256) + 임상(128) = 640dim

### 6. 모델 저장 (`save_model.py`)
- 학습된 MLP 모델을 `.pt` 파일로 저장
- 결과: `radiomics_mlp_256dim.pt`

## 최종 선택된 15개 특징 (LASSO)
- original_shape_LeastAxisLength
- original_firstorder_10Percentile
- original_firstorder_Minimum
- original_gldm_DependenceVariance
- wavelet-LLH_glszm_GrayLevelNonUniformityNormalized
- wavelet-LLH_gldm_DependenceVariance
- wavelet-LHL_glszm_GrayLevelNonUniformity
- wavelet-LHL_glszm_GrayLevelNonUniformityNormalized
- wavelet-LHH_glszm_LargeAreaHighGrayLevelEmphasis
- wavelet-LHH_ngtdm_Busyness
- wavelet-HLL_firstorder_Skewness
- wavelet-HLH_firstorder_Median
- wavelet-HLH_firstorder_Skewness
- wavelet-LLL_glszm_LargeAreaHighGrayLevelEmphasis
- wavelet-LLL_gldm_LargeDependenceLowGrayLevelEmphasis

## GitHub 브랜치 파일 구조 (feature/LJY)
```
feature/LJY/
├── README_LJY.md              # 전체 파트 설명
├── params.yaml                # 추출 설정
├── extract.py                 # 특징 추출
├── filter_features.py         # 상관 필터
├── lasso.py                   # LASSO 변수 선택
├── evaluate.py                # 성능 평가
├── dim_radiomics.py           # 256dim 변환
├── optuna_dim.py              # Optuna 최적 dim 탐색
├── save_model.py              # 모델 저장
├── radiomics_mlp_256dim.pt    # 학습된 MLP 모델
└── features/
    ├── README.md              # features 폴더 설명
    ├── radiomics_features.csv         # 851개 원본 특징 (421명 x 851개)
    ├── features_corr_filtered.csv     # 상관 필터 후 특징 (421명 x 269개)
    ├── final_features.csv             # LASSO 최종 특징 (421명 x 15개)
    ├── radiomics_embed_256.npy        # 256dim 임베딩 - Fusion 투입 파일
    └── radiomics_embed_256.csv        # 256dim 임베딩 확인용
```