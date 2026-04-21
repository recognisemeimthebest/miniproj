# 라디오믹스 특징 추출 및 전처리 - LJY

## 개요
LUNG1 데이터셋을 이용한 라디오믹스 특징 추출 및 MLP Fusion 전처리

## 환경
- Python 3.9
- pyradiomics
- scikit-learn
- torch

## 데이터
- 환자 수: 421명 (LUNG1-128 마스크 없음으로 제외)
- 입력: CT 영상 (`ct.nii.gz`) + 종양 마스크 (`gtv_mask.nii.gz`)
- 목표 변수: 2년 이내 생존여부 (`label_2yr`)

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
- 모델: GBM (Gradient Boosting)
- Test AUROC: 0.6094 ✅ (기준 > 0.55)
- Val-Test 차이: 0.0065 ✅ (기준 < 0.15)
- Recall: 0.4118 ✅ (기준 > 0.1)

### 4. MLP Dim 변환 (`dim_radiomics.py`)
- 15개 특징 → MLP → 128dim 임베딩
- 결과: `features/radiomics_embed_128.npy`
- Fusion 총 차원: CT(256) + 라디오믹스(128) + 임상(128) = 512dim

## 파일 구조
```
radiomics_project/
├── params.yaml                        # 추출 설정
├── extract.py                         # 특징 추출
├── filter_features.py                 # 상관 필터
├── lasso.py                           # LASSO 변수 선택
├── evaluate.py                        # 성능 평가
├── dim_radiomics.py                   # 128dim 변환
└── features/
    ├── radiomics_features.csv         # 851개 특징
    ├── features_corr_filtered.csv     # 269개 특징
    ├── final_features.csv             # 15개 최종 특징
    ├── radiomics_embed_128.npy        # Fusion 투입 파일
    └── radiomics_embed_128.csv        # 확인용
```

## GitHub 브랜치 파일 구조 (feature/LJY)
```
feature/LJY/
├── README_LJY.md              # 전체 파트 설명
├── params.yaml                # 추출 설정
├── extract.py                 # 특징 추출
├── filter_features.py         # 상관 필터
├── lasso.py                   # LASSO 변수 선택
├── evaluate.py                # 성능 평가
├── dim_radiomics.py           # 128dim 변환
└── features/
    ├── README.md              # features 폴더 설명
    ├── radiomics_features.csv
    ├── features_corr_filtered.csv
    ├── final_features.csv
    ├── radiomics_embed_128.npy
    └── radiomics_embed_128.csv
```