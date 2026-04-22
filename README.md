# LUNA-XAI Mini — NSCLC 2년 생존 예측 멀티모달 XAI

> **NSCLC CT + Radiomics + Clinical 변수를 결합한 멀티모달 딥러닝 생존 예측 + XAI 파이프라인**
> 10일 내부 스터디 · 팀 4명 · end-to-end (데이터 → 학습 → XAI → Streamlit 데모)

본 레포는 **실제 학습이 완료된 현 시점 상태** (단일 모달 3종 + triple linear L2 fusion) 를
기준으로 정리되어 있다. 상세 기획은 [`NSCLC_Multimodal_XAI_Project_Plan.md`](NSCLC_Multimodal_XAI_Project_Plan.md) (v4.0, 2026-04-22) 참조.

---

## 프로젝트 개요 (현재 상태 = v4.0)

| 항목 | 내용 |
|---|---|
| 데이터 | NSCLC-Radiomics (TCIA Lung1), censoring 제외 후 **n = 420** |
| 타겟 | 2년 생존 이진분류 (`survival ≥ 730d AND not dead` vs `< 730d AND dead`) |
| Split | **70 / 15 / 15 label-stratified = 293 / 64 / 63**, seed = 99 (전 모델 공통) |
| CT 인코더 | **Hosny 2018 shallow 3D CNN** (4 block, ~550k params, 80³ ROI, from-scratch) |
| Radiomics | PyRadiomics → LASSO 15개 피처 → MLP (Optuna) → **256-dim 임베딩** |
| Clinical | 11 피처 → MLP (Optuna) → **128-dim 임베딩** |
| **Fusion (M4)** | **Triple Linear L2 LogReg** on 640-dim concat (C=1e-3, sklearn `LogisticRegressionCV`) |
| 헤드라인 성능 | **M4 Test AUROC = 0.6589** (trainval 재학습, n=63) |
| XAI | Grad-CAM (CT), SHAP Top-10 (LinearExplainer), Modality Ablation |
| 서비스 | Streamlit 로컬 데모 (발표 당일 라이브 시연) |

**단계별 최고점 (동일 test 분할, n=63):**

| 단계 | 최고 모델 | Test AUROC | Δ vs previous |
|---|---|---|---|
| Single (M1/M2/M3) | CT Hosny CNN (M2, MLP head) | 0.6232 | — |
| Double | Clin+Rad intermediate | 0.6284 | +0.0052 |
| **Triple (M4)** | **Linear L2 on 640-concat** | **0.6589** | **+0.0305** 🏆 |

> Target AUROC 0.75 는 현 모델로 달성되지 않음. 기획서 v4.0 은 이를 투명하게 반영하고 **XAI 분석·서비스 데모를 중심 산출물**로 재조정.

---

## 폴더 구조 (실제 repo 상태)

```
.
├── NSCLC_Multimodal_XAI_Project_Plan.md   v4.0 기획서 (단일 진실 원천)
├── CONTRIBUTING.md                        팀 기여 가이드
├── README.md                              본 파일
│
├── single_modal_baseline/                 M1 / M2 / M3 단독 baseline
│   ├── scripts/
│   │   ├── clinical/                      Clinical MLP + Optuna
│   │   ├── radiomics/                     Radiomics MLP + Optuna
│   │   ├── ct/                            Hosny 3D CNN + Optuna MLP head
│   │   └── common_split.py                공용 split (293/64/63, seed=99)
│   └── results/{clinical,radiomics,ct}/   best.pt + features/{train,val,test}.npz
│
├── double_model/                          2-모달 fusion (3 pair × {late, intermediate})
│   ├── scripts/
│   └── results/{clin_rad,clin_ct,rad_ct}/
│
├── triple_model/                          3-모달 fusion = M4
│   ├── scripts/
│   │   ├── late_fusion.py                 equal-weight arith/logit
│   │   ├── intermediate_fusion.py         v1: Optuna MLP head (val-max)
│   │   ├── weighted_late_fusion.py        simplex grid on val
│   │   ├── intermediate_fusion_v2.py      v2: 5-fold CV Optuna + BN + modality dropout
│   │   ├── linear_fusion.py               ★ LogRegCV(L2) + ElasticNet on 640-concat
│   │   └── visualize.py                   통합 bar + ROC + CM
│   ├── embeddings/                        frozen encoder 3종 임베딩 캐시
│   ├── results/triple/
│   │   ├── {late,weighted_late,intermediate,intermediate_v2,linear}/
│   │   └── summary.json
│   ├── figures/                           comparison_bar, fusion_tuning_bar, roc_curves, CM
│   └── triple_linear_l2_best.pt           ★ M4 PyTorch drop-in 체크포인트 (12K)
│
├── preprocessing/                         clinical preprocessing & ML baseline
├── experiments/                           Optuna 아티팩트 + 재현용 예측 CSV
└── app/                                   Streamlit 데모 (Day 9 구현 예정)
```

`data/`, `checkpoints/`, 대용량 NIfTI/DICOM 은 `.gitignore`. TCIA 라이선스 때문에 원본
데이터는 팀 워크스테이션 로컬에만 보관.

---

## M4 체크포인트 (`triple_model/triple_linear_l2_best.pt`)

Drop-in PyTorch 포팅 — 학습은 sklearn `LogisticRegressionCV` 로 fit, 추론·XAI 는 동등한
`TripleLinearL2` 모듈로 로드.

```python
import torch
ckpt = torch.load("triple_model/triple_linear_l2_best.pt", map_location="cpu", weights_only=False)
# ckpt['arch'] == 'TripleLinearL2'
# ckpt['in_dim'] == 640  (clin 128 + radiomics 256 + ct 256)
# ckpt['modality_slices'] == {'clinical': (0,128), 'radiomics': (128,384), 'ct': (384,640)}
# ckpt['chosen_C'] == 1e-3
# ckpt['metrics'] == {'test_auroc_sklearn': 0.6589, 'test_auroc_torch': 0.6589, ...}
# ckpt['state_dict'] 에 mean(640,), scale(640,), linear.weight(1,640), linear.bias(1,)
```

`test_auroc_sklearn == test_auroc_torch == 0.6589` 로 드롭인 포팅이 sklearn 과 완전 일치함.

---

## 빠른 시작

### 1. 환경

```bash
conda env create -f environment.yml   # Python 3.10 / PyTorch 2.x / CUDA 12 / MONAI
conda activate luna-xai
```

### 2. 데이터 준비

- TCIA `NSCLC-Radiomics` 다운로드 (DOI: 10.7937/K9/TCIA.2015.PF0M9REI, CC BY-NC 3.0)
- DICOM → NIfTI 변환, GTV-1 mask 추출, Radiomics 추출 (이미 완료된 상태 가정)
- Split CSV: `single_modal_baseline/scripts/common_split.py` 가 seed=99 로 고정

### 3. 재현 (단계별)

```bash
# 단일 모달 baseline (M1 / M2 / M3)
python single_modal_baseline/scripts/clinical/train_clinical.py
python single_modal_baseline/scripts/radiomics/train_radiomics.py
python single_modal_baseline/scripts/ct/train_ct.py            # + optuna_mlp_head.py

# 2-모달 fusion (상보성 측정용, double_model/)
python double_model/scripts/late_fusion.py
python double_model/scripts/intermediate_fusion.py --n-trials 30

# 3-모달 fusion = M4 (triple_model/)
python triple_model/scripts/late_fusion.py                     # baseline
python triple_model/scripts/intermediate_fusion.py --n-trials 30
python triple_model/scripts/weighted_late_fusion.py
python triple_model/scripts/intermediate_fusion_v2.py --n-trials 40 --n-splits 5
python triple_model/scripts/linear_fusion.py                   # ★ M4 winner
python triple_model/scripts/visualize.py                       # 모든 figure 재생성
```

### 4. Streamlit 데모 (Day 9, 개발 예정)

```bash
streamlit run app/main.py    # localhost:8501 에서 4-tab UI (예측 / Grad-CAM / SHAP / Ablation)
```

---

## 브랜치 전략

```
main            보호, PR + 리뷰 1명 필수
├── dev
└── feature/LJW   ← 현재 모든 baseline + M4 가 이 브랜치에 있음
    feature/CSJ
    feature/LJY
```

- **`feature/LJW`** = 단일 모달 baseline 3종 + double/triple fusion + `triple_linear_l2_best.pt` 포함
- `main` 머지는 Day 10 전 통합 PR 로 진행

---

## 팀 역할 (v4.0 §4)

| 역할 | 담당자 | 주 책임 |
|---|---|---|
| PM / 통합 리드 | A | 일정·회의·split 관리, 최종 통합, Streamlit 앱 통합 |
| 영상 파이프라인 리드 | B | GTV crop, Hosny CNN(M2), Grad-CAM, IoU/Pointing game, 앱 Grad-CAM 탭 |
| Radiomics & Tabular 리드 | C | LASSO/MLP(M1/M3), SHAP Top-10 (LinearExplainer), 임상 전처리, 앱 입력·SHAP 탭 |
| XAI & 평가 리드 | D | 평가 지표, DeLong, KM log-rank, Modality Ablation, 앱 Ablation 탭 |

**GPU:** RTX 4070 Ti Super (16GB VRAM) 팀 공용 1대 — 시간 분할 공유.

---

## 일정 (10일 — 현재 Day 7 진입)

| Phase | Day | 마일스톤 | 상태 |
|---|---|---|---|
| Setup & Single | 1 | Kick-off, 환경, EDA, 통일 split | ✅ |
| | 2 | Radiomics LASSO + M3 MLP | ✅ |
| | 3 | CT 전처리 + Hosny CNN(M2) | ✅ |
| | 4 | M1 Clinical MLP, M2 튜닝 | ✅ |
| | 5 | 중간 체크포인트 #1 — M1/M2/M3 공유 | ✅ |
| Multimodal & XAI | 6 | **M4 triple linear L2 fit → `triple_linear_l2_best.pt`** | ✅ |
| | 7 | Test 평가 (DeLong, Brier, KM log-rank) | 진행 |
| | 8 | **XAI 3종:** Grad-CAM + IoU/Pointing + SHAP Top-10 + Ablation | 예정 |
| Service & 발표 | 9 | **Streamlit 앱 개발** (추론 + 4-tab XAI) | 예정 |
| | 10 | 앱 검증 + 발표자료 + 리허설 + **최종 발표** | 예정 |

---

## 데이터 인용 (필수)

> Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., et al. (2014). **Data From NSCLC-Radiomics (version 4)** [Data set]. The Cancer Imaging Archive.
> https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
> License: CC BY-NC 3.0

### 핵심 참고 논문

- Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach.* **Nat Commun** 2014;5:4006.
- Hosny A, et al. *Deep learning for lung cancer prognostication.* **PLOS Medicine** 2018;15(11):e1002711. (M2 CT CNN 구조)
- Simon BD, et al. *A multimodal automated deep learning-based model for predicting biochemical recurrence of prostate cancer.* **Clin Imaging** 2025;126:110579. (M1~M4 구조 패턴)
- Selvaraju RR, et al. *Grad-CAM.* **ICCV** 2017.
- Lundberg SM, Lee SI. *A unified approach to interpreting model predictions (SHAP).* **NeurIPS** 2017.

---

*Python 3.10 / PyTorch 2.x / CUDA 12 / MONAI ≥ 1.3 / scikit-learn / SHAP ≥ 0.44 / pytorch-grad-cam (3D) / Streamlit ≥ 1.30*
