# NSCLC CT 기반 멀티모달 2년 생존 예측 및 설명가능 AI 프로젝트 기획서

**프로젝트명(가제):** *LUNA-XAI (LUng NSCLC Analysis with eXplainable AI)*
**기간:** 10일 (Day 1 ~ Day 10)
**인원:** 4명
**목적:** 내부 스터디 — 멀티모달 딥러닝 + 생존 예측 + XAI 파이프라인을 팀 전체가 end-to-end로 경험하고 최종 발표자료로 정리

---

## 1. 프로젝트 개요

### 1.1 배경 및 동기

비소세포폐암(NSCLC, Non-Small Cell Lung Cancer)은 전체 폐암의 약 85%를 차지하며, 1차 치료 후 2년 시점의 생존 여부는 예후 판정과 치료 계획에서 핵심 지표로 쓰인다. 단일 영상 소견이나 단일 임상 변수만으로 예측하는 기존 접근은 한계가 명확하며, 최근에는 **CT + 정량적 영상 특징(radiomics) + 임상 변수**를 결합한 멀티모달 딥러닝 접근이 성능과 임상 해석력 측면에서 우위를 보이고 있다. 동시에 **"왜 그렇게 예측했는지"**를 설명할 수 있는 XAI는 임상 수용성 확보의 전제 조건이다.

본 스터디는 공개 데이터셋(NSCLC-Radiomics, TCIA)을 이용해 NIH-MIP의 `Multimodal_RPModel` 아키텍처 패턴을 **NSCLC 도메인에 맞게 이식**하고, 여기에 `cgiova/multimodal-xai-prostate`의 설명 방식을 **데이터 특성에 맞게 재구현**하는 것을 목표로 한다.

### 1.2 프로젝트 질문

> **"NSCLC 환자의 pre-treatment CT 영상 + PyRadiomics 피처 + 임상 변수를 통합한 멀티모달 모델이, 단일 모달리티 대비 2년 생존 예측에서 얼마나 개선되며, 그 예측 근거는 임상적으로 타당한가?"**

### 1.3 목표 (SMART)

| 구분 | 목표 | 현재 상태 |
|---|---|---|
| 주요 목표 | 2년 생존 예측 이진분류에서 멀티모달 모델(M4) 학습·평가 완료 | **완료 (Test AUROC 0.659)** |
| 부수 목표 | 단일 모달리티(M1/M2/M3) 대비 멀티모달의 이득 정량화 (AUROC 비교 + DeLong test) | M1~M3: 각 0.55~0.62, M4: 0.659 → DeLong 수행 예정 |
| 해석 목표 | Grad-CAM 히트맵이 GTV-1 영역과 중첩되는지 정성·정량 평가, SHAP 상위 10개 피처가 문헌상 예후인자와 일치하는지 검증 | XAI 단계(Day 8) 수행 예정 |
| 학습 목표 | 팀원 4명 전원이 모달리티별 학습 → 멀티모달 fusion → XAI 적용의 전 과정을 설명·재현 가능 | 진행 중 |

> **현실 조정(v4.0):** 단일 모달리티 baseline들과 triple linear L2 fusion이 모두 학습 완료된 시점에서 M4 Test AUROC가 **0.659**로 측정되어, 기획서 초안의 Target(AUROC ≥ 0.75)에는 도달하지 못했다. 본 버전은 기 학습된 모델을 전제로 **XAI 분석과 서비스 데모를 중심 산출물**로 재조정한다.

---

## 2. 데이터

### 2.1 데이터셋: NSCLC-Radiomics (TCIA Lung1)

- **출처:** The Cancer Imaging Archive, DOI: 10.7937/K9/TCIA.2015.PF0M9REI
- **라이선스:** CC BY-NC 3.0 (비상업적 연구 목적 허용, 인용 필수)
- **원본 규모:** 422명 NSCLC 환자, 사전 치료(pre-treatment) CT + 수동 delineation + 임상 outcome
- **본 프로젝트 사용 n:** **420명** (GTV mask 이상·불완전 volume 등 제외 후)
- **데이터 준비 상태:** 다운로드·DICOM→NIfTI 변환·GTV mask 추출·Radiomics 피처 추출 **완료** (프로젝트 시작 시점 기준)

### 2.2 데이터 구성

| 데이터 유형 | 포맷 | 설명 |
|---|---|---|
| CT 영상 | DICOM → NIfTI (전처리 완료) | 축방향 slice **512×512**, 환자별 slice 수 상이 |
| 종양 segmentation | SEG/RTSTRUCT → binary mask | GTV-1 mask (primary lung lesion) |
| 임상 데이터 | CSV (Lung1.clinical) | age, clinical.T/N/M.Stage, Overall.Stage, Histology, gender, Survival.time, deadstatus.event |
| Radiomics 피처 | CSV (추출 완료) | IBSI-compliant hand-crafted features |

### 2.3 타겟 정의

- **2년 생존 이진분류:**
  - `survival.time ≥ 730 days AND deadstatus.event == 0` → **Class 1 (Survivor)**
  - `survival.time < 730 days AND deadstatus.event == 1` → **Class 0 (Non-survivor)**
  - `survival.time < 730 days AND deadstatus.event == 0` (right-censored) → **제외**
- 실제 분석 n은 censoring 제외 후 **420명** (단일 모달 베이스라인 통합 split 기준)

### 2.4 데이터 분할 전략 (단일 모달 베이스라인 통합 split 기준)

- **Train : Validation : Test = 70 : 15 : 15 (label-stratified)**
  - 실제 분할: **293 / 64 / 63**
- 분할 근거: 초기에 세 모달리티가 각자 다른 split을 쓰던 문제를 해소하기 위해 `single_modal_baseline/scripts/common_split.py`로 **통일 split** 고정
- **random seed = 99** (단일 모달 베이스라인 및 triple linear L2 fusion 전 구간 공통)
- split indices CSV로 팀 공유, Test set은 프로젝트 후반부까지 **blind** 유지 (data leakage 방지)

> **참고:** 초안(v3.0)은 60/20/20 + seed=42였으나, 실제 수행된 학습 파이프라인(`feature/LJW` 브랜치)이 70/15/15 + seed=99를 사용해 삼중 임베딩을 생성했기 때문에 기획서를 현실에 맞춰 갱신함.

---

## 3. 방법론

### 3.1 전체 파이프라인 아키텍처

```
┌────────────────────────────────────────────────────────────────────────┐
│                 NSCLC-Radiomics (n=420, 전처리 완료)                   │
│                   Split 70/15/15 = 293/64/63 (seed=99)                 │
└────────────────────────────────────────────────────────────────────────┘
          │                    │                      │
          ▼                    ▼                      ▼
    ┌──────────┐        ┌────────────┐        ┌────────────┐
    │   CT     │        │  Radiomics │        │  Clinical  │
    │  NIfTI   │        │   (CSV,    │        │  (CSV)     │
    │          │        │  추출완료) │        │            │
    └────┬─────┘        └──────┬─────┘        └──────┬─────┘
         │                     │                     │
         ▼                     ▼                     ▼
    Preproc: HU clip,     LASSO 15 feat 선택    11 임상 피처 인코딩
    GTV ROI crop 80³,     + z-score             (age, T/N/M, Overall,
    z-score, TTA          (train fit only)      Histology, Gender)
         │                     │                     │
         ▼                     ▼                     ▼
    ┌────────────┐      ┌────────────┐        ┌────────────┐
    │ Hosny 3D   │      │ Radiomics  │        │ Clinical   │
    │ CNN        │      │ MLP head   │        │ MLP head   │
    │ (4 block,  │      │ (Optuna)   │        │ (Optuna)   │
    │ ~550k par, │      │            │        │            │
    │ from       │      │            │        │            │
    │ scratch)   │      │            │        │            │
    │ → 256-dim  │      │ → 256-dim  │        │ → 128-dim  │
    └────┬───────┘      └──────┬─────┘        └──────┬─────┘
         │                     │                     │
         └─────────────┬───────┴─────────────┬───────┘
                       │                     │
                       ▼                     ▼
                  ┌────────────────────────────────┐
                  │  Concat 640-dim                │
                  │  (clin 128 + radio 256 + ct 256)│
                  │  → standardize                 │
                  │  → Linear(640,1) L2 LogReg     │
                  │  → Sigmoid                     │
                  │  (chosen_C=1e-3, trainval fit) │
                  └────────────┬───────────────────┘
                               │
                               ▼
                    P(2-year survival)
                    Test AUROC = 0.659
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
      Grad-CAM             SHAP                  Modality
      (Hosny CNN)         (LinearExplainer      Ablation
       last conv)          on 640-dim            (zero-mask
                           → top-10 feature)     per modality)
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │   Streamlit 로컬 데모 (서비스)    │
              │   ──────────────────────────     │
              │   [입력] 임상 변수 폼            │
              │        + CT NIfTI 업로드         │
              │   [추론] M4 예측 P(생존)         │
              │   [설명] Grad-CAM · SHAP ·       │
              │          Modality Ablation       │
              └──────────────────────────────────┘
```

### 3.2 예측 모델 (NIH-MIP `Multimodal_RPModel` 패턴 이식)

NIH-MIP 원본 연구(Simon et al., *Clinical Imaging* 2025)는 4개 모델을 비교하는 구조를 가진다. 본 프로젝트는 이 패턴을 채용하되, 구현은 **frozen encoder + linear fusion** 2단계로 수행했다.

| 모델 ID | 입력 모달리티 | 인코더 / 분류기 | 상태 | Test AUROC |
|---|---|---|---|---|
| **M1** | Clinical only (11 feat) | MLP (Optuna) + linear head | 완료 | ~0.55–0.62 |
| **M2** | CT only | Hosny 3D CNN (from-scratch) + MLP head | 완료 | ~0.55–0.62 |
| **M3** | Radiomics only (15 LASSO) | MLP (Optuna) + linear head | 완료 | ~0.55–0.62 |
| **M4** | CT + Radiomics + Clinical (Triple Linear L2) | 3개 frozen encoder 임베딩 concat (640-dim) + L2 LogReg | 완료 | **0.659** |

> **v3.0과의 차이:** 초안은 M3을 "Radiomics+Clinical fusion"으로 정의했으나, 실제 `feature/LJW/single_modal_baseline` 브랜치에서는 세 모달리티 모두 **단독 baseline**으로 학습됐다 (M3 = Radiomics-only). M4는 세 인코더의 출력을 frozen으로 고정한 뒤 **로지스틱 회귀(L2) 형태의 late linear fusion**으로 구현됨.

Baseline 비교군으로 **TNM Stage 단독 logistic regression** 모델을 함께 평가한다.

### 3.3 각 모달리티 처리 방법 (실제 학습된 구성)

#### 3.3.1 CT 이미지 (M2 — Hosny shallow 3D CNN)

**원본 데이터:** 축방향 slice **512×512**, voxel spacing은 환자별로 다를 수 있음.

- **전처리 (`scripts/ct/ct_preprocess.py`):**
  - HU clipping: [-1000, 400] → 정규화 [0, 1]
  - GTV-1 중심 3D ROI crop: **80×80×80 voxels** (Hosny 2018 원논문 50³에서 stage III 중심 LUNG1 코호트의 tumor clip 문제 해결 위해 확장)
  - isotropic resampling 1×1×1 mm³
- **모델 (`scripts/ct/hosny_cnn.py`):** Hosny 2018 shallow 3D CNN
  - 4 conv blocks: (32 → 64 → 128 → 256 channels), 각 block = Conv3d(3×3×3) → BN → ReLU → MaxPool3d(2) → Dropout3d(0.1)
  - GlobalAvgPool → Flatten → Dropout(0.3~0.4) → Linear(256 → 2)
  - **총 ~550k params**, from-scratch 학습 (MedicalNet pretrained 미사용)
- **Augmentation:** random flip, rotation (±10°), intensity shift (±5%)
- **Test-time augmentation:** L-R flip 평균 (feature extraction 단계)
- **임베딩 추출 (`scripts/ct/extract_features.py`):** 학습 완료된 checkpoint에서 GAP 출력 256-dim을 저장 → `results/ct/features/{train,val,test}.npz`

> **v3.0과의 차이:** 초안은 MedicalNet ResNet-10/18 + 128×128×64 ROI + 512-dim 임베딩을 계획했으나, 소규모 코호트(n≈293 train)에서 깊은 ResNet의 overfit 위험과 pretrained weight 수급 문제 때문에 **Hosny 4-block shallow CNN (from-scratch, 80³, 256-dim)** 으로 전환.

#### 3.3.2 Radiomics (M3 — MLP encoder)

- **추출 완료 가정** — 프로젝트는 추출된 CSV(`radiomics_final_features.csv`)에서 시작
- **피처군 (IBSI-compliant, PyRadiomics 기반):** Shape, First-order, GLCM, GLRLM, GLSZM, NGTDM, GLDM (약 100개, wavelet 제외 기준)
- **피처 선택:** train set에서 LASSO로 **15개 피처 선택**, val/test는 동일 세트만 사용
- **정규화:** Z-score standardization (train 평균/표준편차 적용)
- **MLP 인코더:** Optuna로 하이퍼파라미터 탐색 → **256-dim 임베딩 출력**
- **출력 체크포인트:** `radiomics_embed_128.npy`는 초기 시도(128-dim)본이며, 최종 채택된 인코더는 **256-dim** (triple linear L2 입력과 일치)

#### 3.3.3 임상 변수 (M1 — MLP encoder)

- **입력 11개 피처:**
  - Age (연속, z-score)
  - Clinical T/N/M stage (ordinal)
  - Overall Stage (one-hot)
  - Histology (one-hot: adenocarcinoma, large-cell, squamous-cell, NOS)
  - Gender (binary)
  - (+ 결측 flag 1~2개)
- **결측 처리:** train set의 중앙값/최빈값 대체, 결측 flag 피처 추가
- **MLP 인코더:** Optuna로 탐색 → **128-dim 임베딩 출력**
- **출력 체크포인트:** `clinical_branch.pt` (단, 체크포인트 이름은 실제 레포지토리 파일에 따름)

### 3.4 Fusion 전략 (M4 — Triple Linear L2)

**실제 구현 (`checkpoints/triple_linear_l2_best.pt`):**

```
[Clinical 128] ⊕ [Radiomics 256] ⊕ [CT 256]  → 640-dim concat
     → standardize (mean / scale in checkpoint)
     → Linear(640, 1)  [L2-regularized logistic regression]
     → Sigmoid
```

- **학습 방식:** scikit-learn `LogisticRegressionCV`
  - `Cs = np.logspace(-4, 4, 17)`, CV = 5-fold, scoring = `roc_auc`
  - `penalty=l2`, `solver=lbfgs`, `class_weight=balanced`, `max_iter=2000`
  - `train_subset = "train+val (trainval headline)"` — 최종 fit은 train+val 합쳐 재학습
- **선택된 하이퍼파라미터:** `chosen_C = 1e-3`
- **체크포인트 구성:**
  - `state_dict.mean` / `state_dict.scale` (640,) — 표준화 상수
  - `state_dict.linear.weight` (1, 640), `state_dict.linear.bias` (1,)
  - `modality_slices` = {clinical: [0,128], radiomics: [128,384], ct: [384,640]}
- **PyTorch drop-in 포팅:** 학습 시 sklearn으로 fit, 추론·XAI 단계에서는 동등한 PyTorch 모듈(`TripleLinearL2`)로 로드

> **v3.0과의 차이:** 초안은 "Concat(672-dim) → FC(128) → Dropout → FC(1)"의 MLP late fusion이었으나, 소규모 코호트에서 추가 MLP 레이어의 이득이 불분명했고 선형 L2가 더 안정적이었기에 **Linear L2 LogReg**로 단순화됨.
>
> **대안 실험 (Stretch, 여유 시):** attention-based fusion, gating mechanism, 얕은 MLP fusion.

### 3.5 학습 설정 요약

| 항목 | M1 / M2 / M3 (encoder+head) | M4 (triple linear L2) |
|---|---|---|
| Loss | Cross-Entropy | Weighted log-loss (class_weight=balanced) |
| Optimizer | AdamW (Optuna search) | sklearn lbfgs |
| Scheduler | CosineAnnealingLR | (N/A — 1-shot convex fit) |
| Batch size | 8 (CT), 32 (tabular) | (N/A, full-batch) |
| Epochs | 50 (early stop patience=10 on val AUROC) | max_iter=2000 |
| Regularization | Dropout 0.1~0.4, weight_decay | L2 (C grid 17점, CV=5) |
| 재현성 | seed=99, `torch.use_deterministic_algorithms(True)` | seed=99 |
| 임베딩 생성 | TTA (L-R flip 평균, CT) | — |

### 3.6 평가 지표

- **주 지표:** AUROC, Sensitivity, Specificity, F1
- **보조 지표:** Balanced Accuracy, AUPRC (class imbalance 고려)
- **통계 검정:** DeLong test로 M1/M2/M3 vs M4 AUROC 차이 유의성 검증
- **보정(Calibration):** Brier score, calibration plot
- **생존 관점 보조 분석:** M4 예측 확률로 high/low risk 이분화 후 Kaplan-Meier + log-rank test
- **현재 측정된 headline 수치 (Test n=63):**
  - M4 Test AUROC (sklearn) = **0.6589**
  - M4 Test AUROC (torch port) = **0.6589** — 드롭인 포팅 일치 확인

### 3.7 XAI 전략 (실제 모델 기준 재정의)

초안(v3.0) §3.7은 end-to-end MLP fusion을 가정했으나, 실제 M4는 **frozen 인코더 3종 + linear L2 head** 구조다. 이에 맞춰 XAI를 다음 **3종으로 단순화·고정**한다.

#### 3.7.1 CT 이미지 설명 — Grad-CAM (Hosny 3D CNN 단독 backward)

- **대상 모델:** `single_modal_baseline/results/ct/best.pt` (Hosny 4-block 3D CNN)
- **Target layer:** 마지막 conv block `features[3]`의 ReLU 직후 출력
  - 입력 80³ 기준 feature map 크기 (256, 5, 5, 5)
- **방식:** `pytorch-grad-cam` 라이브러리의 3D Grad-CAM
  - score = Hosny CNN head의 class-1(생존) logit (M4 파이프라인의 최종 확률이 아니라 **CT branch 단독 로짓** 기준)
  - 3D heatmap을 원본 80³ 해상도로 upsample → 저장
- **시각화:** axial MIP(최대 강도 투영) + 3-plane viewer (axial / coronal / sagittal) 오버레이
- **정량 평가 (Test set 전수, n=63):**
  - **IoU:** Grad-CAM 상위 k% voxel binarize vs. GTV-1 mask (k = 25%, 50% 두 기준)
  - **Pointing game:** heatmap peak voxel이 GTV-1 내부에 위치하는 case 비율
  - 결과: 평균 ± 표준편차, 히스토그램 제시
- **M4와의 연결 주석:** Grad-CAM은 CT 단독 branch로 생성하지만, 이 CT 임베딩은 frozen 상태로 M4에 그대로 투입되므로 "M4가 CT에서 본 단서"의 근사로 해석 가능하다는 점을 리포트에 명시한다.

#### 3.7.2 Tabular 설명 — SHAP Top-10 (LinearExplainer on 640-dim concat)

- **대상 모델:** Triple Linear L2 head (M4의 선형 분류기)
- **방식:** `shap.LinearExplainer`
  - 배경 분포: train+val 640-dim 임베딩 행렬
  - 선형 모델이므로 SHAP 값은 해석적으로 `w · (x − μ) / σ` 형태 → 노이즈 없이 정확
- **출력:**
  - **Global Top-10 피처:** `mean(|SHAP|)` 내림차순 상위 10개를 막대그래프로 표시
    - 각 피처에 모달리티 라벨 태깅 (Clinical / Radiomics / CT)
    - 원 임상·radiomics 피처의 경우 피처명, CT 임베딩의 경우 `CT_emb[idx]` 형식
  - **Local (개별 환자):** 해당 환자 예측에 대한 waterfall plot (Top-10 bar와 동일 기준)
- **해석 가이드:** Top-10에 CT 임베딩 차원이 다수 포함될 경우 "고수준 영상 단서가 예측을 주도", clinical feature가 다수면 "임상 지표 주도" 식의 내러티브로 연결. 개별 피처 의미는 radiomics/clinical만 직접 해석 가능하며, CT 임베딩 차원의 의미는 Grad-CAM(3.7.1)으로 간접 보조.

> **v3.0과의 차이:** 초안은 "Global summary + Local waterfall + KernelExplainer"였으나, 선형 head에서는 LinearExplainer가 정확·고속이며, 사용자 요청에 따라 **Top-10 피처 bar + 개별 waterfall 두 가지 view**로 한정.

#### 3.7.3 모달리티 기여도 — Ablation 단일화

- **방식:** 학습된 M4의 640-dim 입력에서 모달리티별 슬라이스(`modality_slices`)를 **zero-vector로 masking**
  - Clinical mask: `x[0:128] = 0`
  - Radiomics mask: `x[128:384] = 0`
  - CT mask: `x[384:640] = 0`
- **측정:**
  - Test set 전수에 대해 **ΔAUROC** (전체 AUROC − 해당 모달리티 masking 후 AUROC)
  - 개별 환자 기준 **ΔP(생존)** (서비스 화면용)
- **출력:**
  - Global: 3-모달리티 ΔAUROC bar chart (bootstrap 95% CI 포함, n=1000)
  - Local: 개별 환자 ΔP bar (Streamlit 데모 Tab 4)
- **해석:** ΔAUROC가 큰 모달리티 = M4 성능 기여도 큰 모달리티.

> **v3.0과의 차이:** 초안의 "Modality ablation + Modality-level KernelSHAP" 두 트랙에서 **ablation 한 가지로 단순화** (사용자 결정).

#### 3.7.4 설명 품질 평가 (Stretch)

- **Faithfulness** (deletion metric), **Stability** (cosine similarity), **Plausibility** (문헌 대조) — 일정 여유 시에만 수행, 발표 필수 항목 아님.

> **필수 정량평가:** IoU + Pointing game (Grad-CAM). 그 외는 Stretch.

### 3.8 서비스 구현 (Streamlit 로컬 데모)

학습·평가·XAI 완료 후, M4 모델을 실시간 추론 가능한 **Streamlit 웹 앱**으로 포장한다. 발표 당일 로컬에서 실행하고 화면 공유 방식으로 시연한다. **배포는 로컬에 한정**, 외부 호스팅 없음.

#### 3.8.1 서비스 아키텍처

```
┌────────────────────── Streamlit App ──────────────────────┐
│                                                            │
│  [Sidebar]                  [Main Area]                    │
│  ─────────                  ─────────────                  │
│   Step 1. 임상 변수 입력     [입력 요약 카드]                │
│    ├ Age (slider)           ▼                              │
│    ├ T/N/M stage (select)   [추론 버튼 "Predict"]          │
│    ├ Overall Stage          ▼                              │
│    ├ Histology              ┌─────────────────────────┐   │
│    └ Gender                 │  Output Panel (tabs)    │   │
│                             │  ─────────────────────  │   │
│   Step 2. CT 업로드         │  [Tab 1] 예측 결과       │   │
│    ├ .nii.gz / .nii         │   • P(2-yr survival)    │   │
│    └ GTV mask (.nii)        │   • Risk category       │   │
│                             │   • Confidence bar      │   │
│   [Run Inference 버튼]       │                         │   │
│                             │  [Tab 2] Grad-CAM       │   │
│                             │   • CT + heatmap overlay│   │
│                             │   • 3-plane view        │   │
│                             │   • IoU (GTV 있을 때)   │   │
│                             │                         │   │
│                             │  [Tab 3] SHAP Top-10    │   │
│                             │   • Top-10 bar (global) │   │
│                             │   • Waterfall (local)   │   │
│                             │                         │   │
│                             │  [Tab 4] Modality       │   │
│                             │         Ablation        │   │
│                             │   • ΔAUROC bar (global) │   │
│                             │   • ΔP bar (이 환자)    │   │
│                             └─────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

#### 3.8.2 입력 화면 (Input Panel)

**Sidebar Step 1: 임상 변수 입력 (11개 피처)**

| 변수 | 위젯 | 설정 |
|---|---|---|
| Age | `st.slider` | 30 ~ 90, default 65 |
| Clinical T stage | `st.selectbox` | T1 / T2 / T3 / T4 |
| Clinical N stage | `st.selectbox` | N0 / N1 / N2 / N3 |
| Clinical M stage | `st.selectbox` | M0 / M1 |
| Overall Stage | `st.selectbox` | I / II / IIIA / IIIB / IV |
| Histology | `st.selectbox` | adenocarcinoma / large-cell / squamous-cell / NOS |
| Gender | `st.radio` | male / female |

**Sidebar Step 2: CT 업로드**
- `st.file_uploader`: NIfTI 파일(`.nii`, `.nii.gz`) 1개
- (선택) GTV mask NIfTI 파일 1개 — 없으면 예측만 가능, Grad-CAM IoU 계산 불가
- 업로드 후 middle slice 썸네일 미리보기

#### 3.8.3 추론 파이프라인

1. **입력 검증:** 필수 필드 누락 시 오류 메시지
2. **CT 전처리:** 학습 때와 동일(`scripts/ct/ct_preprocess.py`와 동일 로직 재사용) — HU clip, GTV-1 중심 80³ crop, z-score
3. **Radiomics 추출 (선택):** 업로드된 CT + GTV mask가 있으면 PyRadiomics로 실시간 추출 (시간 소요 시 사전 추출 CSV에서 샘플 선택 모드 병행 제공)
4. **임베딩 생성:**
   - Hosny 3D CNN frozen forward → CT 256-dim
   - Radiomics MLP frozen forward → 256-dim
   - Clinical MLP frozen forward → 128-dim
   - → concat 640-dim
5. **M4 추론:** 캐시된 체크포인트(`checkpoints/triple_linear_l2_best.pt`) 로드 → standardize → Linear → sigmoid
6. **XAI 생성:** Grad-CAM (Hosny CNN) + SHAP (LinearExplainer, top-10) + Ablation (3-모달리티 zero mask) 병렬 계산
7. **결과 렌더:** tabs로 구분해 순차 표시

**성능 타겟:** 단일 환자 추론 + XAI 전체 ≤ **30초** (GPU 사용 시). Linear head는 즉시, Grad-CAM이 대부분의 지연 요인.

#### 3.8.4 출력 화면 (Output Panel — 4-Tab 구조)

**Tab 1: 예측 결과**
- 큰 숫자로 2년 생존 확률 표시 (`st.metric`)
- Risk category (Low/High) 배지, cutoff는 val set Youden index 기반 threshold
- 예측 신뢰도 bar

**Tab 2: Grad-CAM 시각화**
- CT slice 3-plane(axial / coronal / sagittal)에 heatmap 오버레이
- 슬라이스 번호 `st.slider`로 스크롤
- GTV mask가 업로드된 경우 IoU(25%, 50%) + Pointing-game 결과 동반 표시
- 다운로드 버튼(PNG)

**Tab 3: SHAP Top-10**
- Global Top-10 feature bar chart (모달리티별 색상)
- 해당 환자 단일 waterfall plot (Top-10 기준)
- 각 피처의 현재 값 + SHAP contribution 표

**Tab 4: Modality Ablation (단일 관점으로 축소)**
- 3-모달 각각을 zero-masking 했을 때의 **ΔAUROC** bar chart (global, 사전 계산 결과)
- 이 환자 기준 **ΔP(생존)** bar chart (실시간 계산)
- 1~2줄 자연어 해설 (예: "이 환자의 예측은 CT 임베딩 제거 시 가장 크게 흔들림 → CT 영상이 주된 근거")

> **v3.0과의 차이:** Tab 4의 "Modality-level SHAP pie chart"는 제거, **ablation 단일 뷰**로 축소.

#### 3.8.5 구현 스택

| 구성 요소 | 기술 |
|---|---|
| UI 프레임워크 | **Streamlit** (≥1.30) |
| 모델 로딩 | PyTorch + `@st.cache_resource` (체크포인트 1회만 로드) |
| CT 시각화 | `matplotlib` + `nibabel` 기반 3-plane viewer |
| Grad-CAM | `pytorch-grad-cam` (3D, Hosny CNN `features[3]` 타깃) |
| SHAP | `shap.LinearExplainer` + `shap.plots.waterfall` / `bar` |
| 실행 방식 | `streamlit run app/main.py` → 브라우저에서 `localhost:8501` |
| 배포 | **로컬 전용**, 발표 당일 워크스테이션에서 실행 |

#### 3.8.6 파일 구조

```
app/
├── main.py              # Streamlit 엔트리포인트
├── pages/
│   ├── 1_input.py       # 입력 페이지 (변수·업로드)
│   └── 2_result.py      # 결과 페이지 (4 tabs)
├── inference/
│   ├── preprocess.py    # 학습 파이프라인 재사용 (HU clip, ROI crop 80³)
│   ├── encoders.py      # 3개 frozen encoder 로더 (Hosny CNN / Radio MLP / Clin MLP)
│   ├── predict.py       # triple linear L2 forward pass wrapper
│   └── xai.py           # Grad-CAM / SHAP LinearExplainer / Ablation 통합 API
├── ui/
│   ├── sidebar.py       # 입력 위젯 모듈
│   └── plots.py         # 시각화 헬퍼
└── assets/
    └── sample_cases/    # 데모용 테스트 환자 2~3명 사전 배치
```

#### 3.8.7 데모 시나리오 (발표 당일)

1. **준비 단계:** Day 10 오전에 사전 로드 스모크 테스트, 샘플 케이스 2~3명 `app/assets/sample_cases/`에 배치 (전처리·임베딩 사전 계산된 버전 동반 저장으로 실시간 경로 문제 시 fallback)
2. **발표 중 시연 흐름 (3~4분):**
   - 샘플 환자 A (low-risk 예상): 업로드 → 예측 → Grad-CAM으로 종양 영역 확인 → SHAP Top-10에서 어느 모달리티가 주도했는지 확인
   - 샘플 환자 B (high-risk 예상): 동일 흐름 → Modality Ablation에서 각 모달리티의 ΔP 비교
3. **Fallback:** 라이브 시연 실패 대비 **사전 녹화 영상(30초)** 준비, 슬라이드에 임베드
4. **제약 고지:** 이것은 **연구용 데모**이며 임상 진단 도구가 아님을 화면 footer와 발표 초반에 명시. Test AUROC 0.659의 한계도 투명하게 공개.

---

## 4. 팀 구성 및 역할 분담

팀원 4명 전원이 ML 경험 보유, 그중 2명 이상이 DICOM/의료영상에 익숙한 점을 반영해 **병렬 처리 + 주기적 통합** 구조로 설계.

| 역할 | 담당자 | 주 책임 | 보조 책임 |
|---|---|---|---|
| **PM / 통합 리드** | 팀원 A | 일정 관리, 회의 진행, 데이터 split 관리, 최종 통합·발표, **Streamlit 앱 통합** | M4 fusion 포팅 |
| **영상 파이프라인 리드** (DICOM 경험) | 팀원 B | GTV ROI crop, Hosny 3D CNN(M2) 학습/임베딩, **Grad-CAM (3D, `pytorch-grad-cam`)**, IoU/Pointing game, **앱 Grad-CAM 탭** | CT QC |
| **Radiomics & Tabular 리드** (DICOM 경험) | 팀원 C | Radiomics 피처 선택, M1/M3 학습/임베딩, **SHAP Top-10 (LinearExplainer)**, 임상변수 전처리, **앱 입력 폼·SHAP 탭** | Calibration |
| **XAI & 평가 리드** | 팀원 D | 멀티모달 XAI 통합, 평가 지표 구현, DeLong, Kaplan-Meier, **Modality Ablation**, **앱 Ablation 탭** | 발표자료 디자인 |

### 4.1 GPU 자원 및 분담 전략

- **하드웨어:** **RTX 4070 Ti Super (16GB VRAM, 32GB RAM) — 단독 사용 (팀 공용 1대)**
- **공유 방식:** GPU는 1대이므로 **시간 분할 공유** 필요
  - M2 (3D CNN) 학습·Grad-CAM: 주간 시간대에 담당자 독점 사용
  - M1, M3 (tabular) 학습: CPU로도 가능
  - M4 학습: 이미 완료, 재학습 필요 없음
- **실험 추적:** wandb/MLflow
- **체크포인트 정책:** `checkpoints/triple_linear_l2_best.pt` 고정, 단일 모달 베이스라인은 `feature/LJW/single_modal_baseline/results/{clinical,radiomics,ct}/` 참조

### 4.2 협업 인프라

- **코드:** GitHub private repo, `main`/`dev`/feature branch 전략, PR 리뷰 필수
  - 단일 모달 베이스라인: `feature/LJW/single_modal_baseline` 브랜치
- **실험 추적:** Weights & Biases (무료 tier 충분)
- **데이터/모델 저장소:** 전처리된 NIfTI는 워크스테이션 로컬, 체크포인트·결과는 Google Drive/NAS로 백업
- **커뮤니케이션:** Slack/Discord, Daily 15분 stand-up (원격 포함)
- **문서:** Notion 또는 공유 Google Docs
- **환경 통일:** `environment.yml` (conda), `requirements.txt` 공용, Python 3.10 / PyTorch 2.x / CUDA 12 / MONAI

---

## 5. 일정 (10일 — 현재 Day 6~7 시점 재조정)

데이터 준비와 **단일 모달 베이스라인(M1/M2/M3) + Triple Linear L2 fusion(M4)** 이 이미 완료된 시점을 전제로, 남은 일정은 **XAI 3종 수행 + 서비스 구현 + 발표 준비**에 집중한다.

| Phase | Day | 주요 마일스톤 | 담당 | 상태 |
|---|---|---|---|---|
| **Phase 1 — Setup & Single Modality** | | | | |
| | Day 1 | Kick-off, 환경 셋업, EDA, 타겟 라벨 생성, 통일 split(70/15/15, seed=99) 확정 | 전원 | ✅ |
| | Day 2 | Radiomics 피처 선택 (LASSO 15), M3 MLP 학습 | C | ✅ |
| | Day 3 | CT 전처리(80³ ROI) + Hosny 3D CNN(M2) 학습 | B | ✅ |
| | Day 4 | M2 학습 완료 / 튜닝, M1 Clinical MLP 학습 | B, C | ✅ |
| | Day 5 | **중간 체크포인트 #1** — M1/M2/M3 결과 공유 | 전원 | ✅ |
| **Phase 2 — Multimodal & XAI** | | | | |
| | Day 6 | **M4 triple linear L2 fit(LogisticRegressionCV, trainval) 완료, 체크포인트(`triple_linear_l2_best.pt`) 확보 → 기획서 v4.0 재정렬** | A, B, C | ✅ |
| | Day 7 | Test set 평가 (M1/M2/M3/M4 AUROC, DeLong, Brier, KM log-rank) | A, D | 진행 |
| | Day 8 | **XAI 3종:** Grad-CAM(3D, Hosny CNN) + IoU/Pointing game, SHAP Top-10 (LinearExplainer), Modality Ablation (ΔAUROC bootstrap) | B, C, D | 예정 |
| **Phase 3 — 서비스 구현 & 발표** | | | | |
| | Day 9 | **Streamlit 앱 개발** — 입력 폼 + 추론 파이프라인(3-encoder frozen forward + linear head) + XAI 3-tab | A, B, C, D | 예정 |
| | Day 10 | 앱 통합 테스트 + 샘플 케이스 검증, 발표자료 통합, 리허설, 최종 발표 + 라이브 데모 | 전원 | 예정 |

### 5.1 Day 별 상세 — XAI + 서비스 세부 분해

#### Day 8 — XAI 생성 + 정량평가 통합

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 | Grad-CAM 3D 생성 + MIP 오버레이 (Hosny CNN `features[3]` 타깃) | B | `results/gradcam/{train,test}/*.nii.gz`, `figures/gradcam_overlays/*.png` |
| 오전 | SHAP LinearExplainer on 640-dim → Top-10 bar + waterfall | C | `results/shap_values.npy`, `figures/shap_top10_bar.png`, `figures/shap_waterfall_*.png` |
| 오후 | Modality Ablation (zero-mask, bootstrap CI) | D | `results/ablation.csv`, `figures/ablation_bar.png` |
| 오후 | IoU + Pointing game 정량평가 (Test n=63 전수) | B | `results/gradcam_iou.csv`, `results/pointing_game.csv` |
| 저녁 | 결과 통합 + 임상 해석 초안 + Auditor 품질 검사 | A, C, Auditor | `report/xai_interpretation.md` |

> **drop 항목:** Faithfulness, Stability는 Stretch로 이동. Modality-level SHAP도 제거 (Ablation 단일화).

#### Day 9 — Streamlit 앱 개발

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 | 앱 스켈레톤 + Sidebar 입력 폼 구현 (11개 clinical feat) | C | `app/main.py`, `app/ui/sidebar.py` |
| 오전 | 추론 파이프라인 래핑 (3-encoder frozen + M4 linear head) | A | `app/inference/encoders.py`, `predict.py` |
| 오후 | Grad-CAM 3-plane viewer 탭 (`features[3]` 3D CAM) | B | `app/ui/plots.py` 내 gradcam 섹션 |
| 오후 | SHAP Top-10 bar + waterfall 탭 | C | SHAP 탭 완성 |
| 오후 | Modality Ablation 탭 (ΔAUROC + ΔP) | D | Ablation 탭 완성 |
| 저녁 | 전체 통합 + 샘플 케이스 2~3명 동작 확인 | A (전원) | `app/assets/sample_cases/` |

#### Day 10 — 앱 검증 + 발표자료 + 리허설 + 발표

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 (1) | 앱 smoke test (샘플 케이스 2~3명 전수 시연) | A, B | 시연 영상 녹화 (fallback용) |
| 오전 (2) | 발표 슬라이드 통합 (모델·XAI·서비스 섹션) | 담당별 병렬 | 최종 pptx |
| 오후 (1) | 내부 리허설 (앱 라이브 시연 포함 25분) | 전원 | 피드백 메모 |
| 오후 (2) | 피드백 반영, 앱 마이너 버그 수정 | 담당별 | 최종본 |
| 저녁 | **최종 발표 (라이브 데모 + 슬라이드)** | 전원 | 발표 수행, 회고 |

### 5.2 일정 압박 대응 전략

- **Day 7 체크포인트**: Test 평가·DeLong 미완 시 Day 8 오전으로 이월
- **Day 8 체크포인트**: XAI 3종 결과 미확보 시 Day 9 서비스 구현에 영향 → 팀원 1명 XAI 마무리에 추가 할당
- **Day 9 체크포인트**: 앱 기본 골격이 안 되면 **사전 녹화 영상으로 대체**하고 라이브 데모 포기
- **서비스 기능 축소 순서 (위부터 drop)**:
  1. Modality Ablation 탭 → 정적 PNG로 대체
  2. SHAP 탭 → Top-10 bar 1장 이미지로 대체
  3. CT 업로드 기능 → 사전 로드 샘플 3명 드롭다운 선택만 제공
  4. 위 3개가 모두 실패해도 Grad-CAM + 예측 결과만은 반드시 유지

### 5.3 진행률 가시성

- Daily stand-up에서 각자 "어제/오늘/장애물" 1분씩 공유
- Day 7 / Day 8 체크포인트 후 traffic-light(녹/황/적) 공유
- 적색 전환 시 Stretch(Faithfulness/Stability/Attention fusion) 전부 drop, 서비스는 최소 기능으로 축소

---

## 6. 리스크 관리

| 리스크 | 가능성 | 영향도 | 대응 전략 |
|---|---|---|---|
| M4 AUROC(0.659)가 단일 모달 대비 유의한 개선이 아닐 가능성 | 높음 | 중 | DeLong test 결과를 투명하게 보고, "멀티모달의 성능 이득이 크지 않음"을 결론의 한 축으로 수용. XAI 분석 자체가 핵심 산출물로 위치 조정 |
| Grad-CAM이 GTV 외 영역(폐 실질·배경)에 강하게 반응 | 중 | 중 | 저성능(0.659) 모델의 attention이 noisy할 수 있음. IoU 낮은 case를 **failure case analysis**로 포함, lung mask 적용 후 heatmap 재시각화 고려 |
| CT 인코더 체크포인트(`feature/LJW`) 접근 문제 | 낮음 | 높음 | 브랜치 clone 즉시 수행, `results/ct/best.pt`를 `checkpoints/`로 미러링 |
| Linear head라 SHAP Top-10이 거의 feature magnitude만 반영 | 중 | 낮음 | SHAP 값 = `coef × standardized input`임을 리포트에 명시, 별도로 coefficient 기여도 table도 함께 제시 |
| Streamlit 앱에서 3D CNN forward 지연 (Grad-CAM 포함) | 중 | 중 | 샘플 케이스 사전 계산 결과 cache, 실시간 경로는 GPU inference 필수 |
| 발표 중 라이브 데모 실패 (업로드/추론 에러) | 중 | 중 | 사전 녹화 영상 준비(Day 10 오전), 샘플 케이스 드롭다운 fallback 경로 확보 |
| 10일 일정 지연 (버퍼 없음) | 중 | 높음 | Stretch 항목 즉시 drop, XAI 정량 평가 중 선택적 항목 생략, Minimum 기준 우선 달성 |
| Class imbalance | 중 | 낮음 | Linear head는 이미 `class_weight=balanced`로 fit됨, 평가 시 AUPRC도 함께 보고 |

---

## 7. 최종 산출물 (내부 스터디 발표용)

### 7.1 발표 자료 (핵심 deliverable)

1. **프로젝트 개요 슬라이드** (1~2장): 문제 정의, 데이터(n=420), 목표, **실제 달성치 투명 공개 (M4 AUROC 0.659)**
2. **데이터 EDA** (2~3장): 코호트 특성, 생존 분포, stage 분포, censoring 처리 후 유효 n, 통일 split(70/15/15)
3. **방법론** (3~4장): 전체 파이프라인 다이어그램, 각 모달리티 전처리, M1~M4 구조 (Hosny CNN 강조, Linear L2 fusion)
4. **결과** (4~5장):
   - M1~M4 성능 비교 테이블 (AUROC, Sens, Spec, F1, AUPRC, Brier)
   - ROC curve (4개 모델 오버레이)
   - DeLong p-value 매트릭스
   - Kaplan-Meier 생존곡선 (M4 기반 risk stratification)
5. **XAI 분석** (3~4장):
   - **Grad-CAM** 예시 3~5명 (성공 1~2명 + 실패 1명) + IoU@25/50% + Pointing game 수치
   - **SHAP Top-10** bar chart + 모달리티별 기여 비율
   - **Modality Ablation** ΔAUROC bar chart (bootstrap CI)
6. **서비스 데모** (2~3장 + 라이브 시연 3~4분):
   - 앱 아키텍처·UI 스크린샷
   - **라이브 데모:** 샘플 환자 A(low-risk) → B(high-risk) 순차 시연
   - Fallback 영상(30초) 슬라이드에 임베드
7. **논의 및 한계** (1~2장): **M4 AUROC 0.659의 해석** (small-n, linear fusion의 한계), 개선 방향 (deeper CT backbone, attention fusion, 외부 validation)
8. **배운 점 / 팀 회고** (1장): 스터디 목적에 부합하는 learning 정리

### 7.2 부속 산출물

- GitHub repo (재현 가능한 코드 + README + 환경설정)
- `app/` 디렉토리: Streamlit 앱 소스 + 샘플 케이스 + 실행 가이드
- `results/` 폴더: 평가 지표 CSV, 그림 PNG/PDF, Grad-CAM NIfTI
- 모델 체크포인트 (`checkpoints/triple_linear_l2_best.pt` + 단일 모달 인코더들)
- `REPORT.md`: 발표 내용의 텍스트 버전
- 라이브 데모 fallback용 녹화 영상 (30초~1분)

---

## 8. 참고 자료

### 8.1 주요 참조 레포지토리

| 용도 | 링크 |
|---|---|
| 예측 모델 패턴 | [NIH-MIP/Multimodal_RPModel](https://github.com/NIH-MIP/Multimodal_RPModel) |
| XAI 원리 | [cgiova/multimodal-xai-prostate](https://github.com/cgiova/multimodal-xai-prostate) |
| 데이터셋 | [TCIA NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) |
| 내부 단일 모달 baseline | `recognisemeimthebest/miniproj @ feature/LJW/single_modal_baseline` |
| Hosny 2018 shallow 3D CNN | Hosny et al., PLOS Medicine 2018 (원논문) |
| Radiomics 추출 | [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) |
| Grad-CAM | [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| SHAP | [slundberg/shap](https://github.com/shap/shap) (LinearExplainer) |

### 8.2 핵심 논문

- Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach.* **Nat Commun** 2014;5:4006. (원 Lung1 데이터셋 논문)
- Hosny A, et al. *Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study.* **PLOS Medicine** 2018;15(11):e1002711. (M2 CT CNN 구조)
- Simon BD, Harmon SA, Turkbey B, et al. *A multimodal automated deep learning-based model for predicting biochemical recurrence of prostate cancer.* **Clin Imaging** 2025;126:110579. (M1~M4 모델 구조 참조)
- Selvaraju RR, et al. *Grad-CAM: Visual explanations from deep networks.* **ICCV** 2017.
- Lundberg SM, Lee SI. *A unified approach to interpreting model predictions.* **NeurIPS** 2017. (SHAP)

### 8.3 데이터 인용 (필수)

> Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., Leijenaar, R. T. H., Parmar, C., Grossmann, P., Carvalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., Rietveld, D., Hoebers, F., Rietbergen, M. M., Leemans, C. R., Dekker, A., Quackenbush, J., Gillies, R. J., Lambin, P. (2014). **Data From NSCLC-Radiomics (version 4)** [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

---

## 9. 부록

### 9.1 예상 소프트웨어 스택

```
- Python 3.10
- PyTorch 2.x (CUDA 12)
- MONAI (의료영상 딥러닝 유틸, resnet18/persistent dataset)
- SimpleITK, nibabel (NIfTI I/O)
- PyRadiomics (이미 추출 완료, 재현용)
- scikit-learn (LogisticRegressionCV for M4)
- pandas, numpy
- shap (≥0.44, LinearExplainer), pytorch-grad-cam (3D)
- lifelines (KM curve, log-rank)
- wandb
- matplotlib, seaborn
- Streamlit (≥1.30)
```

### 9.2 윤리 및 데이터 사용 규정

- TCIA 데이터는 **de-identified public dataset**이므로 추가 IRB 불필요 (기관 정책에 따라 내부 확인 권장)
- CC BY-NC 3.0 라이선스 — **비상업적 연구 목적에 한함**, 재배포 시 인용 필수
- 발표·문서에서 데이터 citation 누락 금지

### 9.3 성공 기준 요약 (v4.0 재조정)

| 수준 | 기준 | 현재 상태 |
|---|---|---|
| **Minimum** (반드시 달성) | M1~M4 모두 학습 완료, 성능 비교 테이블 제시, Grad-CAM + SHAP + Ablation 각 최소 1건씩, **Streamlit 앱 — 사전 로드 샘플 케이스 1명 이상 시연 가능** | M4 학습 완료 (AUROC 0.659), XAI·앱 예정 |
| **Target** (목표) | DeLong 기반 M4 vs 단일모달 비교 결과 해석, XAI 정량 평가 1개 이상 (IoU + Pointing game), **앱 — CT 업로드 포함 full flow, 3-tab XAI 모두 동작** | 예정 |
| **Stretch** (여유 시) | Attention fusion 실험, Faithfulness/Stability 평가, 반복 실험 신뢰구간, **앱 — 실시간 Radiomics 재추출, 3-plane CT viewer 완성도** | 조건부 |

> **AUROC Target 0.75는 현 모델로 달성되지 않았음.** v4.0에서는 이를 숨기지 않고 "멀티모달 late fusion이 소규모(n≈293 train) NSCLC 2년 생존 예측에서 보인 현실적 한계"로 리포트하고, **XAI 분석의 교육적·해석적 가치를 중심 산출물**로 재배치한다.

---

*작성일 (v4.0): 2026-04-22*
*문서 버전: v4.0*
*주요 변경:*
- *v2.0: 10일 일정 압축, GPU 단독 사용(RTX 4070 Ti Super) 반영, 데이터 준비 완료 전제, n=419 명시, CT slice shape 512×512 반영*
- *v3.0: **서비스 구현 단계(Streamlit 로컬 데모) 추가** — 3.8절 신설, Day 9 서비스 개발 / Day 10 통합·발표로 재편, 기존 Day 8+9 XAI를 Day 8에 압축, Faithfulness/Stability는 Stretch로 이동*
- *v4.0: **실제 학습 완료 모델(`triple_linear_l2_best.pt`)에 맞춰 기획서 전면 재정렬.** (1) Split 70/15/15, seed=99로 갱신, (2) CT backbone을 MedicalNet ResNet-10 → **Hosny 2018 shallow 3D CNN(80³, 256-dim, from-scratch)** 로 교체, (3) Radiomics/Clinical 인코더 출력 차원 256/128로 명시, (4) M4 fusion을 FC MLP → **Linear L2 LogReg (640→1)** 로 교체, (5) XAI를 사용자 확정 3종(Grad-CAM / SHAP Top-10 / Modality Ablation)으로 단순화 — Modality-level SHAP 제거, (6) Success criteria에 **Test AUROC 0.659 현실 반영** 및 XAI 중심 산출물 재배치.*
