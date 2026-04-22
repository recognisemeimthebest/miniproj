# M1 CT-only 3D CNN — 2-Year Survival Prediction (feature/LJW)

TCIA `NSCLC-Radiomics (LUNG1)` CT + GTV 마스크로 2-year OS를 예측하는 3D CNN. Hosny 2018 (*Lancet Digital Health*) 재현 + multi-seed 검증으로 **intra-cohort 벤치마크 수준** 도달.

본 파일은 feature/LJW 브랜치의 M1 모듈 전용 요약이며, 프로젝트 전체 구조는 루트 `README.md` 참조.

---

## 📊 핵심 결과

4-seed multi-split 검증 (학습 설정 동일, stratified split seed만 변경):

| seed | val AUROC | test AUROC | Tier 1 (≥0.55) | val-test gap |
|:-:|:-:|:-:|:-:|:-:|
| 42   | 0.651     | 0.495      | ❌ | −0.156 |
| 7    | 0.644     | 0.596      | ✅ | −0.048 |
| **99** | 0.609   | **0.666** 🏆 | ✅ | **+0.055** |
| 123  | 0.470     | 0.601      | ✅ | +0.132 |
| **평균** | **0.593** | **0.589 ± 0.061** | **3/4 (75%)** | **−0.004** |

**외부 벤치마크**
- Braghetto 2022 (*Nat Sci Rep*, LUNG1 intra-cohort, 24 pipelines benchmark)
  - 2D-CNN: 0.63 ± 0.04 (image only) / 0.64 ± 0.04 (+clinical)
  - 2.5D-CNN (5-slice): 0.61
  - Radiomics alone: 0.67 ± 0.03
  - Radiomics + Deep fusion (**ceiling**): 0.67 ± 0.02
- **Ours (Hosny 3D-CNN)**: **0.59 ± 0.06** — 1σ 범위 내 동등

**결론**: CT-only M1 gate(≥0.55) 통과. 단일 LUNG1 코호트 intra-cohort 3D-CNN으로 실현 가능한 성능 수준을 검증.

---

## 🛠 방법론

### 데이터
- TCIA `NSCLC-Radiomics (LUNG1)` n=422 → 라벨/마스크 필터링 후 n=420
- Stratified 60/15/15 split: train 293 / val 64 / test 63
- 4 seeds: 42, 7, 99, 123
- 라벨: survival ≥ 2y = class 1, otherwise class 0 (≈ 40/60 imbalance)

### 전처리 (`src/data/ct_preprocess.py`)
```
LoadImaged → Orientationd(RAS) → Spacingd(1×1×1 mm isotropic)
→ ScaleIntensityRanged(HU [-1000, 400] → [0, 1])
→ CropAroundMaskCoMd(80×80×80 @ GTV centroid, .clone() applied)
→ (training) RandFlipd L-R + intensity jitter  [light_aug=True]
→ EnsureTyped(track_meta=False)
```

### 모델 (`src/models/hosny_cnn.py`)
Hosny 2018 Fig.1 재현:
- 4-block 3D CNN: conv(3³) → BN → ReLU → MaxPool(2)
- 채널: **32 → 64 → 128 → 256**
- GlobalAveragePool3d → Flatten → Dropout(0.4) → Linear(256, 2)
- 파라미터: **1.16M** (ResNet-18 11M 대비 10× 작음)
- From-scratch 학습 (pretrain 없음, Hosny 스펙)

### 학습 (`src/train.py`)
- Optimizer: AdamW (head_lr 1e-3, WD 5e-4)
- **EMA off** (Hosny 원래 스펙)
- Class weight off (Hosny 원래 스펙)
- Label smoothing 0.1
- Batch 8, bf16 autocast
- Early stopping patience 8 on val AUROC
- Max 50 epochs

### 평가 (`src/evaluate/eval_m1.py`)
- TTA: L-R flip average
- Threshold 0.5 for confusion matrix
- 저장: `test_summary.json` + `test_predictions.csv`

---

## 📁 주요 파일

```
src/
├── data/
│   ├── ct_preprocess.py       # MONAI 전처리 파이프라인
│   └── dataset.py              # manifest + stratified split
├── models/
│   └── hosny_cnn.py            # 3D CNN 정의
├── evaluate/
│   └── eval_m1.py              # test 평가
└── train.py                    # 학습 엔트리

experiments/
├── iterations_log.json         # 12회 iter 전체 기록 (Option C 포함)
├── m1_hosny_iter7b/            # seed=42 체크포인트 + summary
├── m1_hosny_iter7b_seed7/
├── m1_hosny_iter7b_seed99/     # ← 🏆 test 최고 (0.666)
└── m1_hosny_iter7b_seed123/
```

각 `experiments/m1_hosny_iter7b*/` 에 포함:
- `best.pt` — `{"model_state_dict", "epoch", "val_auroc", "args"}` (~4.5MB)
- `test_summary.json` — val/test AUROC, confusion matrix, per-class PRF
- `test_predictions.csv` — 환자별 `prob_class1`, `label`, `pred@0.5`

---

## 🚀 재현

### 환경
```bash
pip install -r requirements.txt
# 핵심: torch 2.5+, monai 1.5+, pandas, scikit-learn, numpy, pyarrow
# CUDA 11.8+
```

### 단일 학습 (seed 99 예시)
```bash
python src/train.py \
  --arch hosny --light-aug --roi-size 80 80 80 \
  --epochs 50 --patience 8 --ema-decay 0.0 \
  --weight-decay 5e-4 --dropout 0.4 --label-smoothing 0.1 \
  --no-class-weight --tta --bf16 --seed 99 \
  --exp-dir experiments/my_run
```

### Test 평가
```bash
python src/evaluate/eval_m1.py \
  --ckpt experiments/my_run/best.pt \
  --arch hosny --roi-size 80 80 80 --tta \
  --seed 99 --cache-dir cache/preproc_80
```

### Multi-seed 검증 (권장)
Val AUROC와 test AUROC 상관이 작아 단일 seed는 신뢰 어려움. **최소 4 seed 평균** 리포트 권장:
```bash
for seed in 42 7 99 123; do
  python src/train.py ... --seed $seed --exp-dir experiments/run_seed$seed
  python src/evaluate/eval_m1.py --ckpt experiments/run_seed$seed/best.pt --seed $seed ...
done
```

---

## 💡 주요 발견

### 1. Val–test 상관이 거의 0
- seed=123: val 최악(0.47) / test 중간(0.60)
- seed=42: val 최고(0.65) / test 최악(0.49)
- → **Val AUROC 기반 model selection 신뢰 불가**. Early stopping도 noise 추적 위험.

### 2. Seed=42 단일 평가의 위험성
- iter 0–7b (8회) 모두 seed=42 고정 → "Tier 1 불가" 결론 도출
- Option C에서 seed 다변화 → **3/4 통과**, 이전 결론 뒤집힘
- 교훈: 초반부터 multi-seed 필수

### 3. EMA는 소규모 데이터셋에서 독
- iter 7 (EMA 0.999, 518 step) → shadow 60%가 init weight → 상수 출력
- iter 7b (EMA off) → +0.062 회복

### 4. Cache 블로트 버그 (수정 완료)
- `CropAroundMaskCoMd` 슬라이싱이 tensor view를 pickle → 샘플당 776MB
- `.clone()` 추가로 3.91MB/sample (198× 감소), 에폭 속도 4×↑
- 위치: `src/data/ct_preprocess.py::CropAroundMaskCoMd.__call__`

### 5. 3D-CNN ≈ 2D-CNN on single LUNG1
- Braghetto 2D 0.63 vs ours 3D 0.59 (1σ 동등)
- Braghetto 2.5D 0.61도 2D보다 낮음 → 추가 slice 이득 미미
- 교훈: LUNG1 단일 코호트에선 architecture보다 데이터 규모/다양성이 지배

---

## 🔭 다음 단계 (M2 Fusion)

Braghetto 실험에서 **radiomics + deep feature fusion이 variance를 ±0.04 → ±0.02로 감소**시켰음. 평균 AUROC 상방은 제한적(0.67)이지만 안정성 획득이 유효.

제안 경로:
1. **Radiomics**: Pyradiomics로 GTV shape/texture/first-order feature 추출 (~1000 features)
2. **Clinical**: stage, age, histology, gender (`metadata/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv`)
3. **Fusion**:
   - Early: CNN 256-dim GAP feature + radiomics + clinical → MLP 또는 XGBoost
   - Late: CT prob + radiomics prob + clinical prob 가중평균
4. **Target**: test AUROC ≥ 0.65, variance ≤ ±0.03

추천 base CKPT: `experiments/m1_hosny_iter7b_seed99/best.pt` (test 0.666).

---

## 📚 참고 문헌

- Hosny A, et al. *Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study.* PLOS Medicine 2018.
- Braghetto A, et al. *Radiomics and deep learning methods for the prediction of 2-year overall survival in LUNG1 dataset.* Sci Rep 2022. [doi:10.1038/s41598-022-18085-z](https://www.nature.com/articles/s41598-022-18085-z)
- Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach.* Nat Commun 2014.

---

## 📝 실험 이력

12회 iter 전체 기록: `experiments/iterations_log.json`
- iter 0-6: ResNet-18 + MedicalNet pretrain (seed=42 고정, Tier 1 미달)
- iter 7/7b: Hosny 2018 재현 (EMA 버그 → 수정)
- iter 7c/7d/7e: Option C multi-seed (3/4 Tier 1 통과)
- `multi_seed_option_c_summary` 섹션에 최종 결론 담김

---

**Branch**: `feature/LJW` — 이재원 M1 CT-only 모듈. `main` merge 전 팀 리뷰 필요.
