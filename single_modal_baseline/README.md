# single_modal_baseline

NSCLC 2-year survival prediction — **단일 모달(single-modal) 성능 비교**.
Clinical / Radiomics / CT 이미지 세 개 모달을 **동일한 split + 동일한 튜닝 프로토콜(Optuna)** 로 평가해
fusion 이전의 기준치(baseline)를 정리한 폴더입니다.

---

## 왜 이 폴더를 따로 만들었나

프로젝트는 3개의 feature 브랜치로 나뉘어 진행됐습니다.

| 브랜치 | 담당 모달 | 기존 split |
|---|---|---|
| `feature/CSJ` | Clinical (11 feat) | 자체 split |
| `feature/LJY` | Radiomics (15 LASSO feat) | 자체 split |
| `feature/LJW` | CT 3D CNN (256-dim) | 자체 split |

세 브랜치가 **서로 다른 train/val/test 분할**을 썼기 때문에
"clinical AUROC 0.X vs radiomics AUROC 0.Y" 를 그대로 비교하면 split 효과가 섞여 해석이 불가능합니다.

따라서 fusion 직전 단계에서 **공정한 1:1:1 비교**를 하기 위해
- split: **70 / 15 / 15 label-stratified** 공용 분할 (293 / 64 / 63) 채택
- 튜닝: 각 모달마다 **Optuna** 로 MLP 하이퍼파라미터 재탐색
- 코드·결과·그림 전부: 이 폴더 하나에 자립(self-contained)하도록 배치

하여 이 폴더 결과만 봐도 **fusion 에 어떤 embedding 차원을 쓰면 되는지** 바로 알 수 있도록 했습니다.

---

## Split (공용 분할)

- 원본: `data/NSCLC-Radiomics-Lung1`, n=420
- **train = 293 / val = 64 / test = 63** (label-stratified 70/15/15)
- 세 모달 모두 **같은 PatientID 리스트**로 분할되므로 split 효과는 완전히 통제됩니다.

```
results/ct/features/
 ├─ train.npz   (pids=293)
 ├─ val.npz     (pids= 64)
 └─ test.npz    (pids= 63)
```

`scripts/common_split.py` 가 이 npz 3개로부터 PID/label 을 읽어
clinical/radiomics 스크립트에 주입합니다. CT 쪽은 같은 분할로 3D CNN 이 학습된 후
256-dim GAP feature 를 뽑아 `results/ct/features/*.npz` 로 캐시해둔 상태입니다.

---

## Optuna 로 뽑은 dims (fusion 설계용)

각 모달의 MLP 인코더는 마지막 linear 가 **embed_dim** 을 만들고, 그 위에 단일 logit classifier 가 붙습니다.
Optuna 가 val AUROC 기준으로 최적화한 결과:

| Modality  | input_dim | hidden_dim | **embed_dim (fusion concat 용)** | dropout | best lr  | best epochs |
|-----------|-----------|------------|----------------------------------|---------|----------|-------------|
| Clinical  | 11        | 64         | **128**                          | 0.46    | 2.5e-4   | 81          |
| Radiomics | 15        | 128        | **256**                          | 0.16    | 2.0e-4   | 132         |
| CT        | 256\*     | 128 (head) | **256** (encoder 출력 그대로)    | 0.32    | 2.6e-4   | 148         |

\* CT 의 `input_dim=256` 은 3D CNN encoder (`results/ct/best.pt`) 에서 뽑아낸 frozen feature.
CT MLP head 는 `n_hidden=1` (128-dim 중간층) 이 최적이었음.

**→ fusion stage 권장 concat 차원 = 128 (clinical) + 256 (radiomics) + 256 (ct) = 640**

Optuna 탐색 설정:
- Clinical : 50 trials, TPE sampler, val AUROC 최대화
- Radiomics: 100 trials, TPE sampler, val AUROC 최대화
- CT MLP head: 200 trials, TPE sampler, frozen 256-dim feature 위 MLP head 탐색

---

## 결과 (공용 split, test n=63)

| Modality  | Val AUROC (Optuna best) | Test AUROC (train only) | **Test AUROC (train+val retrain, headline)** |
|-----------|-------------------------|-------------------------|----------------------------------------------|
| Clinical  | 0.6822                  | 0.5589                  | 0.5537                                       |
| Radiomics | 0.7247                  | 0.6095                  | 0.5726                                       |
| CT        | 0.6255                  | 0.5537                  | **0.6232**                                   |

Headline 은 "train+val 합쳐서 best 하이퍼파라미터로 재학습 후 test" 기준.
시각화는 `figures/` 의 3장 (bar / ROC / confusion matrix) 참고.

해석 메모:
- **CT 가 단독 1위** (0.623) — 다만 clinical/radiomics 와의 격차가 크지 않음.
- Radiomics 는 val 에서 가장 강함(0.725) 이나 test trainval 에서는 떨어짐 → 소규모 val(64)에 과적합한 징후.
- Clinical 은 전 구간 chance 근처 — 단독으로는 약하지만 fusion 에서 상보성이 기대됨.
- 세 모달이 **비슷한 수준 (0.55 ~ 0.62)** 으로 모여 있어서 fusion 의 상보성 효과를 테스트하기 좋은 조건.

---

## 폴더 구조

```
single_modal_baseline/
├─ README.md                ← 이 파일
├─ data/
│  ├─ clinical_encoded.csv           (420 patients × 11 feat + label, CSJ 전처리 결과)
│  └─ radiomics_final_features.csv   (420 patients × 15 LASSO feat + label, LJY 전처리 결과)
├─ scripts/
│  ├─ common_split.py        공용 split PID/label 로더
│  ├─ eval_clinical.py       Clinical MLP Optuna + 최종 평가
│  ├─ eval_radiomics.py      Radiomics MLP Optuna + 최종 평가
│  ├─ summarize.py           3개 모달 결과 → summary.json 통합
│  ├─ visualize.py           bar / ROC / confusion-matrix 3종 플롯
│  └─ ct/                    참고용 CT 파이프라인 스냅샷
│     ├─ ct_preprocess.py    DICOM → npz 전처리
│     ├─ dataset.py          3D CT Dataset
│     ├─ cnn3d.py · hosny_cnn.py   Hosny 2018 3D CNN
│     ├─ train.py            encoder 학습 엔트리
│     ├─ extract_features.py 256-dim feature 추출
│     └─ optuna_mlp_head.py  frozen feature 위 MLP head Optuna
├─ results/
│  ├─ clinical/   best.pt, results.json, test_predictions.csv
│  ├─ radiomics/  best.pt, results.json, test_predictions.csv
│  ├─ ct/
│  │  ├─ best.pt                     (3D CNN encoder)
│  │  ├─ features/{train,val,test}.npz   (256-dim frozen features + pids + labels)
│  │  ├─ history.json · test_summary.json · test_predictions.csv   (3D CNN+TTA 원본 결과)
│  │  └─ optuna_mlp/                 ← headline 에 쓰는 MLP head 결과
│  │     ├─ best_optuna_mlp.pt
│  │     ├─ optuna_mlp_summary.json
│  │     └─ test_predictions_optuna_mlp.csv
│  └─ summary.json           세 모달 합쳐놓은 metric JSON
└─ figures/
   ├─ comparison_bar.png     Val / Test-trainonly / Test-trainval grouped bar
   ├─ roc_curves.png         test 3모달 ROC 오버레이 (trainval)
   └─ confusion_matrices.png 1×3 confusion matrix (threshold=0.5, trainval)
```

---

## 재현 방법

필요 패키지: `torch`, `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`.

```bash
# 1) Clinical: Optuna(50 trials) → best 재학습(train-only) + trainval 재학습 → results/clinical/
PYTHONPATH=. python single_modal_baseline/scripts/eval_clinical.py

# 2) Radiomics: Optuna(100 trials) 동일 절차 → results/radiomics/
PYTHONPATH=. python single_modal_baseline/scripts/eval_radiomics.py

# 3) CT MLP head (200 trials) → results/ct/optuna_mlp/
PYTHONPATH=. python single_modal_baseline/scripts/ct/optuna_mlp_head.py \
    --ckpt single_modal_baseline/results/ct/best.pt \
    --features single_modal_baseline/results/ct/features \
    --n-trials 200 --n-jobs 4 \
    --out single_modal_baseline/results/ct/optuna_mlp
#    (3D CNN 자체는 이미 학습되어 results/ct/best.pt + features/*.npz 에 포함됨.
#     encoder 부터 다시 돌리려면 scripts/ct/ 의 train.py / extract_features.py 참고.)

# 4) 3개 모달 요약 JSON 생성
python single_modal_baseline/scripts/summarize.py

# 5) 그림 3장 생성
python single_modal_baseline/scripts/visualize.py
```

모든 스크립트는 이 폴더 기준으로 경로가 잡혀 있어서 repo root 에서 `PYTHONPATH=.` 만 설정하면 동작합니다.

---

## Fusion 단계로 넘길 때 참고

- 세 모달 모두 **같은 293/64/63 PatientID 매핑**을 쓰고 있으므로 concat-level fusion 이 바로 가능.
- 권장 concat 차원: `128 (clin) + 256 (rad) + 256 (ct) = 640`.
- Headline 은 `train+val retrain` 기준 (CT 단독 0.623 이 현재 상한선).
- fusion 에서는 각 encoder 의 **embed layer 출력** (classifier 직전) 을 뽑아 쓰면 됨:
  - clinical: `ClinicalBranch.encoder` 의 마지막 Linear → 128-dim
  - radiomics: `RadiomicsMLP.encoder` 의 마지막 ReLU 출력 → 256-dim
  - ct: `results/ct/features/*.npz` 의 256-dim frozen feature
