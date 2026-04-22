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
| `feature/LJW` | CT 3D CNN (256-dim) | `seed99`, 293/64/63 |

세 브랜치가 **서로 다른 train/val/test 분할**을 썼기 때문에
“clinical AUROC 0.X vs radiomics AUROC 0.Y” 를 그대로 비교하면 split 효과가 섞여 해석이 불가능합니다.

따라서 fusion 직전 단계에서 **공정한 1:1:1 비교**를 하기 위해
- split: LJW `seed99` 의 **293 / 64 / 63** (label-stratified, seed=99) 을 공용으로 채택
- 튜닝: 각 모달마다 **Optuna** 로 MLP 하이퍼파라미터 재탐색
- 코드·결과·그림 전부: 이 폴더 하나에 자립(self-contained)하도록 배치

하여 이 폴더 결과만 봐도 **fusion 에 어떤 embedding 차원을 쓰면 되는지** 바로 알 수 있도록 했습니다.

---

## Split (공용 분할)

- 원본: `data/NSCLC-Radiomics-Lung1`, n=420
- **train = 293 / val = 64 / test = 63** (label-stratified, seed=99)
- 출처: LJW 브랜치의 `m1_hosny_iter7b_seed99` 파이프라인 → `results/ct/features/{train,val,test}.npz` 의 `pids`

세 모달 모두 **같은 PatientID 리스트**로 분할되므로 split 효과는 완전히 통제됩니다.

```
results/ct/features/
 ├─ train.npz   (pids=293)
 ├─ val.npz     (pids= 64)
 └─ test.npz    (pids= 63)
```

`scripts/common_split.py` 가 이 npz 3개로부터 PID/label 을 읽어
clinical/radiomics 스크립트에 주입합니다.

---

## Optuna 로 뽑은 dims (fusion 설계용)

각 모달의 MLP 인코더는 마지막 linear 가 **embed_dim** 을 만들고, 그 위에 단일 logit classifier 가 붙습니다.
Optuna 가 val AUROC 기준으로 최적화한 결과:

| Modality  | input_dim | hidden_dim | **embed_dim (fusion concat 용)** | dropout | best lr  | best epochs |
|-----------|-----------|------------|----------------------------------|---------|----------|-------------|
| Clinical  | 11        | 256        | **32**                           | 0.19    | 7.7e-3   | 97          |
| Radiomics | 15        | 128        | **256**                          | 0.25    | 5.5e-3   | 54          |
| CT        | 256*      | 32 (head)  | **256** (encoder 출력 그대로)    | 0.47    | 3.1e-3   | 159         |

\* CT 의 `input_dim=256` 은 LJW seed99 3D CNN encoder (`results/ct/best.pt`) 에서 뽑아낸 frozen feature.

**→ fusion stage 권장 concat 차원 = 32 (clinical) + 256 (radiomics) + 256 (ct) = 544**

Optuna 탐색 설정:
- Clinical : 50 trials, TPE sampler, val AUROC 최대화
- Radiomics: 100 trials, TPE sampler, val AUROC 최대화
- CT MLP head: 100 trials (LJW 원본 `optuna_mlp_head.py`, frozen encoder features 위)

---

## 결과 (LJW seed99 공용 split, test n=63)

| Modality  | Val AUROC (Optuna best) | Test AUROC (train only) | **Test AUROC (train+val retrain, headline)** |
|-----------|-------------------------|-------------------------|----------------------------------------------|
| Clinical  | 0.6569                  | 0.5484                  | 0.5795                                       |
| Radiomics | 0.7217                  | 0.6126                  | 0.5453                                       |
| CT        | 0.6549                  | 0.5547                  | **0.6337**                                   |

Headline 은 "train+val 합쳐서 best 하이퍼파라미터로 재학습 후 test" 기준.
시각화는 `figures/` 의 3장 (bar / ROC / confusion matrix) 참고.

해석 메모:
- Radiomics 는 val 에서 가장 강하지만 test trainval 에서는 오히려 떨어짐 → 소규모 val(64)에 과적합한 징후.
- CT 는 val 기준으로는 평범해도 trainval retrain 에서 가장 안정적.
- Clinical 은 전 구간 chance 근처 — 단독으로는 약하지만 fusion 에서 상보성이 기대됨.

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
│  └─ ct/                    참고용 CT 파이프라인 스냅샷 (LJW src/ 사본)
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
│  │  ├─ best.pt                     (LJW seed99 3D CNN encoder)
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
(현재 로컬 실행은 base conda env `C:/Users/dwd00/anaconda3/python.exe` 기준으로 검증됨.)

```bash
# 1) Clinical: Optuna(50 trials) → best 재학습(train-only) + trainval 재학습 → results/clinical/
python single_modal_baseline/scripts/eval_clinical.py

# 2) Radiomics: Optuna(100 trials) 동일 절차 → results/radiomics/
python single_modal_baseline/scripts/eval_radiomics.py

# 3) CT MLP head 는 이미 results/ct/optuna_mlp/ 에 포함됨
#    (재현하려면 LJW seed99 3D CNN 학습 → feature 추출 → MLP head Optuna 순.
#     scripts/ct/ 하위에 원본 스크립트가 참고용으로 들어있음.)

# 4) 3개 모달 요약 JSON 생성
python single_modal_baseline/scripts/summarize.py

# 5) 그림 3장 생성
python single_modal_baseline/scripts/visualize.py
```

모든 스크립트는 이 폴더 기준으로 경로가 잡혀 있어서 repo 어디에 두고 실행해도 동작합니다.

---

## Fusion 단계로 넘길 때 참고

- 세 모달 모두 **같은 293/64/63 PatientID 매핑**을 쓰고 있으므로 concat-level fusion 이 바로 가능.
- 권장 concat 차원: `32 (clin) + 256 (rad) + 256 (ct) = 544`.
- Headline 은 `train+val retrain` 기준 (CT MLP head 의 0.634 와 직접 비교 가능).
- fusion 에서는 각 encoder 의 **embed layer 출력** (classifier 직전) 을 뽑아 쓰면 됨:
  - clinical: `ClinicalBranch.encoder` 의 마지막 Linear → 32-dim
  - radiomics: `RadiomicsMLP.encoder` 의 마지막 ReLU 출력 → 256-dim
  - ct: `results/ct/features/*.npz` 의 256-dim frozen feature
