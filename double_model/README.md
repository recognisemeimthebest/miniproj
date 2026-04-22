# double_model — 2-modal fusion (late + intermediate)

`single_modal_baseline/` 을 기반으로 **두 모달씩 조합한 3쌍** (clin+rad, clin+ct, rad+ct)
에 대해 두 가지 fusion 전략을 돌리고 결과를 같이 기록한 폴더입니다.

1. **Late fusion** (prob-level, equal weight, arithmetic / logit mean) — 즉시 계산
2. **Intermediate fusion** (frozen encoder + concat + MLP head, Optuna-tuned) — head 만 학습

다음 단계 `triple_model/` 에서 **세 모달 전체**를 같은 두 전략으로 돌려서
`single → double → triple` 개선 폭을 직접 비교할 수 있도록 스키마를 통일했습니다.

---

## 목적

- single_modal_baseline 의 단일 모달 AUROC (`clin=0.580, rad=0.545, ct=0.634`) 대비
  **두 모달 조합이 상보적 신호를 주는지** 확인.
- fusion 전략 2종 (late, intermediate) 에 대해 상한을 각각 보여서 triple 에서
  어떤 전략을 집중해야 할지 방향을 잡음.

---

## 실험 설계 (요약)

| 항목 | Late fusion | Intermediate fusion |
|---|---|---|
| Split | LJW seed99 **293 / 64 / 63** (동일) | 동일 |
| 입력 | single_modal_baseline 의 `prob_trainval` (test CSV) | single_modal_baseline 의 **train-only encoder** embedding |
| 모달 연결 위치 | 확률 (1-dim × 2) | embedding (D_A + D_B dim) |
| 학습 파라미터 | 없음 (평균만) | FusionHead MLP만 (encoder 는 freeze) |
| 튜닝 | 없음 (equal weight) | Optuna 30 trials, val AUROC 최대화 |
| Threshold | 0.5 | 0.5 |
| Seed | — | 99 (모델 init + Optuna sampler) |

**Intermediate 의 encoder 가 train-only 인 이유**
: trainval encoder 는 val 로 학습되었기 때문에, 그걸로 val embedding 을 뽑으면
  Optuna 가 val AUROC 를 올리는 과정에서 encoder 의 val 정보가 누수됨.
  Train-only encoder 는 val 을 한 번도 안 봤으므로 val 이 honest 한 eval 이 됨.
  단점: 최종 trainval 재학습에서도 encoder 는 train-only 로 고정 → baseline 의
  trainval 재학습(end-to-end) 보다 encoder 가 약할 수 있음. 그래도 헤드가 더
  자유롭게 조합할 수 있는 구조적 이점이 더 크다고 판단.

**Intermediate FusionHead 구조**
```
concat(emb_A, emb_B) → [Linear → ReLU → Dropout] * n_hidden → Linear(1)
```
Optuna 탐색 공간:
- `n_hidden` ∈ {1, 2}
- `hidden_dim` ∈ {32, 64, 128, 256}
- `dropout` ∈ [0.1, 0.5]
- `lr` log-uniform [1e-4, 1e-2]
- `weight_decay` log-uniform [1e-6, 1e-3]
- `batch_size` ∈ {16, 32, 64}
- `epochs` ∈ [30, 120]
- Loss: `BCEWithLogitsLoss(pos_weight=neg/pos)` (clinical/radiomics 와 동일)

---

## 결과 (test n=63)

| Pair     | input_dim | Late arith | Late logit | **Intermediate (headline)** | Optuna val (inter) |
|----------|-----------|------------|------------|------------------------------|---------------------|
| clin_rad | 32+256=288| 0.5653     | 0.5674     | **0.5947**                   | 0.7196             |
| clin_ct  | 32+256=288| 0.6274     | 0.6168     | 0.6095                       | 0.6670             |
| rad_ct   | 256+256=512| 0.5821    | 0.5516     | **0.6100**                   | 0.7571             |

Single-modal 참고 (train+val retrain test AUROC):
- clinical : 0.5795
- radiomics: 0.5453
- **ct      : 0.6337 ← 현 최강 단독**

관찰:
- **어떤 fusion pair 도 CT 단독(0.634)을 아직 못 넘음.** 그래도 intermediate 가 late 대비
  `clin_rad` (+0.029) 와 `rad_ct` (+0.028) 에서 개선.
- `clin_ct` 는 예외적으로 late (0.627) 가 intermediate (0.610) 보다 더 좋음 →
  clinical 의 32-dim embedding 이 CT 의 256-dim 신호를 head 에서 희석시킨 듯.
  → triple 에서도 clinical embed_dim 을 키우는(예: 64 또는 128) 재튜닝 고려 여지.
- Optuna val 과 test trainval 의 gap 이 0.1+ → val 에 살짝 과적합. n=64 의 한계.
- Intermediate 를 2/3 pair 에서 본 개선(~0.03) 이 "intermediate fusion 의 기대이득 +0.03~0.05"
  예상치와 대체로 일치.

시사점:
- 현재 규모(train 293, val 64)에서는 **intermediate fusion 만으로 CT 단독을 큰 폭 초과하기
  는 힘듦**. triple 에서 세 모달 동시 사용 + 같은 Optuna 튜닝이 현실적 다음 단계.
- 추후 개선안:
  - weighted late fusion (val 에서 w 를 grid search, train-only 모델의 val 예측 필요)
  - gated / attention fusion
  - end-to-end 재학습 (encoder 까지 같이 훈련) — 과적합 위험 큼

---

## 폴더 구조

```
double_model/
├─ README.md
├─ scripts/
│  ├─ late_fusion.py           3 pair × (arith, logit) 확률 평균
│  ├─ intermediate_fusion.py   frozen encoder → concat → Optuna-tuned MLP head
│  └─ visualize.py             bar (late arith/logit + intermediate) / ROC / CM
├─ embeddings/                 intermediate 시 캐시된 train-only encoder embeddings
│  └─ {clinical,radiomics,ct}_{train,val,test}.npz   (emb, y, pids)
├─ results/
│  ├─ clin_rad/
│  │  ├─ predictions_arithmetic.csv   (late, equal weight)
│  │  ├─ predictions_logit.csv        (late, equal weight)
│  │  └─ intermediate/
│  │     ├─ best.pt                   (trainonly + trainval FusionHead state dicts)
│  │     ├─ test_predictions.csv      (prob_trainonly, prob_trainval, pred@0.5)
│  │     └─ results.json              (best params, val/test AUROC, metrics)
│  ├─ clin_ct/    (동일 구조)
│  ├─ rad_ct/     (동일 구조)
│  └─ summary.json                    late + intermediate 통합
└─ figures/
   ├─ comparison_bar.png               pair × (late arith / late logit / inter) + single 앵커
   ├─ roc_curves.png                   pair best ROC + single ROC overlay
   └─ confusion_matrices.png           intermediate 기준 1x3 CM (threshold=0.5)
```

---

## 재현

```bash
# single_modal_baseline/ 의 best.pt + features/*.npz 가 있어야 함

# 1) Late fusion (수초)
python double_model/scripts/late_fusion.py

# 2) Intermediate fusion — Optuna 30 trials × 3 pair (GPU 기준 2~3분)
python double_model/scripts/intermediate_fusion.py --n-trials 30

# 3) Figures
python double_model/scripts/visualize.py
```

Intermediate 쪽은 `single_modal_baseline/scripts/` 의 `ClinicalBranch` / `RadiomicsMLP` 클래스
+ `common_split.load_split_pids` 를 import 하므로 single_modal_baseline 폴더가 repo 에 같이
있어야 동작 (경로: `../single_modal_baseline/`).

---

## triple_model 단계에서 기록해둘 것

(나중에 세 모달 합칠 때 이 폴더와 1:1 비교 가능하도록 동일하게 남기면 좋음)

- `split`: 293/64/63 (동일 사용 — seed99)
- `encoder_source`: train-only / trainval / joint (어떤 버전을 썼는지)
- `method`: late (arith/logit/weighted) / intermediate (frozen concat + MLP) / early / joint
- `modalities`: 항상 `["clinical","radiomics","ct"]` 순서
- `test_auroc_trainval`: headline (이 폴더와 1:1 비교)
- `optuna_best_val_auroc`: val side (과적합 여부 확인용)
- 앵커 숫자:
  - single best (CT) : **0.6337**
  - double best (현재 intermediate rad_ct) : **0.6100**
  - triple 목표 : > 0.6337 (CT 단독 초과) 가 의미 있는 결과

이 규약을 유지하면 `summary.json` 세 개(single / double / triple) 를 한 스크립트로
읽어 통합 bar chart 를 그릴 수 있습니다.
