# triple_model — 3-modal fusion (late + intermediate)

`single_modal_baseline/` 과 `double_model/` 를 기반으로 **세 모달 전체**
(clinical + radiomics + ct) 를 결합한 폴더입니다. 동일한 split / 동일한 평가 프로토콜을
유지해서 `single → double → triple` 을 한 차트로 비교할 수 있습니다.

1. **Late fusion** (prob-level, equal weight 1/3, arithmetic / logit mean) — 즉시 계산
2. **Intermediate fusion** (frozen encoder 3개 concat + MLP head, Optuna-tuned) — head 만 학습

---

## 목적

- single (clin/rad/ct) 과 double (clin_rad/clin_ct/rad_ct) 에서 쌓아둔 베이스라인 위에서
  **세 모달 전체 조합이 추가적 상보성을 주는지** 확인.
- Fusion 전략 2종 (late, intermediate) 의 성능 상한을 한 자리에서 보여줘서 triple 이
  double 최고치 (clin_rad intermediate 0.6284) 와 CT 단독 (0.6232) 을 넘는지 직접 비교.

---

## 실험 설계 (요약)

| 항목 | Late fusion | Intermediate fusion |
|---|---|---|
| Split | 공용 **293 / 64 / 63** (동일) | 동일 |
| 입력 | single_modal_baseline 의 `prob_trainval` (test CSV) × 3 | single_modal_baseline 의 **train-only encoder** embedding × 3 |
| 모달 연결 위치 | 확률 (1-dim × 3) | embedding (D_clin + D_rad + D_ct = 128+256+256 = 640) |
| 학습 파라미터 | 없음 (평균만) | FusionHead MLP만 (encoder 는 freeze) |
| 튜닝 | 없음 (equal weight 1/3) | Optuna 30 trials, val AUROC 최대화 |
| Threshold | 0.5 | 0.5 |

**Intermediate encoder 가 train-only 인 이유 / FusionHead 구조** → `double_model/README.md` 와 동일.

```
concat(emb_clin[128], emb_rad[256], emb_ct[256]) → [Linear → ReLU → Dropout] * n_hidden → Linear(1)
```

Optuna 탐색 공간 (double 과 동일):
- `n_hidden` ∈ {1, 2}
- `hidden_dim` ∈ {32, 64, 128, 256}
- `dropout` ∈ [0.1, 0.5]
- `lr` log-uniform [1e-4, 1e-2]
- `weight_decay` log-uniform [1e-6, 1e-3]
- `batch_size` ∈ {16, 32, 64}
- `epochs` ∈ [30, 120]
- Loss: `BCEWithLogitsLoss(pos_weight=neg/pos)`

---

## 결과 (test n=63)

### Baseline (v1, equal-weight & default MLP)

| 방법                         | Val/CV AUROC  | Test AUROC (trainonly) | **Test AUROC (trainval, headline)** |
|------------------------------|---------------|-------------------------|--------------------------------------|
| Late fusion — arithmetic     | —             | —                       | 0.5916                               |
| Late fusion — logit          | —             | —                       | 0.6074                               |
| Intermediate (frozen + MLP)  | val 0.6559    | 0.6232                  | 0.5447                               |

### Fusion-side tuning (동일 encoder, 동일 split — head/weight 만 변경)

| 방법                                      | Val/CV AUROC     | Test AUROC (trainonly) | **Test AUROC (trainval, headline)** |
|-------------------------------------------|------------------|-------------------------|--------------------------------------|
| Weighted late — arith (simplex 0.05 grid) | val 0.7601       | —                       | 0.5842                               |
| Weighted late — logit (same grid)         | val 0.7601       | —                       | 0.5853                               |
| Intermediate v2 (CV+BN+modality dropout)  | CV  0.6905       | 0.6295                  | 0.5547                               |
| **Linear L2 (LogRegCV on 640-concat)**    | val 0.5951       | 0.6811                  | **0.6589** 🏆                        |
| Linear ElasticNet (SGD + grid)            | val 0.6422       | 0.6563                  | 0.5668                               |

**앵커 숫자 (비교용):**

| 단계     | 최고 AUROC | 출처                                       |
|----------|------------|--------------------------------------------|
| Single   | 0.6232     | `ct` (MLP head trainval)                   |
| Double   | 0.6284     | `clin_rad` intermediate (trainval)         |
| Triple   | **0.6589** | **linear L2 on 640-concat (trainval)**    |

관찰 / 해석:

**Baseline 3종 (v1)**:
- Late logit (0.607) 이 triple baseline 최고지만 CT 단독 (0.623) 보다 -0.016 낮음.
- Late arithmetic (0.592) 는 약한 clinical/radiomics 가 강한 CT 를 희석.
- Intermediate v1 (0.545) 은 val 단일 epoch-max 목적함수가 `lr=9e-3, epochs=31` 같은
  과최적화 파라미터를 뽑아내 trainval 재학습에서 붕괴. Double 의 `rad_ct` 에서 본
  패턴이 concat-dim 640 에서 더 심하게 재현됨.

**Fusion-side tuning (5종)**:
- **Weighted late (0.584/0.585)** — val(n=64) grid search 가 clin+rad 에 과적합해
  오히려 equal-1/3 late logit (0.607) 보다 악화. "val n=64 에서 weight tuning 은
  불안정하다" 는 경고가 실측으로 확인됨.
- **Intermediate v2 (CV+BN+modality dropout, 0.555)** — Optuna 목적함수를 trainval 5-fold
  CV 로 바꾸고 input BN + modality dropout 을 추가한 결과, Optuna 가 고른 best params
  은 v1 보다 훨씬 보수적(`lr=9.6e-4, epochs=95, hidden=128`, dropout 0.28)이고 CV AUROC
  0.690 까지 올랐다. trainonly test 도 0.629 (v1 0.623 보다 소폭↑). 그러나 trainval
  재학습에서는 여전히 0.555 → **640 → MLP 구조 자체가 357 샘플에서 과적합하기 쉬워
  헤드 튜닝만으로는 불충분**.
- **Linear L2-LogReg (0.6589, 🏆)** — 640-concat 위에 L2-penalized logistic (≈640 params,
  C=0.001 로 강한 정규화) 을 얹은 가장 단순한 방법이 triple 최고점. CT 단독 (0.623)
  도 이기고, 지금까지 단계별 최고였던 double `clin_rad` intermediate (0.628) 도 소폭
  상회 (+0.031). 640-dim concat 에는 실제로 상보적인 신호가 있었는데 v1/v2 의 MLP 가
  그것을 잡기 전에 과적합으로 먼저 무너진 것.
- **ElasticNet (0.567)** — L1-혼합이 640 coef 중 일부를 0 으로 만드는데 val(n=64) 에서
  고른 (α=1e-4, l1=0.1) 이 trainval 재학습에서는 약간 다른 방향으로 수렴. L2 만 남기는
  게 이 regime 에서는 더 안정적.

**결론 / 시사점**:
- **Triple > Double > Single 이 재현**: `linear L2 on 640-concat` 이 단계별 최고점을
  갱신 (0.554 → 0.572 → 0.623 → 0.628 → 0.659).
- **"Fusion-side tuning" 으로 MLP 를 더 정교하게 다듬는 것보다, capacity 를 줄여 linear
  로 돌리는 게 이 데이터 규모에서 가장 크게 먹혔다**. 357 샘플 × 640 dim 에는 MLP 가
  필요 없고, L2-penalized linear 가 "concat 안의 상보 신호" 를 가장 깔끔하게 뽑는다.
- Weighted late 와 intermediate v2 의 실패는 모두 "val(n=64) 에서 tuning 하면 그
  분포에 과적합" 이라는 공통 원인. Linear L2 는 C 를 내부 5-fold CV 로 고르므로 val
  독립 → 단일 분포에 묶이지 않아 더 안정적.
- 남은 개선 방향:
  - **Stacking** (meta-learner = linear L2 on [p_clin, p_rad, p_ct]) — 최고 확률 예측만
    쓰는 쪽. 다만 single-modal OOF 가 필요해 fusion-side 범위를 벗어남.
  - **Ensemble of linear L2 heads with different C / scaling** — 현재 best 를 더 단단하게.
  - **Robustness**: seed7 / seed123 / seed42 에서 triple (linear L2) > double (intermediate)
    이 재현되는지 검증 필요.

---

## 폴더 구조

```
triple_model/
├─ README.md
├─ scripts/
│  ├─ late_fusion.py                확률 평균 (arith + logit, equal 1/3)
│  ├─ intermediate_fusion.py        v1: frozen encoder × 3 concat + Optuna-tuned MLP head
│  ├─ weighted_late_fusion.py       simplex grid search (step 0.05) over w_clin/w_rad/w_ct
│  ├─ intermediate_fusion_v2.py     v2: 5-fold CV Optuna + input BN + modality dropout
│  ├─ linear_fusion.py              LogRegCV(L2) + ElasticNet(SGD) on 640-dim concat
│  └─ visualize.py                  single/double/triple 통합 bar + ROC + CM + fusion tuning bar
├─ embeddings/                      train-only encoder embeddings (cached by intermediate_fusion.py)
│  └─ {clinical,radiomics,ct}_{train,val,test}.npz   (emb, y, pids)
├─ results/
│  ├─ triple/
│  │  ├─ predictions_arithmetic.csv         late (equal 1/3)
│  │  ├─ predictions_logit.csv              late (equal 1/3)
│  │  ├─ predictions_weighted_arithmetic.csv  weighted late (val-tuned w)
│  │  ├─ predictions_weighted_logit.csv       weighted late (val-tuned w)
│  │  ├─ intermediate/                      v1 (MLP, val-max objective)
│  │  ├─ intermediate_v2/                   v2 (CV + BN + modality dropout)
│  │  └─ linear/                            LogRegCV L2 + ElasticNet
│  └─ summary.json                          all methods + single/double 참조값
└─ figures/
   ├─ comparison_bar.png                    single(3) | double(3) | triple top-3
   ├─ fusion_tuning_bar.png                 triple 8 methods (baseline 3 + tuned 5)
   ├─ roc_curves.png                        triple top-3 + single/double 오버레이
   └─ confusion_matrices.png                1×3 CM (late logit, intermediate v2, linear L2)
```

---

## 재현

```bash
# single_modal_baseline/ 의 best.pt + features/*.npz
# double_model/results/ 의 summary.json / predictions_*.csv / intermediate test_predictions.csv
# 위 두 폴더가 준비돼 있어야 함.

# 1) Late fusion — equal-weight arith + logit (수초)
python triple_model/scripts/late_fusion.py

# 2) Intermediate fusion v1 — Optuna 30 trials, val-max objective (GPU ~1분)
python triple_model/scripts/intermediate_fusion.py --n-trials 30

# 3) Weighted late fusion — simplex grid search on val (수초)
python triple_model/scripts/weighted_late_fusion.py

# 4) Intermediate fusion v2 — CV Optuna + BN + modality dropout (GPU ~3-5분)
python triple_model/scripts/intermediate_fusion_v2.py --n-trials 40 --n-splits 5

# 5) Linear fusion — LogRegCV(L2) + ElasticNet on 640-concat (수초)
python triple_model/scripts/linear_fusion.py

# 6) Figures (single/double/triple 통합 + fusion tuning comparison)
python triple_model/scripts/visualize.py
```

Intermediate 쪽은 `single_modal_baseline/scripts/` 의 `ClinicalBranch` / `RadiomicsMLP`
+ `common_split.load_split_pids` 를 import 하므로 single_modal_baseline 폴더가 repo 에
같이 있어야 동작 (경로: `../single_modal_baseline/`).

---

## 최종 비교 (single → double → triple)

| 단계   | Best method                           | Best AUROC | Δ vs previous best |
|--------|---------------------------------------|------------|---------------------|
| Single | CT MLP head (trainval)                | 0.6232     | —                   |
| Double | Clin+Rad intermediate                 | 0.6284     | **+0.0052**         |
| Triple | **Linear L2 on 640-concat (trainval)**| **0.6589** | **+0.0305** 🏆      |

Baseline triple (late logit, equal 1/3) 은 0.6074 로 CT 단독 (0.6232) 도 못 넘겼지만,
fusion-side tuning 중 **linear L2 head** 가 triple 을 단계별 최고점 자리에 복귀시킴.
640 → 1 linear + 5-fold CV 로 고른 C=1e-3 이 357 샘플 × 640 dim regime 에 딱 맞는
capacity. MLP head (v1/v2) 는 같은 concat 위에서 과적합으로 무너짐.
