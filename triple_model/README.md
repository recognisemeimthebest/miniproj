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

| 방법                         | Val AUROC (trainonly) | Test AUROC (trainonly) | **Test AUROC (trainval, headline)** |
|------------------------------|------------------------|-------------------------|--------------------------------------|
| Late fusion — arithmetic      | —                      | —                       | 0.5916                               |
| Late fusion — logit           | —                      | —                       | **0.6074**                           |
| Intermediate (frozen + MLP)  | 0.6559                 | 0.6232                  | 0.5447                               |

**앵커 숫자 (비교용):**

| 단계     | 최고 AUROC | 출처                                       |
|----------|------------|--------------------------------------------|
| Single   | **0.6232** | `ct` (MLP head trainval)                   |
| Double   | **0.6284** | `clin_rad` intermediate (trainval)         |
| Triple   | **0.6074** | `triple` late logit mean (본 폴더, trainval)|

관찰 / 해석:
- **Triple 에서는 fusion 이 오히려 single/double 최고치를 못 넘음**.
  - Late logit (0.607) 이 triple 최고지만 CT 단독 (0.623) 보다 -0.016 낮음.
  - Late arithmetic (0.592) 는 세 모달 확률을 평균내느라 약한 clinical/radiomics 가
    강한 CT 신호를 희석시킨 결과로 보임.
- **Intermediate triple (trainval = 0.545) 은 확연히 나쁨**.
  - trainonly head 는 0.6232 로 CT 단독과 동률 → 640-dim concat 에서
    **헤드가 결국 CT 채널에 의존하도록 학습** (clinical/radiomics 는 noise 취급).
  - 그런데 val 지운 trainval 재학습에서 `lr=9.0e-3, epochs=31` 조합이 작은 train 에
    과적합해 test 는 0.545 로 붕괴 (혼동행렬: 다수 클래스 0 쪽으로 쏠림).
  - double 의 `rad_ct` 에서 봤던 "Optuna val 은 0.7 인데 trainval 재학습에서 붕괴"
    패턴이 triple 에도 재현됨 — concat dim 이 커질수록 head 과적합 위험.
- Optuna val = 0.656 vs test trainonly = 0.623 차이는 **val(n=64) 자체가 noisy** 하기
  때문. Triple 에서의 val-test gap 이 double/single 과 비슷한 크기 (~0.03) → fusion
  자체가 과적합한 게 아니라 val 분포의 한계.

시사점:
- **현재 데이터 규모(train 293, val 64, test 63) 에서 triple fusion 만으로는 single(CT) 을
  넘기지 못함**. fusion 상한은 double (clin_rad intermediate 0.6284) 이 잡고 있음.
- 세 모달이 상보적이라면 triple 은 double 을 소폭이라도 올려야 하는데 그렇지 못한
  건, clinical 과 radiomics 의 신호 품질 (둘 다 test trainval 0.55~0.57) 이 CT 를
  희석시키는 수준에 머무르기 때문으로 해석됨.
- 추후 개선안:
  - **Weighted late fusion**: val 에서 `w_clin, w_rad, w_ct` grid search → CT 쪽에
    높은 weight 를 자동 할당하면 late arithmetic 의 희석 문제가 완화될 것.
  - **Gated / attention fusion**: concat 대신 modality-wise gating 으로 CT 의존을
    학습적으로 강화.
  - **Learnable weights 재학습**: train-only encoder 까지 같이 풀어주는 end-to-end
    (과적합 위험 크므로 데이터 확장 or 강한 정규화 필수).
  - **Robustness check**: 다른 split seed 에서도 triple ≥ double 이 재현 안 되면
    "현 데이터 규모의 한계" 로 결론 짓는 쪽이 정직.

---

## 폴더 구조

```
triple_model/
├─ README.md
├─ scripts/
│  ├─ late_fusion.py           확률 평균 (arith + logit, equal 1/3)
│  ├─ intermediate_fusion.py   frozen encoder × 3 → concat → Optuna-tuned MLP head
│  └─ visualize.py             single/double/triple 통합 bar + ROC + CM
├─ embeddings/                 intermediate 시 캐시된 train-only encoder embeddings
│  └─ {clinical,radiomics,ct}_{train,val,test}.npz   (emb, y, pids)
├─ results/
│  ├─ triple/
│  │  ├─ predictions_arithmetic.csv   (late, equal 1/3)
│  │  ├─ predictions_logit.csv        (late, equal 1/3)
│  │  └─ intermediate/
│  │     ├─ best.pt                   (trainonly + trainval FusionHead state dicts)
│  │     ├─ test_predictions.csv      (prob_trainonly, prob_trainval, pred@0.5)
│  │     └─ results.json              (best params, val/test AUROC, metrics)
│  └─ summary.json                    late + intermediate + single/double 참조값
└─ figures/
   ├─ comparison_bar.png               single(3) | double(3) | triple(3 methods) 통합
   ├─ roc_curves.png                   triple late arith + intermediate + single + double 오버레이
   └─ confusion_matrices.png           1×2 CM (late arith, intermediate; threshold=0.5)
```

---

## 재현

```bash
# single_modal_baseline/ 의 best.pt + features/*.npz
# double_model/results/ 의 summary.json / predictions_*.csv / intermediate test_predictions.csv
# 위 두 폴더가 준비돼 있어야 함.

# 1) Late fusion (수초)
python triple_model/scripts/late_fusion.py

# 2) Intermediate fusion — Optuna 30 trials (GPU 기준 ~1분)
python triple_model/scripts/intermediate_fusion.py --n-trials 30

# 3) Figures (single/double/triple 통합)
python triple_model/scripts/visualize.py
```

Intermediate 쪽은 `single_modal_baseline/scripts/` 의 `ClinicalBranch` / `RadiomicsMLP`
+ `common_split.load_split_pids` 를 import 하므로 single_modal_baseline 폴더가 repo 에
같이 있어야 동작 (경로: `../single_modal_baseline/`).

---

## 최종 비교 (single → double → triple)

| 단계   | Best method              | Best AUROC | Δ vs previous best |
|--------|--------------------------|------------|---------------------|
| Single | CT MLP head (trainval)   | 0.6232     | —                   |
| Double | Clin+Rad intermediate    | 0.6284     | **+0.0052**         |
| Triple | Late logit (equal 1/3)   | 0.6074     | **-0.0210** (퇴보)  |

본 실험에서는 **double 이 triple 보다 잘 하는 상황**. triple fusion 을 살리려면
weighted late (val 에서 CT 쪽 가중치 자동 학습) 또는 attention-gated fusion 이 필요.
