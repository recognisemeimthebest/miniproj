# double_model — 2-modal late fusion

`single_modal_baseline/` 의 각 모달 test 확률을 입력으로,
**두 모달씩 조합한 3쌍**에 대해 **late fusion (확률 평균)** 을 평가한 폴더입니다.

다음 단계(`triple_model/` 에서 세 모달 합치는 실험)와 **직접 비교 가능한 스키마**로
결과를 기록하는 것이 1차 목적이므로, 실험 조건을 **최대한 단순**하게 고정했습니다.

---

## 목적

- single_modal_baseline 의 단일 모달 AUROC (`clin=0.580, rad=0.545, ct=0.634`) 대비
  **두 모달 조합이 상보적 신호를 주는지** 확인.
- triple fusion 에서 같은 평가 프로토콜을 재사용하여
  `single → double → triple` 차이를 한눈에 보도록 **스키마를 통일**.

---

## 실험 설계 (요약)

| 항목 | 값 |
|---|---|
| Split | LJW seed99 **293 / 64 / 63** (label-stratified) — single_modal_baseline 과 동일 |
| 입력 확률 | single_modal_baseline 의 **train+val retrain** test 예측 (headline 일치) |
| 조합 | `clin_rad`, `clin_ct`, `rad_ct` (3 pair) |
| 가중치 | **equal weight (0.5, 0.5)** — val 튜닝 없이 honest baseline |
| 평균 방식 | (A) **arithmetic**: `(p_A + p_B) / 2` <br>(B) **logit**: `σ((logit p_A + logit p_B) / 2)` |
| Threshold | 0.5 (single_modal_baseline 와 동일) |
| Seed | 계산 결정론적 — pure arithmetic on saved probs, 별도 seed 없음 |

왜 val 로 weight 튜닝을 안 했나:
- 확률 재사용(이미 trainval 재학습된 값)으로 val 은 이미 학습에 포함됨 → val 에서 weight 를
  고르면 leakage.
- 엄밀한 weight 튜닝은 train-only 모델의 val 예측을 새로 뽑아야 가능 (향후 작업으로 보류).

---

## 결과 (test n=63)

| Pair     | modalities            | Arithmetic AUROC | Logit AUROC | vs best single |
|----------|-----------------------|------------------|-------------|----------------|
| clin_rad | clinical + radiomics  | 0.5653           | 0.5674      | clin 0.5795 ↓  |
| clin_ct  | clinical + ct         | **0.6274**       | 0.6168      | ct 0.6337 ↓    |
| rad_ct   | radiomics + ct        | 0.5821           | 0.5516      | ct 0.6337 ↓    |

Single-modal 참고 (train+val retrain test AUROC):
- clinical : 0.5795
- radiomics: 0.5453
- **ct      : 0.6337 ← 현 최강 단독**

관찰:
- **어떤 pair 도 CT 단독(0.634)을 못 넘음.** Equal-weight 로 약한 신호(clin/rad)가 CT 를
  오히려 희석시키는 전형적 현상.
- `clin_ct` 가 그나마 CT 에 근접(0.627) — clinical 이 CT 를 크게 망치지는 않음.
- `clin_rad` 는 단일 clinical(0.580) 보다도 아래 → 두 모달이 비슷한 방향의 오류를 공유.
- Arithmetic vs Logit: 큰 차이 없음 (max Δ ≈ 0.03). 극단 확률이 많지 않은 test set.

시사점:
- 단순 equal-weight late fusion 만으로는 여기서 이득 없음 →
  (1) CT 쪽에 더 큰 가중치 주는 weighted late fusion, 또는
  (2) embedding 레벨에서 합치는 intermediate fusion (concat + MLP) 이 필요.
- **triple fusion** 에서도 equal-weight 만 쓰면 비슷한 한계가 예상되므로, 이 숫자들을
  삼중 결과와 비교해서 "3 모달을 전부 쓰는 게 정말 도움이 되는가" 를 판단할 근거로 사용.

---

## 폴더 구조

```
double_model/
├─ README.md               ← 이 파일
├─ scripts/
│  ├─ late_fusion.py       3 pair × (arith, logit) 계산, summary.json 출력
│  └─ visualize.py         bar / ROC / confusion matrix 3종 플롯
├─ results/
│  ├─ clin_rad/
│  │  ├─ predictions_arithmetic.csv   (pid, y_true, prob_A, prob_B, prob_fused, pred@0.5)
│  │  └─ predictions_logit.csv
│  ├─ clin_ct/   (동일 구조)
│  ├─ rad_ct/    (동일 구조)
│  └─ summary.json         triple fusion 이 같은 스키마로 붙일 수 있는 형태
└─ figures/
   ├─ comparison_bar.png   3 pair × 2 rule + 3 single baseline lines
   ├─ roc_curves.png       pair 3 (실선) + single 3 (점선) overlay
   └─ confusion_matrices.png   1×3 pair CM (arithmetic, threshold=0.5)
```

---

## 재현

```bash
# single_modal_baseline/ 결과가 이미 있어야 함 (test_predictions CSV 들)
python double_model/scripts/late_fusion.py    # → results/ 채우고 표 출력
python double_model/scripts/visualize.py      # → figures/ 3장
```

두 스크립트 모두 이 폴더 기준 상대 경로 + `../single_modal_baseline/results/` 참조.
추가 모델 훈련 없이 확률 CSV 연산만 수행하므로 몇 초면 끝남.

---

## triple_model 단계에서 기록해둘 것

(나중에 세 모달 합칠 때 이 폴더와 1:1 비교 가능하도록 동일하게 남기면 좋음)

- `split`: 293/64/63 (동일 사용 — seed99)
- `source_probs`: 어떤 확률을 썼는지 (train+val retrain / train-only / encoder embedding)
- `method`: late / intermediate / early, weight scheme
- `modalities`: 항상 `["clinical","radiomics","ct"]` 순서
- `test_auroc`: arithmetic / logit / weighted / trained-head 등 변형별
- `single_modal_reference_test_auroc_trainval`: 비교 앵커 숫자 (위 표의 3개)
- `double_model_reference_test_auroc`: 이 폴더 `clin_ct=0.6274` 같은 상한 체크 숫자

이 규약을 유지하면 `summary.json` 세 개(single / double / triple) 를 한 스크립트로
읽어 통합 bar chart 를 그릴 수 있습니다.
