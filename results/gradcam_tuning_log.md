# Grad-CAM 5-Iteration Tuning Log

**목표**: seed=123 ckpt (`single_modal_baseline/results/ct/best.pt`, test AUROC 0.601) 고정, **Grad-CAM 설명 품질**(IoU@25%, IoU@50%, Pointing-game) 개선.

**제약**: 모델 재학습 금지. 오직 CAM 방법·target layer·후처리·정규화 등만 수정.

**Test set**: n=63, 80³ ROI 공간.

**Random baselines** (GTV가 ROI의 45.9% 차지):
- IoU@25%: **0.193**
- IoU@50%: **0.315**
- Pointing-game: **0.459**

---

## Iter 0 — Baseline

**설정**: `GradCAM` on `model.features[3].act` (ReLU 후, 10³ → trilinear 80³), class=1 logit, ReLU-normalize, min-max [0,1].

**결과**:
| 지표 | 값 | vs random |
|---|---|---|
| IoU@25% | 0.149 ± 0.069 | −0.044 |
| IoU@50% | 0.282 ± 0.093 | −0.033 |
| Pointing-game | **0.127 (8/63)** | **0.28× random** ⚠️ |

**해석**: 모든 지표가 random baseline 이하. Grad-CAM이 tumor 바깥을 보고 있음.

---

## Iter 1 — LayerCAM on features[2].act (pre-pool, 20³)

**평가에이전트 권고 요지**:
- GradCAM의 채널 전역평균(GAP) 가중치가 shallow feature map에서 공간 신호를 과도하게 평탄화.
- LayerCAM은 **양(+)인 gradient만** channel·spatial-wise로 사용 → 공간 국소성↑.
- `features[3].act`(10³) → `features[2].act`(20³, pre-pool): 해상도 8× (voxel 개수 기준), interpolation artifact 감소.

**설정**: `LayerCAM` on `model.features[2].act` (ReLU 후, 20³ → trilinear 80³), class=1 logit, ReLU-normalize, min-max [0,1].

**결과**:
| 지표 | Iter 1 | vs Iter 0 | vs random |
|---|---|---|---|
| IoU@25% | **0.169 ± 0.043** | +0.020 (+13%) | −0.024 (0.88× random) |
| IoU@50% | **0.305 ± 0.100** | +0.023 (+8%) | −0.010 (0.97× random) |
| Pointing-game | **0.254 (16/63)** | **+0.127 (2.00×)** | 0.55× random |

**해석**:
- Pointing-game 2배 개선 — peak이 tumor 안쪽으로 이동 중.
- IoU@50%는 거의 random 수준까지 올라옴 (0.97×).
- 그러나 PG random(0.459)까진 여전히 격차. Iter 2에서 추가 개선 필요.

**산출물**: `results/gradcam/iter1_layercam_b2/` (63 NIfTI), `figures/gradcam_overlays_iter1/` (63 PNG)

---

## Iter 2 — HiResCAM on features[1].act (40³) + p75p99 post

**평가에이전트 권고 요지**:
- 더 얕은 층(40³)으로 ↓ → 공간 해상도 ↑.
- HiResCAM: grad ⊙ activation element-wise (GradCAM GAP 없음) → 공간 기울기 구조 보존.
- Post-proc: ReLU+min-max → [p75, p99] clip+rescale (배경 억제).

**설정**: `HiResCAM` on `features[1].act` (40³ → trilinear 80³), post=p75p99.

**결과**:
| 지표 | Iter 2 | vs Iter 1 | vs random |
|---|---|---|---|
| IoU@25% | 0.159 ± 0.043 | −0.010 | 0.82× |
| IoU@50% | **0.459 ± 0.199** | **+0.154** | **1.46× (>random!)** |
| Pointing-game | 0.127 (8/63) | **−0.127 (regression!)** | 0.28× |

**해석**:
- **IoU@50% 최초로 random 상회 (0.46 > 0.32)** — top 50% voxel mass가 tumor와 잘 겹침.
- 그러나 **peak voxel이 tumor 바깥으로 이동** — PG 원점 회귀.
- 원인 가설: 40³ shallow layer에선 edge/경계 쪽 spike가 peak을 차지. HiResCAM이 공간 gradient를 직접 노출 → texture noise peak이 강해짐.
- **트레이드오프 확인**: block 깊이 ↓ → area coverage ↑, peak stability ↓.

**산출물**: `results/gradcam/iter2_hirescam_b1_p75p99/`, `figures/gradcam_overlays_iter2/`

---

## Iter 3 — LayerCAM on features[2].act + p75p99

**평가에이전트 권고 요지**: Iter 1 (좋은 PG) × Iter 2 (좋은 IoU@50)의 대각 조합. 퍼센타일 rescale은 monotonic → peak 불변이라 예측.

**결과**:
| 지표 | Iter 3 | vs Iter 1 | vs Iter 2 | vs random |
|---|---|---|---|---|
| IoU@25% | 0.169 ± 0.043 | = | +0.010 | 0.88× |
| IoU@50% | **0.459 ± 0.199** | +0.154 | = | **1.46×** |
| Pointing-game | **0.095 (6/63)** | **−0.159** | −0.032 | 0.21× (worse!) |

**해석 — 에이전트 예측 빗나감**:
- "percentile rescale은 monotonic이라 argmax 불변" 가정이 깨짐.
- 원인: p99 clip으로 **상위 1% voxel이 전부 1.0에 plateau** → `np.argmax` tie breaking은 **row-major 첫 인덱스**를 리턴. ROI corner (air)가 tumor보다 먼저 나옴 → PG 붕괴.
- 증거: Iter 1 (같은 layercam+block2, post=relu_minmax)은 PG 0.254. 같은 CAM 원본에서 **post-proc만 추가로 적용**했는데 peak이 바뀜 = tie issue 확실.

**결론**: p75p99 post는 **IoU 계산 전용**, PG는 원본 CAM(혹은 pre-clip)에서 계산해야 정확.

**산출물**: `results/gradcam/iter3_layercam_b2_p75p99/`, `figures/gradcam_overlays_iter3/`

---

## Iter 4 — LayerCAM b[2] + Gaussian σ=2 (honest PG from pre-post-proc CAM)

**평가에이전트 권고 요지**:
- PG 메트릭 버그 수정: argmax를 p99-clip 이전 CAM에서 계산.
- σ=2 vx Gaussian이 isolated spike noise를 없애 PG 2배 기대 (0.48–0.55).

**구현 변경**:
- `run_gradcam()` → `(cam_post, cam_pre_post)` 쌍 반환.
- `compute_metrics(cam_iou, cam_peak, ...)` — IoU는 cam_post, PG는 cam_pre.
- `--smooth-sigma` CLI 플래그.

**결과**:
| 지표 | Iter 4 | vs Iter 1 | vs random |
|---|---|---|---|
| IoU@25% | 0.147 ± 0.052 | −0.022 | 0.76× |
| IoU@50% | 0.291 ± 0.094 | −0.014 | 0.92× |
| Pointing-game | **0.254 (16/63)** | = | 0.55× |

**해석 — 예측 빗나감 again**:
- PG 변화 **없음** (Iter 1과 정확히 동일). Gaussian smoothing이 argmax를 움직이지 못함.
- 원인: 20³ 원본 → trilinear upsample 80³ 과정에서 이미 ~4 vx 공간 smoothing이 내재. 추가 σ=2는 monotonic transform에 가까워 argmax 불변.
- 결론: **LayerCAM b[2] peak은 8/63(=Iter0) → 16/63(=Iter1) 이후 **천장**을 쳤음.** 메서드·smoothing만으로는 0.254 이상 못 감.

**남은 지렛대**: 공간 사전 mask(e.g. lung HU > −500) 혹은 layer fusion. Iter 5 대상.

**산출물**: `results/gradcam/iter4_layercam_b2_sigma2/`, `figures/gradcam_overlays_iter4/`

---

## Iter 5 (FINAL) — LayerCAM b[2] + Lung-tissue mask

**평가에이전트 권고 요지**: smoothing/percentile 같은 monotonic 변환은 argmax를 못 옮김. peak이 air/배경으로 빠지는 게 failure mode. 공간 사전(CT HU > −500) mask로 argmax를 강제로 tissue 안에 배치.

**구현**: `--lung-mask` 플래그. CT normalized > 0.357 (HU=−500 equivalent, since [-1000,400]→[0,1]) 을 tissue로 간주, CAM에 곱한 뒤 min-max.

**결과**:
| 지표 | Iter 5 | vs Iter 4 | vs Iter 1 | vs random |
|---|---|---|---|---|
| IoU@25% | **0.219 ± 0.068** | +0.072 | +0.050 | **1.13× (>random!)** |
| IoU@50% | **0.367 ± 0.146** | +0.076 | +0.062 | **1.17× (>random!)** |
| Pointing-game | **0.270 (17/63)** | +0.016 | +0.016 | 0.59× |

**해석**:
- **최초로 IoU@25%·IoU@50% 동시에 random baseline 상회** (각 1.13×, 1.17×).
- PG는 여전히 random 하회(0.59×)지만 Iter 1/4의 0.254 → 0.270 소폭 개선.
- lung mask가 top-25%/50% voxel mass에 tumor 외 air 영역을 제거 → IoU denominator 줄고 intersection 유지 → IoU 상승.
- PG 개선 제한적인 이유: tumor 안이 아니어도 tissue 안(늑골, 흉벽, 종격동)에 강한 peak이 여전히 존재. GTV mask는 tissue mask보다 훨씬 작음.

**산출물**: `results/gradcam/iter5_layercam_b2_lungmask/`, `figures/gradcam_overlays_iter5/`

---

## 최종 요약 (5-iter 튜닝 결과)

| Iter | Config | IoU@25 | IoU@50 | PG | 평가 |
|---|---|---|---|---|---|
| random | — | 0.193 | 0.315 | 0.459 | baseline |
| 0 | GradCAM, b[3] | 0.149 | 0.282 | 0.127 | 전부 random 이하 |
| 1 | LayerCAM, b[2] | 0.169 | 0.305 | 0.254 | PG 2× |
| 2 | HiResCAM, b[1], p75p99 | 0.159 | 0.459 | 0.127 | IoU@50 up, PG 회귀 |
| 3 | LayerCAM, b[2], p75p99 | 0.169 | 0.459 | 0.095 | PG artifact(tie-break) |
| 4 | LayerCAM, b[2] + σ=2 | 0.147 | 0.291 | 0.254 | smoothing = monotonic, no-op on argmax |
| **5** | **LayerCAM b[2] + lung mask** | **0.219** | **0.367** | **0.270** | **IoU 두 지표 random 상회 (최초)** |

**방법론적 교훈** (기획서 §6에 반영할만):
1. **Pointing-game metric은 post-proc에 민감**: p99-clip이 plateau를 만들면 `np.argmax` tie-break이 PG를 왜곡. IoU는 pre/post-proc 버전, PG는 pre-clip CAM에서 계산해야 정확.
2. **Monotonic transform은 peak을 옮길 수 없다**: smoothing, percentile rescale 모두 argmax 불변. Peak 위치를 바꾸려면 **spatial prior**(mask multiply, fusion 등 non-monotonic)가 필요.
3. **Depth–peak 트레이드오프**: 얕은 layer(40³) → area 커버리지↑, peak noise↑. 깊은 layer(10³) → peak stable but 공간 coarse. block[2] (20³) 이 sweet spot.
4. **Lung/tissue mask가 random 상회의 가장 확실한 레버**: 저성능 모델(0.60 AUROC)의 CAM은 공기 영역에 spurious peak을 만들기 쉬움. tissue-only 제약이 interpretability와 metric 양쪽에 가장 효과적.

**권장 사용 설정 (final)**:
```bash
python src/explain/gradcam_ct.py --all-test \
  --cam-method layercam --target-block 2 --post relu_minmax --lung-mask \
  --out-nii results/gradcam/final --out-png figures/gradcam_overlays
```

**한계 및 남은 과제**:
- PG < random(0.459). backbone model AUROC 0.601은 tumor 이외의 신호(흉벽, 종격동)에 의존하는 경향 → peak이 GTV 대신 tissue 전체에 분산. 기획서 §6 "저성능 모델의 noisy attention" risk가 정량적으로 확인됨.
- 모델 재학습 금지 조건 하에 CAM-level 개선은 여기가 실질적 천장. 근본 해결은 checkpoint 교체(예: seed=99, test 0.666)나 radiomics-stream fusion으로만 가능할 것.
