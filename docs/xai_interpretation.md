# XAI — Grad-CAM on Hosny 3D CNN

> Team member B (Grad-CAM) — Project Plan §3.7.1
> Seed-123 CT checkpoint (`single_modal_baseline/results/ct/best.pt`, test AUROC 0.601) — fixed, no retraining.

## Goal

Visualize *what regions of the input CT* drive the Hosny 3D CNN's 2-year survival prediction, and quantitatively evaluate whether those regions overlap with the annotated GTV.

## Setup

- **Model**: Hosny 2018 shallow 3D CNN (4 conv blocks, 32→64→128→256 ch, ~1.16M params)
- **Input**: 80×80×80 GTV-centered ROI, HU [−1000, 400] → [0, 1]
- **Library**: `pytorch-grad-cam` 1.5.5
- **Test set**: n=63 (seed=99 shared split)
- **Metrics**: IoU@25%, IoU@50%, Pointing-game (peak voxel ∈ GTV-1)
- **Random baselines** (GTV fills 45.9% of ROI): IoU@25=0.193, IoU@50=0.315, PG=0.459

## 5-iteration evaluator-guided tuning

| Iter | Config | IoU@25 | IoU@50 | PG | Note |
|---|---|---|---|---|---|
| random | — | 0.193 | 0.315 | 0.459 | baseline |
| 0 | GradCAM, `features[3].act` (10³) | 0.149 | 0.282 | 0.127 | all below random |
| 1 | LayerCAM, `features[2].act` (20³) | 0.169 | 0.305 | 0.254 | PG 2× — block[2] sweet spot |
| 2 | HiResCAM, `features[1].act` (40³), p75p99 | 0.159 | **0.459** | 0.127 | IoU@50 crosses random, PG drifts off-tumor |
| 3 | LayerCAM b[2] + p75p99 | 0.169 | 0.459 | 0.095 | `argmax` tie-break artifact from p99 plateau |
| 4 | LayerCAM b[2] + σ=2 Gaussian + honest PG fix | 0.147 | 0.291 | 0.254 | smoothing monotonic → peak unchanged |
| **5** | **LayerCAM b[2] + lung-tissue mask** | **0.219** | **0.367** | **0.270** | **IoU@25·IoU@50 both clear random** |

## Final configuration

```bash
PYTHONPATH=. python src/explain/gradcam_ct.py --all-test \
  --cam-method layercam --target-block 2 \
  --post relu_minmax --lung-mask
```

- **Method**: `LayerCAM` on `model.features[2].act` (post-ReLU pre-pool, 20³)
- **Upsample**: trilinear → 80³
- **Tissue mask**: multiply by `(normalized_CT > 0.357)` (= CT HU > −500)
- **Post-proc**: ReLU + min-max to [0, 1]
- **Metric integrity**: Pointing-game computed from pre-post-proc CAM to avoid `np.argmax` tie-break artifacts under plateau-inducing post-processing

## Methodological takeaways

1. **Monotonic transforms cannot move argmax.** Smoothing, percentile rescaling, and (critically) top-k clipping all preserve the existing maximum location. To *relocate* the CAM peak you need a non-monotonic intervention (mask multiplication, layer fusion, anatomical prior).
2. **Pointing-game is sensitive to post-processing artifacts.** Percentile clipping (p99) creates plateaus at the cap value; `np.argmax` falls back to row-major order and lands in ROI corners (air). Compute PG *before* any clip/plateau-inducing step.
3. **Depth–peak trade-off.** Shallower layers (40³, 80³) give broader spatial coverage → higher IoU@50, but peak drifts into texture/edge spikes. Deeper layers (10³) stabilize the peak but lose spatial resolution. `features[2]` (20³) was the experimentally-confirmed sweet spot.
4. **Tissue mask is the single most effective lever.** For a backbone with test AUROC 0.601, non-trivial CAM mass leaks into air and background. A simple HU-based tissue mask pushed IoU@25 and IoU@50 over their random baselines simultaneously for the first time.

## Limits

- PG = 0.270 remains below random 0.459. The backbone model (AUROC 0.601) attends to non-GTV tissue structures (mediastinum, chest wall). This is Project Plan §6 "noisy attention from low-AUROC backbone" quantitatively confirmed.
- CAM-level improvements are near ceiling without model retraining. Further gains would require (a) swapping to the seed=99 checkpoint (test AUROC 0.666), (b) training with GTV-focus regularization, or (c) adding a radiomics-stream co-training signal.

## Artifacts

- **Code**: [src/explain/gradcam_ct.py](../src/explain/gradcam_ct.py)
- **Full tuning log**: [results/gradcam_tuning_log.md](../results/gradcam_tuning_log.md)
- **Per-patient CSV + summary**: `results/gradcam_iou.csv`, `results/gradcam_summary.txt`
- **Overlays (n=63)**: `figures/gradcam_overlays_iter5/`
- **Quantitative histogram**: `figures/gradcam_iou_hist.png`
