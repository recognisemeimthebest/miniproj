# XAI Analysis — NSCLC 2-Year Survival Prediction

최종 모델 `triple_linear_l2_best.pt`에 대한 설명 가능 AI(XAI) 분석 스크립트

## 파일 구성

| 파일 | 설명 |
|------|------|
| `modality_ablation_xai.py` | Modality Ablation — 각 모달리티를 제거했을 때 AUROC 변화로 기여도 측정 |
| `shap_xai.py` | SHAP — 선형 모델 정확 해 (Linear SHAP), feature/modality별 기여도 분석 |

## 실행 방법

```bash
cd miniproj   # 프로젝트 루트
python XAI/modality_ablation_xai.py
python XAI/shap_xai.py
```

## 사전 조건

- `triple_linear_l2_best.pt` — 최종 융합 모델
- `triple_model/embeddings/{clinical,radiomics,ct}_test.npz` — 테스트셋 임베딩

출력 파일은 `output_xai/` 폴더에 저장됩니다.

## 결과 요약

### Modality Ablation (ΔAUROC, Full=0.6589)

| Modality | ΔAUROC | 해석 |
|----------|--------|------|
| Radiomics | +0.0863 | 제거 시 AUC 하락 폭 최대 → **기여도 1위** |
| CT | −0.0147 | |
| Clinical | −0.0158 | |

### SHAP (Sum |SHAP| per patient)

| Modality | Sum \|SHAP\| |
|----------|-------------|
| Radiomics | 0.795 ← 1위 |
| CT | 0.460 |
| Clinical | 0.426 |

> Ablation과 SHAP 모두 **Radiomics가 예측에 가장 핵심적인 모달리티**임을 일관되게 보여줌