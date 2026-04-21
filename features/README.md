# Features 폴더 설명 - LJY

## 파일 목록

| 파일명 | 설명 | 크기 |
|---|---|---|
| `radiomics_features.csv` | pyradiomics 원본 추출 특징 | 421명 x 851개 |
| `features_corr_filtered.csv` | 상관 필터 적용 후 특징 | 421명 x 269개 |
| `final_features.csv` | LASSO 최종 선택 특징 | 421명 x 15개 |
| `radiomics_embed_128.npy` | MLP 128dim 임베딩 (Fusion 투입용) | 421명 x 128 |
| `radiomics_embed_128.csv` | MLP 128dim 임베딩 확인용 | 421명 x 128 |

## Fusion 모델 투입 파일
- **`radiomics_embed_128.npy`** 를 Fusion 모델에 투입
- 차원: 128dim
- Fusion 총 차원: CT(256) + 라디오믹스(128) + 임상(128) = 512dim