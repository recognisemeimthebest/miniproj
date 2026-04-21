# clinical_dim

임상 데이터를 멀티모달 fusion 모델에 연결하기 위한 ClinicalBranch MLP 모듈입니다.  
Optuna를 사용한 하이퍼파라미터 자동 탐색으로 최적 임베딩을 학습합니다.

## 파일

| 파일 | 설명 |
|------|------|
| `clinical_branch.py` | ClinicalBranch MLP 정의 + Optuna 튜닝 스크립트 |

## 입력 데이터

`preprocessing/output_clinical/lung1_clinical_encoded.csv`

## 모델 구조

```
입력 (11개 피처)
  ↓
Linear(input_dim → hidden_dim) + BatchNorm + ReLU + Dropout
  ↓
Linear(hidden_dim → hidden_dim//2) + BatchNorm + ReLU + Dropout
  ↓
Linear(hidden_dim//2 → embed_dim)
  ↓
[standalone 모드] ReLU → Linear(embed_dim, 1) → Sigmoid → 이진 분류
[fusion 모드]     embed_dim 벡터 반환 → 다른 브랜치와 concat
```

## Fusion 모델 연결 방법

```python
from clinical_branch import ClinicalBranch

model = ClinicalBranch(input_dim=11, hidden_dim=128, embed_dim=128, dropout=0.3)
model.load_state_dict(torch.load("output_clinical_branch/clinical_branch.pt"))

# 임베딩 벡터 추출 (fusion 시 사용)
embedding = model(clinical_tensor, return_embedding=True)  # shape: (batch, 128)
```

CT 브랜치(128) + Radiomics 브랜치(64) + Clinical 브랜치(128) → concat(320) → Fusion MLP

## Optuna 탐색 범위

| 하이퍼파라미터 | 범위 |
|---------------|------|
| hidden_dim | 64, 128, 256 |
| embed_dim | 32, 64, 128 |
| dropout | 0.1 ~ 0.5 |
| learning rate | 1e-4 ~ 1e-2 |
| batch_size | 16, 32, 64 |
| weight_decay | 1e-5 ~ 1e-3 |

- 탐색 횟수: 50 trials
- 평가: Stratified 5-Fold CV AUC (평균)

## 실행 방법

```bash
python clinical_branch.py
```

> GPU가 없어도 CPU로 실행 가능합니다.

## 출력 파일

```
output_clinical_branch/
  ├── clinical_branch.pt     # 최적 파라미터로 학습된 모델 가중치
  ├── best_params.json       # Optuna 최적 하이퍼파라미터
  └── optuna_results.png     # Trial별 AUC 수렴 그래프
```

## 튜닝 결과

| 항목 | 값 |
|------|-----|
| Best embed_dim | 128 |
| Best CV AUC | 0.5664 |

임상 데이터 단독으로는 AUC 0.57 수준으로, CT/Radiomics 브랜치와 결합하면  
성능이 크게 향상될 것으로 예상합니다.

## 의존 패키지

```
torch, optuna, scikit-learn, pandas, numpy, matplotlib
```

```bash
pip install torch optuna scikit-learn pandas numpy matplotlib
```
