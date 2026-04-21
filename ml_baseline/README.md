# ml_baseline

임상 정형 데이터만 사용한 2년 생존 예측 머신러닝 베이스라인입니다.  
멀티모달 fusion 모델 대비 임상 데이터 단독 성능의 기준선을 설정합니다.

## 파일

| 파일 | 설명 |
|------|------|
| `ml_clinical_baseline.py` | 4가지 ML 모델 비교 스크립트 |

## 입력 데이터

`preprocessing/output_clinical/lung1_clinical_encoded.csv`

- PatientID를 인덱스로 사용
- `label_2yr` 컬럼이 target (NaN 행 제외 후 학습)
- `Survival.time`, `deadstatus.event`는 **데이터 누수 방지를 위해 제외**

## 사용 피처 (11개)

```
age, clinical.T.Stage, Clinical.N.Stage, Clinical.M.Stage,
Overall.Stage, gender,
hist_adenocarcinoma, hist_large cell, hist_squamous cell carcinoma,
hist_squamous cell carcinoma nos, hist_unknown
```

## 비교 모델

| 모델 | 설명 |
|------|------|
| Logistic Regression | L2 정규화, class_weight=balanced |
| Random Forest | 100 trees, class_weight=balanced |
| XGBoost | scale_pos_weight로 클래스 불균형 보정 |
| SVM | RBF kernel, 표준화 포함 |

## 평가 방법

- **Hold-out**: Stratified train/test split (80:20)
- **Cross-validation**: Stratified 5-Fold CV (AUC)
- **지표**: ROC-AUC (주요), Accuracy, F1

## 결과 (예시)

| 모델 | Test AUC | 5-CV AUC |
|------|----------|----------|
| Logistic Regression | ~0.59 | ~0.58 |
| Random Forest | ~0.60 | ~0.57 |
| **XGBoost** | **~0.618** | **~0.606** |
| SVM | ~0.57 | ~0.56 |

→ XGBoost 단독 최고 성능, 임상 데이터만으로는 AUC 0.62 수준

## 실행 방법

```bash
# preprocessing 결과가 output_clinical/ 폴더에 있어야 함
python ml_clinical_baseline.py
```

## 출력 파일

```
output_ml/
  ├── ml_results.csv           # 모델별 전체 성능 지표
  └── roc_curves.png           # ROC 커브 비교 그래프
```

## 해석

임상 데이터 단독 AUC ≈ 0.62는 예측력이 제한적임을 보여줍니다.  
CT 이미지 및 Radiomics 피처를 추가한 멀티모달 fusion 모델에서  
성능 향상을 기대합니다.
