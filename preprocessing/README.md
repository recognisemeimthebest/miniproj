# preprocessing

TCIA NSCLC-Radiomics Lung1 임상 데이터 전처리 모듈입니다.

## 파일

| 파일 | 설명 |
|------|------|
| `clinical_preprocessing.py` | 임상 CSV 전처리 메인 스크립트 |

## 데이터

- **원본**: `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv`
- **환자 수**: 422명 / 컬럼: 10개

| 컬럼 | 설명 |
|------|------|
| PatientID | 환자 ID (LUNG1-xxx) |
| age | 나이 |
| clinical.T.Stage | 종양 크기·침범 단계 (1~5) |
| Clinical.N.Stage | 림프절 전이 단계 (0~4) |
| Clinical.M.Stage | 원격 전이 단계 |
| Overall.Stage | 종합 병기 (I, II, IIIa, IIIb) |
| Histology | 조직형 |
| gender | 성별 |
| Survival.time | 생존 기간 (days) |
| deadstatus.event | 사망 여부 (1=사망, 0=생존/중도절단) |

## 전처리 흐름

```
CSV 로드
  ↓
결측치 처리
  - age (22명): 중앙값 대체
  - clinical.T.Stage (1명): 최빈값 대체
  - Overall.Stage (1명): 최빈값 대체
  - Histology (42명): 'unknown' 범주 추가
  ↓
2년 생존 레이블 생성 (label_2yr)
  - 1 : Survival.time >= 730일
  - 0 : 730일 이내 사망 확인
  - NaN : 중도절단 → 모델 학습에서 제외
  ↓
범주형 인코딩
  - gender: male=1, female=0
  - Overall.Stage: 순서형 (I=1, II=2, IIIa=3, IIIb=4)
  - Histology: One-Hot Encoding (hist_xxx 컬럼 생성)
  ↓
CSV 저장 + 시각화 PNG 저장
```

## 레이블 분포 (결과 예시)

| 레이블 | 의미 | 환자 수 |
|--------|------|---------|
| 1 | 2년 이상 생존 | ~135명 |
| 0 | 2년 이내 사망 | ~215명 |
| NaN | 중도절단 (제외) | ~72명 |

학습에 사용되는 환자: **약 350명**

## 실행 방법

```bash
# CSV 파일을 스크립트와 같은 폴더에 위치시킨 뒤
python clinical_preprocessing.py
```

## 출력 파일

```
output_clinical/
  ├── lung1_clinical_cleaned.csv    # 결측치 처리 완료, label_2yr 포함
  ├── lung1_clinical_encoded.csv   # 범주형 인코딩 완료 (모델 입력용)
  └── clinical_overview.png        # 생존기간, 병기, 조직형 시각화
```

## 다음 단계 연결

- `lung1_clinical_encoded.csv` → `ml_baseline/` ML 모델 학습에 사용
- `lung1_clinical_encoded.csv` → `clinical_dim/` ClinicalBranch MLP 학습에 사용
- PatientID(`LUNG1-xxx`)를 join key로 CT/Radiomics 데이터와 연결 예정
