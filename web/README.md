# LUNA-XAI 웹 구현 - LJY

## 개요
NSCLC(비소세포폐암) 2년 생존 예측 Streamlit 웹 애플리케이션

## 환경
- Python 3.9
- streamlit
- torch
- numpy
- pandas
- plotly
- shap
- scikit-learn
- matplotlib

## 실행 방법
```bash
conda activate radiomics_env
cd web
streamlit run app.py
```

## 접속 방법
- 로컬: http://localhost:8501
- 네트워크(같은 WiFi): http://192.168.0.44:8501

## 기능

### 1. 예측 결과 탭
- 환자 정보 입력 (Patient ID, 나이, 성별, TNM Stage, Histology)
- M4 Fusion 모델 기반 2년 생존 확률 예측
- 고위험/저위험 분류
- 모달리티별 기여도 파이차트
- 모달리티별 AUROC 표

### 2. 모달리티 기여도 탭
- Modality Ablation Study 결과
- M1(Clinical) / M2(CT) / M3(Rad+Clin) / M4(Fusion) AUROC 비교
- SHAP 특징 기여도 분석 (라디오믹스 15개 + 임상 변수)
- 상위 10개 주요 특징 표

### 3. 모델 성능 탭
- 라디오믹스 단독 모델 성능 비교 (256dim)
- GBM 최고 성능 (Test AUROC 0.6112)

## 파일 구조
web/
├── app.py                    # 메인 Streamlit 앱
├── README.md                 # 웹 구현 설명
├── checkpoints/
│   └── triple_linear_l2_best.pt  # Fusion 모델
└── embeddings/
├── ct_train/val/test.npz     # CT 임베딩
├── radiomics_train/val/test.npz  # 라디오믹스 임베딩
└── clinical_train/val/test.npz   # 임상 임베딩

## 모델 정보
- 모델명: TripleLinearL2
- 입력 차원: 640 (CT 256 + Radiomics 256 + Clinical 128)
- Test AUROC: 0.781

## 사용 가능한 환자 목록 (Test set, 63명)
LUNG1-004, LUNG1-015, LUNG1-026, LUNG1-043, LUNG1-054,
LUNG1-055, LUNG1-066, LUNG1-070, LUNG1-078, LUNG1-082,
LUNG1-088, LUNG1-099, LUNG1-100, LUNG1-104, LUNG1-111,
LUNG1-123, LUNG1-129, LUNG1-131, LUNG1-134, LUNG1-139,
LUNG1-155, LUNG1-159, LUNG1-162, LUNG1-165, LUNG1-173,
LUNG1-177, LUNG1-179, LUNG1-182, LUNG1-202, LUNG1-206,
LUNG1-212, LUNG1-213, LUNG1-219, LUNG1-222, LUNG1-231,
LUNG1-237, LUNG1-259, LUNG1-260, LUNG1-267, LUNG1-278,
LUNG1-291, LUNG1-295, LUNG1-317, LUNG1-322, LUNG1-325,
LUNG1-327, LUNG1-353, LUNG1-355, LUNG1-363, LUNG1-366,
LUNG1-370, LUNG1-375, LUNG1-376, LUNG1-378, LUNG1-384,
LUNG1-387, LUNG1-394, LUNG1-399, LUNG1-400, LUNG1-409,
LUNG1-411, LUNG1-417, LUNG1-418