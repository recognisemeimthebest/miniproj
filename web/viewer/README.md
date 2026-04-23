# LUNA-XAI Web Viewer

NSCLC(비소세포폐암) 환자의 **2년 생존 확률 예측 + 설명가능성(XAI)** 을 브라우저에서 시각화하는 정적 웹 뷰어.

`file://` 로컬 실행 가능한 단일 HTML 파일 구조(서버 불필요)이며, Lung1 공개 데이터셋의
**test 63명**에 대한 사전 추출 임베딩·Grad-CAM·DICOM 원본 기반 3D 재구성을 모두 포함한다.

---

## 🎯 무엇을 보여주는가

### 1) 예측 탭 (🔍)
| 영역 | 내용 |
|------|------|
| **왼쪽 사이드바** | 환자 ID 드롭다운 선택 · 임상 정보(나이/성별/병기/TNM/조직형) · 🫁 **반투명 3D 폐+종양 미니뷰** |
| **중앙 (메인)** | (상단) **3-plane MPR** — Axial/Sagittal/Coronal 동시 표시 + 슬라이스/HU 윈도잉/종양 토글/측정  <br>(하단) **3D Volume** — Plotly Mesh3d 기반 · 드래그=회전 / 스크롤=줌 / Shift-드래그=팬 |
| **오른쪽** | 2년 생존 확률 큰 카드 + 🟢/🟡/🔴 위험도 · M4 vs M1(Clinical) vs M3(Clin+Rad) 비교 바 · 📝 **AI 판단 근거 자연어 요약** |

### 2) Grad-CAM 탭 (🖼️)
- **CT axial 슬라이스 + Grad-CAM overlay** (iter0 · GradCAM + ReLU+min-max, `features[3].act`)
- 정량 지표: **IoU@25 = 0.149, IoU@50 = 0.282, Pointing Game = 0.127**

### 3) SHAP 분석 탭 (📊)
- **환자별 Top-20 차원 SHAP** — 선형 모델에서 정확히 `W_i · (x_i − μ_i)/σ_i`
- **Modality 기여도** (임상 / Radiomics / CT) — logit 합산 검증식 표시
- **Global importance** — test n=63 전체 평균 `|SHAP|` 의 modality별 합계
- 임상 latent dim 중 원본 변수(나이·T/N/M·병기·조직형)와 trainval Pearson `|r| ≥ 0.40` 인 차원은 **한글 변수명**으로 표기(예: `나이`, `대세포암`, `M병기`). 그 외는 `dim <N>` 유지.

---

## 🏗️ 기술 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│  static_app.html  (단일 파일 · ~6 MB)                              │
│  ├─ 모델 가중치 (JSON 인라인) — W, bias, scaler mean/scale (640)    │
│  ├─ 환자 임베딩 (63 × 640) · 임상정보 · GT 라벨                      │
│  ├─ Grad-CAM PNG 63장 (base64 인라인)                              │
│  └─ JS 추론: logit = b + Σ W_i · (x_i − μ_i)/σ_i    sigmoid         │
└────────────────────────────────────────────────────────────────────┘
                    │
                    ▼  iframe (같은 폴더 상대경로)
     ┌──────────────┼──────────────┐──────────────────┐
     │              │              │                  │
 patients/      patients3d/    patients3d_lung/    rt_data/
 (3-plane MPR)  (풀 3D Volume)   (폐+종양 미니뷰)   (DICOM 원본 GTV-1)
 422 × HTML     63 × HTML        63 × HTML           63 × NIfTI 쌍
```

### 모델 추론 — 순수 JS
`M4 = TripleLinearL2 (640 → 1)`이므로 **표준화 스칼라 + Linear = JS 한 줄 연산**:
```js
logit = b + Σ W[i] · (x[i] − mean[i]) / scale[i]
```
PyTorch 예측과 `|Δ logit| = 2e-8` 수준 일치, **Test AUROC = 0.6589** 완전 재현.

### 3D 렌더링 — Plotly Mesh3d
1. `rt_data/<PID>/image.nii.gz` + `mask_GTV-1.nii.gz` 로드 (DICOM 원본에서 재변환)
2. HU 정규화 → `scipy.ndimage.zoom` 으로 다운샘플 (80³ / 72³)
3. **`skimage.measure.marching_cubes(..., spacing=dsz_mm)`** 로 mm 단위 mesh 생성 — 비등방성 voxel(0.98 × 0.98 × 3.0 mm) 보정
4. `Plotly Mesh3d` 로 body + GTV-1 표면 렌더 (flatshading, 반투명)
5. 의료영상 convention(환자 왼쪽 = 화면 오른쪽)에 맞춰 `scene.xaxis.autorange="reversed"`

### DICOM → GTV-1 정확 추출
원본 `gtv_mask.nii.gz`는 RTSTRUCT의 여러 구조(`Lung-Right`, `Lung-Left`, `Heart`, `Esophagus`, `Spinal-Cord`, `gtv-2`, `gtv-3`, `GTV-1`)가 **합쳐진 합본**이라 종양만 보이지 않음. 이를 해결하기 위해:

- `dcmrtstruct2nii` 로 **"GTV-1" ROI만** 추출
- DICOM 원본 CT도 같은 voxel grid로 재조립 → `rt_data/<PID>/image.nii.gz + mask_GTV-1.nii.gz`
- 결과: 63명 전원 정상 NSCLC 크기 (**중앙값 17cc**, 범위 1.7 ~ 197 cc)

| 환자 | 이전 합본 | DICOM GTV-1 |
|---|---:|---:|
| LUNG1-004 | 820 cc | **84.5 cc** |
| LUNG1-134 | 1,781 cc | **2.9 cc** |
| LUNG1-111 | 1,968 cc | **2.6 cc** |
| LUNG1-355 | 130 cc | **15.4 cc** |

### 임상 latent 해석 (Pearson 상관)
Clinical encoder의 128 latent dim과 원본 임상 변수 13개(나이·성별·T/N/M·병기 one-hot·조직형 one-hot) 간 trainval Pearson 상관을 계산. `|r| ≥ 0.40` 인 dim은 SHAP 탭에서 한글 변수명으로 표시. (114 / 128 dim 매핑)

---

## 📂 파일 구조

```
web/
├── static_app.html              # 최종 정적 앱 (더블클릭 실행)
├── build_static.py              # payload + HTML 빌더 (M4 + baselines + Grad-CAM base64)
├── build_3d_viewers.py          # 풀 3D volume (mesh3d) 빌더
├── build_lung_thumbs.py         # 폐+종양 미니뷰 빌더
├── rebuild_gtv_from_dicom.py    # DICOM RTSTRUCT → GTV-1 NIfTI 변환
│
├── checkpoints/
│   └── triple_linear_l2_best.pt # M4 체크포인트 (11 KB)
├── embeddings/                  # clinical/radiomics/ct · train/val/test npz
│
├── patients/        → symlink → /home/team4/LJW/ct_viewer/patients/
│   └── LUNG1-XXX.html           # LJW 3-plane MPR viewer (422명)
├── patients3d/
│   └── LUNG1-XXX.html           # Plotly 3D volume (63 × ~700 KB)
├── patients3d_lung/
│   └── LUNG1-XXX.html           # 폐+종양 미니뷰 (63 × ~250 KB)
├── rt_data/
│   └── LUNG1-XXX/
│       ├── image.nii.gz          # DICOM 재조립 CT
│       └── mask_GTV-1.nii.gz     # RTSTRUCT의 GTV-1만
│
└── viewer/
    └── README.md                # (this file)
```

---

## 🔧 빌드 파이프라인

```bash
cd web/

# 1) DICOM → GTV-1 + CT 재변환 (최초 1회)
python rebuild_gtv_from_dicom.py                  # → rt_data/

# 2) 3D 뷰어 생성
python build_3d_viewers.py  --size 80             # → patients3d/
python build_lung_thumbs.py --size 72             # → patients3d_lung/

# 3) 최종 HTML (모델 가중치 + 임베딩 + Grad-CAM 인라인)
python build_static.py                            # → static_app.html
```

Grad-CAM overlay PNG는 상위 `results/gradcam_iter0.py --all-test` 로 생성
(`features[3].act` 대상 · GradCAM · ReLU+min-max 후처리).

---

## 🚀 실행

```
file:///.../web/static_app.html
```

브라우저에서 바로 열면 됨(서버 불필요). 네트워크 접근 1건: 3D iframe 이 Plotly.js CDN 로드 (오프라인 환경이면 `build_3d_viewers.py` 의 `include_plotlyjs="cdn"` → `"inline"` 으로 전환, 용량은 환자당 +~4 MB).

---

## 📊 성능 지표

| 모델 | 입력 | Test AUROC |
|------|------|-----------:|
| **M4 TripleLinearL2** | Clinical(128) + Radiomics(256) + CT(256) | **0.6589** |
| M1 Clinical (재적합) | Clinical(128) | 0.57 ~ |
| M3 Clin + Rad (재적합) | Clinical + Radiomics | 0.60 ~ |

| Grad-CAM (iter0) | 값 |
|---|---:|
| IoU@25 | 0.149 |
| IoU@50 | 0.282 |
| Pointing Game | 0.127 (8/63) |

---

## 📦 Python 의존성

```
torch          # M4 체크포인트 로드
nibabel        # NIfTI I/O
pydicom        # DICOM 메타 파싱
dcmrtstruct2nii  # RTSTRUCT → NIfTI
scikit-image   # marching_cubes
scipy          # ndimage.zoom, label
numpy, pandas, plotly  # 시각화·전처리
scikit-learn   # StandardScaler, LogisticRegression (baselines)
```

---

## ⚠️ 연구용 프로토타입

본 뷰어는 **Lung1 공개 데이터셋의 시연·해석 목적**으로 제작되었으며, 실제 임상 의사결정에
사용할 수 없음. M4 = TripleLinearL2(feature/LJW), seed=99, trainval refit, test AUROC = 0.6589.
