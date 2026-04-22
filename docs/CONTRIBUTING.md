# 기여 가이드 (Contributing Guide)

## 브랜치 규칙

- `main` 브랜치에 **직접 push 금지** — 반드시 PR로만 변경
- PR merge 전 **팀원 1명 이상 review 필수**
- 브랜치명 규칙: `feat/기능명` (예: `feat/segresnet`, `feat/gradcam`)

```
main        ← 보호됨
└── dev     ← 통합 브랜치 (PM 관리)
    └── feat/xxx  ← 각자 작업 브랜치
```

---

## 폴더 소유권

다른 사람 담당 폴더를 수정할 경우 **반드시 해당 담당자를 PR reviewer로 지정**

| 폴더 / 파일 | 담당자 | PR reviewer |
|---|---|---|
| `src/data/` | 팀원 C | 팀원 C |
| `src/models/image_encoders/` | 팀원 B | 팀원 B |
| `src/models/tabular_encoders/` | 팀원 C | 팀원 C |
| `src/models/fusion.py` | 팀원 A | 팀원 A |
| `src/train/` | 공용 | 팀원 A |
| `src/evaluate/` | 팀원 D | 팀원 D |
| `src/explain/gradcam.py` | 팀원 B | 팀원 B |
| `src/explain/shap_utils.py` | 팀원 C | 팀원 C |
| `src/explain/ablation.py` | 팀원 D | 팀원 D |
| `app/` | 팀원 A (통합) | 팀원 A |
| `configs/` | 공용 | 없음 |
| `experiments/` | 공용 | 없음 |

---

## AI 툴 사용 시 주의사항

팀원마다 다른 AI 코딩 툴을 사용합니다. 아래 파일들이 실수로 commit되지 않도록 주의하세요.

| 툴 | 생성 폴더/파일 | 처리 |
|---|---|---|
| Claude Code | `.claude/` | `.gitignore` 처리 |
| Antigravity | `.antigravity/cache/` | `.gitignore` 처리 |
| Cursor | `.cursor/` | `.gitignore` 처리 |

**Day 1 환경 셋업 시 각자 본인 툴의 캐시 폴더가 `.gitignore`에 있는지 확인 필수**

---

## Commit 메시지 규칙

```
feat: SegResNet encoder 구현
fix: CT 전처리 HU clipping 버그 수정
data: Radiomics feature selection 스크립트 추가
exp: exp01_segresnet 결과 추가
docs: README 업데이트
```

---

## 실험 결과 기록 방법

새 실험 시작 시 `experiments/` 아래 폴더 생성:

```
experiments/
└── exp01_segresnet/
    ├── config.yaml      # 실험에 사용한 설정 (복사본)
    └── results.csv      # AUROC, Sensitivity, Specificity, F1
```

`results.csv` 형식:
```
model,split,auroc,sensitivity,specificity,f1,epoch
segresnet,val,0.72,0.68,0.75,0.70,42
segresnet,test,0.74,0.70,0.77,0.72,-
```

---

## 환경 통일

```bash
# 환경 생성
conda env create -f environment.yml
conda activate luna-xai

# 패키지 추가 시 반드시 environment.yml 업데이트 후 PR
conda env export --no-builds | grep -v "^prefix" > environment.yml
```
