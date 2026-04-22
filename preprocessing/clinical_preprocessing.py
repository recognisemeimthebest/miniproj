"""
TCIA NSCLC-Radiomics Lung1 - 임상 데이터 전처리
================================================
파일: NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
환자: 422명 / 컬럼: 10개

컬럼 설명:
  PatientID         : 환자 고유 ID (LUNG1-xxx)
  age               : 나이 (결측 22명)
  clinical.T.Stage  : 종양 크기/침범 단계 (1~5)
  Clinical.N.Stage  : 림프절 전이 단계 (0~4)
  Clinical.M.Stage  : 원격 전이 단계 (0,1,3)
  Overall.Stage     : 종합 병기 (I, II, IIIa, IIIb)
  Histology         : 조직형 (결측 42명)
  gender            : 성별 (male/female)
  Survival.time     : 생존 기간 (일, days)
  deadstatus.event  : 사망 여부 (1=사망, 0=생존/중도절단)

목표 변수 (label_2yr):
  1 = 2년(730일) 이상 생존
  0 = 2년 이내 사망
  NaN = 중도절단(730일 미만) → 모델 학습에서 제외
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
CSV_PATH   = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
OUTPUT_DIR = "output_clinical"
THRESHOLD  = 730  # 2년 기준 (days)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}  →  {df.shape[0]}명, {df.shape[1]}개 변수")
print(df.head())

# ─────────────────────────────────────────────
# 2. 결측치 현황
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 결측치 현황")
print("=" * 60)

null_df = pd.DataFrame({
    "null_count": df.isna().sum(),
    "null_%":     (df.isna().mean() * 100).round(1)
})
print(null_df[null_df["null_count"] > 0])

# ─────────────────────────────────────────────
# 3. 결측치 처리
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 결측치 처리")
print("=" * 60)

df_clean = df.copy()

# age: 중앙값 대체 (22명, 5.2%)
age_median = df_clean["age"].median()
df_clean["age"] = df_clean["age"].fillna(age_median)
print(f"  age               : {df['age'].isna().sum()}명 결측 → 중앙값 {age_median:.1f} yrs")

# clinical.T.Stage: 최빈값 대체 (1명, 0.2%)
t_mode = df_clean["clinical.T.Stage"].mode()[0]
df_clean["clinical.T.Stage"] = df_clean["clinical.T.Stage"].fillna(t_mode)
print(f"  clinical.T.Stage  : {df['clinical.T.Stage'].isna().sum()}명 결측 → 최빈값 {t_mode}")

# Histology: 'unknown' 범주 추가 (42명, 10%)
df_clean["Histology"] = df_clean["Histology"].fillna("unknown")
print(f"  Histology         : {df['Histology'].isna().sum()}명 결측 → 'unknown' 범주")

# Overall.Stage: 최빈값 대체 (1명, 0.2%)
stage_mode = df_clean["Overall.Stage"].mode()[0]
df_clean["Overall.Stage"] = df_clean["Overall.Stage"].fillna(stage_mode)
print(f"  Overall.Stage     : {df['Overall.Stage'].isna().sum()}명 결측 → '{stage_mode}'")

print(f"\n  처리 후 총 결측치: {df_clean.isna().sum().sum()}")

# ─────────────────────────────────────────────
# 4. PatientID 인덱스 설정
# ─────────────────────────────────────────────
df_clean = df_clean.set_index("PatientID")

# ─────────────────────────────────────────────
# 5. 2년 생존 레이블 생성 (목표 변수)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 2년 생존 레이블 생성 (목표 변수)")
print("=" * 60)

def assign_2yr_label(row):
    if row["Survival.time"] >= THRESHOLD:
        return 1          # 2년 이상 생존
    elif row["deadstatus.event"] == 1:
        return 0          # 2년 이내 사망 확인
    else:
        return np.nan     # 중도절단 → 제외

df_clean["label_2yr"] = df_clean.apply(assign_2yr_label, axis=1)

n_total    = len(df_clean)
n_survive  = int((df_clean["label_2yr"] == 1).sum())
n_dead     = int((df_clean["label_2yr"] == 0).sum())
n_excluded = int(df_clean["label_2yr"].isna().sum())
n_usable   = n_survive + n_dead

print(f"  기준             : {THRESHOLD}일 (2년)")
print(f"  label=1 (2yr+)  : {n_survive}명  ({n_survive/n_total*100:.1f}%)")
print(f"  label=0 (<2yr)  : {n_dead}명  ({n_dead/n_total*100:.1f}%)")
print(f"  NaN (censored)  : {n_excluded}명  ({n_excluded/n_total*100:.1f}%)")
print(f"  학습 사용 가능   : {n_usable}명")
print(f"  클래스 비율      : {n_survive}:{n_dead}  "
      f"(pos {n_survive/n_usable*100:.1f}% / neg {n_dead/n_usable*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. 생존 변수 요약
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 생존 변수 요약")
print("=" * 60)

print(f"  전체 환자        : {n_total}명")
print(f"  사망 (event=1)  : {int(df_clean['deadstatus.event'].sum())}명 "
      f"({df_clean['deadstatus.event'].mean()*100:.1f}%)")
print(f"  생존기간 중앙값  : {df_clean['Survival.time'].median():.0f}일 "
      f"({df_clean['Survival.time'].median()/30.4:.1f}개월)")
print(f"  생존기간 범위    : {df_clean['Survival.time'].min():.0f} ~ "
      f"{df_clean['Survival.time'].max():.0f}일")

# ─────────────────────────────────────────────
# 7. 범주형 인코딩
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 범주형 인코딩")
print("=" * 60)

df_encoded = df_clean.copy()

# gender: male=1, female=0
df_encoded["gender"] = df_encoded["gender"].map({"male": 1, "female": 0})
print("  gender       : male=1, female=0")

# Overall.Stage: 순서형 인코딩
stage_map = {"I": 1, "II": 2, "IIIa": 3, "IIIb": 4}
df_encoded["Overall.Stage"] = df_encoded["Overall.Stage"].map(stage_map)
print(f"  Overall.Stage: {stage_map}")

# Histology: One-Hot Encoding
hist_dummies = pd.get_dummies(df_encoded["Histology"], prefix="hist")
df_encoded = pd.concat([df_encoded.drop("Histology", axis=1), hist_dummies], axis=1)
print(f"  Histology    : One-Hot → {hist_dummies.columns.tolist()}")

# clinical.T.Stage: 정수 통일
df_encoded["clinical.T.Stage"] = df_encoded["clinical.T.Stage"].astype(int)

print(f"\n  최종 컬럼: {df_encoded.columns.tolist()}")
print(f"  Shape: {df_encoded.shape}")

# ─────────────────────────────────────────────
# 8. 시각화
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. 시각화 저장")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("NSCLC-Radiomics Lung1 - Clinical Data Overview", fontsize=14, fontweight="bold")

# (1) 생존 기간 분포
axes[0, 0].hist(df_clean["Survival.time"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0, 0].axvline(THRESHOLD, color="orange", linestyle="--", linewidth=2, label="2yr (730d)")
axes[0, 0].axvline(df_clean["Survival.time"].median(), color="red", linestyle="--",
                   label=f"Median: {df_clean['Survival.time'].median():.0f}d")
axes[0, 0].set_title("Survival Time Distribution")
axes[0, 0].set_xlabel("Days")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend(fontsize=8)

# (2) 2년 생존 레이블 분포
label_counts = df_clean["label_2yr"].value_counts(dropna=False)
labels_str   = ["< 2yr (label=0)", ">= 2yr (label=1)", "Censored (NaN)"]
values       = [n_dead, n_survive, n_excluded]
colors_pie   = ["#e74c3c", "#2ecc71", "#95a5a6"]
axes[0, 1].bar(labels_str, values, color=colors_pie, edgecolor="white")
axes[0, 1].set_title("2-Year Survival Label")
axes[0, 1].set_ylabel("Count")
for i, v in enumerate(values):
    axes[0, 1].text(i, v + 2, str(v), ha="center", fontsize=10)

# (3) 병기 분포
stage_counts = df_clean["Overall.Stage"].value_counts().reindex(["I", "II", "IIIa", "IIIb"])
axes[0, 2].bar(stage_counts.index, stage_counts.values,
               color=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"], edgecolor="white")
axes[0, 2].set_title("Overall Stage Distribution")
axes[0, 2].set_xlabel("Stage")
axes[0, 2].set_ylabel("Count")
for i, v in enumerate(stage_counts.values):
    axes[0, 2].text(i, v + 2, str(v), ha="center", fontsize=10)

# (4) 조직형 분포
hist_counts = df_clean["Histology"].value_counts()
bar_colors  = ["#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#95a5a6"]
axes[1, 0].barh(hist_counts.index, hist_counts.values,
                color=bar_colors[:len(hist_counts)], edgecolor="white")
axes[1, 0].set_title("Histology Distribution")
axes[1, 0].set_xlabel("Count")
for i, v in enumerate(hist_counts.values):
    axes[1, 0].text(v + 1, i, str(v), va="center", fontsize=9)

# (5) 나이 분포
axes[1, 1].hist(df_clean["age"], bins=25, color="darkorange", edgecolor="white", alpha=0.85)
axes[1, 1].axvline(df_clean["age"].median(), color="red", linestyle="--",
                   label=f"Median: {df_clean['age'].median():.1f} yrs")
axes[1, 1].set_title("Age Distribution")
axes[1, 1].set_xlabel("Age (years)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend(fontsize=8)

# (6) 병기별 생존기간 박스플롯
stage_order_list = ["I", "II", "IIIa", "IIIb"]
data_by_stage = [
    df_clean[df_clean["Overall.Stage"] == s]["Survival.time"].dropna()
    for s in stage_order_list
]
bp = axes[1, 2].boxplot(data_by_stage, tick_labels=stage_order_list, patch_artist=True)
box_colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[1, 2].axhline(THRESHOLD, color="orange", linestyle="--", linewidth=1.5, label="2yr threshold")
axes[1, 2].set_title("Survival Time by Stage")
axes[1, 2].set_xlabel("Stage")
axes[1, 2].set_ylabel("Days")
axes[1, 2].legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "clinical_overview.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  시각화 저장: {fig_path}")

# ─────────────────────────────────────────────
# 9. 결과 저장
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. 결과 저장")
print("=" * 60)

cleaned_path = os.path.join(OUTPUT_DIR, "lung1_clinical_cleaned.csv")
encoded_path = os.path.join(OUTPUT_DIR, "lung1_clinical_encoded.csv")

df_clean.to_csv(cleaned_path)
df_encoded.to_csv(encoded_path)

print(f"  정제 데이터:   {cleaned_path}  {df_clean.shape}")
print(f"  인코딩 데이터: {encoded_path}  {df_encoded.shape}")
print("\n✅ 전처리 완료!")