"""
NSCLC-Radiomics Lung1 - Clinical Data ML Baseline
목표: 2년 생존 예측 (label_2yr)
모델: Logistic Regression, Random Forest, XGBoost, SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             classification_report, roc_curve, confusion_matrix)
from xgboost import XGBClassifier

matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_PATH  = "output_clinical/lung1_clinical_encoded.csv"
OUTPUT_DIR = "output_ml"
SEED       = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드 및 준비
# ─────────────────────────────────────────────
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col="PatientID")

# 데이터 누수(leakage) 방지: Survival.time, deadstatus.event 제거
# (이 두 변수는 label_2yr을 만드는 데 사용됐기 때문에 모델 입력에서 제외)
LEAKAGE_COLS = ["Survival.time", "deadstatus.event"]
TARGET_COL   = "label_2yr"

# NaN 레이블(중도절단 2명) 제거
df = df.dropna(subset=[TARGET_COL])

feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS + [TARGET_COL]]
X = df[feature_cols].fillna(0)
# bool 컬럼 → int 변환 (get_dummies 결과물)
X = X.apply(lambda c: c.astype(int) if c.dtype == bool else c)
y = df[TARGET_COL].astype(int)

print(f"  Features : {feature_cols}")
print(f"  Shape    : X={X.shape}, y={y.shape}")
print(f"  Label 분포: 0(사망)={int((y==0).sum())}명, 1(생존)={int((y==1).sum())}명")

# ─────────────────────────────────────────────
# 2. Train / Test Split (stratified 80:20)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Train/Test Split (80:20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"  Train: {X_train.shape[0]}명  |  Test: {X_test.shape[0]}명")

# LR / SVM은 스케일링 필요
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. 모델 정의
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced"),
    "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                         random_state=SEED, eval_metric="logloss",
                                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=SEED, class_weight="balanced"),
}

# 스케일링이 필요한 모델
scaled_models = {"Logistic Regression", "SVM"}

# ─────────────────────────────────────────────
# 4. 학습 및 평가
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 모델 학습 및 평가")
print("=" * 60)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, model in models.items():
    Xtr = X_train_sc if name in scaled_models else X_train.values
    Xte = X_test_sc  if name in scaled_models else X_test.values

    # 학습
    model.fit(Xtr, y_train)

    # 예측
    y_pred      = model.predict(Xte)
    y_prob      = model.predict_proba(Xte)[:, 1]

    # 5-Fold CV AUC
    X_cv = X_train_sc if name in scaled_models else X_train.values
    cv_auc = cross_val_score(model, X_cv, y_train, cv=cv, scoring="roc_auc").mean()

    # 메트릭
    auc  = roc_auc_score(y_test, y_prob)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "AUC": auc, "CV_AUC": cv_auc, "Accuracy": acc, "F1": f1
    }

    print(f"\n  [{name}]")
    print(f"    AUC (test)   : {auc:.4f}")
    print(f"    AUC (5-CV)   : {cv_auc:.4f}")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    F1 Score     : {f1:.4f}")

# ─────────────────────────────────────────────
# 5. 결과 요약 테이블
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 결과 요약")
print("=" * 60)

summary = pd.DataFrame({
    name: {"AUC(test)": v["AUC"], "AUC(5-CV)": v["CV_AUC"],
           "Accuracy": v["Accuracy"], "F1": v["F1"]}
    for name, v in results.items()
}).T.round(4)

print(summary.to_string())
summary.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

# ─────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 시각화 저장")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Clinical Baseline - Model Comparison", fontsize=13, fontweight="bold")

colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

# (1) ROC Curve
for (name, res), c in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.3f})", color=c, linewidth=2)
axes[0].plot([0,1],[0,1], "k--", linewidth=1)
axes[0].set_title("ROC Curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# (2) AUC 비교 바 차트
names = list(results.keys())
aucs  = [results[n]["AUC"] for n in names]
cv_aucs = [results[n]["CV_AUC"] for n in names]
x = np.arange(len(names))
w = 0.35
axes[1].bar(x - w/2, aucs, w, label="Test AUC", color=colors, alpha=0.85, edgecolor="white")
axes[1].bar(x + w/2, cv_aucs, w, label="5-CV AUC", color=colors, alpha=0.45, edgecolor="white", hatch="//")
axes[1].set_xticks(x); axes[1].set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
axes[1].set_ylim(0.5, 1.0); axes[1].set_title("AUC Comparison")
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
for i, (a, ca) in enumerate(zip(aucs, cv_aucs)):
    axes[1].text(i-w/2, a+0.01, f"{a:.3f}", ha="center", fontsize=8)
    axes[1].text(i+w/2, ca+0.01, f"{ca:.3f}", ha="center", fontsize=8)

# (3) Feature Importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
top_n = importances.tail(10)
axes[2].barh(top_n.index, top_n.values, color="#2ecc71", edgecolor="white", alpha=0.85)
axes[2].set_title("RF Feature Importance (Top 10)"); axes[2].set_xlabel("Importance")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone\!")
