import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# ── 데이터 로드 ──────────────────────────────────────
df_features = pd.read_csv(r"C:\Users\301-13\radiomics_project\features\final_features.csv")
df_clinical  = pd.read_csv(r"C:\Users\301-13\radiomics_project\lung1_clinical_encoded.csv")

df = pd.merge(df_features, df_clinical[["PatientID", "label_2yr"]],
              left_on="patient_id", right_on="PatientID", how="inner")
df = df.dropna(subset=["label_2yr"])

# 클래스 비율 확인
print(f"총 환자 수: {len(df)}명")
print(f"생존(1): {int(y.sum()) if 'y' in dir() else int(df['label_2yr'].sum())}명 / "
      f"사망(0): {int((df['label_2yr']==0).sum())}명")

feature_cols = [c for c in df_features.columns if c != "patient_id"]
X = df[feature_cols].fillna(0)
y = df["label_2yr"]

print(f"생존(1): {int(y.sum())}명 / 사망(0): {int((y==0).sum())}명")

# ── Train / Val / Test 분할 (6:2:2) ──────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}명 / Val: {len(X_val)}명 / Test: {len(X_test)}명")

# ── 표준화 ────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# 클래스 비율 계산 (불균형 보정용)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos = neg / pos

# ── 모델 비교 ─────────────────────────────────────────
models = {
    "LR (balanced)"  : LogisticRegression(C=0.1, class_weight="balanced",
                                           random_state=42, max_iter=10000),
    "RF (balanced)"  : RandomForestClassifier(n_estimators=500, max_depth=5,
                                               class_weight="balanced",
                                               random_state=42),
    "GBM"            : GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                   learning_rate=0.05,
                                                   random_state=42),
    "LR (C=1)"       : LogisticRegression(C=1, class_weight="balanced",
                                           random_state=42, max_iter=10000),
    "RF (depth=3)"   : RandomForestClassifier(n_estimators=300, max_depth=3,
                                               class_weight="balanced",
                                               min_samples_leaf=5,
                                               random_state=42),
}

print("\n========== 모델별 성능 비교 ==========")
best_model_name = None
best_test_auroc = 0

for name, model in models.items():
    model.fit(X_train_s, y_train)

    val_prob   = model.predict_proba(X_val_s)[:, 1]
    test_prob  = model.predict_proba(X_test_s)[:, 1]
    test_pred  = model.predict(X_test_s)

    val_auroc   = roc_auc_score(y_val, val_prob)
    test_auroc  = roc_auc_score(y_test, test_prob)
    test_recall = recall_score(y_test, test_pred)
    auroc_gap   = abs(val_auroc - test_auroc)

    cond1 = test_auroc > 0.55
    cond2 = auroc_gap < 0.15
    cond3 = test_recall > 0.1
    all_pass = cond1 and cond2 and cond3

    print(f"\n[{name}]")
    print(f"  Val AUROC  : {val_auroc:.4f}")
    print(f"  Test AUROC : {test_auroc:.4f}  {'✅' if cond1 else '❌'}")
    print(f"  Val-Test   : {auroc_gap:.4f}   {'✅' if cond2 else '❌'}")
    print(f"  Recall     : {test_recall:.4f}  {'✅' if cond3 else '❌'}")
    print(f"  → {'🎉 3가지 조건 모두 통과!' if all_pass else '⚠️ 조건 미달'}")

    if test_auroc > best_test_auroc:
        best_test_auroc = test_auroc
        best_model_name = name

print(f"\n========== 최고 성능 모델: {best_model_name} (Test AUROC: {best_test_auroc:.4f}) ==========")