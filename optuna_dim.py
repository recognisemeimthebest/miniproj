import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ── 데이터 로드 ──────────────────────────────────────
df_features = pd.read_csv(r"C:\Users\301-13\radiomics_project\features\final_features.csv")
df_clinical  = pd.read_csv(r"C:\Users\301-13\radiomics_project\lung1_clinical_encoded.csv")

df = pd.merge(df_features, df_clinical[["PatientID", "label_2yr"]],
              left_on="patient_id", right_on="PatientID", how="inner")
df = df.dropna(subset=["label_2yr"])

feature_cols = [c for c in df_features.columns if c != "patient_id"]
X = df[feature_cols].fillna(0).values
y = df["label_2yr"].values

# ── 표준화 ────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Val 분할 ──────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# ── MLP 모델 ─────────────────────────────────────────
class RadiomicsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x).squeeze()

# ── Optuna 목적 함수 ──────────────────────────────────
def objective(trial):
    embed_dim  = trial.suggest_categorical("embed_dim", [16, 32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    dropout    = trial.suggest_float("dropout", 0.1, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs     = trial.suggest_int("epochs", 30, 100)

    model = RadiomicsMLP(
        input_dim=15,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 학습
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        val_prob = model(X_val_t).numpy()

    auroc = roc_auc_score(y_val, val_prob)
    return auroc

# ── Optuna 실행 ───────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# ── 결과 출력 ─────────────────────────────────────────
print("\n========== Optuna 최적 결과 ==========")
print(f"최적 Val AUROC : {study.best_value:.4f}")
print(f"최적 embed_dim : {study.best_params['embed_dim']}")
print(f"최적 hidden_dim: {study.best_params['hidden_dim']}")
print(f"최적 dropout   : {study.best_params['dropout']:.4f}")
print(f"최적 lr        : {study.best_params['lr']:.6f}")
print(f"최적 epochs    : {study.best_params['epochs']}")