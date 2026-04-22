"""
NSCLC-Radiomics Lung1 - Clinical Branch (MLP) with Optuna Tuning
=================================================================
역할: 임상 데이터 11개 변수 → embedding 벡터 변환
      Fusion 모델의 임상 브랜치로 사용

출력 파일:
  output_clinical_branch/best_params.json   → 최적 하이퍼파라미터
  output_clinical_branch/clinical_branch.pt → 최적 모델 가중치
  output_clinical_branch/optuna_results.png → 튜닝 결과 시각화
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_PATH  = "output_clinical/lung1_clinical_encoded.csv"
OUTPUT_DIR = "output_clinical_branch"
SEED       = 42
N_TRIALS   = 50    # Optuna 시도 횟수 (시간 있으면 100으로)
N_EPOCHS   = 100
N_FOLDS    = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

LEAKAGE = ["Survival.time", "deadstatus.event"]
TARGET  = "label_2yr"

# ─────────────────────────────────────────────
# 1. 데이터 준비
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, index_col="PatientID")
df = df.dropna(subset=[TARGET])

feat_cols = [c for c in df.columns if c not in LEAKAGE + [TARGET]]
X_raw = df[feat_cols].fillna(0)
X_raw = X_raw.apply(lambda c: c.astype(int) if c.dtype == bool else c)
X_raw = X_raw.values.astype(np.float32)
y_raw = df[TARGET].values.astype(np.float32)

INPUT_DIM = X_raw.shape[1]
print(f"Input dim: {INPUT_DIM}, Samples: {len(y_raw)}")
print(f"Label: 0={int((y_raw==0).sum())}, 1={int((y_raw==1).sum())}")

# ─────────────────────────────────────────────
# 2. Dataset / DataLoader
# ─────────────────────────────────────────────
class ClinicalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ─────────────────────────────────────────────
# 3. ClinicalBranch 모델 정의
# ─────────────────────────────────────────────
class ClinicalBranch(nn.Module):
    """
    임상 데이터 MLP 인코더
    Input:  (batch, input_dim)  ← 11개 임상 변수
    Output: (batch, embed_dim)  ← Fusion에 넘겨줄 embedding
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
            # ← 여기서 끝 (sigmoid 없음 — Fusion 이후에 붙임)
        )
        # 단독 학습용 분류 헤드 (튜닝 시에만 사용, fusion 시엔 제거)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_embedding=False):
        emb = self.encoder(x)
        if return_embedding:
            return emb                    # Fusion 모델에서 사용
        return self.classifier(emb)       # 단독 학습 시 사용

# ─────────────────────────────────────────────
# 4. 학습 함수
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b).squeeze()
        loss = criterion(pred, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_auc(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = model(X_b.to(device)).squeeze().cpu().numpy()
            preds.extend(p if p.ndim > 0 else [p])
            labels.extend(y_b.numpy())
    return roc_auc_score(labels, preds)

# ─────────────────────────────────────────────
# 5. Optuna Objective
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 클래스 불균형 보정 가중치
pos_weight = torch.tensor([(y_raw==0).sum() / (y_raw==1).sum()]).to(device)

def objective(trial):
    # 탐색할 하이퍼파라미터
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    embed_dim  = trial.suggest_categorical("embed_dim",  [32, 64, 128])
    dropout    = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y_raw)):
        # 스케일링 (fold 내부에서)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr_idx])
        X_val = scaler.transform(X_raw[val_idx])

        tr_loader  = DataLoader(ClinicalDataset(X_tr, y_raw[tr_idx]),
                                batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ClinicalDataset(X_val, y_raw[val_idx]),
                                batch_size=batch_size)

        model = ClinicalBranch(INPUT_DIM, hidden_dim, embed_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(N_EPOCHS):
            train_epoch(model, tr_loader, optimizer, criterion, device)

        auc = eval_auc(model, val_loader, device)
        fold_aucs.append(auc)

    return np.mean(fold_aucs)

# ─────────────────────────────────────────────
# 6. Optuna 실행
# ─────────────────────────────────────────────
print(f"\nOptuna 탐색 시작 ({N_TRIALS} trials, {N_FOLDS}-Fold CV)...")
sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n최적 AUC (CV): {study.best_value:.4f}")
print(f"최적 파라미터: {study.best_params}")

# 결과 저장
params_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(params_path, "w") as f:
    json.dump({"best_auc_cv": study.best_value, **study.best_params}, f, indent=2)
print(f"파라미터 저장: {params_path}")

# ─────────────────────────────────────────────
# 7. 최적 파라미터로 전체 데이터 재학습
# ─────────────────────────────────────────────
print("\n최적 파라미터로 전체 데이터 재학습...")
bp = study.best_params
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_raw).astype(np.float32)

full_loader = DataLoader(ClinicalDataset(X_scaled, y_raw),
                         batch_size=bp["batch_size"], shuffle=True)

best_model = ClinicalBranch(INPUT_DIM, bp["hidden_dim"], bp["embed_dim"], bp["dropout"]).to(device)
optimizer  = torch.optim.Adam(best_model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(N_EPOCHS):
    loss = train_epoch(best_model, full_loader, optimizer, criterion, device)
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  loss={loss:.4f}")

# 모델 저장 (embedding 부분만 — Fusion 팀원에게 전달)
model_path = os.path.join(OUTPUT_DIR, "clinical_branch.pt")
torch.save({
    "model_state_dict": best_model.state_dict(),
    "scaler_mean":      scaler_final.mean_.tolist(),
    "scaler_scale":     scaler_final.scale_.tolist(),
    "input_dim":        INPUT_DIM,
    "hidden_dim":       bp["hidden_dim"],
    "embed_dim":        bp["embed_dim"],
    "dropout":          bp["dropout"],
    "feature_cols":     feat_cols,
}, model_path)
print(f"모델 저장: {model_path}")

# ─────────────────────────────────────────────
# 8. 시각화
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt

trial_nums = [t.number for t in study.trials]
trial_aucs = [t.value for t in study.trials]
best_so_far = [max(trial_aucs[:i+1]) for i in range(len(trial_aucs))]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Optuna Hyperparameter Tuning - Clinical Branch", fontsize=13, fontweight="bold")

# (1) Trial별 AUC
axes[0].scatter(trial_nums, trial_aucs, alpha=0.5, color="steelblue", s=30, label="Trial AUC")
axes[0].plot(trial_nums, best_so_far, color="red", linewidth=2, label="Best so far")
axes[0].axhline(study.best_value, color="orange", linestyle="--",
                label=f"Best={study.best_value:.4f}")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("CV AUC")
axes[0].set_title("AUC per Trial"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# (2) 파라미터 중요도
try:
    importances = optuna.importance.get_param_importances(study)
    names  = list(importances.keys())
    values = list(importances.values())
    axes[1].barh(names, values, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].set_title("Parameter Importance"); axes[1].set_xlabel("Importance")
    axes[1].grid(axis="x", alpha=0.3)
except Exception:
    axes[1].text(0.5, 0.5, "Not enough trials\nfor importance", ha="center", va="center")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "optuna_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"시각화 저장: {fig_path}")

print("\n=== 완료 ===")
print(f"  embed_dim  : {bp['embed_dim']}  ← Fusion 팀원에게 전달")
print(f"  Best CV AUC: {study.best_value:.4f}")
