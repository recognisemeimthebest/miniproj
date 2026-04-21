import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# ── 데이터 로드
df_features = pd.read_csv(r"C:\Users\301-13\radiomics_project\features\final_features.csv")
df_clinical = pd.read_csv(r"C:\Users\301-13\radiomics_project\lung1_clinical_encoded.csv")

df = pd.merge(df_features, df_clinical[["PatientID", "label_2yr"]],
              left_on="patient_id", right_on="PatientID", how="inner")
df = df.dropna(subset=["label_2yr"])

feature_cols = [c for c in df_features.columns if c != "patient_id"]
X = df[feature_cols].fillna(0).values
y = df["label_2yr"].values

# ── 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── MLP 모델 정의
class RadiomicsMLP(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, embed_dim=256, dropout=0.1409):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.classifier(self.embed(x)).squeeze()

# ── 학습
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = RadiomicsMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003147)
criterion = nn.BCEWithLogitsLoss()

print("모델 학습 중...")
model.train()
for epoch in range(84):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/84 - Loss: {loss.item():.4f}")

# ── 저장
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": 15,
    "hidden_dim": 128,
    "embed_dim": 256,
    "dropout": 0.1409,
    "lr": 0.003147,
    "epochs": 84,
    "optuna_val_auroc": 0.6400,
}, r"C:\Users\301-13\radiomics_project\radiomics_mlp_256dim.pt")

print("\n저장 완료!")
print("  - radiomics_mlp_256dim.pt")