import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# ── 데이터 로드 ──────────────────────────────────────
df_features = pd.read_csv(r"C:\Users\301-13\radiomics_project\features\final_features.csv")
df_clinical  = pd.read_csv(r"C:\Users\301-13\radiomics_project\lung1_clinical_encoded.csv")

df = pd.merge(df_features, df_clinical[["PatientID", "label_2yr"]],
              left_on="patient_id", right_on="PatientID", how="inner")
df = df.dropna(subset=["label_2yr"])

feature_cols = [c for c in df_features.columns if c != "patient_id"]
X = df[feature_cols].fillna(0).values

print(f"환자 수: {len(X)}명")
print(f"입력 특징 수: {X.shape[1]}개")

# ── 표준화 ────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── MLP: 15 → 256dim (Optuna 최적값) ─────────────────
class RadiomicsMLP(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, embed_dim=256, dropout=0.1409):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# ── 임베딩 추출 ───────────────────────────────────────
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
model = RadiomicsMLP(input_dim=15, hidden_dim=128, embed_dim=256, dropout=0.1409)
model.eval()

with torch.no_grad():
    embeddings = model(X_tensor).numpy()

print(f"\n입력 차원: {X_scaled.shape}  →  (환자수, 15)")
print(f"출력 차원: {embeddings.shape}  →  (환자수, 256)")

# ── 저장 ─────────────────────────────────────────────
np.save(r"C:\Users\301-13\radiomics_project\features\radiomics_embed_256.npy", embeddings)

df_embed = pd.DataFrame(embeddings, columns=[f"radio_dim_{i}" for i in range(256)])
df_embed.insert(0, "patient_id", df["patient_id"].values)
df_embed.to_csv(r"C:\Users\301-13\radiomics_project\features\radiomics_embed_256.csv", index=False)

print("\n저장 완료!")
print("  - radiomics_embed_256.npy  ← Fusion 모델에 투입할 파일")
print("  - radiomics_embed_256.csv  ← 확인용 파일")