"""LUNA-XAI Streamlit demo (Option A — 임베딩 lookup 모드).

실행:
    cd /home/team4/miniproj_ljw/web
    streamlit run app.py --server.port 8501

입력: Lung1 test/val/train 환자 ID 선택 (63 + 64 + 293 = 420명).
파이프라인: 사전 추출된 640-dim 임베딩 lookup → TripleLinearL2(M4) → 생존 확률.
비교군: M1(Clinical-only 로지스틱), M3(Clinical+Radiomics 로지스틱).
XAI: 선형 모델이므로 SHAP = coef * (x - mean)/scale per-modality 기여도로 분해.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

# ── 경로 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "triple_linear_l2_best.pt"
EMB_DIR = ROOT / "embeddings"
GRADCAM_DIR = ROOT.parent / "figures" / "gradcam_overlays_iter0"
CLINICAL_CSV = Path("/home/team4/LJW/metadata/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")

MODALITIES = ["clinical", "radiomics", "ct"]
SPLITS = ("train", "val", "test")

st.set_page_config(
    page_title="LUNA-XAI | NSCLC 2yr Survival",
    page_icon="🫁",
    layout="wide",
)


# ── 모델 ────────────────────────────────────────────────────────────
class TripleLinearL2(nn.Module):
    def __init__(self, in_dim: int = 640):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("scale", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((x - self.mean) / self.scale).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


@st.cache_resource
def load_model():
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    m = TripleLinearL2(in_dim=ck["in_dim"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@st.cache_data
def load_embeddings():
    """Returns dict: split -> pid -> {clinical,radiomics,ct,y,concat}."""
    banks = {
        mod: {s: np.load(EMB_DIR / f"{mod}_{s}.npz", allow_pickle=True) for s in SPLITS}
        for mod in MODALITIES
    }
    out = {}
    for s in SPLITS:
        ref_pids = [str(p) for p in banks["clinical"][s]["pids"]]
        ref_y = banks["clinical"][s]["y"].astype(int)
        pid_records = {}
        # build index for each modality
        idx = {
            m: {str(p): i for i, p in enumerate(banks[m][s]["pids"])} for m in MODALITIES
        }
        for pid_i, pid in enumerate(ref_pids):
            parts = []
            rec = {}
            for m in MODALITIES:
                j = idx[m][pid]
                e = banks[m][s]["emb"][j].astype(np.float32)
                rec[m] = e
                parts.append(e)
            rec["y"] = int(ref_y[pid_i])
            rec["concat"] = np.concatenate(parts, axis=0)  # (640,)
            pid_records[pid] = rec
        out[s] = pid_records
    return out


@st.cache_data
def load_clinical_csv() -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_CSV)
    df["PatientID"] = df["PatientID"].astype(str)
    return df


# ── 비교군 로지스틱 (on-the-fly fit, 임베딩 레벨) ──────────────────
@st.cache_resource
def fit_baselines(_emb_cache):
    """trainval 로지스틱: Clinical only(M1-style) & Clinical+Radiomics(M3-style)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def stack(split_names, dims):
        X, y = [], []
        for s in split_names:
            for pid, rec in _emb_cache[s].items():
                X.append(np.concatenate([rec[m] for m in dims]))
                y.append(rec["y"])
        return np.asarray(X, np.float32), np.asarray(y, int)

    models = {}
    for name, dims in (("M1 Clinical", ["clinical"]),
                       ("M3 Clin+Rad", ["clinical", "radiomics"])):
        Xtv, ytv = stack(["train", "val"], dims)
        scaler = StandardScaler().fit(Xtv)
        clf = LogisticRegression(
            penalty="l2", C=0.01, class_weight="balanced",
            max_iter=2000, random_state=99,
        ).fit(scaler.transform(Xtv), ytv)
        models[name] = (scaler, clf, dims)
    return models


# ── UI ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1f6feb,#388bfd);padding:12px 20px;
                border-radius:10px;color:#fff;margin-bottom:10px;">
      <h2 style="margin:0">🫁 LUNA-XAI  <span style="font-weight:400;opacity:.85;font-size:16px">
      | NSCLC 2-Year Survival Prediction (demo)</span></h2>
    </div>
    """,
    unsafe_allow_html=True,
)

model, ckpt = load_model()
emb_cache = load_embeddings()
clinical_df = load_clinical_csv()
baselines = fit_baselines(emb_cache)

# ── Sidebar: 환자 선택 ──────────────────────────────────────────────
st.sidebar.header("📋 환자 선택")
split = st.sidebar.selectbox("Split", SPLITS, index=SPLITS.index("test"))
pid_list = sorted(emb_cache[split].keys())
pid = st.sidebar.selectbox(f"Patient ID ({len(pid_list)}명)", pid_list)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"모델: **TripleLinearL2 (M4)**  \n"
    f"Test AUROC: **{ckpt['metrics']['test_auroc_torch']:.4f}**  \n"
    f"입력 차원: {ckpt['in_dim']} "
    f"(clin={ckpt['modality_dims']['clinical']}, "
    f"rad={ckpt['modality_dims']['radiomics']}, "
    f"ct={ckpt['modality_dims']['ct']})"
)

# ── 현재 환자 정보 ─────────────────────────────────────────────────
rec = emb_cache[split][pid]
row = clinical_df[clinical_df["PatientID"] == pid]
clin_info = row.iloc[0].to_dict() if len(row) else {}

# ── 메인 레이아웃 ──────────────────────────────────────────────────
left, center, right = st.columns([1.0, 1.3, 1.2])

# ===== LEFT: 임상 정보 =====
with left:
    st.subheader("🧑‍⚕️ 임상 정보")
    if clin_info:
        age = clin_info.get("age", float("nan"))
        st.metric("Age", f"{age:.1f}" if pd.notna(age) else "—")
        c1, c2 = st.columns(2)
        c1.metric("Gender", str(clin_info.get("gender", "—")))
        c2.metric("Stage", str(clin_info.get("Overall.Stage", "—")))
        st.write(
            f"**Histology**: {clin_info.get('Histology','—')}  \n"
            f"**TNM**: T{int(clin_info.get('clinical.T.Stage', 0))} "
            f"N{int(clin_info.get('Clinical.N.Stage', 0))} "
            f"M{int(clin_info.get('Clinical.M.Stage', 0))}"
        )
        st.markdown("---")
        surv_days = clin_info.get("Survival.time", np.nan)
        dead = clin_info.get("deadstatus.event", np.nan)
        st.caption("Ground truth (표시용)")
        c1, c2 = st.columns(2)
        c1.metric("Survival (d)", f"{int(surv_days)}" if pd.notna(surv_days) else "—")
        c2.metric("2yr label y", "생존" if rec["y"] == 1 else "사망")
    else:
        st.info("임상 CSV에서 해당 환자 레코드를 찾지 못했습니다.")

# ===== CENTER: 예측 결과 =====
with center:
    st.subheader("📈 M4 예측 — 2년 생존 확률")
    x = torch.from_numpy(rec["concat"]).unsqueeze(0)
    with torch.no_grad():
        prob_m4 = float(model.predict_proba(x).item())

    # 바이너리 분류 확률이지만, 보여주는 건 "2년 생존 확률"로 해석
    pct = prob_m4 * 100
    bg = "#238636" if pct >= 60 else ("#d29922" if pct >= 40 else "#da3633")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{bg}cc,{bg});color:#fff;
             border-radius:12px;padding:18px;text-align:center">
          <div style="font-size:12px;letter-spacing:2px;opacity:.85">2-YEAR SURVIVAL PROB</div>
          <div style="font-size:60px;font-weight:700;line-height:1.1">{pct:.1f}<span style="font-size:24px">%</span></div>
          <div style="font-size:12px;opacity:.85">
            logit = {float(model(x).item()):.3f} · chosen C = {ckpt['chosen_C']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 모델별 비교
    st.markdown("##### 📊 모델별 비교 (동일 환자)")
    comp = {"M4 Fusion (CT+Rad+Clin)": prob_m4}
    for name, (sc, clf, dims) in baselines.items():
        xb = np.concatenate([rec[m] for m in dims]).reshape(1, -1)
        comp[name] = float(clf.predict_proba(sc.transform(xb))[0, 1])

    fig = go.Figure(
        go.Bar(
            x=list(comp.values()),
            y=list(comp.keys()),
            orientation="h",
            marker_color=["#3fb950", "#58a6ff", "#8957e5"],
            text=[f"{v*100:.1f}%" for v in comp.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=220, margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 참고 — 이 데모의 입력/모델 동작"):
        st.markdown(
            f"""
            - **데이터**: Lung1 공개 셋 {sum(len(v) for v in emb_cache.values())}명을
              train/val/test 분할 (train 293 · val 64 · test 63)
            - **입력**: 사전 추출된 concat 임베딩 clinical(128)+radiomics(256)+ct(256) = **640-dim**
            - **모델**: StandardScaler → Linear(640,1) → sigmoid (L2 regularized, CV로 C={ckpt['chosen_C']})
            - **성능**: Test AUROC = **{ckpt['metrics']['test_auroc_torch']:.4f}**
              ({ckpt['encoder_source']})
            - **비고**: 원본 CT/임상값을 받아 실시간 인코딩하는 모드는 옵션 B에서 추가 예정
            """
        )

# ===== RIGHT: XAI (modality contribution) =====
with right:
    st.subheader("🔬 XAI — Modality 기여도")
    W = model.linear.weight.detach().numpy().reshape(-1)   # (640,)
    b = float(model.linear.bias.detach().item())
    x_std = (rec["concat"] - model.mean.numpy()) / model.scale.numpy()
    contrib_all = W * x_std  # (640,)

    slices = ckpt["modality_slices"]
    per_mod = {m: float(contrib_all[a:b_].sum()) for m, (a, b_) in slices.items()}

    fig2 = go.Figure(
        go.Bar(
            x=[v for v in per_mod.values()],
            y=[m.upper() for m in per_mod.keys()],
            orientation="h",
            marker_color=["#3fb950" if v >= 0 else "#f85149" for v in per_mod.values()],
            text=[f"{v:+.3f}" for v in per_mod.values()],
            textposition="outside",
        )
    )
    fig2.add_vline(x=0, line_color="#484f58", line_width=1)
    fig2.update_layout(
        height=180, margin=dict(l=10, r=30, t=10, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        xaxis_title="logit 기여도 (+ 생존 / − 사망)",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"모든 기여도 합 + bias({b:+.3f}) = logit")

    # Grad-CAM overlay (pre-generated, test split only)
    st.markdown("##### 🖼️ Grad-CAM — CT overlay")
    gc_path = GRADCAM_DIR / f"{pid}.png"
    if gc_path.exists():
        st.image(
            str(gc_path),
            caption=f"{pid} · CT axial slice + Grad-CAM (CT encoder, iter0 · GradCAM+relu_minmax)",
            use_container_width=True,
        )
    else:
        st.info(
            f"Grad-CAM 오버레이는 **test split 63명**에 대해서만 사전 생성되어 있습니다.  \n"
            f"현재 선택: split=`{split}`, pid=`{pid}`"
        )

# ── footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **연구용 프로토타입**. 본 결과는 실제 임상 의사결정에 사용할 수 없습니다. "
    "M4 = TripleLinearL2 (feature/LJW), seed=99, trainval refit. "
    f"test AUROC = {ckpt['metrics']['test_auroc_torch']:.4f}."
)
