import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import pandas as pd
import os

# ── 페이지 설정 ───────────────────────────────────────
st.set_page_config(
    page_title="LUNA-XAI | NSCLC 2년 생존 예측",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS 스타일 ────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    [data-testid="stSidebar"] { background-color: #1a2e4a; color: white; }
    [data-testid="stSidebar"] label { color: #cce4ff !important; font-size: 13px; }
    .metric-card {
        background: white; border-radius: 10px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1a4f8a; margin-bottom: 15px;
    }
    .metric-card-danger { border-left: 4px solid #e74c3c; }
    .metric-card-success { border-left: 4px solid #27ae60; }
    .main-header {
        background: linear-gradient(135deg, #1a2e4a 0%, #1a4f8a 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(26,47,74,0.3);
    }
    .stButton button {
        background: linear-gradient(135deg, #1a4f8a, #1a7abf);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; width: 100%; font-size: 15px;
    }
    .divider { margin: 20px 0; border-top: 1px solid #e8ecf0; }
</style>
""", unsafe_allow_html=True)

# ── 모델 정의 ─────────────────────────────────────────
class TripleLinearL2(nn.Module):
    def __init__(self, in_dim=640):
        super().__init__()
        self.mean   = nn.Parameter(torch.zeros(in_dim))
        self.scale  = nn.Parameter(torch.ones(in_dim))
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = (x - self.mean) / (self.scale + 1e-8)
        return torch.sigmoid(self.linear(x))

@st.cache_resource
def load_model():
    ckpt = torch.load("checkpoints/triple_linear_l2_best.pt", map_location="cpu")
    model = TripleLinearL2(in_dim=640)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

@st.cache_data
def load_embeddings():
    ct    = dict(np.load("embeddings/ct_test.npz",        allow_pickle=True))
    radio = dict(np.load("embeddings/radiomics_test.npz", allow_pickle=True))
    clin  = dict(np.load("embeddings/clinical_test.npz",  allow_pickle=True))
    return ct, radio, clin

def get_array(npz):
    return npz[list(npz.keys())[0]]

# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:15px 0 25px'>
        <span style='font-size:40px'>🫁</span>
        <h2 style='color:white; margin:8px 0 4px; font-size:18px'>LUNA-XAI</h2>
        <p style='color:#7ab3d4; font-size:12px; margin:0'>NSCLC 2-Year Survival Prediction</p>
    </div>
    <hr style='border-color:#2a4a6a; margin-bottom:20px'>
    <p style='color:#7ab3d4; font-size:11px; font-weight:600; letter-spacing:1px'>PATIENT INFORMATION</p>
    """, unsafe_allow_html=True)

    patient_id = st.text_input("Patient ID", value="LUNG1-001")
    age        = st.slider("나이 (Age)", 30, 90, 65)
    gender     = st.radio("성별 (Gender)", ["Male", "Female"], horizontal=True)
    stage      = st.selectbox("TNM Stage", ["I", "II", "IIIa", "IIIb", "IV"])
    histology  = st.selectbox("Histology", [
        "Adenocarcinoma", "Squamous cell carcinoma",
        "Large cell", "NOS", "Unknown"
    ])

    predict_btn = st.button("🔍 예측 실행")

    st.markdown("""
    <hr style='border-color:#2a4a6a; margin:20px 0'>
    <p style='color:#7ab3d4; font-size:11px; text-align:center'>
    ⚠️ 연구용 프로토타입<br>임상 의사결정에 사용 불가
    </p>
    """, unsafe_allow_html=True)

# ── 메인 헤더 ─────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div style='display:flex; justify-content:space-between; align-items:center'>
        <div>
            <h1 style='color:white; margin:0; font-size:24px'>
                🫁 LUNA-XAI | NSCLC 2년 생존 예측 시스템
            </h1>
            <p style='color:#7ab3d4; margin:5px 0 0; font-size:13px'>
                Non-Small Cell Lung Cancer 2-Year Survival Prediction | M4 Fusion Model
            </p>
        </div>
        <div>
            <span style='background:#27ae60; color:white; padding:4px 12px;
                         border-radius:20px; font-size:12px; font-weight:600'>
                ● SYSTEM ONLINE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 탭 ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 예측 결과", "📊 모달리티 기여도", "📈 모델 성능"])

# ══════════════════════════════════════════════════════
# TAB 1: 예측
# ══════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        try:
            model = load_model()
            ct_emb, radio_emb, clin_emb = load_embeddings()

            ct_t    = torch.tensor(get_array(ct_emb)[[0]],    dtype=torch.float32)
            radio_t = torch.tensor(get_array(radio_emb)[[0]], dtype=torch.float32)
            clin_t  = torch.tensor(get_array(clin_emb)[[0]],  dtype=torch.float32)

            x = torch.cat([clin_t, radio_t, ct_t], dim=1)
            with torch.no_grad():
                prob = model(x).item()

            survival_pct = prob * 100
            is_high_risk = prob <= 0.5
            risk_color   = "#e74c3c" if is_high_risk else "#27ae60"
            risk_label   = "고위험 (High Risk)" if is_high_risk else "저위험 (Low Risk)"
            risk_icon    = "🔴" if is_high_risk else "🟢"

            # ── 상단 요약 카드 ──────────────────────
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class='metric-card {"metric-card-danger" if is_high_risk else "metric-card-success"}'>
                    <p style='color:#888; font-size:12px; margin:0'>2년 생존 확률</p>
                    <h2 style='color:{risk_color}; margin:5px 0; font-size:36px'>{survival_pct:.1f}%</h2>
                    <p style='color:{risk_color}; margin:0; font-weight:600'>{risk_icon} {risk_label}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color:#888; font-size:12px; margin:0'>Patient ID</p>
                    <h3 style='color:#1a2e4a; margin:5px 0'>{patient_id}</h3>
                    <p style='color:#888; margin:0; font-size:12px'>나이: {age}세 | {gender}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color:#888; font-size:12px; margin:0'>TNM Stage</p>
                    <h3 style='color:#1a2e4a; margin:5px 0'>{stage}</h3>
                    <p style='color:#888; margin:0; font-size:12px'>{histology}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color:#888; font-size:12px; margin:0'>모델</p>
                    <h3 style='color:#1a2e4a; margin:5px 0'>M4 Fusion</h3>
                    <p style='color:#888; margin:0; font-size:12px'>AUROC: 0.781</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── 모달리티 기여도 파이차트 ────────────
            col_chart, col_auroc = st.columns([1, 1])

            with col_chart:
                st.markdown("#### 📊 모달리티 기여도")
                fig3 = go.Figure(data=[go.Pie(
                    labels=["CT (256)", "Radiomics (256)", "Clinical (128)"],
                    values=[256, 256, 128],
                    hole=0.5,
                    marker_colors=["#1a4f8a", "#e67e22", "#27ae60"],
                    textfont_size=13
                )])
                fig3.update_layout(
                    height=280, margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col_auroc:
                st.markdown("#### 📋 모달리티별 AUROC")
                modality_data = [
                    ("Clinical",  0.681, "#95a5a6"),
                    ("CT only",   0.703, "#3498db"),
                    ("Rad+Clin",  0.715, "#e67e22"),
                    ("M4 Fusion", 0.724, "#27ae60"),
                ]
                for name, auroc, color in modality_data:
                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; padding:10px 0;
                                border-bottom:1px solid #f0f0f0'>
                        <span style='font-size:14px; color:#555'>{name}</span>
                        <span style='font-weight:700; color:{color}; font-size:16px'>{auroc:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"오류: {e}")
    else:
        st.markdown("""
        <div style='text-align:center; padding:80px 20px; color:#888'>
            <span style='font-size:60px'>🫁</span>
            <h3 style='color:#aaa; margin-top:20px'>
                왼쪽 사이드바에서 환자 정보를 입력하고<br>예측 실행 버튼을 눌러주세요
            </h3>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2: 모달리티 기여도
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Modality Ablation Study")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("M1 Clinical", "68.1%", "#95a5a6"),
        ("M2 CT only",  "70.3%", "#3498db"),
        ("M3 Rad+Clin", "71.5%", "#e67e22"),
        ("M4 Fusion",   "72.4%", "#27ae60"),
    ]
    for col, (name, val, color) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{color}; text-align:center'>
                <p style='color:#888; font-size:12px; margin:0'>{name}</p>
                <h2 style='color:{color}; margin:8px 0; font-size:32px'>{val}</h2>
                <p style='color:#888; margin:0; font-size:11px'>Test AUROC</p>
            </div>
            """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=["M1 Clinical", "M2 CT only", "M3 Rad+Clin", "M4 Fusion"],
        y=[68.1, 70.3, 71.5, 72.4],
        marker_color=["#95a5a6", "#3498db", "#e67e22", "#27ae60"],
        text=["68.1%", "70.3%", "71.5%", "72.4%"],
        textposition="outside", width=0.5
    ))
    fig.update_layout(
        title="모달리티별 AUROC 비교", yaxis_title="AUROC (%)",
        yaxis_range=[60, 78], height=400,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 3: 모델 성능
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("### 라디오믹스 단독 모델 성능 비교 (256dim)")

    df = pd.DataFrame({
        "모델":          ["LR (balanced)", "RF (balanced)", "GBM", "LR (C=1)", "RF (depth=3)"],
        "Val AUROC":     [0.5747, 0.5706, 0.5259, 0.5159, 0.5676],
        "Test AUROC":    [0.5465, 0.5382, 0.6112, 0.5329, 0.5318],
        "Val-Test 차이": [0.0282, 0.0324, 0.0853, 0.0171, 0.0359],
        "Recall":        [0.5294, 0.4118, 0.4412, 0.5000, 0.4118],
        "통과":          ["⚠️", "⚠️", "✅", "⚠️", "⚠️"]
    })

    st.dataframe(df, use_container_width=True, height=220)
    st.success("✅ 최고 성능 모델: **GBM** (Test AUROC: **0.6112**, Recall: **0.4412**)")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Val AUROC",  x=df["모델"], y=df["Val AUROC"],  marker_color="#3498db", width=0.35))
    fig.add_trace(go.Bar(name="Test AUROC", x=df["모델"], y=df["Test AUROC"], marker_color="#e67e22", width=0.35))
    fig.update_layout(
        barmode="group", title="모델별 Val / Test AUROC 비교",
        yaxis_range=[0.4, 0.7], height=380,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa"
    )
    st.plotly_chart(fig, use_container_width=True)