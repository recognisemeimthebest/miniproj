"""Build single-file static demo (file:// friendly).

Run:
    cd /home/team4/miniproj_ljw/web
    python build_static.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CKPT = ROOT / "checkpoints" / "triple_linear_l2_best.pt"
EMB = ROOT / "embeddings"
GRADCAM_DIR = REPO / "figures" / "gradcam_overlays_iter0"
CLINICAL_CSV = Path("/home/team4/LJW/metadata/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
OUT = ROOT / "static_app.html"

SPLITS = ("train", "val", "test")
MODS = ("clinical", "radiomics", "ct")


# ── 1. M4 weights ─────────────────────────────────────────────────
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
W = ck["state_dict"]["linear.weight"].numpy().reshape(-1).astype(np.float32)
b = float(ck["state_dict"]["linear.bias"].numpy().reshape(-1)[0])
mean = ck["state_dict"]["mean"].numpy().astype(np.float32)
scale = ck["state_dict"]["scale"].numpy().astype(np.float32)
slices = ck["modality_slices"]
test_auroc = ck["metrics"]["test_auroc_torch"]


def load_split(split):
    banks = {m: np.load(EMB / f"{m}_{split}.npz", allow_pickle=True) for m in MODS}
    ref_pids = [str(p) for p in banks["clinical"]["pids"]]
    ref_y = banks["clinical"]["y"].astype(int)
    idx = {m: {str(p): i for i, p in enumerate(banks[m]["pids"])} for m in MODS}
    recs = {}
    for pid_i, pid in enumerate(ref_pids):
        parts = [banks[m]["emb"][idx[m][pid]].astype(np.float32) for m in MODS]
        recs[pid] = {"concat": np.concatenate(parts).tolist(), "y": int(ref_y[pid_i])}
    return recs


test_data = load_split("test")


# ── 2. Baselines ──────────────────────────────────────────────────
def stack(splits, dims):
    X, y = [], []
    for s in splits:
        banks = {m: np.load(EMB / f"{m}_{s}.npz", allow_pickle=True) for m in MODS}
        pids = [str(p) for p in banks["clinical"]["pids"]]
        yy = banks["clinical"]["y"].astype(int)
        idx = {m: {str(p): i for i, p in enumerate(banks[m]["pids"])} for m in MODS}
        for pid_i, pid in enumerate(pids):
            parts = [banks[m]["emb"][idx[m][pid]].astype(np.float32) for m in dims]
            X.append(np.concatenate(parts))
            y.append(int(yy[pid_i]))
    return np.asarray(X, np.float32), np.asarray(y, int)


baselines = {}
for name, dims in [("M1 Clinical", ["clinical"]),
                   ("M3 Clin+Rad", ["clinical", "radiomics"])]:
    Xtv, ytv = stack(["train", "val"], dims)
    sc = StandardScaler().fit(Xtv)
    clf = LogisticRegression(penalty="l2", C=0.01, class_weight="balanced",
                             max_iter=2000, random_state=99).fit(sc.transform(Xtv), ytv)
    baselines[name] = {
        "dims": dims,
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "W": clf.coef_.reshape(-1).tolist(), "b": float(clf.intercept_.reshape(-1)[0]),
    }


# ── 3. Clinical info ──────────────────────────────────────────────
cdf = pd.read_csv(CLINICAL_CSV)
cdf["PatientID"] = cdf["PatientID"].astype(str)
clin_info = {}
for pid in test_data:
    r = cdf[cdf["PatientID"] == pid]
    if len(r):
        d = r.iloc[0].to_dict()
        clin_info[pid] = {
            "age": None if pd.isna(d["age"]) else round(float(d["age"]), 1),
            "gender": str(d["gender"]),
            "stage": str(d["Overall.Stage"]),
            "histology": str(d["Histology"]),
            "T": int(d["clinical.T.Stage"]) if pd.notna(d["clinical.T.Stage"]) else None,
            "N": int(d["Clinical.N.Stage"]), "M": int(d["Clinical.M.Stage"]),
            "surv_days": int(d["Survival.time"]),
            "dead": int(d["deadstatus.event"]),
        }
    else:
        clin_info[pid] = None


# ── 4. Grad-CAM as base64 ────────────────────────────────────────
gradcam_b64 = {}
for pid in test_data:
    p = GRADCAM_DIR / f"{pid}.png"
    gradcam_b64[pid] = ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
                        if p.exists() else None)


# ── 5. Global feature importance across test set ─────────────────
X_test = np.asarray([test_data[p]["concat"] for p in test_data], np.float32)
X_std = (X_test - mean) / scale                      # (n,640)
contribs = W * X_std                                  # (n,640)  per-patient SHAP
global_imp = np.abs(contribs).mean(axis=0)            # (640,)


# ── 5b. Clinical latent dim ↔ 원본 임상 변수 매핑 (trainval Pearson) ──
HIST_KEYS = ["adenocarcinoma", "squamous cell carcinoma", "large cell", "nos"]
STAGE_KEYS = ["I", "II", "IIIa", "IIIb"]
CLIN_KOR = {
    "age": "나이",
    "gender_male": "성별(남)",
    "T_stage": "T병기",
    "N_stage": "N병기",
    "M_stage": "M병기",
    **{f"stage_{s}": f"병기{s}" for s in STAGE_KEYS},
    **{f"histo_{h}": k for h, k in zip(
        HIST_KEYS, ["선암", "편평세포암", "대세포암", "조직형기타"])},
}


def build_clin_X(df):
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"].fillna(df["age"].mean())
    X["gender_male"] = (df["gender"].str.lower() == "male").astype(int)
    X["T_stage"] = df["clinical.T.Stage"].fillna(0)
    X["N_stage"] = df["Clinical.N.Stage"].fillna(0)
    X["M_stage"] = df["Clinical.M.Stage"].fillna(0)
    for s in STAGE_KEYS:
        X[f"stage_{s}"] = (df["Overall.Stage"] == s).astype(int)
    for h in HIST_KEYS:
        X[f"histo_{h}"] = (df["Histology"].str.lower() == h).astype(int)
    return X


# trainval clinical latent 357×128 + 원본 변수 357×13 → Pearson corr
tv_pids = []
tv_clin_emb = []
for s in ("train", "val"):
    z = np.load(EMB / f"clinical_{s}.npz", allow_pickle=True)
    for pid, e in zip(z["pids"], z["emb"]):
        tv_pids.append(str(pid))
        tv_clin_emb.append(e.astype(np.float32))
tv_clin_emb = np.asarray(tv_clin_emb, np.float32)      # (357, 128)

cdf_indexed = cdf.set_index("PatientID")
X_raw_tv = build_clin_X(cdf_indexed.loc[tv_pids].reset_index()).values.astype(np.float32)
feat_names = list(build_clin_X(cdf_indexed.iloc[:1].reset_index()).columns)

# Pearson correlation (128 x 13)
def pearson(a, b):
    a = (a - a.mean(0)) / (a.std(0) + 1e-9)
    b = (b - b.mean(0)) / (b.std(0) + 1e-9)
    return (a.T @ b) / a.shape[0]


corr = pearson(tv_clin_emb, X_raw_tv)                  # (128, 13)

THRESH = 0.40
clin_dim_labels = []
for i in range(128):
    j = int(np.argmax(np.abs(corr[i])))
    r = float(corr[i, j])
    if abs(r) >= THRESH:
        clin_dim_labels.append({
            "kor": CLIN_KOR[feat_names[j]],
            "r": round(r, 2),
        })
    else:
        clin_dim_labels.append(None)
n_mapped = sum(v is not None for v in clin_dim_labels)
print(f"[clin_map] {n_mapped}/128 clinical latent dims mapped to raw vars (|r|>={THRESH})")




# ── 6. HTML ───────────────────────────────────────────────────────
payload = {
    "model": {
        "modality_slices": slices,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "W": W.tolist(), "b": b,
        "test_auroc": test_auroc, "chosen_C": ck["chosen_C"],
    },
    "baselines": baselines,
    "test": test_data,
    "clinical": clin_info,
    "gradcam": gradcam_b64,
    "global_importance": global_imp.tolist(),
    "clin_dim_labels": clin_dim_labels,   # length 128; entry or null
}

print(f"[stats] test={len(test_data)}  gradcam={sum(v is not None for v in gradcam_b64.values())}")

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>LUNA-XAI · NSCLC 2yr Survival</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
       background:#0d1117; color:#e6edf3; min-height:100vh; }

header { background:linear-gradient(135deg,#1f6feb,#388bfd); padding:12px 28px;
         display:flex; justify-content:space-between; align-items:center;
         box-shadow:0 2px 12px rgba(0,0,0,.3); }
header h1 { font-size:20px; display:flex; align-items:center; gap:10px; }
header .logo { font-size:26px; }
header .sub { font-size:12px; opacity:.85; font-weight:400; margin-left:6px; }
header .stat { font-size:12px; font-family:monospace; opacity:.85; }

nav.tabs { background:#161b22; border-bottom:1px solid #30363d; display:flex;
           padding:0 28px; gap:4px; }
nav.tabs button { background:transparent; border:none; color:#8b949e;
                  font-size:14px; padding:12px 22px; cursor:pointer;
                  border-bottom:2px solid transparent; font-weight:500; }
nav.tabs button:hover { color:#c9d1d9; }
nav.tabs button.active { color:#58a6ff; border-bottom-color:#58a6ff; }

.tab-body { padding:16px; }
.tab-body.hidden { display:none; }

.panel { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px; }
.panel h2 { font-size:14px; text-transform:uppercase; letter-spacing:.8px;
            color:#7d8590; margin-bottom:14px; padding-bottom:10px;
            border-bottom:1px solid #30363d; }
.panel h3 { font-size:12px; color:#8b949e; margin:14px 0 8px;
            text-transform:uppercase; letter-spacing:.5px; }

.grid-pred { display:grid; grid-template-columns:300px 1fr 360px; gap:14px;
             height:calc(100vh - 120px); align-items:stretch; }
.grid-pred > aside, .grid-pred > main {
  overflow-y:auto; max-height:100%;
  scrollbar-width:thin; scrollbar-color:#30363d #0d1117;
}
.grid-pred > *::-webkit-scrollbar { width:8px; }
.grid-pred > *::-webkit-scrollbar-track { background:#0d1117; }
.grid-pred > *::-webkit-scrollbar-thumb { background:#30363d; border-radius:4px; }
.grid-pred > *::-webkit-scrollbar-thumb:hover { background:#484f58; }

select { width:100%; background:#0d1117; border:1px solid #30363d;
         color:#e6edf3; padding:8px 10px; border-radius:6px; font-size:13px; }
label { display:block; font-size:12px; color:#8b949e; margin-bottom:6px; }

.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.info-cell { background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:8px 10px; }
.info-cell .k { color:#8b949e; font-size:10px; text-transform:uppercase; letter-spacing:.5px; }
.info-cell .v { color:#e6edf3; font-family:monospace; font-size:14px; margin-top:2px; }

.ct-stage { display:flex; flex-direction:column; gap:10px; }
.ct-pane { background:#000; border:1px solid #30363d; border-radius:10px;
           padding:0; overflow:hidden; position:relative; }
.ct-pane iframe { width:100%; border:0; display:block; background:#111; }
.ct-pane.mpr iframe { height:560px; }
.ct-pane.vol iframe { height:460px; }
.ct-pane .cap { position:absolute; top:10px; left:14px; color:#7fdbca;
                font-size:11px; font-family:monospace; padding:3px 10px;
                background:rgba(0,0,0,.6); border-radius:4px; z-index:2; pointer-events:none; }

.gradcam-mini { background:#000; border:1px solid #30363d; border-radius:8px;
                padding:10px; margin-bottom:14px; text-align:center; }
.gradcam-mini img { max-width:100%; border-radius:4px; }
.gradcam-mini .cap { color:#8b949e; font-size:11px; margin-top:6px; font-family:monospace; }

.lung-thumb { background:#0d1117; border:1px solid #30363d; border-radius:8px;
              margin-top:14px; overflow:hidden; position:relative; }
.lung-thumb iframe { width:100%; height:260px; border:0; display:block; background:#0d1117; }
.lung-thumb .cap { position:absolute; top:6px; left:8px; color:#7fdbca;
                   font-size:10px; font-family:monospace; padding:2px 6px;
                   background:rgba(0,0,0,.6); border-radius:3px; pointer-events:none;
                   z-index:2; }
.lung-thumb .legend { position:absolute; bottom:6px; right:8px;
                      display:flex; gap:8px; font-size:10px; z-index:2; pointer-events:none; }
.lung-thumb .legend span { background:rgba(0,0,0,.6); padding:2px 6px; border-radius:3px; }

.gc-grid { display:grid; grid-template-columns:1.6fr 1fr; gap:14px; }
.gc-image { display:flex; flex-direction:column; }
.gradcam-big { background:#000; border-radius:8px; padding:16px; text-align:center;
               flex:1; display:flex; align-items:center; justify-content:center; min-height:500px; }
.gradcam-big img { max-width:100%; max-height:700px; border-radius:6px;
                   box-shadow:0 4px 24px rgba(0,0,0,.5); }
.gc-image .cap { color:#7fdbca; font-size:12px; margin-top:10px;
                 font-family:monospace; text-align:center; }

.prob-card { border-radius:10px; padding:14px 16px; text-align:center; color:#fff;
             margin-bottom:12px; }
.prob-card .label { font-size:10px; opacity:.9; letter-spacing:2px; }
.prob-card .prob { font-size:42px; font-weight:700; line-height:1.1; margin:4px 0 2px; }
.prob-card .ci { font-size:11px; opacity:.85; font-family:monospace; }
.risk-chip { display:inline-block; margin-top:6px; background:rgba(0,0,0,.25);
             padding:3px 10px; border-radius:12px; font-size:11px; }

.bar-row { display:grid; grid-template-columns:1fr 55px; gap:8px;
           align-items:center; font-size:12px; margin-bottom:6px; }
.bar-row .name { color:#c9d1d9; font-size:11px; margin-bottom:3px; }
.bar-track { height:10px; background:#30363d; border-radius:5px; overflow:hidden; }
.bar-fill { height:100%; border-radius:5px; }
.bar-row .val { text-align:right; font-family:monospace; color:#c9d1d9; font-size:12px; }
.bar-wrap { margin-bottom:10px; }

.summary-box { background:#0d1117; border:1px solid #30363d; border-radius:8px;
               padding:12px 14px; font-size:13px; line-height:1.65; color:#c9d1d9; }
.summary-box .tag { display:inline-block; background:#1f6feb33; color:#58a6ff;
                    padding:1px 7px; border-radius:10px; font-size:11px;
                    font-family:monospace; margin:0 2px; }
.summary-box .pos { color:#3fb950; font-weight:600; }
.summary-box .neg { color:#f85149; font-weight:600; }
.summary-box h4 { color:#e6edf3; font-size:12px; margin-bottom:8px;
                  text-transform:uppercase; letter-spacing:1px; }

/* SHAP tab */
.grid-shap { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.shap-panel { background:#161b22; border:1px solid #30363d; border-radius:10px;
              padding:18px; }

.shaprow { display:grid; grid-template-columns:90px 1fr 60px; gap:8px;
           align-items:center; font-size:12px; margin-bottom:4px; }
.shaptrack { position:relative; height:18px; background:#0d1117;
             border:1px solid #21262d; border-radius:3px; }
.shapmid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:#484f58; }
.shapfill { position:absolute; height:100%; top:0; border-radius:2px; }
.shapfill.pos { background:#3fb950; left:50%; }
.shapfill.neg { background:#f85149; right:50%; }
.shaprow .val { text-align:right; font-family:monospace; font-size:11px; }
.shaprow .lbl { font-family:monospace; font-size:11px; color:#8b949e; }

.tag-mod-clinical  { background:#1f6feb33; color:#58a6ff; }
.tag-mod-radiomics { background:#8957e533; color:#bc8cff; }
.tag-mod-ct        { background:#da363333; color:#ff7b72; }
.mod-tag { display:inline-block; padding:1px 6px; border-radius:8px;
           font-size:10px; font-family:monospace; }

.warn { background:#f8514918; border-left:3px solid #f85149; padding:8px 12px;
        border-radius:4px; font-size:11px; color:#ffa198; line-height:1.5; margin-top:12px; }
.footer { padding:10px 20px; font-size:10.5px; color:#6e7681; line-height:1.5;
          border-top:1px solid #30363d; text-align:center; }
.footer b { color:#8b949e; font-weight:600; }
</style>
</head>
<body>

<header>
  <h1><span class="logo">🫁</span> LUNA-XAI <span class="sub">| NSCLC 2-Year Survival · static demo</span></h1>
  <span class="stat" id="modelstat"></span>
</header>

<nav class="tabs">
  <button class="active" data-tab="pred">🔍 예측</button>
  <button data-tab="gradcam">🖼️ Grad-CAM</button>
  <button data-tab="shap">📊 SHAP 분석</button>
</nav>

<!-- ====== TAB 1: PRED ====== -->
<div class="tab-body" id="tab-pred">
  <div class="grid-pred">

    <!-- LEFT: selection + clinical -->
    <aside class="panel">
      <h2>📋 환자 선택 & 임상정보</h2>
      <label>Test split (n=<span id="npts"></span>)</label>
      <select id="pid"></select>
      <h3>🧑‍⚕️ Clinical</h3>
      <div id="clin"></div>

      <h3>🫁 3D 폐 + 종양 (입체)</h3>
      <div class="lung-thumb">
        <span class="cap" id="lung_cap">폐(파랑) · 종양(빨강)</span>
        <span class="legend">
          <span style="color:#6ab7ff">■ 폐</span>
          <span style="color:#ff3b3b">■ 종양</span>
        </span>
        <iframe id="lungthumb" src="about:blank" title="Lung + GTV 3D mini"></iframe>
      </div>

      <div class="warn">
        사전 추출된 <b>test 63명 임베딩</b>을 사용하는 데모.<br>
        M4 = Linear L2 on 640-dim concat (test AUROC 0.6589).
      </div>
    </aside>

    <!-- CENTER: 4 views at once — 상단 3-plane MPR + 하단 3D volume -->
    <main class="ct-stage">
      <div class="ct-pane mpr">
        <div class="cap" id="ct_cap">Axial / Sagittal / Coronal — MPR</div>
        <iframe id="ctviewer" src="about:blank" title="CT MPR viewer"></iframe>
      </div>
      <div class="ct-pane vol">
        <div class="cap" id="ct3d_cap">3D Volume · drag=rotate · scroll=zoom · shift-drag=pan</div>
        <iframe id="ctviewer3d" src="about:blank" title="CT 3D viewer"></iframe>
      </div>
    </main>

    <!-- RIGHT: prob + compare + summary -->
    <aside class="panel">
      <h2>📈 AI 판단 결과</h2>
      <div id="result"></div>

      <h3>📊 모델별 비교</h3>
      <div id="compare"></div>

      <h3>📝 판단 근거 요약</h3>
      <div class="summary-box" id="summary"></div>
    </aside>
  </div>
</div>

<!-- ====== TAB 2: Grad-CAM ====== -->
<div class="tab-body hidden" id="tab-gradcam">
  <div class="gc-grid">
    <section class="shap-panel gc-image">
      <h2>🖼️ Grad-CAM — CT overlay (iter0)</h2>
      <div class="gradcam-big">
        <img id="gradcam" alt="Grad-CAM overlay">
      </div>
      <div class="cap" id="gradcam_cap"></div>
    </section>

    <section class="shap-panel">
      <h2>📐 설명 & 정량 지표</h2>
      <p style="font-size:13px;color:#c9d1d9;line-height:1.7">
        Grad-CAM은 <b>M2 CT encoder</b>(Hosny 3D CNN) 가 "2년 생존" 예측 logit을 만드는 데
        공간적으로 <b>어느 영역을 활성화</b>했는지를 보여줍니다.
        빨갛게 타는 영역일수록 모델이 생존/사망 판단에 강하게 사용한 부위입니다.
      </p>
      <h3>⚙️ iter0 설정 (frozen)</h3>
      <div class="info-grid">
        <div class="info-cell"><div class="k">CAM Method</div><div class="v">GradCAM</div></div>
        <div class="info-cell"><div class="k">Target Layer</div><div class="v" style="font-size:11px">features[3].act</div></div>
        <div class="info-cell"><div class="k">Post-proc</div><div class="v" style="font-size:12px">ReLU + min-max</div></div>
        <div class="info-cell"><div class="k">Target Class</div><div class="v">1 (survivor)</div></div>
      </div>
      <h3>📊 정량 지표 (test n=63 평균)</h3>
      <div class="info-grid">
        <div class="info-cell"><div class="k">IoU @ 25%</div><div class="v">0.149</div></div>
        <div class="info-cell"><div class="k">IoU @ 50%</div><div class="v">0.282</div></div>
        <div class="info-cell"><div class="k">Pointing Game</div><div class="v">0.127 (8/63)</div></div>
        <div class="info-cell"><div class="k">vs Random</div><div class="v" style="color:#f85149;font-size:12px">PG below 0.459</div></div>
      </div>
      <p style="margin-top:14px;font-size:11px;color:#8b949e;line-height:1.6">
        ℹ️ iter0는 <b>가장 깊은 층</b>(features[3])을 target으로 하여
        정량 IoU/PG는 낮지만 시각적으로는 <b>GTV에 가장 집중된 단일 blob</b>을 생성합니다
        (iter5 대비 공간 노이즈 최소).
      </p>
    </section>
  </div>
</div>

<!-- ====== TAB 3: SHAP ====== -->
<div class="tab-body hidden" id="tab-shap">
  <div class="grid-shap">

    <section class="shap-panel">
      <h2>🔬 현재 환자 — 차원별 SHAP 기여도 (Top 20)</h2>
      <div id="shap_top"></div>
      <p style="margin-top:10px;font-size:11px;color:#8b949e;line-height:1.55">
        <b>SHAP</b> (선형 모델에서는 정확히 <code>W<sub>i</sub> · (x<sub>i</sub>−μ<sub>i</sub>)/σ<sub>i</sub></code>).
        양수(초록) = 2년 생존 확률을 높이는 방향, 음수(빨강) = 사망 방향.
        임상 latent dim 중 원본 변수(나이·T/N/M·병기·조직형)와 trainval Pearson |r|≥0.40인
        차원은 해당 변수명(예: <span style="color:#7fdbca">대세포암</span>, <span style="color:#7fdbca">나이</span>)
        으로만 표시합니다. Radiomics·CT는 해석 불가능한 latent라 영문 dim 번호 유지.
      </p>
    </section>

    <section class="shap-panel">
      <h2>📐 Modality 요약 + 재구성</h2>
      <div id="shap_mod"></div>
      <div id="shap_sum" style="margin-top:10px;font-size:12px;color:#8b949e;font-family:monospace"></div>
      <h3>🌍 Global importance (test n=63)</h3>
      <div id="shap_global"></div>
      <p style="font-size:11px;color:#8b949e;margin-top:6px;line-height:1.55">
        test 환자 전체에서 <code>|SHAP<sub>i</sub>|</code>를 평균한 모달리티별 합계.
        어느 modality가 평균적으로 "더 많이 말하는지" 보여줌.
      </p>
    </section>

  </div>
</div>

<div class="footer">
  <b>⚠️ 연구용 프로토타입.</b> 본 결과는 실제 임상 의사결정에 사용할 수 없습니다.
  M4 = TripleLinearL2 (feature/LJW), seed=99, trainval refit. test AUROC = __AUROC__.
</div>

<script id="DATA" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("DATA").textContent);
const pids = Object.keys(D.test).sort();
const sel = document.getElementById("pid");
pids.forEach(p => { const o=document.createElement("option"); o.value=p; o.textContent=p; sel.appendChild(o); });
document.getElementById("npts").textContent = pids.length;
document.getElementById("modelstat").textContent =
  `Test AUROC ${D.model.test_auroc.toFixed(4)} · n=${pids.length} · C=${D.model.chosen_C}`;

// Tabs
document.querySelectorAll("nav.tabs button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav.tabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-body").forEach(t => t.classList.add("hidden"));
    document.getElementById("tab-"+btn.dataset.tab).classList.remove("hidden");
  });
});

const sigm = z => 1/(1+Math.exp(-z));

function standardised(x){
  const {mean, scale} = D.model;
  const out = new Array(x.length);
  for (let i=0;i<x.length;i++) out[i] = (x[i]-mean[i])/scale[i];
  return out;
}
function contribFull(x){
  const xs = standardised(x);
  const W = D.model.W;
  const c = new Array(W.length);
  for (let i=0;i<W.length;i++) c[i] = W[i]*xs[i];
  return c;
}
function modalityOf(i){
  for (const [m,[a,b]] of Object.entries(D.model.modality_slices)) if (i>=a && i<b) return m;
  return "?";
}
function sliceOf(x, dims){
  const sl = D.model.modality_slices; const out=[];
  dims.forEach(m => { const [a,b]=sl[m]; for (let i=a;i<b;i++) out.push(x[i]); });
  return out;
}
function predictBaseline(x, base){
  let z = base.b;
  for (let i=0;i<x.length;i++) z += base.W[i]*(x[i]-base.mean[i])/base.scale[i];
  return sigm(z);
}
function sum(a){ return a.reduce((s,v)=>s+v,0); }

function writeSummary(pid, rec, prob, logit, perMod){
  const ci = D.clinical[pid];
  const pct = (prob*100).toFixed(1);
  const verdict = prob>=0.6 ? ["낮음","pos","생존 가능성이 높다"]
               : prob>=0.4 ? ["중등도","",  "판정이 경계 구간이다"]
               :             ["높음","neg","사망 위험이 높다"];
  const modSorted = Object.entries(perMod).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const [topMod, topVal] = modSorted[0];
  const dir = topVal>=0 ? "긍정" : "부정";
  const dirCls = topVal>=0 ? "pos" : "neg";
  const gtTxt = rec.y===1 ? `<span class="pos">2년 생존 (y=1)</span>` : `<span class="neg">2년 내 사망 (y=0)</span>`;

  let ageStagePart = "";
  if (ci){
    ageStagePart = `이 환자는 <b>${ci.age ?? "?"}세 ${ci.gender==='male'?'남성':'여성'}</b>, `
      + `<span class="tag">${ci.stage}</span> 병기의 <span class="tag">${ci.histology}</span> 입니다. `;
  }
  const allAbs = Object.entries(perMod).map(([m,v])=>Math.abs(v));
  const totalAbs = allAbs.reduce((s,v)=>s+v,0) + 1e-9;
  const share = Object.fromEntries(Object.entries(perMod).map(([m,v])=>[m, 100*Math.abs(v)/totalAbs]));

  document.getElementById("summary").innerHTML = `
    <h4>AI 판단 내용</h4>
    <p>${ageStagePart}<b>M4 Fusion 모델</b>은 2년 생존 확률을 <b style="font-size:16px">${pct}%</b>로 예측했습니다
       (logit ${logit.toFixed(3)}) — <span class="${dirCls}">${verdict[2]}</span>는 판단 (<span class="tag">위험 ${verdict[0]}</span>).</p>
    <p style="margin-top:10px">
       판단에 가장 크게 기여한 입력은 <span class="mod-tag tag-mod-${topMod}">${({clinical:"임상",radiomics:"Radiomics",ct:"CT"})[topMod]||topMod}</span>
       모달리티로, <span class="${dirCls}">${dir} 방향(${topVal>=0?'+':''}${topVal.toFixed(3)})</span>으로 작용했습니다.
       세 모달리티의 logit 기여 비중은
       임상 ${share.clinical.toFixed(0)}% /
       Radiomics ${share.radiomics.toFixed(0)}% /
       CT ${share.ct.toFixed(0)}% 입니다.</p>
    <p style="margin-top:10px">
       정답 라벨: ${gtTxt} · 실제 생존일 ${ci ? ci.surv_days+"d" : "?"}
       ${ci ? (ci.dead?" (사망 이벤트)":" (censored)") : ""}.
       ${prob>=0.5 === (rec.y===1) ? `<span class="pos">모델 판정이 정답과 일치합니다.</span>`
                                   : `<span class="neg">모델 판정이 정답과 어긋납니다.</span>`}</p>
  `;
}

function renderPred(pid){
  const rec = D.test[pid];
  const x = rec.concat;
  const xs = standardised(x);
  const c = contribFull(x);
  const W = D.model.W, b = D.model.b;
  const logit = b + sum(c);
  const prob = sigm(logit);
  const pct = (prob*100).toFixed(1);

  const bg = prob>=0.6 ? "#238636" : (prob>=0.4 ? "#d29922" : "#da3633");
  const risk = prob>=0.6 ? "🟢 저위험" : (prob>=0.4 ? "🟡 중등도" : "🔴 고위험");
  document.getElementById("result").innerHTML = `
    <div class="prob-card" style="background:linear-gradient(135deg,${bg}cc,${bg})">
      <div class="label">2-YEAR SURVIVAL PROB</div>
      <div class="prob">${pct}<span style="font-size:22px">%</span></div>
      <div class="ci">logit ${logit.toFixed(3)}  ·  GT ${rec.y===1?"생존":"사망"}</div>
      <div class="risk-chip">${risk}</div>
    </div>`;

  // Model comparison
  const comp = {"M4 Fusion": prob};
  for (const [n, base] of Object.entries(D.baselines)){
    comp[n] = predictBaseline(sliceOf(x, base.dims), base);
  }
  const colors = {"M4 Fusion":"#3fb950","M1 Clinical":"#58a6ff","M3 Clin+Rad":"#8957e5"};
  document.getElementById("compare").innerHTML = Object.entries(comp).map(([n,v])=>`
    <div class="bar-wrap">
      <div class="name">${n}</div>
      <div class="bar-row">
        <div class="bar-track"><div class="bar-fill" style="width:${(v*100).toFixed(1)}%;background:${colors[n]||'#58a6ff'}"></div></div>
        <span class="val">${(v*100).toFixed(1)}%</span>
      </div>
    </div>`).join("");

  // Modality contribution
  const perMod = {};
  for (const [m,[a,bv]] of Object.entries(D.model.modality_slices)){
    let s=0; for (let i=a;i<bv;i++) s+=c[i]; perMod[m]=s;
  }

  // CT viewers — 3-plane MPR + 3D volume iframes
  const iframe = document.getElementById("ctviewer");
  if (iframe.dataset.pid !== pid){
    iframe.src = `patients/${pid}.html`;
    iframe.dataset.pid = pid;
  }
  const iframe3d = document.getElementById("ctviewer3d");
  if (iframe3d.dataset.pid !== pid){
    iframe3d.src = `patients3d/${pid}.html`;
    iframe3d.dataset.pid = pid;
  }
  const lungThumb = document.getElementById("lungthumb");
  if (lungThumb && lungThumb.dataset.pid !== pid){
    lungThumb.src = `patients3d_lung/${pid}.html`;
    lungThumb.dataset.pid = pid;
  }
  document.getElementById("ct_cap").textContent = `${pid} · Axial / Sagittal / Coronal MPR`;
  document.getElementById("ct3d_cap").textContent = `${pid} · 3D Volume · drag=rotate · scroll=zoom · shift-drag=pan`;
  const lc = document.getElementById("lung_cap");
  if (lc) lc.textContent = `${pid} · 폐(파랑) + 종양(빨강)`;

  // Grad-CAM (SHAP tab)
  const img = D.gradcam[pid];
  document.getElementById("gradcam").src = img || "";
  document.getElementById("gradcam_cap").textContent =
    img ? `${pid} · CT axial + Grad-CAM (iter0 · GradCAM+relu_minmax)`
        : `${pid} · Grad-CAM 없음`;

  const ci = D.clinical[pid];
  document.getElementById("clin").innerHTML = ci ? `
    <div class="info-grid">
      <div class="info-cell"><div class="k">Age</div><div class="v">${ci.age ?? '—'}</div></div>
      <div class="info-cell"><div class="k">Gender</div><div class="v">${ci.gender}</div></div>
      <div class="info-cell"><div class="k">Stage</div><div class="v">${ci.stage}</div></div>
      <div class="info-cell"><div class="k">TNM</div><div class="v">T${ci.T??0}N${ci.N}M${ci.M}</div></div>
      <div class="info-cell" style="grid-column:span 2"><div class="k">Histology</div><div class="v" style="font-size:12px">${ci.histology}</div></div>
      <div class="info-cell"><div class="k">Survival (d)</div><div class="v">${ci.surv_days}</div></div>
      <div class="info-cell"><div class="k">Dead event</div><div class="v">${ci.dead?'yes':'no'}</div></div>
    </div>` : `<div style="color:#8b949e;font-size:12px">임상 CSV에 없음</div>`;

  writeSummary(pid, rec, prob, logit, perMod);
  renderShap(pid, c, perMod, logit);
}

const MOD_LABEL = {clinical:"임상", radiomics:"Radiomics", ct:"CT"};

function shapBar(name, val, maxAbs, modTag){
  const w = Math.min(50, 50*Math.abs(val)/(maxAbs+1e-9));
  const cls = val>=0 ? "pos" : "neg";
  const display = MOD_LABEL[modTag] || modTag;
  const mod = modTag ? `<span class="mod-tag tag-mod-${modTag}">${display}</span>` : "";
  return `<div class="shaprow">
    <span class="lbl">${name} ${mod}</span>
    <div class="shaptrack"><div class="shapmid"></div>
      <div class="shapfill ${cls}" style="width:${w}%"></div>
    </div>
    <span class="val" style="color:${val>=0?'#3fb950':'#f85149'}">${val>=0?'+':''}${val.toFixed(3)}</span>
  </div>`;
}

function dimLabel(i, mod){
  // Clinical latent dim(0~127) 중 원본 변수와 강한 상관이 있는 경우 한글 이름만 표시
  if (mod === "clinical"){
    const [a] = D.model.modality_slices.clinical;
    const lab = (D.clin_dim_labels || [])[i - a];
    if (lab){
      return `<span style="color:#c9d1d9">${lab.kor}</span>`;
    }
  }
  return `dim ${i}`;
}

function renderShap(pid, c, perMod, logit){
  // Top-20 dims
  const idxs = [...c.keys()].sort((a,b)=>Math.abs(c[b])-Math.abs(c[a])).slice(0,20);
  const maxAbs = Math.max(...idxs.map(i=>Math.abs(c[i])), 1e-6);
  document.getElementById("shap_top").innerHTML = idxs.map(i => {
    const mod = modalityOf(i);
    return shapBar(dimLabel(i, mod), c[i], maxAbs, mod);
  }).join("");

  // Modality-level
  const maxAbsM = Math.max(...Object.values(perMod).map(Math.abs), 1e-6);
  document.getElementById("shap_mod").innerHTML = Object.entries(perMod).map(([m,v]) =>
    shapBar(m, v, maxAbsM, m)
  ).join("");
  const total = Object.values(perMod).reduce((s,v)=>s+v,0);
  document.getElementById("shap_sum").textContent =
    `∑ modality (${total.toFixed(3)}) + bias (${D.model.b.toFixed(3)}) = logit (${logit.toFixed(3)})`;

  // Global importance (per modality sum of mean|contrib|)
  const g = D.global_importance;
  const gMod = {clinical:0, radiomics:0, ct:0};
  for (const [m,[a,b]] of Object.entries(D.model.modality_slices)){
    for (let i=a;i<b;i++) gMod[m]+=g[i];
  }
  const gMax = Math.max(...Object.values(gMod), 1e-6);
  document.getElementById("shap_global").innerHTML = Object.entries(gMod).map(([m,v])=>{
    const w = Math.min(100, 100*v/gMax);
    const disp = MOD_LABEL[m] || m;
    return `<div class="shaprow">
      <span class="lbl">${disp} <span class="mod-tag tag-mod-${m}">${disp}</span></span>
      <div class="shaptrack">
        <div class="shapfill pos" style="left:0;width:${w}%"></div>
      </div>
      <span class="val" style="color:#c9d1d9">${v.toFixed(3)}</span>
    </div>`;
  }).join("");
}

sel.addEventListener("change", e => renderPred(e.target.value));
renderPred(pids[0]);
</script>
</body>
</html>
"""

html_final = (
    HTML
    .replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
    .replace("__AUROC__", f"{test_auroc:.4f}")
)
OUT.write_text(html_final, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"[ok] wrote {OUT}  ({size_mb:.2f} MB)")
print(f"     file://{OUT}")
