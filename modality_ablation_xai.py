"""Modality Ablation XAI - triple_linear_l2_best.pt"""
import os, zipfile, pickle, io
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODEL_PATH = "triple_linear_l2_best.pt"
EMB_DIR    = "triple_model/embeddings"
OUTPUT_DIR = "output_xai"
SPLIT      = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── AUROC / ROC (numpy only) ─────────────────────────────────
def auroc(y, p):
    y, p = np.asarray(y, np.float32), np.asarray(p, np.float32)
    pos, neg = p[y==1], p[y==0]
    if len(pos)==0 or len(neg)==0: return 0.5
    order = np.argsort(p)[::-1]
    ys = y[order]
    cum_p = np.cumsum(ys)
    auc = sum(cum_p[i] for i,v in enumerate(ys) if v==0)
    return auc / (len(pos)*len(neg))

def roc(y, p):
    y, p = np.asarray(y, np.float32), np.asarray(p, np.float32)
    thrs = np.sort(np.unique(p))[::-1]
    np_, nn = (y==1).sum(), (y==0).sum()
    fprs, tprs = [0.], [0.]
    for thr in thrs:
        pp = p >= thr
        tprs.append(((pp)&(y==1)).sum()/np_ if np_ else 0.)
        fprs.append(((pp)&(y==0)).sum()/nn  if nn  else 0.)
    fprs.append(1.); tprs.append(1.)
    return np.array(fprs), np.array(tprs)

# ── 모델 로드 ─────────────────────────────────────────────────
class _OD(dict): pass

class _UP(pickle.Unpickler):
    def __init__(self, f, sm):
        super().__init__(f); self._sm = sm
    def find_class(self, m, n):
        if m=="torch._utils" and n=="_rebuild_tensor_v2": return self._rb
        if "torch" in m or (m=="collections" and n=="OrderedDict"): return _OD
        try: return super().find_class(m, n)
        except: return _OD
    def persistent_load(self, pid):
        if isinstance(pid,(list,tuple)) and len(pid)>=5: return (pid[2],pid[4])
        if isinstance(pid,(list,tuple)) and len(pid)==4: return (pid[1],pid[3])
        return pid
    def _rb(self, spid, off, shape, stride, rg, bw):
        key,_ = spid
        raw = self._sm.get(key)
        if raw is None: return np.zeros(shape, np.float32)
        flat = np.frombuffer(raw, np.float32)
        n = int(np.prod(shape)) if shape else 1
        return flat[off:off+n].reshape(shape)

def load_model(path):
    with zipfile.ZipFile(path,"r") as zf:
        names  = zf.namelist()
        prefix = names[0].split("/")[0]+"/"
        sm = {n.split("/data/")[1]: zf.read(n) for n in names if "/data/" in n}
        pkl = zf.read(prefix+"data.pkl")
    obj = _UP(io.BytesIO(pkl), sm).load()
    sd  = obj.get("state_dict", obj)
    mean  = np.array(sd["mean"],          np.float32)
    scale = np.array(sd["scale"],         np.float32)
    W     = np.array(sd["linear.weight"], np.float32)
    b     = np.array(sd["linear.bias"],   np.float32)
    meta  = {k:v for k,v in obj.items() if k!="state_dict"}
    return mean, scale, W, b, meta

def predict(mean, scale, W, b, X):
    Xn = (X - mean) / (scale + 1e-8)
    return 1./(1.+np.exp(-((Xn @ W.T).reshape(-1) + b[0])))

# ── 데이터 ───────────────────────────────────────────────────
MODALS  = ["clinical","radiomics","ct"]
SLICES  = {"clinical":(0,128),"radiomics":(128,384),"ct":(384,640)}
MCOLORS = {"clinical":"#e66101","radiomics":"#5e3c99","ct":"#008837"}

def load_emb(emb_dir, split):
    parts, y = [], None
    for m in MODALS:
        fp = os.path.join(emb_dir, f"{m}_{split}.npz")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"임베딩 없음: {fp}\n"
                "triple_model/embeddings/ 에 npz 파일을 두거나\n"
                "single_modal_baseline 인코더를 먼저 실행하세요.")
        d = np.load(fp, allow_pickle=True)
        parts.append(d["emb"].astype(np.float32))
        if y is None: y = d["y"].astype(np.float32)
    return np.concatenate(parts, 1), y

def ablate(X, zeros):
    Xc = X.copy()
    for m in zeros:
        s,e = SLICES[m]; Xc[:,s:e] = 0.
    return Xc

# ── 시나리오 ─────────────────────────────────────────────────
SCEN = [
    ("Full",           [],                    ["clinical","radiomics","ct"]),
    ("-Clinical",      ["clinical"],           ["radiomics","ct"]),
    ("-Radiomics",     ["radiomics"],          ["clinical","ct"]),
    ("-CT",            ["ct"],                ["clinical","radiomics"]),
    ("Clinical only",  ["radiomics","ct"],     ["clinical"]),
    ("Radiomics only", ["clinical","ct"],      ["radiomics"]),
    ("CT only",        ["clinical","radiomics"],["ct"]),
]
CMAP = {"Full":"#2c7bb6","-Clinical":"#d7191c","-Radiomics":"#fdae61",
        "-CT":"#abd9e9","Clinical only":"#1a9641",
        "Radiomics only":"#a6d96a","CT only":"#b8e186"}

# ── 실행 ─────────────────────────────────────────────────────
print("="*60)
print("  Modality Ablation XAI")
print("="*60)

print(f"\n[1] 모델 로드: {MODEL_PATH}")
mean, scale, W, b, meta = load_model(MODEL_PATH)
saved = meta.get("metrics",{}).get("test_auroc_sklearn","?")
print(f"    arch={meta.get('arch')}  C={meta.get('chosen_C')}  saved_AUC={saved}")

print(f"\n[2] 임베딩 로드: {EMB_DIR}/{SPLIT}")
X, y = load_emb(EMB_DIR, SPLIT)
print(f"    X={X.shape}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")

print(f"\n[3] Ablation")
results = []
for name, zeros, keeps in SCEN:
    prob = predict(mean, scale, W, b, ablate(X, zeros))
    auc  = auroc(y, prob)
    results.append({"name":name,"zeros":zeros,"auc":auc,"prob":prob})
    print(f"    {name:<22} AUROC={auc:.4f}")

full_auc = results[0]["auc"]
contrib  = {r["zeros"][0]: full_auc - r["auc"] for r in results[1:4]}
max_c    = max(abs(v) for v in contrib.values()) if any(abs(v)>1e-9 for v in contrib.values()) else 1e-9

print(f"\n[4] 기여도 DELTA-AUROC (Full={full_auc:.4f})")
if full_auc < 0.51:
    print("    WARNING: Full AUROC~0.5 - 임베딩/모델 확인 필요")
for m,d in sorted(contrib.items(),key=lambda x:-x[1]):
    bar = "X"*max(1,int(abs(d)/max_c*20)) if max_c>1e-6 else ""
    print(f"    {m:<12} {d:+.4f}  {bar}")

pd.DataFrame([{"scenario":r["name"],"zeroed":",".join(r["zeros"]) or "none",
               "auroc":r["auc"],"delta":r["auc"]-full_auc}
              for r in results]).to_csv(
    f"{OUTPUT_DIR}/modality_ablation_results.csv",index=False,float_format="%.4f")

# ── 시각화 1: Bar + 기여도 ───────────────────────────────────
fig,(ax0,ax1) = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Modality Ablation XAI — TripleLinearL2",fontsize=11,fontweight="bold")

names=[r["name"] for r in results]; aucs=[r["auc"] for r in results]
bc=[CMAP.get(n,"#888") for n in names]
bars=ax0.barh(names,aucs,color=bc,edgecolor="white",height=0.6,alpha=0.88)
ax0.axvline(full_auc,color="black",linestyle="--",lw=1.2,label=f"Full={full_auc:.4f}")
for bar,auc in zip(bars,aucs):
    d=auc-full_auc
    lbl=f"{auc:.4f}" if abs(d)<1e-4 else f"{auc:.4f} ({d:+.4f})"
    ax0.text(auc+.003,bar.get_y()+bar.get_height()/2,lbl,va="center",fontsize=8)
ax0.set_xlabel("AUROC"); ax0.set_title("Ablation Scenario AUROC")
ax0.set_xlim(max(0,min(aucs)-.06),min(1,max(aucs)+.09))
ax0.legend(fontsize=9); ax0.grid(axis="x",alpha=.3); ax0.invert_yaxis()

mc=[MCOLORS[m] for m in contrib]
bars2=ax1.bar(list(contrib),list(contrib.values()),color=mc,edgecolor="white",alpha=.88,width=.45)
ax1.axhline(0,color="black",lw=.8)
for bar,d in zip(bars2,contrib.values()):
    ax1.text(bar.get_x()+bar.get_width()/2,d+(.001 if d>=0 else -.003),
             f"{d:+.4f}",ha="center",va="bottom" if d>=0 else "top",fontsize=11,fontweight="bold")
ax1.set_ylabel("DELTA-AUROC (Full - Ablated)\n+높을수록 중요")
ax1.set_title("Modality Contribution"); ax1.grid(axis="y",alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/modality_ablation_bar.png",dpi=150,bbox_inches="tight")
plt.close()
print("    bar.png 저장")

# ── 시각화 2: ROC ────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(7,6))
ax.plot([0,1],[0,1],"k--",lw=.8,alpha=.4)
styles=["-","--","-.",":","-","--","-."]; lws=[2.2]+[1.4]*6
for i,r in enumerate(results):
    fpr,tpr=roc(y,r["prob"])
    ax.plot(fpr,tpr,linestyle=styles[i],lw=lws[i],
            color=CMAP.get(r["name"],"#888"),label=f"{r['name']}  {r['auc']:.4f}")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves — Modality Ablation",fontweight="bold")
ax.legend(fontsize=8,loc="lower right"); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/modality_ablation_roc.png",dpi=150,bbox_inches="tight")
plt.close()
print("    roc.png 저장")

# ── 시각화 3: Weight 분석 ────────────────────────────────────
W_vec=W.reshape(-1)
fig,(axW,axS)=plt.subplots(1,2,figsize=(14,4))
fig.suptitle("TripleLinearL2 Weight Analysis",fontweight="bold")
seg_c=["#e66101"]*128+["#5e3c99"]*256+["#008837"]*256
axW.bar(np.arange(640),W_vec,color=seg_c,alpha=.7,linewidth=0)
for s,e,lbl,c in [(0,128,"clinical\n[0:128]","#e66101"),
                   (128,384,"radiomics\n[128:384]","#5e3c99"),
                   (384,640,"ct\n[384:640]","#008837")]:
    axW.axvline(s,color="black",lw=1.,linestyle=":")
    y_=axW.get_ylim()[1]*0.85 if axW.get_ylim()[1]!=0 else .01
    axW.text((s+e)/2,y_,lbl,ha="center",fontsize=9,color=c,fontweight="bold")
axW.axhline(0,color="black",lw=.5)
axW.set_xlabel("dim"); axW.set_ylabel("Weight")
axW.set_title("Linear Weights (640 dims)"); axW.grid(axis="y",alpha=.2)
mstats={m:{"mean_abs":float(np.mean(np.abs(W_vec[s:e]))),
            "sum_abs":float(np.sum(np.abs(W_vec[s:e]))),
            "std":float(np.std(W_vec[s:e]))} for m,(s,e) in SLICES.items()}
tot=sum(v["sum_abs"] for v in mstats.values())
x=np.arange(3); width=.25
for i,m in enumerate(MODALS):
    axS.bar(x+(i-1)*width,[mstats[m][k] for k in ["mean_abs","sum_abs","std"]],
            width,label=m,color=MCOLORS[m],alpha=.82,edgecolor="white")
for i,m in enumerate(MODALS):
    pct=mstats[m]["sum_abs"]/tot*100
    axS.text(1+(i-1)*width,mstats[m]["sum_abs"]+tot*.012,
             f"{pct:.1f}%",ha="center",fontsize=8,color=MCOLORS[m],fontweight="bold")
axS.set_xticks(x); axS.set_xticklabels(["|W| mean","|W| sum","W std"])
axS.set_title("Weight Stats per Modality")
axS.legend(fontsize=9); axS.grid(axis="y",alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/modality_ablation_coef.png",dpi=150,bbox_inches="tight")
plt.close()
print("    coef.png 저장")

# ── 시각화 4: Heatmap ────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,3))
usage=np.array([[0. if m in zeros else 1. for _,zeros,_ in SCEN] for m in MODALS])
cmap2=LinearSegmentedColormap.from_list("rg",["#d73027","#1a9641"],N=2)
ax.imshow(usage,aspect="auto",cmap=cmap2,vmin=0,vmax=1,interpolation="nearest")
ax.set_xticks(np.arange(len(SCEN)))
ax.set_xticklabels([f"{r['name']}\n{r['auc']:.4f}" for r in results],fontsize=9,rotation=25,ha="right")
ax.set_yticks(np.arange(3))
ax.set_yticklabels(["Clinical\n128-dim","Radiomics\n256-dim","CT\n256-dim"],fontsize=10)
ax.set_title("Modality Ablation Heatmap   green=used   red=zeroed",fontweight="bold")
for j in range(len(SCEN)):
    for i in range(3):
        ax.text(j,i,"O" if usage[i,j] else "X",
                ha="center",va="center",fontsize=15,color="white",fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/modality_ablation_heatmap.png",dpi=150,bbox_inches="tight")
plt.close()
print("    heatmap.png 저장")

# ── 요약 ─────────────────────────────────────────────────────
print("\n"+"="*60)
print(f"  Full AUROC = {full_auc:.4f}")
print()
print("  기여도 순위 (DELTA-AUROC):")
for rank,(m,d) in enumerate(sorted(contrib.items(),key=lambda x:-x[1]),1):
    bar="X"*max(1,int(abs(d)/max_c*20)) if max_c>1e-6 else ""
    print(f"  {rank}. {m:<12} {d:+.4f}  {bar}")
print()
print("  단일 모달 AUC:")
for r in results[4:]:
    print(f"    {r['name']:<22} {r['auc']:.4f}")
print()
print("  Weight 비중 (|W| sum):")
for m in MODALS:
    pct=mstats[m]["sum_abs"]/tot*100
    print(f"    {m:<12} {pct:5.1f}%  {'X'*int(pct/4)}")
print()
print(f"  출력: {OUTPUT_DIR}/")
print("="*60)
print("Done!")
