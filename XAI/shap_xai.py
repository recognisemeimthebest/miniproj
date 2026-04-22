"""
SHAP XAI - triple_linear_l2_best.pt (Linear Model)
Linear SHAP: phi_j(x) = W[j] * (Xn[j] - E_bg[Xn[:,j]])
Modality importance: Sum |SHAP| per patient (avg over patients)
"""
import os, zipfile, pickle, io
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

MODEL_PATH = "triple_linear_l2_best.pt"
EMB_DIR    = "triple_model/embeddings"
OUTPUT_DIR = "output_xai"
SPLIT      = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODALS = ["clinical","radiomics","ct"]
SLICES = {"clinical":(0,128),"radiomics":(128,384),"ct":(384,640)}
MCOLS  = {"clinical":"#e66101","radiomics":"#5e3c99","ct":"#008837"}

def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-30,30)))

def auroc_np(y, p):
    y,p = np.asarray(y,np.float32), np.asarray(p,np.float32)
    pos,neg = p[y==1], p[y==0]
    if not len(pos) or not len(neg): return .5
    order = np.argsort(p)[::-1]; ys=y[order]
    cp = np.cumsum(ys)
    return sum(cp[i] for i,v in enumerate(ys) if v==0)/(len(pos)*len(neg))

class _OD(dict): pass

class _UP(pickle.Unpickler):
    def __init__(self, f, sm): super().__init__(f); self._sm=sm
    def find_class(self,m,n):
        if m=="torch._utils" and n=="_rebuild_tensor_v2": return self._rb
        if "torch" in m or (m=="collections" and n=="OrderedDict"): return _OD
        try: return super().find_class(m,n)
        except: return _OD
    def persistent_load(self,pid):
        if isinstance(pid,(list,tuple)) and len(pid)>=5: return (pid[2],pid[4])
        if isinstance(pid,(list,tuple)) and len(pid)==4: return (pid[1],pid[3])
        return pid
    def _rb(self,spid,off,shape,stride,rg,bw):
        key,_ = spid; raw=self._sm.get(key)
        if raw is None: return np.zeros(shape,np.float32)
        flat=np.frombuffer(raw,np.float32); n=int(np.prod(shape)) if shape else 1
        return flat[off:off+n].reshape(shape)

def load_model(path):
    with zipfile.ZipFile(path,"r") as zf:
        names=zf.namelist(); prefix=names[0].split("/")[0]+"/"
        sm={n.split("/data/")[1]:zf.read(n) for n in names if "/data/" in n}
        pkl=zf.read(prefix+"data.pkl")
    obj=_UP(io.BytesIO(pkl),sm).load()
    sd=obj.get("state_dict",obj)
    mean =np.array(sd["mean"],  np.float32)
    scale=np.array(sd["scale"], np.float32)
    W    =np.array(sd["linear.weight"],np.float32)
    b    =np.array(sd["linear.bias"],  np.float32)
    return mean,scale,W.flatten(),b[0]

def load_emb(split):
    parts,y=[],None
    for m in MODALS:
        fp=os.path.join(EMB_DIR,f"{m}_{split}.npz")
        d=np.load(fp,allow_pickle=True)
        parts.append(d["emb"].astype(np.float32))
        if y is None: y=d["y"].astype(np.float32)
    return np.concatenate(parts,1), y

# ── 1. Load ─────────────────────────────────────────────
print("="*60); print("  SHAP XAI - Linear Model"); print("="*60)
mean,scale,W,b = load_model(MODEL_PATH)
X,y = load_emb(SPLIT)
Xn  = (X - mean)/(scale+1e-8)
N,D = Xn.shape
logits = Xn @ W + b
probs  = sigmoid(logits)
auc    = auroc_np(y,probs)
print(f"\nAUROC={auc:.4f}  N={N}  features={D}")

# ── 2. Linear SHAP ──────────────────────────────────────
bg        = Xn.mean(axis=0)
shap_vals = (Xn - bg) * W
base_val  = sigmoid(bg @ W + b)
print(f"base value: {base_val:.4f}")

# ── 3. Feature names ────────────────────────────────────
feat_names = []
for m in MODALS:
    s,e = SLICES[m]
    for i in range(e-s): feat_names.append(f"{m}_{i}")
feat_names = np.array(feat_names)

# ── 4. Modality importance: Sum |SHAP| per patient ──────
# correct metric: sum over dims then average over patients
print("\n[Modality Sum |SHAP| per patient (avg)]")
modal_shap = {}
for m in MODALS:
    s,e = SLICES[m]
    ms  = np.abs(shap_vals[:,s:e]).sum(axis=1).mean()
    modal_shap[m] = ms
    print(f"  {m:12s}: {ms:.5f}  (dims={e-s})")

# ── 5. Top features by mean |SHAP| ──────────────────────
mean_abs = np.abs(shap_vals).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:20]

def feat_color(idx):
    for m,(s,e) in SLICES.items():
        if s<=idx<e: return MCOLS[m]
    return "gray"

# ═══ Fig1: Modality bar (Sum |SHAP|, sorted) ════════════
fig,ax = plt.subplots(figsize=(7,4))
sorted_modals = sorted(MODALS, key=lambda m: modal_shap[m])
sorted_vals   = [modal_shap[m] for m in sorted_modals]
sorted_colors = [MCOLS[m] for m in sorted_modals]
bars = ax.barh(sorted_modals, sorted_vals, color=sorted_colors,
               edgecolor="white", alpha=0.88)
best_m = max(modal_shap, key=modal_shap.get)
for bar,m,v in zip(bars,sorted_modals,sorted_vals):
    if m==best_m: bar.set_edgecolor("black"); bar.set_linewidth(2)
    ax.text(v*1.01, bar.get_y()+bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean Sum |SHAP| per patient  (sum over dims, avg over patients)")
ax.set_title("SHAP Modality Importance  (dimension-count-aware)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_modality_bar.png",dpi=150,bbox_inches="tight")
plt.close(); print("\nshap_modality_bar.png saved")

# ═══ Fig2: Top-20 feature bar ═══════════════════════════
fig,ax = plt.subplots(figsize=(8,7))
top_names  = feat_names[top_idx]
top_vals   = mean_abs[top_idx]
top_colors = [feat_color(i) for i in top_idx]
ys = np.arange(len(top_idx))
ax.barh(ys, top_vals, color=top_colors, edgecolor="white", alpha=0.85)
ax.set_yticks(ys); ax.set_yticklabels(top_names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top-20 Features - Mean |SHAP|")
legend=[Patch(color=MCOLS[m],label=m) for m in MODALS]
ax.legend(handles=legend, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_top20_bar.png",dpi=150,bbox_inches="tight")
plt.close(); print("shap_top20_bar.png saved")

# ═══ Fig3: Beeswarm (top 15) ════════════════════════════
top15_idx   = top_idx[:15]
top15_names = feat_names[top15_idx]
fig,ax = plt.subplots(figsize=(9,6))
norm_c = Normalize(vmin=0, vmax=1)
for row, fi in enumerate(top15_idx[::-1]):
    sv  = shap_vals[:, fi]
    fv  = Xn[:, fi]
    fv_n= (fv-fv.min())/(fv.max()-fv.min()+1e-9)
    jit = np.random.default_rng(fi).uniform(-0.25,0.25, N)
    ax.scatter(sv, np.full(N,row)+jit, c=fv_n,
               cmap="coolwarm", s=12, alpha=0.7, linewidths=0, vmin=0, vmax=1)
ax.set_yticks(np.arange(len(top15_idx)))
ax.set_yticklabels(top15_names[::-1], fontsize=8)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("SHAP value (log-odds contribution)")
ax.set_title("SHAP Summary Plot (Beeswarm) - Top 15 Features")
sm2 = cm.ScalarMappable(cmap="coolwarm", norm=norm_c)
sm2.set_array([])
cb = plt.colorbar(sm2, ax=ax, shrink=0.6, pad=0.01)
cb.set_label("Feature value (normalized)", fontsize=8)
cb.set_ticks([0,0.5,1]); cb.set_ticklabels(["Low","Mid","High"])
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png",dpi=150,bbox_inches="tight")
plt.close(); print("shap_beeswarm.png saved")

# ═══ Fig4: Waterfall (2 patients) ═══════════════════════
def waterfall(ax, sv, base, pred, title, top_n=10):
    order     = np.argsort(np.abs(sv))[::-1][:top_n]
    rest      = sv.sum() - sv[order].sum()
    entries_v = list(sv[order]) + [rest]
    entries_l = list(feat_names[order]) + ["other features"]
    cols      = ["#d73027" if v>0 else "#4575b4" for v in entries_v]
    cumulative= base
    bottoms, heights = [], []
    for v in entries_v:
        if v >= 0: bottoms.append(cumulative); heights.append(v)
        else:      bottoms.append(cumulative+v); heights.append(-v)
        cumulative += v
    ys = np.arange(len(entries_v))
    ax.barh(ys, heights, left=bottoms, color=cols,
            edgecolor="white", alpha=0.88, height=0.7)
    ax.axvline(base, color="gray", lw=1, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels(entries_l, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP (log-odds contribution)")
    ax.set_title(f"{title}\nbase={base:.3f}  pred={pred:.3f}", fontsize=9)
    for i,(v,bot,h) in enumerate(zip(entries_v,bottoms,heights)):
        ax.text(bot+h+0.0002, i, f"{v:+.4f}", va="center", fontsize=6.5)
    ax.grid(axis="x", alpha=0.2)

top_pos = np.argmax(probs * (y==1))
top_neg = np.argmin(probs + (y==1)*10)
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle("SHAP Waterfall - Individual Patients", fontsize=12, fontweight="bold")
waterfall(axes[0], shap_vals[top_pos], base_val, probs[top_pos],
          f"Patient #{top_pos}  (True=Survived, Pred={probs[top_pos]:.3f})")
waterfall(axes[1], shap_vals[top_neg], base_val, probs[top_neg],
          f"Patient #{top_neg}  (True=Died, Pred={probs[top_neg]:.3f})")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png",dpi=150,bbox_inches="tight")
plt.close(); print("shap_waterfall.png saved")

# ── Summary ─────────────────────────────────────────────
print("\n"+"="*60)
print("  SHAP Result Summary")
print("="*60)
print(f"\nBase value (background pred): {base_val:.4f}")
print(f"Model AUROC:                   {auc:.4f}\n")
print("Modality Sum |SHAP| per patient (ranked):")
for m,v in sorted(modal_shap.items(), key=lambda x:-x[1]):
    s,e = SLICES[m]
    bar = "X"*int(v/max(modal_shap.values())*20)
    print(f"  {m:12s}  {v:.5f}  (dims={e-s})  {bar}")
print("\nTop-5 features (mean |SHAP|):")
for rank,fi in enumerate(top_idx[:5],1):
    print(f"  {rank}. {feat_names[fi]:22s}  {mean_abs[fi]:.5f}")
print(f"\nOutput: {OUTPUT_DIR}/shap_*.png  (4 files)")
print("="*60)
