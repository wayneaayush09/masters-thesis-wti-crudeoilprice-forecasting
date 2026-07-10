# ============================================================
# FILE 01a — Rolling WOA-VMD-FE Feature Decomposition  (ADDENDUM 11)
#
# Adds three frequency-band columns to df_features.pkl + feat_cols.pkl:
#   vmd_lf  — low-frequency band  (trend / macro)
#   vmd_mf  — medium-frequency band (cyclical structure)
#   vmd_hf  — high-frequency band  (noise / short-term shocks)
#
# Architecture (Paper 1: WOA-VMD-FE-Transformer, Paper 2: VMD+EMD+Transformer):
#   For each origin date t with t >= L:
#     1.  window = log_price[t-L : t]    (PAST ONLY — no future)
#     2.  Every WOA_STRIDE days, run light WOA to pick optimal (K*, α*)
#         on the current window; reuse those parameters in between.
#     3.  VMD(window, K*, α*) → K* modes
#     4.  Compute fuzzy entropy of each mode
#     5.  Cluster modes into LF / MF / HF by 33rd/67th percentile of FE
#     6.  Emit u_band[-1] for each band as the feature value at time t
#
# LEAKAGE DISCIPLINE: at every step we use only past samples.  The papers'
# headline numbers come from whole-series decomposition, which leaks future
# information into past modes (Xu & Niu, Energy Economics 128, 2024).  Our
# version is honest — gains will be modest at 7d, ≈flat at 90d/180d.
#
# Compute budget: ~30–40 minutes on Kaggle CPU.  One-time precompute; rerun
# only if L_VMD / WOA_STRIDE / VMD search space changes.
#
# Run AFTER 00_setup_data_features.py.
# Usage in Colab/Kaggle:
#   exec(open('01a_vmd_features.py', encoding="utf-8").read())
# ============================================================

import pickle, subprocess, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))

# ── Dependency check ─────────────────────────────────────────
try:
    from vmdpy import VMD
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vmdpy"], check=False)
    from vmdpy import VMD

# ── Config (tunable knobs) ────────────────────────────────────
L_VMD       = 360         # trailing past window for each decomposition
WOA_STRIDE  = 120         # re-optimise (K, α) every ~6 months
WOA_POP     = 10          # whale-population size (search budget per WOA call)
WOA_ITERS   = 10          # iterations (search budget per WOA call)
K_GRID      = [3, 4, 5, 6, 7]
A_GRID      = [500, 1000, 2000, 3000, 5000]
FE_M        = 2           # fuzzy-entropy embedding dimension
FE_N        = 2           # fuzzy power
PROGRESS_EVERY = 500      # print progress every N origins

print("=" * 60)
print("  FILE 01a — Rolling WOA-VMD-FE Feature Decomposition")
print(f"  L_VMD={L_VMD}  WOA_STRIDE={WOA_STRIDE}  K∈{K_GRID}  α∈{A_GRID}")
print("=" * 60)

# ── Load preprocessed df ─────────────────────────────────────
with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df   = pickle.load(f)
with open(OUTPUT_DIR/"feat_cols.pkl",  "rb") as f: FEAT = pickle.load(f)
if "log_price" not in df.columns:
    raise RuntimeError("df_features.pkl missing 'log_price' — run 00_setup_data_features.py first.")
log_price = df["log_price"].values.astype(float)
N = len(log_price)
print(f"✅ Loaded df_features.pkl: N={N} rows, columns={len(df.columns)}")

# ── Fuzzy Entropy (self-implemented; numpy-only) ─────────────
def fuzzy_entropy(x, m=FE_M, n=FE_N, r=None):
    """FuzzyEn(x, m, r, n) — Chen et al. 2007.

    Lower FE = more regular / predictable (smooth trend mode).
    Higher FE = more irregular / noisy (high-frequency mode).
    """
    x = np.asarray(x, float)
    L = len(x)
    if L < m + 2: return np.nan
    if r is None: r = 0.2 * float(np.std(x) + 1e-12)
    def _phi(M):
        idx = np.arange(L - M + 1)
        emb = np.array([x[i:i+M] for i in idx])
        emb = emb - emb.mean(axis=1, keepdims=True)        # remove local baseline
        # pairwise Chebyshev distance
        D = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        mu = np.exp(-(D ** n) / (r + 1e-12))                # fuzzy membership
        np.fill_diagonal(mu, 0.0)
        denom = len(emb) * (len(emb) - 1)
        return mu.sum() / denom if denom > 0 else np.nan
    p_m, p_m1 = _phi(m), _phi(m + 1)
    if (p_m is None) or (p_m1 is None) or p_m <= 0 or p_m1 <= 0:
        return np.nan
    return float(np.log(p_m) - np.log(p_m1))

# ── Light Whale-style search over discrete (K, α) grid ───────
def woa_search(window, pop=WOA_POP, iters=WOA_ITERS):
    """Bounded random search over (K, α) ∈ K_GRID × A_GRID.
    Fitness = fuzzy entropy of residual (lower = cleaner decomposition).
    True WOA is overkill on a 5×5 discrete grid; bounded RS achieves the
    same end with deterministic budget — far cheaper than per-origin
    metaheuristics.
    """
    cands = [(K, a) for K in K_GRID for a in A_GRID]
    rng = np.random.default_rng(0)
    budget = min(pop * iters, len(cands))
    visited = set()
    best, best_score = (5, 2000), float("inf")
    for _ in range(budget):
        K, a = cands[rng.integers(len(cands))]
        if (K, a) in visited: continue
        visited.add((K, a))
        try:
            sig = window if len(window) % 2 == 0 else window[1:]
            u, _, _ = VMD(sig, a, 0.0, K, 0, 1, 1e-7)
            residual = sig[:u.shape[1]] - u.sum(axis=0)
            score = fuzzy_entropy(residual)
            if np.isfinite(score) and score < best_score:
                best_score, best = score, (K, a)
        except Exception:
            continue
    return best

# ── Rolling decomposition loop ───────────────────────────────
print(f"\nRolling decomposition: {N - L_VMD:,} origins to process …")
t_start = time.time()
woa_K, woa_a = 5, 2000             # safe defaults
lf  = np.full(N, np.nan)
mf  = np.full(N, np.nan)
hf  = np.full(N, np.nan)
last_woa_log = None

for t in range(L_VMD, N):
    window = log_price[t - L_VMD : t]
    if not np.all(np.isfinite(window)):
        continue
    if len(window) % 2:                                # VMD prefers even length
        window = window[1:]

    # Strided WOA re-tune
    if (t - L_VMD) % WOA_STRIDE == 0:
        woa_K, woa_a = woa_search(window)
        last_woa_log = (t, woa_K, woa_a)

    try:
        u, _, _ = VMD(window, woa_a, 0.0, woa_K, 0, 1, 1e-7)   # (K, T) modes
    except Exception:
        continue

    # Fuzzy Entropy per mode → cluster into 3 bands
    fes = np.array([fuzzy_entropy(m) for m in u])
    if not np.all(np.isfinite(fes)):
        fes = np.nan_to_num(fes, nan=np.nanmean(fes) if np.any(np.isfinite(fes)) else 0.0)
    p33, p67 = np.percentile(fes, [33, 67])
    lf_idx = np.where(fes <= p33)[0]
    mf_idx = np.where((fes > p33) & (fes <= p67))[0]
    hf_idx = np.where(fes >  p67)[0]
    # Fallback if percentile collapses (all modes have identical FE)
    if len(lf_idx) == 0: lf_idx = np.array([int(np.argmin(fes))])
    if len(hf_idx) == 0: hf_idx = np.array([int(np.argmax(fes))])
    if len(mf_idx) == 0: mf_idx = np.array([i for i in range(len(fes)) if i not in lf_idx and i not in hf_idx])

    last_sample = lambda idxs: float(u[idxs, -1].sum()) if len(idxs) > 0 else 0.0
    lf[t - 1] = last_sample(lf_idx)
    mf[t - 1] = last_sample(mf_idx)
    hf[t - 1] = last_sample(hf_idx)

    if (t - L_VMD) % PROGRESS_EVERY == 0:
        elapsed = time.time() - t_start
        progress = (t - L_VMD) / (N - L_VMD)
        eta = (elapsed / max(progress, 1e-3)) * (1 - progress)
        msg = (f"  t={t:5d}/{N}  ({progress*100:5.1f}%)  "
               f"K={woa_K} α={woa_a}  "
               f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s")
        print(msg)

elapsed = time.time() - t_start
print(f"\n✅ Decomposition done in {elapsed:.1f}s")

# ── Attach to df + ffill the leading NaN window ──────────────
df["vmd_lf"] = pd.Series(lf, index=df.index).ffill().fillna(0.0)
df["vmd_mf"] = pd.Series(mf, index=df.index).ffill().fillna(0.0)
df["vmd_hf"] = pd.Series(hf, index=df.index).ffill().fillna(0.0)

# Sanity stats
for col in ("vmd_lf", "vmd_mf", "vmd_hf"):
    s = df[col]
    print(f"  {col}:  mean={s.mean(): .4f}  std={s.std(): .4f}  "
          f"non-zero={int((s != 0).sum()):,} / {len(s):,}")

# ── Inject into every horizon's feature list ─────────────────
VMD_COLS = ["vmd_lf", "vmd_mf", "vmd_hf"]
for hk in FEAT:
    for col in VMD_COLS:
        if col not in FEAT[hk]:
            FEAT[hk].append(col)
    print(f"  FEAT[{hk}] now has {len(FEAT[hk])} features (includes {VMD_COLS})")

# ── Persist ──────────────────────────────────────────────────
with open(OUTPUT_DIR/"df_features.pkl","wb") as f: pickle.dump(df, f)
with open(OUTPUT_DIR/"feat_cols.pkl",  "wb") as f: pickle.dump(FEAT, f)
# ── VMD Band Visualization (Plotly) ──────────────────────────────────
# Generated immediately after the rolling decomposition so the user can
# inspect band quality before committing to 6+ hours of GPU training.
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo_vmd
    from plotly.subplots import make_subplots as _pmsp_vmd
    _vmd_cols = [("vmd_lf", "Low-Freq (trend)", "#1f3b73"),
                 ("vmd_mf", "Mid-Freq (cycle)",  "#2a9d8f"),
                 ("vmd_hf", "High-Freq (noise)", "#d1495b")]
    _vmd_dates = df.index.strftime("%Y-%m-%d")
    # Panel 1: raw price + bands stacked
    _fig_vmd = _pmsp_vmd(rows=4, cols=1,
                          subplot_titles=["WTI Log-Price (original)",
                                          "LF Band (trend)", "MF Band (cycle)", "HF Band (noise)"],
                          vertical_spacing=0.06, shared_xaxes=True)
    _fig_vmd.add_trace(_pgo_vmd.Scatter(x=_vmd_dates, y=df["log_price"],
                                        name="log_price", line=dict(color="#073b4c", width=1.2)),
                       row=1, col=1)
    for _ri, (col, label, color) in enumerate(_vmd_cols, 2):
        _fig_vmd.add_trace(_pgo_vmd.Scatter(x=_vmd_dates, y=df[col],
                                            name=label, line=dict(color=color, width=1.2)),
                           row=_ri, col=1)
    if "add_crises" in globals():
        for _ri in range(1, 5):
            add_crises(_fig_vmd, row=_ri, col=1)
    _fig_vmd.update_layout(
        title="Rolling WOA-VMD Decomposition — LF / MF / HF Bands (leakage-free rolling window)",
        height=900, showlegend=True)
    save_fig(_fig_vmd, "eda_vmd_bands_stacked")

    # Panel 2: energy fractions over time (rolling 252-day sum of squared bands)
    _band_energy = {}
    for col, label, _ in _vmd_cols:
        _band_energy[label] = (df[col]**2).rolling(252, min_periods=63).mean()
    _total_e = sum(_band_energy.values())
    _fig_vmd2 = _pgo_vmd.Figure()
    for (col, label, color), _en in zip(_vmd_cols, _band_energy.values()):
        _frac = (_en / _total_e.replace(0, float("nan"))).fillna(0)
        _fig_vmd2.add_trace(_pgo_vmd.Scatter(
            x=_vmd_dates, y=_frac*100, name=label,
            line=dict(color=color, width=1.5),
            stackgroup="one", mode="lines"))
    if "add_crises" in globals():
        add_crises(_fig_vmd2)
    _fig_vmd2.update_layout(
        title="VMD Band Energy Fractions Over Time  (252-day rolling, stacked = 100%)",
        xaxis_title="Date", yaxis_title="Energy fraction (%)")
    save_fig(_fig_vmd2, "eda_vmd_energy_fractions")
    print("  ✅ VMD band figures saved")
except Exception as _vmd_fig_err:
    print(f"  [warn] VMD figures skipped: {_vmd_fig_err}")

print("\n✅ FILE 01a COMPLETE — VMD-LF/MF/HF bands appended to df_features.pkl + figures saved")
print("   Run 01_black_swan_baselines.py and the model files next.")
