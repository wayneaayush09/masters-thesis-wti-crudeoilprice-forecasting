# DDP_v2 — Step-by-Step Running Procedure

## The 3-Phase Workflow at a Glance

```
Phase 1 — LOCAL (free, CPU)       Phase 2 — CLOUD GPU               Phase 3 — LOCAL (analysis)
────────────────────────────  →   ────────────────────────────  →   ────────────────────────────
run_local_cpu.py                  run_short.py  (1d + 7d)           C_ensemble.py
  builds data, features,          run_long.py   (90d + 180d)        C_compare.py
  baselines, CPU models,          trains all neural-net archs        06_analysis_plots.py
  decomposition caches                                               17_audit_pipeline.py

  ~1–2 hours                          ~3–6 hours                          ~30 minutes
```

> **Why this split?** Everything in Phase 1 is pure math / data loading — no GPU needed.
> Running it locally for free means the expensive cloud GPU session starts
> *straight into NN training*, with data and caches already on disk.

---

## Phase 1 — Local Machine (CPU only, run ONCE)

### 1a. Set your storage location (pick ONE)

```python
import os

# Option A — results saved locally (easiest to start with)
os.environ['DDP_OUTPUT_DIR'] = r'C:\Users\anand\Desktop\DDP\results'

# Option B — results on Google Drive (auto-available in Colab later)
# os.environ['DDP_OUTPUT_DIR'] = r'G:\My Drive\DDP_results'

os.chdir(r'C:\Users\anand\Desktop\DDP\DDP code updated\DDP_v2')
```

### 1b. Run the CPU prep orchestrator

```python
exec(open('run_local_cpu.py').read())
```

**What this builds (in order, auto-skipping anything already done):**

| Step | File | What it creates | Time (first run) |
|------|------|-----------------|------------------|
| 1 | `00_setup_data_features.py` | `df_features.pkl`, `feat_cols.pkl`, `wti_raw.pkl`, `regimes.pkl` | ~5–15 min |
| 2 | `01a_vmd_features.py` | Rolling VMD columns added to `df_features.pkl` | ~30–40 min |
| 3 | `01_black_swan_baselines.py` | `baselines_results.pkl`, `baseline_metrics.csv` | ~5 min |
| 4 | `10_text_features.py` | `text_features_daily.csv` *(needs internet; skip if slow)* | ~10 min |
| 5 | `09_vmd_hybrid.py` | `predictions_VMD_{hk}.csv` (leakage-free hybrid) | ~60 min |
| 6 | `A_components_linear.py` | `predictions_LHYu_dlinear_{hk}.csv` etc. **+ decomp caches** | ~15 min |
| 7 | `A_components_classical.py` | `predictions_LHYu_ar_{hk}.csv` etc. | ~10 min |
| 8 | `A_components_hybrid.py` | `predictions_LHYu_select_{hk}.csv` etc. | ~20 min |
| 9 | `B_honest_decomp.py` | `predictions_LHStandalone_honest_{hk}.csv` | ~60 min |
| 10 | `R_timerag.py` | `predictions_RAGForecaster_{hk}.csv` | ~5 min |
| 11 | `07_risk_evt.py` | `risk_evt_{hk}.pkl` (GPD tail risk) | ~1 min |
| 12 | `07b_garch_evt.py` | `garch_evt_{hk}.pkl` (dynamic VaR/ES) | ~2 min |

> **Total first run: ~3–4 hours.** After the first run, re-running is ~5 min because every
> file checks if its output already exists and skips automatically.

> **Most important output from Step 6:** `cache/yu_decomp_{hk}.pkl` and
> `cache/huang_decomp_{hk}.pkl`. These save the ~7-minute WOA-VMD decomposition so
> the Colab session loads it in **less than 1 second** instead of recomputing it.

### 1c. Sync results to Google Drive

If you chose Option A (local path), copy the `results/` folder to your Drive:
- **Windows**: drag the `results/` folder into the Google Drive desktop app folder (syncs automatically)
- **Or**: set `DDP_OUTPUT_DIR` to your Google Drive path from the start (Option B above) so it writes there directly

---

## Phase 2 — Google Colab or Kaggle (GPU required)

### 2a. Upload the code (one-time)

Upload the entire `DDP_v2/` folder to Colab/Kaggle, or keep it in Drive and cd into it.

### 2b. Bootstrap cell (run at the start of EVERY GPU session)

```python
from google.colab import drive
drive.mount('/content/drive')
import os

# Point to the SAME results folder as Phase 1
os.environ['DDP_OUTPUT_DIR'] = '/content/drive/MyDrive/DDP_results'
os.chdir('/content/drive/MyDrive/DDP_v2')   # or wherever your code lives

exec(open('_env.py').read())       # sets OUTPUT_DIR, TRACK_HORIZONS, already_done()
exec(open('_plotting.py').read())  # Plotly theme, save_fig(), add_crises()
```

### 2c. Run the GPU training (choose one track per session)

**Short horizons — 1-day and 7-day:**
```python
os.environ['DDP_TRACK'] = 'short'
exec(open('run_short.py').read())
```

**Long horizons — 90-day and 180-day (separate session recommended):**
```python
os.environ['DDP_TRACK'] = 'long'
exec(open('run_long.py').read())
```

**The orchestrator runs these GPU-heavy files (in order):**
```
A_yu_patchtst.py          ← Track A: WOA-VMD + PatchTST  (leaky benchmark)
A_huang_patchtst.py       ← Track A: VMD+EMD + PatchTST
B_wf_patchtst.py          ← Track B: honest walk-forward PatchTST  ← YOUR THESIS NUMBERS
B_wf_itransformer.py      ← Track B: iTransformer
B_foundation_chronos.py   ← zero-shot Amazon Chronos
B_foundation_timesfm.py   ← zero-shot Google TimesFM
... (add more B_wf_*.py / A_*.py to the STAGES list in run_short.py)
```

### 2d. After a disconnect — just re-run 2b + 2c

The skip-guard checks which `predictions_*.csv` files already exist:
- You'll see `⏭️  skip PatchTST 7d (already done)` for finished models
- Per-fold checkpoints inside long training runs let it resume mid-model too
- Decomposition caches load in <1 second — no recompute

### 2e. Add more architectures (optional)

Edit the `STAGES` list inside `run_short.py` or `run_long.py` and add any of:
```python
"A_yu_itransformer.py",    "A_huang_itransformer.py",
"B_wf_tft.py",             "B_wf_nhits.py",
"B_wf_timemixer.py",       "B_wf_fedformer.py",
"B_wf_segrnn.py",          "B_wf_tcn.py",
"B_regime_conditional.py", # regime-aware (4× slower, needs GPU)
```

---

## Phase 3 — Local Machine Again (analysis & reports)

After the Colab session finishes (Drive will have synced the new `predictions_*.csv` files):

```python
import os
os.environ['DDP_OUTPUT_DIR'] = r'C:\Users\anand\Desktop\DDP\results'
os.chdir(r'C:\Users\anand\Desktop\DDP\DDP code updated\DDP_v2')
exec(open('_env.py').read())
exec(open('_plotting.py').read())

exec(open('13_probabilistic_forecast.py').read())  # quantile fans + scenarios
exec(open('C_ensemble.py').read())                  # SLSQP blend of all models
exec(open('C_compare.py').read())                   # rank all models vs benchmark
exec(open('06_analysis_plots.py').read())           # publication figures (Plotly)
exec(open('17_audit_pipeline.py').read())           # final audit report
```

---

## Complete File Routing Reference

### 🟢 Run on LOCAL MACHINE (CPU, free)

| File | Purpose |
|------|---------|
| `00_setup_data_features.py` | Master data build — **run this first** |
| `01a_vmd_features.py` | Rolling VMD feature columns |
| `01_black_swan_baselines.py` | Naive / ARIMA / ETS baselines |
| `10_text_features.py` | News sentiment *(optional, needs internet)* |
| `A_components_linear.py` | DLinear / NLinear / Ridge on leaky VMD bands |
| `A_components_classical.py` | AR / Theta on leaky VMD bands |
| `A_components_hybrid.py` | Best-method-per-band auto-selection |
| `B_honest_decomp.py` | Leakage-free rolling VMD + AR |
| `09_vmd_hybrid.py` | Leakage-free VMD hybrid |
| `R_timerag.py` | Analog-retrieval forecaster |
| `07_risk_evt.py` | GPD tail risk (VaR / ES) |
| `07b_garch_evt.py` | Dynamic GARCH-EVT (conditional VaR) |
| `13_probabilistic_forecast.py` | Quantile fan + 500-path scenarios |
| `C_ensemble.py` | SLSQP blend of all model predictions |
| `C_compare.py` | Model ranking vs Huang 3.45 benchmark |
| `06_analysis_plots.py` | Publication figures (all cross-model) |
| `17_audit_pipeline.py` | Final connectivity audit |

### 🔴 Run on COLAB / KAGGLE (GPU required — neural-net training)

| Group | Files | What they do |
|-------|-------|-------------|
| Track A (leaky benchmark) | `A_yu_*.py` (18 files) | WOA-VMD + each transformer/RNN arch |
| Track A (leaky benchmark) | `A_huang_*.py` (18 files) | VMD+EMD + each arch |
| Track B (honest — thesis) | `B_wf_*.py` (17 files) | Walk-forward training for each arch |
| Track B special | `B_regime_conditional.py` | 4 regime-specific specialists |

### 🟡 Optional — either machine (foundation models, slow on CPU)

| File | What it does |
|------|-------------|
| `B_foundation_chronos.py` | Amazon Chronos-Bolt zero-shot |
| `B_foundation_timesfm.py` | Google TimesFM zero-shot |
| `12_agentic_rag.py` | Local LLM directional signals |
| `D_llm_news_scenario.py` | News headline what-if scenarios |

---

## Quick Tips

**Check what's already done:**
```
Open results/logs/run_manifest.jsonl
```

**Force a model to rerun** (ignore existing output):
```python
os.environ['DDP_FORCE_RERUN'] = '1'
exec(open('B_wf_patchtst.py').read())
os.environ['DDP_FORCE_RERUN'] = '0'  # reset when done
```

**Run just ONE model without the orchestrator:**
```python
exec(open('_env.py').read())
exec(open('B_wf_patchtst.py').read())   # runs for all horizons in TRACK_HORIZONS
```

**Run just ONE horizon:**
```python
TRACK_HORIZONS = ['90d']
exec(open('B_wf_patchtst.py').read())   # only 90-day
```

**See current leaderboard at any time** (after at least some models have run):
```python
exec(open('C_compare.py').read())
```

---

## Dependency Diagram

```
00_setup_data_features.py          ← START HERE (always)
         │
         ├── 01a_vmd_features.py   ← adds VMD columns to df_features
         ├── 01_black_swan_baselines.py
         │
         ├── A_components_linear.py   ┐
         ├── A_components_classical.py ├── CPU Track A (builds decomp caches too)
         ├── A_components_hybrid.py   ┘
         │
         ├── 09_vmd_hybrid.py      ← CPU Track B honest VMD
         ├── B_honest_decomp.py    ← CPU Track B rolling decomp
         ├── R_timerag.py          ← analog retrieval
         │
         ├── [COLAB/KAGGLE] ────────────────────────────────────────
         │    ├── A_yu_*.py / A_huang_*.py   (Track A leaky GPU)
         │    └── B_wf_*.py                  (Track B honest GPU)
         │
         ├── 07_risk_evt.py        ┐
         ├── 07b_garch_evt.py      ├── risk (can run after ANY model)
         ├── 13_probabilistic_forecast.py ┘
         │
         ├── C_ensemble.py         ← blend ALL models (run after all done)
         ├── C_compare.py          ← rank ALL models
         ├── 06_analysis_plots.py  ← publication figures
         └── 17_audit_pipeline.py  ← LAST (final check)
```
