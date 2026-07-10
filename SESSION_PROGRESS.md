# DDP_v2 — Session Progress Summary

> **Last updated:** 2026-06-26  
> **What this file is:** a plain-English record of every significant decision, bug fix, and code
> change made across all Claude sessions for this project. For *how to run* the pipeline see
> `RUNNING_GUIDE.md`; for *methodology* see `THESIS_METHODOLOGY.md`.

---

## Status at a Glance

| Track | Short (1d + 7d) | Long (90d + 180d) |
|---|---|---|
| Track A (leaky repro) | ✅ Complete (on disk in ddp-results1) | ✅ Complete |
| Track B (honest WF) | ❌ Lost — session died before save | ❌ Lost — session died before save |
| Phase 1 (CPU: features, baselines, caches) | ✅ Complete locally | ✅ Complete locally |
| Phase 3 (ensemble, compare, plots) | ⏳ Pending (needs Track B) | ⏳ Pending |
| Final report (LaTeX) | ✅ 1d + 7d results inserted; compiled on Overleaf | — |

**Next action:** Re-run Track B (all 17 B_wf models) on Colab using the Colab cells in this file.
Expected time with speed-up settings: ~2.5–3 hrs (short) per session.

---

## Session History

### Session 1 — Pipeline Architecture Restructure

**Problem:** pipeline was monolithic — all 47 stages ran in one shot, no resume on disconnect,
no GPU/CPU routing, all outputs to a hardcoded relative `outputs/` folder that was wiped on
every Colab session start.

**What was built:**

| File | Purpose |
|---|---|
| `_env.py` | Sets `OUTPUT_DIR` from `DDP_OUTPUT_DIR` env var, defines `already_done()` skip-guard, exposes `FORCE_RERUN` |
| `_plotting.py` | Centralized Plotly business theme: `save_fig()`, `save_table()`, `add_crises()`, `save_animation()` |
| `run_local_cpu.py` | CPU-only orchestrator — builds data, features, decomp caches, baselines locally |
| `run_short.py` | GPU orchestrator for 1d + 7d |
| `run_long.py` | GPU orchestrator for 90d + 180d |
| `COMPUTE_MANIFEST.md` | Routing table: which files are CPU_ONLY vs GPU_HEAVY vs GPU_OPTIONAL |
| `GLOSSARY.md` | Definitions of recurring terms (leaky/honest, WOA-VMD, RevIN, walk-forward, EVT…) |
| `THESIS_METHODOLOGY.md` | Full 11-chapter methodology for thesis examiners |
| `RUNNING_GUIDE.md` | Step-by-step run procedure |

**Key code changes applied across codebase (~25 files):**
- `OUTPUT_DIR` redirect: replaced all `Path("outputs")` literals with
  `Path(os.environ.get("DDP_OUTPUT_DIR","outputs")) / os.environ.get("DDP_TRACK","all")`
- `HORIZONS` global-override: all `B_wf_*.py` + analysis files now read
  `HORIZONS = globals().get("HORIZONS", [...])` so the orchestrator's choice wins
- Resume skip-guard added to `_run_wf_arch.py` and `_run_paper_arch.py`:
  `if already_done(name, hk): continue`
- Disk-backed decomposition caches in `_paper_helpers.py` / `_lh_runners.py`:
  saves `cache/yu_decomp_{hk}.pkl` so the ~7-min WOA-VMD never recomputes on reconnect
- Per-fold checkpoints in `_forecast_template.py`:
  `checkpoints/{MODEL}_{hk}_fold{NN}.pkl` written after each fold; loaded on resume
- `_orchestrate.py` manifest skip-guard: reads `logs/run_manifest.jsonl`, skips
  stages already marked "done" so a kernel restart continues where it stopped
- Plotly conversion: matplotlib removed from `00_setup`, `01_black_swan`, `07_risk_evt`,
  `13_probabilistic_forecast`, `D_llm_news_scenario`; replaced with `save_fig()` calls
- `_paper_helpers.py` torch-optional: soft flag `_TORCH_OK` instead of hard raise,
  so CPU-only environments can run Track A extrapolators without torch installed
- `09_vmd_hybrid.py`: removed from main `STAGES` (it blocked the pipeline for 10+ hrs on CPU
  on 90d/180d); `STRIDE` made configurable via `VMD_STRIDE` env var (default 20, was 5)
- `imageio` auto-install added to `_plotting.py` for GIF/MP4 animation export

**COT / Futures Positioning features added to `00_setup_data_features.py`:**
- Fetches CFTC Commitment of Traders (disaggregated, NYMEX WTI) weekly
- Fetches WTI futures term-structure slope (front vs deferred contract)
- CFTC indexed by **release date** (snapshot + 3 bday lag) — leakage-safe
- 4 derived features: `cot_mm_net`, `cot_mm_net_z`, `cot_mm_net_chg_5d`, `ts_slope`
- Added to `MED_FEATURES` (90d) and `LONG_FEATURES` (180d)
- Graceful fallback: zero-fill if fetch fails; pipeline never crashes

---

### Session 2 — Kaggle GPU Runs (Track A complete; Track B lost)

**What ran:**
- Phase 1 (local CPU): ✅ completed — `df_features.pkl`, decomp caches, baselines all on disk
- Track A (leaky): ✅ completed on Kaggle — 629-min run ("54 ok, 17 failed"); all Track A
  neural models and extrapolators saved to `ddp-results1` dataset
- Track B (honest): ❌ all 17 `B_wf_*` models failed every run

**Root cause of B_wf failures (proven):**

The stale `ddpv2-code` Kaggle dataset hardcodes `OUT = "outputs"` in `_features_plus.py:23`
(ignores `OUTPUT_DIR`). The local repo is already correct. Previous notebook patches used
exact-string matching that never matched the stale file → "already patched" → no-op → same
failure every run.

Evidence: `13_probabilistic_forecast.py` (after all B_wf in the same run) read
`OUTPUT_DIR/df_features.pkl` fine. The failure was isolated to one line in one stale file.

**Other hardcoded `OUT="outputs"` in stale dataset** (not in short/long STAGES, harmless):
`_llm_news.py:20`, `D_llm_news_scenario.py:16`, `E_uni_vs_multi.py:24` — the symlink fix
below also covers these for free.

**Track B also lost to session death:**
Even when Track B was partially running, Kaggle GPU sessions expired before Cell 5 (save
back to dataset) could run. All checkpoint and prediction files were wiped on session clear.

---

### Session 3 — Permanent B_wf Fix + Speed Reduction Plan

**Permanent path fix (notebook-only, no re-upload required):**

1. **Symlink** `DDP_v2/outputs` → real results folder: any `outputs/...` reference now
   resolves regardless of code version (belt)
2. **Hard-rewrite** the `OUT=` line in `_features_plus.py` by reading the file and replacing
   whatever that line is — not exact-string matching (suspenders)

```python
# Symlink (Cell 2 pattern):
REAL = "/content/DDP_results"; LINK = "/content/DDP_v2/outputs"
os.symlink(REAL, LINK)

# Hard-rewrite (Cell 2 pattern):
fp = pathlib.Path("/content/DDP_v2/_features_plus.py")
lines = fp.read_text(encoding="utf-8").splitlines()
for i, l in enumerate(lines[:40]):
    st = l.strip()
    if (st.startswith("OUT ") or st.startswith("OUT=")) and "df_features" not in st:
        lines[i] = 'OUT = os.environ.get("DDP_OUTPUT_DIR") or str(globals().get("OUTPUT_DIR", "outputs"))'
        break
fp.write_text("\n".join(lines), encoding="utf-8")
```

**Speed reduction (accepted tier: "Neutral + 2 seeds + fewer folds", ~3–3.5× faster):**

Three changes, two files:

| Change | File | Effect | Metric impact |
|---|---|---|---|
| `cudnn.benchmark = True` | `02_model_shared.py` | Autotunes conv kernels (biggest win for TCN/TCNBiLSTM/CNNBiLSTMAttn) | None (bit-identical) |
| TF32 flags | `02_model_shared.py` | Free on Ampere/Ada GPUs (L4/A100); no-op on T4 | ~1e-3 on Ampere only |
| `pin_memory=IS_CUDA` on DataLoaders | `02_model_shared.py` | Overlaps host→device copy with compute | None |
| `non_blocking=IS_CUDA` on `.to(DEVICE)` | `02_model_shared.py` | Same as above | None |
| `N_SEEDS_OVERRIDE = 2` (was 3) | set in run cell | −33% trainings; median of 2 instead of 3 | Small disclosed shift |
| `WF_FREQ_MONTHS = {"1d":24,"7d":24}` | `_forecast_template.py` hook | 18→9 folds (−50%); all crisis dates preserved | Small disclosed shift |

**Why `WF_SPLIT_LIMIT` was the wrong fold-reduction tool:**
`WF_SPLITS[-N:]` keeps only the *last* N folds → drops GFC-2008, which breaks the thesis's
stated crisis-OOS design. `WF_FREQ_MONTHS` widens the test window instead → the same date
range is covered but with fewer, wider folds (every OOS date still gets exactly one prediction).

**Combined short track speedup:** `0.67 × 0.5 × ~0.85 ≈ 0.28` → ~9 hrs → **~2.5 hrs**

**Code changes syntax-verified:** both `02_model_shared.py` and `_forecast_template.py`
pass `python -m py_compile`.

---

## Final Report (LaTeX)

All work in `Desktop\DDP\Final Report\`, compiled on Overleaf.

| Round | What was done |
|---|---|
| Round 1 (18 items) | Margins, headings, LOF/LOT prefixes, TikZ figures (framework flowchart, feature bar chart, expected-pattern matrix, ensemble diagram), table restructure, Section 1.5 removed, figure captions shortened, matplotlib→TikZ for patching figure |
| Round 2 (13 items) | Equal 2 cm margins, `parskip`, caption centering, all data tables → `tabular*{\textwidth}` full-width + caption below, Fig 5.1 overflow fix, Fig 1.1 cross-reference added, Section 2.8 page-break |
| Round 3 (scope cut) | Cut from 4 horizons to **1d + 7d only**; filled result placeholders with real numbers from `Desktop\Track A and B.txt`; removed Long-Horizon section + all 4 appendix figure placeholders; all `\figplaceholder` eliminated |

**Real results in report (1d / 7d MAE, honest walk-forward):**
- Best at 7d: Informer 1.96, TimeMixer 2.14, TCN 2.14 — Informer beats Huang benchmark (~$2.25)
- At 1d: all models collapse to naive (~$1.22) — honest finding, reported as such
- Best global-fit (leaky): NLinear 0.60 / DLinear 1.18 — large protocol gap
- Ensemble (reproduction protocol only): SLSQP 0.52/1.09, Ridge 0.36/0.82

---

## Colab Notebook — Track B Only (use these cells)

The permanent fix + speed settings are baked in. Upload updated `DDP_v2` folder and
`DDP_results` folder to `MyDrive` before running.

### Cell 1 — mount + copy + chdir
```python
import subprocess, sys, shutil, os
from google.colab import drive
drive.mount('/content/drive')
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaleido==0.2.1"], check=False)
shutil.copytree('/content/drive/MyDrive/DDP_v2',      '/content/DDP_v2',      dirs_exist_ok=True)
shutil.copytree('/content/drive/MyDrive/DDP_results', '/content/DDP_results', dirs_exist_ok=True)
os.chdir('/content/DDP_v2')
print("df_features present:", os.path.exists('/content/DDP_results/df_features.pkl'))
```

### Cell 2 — permanent path fix
```python
import os, shutil, pathlib
REAL = "/content/DDP_results"; LINK = "/content/DDP_v2/outputs"
os.makedirs(REAL, exist_ok=True)
if os.path.islink(LINK):       os.unlink(LINK)
elif os.path.isdir(LINK):
    for it in os.listdir(LINK):
        s, d = os.path.join(LINK, it), os.path.join(REAL, it)
        if not os.path.exists(d): shutil.move(s, d)
    shutil.rmtree(LINK, ignore_errors=True)
elif os.path.exists(LINK):     os.remove(LINK)
os.symlink(REAL, LINK)
print("✅ symlink | df_features reachable:", os.path.exists(LINK + "/df_features.pkl"))

fp = pathlib.Path("/content/DDP_v2/_features_plus.py")
lines = fp.read_text(encoding="utf-8").splitlines()
for i, l in enumerate(lines[:40]):
    st = l.strip()
    if (st.startswith("OUT ") or st.startswith("OUT=")) and "df_features" not in st:
        lines[i] = 'OUT = os.environ.get("DDP_OUTPUT_DIR") or str(globals().get("OUTPUT_DIR", "outputs"))'
        print(f"✅ _features_plus.py L{i+1} rewritten"); break
fp.write_text("\n".join(lines), encoding="utf-8")

ft = pathlib.Path("/content/DDP_v2/_forecast_template.py")
src = ft.read_text(encoding="utf-8")
if "WF_FREQ_MONTHS" not in src:
    anchor = 'WF_SPLITS = get_wf_splits(df, HORIZON_KEY)'
    hook = ('_wf_freq = globals().get("WF_FREQ_MONTHS", None)\n'
            'if _wf_freq:\n'
            '    for _hk, _fm in _wf_freq.items():\n'
            '        if _hk in WF_CONFIG:\n'
            '            WF_CONFIG[_hk] = {**WF_CONFIG[_hk], "freq_months": int(_fm)}\n'
            '    print(f"  ⏩ WF_FREQ_MONTHS override applied: {_wf_freq}")\n')
    ft.write_text(src.replace(anchor, hook + anchor, 1), encoding="utf-8")
    print("✅ injected WF_FREQ_MONTHS hook")
else:
    print("✅ WF_FREQ_MONTHS hook already present")
```

### Cell 3 — env + speed globals
```python
import os, torch
os.environ['DDP_OUTPUT_DIR'] = '/content/DDP_results'
os.environ['DDP_TRACK']      = 'short'   # 'long' for 90d+180d
exec(open('_env.py',      encoding='utf-8').read())
exec(open('_plotting.py', encoding='utf-8').read())
_pal = {"actual":"#1f3b73","pred":"#d1495b","interval":"#b9c6e0",
        "accent":"#2a9d8f","muted":"#8d99ae","grid":"#e6e9ef"}
try:               PLOT_PALETTE = {**_pal, **PLOT_PALETTE}
except NameError:  PLOT_PALETTE = _pal
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
N_SEEDS_OVERRIDE = 2
WF_FREQ_MONTHS   = {"1d": 24, "7d": 24}   # long: {"90d":36,"180d":36}
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "| seeds:", N_SEEDS_OVERRIDE, "| WF_FREQ_MONTHS:", WF_FREQ_MONTHS)
```

### Cell 4 — run Track B only
```python
import pathlib, shutil
shutil.rmtree('/content/DDP_results/checkpoints', ignore_errors=True)
pathlib.Path('/content/DDP_results/logs/run_manifest.jsonl').unlink(missing_ok=True)
STAGES = [
    "B_wf_patchtst.py","B_wf_itransformer.py","B_wf_autoformer.py","B_wf_informer.py",
    "B_wf_fedformer.py","B_wf_fedformerw.py","B_wf_freeformer.py","B_wf_tft.py",
    "B_wf_timemixer.py","B_wf_nhits.py","B_wf_tide.py","B_wf_tcn.py",
    "B_wf_tcnbilstm.py","B_wf_cnnbilstmattn.py","B_wf_segrnn.py",
    "B_wf_dlinear.py","B_wf_nlinear.py",
    "07_risk_evt.py","07b_garch_evt.py","13_probabilistic_forecast.py",
    "C_ensemble.py","C_compare.py","06_analysis_plots.py","17_audit_pipeline.py",
]
exec(open('_orchestrate.py', encoding='utf-8').read())
```

### Cell 5 — save to Drive (run as soon as session nears time limit)
```python
import shutil
shutil.copytree('/content/DDP_results', '/content/drive/MyDrive/DDP_results', dirs_exist_ok=True)
print("✅ results synced to Drive")
```

**For long track:** change Cell 3 to `DDP_TRACK='long'` and `WF_FREQ_MONTHS={"90d":36,"180d":36}`.  
**Resume after disconnect:** re-run Cells 1→4; `already_done()` skips any model already synced
to Drive via Cell 5.

---

## Key Design Decisions (permanent record)

| Decision | What was chosen | Why |
|---|---|---|
| Path persistence | `DDP_OUTPUT_DIR` env var → all outputs | Survives Colab/Kaggle session clears |
| Fold reduction method | `WF_FREQ_MONTHS` (widen test window) | Preserves GFC/OPEC/COVID in OOS; `WF_SPLIT_LIMIT` drops GFC |
| B_wf path fix | Symlink + hard-rewrite (not exact-string patch) | Exact-string matching silently no-op'd against stale Kaggle dataset |
| Track split | `run_short.py` (1d+7d) vs `run_long.py` (90d+180d) | Fits one Colab session each; results in disjoint folders |
| Report scope | 1d + 7d only (dropped 90d/180d) | Partial pipeline run; 90d/180d results pending Track B completion |
| `WTIDataset` workers | `num_workers=0` | All windows pre-materialized in RAM; extra workers add IPC overhead |
| Loss at 180d | Student-t NLL (df=5) | Fat-tailed returns; gives free σ output for prediction intervals |

---

## Files Modified (code changes)

| File | Change |
|---|---|
| `02_model_shared.py` | cudnn.benchmark + TF32 flags; `pin_memory=IS_CUDA`; `non_blocking=IS_CUDA` on both `.to(DEVICE)` calls |
| `_forecast_template.py` | `WF_FREQ_MONTHS` hook after `WF_CONFIG` dict |
| `_orchestrate.py` | Manifest skip-guard (reads `run_manifest.jsonl`, skips "done" stages) |
| `_env.py` | Created: `OUTPUT_DIR`, `already_done()`, `FORCE_RERUN` |
| `_plotting.py` | Created: Plotly theme, `save_fig/save_table/save_animation/add_crises` |
| `_paper_helpers.py` | Torch-optional; disk-backed decomp cache |
| `_lh_runners.py` | Disk-backed decomp cache |
| `_run_wf_arch.py` | Skip-guard per model×horizon |
| `_run_paper_arch.py` | Skip-guard per model×horizon |
| `00_setup_data_features.py` | COT + term-structure features; matplotlib removed; imageio install |
| `01_black_swan_baselines.py` | matplotlib → Plotly |
| `07_risk_evt.py` | matplotlib → Plotly |
| `13_probabilistic_forecast.py` | matplotlib → Plotly |
| `09_vmd_hybrid.py` | `VMD_STRIDE` configurable; removed from main STAGES |
| `~25 other files` | `OUTPUT_DIR` redirect + `HORIZONS` global-override pattern |

---

*For the full implementation plan including all architectural decisions see the Claude plan file.*
