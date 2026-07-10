# Track A (leaky paper reproductions) on Kaggle — complete run guide

This is the one document to follow top-to-bottom for **Track A** — the *leaky* paper
reproductions (`PaperYu_*` + `PaperHuang_*`, **18 architectures × 2 protocols**). It now
gets the same treatment Track B already had: the **`DDP_SPEED` profile reaches the
per-band trainer** (AMP + batch 256 + 50-epoch early stop → ~2–3× faster, heavy
transformers like Autoformer benefit most), **per-band / per-component checkpoints** so
nothing is lost on a disconnect, and **one independent session per horizon**
(1d / 7d / 90d / 180d).

> **Resume guarantee:** every band/component is checkpointed to the results folder the
> moment it finishes, and the WOA-VMD / VMD-EMD decomposition is disk-cached. If the
> Kaggle session drops, **re-run the setup + run cells**: finished arch×horizon pairs are
> skipped, the in-progress arch resumes from its last band/component, and the
> decomposition reloads in <1 s. A disconnect costs at most the single band/component in
> flight — you never restart an architecture from scratch.

---

## What changed in the code (already applied)

| File | Change | Effect |
|---|---|---|
| `_run_paper_arch.py` | Reads `DDP_SPEED`; injects `EPOCHS`=50 + `batch_size`=256 + `max_patience`=8 (balanced) into the per-band trainer unless a driver set them | The speed profile finally reaches Track A (it never used `02_model_shared`'s knobs) |
| `_paper_helpers.py` | **AMP** in `train_band_model` (env-gated; `full`=fp32-exact) + **per-band/per-component checkpoints** in `run_paper_yu` / `run_paper_huang` | Heavy archs ~2–3× faster; no work lost mid-architecture |
| `run_trackA.py` *(new)* | Per-horizon runner — 36 `A_yu_*`/`A_huang_*` drivers, light→heavy | This is what you run on Kaggle |
| `_run_paper_arch.py` Autoformer | (already done earlier) `_AutoCorrelation` vectorized — host-sync `for b in range(B)` loop removed | Autoformer no longer crawls |

`full` (or CPU) keeps the exact paper numbers — AMP and the epoch cut are byte-for-byte
no-ops there.

---

## Step 1 — Kaggle datasets (one time)

You need **two** datasets attached to the notebook (Add Input):

1. **Code dataset** (your `DDP_v2new` / `ddpv2-code`) — the `DDP_v2` `.py` files. **Make
   sure these are included** (the per-band SOTA + FEDformer archs won't register without
   them, and Track A would silently fall back):
   - `_lh_sota_models.py`  (DLinearNet / NLinearNet / TiDE / FITS / SegRNN)
   - `_fedformer_plus.py`  (FEDformerW / FreEformer)
   - plus the edited `_run_paper_arch.py`, `_paper_helpers.py`, and the new `run_trackA.py`.
2. **Results dataset** (your `ddp-results`) — must contain `df_features.pkl`, `wti_raw.pkl`,
   `feat_cols.pkl`. If it already holds `cache/yu_decomp_*` / `cache/huang_decomp_*` from a
   prior run, the ~7 min + ~15 min decompositions are reused for free.

Set the notebook accelerator to **GPU T4** (Settings → Accelerator).

---

## Step 2 — Run one horizon per session

### Cell 1 — setup (auto-detects code + results regardless of dataset slug)
```python
!pip install -q kaleido==0.2.1 plotly vmdpy EMD-signal
import os, shutil, glob
from pathlib import Path
WORK = Path('/kaggle/working'); RES = WORK/'DDP_results'; RES.mkdir(exist_ok=True)
os.chdir(WORK)

# code: find the folder that holds the .py files, copy them all into /kaggle/working
code_root = Path(glob.glob('/kaggle/input/**/B_wf_patchtst.py', recursive=True)[0]).parent
for f in code_root.glob('*.py'): shutil.copy(f, WORK)

# results: find df_features.pkl, copy all pkl/csv + checkpoints + cache into DDP_results
res_root = Path(glob.glob('/kaggle/input/**/df_features.pkl', recursive=True)[0]).parent
for ext in ('*.pkl', '*.csv'):
    for f in res_root.glob(ext): shutil.copy(f, RES)
for sub in ('checkpoints', 'cache'):
    if (res_root/sub).is_dir():
        shutil.copytree(res_root/sub, RES/sub, dirs_exist_ok=True)

# sanity: the two dependency files must be present or 6 archs fall back to PatchTST
for dep in ('_lh_sota_models.py', '_fedformer_plus.py'):
    print(('OK ' if (WORK/dep).exists() else 'MISSING '), dep)
print("setup ok — df_features:", (RES/'df_features.pkl').exists())
```

### Cell 2 — run ONE horizon (change `DDP_TRACK`, one per session)
```python
import os
os.environ['DDP_OUTPUT_DIR'] = '/kaggle/working/DDP_results'
os.environ['DDP_TRACK']      = '1d'        # <-- '1d' | '7d' | '90d' | '180d'  (one per session)
os.environ['DDP_SPEED']      = 'balanced'  # AMP + batch 256 + 50-epoch early stop
os.environ.pop('DDP_FORCE_RERUN', None)    # keep resume on
exec(open('run_trackA.py', encoding='utf-8').read())
```

Each arch prints a header like
`speed='balanced'  EPOCHS=50  batch=256  patience=8  AMP=True`, then per-band lines
(`[Yu/Autoformer] training band LF …`), and a per-arch forecast-vs-actual figure when it
finishes. Predictions land as `predictions_PaperYu_<arch>_<hk>.csv` /
`predictions_PaperHuang_<arch>_<hk>.csv` + matching `eval_*.pkl`.

### Cell 3 — push results back to the dataset (run after EACH horizon)
So a session teardown can't lose anything, version the results dataset when the run ends:
```python
import json, subprocess
with open('/kaggle/working/DDP_results/dataset-metadata.json', 'w') as f:
    json.dump({"title": "ddp-results", "id": "aayushmanand/ddp-results",
               "licenses": [{"name": "CC0-1.0"}]}, f)
r = subprocess.run(
    ['kaggle', 'datasets', 'version', '-p', '/kaggle/working/DDP_results',
     '-m', 'Track A leaky (PaperYu+PaperHuang) — horizon run', '--dir-mode', 'zip'],
    capture_output=True, text=True)
print(r.stdout); print(r.stderr)
```

### If a session disconnects
Re-run **Cell 1 + Cell 2** for the same horizon (after re-attaching the just-pushed
results dataset version so the partial checkpoints come back). You'll see
`⏭️ skip PaperYu_<arch> <hk> (already done)` for finished archs and
`[Huang] resume: N/15 components already done — skipping them` for the one mid-flight.

### Run all four
Repeat Cells 1–3 with `DDP_TRACK` = `1d`, `7d`, `90d`, `180d` — separate sessions, can run
in parallel across Kaggle accounts/tabs. They share `DDP_results` safely (horizon-suffixed
filenames + track-aware manifest).

---

## Estimated time per horizon (Kaggle T4, balanced, all 36 drivers from scratch)

| Horizon | Total |
|---|---|
| 1d  | ~3 hr |
| 7d  | ~3–3.5 hr |
| 90d | ~5–5.5 hr |
| 180d| ~6–6.5 hr |

Per arch = Yu (3 bands) + Huang (~15 components) = ~18 NN trainings. Heavy archs
(autoformer, informer, fedformer, fedformerw, freeformer, tft) dominate and benefit most
from AMP. Long horizons cost more (lookback 30/60 → 180/360). Each total fits one Kaggle
GPU session (≤9–12 h). **`already_done` skips any `Paper*` arch already on disk**, so if
your `ddp-results` already holds the archs from the 25-06 run, real time is far less.

---

## Step 3 — fold Track A into the final plots (LOCAL, CPU, free)

After the horizons finish, download the latest `ddp-results` dataset version (now holding
Track A + Track B), point Phase 3 at it, and run the CPU consolidation:
```python
import os
os.environ['DDP_OUTPUT_DIR'] = r'C:\Users\anand\Desktop\DDP\results'   # the downloaded folder
os.environ['DDP_TRACK']      = 'all'
os.chdir(r'C:\Users\anand\Desktop\DDP\DDP code updated\DDP_v2')
exec(open('run_phase3.py', encoding='utf-8').read())
```
`C_compare.py` / `06_analysis_plots.py` glob `eval_Paper*`, so the `PaperYu_*` /
`PaperHuang_*` bars appear automatically next to Track B in
`figures/all/16_master_comparator.html`.

---

## Notes / tradeoffs

- **Balanced** shifts the *leaky* Track A numbers slightly (AMP fp16, 50-epoch ceiling,
  batch 256) — acceptable because Track A is the published-benchmark contrast, not the
  defensible thesis claim. For an exact paper number on one headline arch, re-run just it
  with `os.environ['DDP_SPEED']='full'` and `os.environ['DDP_FORCE_RERUN']='1'`, e.g.
  after the Cell 2 env lines: `ARCH='Autoformer'; PROTOCOL='yu'; exec(open('_run_paper_arch.py').read())`.
- Checkpoints/predictions live under `/kaggle/working/DDP_results`; **push the dataset
  version after every horizon** (Cell 3) — Kaggle wipes `/kaggle/working` when the session
  ends.
- `run_standalone` and the `F_fedformer_*` sweep call `train_band_model` directly; they
  inherit the AMP speed-up but not the epoch/batch defaults — out of scope, unchanged.
