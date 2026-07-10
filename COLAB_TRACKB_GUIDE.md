# Fresh Phase 2 (Track B) on Google Colab — complete run guide

This is the one document to follow top-to-bottom. It regenerates the **honest walk-forward
Track B** results (your thesis numbers) that were lost on a Colab disconnect — now writing
**straight to Google Drive** so nothing is ever lost again, running **~6–7× faster**, and split
into **4 independent per-horizon sessions** (1d / 7d / 90d / 180d) you can run in parallel.

> **Resume guarantee:** every result is written to Drive the moment it's produced — after each
> *fold* and after each *model*. If the GPU/session drops, just **re-run the same cell**: it skips
> finished models, reloads the in-progress model from its last per-fold checkpoint, and continues.
> A disconnect costs at most the single fold in flight. You never restart a horizon from scratch.

---

## What changed in the code (already applied)

| File | Change | Effect |
|---|---|---|
| `02_model_shared.py` | `DDP_SPEED` profile + **vectorized Autoformer** auto-correlation | Balanced = 50 epochs / patience 8 / batch 256; Autoformer no longer crawls (removed per-batch `torch.roll` host-sync loop) |
| `_run_wf_arch.py` | Long-horizon epochs 120→60 (balanced); seed knob accepts `N_SEEDS` **or** `N_SEEDS_OVERRIDE` | Big 90d/180d win; fixes a bug where 2 seeds silently trained 3 |
| `_env.py` | `DDP_TRACK` accepts a single horizon (`1d`/`7d`/`90d`/`180d`) | One independent session per horizon |
| `_orchestrate.py` | Stage-skip keyed by `(stage, track)` | The 4 horizons safely share one results folder |
| `run_trackB.py` *(new)* | Per-horizon GPU runner — 17 `B_wf_*` only | This is what you run on Colab |
| `run_phase3.py` *(new)* | CPU consolidation — risk/ensemble/compare/plots/audit | Run locally for free at the end |

---

## Step 1 — Upload to Google Drive (one time)

Put two folders in **`MyDrive/`**:

1. **`MyDrive/DDP_v2`** — the code folder, **excluding** the two huge raw inputs (not needed for
   Phase 2; their signal is already inside `df_features.pkl`) and stray build folders:
   - skip `F_Disagg06_16.txt` (109 MB) and `f_year.txt` (10 MB)
   - skip `__pycache__/`, `outputs/`, `results/`
   - what's left is ~2 MB.
2. **`MyDrive/DDP_results`** — copy your local Phase-1 folder
   `C:\Users\anand\Desktop\DDP\results\` here (37.7 MB). It **must** contain `df_features.pkl`,
   `feat_cols.pkl`, `wti_raw.pkl`, and the `cache/` subfolder. This folder becomes `DDP_OUTPUT_DIR`
   and is where every fresh result lands.

---

## Step 2 — Phase 2 GPU runs (4 sessions, one per horizon)

In each Colab session: set the runtime to **GPU** (Runtime → Change runtime type → T4/L4),
then run **Cell 0** once, then the **Cell 1** for that horizon.

### Cell 0 — mount Drive + install deps (run once per session)
```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q kaleido==0.2.1 plotly
```

### Cell 1 — run one horizon
Change **only** the `DDP_TRACK` line for each of the four sessions.

```python
import os
os.chdir('/content/drive/MyDrive/DDP_v2')
os.environ['DDP_OUTPUT_DIR'] = '/content/drive/MyDrive/DDP_results'  # writes land on Drive instantly
os.environ['DDP_TRACK']      = '1d'        # <-- '1d' | '7d' | '90d' | '180d'  (one per session)
os.environ['DDP_SPEED']      = 'balanced'
N_SEEDS        = 2                          # 2 seeds (median); set 3 for the paper-exact number
WF_FREQ_MONTHS = {'1d': 24, '7d': 24, '90d': 36, '180d': 36}   # ~halves the fold count
exec(open('run_trackB.py', encoding='utf-8').read())
```

Run it **four times in four separate sessions** with `DDP_TRACK` =
`'1d'`, then `'7d'`, then `'90d'`, then `'180d'`. They can run in parallel (different tabs/accounts)
and all share `MyDrive/DDP_results` safely. 90d/180d take longer (longer lookback).

**While it runs you'll see**, per model:
- `💾 [ckpt] fold NN saved (…)` — that fold is now on Drive.
- `✅ <Model> <hk> COMPLETE — predictions_<Model>_<hk>.csv + eval_<Model>_<hk>.pkl saved`
- `📊 per-model figure: model_<Model>_<hk>` — a forecast-vs-actual + residual plot, also on Drive.

> If the very first run prints **"numpy downgraded — restart the kernel"**: do Runtime → Restart
> runtime, then re-run Cell 0 + Cell 1. (One-time environment fix.)

### If a session disconnects
Just re-run **Cell 1** for that same horizon. You'll see `⏭️ skip <Model> <hk> (already done)` for
finished models and `✅ [ckpt] fold NN loaded` for the one that was mid-training. Nothing repeats.

---

## Step 3 — (optional) fold in Track A so the comparison plots show everything

Your Track-A **neural** results (the leaky benchmark) live in your Kaggle **`ddp-results1`**
dataset, not locally. To include them in the final figures:

1. Kaggle → your `ddp-results1` dataset → **Download** → unzip.
2. Copy every `predictions_Paper*.csv` and `eval_Paper*.pkl` into `MyDrive/DDP_results`
   (next to the Track-B files).

The Track-A CPU extrapolators (`predictions_LHYu_*`, `predictions_LHHuang_*`) are already in your
results folder. Skip this step if you don't need the leaky neural bars — the plots still render
with Track B + the LH models.

---

## Step 4 — Phase 3 consolidation (LOCAL, CPU, free)

These stages need no GPU, so run them on your PC. First bring the fresh results down: let the
**Google Drive desktop app** sync `MyDrive/DDP_results`, or download the folder, so it lands back at
`C:\Users\anand\Desktop\DDP\results`.

```python
import os
os.environ['DDP_OUTPUT_DIR'] = r'C:\Users\anand\Desktop\DDP\results'   # the synced folder
os.chdir(r'C:\Users\anand\Desktop\DDP\DDP code updated\DDP_v2')
exec(open('run_phase3.py', encoding='utf-8').read())
```

This builds every cross-model / cross-horizon deliverable into `figures/all/` + `tables/all/`:
master comparator, skill-score vs horizon, DM-test heatmap, model-ranking race animation, ensemble
blend, risk/EVT, and the probabilistic fan charts — over **all** predictions present (Track B fresh
+ Track A).

> Prefer not to sync? Run the **same** cell inside Colab instead, with
> `os.environ['DDP_OUTPUT_DIR'] = '/content/drive/MyDrive/DDP_results'` and
> `os.chdir('/content/drive/MyDrive/DDP_v2')`. It needs no GPU.

---

## Quick checks (worth doing once)

- **Saving works:** a minute into the `1d` session, open `MyDrive/DDP_results` — you should already
  see `predictions_PatchTST_1d.csv`, `eval_PatchTST_1d.pkl`, and
  `checkpoints/PatchTST_1d_fold00.pkl` *while it's still running*.
- **Per-horizon isolation:** after `1d` and `7d` both run, every model has **both**
  `predictions_<Model>_1d.csv` and `predictions_<Model>_7d.csv`.
- **Plots include everything:** after Step 4, open `figures/all/16_master_comparator.html`.

## Notes / tradeoffs

- **Balanced** trades a small, disclosable accuracy shift (fewer seeds/folds, 50-epoch ceiling,
  batch 256) for the big speed-up. State the training budget in the thesis — standard practice.
- Need a specific headline model at the **paper-exact** setting? Re-run just that one in a cell with
  `os.environ['DDP_SPEED']='full'`, `N_SEEDS = 3`, and **omit** `WF_FREQ_MONTHS`, then
  `exec(open('B_wf_<model>.py').read())` (after the Cell 1 env lines). `already_done()` will let it
  overwrite only if you also set `os.environ['DDP_FORCE_RERUN']='1'` for that model.
- Per-fold writes to Drive add minor latency vs local disk; the payoff is zero lost work.
