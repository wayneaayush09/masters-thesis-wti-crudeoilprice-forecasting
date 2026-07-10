# ════════════════════════════════════════════════════════════════════
#  FILE : _env.py
#  ROLE : Environment, persistence & resume backbone  (exec'd FIRST)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Defines WHERE every result is written and HOW the pipeline
#         resumes after a disconnect. Sets RESULTS_ROOT / OUTPUT_DIR,
#         creates the per-track folder tree, and exposes helpers:
#         already_done(), log_stage(), cache_load()/cache_save().
#
#  WHY  — On Colab/Kaggle a disconnect wipes the runtime's local disk.
#         The old code wrote to a relative "outputs/" folder, so a
#         dropped session lost hours of GPU work and forced a full
#         restart. Pointing OUTPUT_DIR at a STABLE location (Google
#         Drive, or a fixed local folder) via one env var makes results
#         survive disconnects; the skip-if-done + on-disk caches make a
#         reconnected session continue instead of recomputing.
#
#  HOW  — Reads two environment variables:
#           DDP_OUTPUT_DIR  → the persistent root  (default "outputs")
#           DDP_TRACK       → "short" | "long" | "all"  (default "all")
#         OUTPUT_DIR = <root>/<track>.  predictions_*.csv / eval_*.pkl
#         stay FLAT inside OUTPUT_DIR (unchanged naming, so every
#         existing glob keeps working); new artifacts go in the
#         figures/ tables/ cache/ checkpoints/ logs/ subfolders.
#
#  READS    : env vars DDP_OUTPUT_DIR, DDP_TRACK, DDP_FORCE_RERUN
#  WRITES   : creates OUTPUT_DIR tree; logs/run_manifest.jsonl
#  COMPUTE  : CPU-ONLY (trivial)
#  PIPELINE : exec'd first by every orchestrator + bootstrap cell
#  RUNTIME  : instant
#
#  STORAGE MODES (set DDP_OUTPUT_DIR before exec'ing this file):
#    • Colab : drive.mount('/content/drive');
#              os.environ['DDP_OUTPUT_DIR']='/content/drive/MyDrive/DDP_results'
#    • Kaggle: os.environ['DDP_OUTPUT_DIR']='/kaggle/working/DDP_results'
#              (zip & download at session end, or push to a Dataset)
#    • Local : os.environ['DDP_OUTPUT_DIR']=r'C:\\Users\\anand\\Desktop\\DDP\\results'
# ════════════════════════════════════════════════════════════════════

import os as _os
import json as _json
import time as _time
import pickle as _pickle
from pathlib import Path as _Path
import sys as _sys_env
import subprocess as _sub_env

# ── numpy ABI guard — must run before ANY sklearn/scipy/pandas import ──
# numpy 2.x changed the dtype struct size (88→96 bytes), breaking all
# C extensions compiled against numpy 1.x (sklearn, pandas, older scipy).
# Detect numpy 2.x here, downgrade it on disk, then stop so the user can
# restart the kernel and pick up the correct version from the start.
try:
    import numpy as _np_guard
    if int(_np_guard.__version__.split('.')[0]) >= 2:
        print(
            f"\n{'='*64}\n"
            f"  ENVIRONMENT FIX: numpy {_np_guard.__version__} detected.\n"
            f"  sklearn/pandas/scipy require numpy<2.0 (ABI incompatibility).\n"
            f"  Downgrading numpy now — this takes ~30 s ...\n"
            f"{'='*64}"
        )
        _sub_env.run(
            [_sys_env.executable, '-m', 'pip', 'install', '-q', 'numpy<2.0'],
            check=False
        )
        print(
            "\n  numpy downgraded successfully.\n"
            "  *** PLEASE RESTART THE KERNEL, then re-run the cell. ***\n"
            "  (Kernel > Restart kernel  /  Runtime > Restart runtime)\n"
        )
        raise SystemExit(
            "numpy downgrade done — kernel restart required before continuing"
        )
except SystemExit:
    raise
except ImportError:
    pass   # numpy not installed yet; 00_setup_data_features.py will install it

# ── Resolve the persistent root + track ──────────────────────────────
# A single env var (DDP_OUTPUT_DIR) decides the storage target. Results
# survive a runtime disconnect because the root is a STABLE location
# (Google Drive, or a fixed local folder).
#
# Layout rationale — data + predictions are SHARED in one root; only the
# human-facing deliverables (figures, tables) are split per track:
#   • OUTPUT_DIR = DATA_DIR = RESULTS_ROOT  → df_features.pkl, feat_cols,
#     wti_*.pkl, regimes, AND every predictions_*_{hk}.csv / eval_*_{hk}.pkl.
#     Because every result filename carries its horizon ({hk}), the short
#     (1d/7d) and long (90d/180d) tracks NEVER collide even in one folder.
#     Keeping them shared is what lets a GPU session reuse the locally-built
#     df_features + decomposition caches (preserves GPU hours; enables resume).
#   • FIGURES_DIR / TABLES_DIR = RESULTS_ROOT/figures|tables/<track>  → the
#     deliverables you actually read are cleanly separated per track.
RESULTS_ROOT = _Path(_os.environ.get("DDP_OUTPUT_DIR", "outputs"))
DDP_TRACK    = _os.environ.get("DDP_TRACK", "all")
OUTPUT_DIR   = RESULTS_ROOT                 # shared data + horizon-suffixed results
DATA_DIR     = RESULTS_ROOT                 # alias (shared inputs live here too)

# ── Folder tree ──────────────────────────────────────────────────────
# predictions_*.csv / eval_*.pkl stay FLAT in OUTPUT_DIR (existing save/
# glob code untouched). Figures/tables are per-track; caches/logs shared.
FIGURES_DIR     = RESULTS_ROOT / "figures" / DDP_TRACK   # per-track deliverables
TABLES_DIR      = RESULTS_ROOT / "tables" / DDP_TRACK
CACHE_DIR       = RESULTS_ROOT / "cache"                 # decomposition caches (SHARED)
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
LOGS_DIR        = RESULTS_ROOT / "logs"
for _d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR, CHECKPOINTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Track → horizon list ─────────────────────────────────────────────
# The run-list of horizons for this track. Named TRACK_HORIZONS (not
# HORIZONS) on purpose: 00_setup.py defines HORIZONS as a config *dict*,
# which would clobber a plain run-list in the shared namespace. Nothing
# reassigns TRACK_HORIZONS, so list-consumers can rely on it.
_TRACK_MAP = {"short": ["1d", "7d"],
              "long":  ["90d", "180d"],
              "all":   ["1d", "7d", "90d", "180d"],
              # Single-horizon tracks → one independent, resumable run per horizon.
              # Predictions stay flat + horizon-suffixed so all four share one
              # results root without collision; figures/tables land under figures/<hk>/.
              "1d":    ["1d"],   "7d":   ["7d"],
              "90d":   ["90d"],  "180d": ["180d"]}
TRACK_HORIZONS = _TRACK_MAP.get(DDP_TRACK, ["1d", "7d", "90d", "180d"])

# ── Resume controls ──────────────────────────────────────────────────
# FORCE_RERUN=1 recomputes everything even if outputs already exist.
FORCE_RERUN = _os.environ.get("DDP_FORCE_RERUN", "0") == "1"


def already_done(name, hk):
    """True if model `name` at horizon `hk` has already produced BOTH its
    prediction CSV and eval pickle (and we're not forcing a rerun).

    Used by the orchestrator loops to skip finished (model × horizon)
    pairs after a disconnect — the primary GPU-hour saver. Matches the
    FLAT naming used by save_paper_predictions / _forecast_template:
        predictions_{name}_{hk}.csv  and  eval_{name}_{hk}.pkl
    """
    if FORCE_RERUN:
        return False
    pred = OUTPUT_DIR / f"predictions_{name}_{hk}.csv"
    ev   = OUTPUT_DIR / f"eval_{name}_{hk}.pkl"
    return pred.exists() and ev.exists()


def log_stage(stage, status="done", **extra):
    """Append a one-line JSON record to logs/run_manifest.jsonl so a
    reconnected session can see what already completed. Never raises."""
    rec = {"stage": stage, "status": status,
           "track": DDP_TRACK, "ts": _time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.update(extra)
    try:
        with open(LOGS_DIR / "run_manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception as _e:                       # logging must never break a run
        print(f"  [warn] log_stage failed: {_e}")
    return rec


def cache_load(key):
    """Load a pickled object from cache/<key>.pkl, or None if absent.
    Used to persist the whole-series VMD/EMD decompositions so a fresh
    session reads them in <1s instead of recomputing (~7 min for Yu)."""
    p = CACHE_DIR / f"{key}.pkl"
    if FORCE_RERUN or not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return _pickle.load(f)
    except Exception as _e:
        print(f"  [warn] cache_load({key}) failed: {_e}")
        return None


def cache_save(key, obj):
    """Persist `obj` to cache/<key>.pkl. Never raises (cache is best-effort)."""
    try:
        with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
            _pickle.dump(obj, f)
    except Exception as _e:
        print(f"  [warn] cache_save({key}) failed: {_e}")


# ── Pre-load torch so DLL initialises ONCE in this process ──────────────────
# On Windows, importing torch inside exec() can trigger WinError 1114 (DLL init
# failed) even when torch works fine in a plain terminal, because the Jupyter
# kernel has a more restricted DLL search path. Pre-loading here puts the module
# into sys.modules so every subsequent `import torch` in exec()'d stage files
# finds it already cached and never re-touches the DLL loader.
import sys as _sys
if "torch" not in _sys.modules:
    # On Windows, torch triggers CUDA DLL initialisation (c10.dll) even on CPU-only
    # machines, causing WinError 1114 before any model code runs.  Setting
    # CUDA_VISIBLE_DEVICES="-1" tells the CUDA runtime there are no devices →
    # torch falls back to CPU cleanly.  setdefault leaves Kaggle/Colab GPU
    # assignments untouched (they already have the var set to "0" etc.).
    if _os.name == "nt":
        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import torch as _torch_preload
        _TORCH_AVAILABLE = True
        print(f"✅ torch {_torch_preload.__version__} pre-loaded (device DLL initialised once)")
    except Exception as _te:
        _TORCH_AVAILABLE = False
        print(f"  ⚠️  torch unavailable ({_te}); neural-network stages will be skipped")
else:
    _TORCH_AVAILABLE = True   # already loaded (e.g. notebook re-run)

print(f"✅ _env.py loaded  |  track='{DDP_TRACK}'  |  horizons={TRACK_HORIZONS}"
      f"  |  OUTPUT_DIR={OUTPUT_DIR}  |  FORCE_RERUN={FORCE_RERUN}")
