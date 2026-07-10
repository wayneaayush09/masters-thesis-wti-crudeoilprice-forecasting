# ============================================================
# run_pipeline.py  —  Sequential runner for DDP_v2
#
# Runs each stage in the SAME Python process so globals
# (loaded data, cached decompositions, model objects) persist
# across stages — no redundant reloading.
#
# Usage:
#   python run_pipeline.py                  # runs STAGES list below
#   python run_pipeline.py --skip-setup     # skip 00 if data already built
#
# To customise what runs, edit the STAGES list.
# ============================================================

import sys, time, traceback
from pathlib import Path

# ── Configuration: edit this list to control what runs ───────
STAGES = [
    "00_setup_data_features.py",   # build df_features.pkl  (run once; slow on first run)
    "A_components_linear.py",      # DLinear / NLinear / Ridge per band
    "A_components_classical.py",   # AR / Theta  (ARIMA skipped unless RUN_SLOW_METHODS=True)
    "A_components_hybrid.py",      # hybrid per-component method selection
    # ── add / comment-out stages below as needed ─────────────
    # "B_wf_patchtst.py",
    # "B_wf_itransformer.py",
    # "C_ensemble.py",
    # "C_compare.py",
]

# ── Optional CLI flag: --skip-setup ──────────────────────────
if "--skip-setup" in sys.argv:
    STAGES = [s for s in STAGES if not s.startswith("00_")]
    print("ℹ️  --skip-setup: skipping 00_setup_data_features.py")

# ── Runner ────────────────────────────────────────────────────
BASE = Path(__file__).parent
_g = {"__name__": "__main__"}          # shared globals across all stages

total = len(STAGES)
passed, failed = 0, 0
wall_start = time.time()

for i, stage in enumerate(STAGES, 1):
    path = BASE / stage
    print(f"\n{'='*62}")
    print(f"  [{i}/{total}]  {stage}")
    print(f"{'='*62}")
    if not path.exists():
        print(f"  ⚠️  File not found — skipping: {path}")
        continue
    t0 = time.time()
    try:
        code = path.read_text(encoding="utf-8")
        exec(compile(code, str(path), "exec"), _g)
        elapsed = time.time() - t0
        print(f"\n  ✅ {stage} done in {elapsed:.1f}s")
        passed += 1
    except Exception:
        elapsed = time.time() - t0
        print(f"\n  ❌ {stage} FAILED after {elapsed:.1f}s:")
        traceback.print_exc()
        failed += 1
        # continue to next stage rather than aborting the whole run

wall = time.time() - wall_start
print(f"\n{'='*62}")
print(f"  Pipeline complete — {passed} passed, {failed} failed  ({wall/60:.1f} min total)")
print(f"{'='*62}")
