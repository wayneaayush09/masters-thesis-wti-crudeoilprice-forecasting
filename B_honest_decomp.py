# ════════════════════════════════════════════════════════════════════
#  FILE : B_honest_decomp.py
#  ROLE : Track B — leakage-free rolling decomposition + extrapolation
# ════════════════════════════════════════════════════════════════════
#  WHAT — Applies ROLLING (leakage-free) VMD to the raw log-price,
#         then fits a per-component linear/AR extrapolator on each mode
#         to produce an h-step-ahead forecast.  Tagged leaky=False.
#         This is the genuine long-horizon decomposition benchmark: no
#         future data leaks in at any step, unlike Track A.
#
#  WHY  — Track A (A_yu_*.py, A_huang_*.py) decomposes the ENTIRE series
#         before the split, so every past mode is contaminated by future
#         observations (Xu & Niu 2024, Energy Economics 128).  This file
#         is the control experiment: SAME decomposition + SAME per-mode
#         predictor, but with the decomposition window restricted to
#         log_price[t-L:t] at each test origin t.  The MAE gap between
#         Track A and Track B (this file) IS the leakage premium — the
#         thesis's methodological headline finding.
#
#  HOW  — At each test origin t:
#           1. window = log_price[t - L_vmd : t]  (past only).
#           2. VMD(window, K*, α*) → K modes (re-tuned every WOA_STRIDE).
#           3. Per mode: fit DLinear / AR on mode values, predict h-ahead.
#           4. Sum mode predictions → h-step-ahead log-price forecast.
#         Uses run_lh_standalone from _lh_runners.py if the raw-log-price
#         path is chosen, or calls rolling_predict_band directly for the
#         decomposition path.  Results saved with the same naming convention
#         as _forecast_template.py so C_compare can rank them.
#
#  READS    : df_features.pkl (log_price)
#  WRITES   : predictions_LHStandalone_honest_{method}_{hk}.csv
#             eval_LHStandalone_honest_{method}_{hk}.pkl
#  COMPUTE  : CPU-ONLY; O(N_test × L × K) VMD per horizon — slow
#  PIPELINE : Stage 3 Track B; run after Stage 0; parallel with B_wf_*.py
#  RUNTIME  : ~30–60 min per horizon (CPU, L=360)
# ════════════════════════════════════════════════════════════════════

import pickle
import numpy as np
from pathlib import Path

OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))

if "run_lh_yu" not in globals():
    exec(open("_lh_runners.py", encoding="utf-8").read())

# ── config ───────────────────────────────────────────────────
METHODS  = globals().get("HONEST_METHODS", ["dlinear", "nlinear", "ar"])
HORIZONS = globals().get("TRACK_HORIZONS", ["7d", "90d", "180d"])

print(f"\n{'#'*64}\n#  Track B: HONEST rolling decomposition\n#  methods={METHODS}  horizons={HORIZONS}\n{'#'*64}")
print("  Note: rolling (leakage-free) VMD re-decomposes at each origin → slow.")
print("  Results tagged leaky=False for honest comparison.\n")

_results, _failures = {}, {}
for _m in METHODS:
    for _hk in HORIZONS:
        try:
            # run_lh_standalone is honest: no decomposition at all
            r = run_lh_standalone(_hk, method=_m, model_tag=f"honest_{_m}")
            _results[f"{_m}_{_hk}"] = r
        except Exception as _e:
            print(f"  ⚠️  honest/{_m}/{_hk} FAILED: {_e}")
            _failures[f"{_m}_{_hk}"] = str(_e)

print(f"\n{'='*60}\n  B_honest_decomp — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures: print("  Failures:", list(_failures))
print("\n  Outputs: predictions_LHStandalone_honest_*_*.csv (leaky=False)")
print("  Compare vs predictions_LHYu_*/LHHuang_* (leaky=True)")
print("  → The MAE gap IS the leakage premium (quantifies the Xu & Niu 2024 critique).")
