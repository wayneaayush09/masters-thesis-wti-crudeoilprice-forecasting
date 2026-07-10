# ════════════════════════════════════════════════════════════════════
#  FILE : A_components_linear.py
#  ROLE : Track A — per-component linear extrapolators (primary benchmark)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs DLinear, NLinear, and Ridge per decomposed band on the
#         whole-series VMD/EMD decomposition (leaky Track A protocol).
#         Both Yu (WOA-VMD-FE, 3 bands) and Huang (VMD-EMD, ~15 IMFs)
#         protocols are run, for all 4 horizons.  Outputs tagged leaky=True.
#
#  WHY  — Linear extrapolators (DLinear = direct linear map from lookback
#         to forecast; Ridge = regularised OLS) are the KEY insight of the
#         "3.45-beater": whole-series VMD decomposes the price into smooth,
#         trend-dominated bands that are ALREADY extrapolatable. Neural nets
#         add mean-reversion bias that hurts at 90d/180d. On these smooth,
#         future-informed modes a simple linear map often outperforms
#         PatchTST. The comparison DLinear vs PatchTST on the same leaky
#         decomposition isolates the "mode-predictability" effect from
#         the "architecture" effect — a thesis-chapter-worthy finding.
#
#  HOW  — Iterates over HORIZONS × {dlinear, nlinear, ridge} × {yu, huang}:
#           run_lh_yu(hk, method=m) or run_lh_huang(hk, method=m)
#         These functions (from _lh_runners.py) reuse the cached
#         whole-series decomposition and call forecast_component per band.
#         Results saved as predictions_LHYu_dlinear_{hk}.csv etc.
#         Resume skip-guard is inherited from run_lh_yu (checks disk).
#
#  READS    : df_features.pkl, cache/yu|huang_decomp_*.pkl (if cached)
#  WRITES   : predictions_LHYu_{dlinear|nlinear|ridge}_{hk}.csv
#             predictions_LHHuang_{dlinear|nlinear|ridge}_{hk}.csv
#             eval_LHYu_*_{hk}.pkl, eval_LHHuang_*_{hk}.pkl
#  COMPUTE  : CPU-ONLY; linear fit < 1 s per band per origin
#  PIPELINE : Stage 2 Track A; run after Stage 0 (+ cached decompositions)
#  RUNTIME  : ~5–15 min per horizon (dominated by prediction loop, not fit)
# ════════════════════════════════════════════════════════════════════
import os as _os
if "run_lh_yu" not in globals():
    exec(open("_lh_runners.py", encoding="utf-8").read())

HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
METHODS  = globals().get("LINEAR_METHODS", ["dlinear", "nlinear", "ridge"])

_results, _failures = {}, {}
for _m in METHODS:
    for _hk in HORIZONS:
        for _proto, _fn in (("Yu", run_lh_yu), ("Huang", run_lh_huang)):
            try:
                _diag = (_hk == "180d")
                _results[f"{_proto}_{_m}_{_hk}"] = _fn(_hk, method=_m, diagnose=_diag)
            except Exception as _e:
                print(f"  ⚠️  {_proto}/{_m}/{_hk} FAILED: {_e}")
                _failures[f"{_proto}_{_m}_{_hk}"] = str(_e)

print(f"\n{'='*60}\n  A_components_linear — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures: print("  Failures:", list(_failures))
