# ════════════════════════════════════════════════════════════════════
#  FILE : A_components_hybrid.py
#  ROLE : Track A — per-component auto-select best forecaster (hybrid)
# ════════════════════════════════════════════════════════════════════
#  WHAT — For EACH decomposed VMD/EMD band, grid-searches a set of
#         forecasters {dlinear, nlinear, ridge, ar, theta} on a held-out
#         validation slice of that band, then locks in the one with
#         the lowest val-MAE for the full test period. Yu + Huang, all
#         horizons. Optional Huang FE-drop: noisiest IMFs (high fuzzy
#         entropy) are predicted as their train-mean (zero forecast cost).
#         Outputs tagged leaky=True.
#
#  WHY  — "One method fits all bands" is a simplification. Smooth trend
#         bands (LF) are best extrapolated linearly; periodic cycle bands
#         (MF) may prefer AR; noisy HF bands degrade all methods equally.
#         Per-band model selection is standard practice in decomposition-
#         ensemble forecasting (e.g., EMD-ARIMA hybrids). This file is
#         the "oracle upper bound" of the linear/classical component suite:
#         if the hybrid doesn't beat the best single method significantly,
#         the band structure is simple enough that selection adds no value.
#
#  HOW  — Uses `select_best_component` from _lh_forecasters.py which:
#           1. Splits the band into train + val (first 70% / last 30%).
#           2. Runs each method on the val slice → computes val-MAE.
#           3. Returns the best method name + all scores (for logging).
#         Then calls the selected method for the full test period.
#         FE-drop: fuzzy_entropy computed on train portion per band;
#         bands above the fe_keep_quantile threshold are replaced with
#         the train-mean (no extrapolation wasted on unpredictable noise).
#
#  READS    : df_features.pkl, cache/yu|huang_decomp_*.pkl (if cached)
#  WRITES   : predictions_LHYu_select_{hk}.csv
#             predictions_LHHuang_select_{hk}.csv
#             eval_LHYu_select_{hk}.pkl, eval_LHHuang_select_{hk}.pkl
#  COMPUTE  : CPU-ONLY; selection overhead ~2× A_components_linear
#  PIPELINE : Stage 2 Track A; run after A_components_linear (reuses cache)
#  RUNTIME  : ~15–30 min per horizon (val-MAE grid search overhead)
# ════════════════════════════════════════════════════════════════════
if "run_lh_yu" not in globals():
    exec(open("_lh_runners.py", encoding="utf-8").read())

HORIZONS    = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
SEL_METHODS = globals().get("SEL_METHODS", ("dlinear", "nlinear", "ridge", "ar", "theta"))
FE_KEEP     = globals().get("FE_KEEP_QUANTILE", 0.8)   # Huang: model lowest-FE 80% of comps

_results, _failures = {}, {}
for _hk in HORIZONS:
    try:
        _results[f"Yu_{_hk}"] = run_lh_yu(_hk, method="select", model_tag="hybrid",
                                          sel_methods=SEL_METHODS, diagnose=(_hk == "180d"))
    except Exception as _e:
        print(f"  ⚠️  Yu/hybrid/{_hk} FAILED: {_e}"); _failures[f"Yu_{_hk}"] = str(_e)
    try:
        _results[f"Huang_{_hk}"] = run_lh_huang(_hk, method="select", model_tag="hybrid",
                                                sel_methods=SEL_METHODS,
                                                fe_keep_quantile=FE_KEEP, diagnose=(_hk == "180d"))
    except Exception as _e:
        print(f"  ⚠️  Huang/hybrid/{_hk} FAILED: {_e}"); _failures[f"Huang_{_hk}"] = str(_e)

print(f"\n{'='*60}\n  A_components_hybrid — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures: print("  Failures:", list(_failures))
