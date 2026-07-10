# ════════════════════════════════════════════════════════════════════
#  FILE : A_components_classical.py
#  ROLE : Track A — per-component classical time-series extrapolators
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs AR, Theta, ARIMA, and ETS per decomposed VMD/EMD band
#         on the whole-series decomposition (leaky Track A protocol).
#         Yu + Huang, all horizons. Outputs tagged leaky=True.
#         Speed tiers: ar / theta are fast (numpy, < 1 min per horizon);
#         arima / ets are strided at stride=50 for tractability
#         (~15–20 min per horizon); both are skipped for 180d by default
#         (AR already outperforms ARIMA at long horizons empirically).
#
#  WHY  — Classical extrapolators (AR, ARIMA, Theta, ETS) form the
#         methodological ladder between the linear map (A_components_linear)
#         and the neural net (A_yu_patchtst). They help isolate WHICH
#         modelling assumption drives performance on each band:
#           LF (trend): AR / Theta / DLinear all extrapolate well.
#           MF (cycle): ARIMA captures periodicity; DLinear doesn't.
#           HF (noise): everything is near-random; AR order matters little.
#         Including classical methods ensures the thesis does not conflate
#         "neural net is better" with "more parameters are better" — on
#         leaky decomposed bands, classical AR often ties or beats PatchTST.
#
#  HOW  — Same routing as A_components_linear:
#           run_lh_yu(hk, method=m) / run_lh_huang(hk, method=m)
#         for m in {ar, theta, arima, ets}.  Skip-guard inherited from
#         _lh_runners (checks predictions_LHYu_ar_{hk}.csv exists).
#         RUN_SLOW_METHODS flag (default False) controls whether arima/ets
#         are included — set True to run the full suite.
#
#  Results from A_components_linear showed (for context):
#   LHHuang_ar 7d: MAE=1.388  (best classical, beats dlinear 1.444)
#   LHHuang_theta 7d: MAE=4.061  (worse than linear)
# So focus is on AR. ARIMA/ETS are included for completeness if time allows.
# Run from inside DDP_v2/.
# ============================================================
if "run_lh_yu" not in globals():
    exec(open("_lh_runners.py", encoding="utf-8").read())

HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])

# ── FAST methods (numpy/pure-Python, ~1 min each) ────────────
FAST_METHODS = globals().get("FAST_CLASSICAL_METHODS", ["ar", "theta"])

# ── SLOW methods (statsmodels ARIMA/ETS, strided) ────────────
# To enable: uncomment the line below (or set before exec).
# Huang decomp has 15 components → ~45–75 min per horizon even at stride=50.
# AR already outperforms ARIMA at all horizons, so this is rarely worth running.
# RUN_SLOW_METHODS = True   # ← uncomment to enable arima/ets
RUN_SLOW_METHODS = False     # always reset — never inherit stale True from session
SLOW_METHODS     = globals().get("SLOW_CLASSICAL_METHODS", ["arima", "ets"])
SLOW_HORIZONS    = [h for h in HORIZONS if h != "180d"]  # skip 180d — too slow
ARIMA_STRIDE     = globals().get("ARIMA_STRIDE", 50)     # refit every 50 days (~14 min)

_results, _failures = {}, {}

# ── Run fast methods (always) ─────────────────────────────────
print(f"\n{'─'*56}\n  FAST classical methods: {FAST_METHODS}\n{'─'*56}")
for _m in FAST_METHODS:
    for _hk in HORIZONS:
        for _proto, _fn in (("Yu", run_lh_yu), ("Huang", run_lh_huang)):
            try:
                _results[f"{_proto}_{_m}_{_hk}"] = _fn(_hk, method=_m,
                                                         diagnose=(_hk == "180d"))
            except Exception as _e:
                print(f"  ⚠️  {_proto}/{_m}/{_hk} FAILED: {_e}")
                _failures[f"{_proto}_{_m}_{_hk}"] = str(_e)

# ── Run slow methods (only if explicitly enabled) ─────────────
if RUN_SLOW_METHODS:
    print(f"\n{'─'*56}")
    print(f"  SLOW classical methods: {SLOW_METHODS}")
    print(f"  stride={ARIMA_STRIDE}, horizons={SLOW_HORIZONS}")
    print(f"  (estimated time: ~{len(SLOW_METHODS)*len(SLOW_HORIZONS)*15} min)")
    print(f"{'─'*56}")
    for _m in SLOW_METHODS:
        for _hk in SLOW_HORIZONS:
            for _proto, _fn in (("Yu", run_lh_yu), ("Huang", run_lh_huang)):
                try:
                    _results[f"{_proto}_{_m}_{_hk}"] = _fn(
                        _hk, method=_m, diagnose=False,
                        stride=ARIMA_STRIDE)   # passed as fc_kw to _statsmodels_component
                except Exception as _e:
                    print(f"  ⚠️  {_proto}/{_m}/{_hk} FAILED: {_e}")
                    _failures[f"{_proto}_{_m}_{_hk}"] = str(_e)
else:
    print(f"\n  ℹ️  Slow methods (arima, ets) SKIPPED — set RUN_SLOW_METHODS=True to run them.")
    print(f"  Based on results, AR already outperforms ARIMA at all horizons.")

print(f"\n{'='*60}\n  A_components_classical — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures: print("  Failures:", list(_failures))
