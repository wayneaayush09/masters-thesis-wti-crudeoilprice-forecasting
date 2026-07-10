# ════════════════════════════════════════════════════════════════════
#  FILE : _run_wf_arch.py
#  ROLE : Track B orchestration template (honest walk-forward, multivariate)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Shared driver exec'd by every B_wf_*.py file. Loops over
#         TRACK_HORIZONS, layering improved features + crisis weighting +
#         post-hoc signal fusion on top of _forecast_template.py.
#         Includes skip-guard: (model × horizon) pairs already on disk
#         are skipped, saving GPU hours after a runtime disconnect.
#
#  WHY  — Centralising the Track B improvements here means the 17 thin
#         B_wf_*.py drivers each stay 3 lines. Any fix (e.g. adding a new
#         signal overlay) is made ONCE and benefits all 17 architectures.
#         The honest walk-forward guarantee is upheld by _forecast_template:
#         no future data leaks in at any step.
#
#  HOW  — Driver protocol:
#           1. Bootstrap shared helpers (MODEL_REGISTRY, _crisis_weighting,
#              _fusion) — idempotent: already-loaded modules are reused.
#           2. For each _hk in HORIZONS:
#              a. Skip-guard: check predictions_*.csv + eval_*.pkl exist.
#              b. Inject HORIZON_KEY + MODEL_NAME_OVERRIDE + N_SEEDS.
#              c. Apply improved per-horizon epoch budget from IMPROVED_CONFIG.
#              d. exec(_forecast_template.py) → runs WF training + saves.
#              e. Optional: fuse_and_save → post-hoc sentiment/EVT overlay.
#              f. Generate per-model Plotly figure (pred vs actual + residuals).
#
#  Optional globals set by the driver BEFORE exec'ing this file:
#     MODEL_NAME, HORIZONS, USE_IMPROVED_FEATURES, USE_CRISIS_WEIGHTS,
#     USE_FUSION, N_SEEDS, EPOCHS_OVERRIDE, IMPROVED_CONFIG
#
#  READS    : df_features.pkl, feat_cols.pkl, wti_raw.pkl
#             predictions_{MODEL}_{hk}.csv  (for skip-guard check)
#  WRITES   : predictions_{MODEL}_{hk}.csv, eval_{MODEL}_{hk}.pkl
#             figures/model_{MODEL}_{hk}.html|.png
#  COMPUTE  : GPU-HEAVY  (exec's _forecast_template which trains on GPU)
#  PIPELINE : exec'd by B_wf_*.py; Stage 3 Track B
#  RUNTIME  : Depends on MODEL and number of horizons in TRACK_HORIZONS
# ════════════════════════════════════════════════════════════════════

import os as _os

# ── bootstrap shared helpers ────────────────────────────────
if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())
if "FreEformer" not in globals() and _os.path.exists("_models_plus.py"):
    exec(open("_models_plus.py", encoding="utf-8").read())         # extends MODEL_REGISTRY with new archs
if "crisis_sample_weights" not in globals() and _os.path.exists("_crisis_weighting.py"):
    exec(open("_crisis_weighting.py", encoding="utf-8").read())
if "fuse_signals" not in globals() and _os.path.exists("_fusion.py"):
    exec(open("_fusion.py", encoding="utf-8").read())

# ── resolve driver inputs ────────────────────────────────────
MODEL_NAME  = globals().get("MODEL_NAME", "PatchTST")
HORIZONS    = globals().get("TRACK_HORIZONS", ["7d", "90d", "180d"])
USE_IMPROVED_FEATURES = globals().get("USE_IMPROVED_FEATURES", True)
USE_CRISIS_WEIGHTS    = globals().get("USE_CRISIS_WEIGHTS", True)
USE_FUSION            = globals().get("USE_FUSION", True)
# Seed count: prefer global N_SEEDS, but also honour N_SEEDS_OVERRIDE so the
# run-cell knob works under either name (the template reads N_SEEDS_OVERRIDE,
# and this driver re-sets it from N_SEEDS below — so without this fallback a
# cell that only set N_SEEDS_OVERRIDE silently trained the default 3 seeds).
N_SEEDS               = globals().get("N_SEEDS", globals().get("N_SEEDS_OVERRIDE", 3))
EPOCHS_OVERRIDE       = globals().get("EPOCHS_OVERRIDE", None)

print("\n" + "#" * 64)
print(f"#  Track B (honest walk-forward)  —  MODEL={MODEL_NAME}  horizons={HORIZONS}")
print(f"#  features+={USE_IMPROVED_FEATURES}  crisis_weights={USE_CRISIS_WEIGHTS}  fusion={USE_FUSION}")
print("#" * 64)

# ── step 1: ensure improved features are in df_features.pkl ──
if USE_IMPROVED_FEATURES and _os.path.exists("_features_plus.py"):
    print("\n[B] Loading improved features …")
    exec(open("_features_plus.py", encoding="utf-8").read())
    # _features_plus enriches df_features.pkl in-place (if outputs/ already exists)

# ── step 2: inject crisis weighting into the template ────────
# The base template picks up N_SEEDS_OVERRIDE and MODEL_NAME_OVERRIDE
# We also patch the HORIZON config to use improved per-horizon epochs/capacity
# Long-horizon epoch budget honours the DDP_SPEED profile (90d/180d are the most
# expensive folds — lookback 336/360). 'balanced' halves the old 120-epoch default.
_long_ep = {"balanced": 60, "aggressive": 45, "full": 120}.get(
    _os.environ.get("DDP_SPEED", "full").lower(), 120)
if "IMPROVED_CONFIG" in globals():
    _LONG_CFG = IMPROVED_CONFIG
else:
    _LONG_CFG = {"90d": {"epochs": _long_ep}, "180d": {"epochs": _long_ep}}

import pickle as _pickle
from pathlib import Path as _Path
_OUTPUT_DIR = globals().get("OUTPUT_DIR", _Path("outputs"))

_results, _failures = {}, {}
for _hk in HORIZONS:
    print(f"\n{'─'*56}\n  Train {MODEL_NAME} @ {_hk}  (horizon loop iteration)\n{'─'*56}")
    # Resume guard — skip (model × horizon) already on disk to save GPU hours after a
    # disconnect.  Mirrors _forecast_template.py's save naming: PatchTST uses no
    # model suffix (predictions_7d.csv), all others use _{MODEL_NAME} prefix.
    _model_suffix = "" if MODEL_NAME == "PatchTST" else f"_{MODEL_NAME}"
    _pred_path = _OUTPUT_DIR / f"predictions{_model_suffix}_{_hk}.csv"
    _eval_path = _OUTPUT_DIR / f"eval{_model_suffix}_{_hk}.pkl"
    if not globals().get("FORCE_RERUN", False) and _pred_path.exists() and _eval_path.exists():
        print(f"  ⏭️  skip {MODEL_NAME} {_hk} (already done; set DDP_FORCE_RERUN=1 to redo)")
        continue
    try:
        # Set template control globals
        globals()["HORIZON_KEY"]          = _hk
        globals()["MODEL_NAME_OVERRIDE"]  = MODEL_NAME
        globals()["N_SEEDS_OVERRIDE"]     = N_SEEDS
        if EPOCHS_OVERRIDE:
            globals()["MAX_EPOCHS_OVERRIDE"] = EPOCHS_OVERRIDE
        elif _hk in _LONG_CFG and "epochs" in _LONG_CFG[_hk]:
            globals()["MAX_EPOCHS_OVERRIDE"] = _LONG_CFG[_hk]["epochs"]

        # Crisis weights: inject into template via WF_CRISIS_WEIGHTS global
        if USE_CRISIS_WEIGHTS and "crisis_sample_weights" in globals():
            globals()["WF_CRISIS_WEIGHTS"] = True   # template checks this flag

        # Capture MODEL_NAME now: _forecast_template.py resets it to 'PatchTST' if the
        # model is absent from MODEL_REGISTRY — using _current_model for all post-exec
        # steps ensures correct file paths and labels even in that (now-fixed) edge case.
        _current_model = MODEL_NAME
        # Run the base walk-forward template
        exec(open("_forecast_template.py", encoding="utf-8").read())

        # step 3: post-hoc fusion on the saved predictions
        if USE_FUSION and "fuse_and_save" in globals():
            fuse_and_save(_current_model, _hk)

        # try to load eval — use _model_suffix (pre-exec) for correct PatchTST path
        eval_p = _OUTPUT_DIR / f"eval{_model_suffix}_{_hk}.pkl"
        if eval_p.exists():
            with open(eval_p, "rb") as _f:
                _ev = _pickle.load(_f)
            _ov = _ev.get(f"Overall_{_hk}", {})
            print(f"  ✅ {_current_model} {_hk}: MAE={_ov.get('MAE',float('nan')):.3f}  "
                  f"RMSE={_ov.get('RMSE',float('nan')):.3f}  MAPE={_ov.get('MAPE',float('nan')):.2f}%")
            _results[_hk] = _ev

        # ── Per-model Plotly figure (pred-vs-actual + residuals) ──────────
        # Generated immediately after this horizon completes so analysts can
        # inspect the result without waiting for the final 06_analysis stage.
        try:
            if "save_fig" in globals():
                _pred_csv = _OUTPUT_DIR / f"predictions{_model_suffix}_{_hk}.csv"
                if not _pred_csv.exists():
                    _pred_csv = _OUTPUT_DIR / f"predictions_{_hk}.csv"
                if _pred_csv.exists():
                    import plotly.graph_objects as _pgo_wf
                    from plotly.subplots import make_subplots as _pmsp_wf
                    _pr = __import__("pandas").read_csv(_pred_csv, index_col=0, parse_dates=True).sort_index()
                    if not _pr.empty and "actual" in _pr.columns and "pred" in _pr.columns:
                        _ya = _pr["actual"].astype(float)
                        _yp = _pr["pred"].astype(float)
                        _err = _ya - _yp
                        _mae_v = float(_ov.get("MAE", float("nan")))
                        _fig_pm = _pmsp_wf(rows=2, cols=1,
                                           subplot_titles=[f"{_current_model} {_hk} — Forecast vs Actual  (MAE=${_mae_v:.2f})",
                                                           "Residuals (actual − predicted)"],
                                           row_heights=[0.65, 0.35], vertical_spacing=0.10)
                        # Panel 1: pred vs actual
                        if "lower" in _pr.columns and "upper" in _pr.columns:
                            _dl = _pr.index.strftime("%Y-%m-%d").tolist()
                            _fig_pm.add_trace(_pgo_wf.Scatter(
                                x=_dl+_dl[::-1],
                                y=list(_pr["upper"].astype(float))+list(_pr["lower"].astype(float)[::-1]),
                                fill="toself", fillcolor="#b9c6e0", opacity=0.4,
                                line_width=0, name="10–90% CI"), row=1, col=1)
                        _fig_pm.add_trace(_pgo_wf.Scatter(x=_pr.index.strftime("%Y-%m-%d"),
                                                           y=_ya, name="Actual",
                                                           line=dict(color="#1f3b73", width=1.2)), row=1, col=1)
                        _fig_pm.add_trace(_pgo_wf.Scatter(x=_pr.index.strftime("%Y-%m-%d"),
                                                           y=_yp, name="Predicted",
                                                           line=dict(color="#d1495b", width=1.2, dash="dash")), row=1, col=1)
                        if "add_crises" in globals():
                            add_crises(_fig_pm, row=1, col=1)
                        # Panel 2: residuals
                        _fig_pm.add_trace(_pgo_wf.Bar(x=_pr.index.strftime("%Y-%m-%d"),
                                                       y=_err, name="Residual",
                                                       marker_color="#8d99ae", opacity=0.7), row=2, col=1)
                        _fig_pm.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                        _fig_pm.update_layout(showlegend=True,
                                              height=600)
                        save_fig(_fig_pm, f"model_{_current_model}_{_hk}")
                        print(f"  📊 per-model figure: model_{_current_model}_{_hk}")
        except Exception as _pfig_err:
            print(f"  [warn] per-model plot skipped: {_pfig_err}")

    except Exception as _e:
        import traceback as _tb
        print(f"  ⚠️  {_current_model}/{_hk} FAILED: {_e}")
        _tb.print_exc()
        _failures[_hk] = str(_e)

    # clean up per-iteration globals
    for _g in ("HORIZON_KEY", "MODEL_NAME_OVERRIDE", "N_SEEDS_OVERRIDE",
               "MAX_EPOCHS_OVERRIDE", "WF_CRISIS_WEIGHTS"):
        globals().pop(_g, None)

print(f"\n{'='*60}\n  Track B {MODEL_NAME} — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures:
    print("  Failures:", _failures)
