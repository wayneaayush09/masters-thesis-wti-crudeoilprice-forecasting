# ════════════════════════════════════════════════════════════════════
#  FILE : _run_paper_arch.py
#  ROLE : Track A orchestration template (LEAKY whole-series protocol)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Shared driver exec'd by every A_yu_*.py and A_huang_*.py file.
#         Loops over TRACK_HORIZONS, routing each (ARCH, PROTOCOL, hk)
#         to the correct engine: neural networks via _paper_helpers, or
#         classical extrapolators via _lh_runners. All results are tagged
#         leaky=True. Resume skip-guard prevents double-computation.
#
#  WHY  — Track A reproduces published paper benchmarks (Yu 2025 / Huang 2024)
#         which use whole-series decomposition BEFORE the train/test split.
#         This is lookahead leakage by design — the paper authors accept it
#         for apples-to-apples comparisons within their own framework. Here
#         it quantifies the "leakage premium": Track A MAE minus Track B MAE
#         at the same horizon is the price of that leakage assumption.
#
#  HOW  — Protocol routing:
#           yu:    _paper_helpers.run_paper_yu  → WOA-VMD-FE → 3 bands
#                  (LF/MF/HF) per-band NN or extrapolator
#           huang: _paper_helpers.run_paper_huang → VMD+EMD → ~15 comps
#         Extrapolator names (dlinear/nlinear/ridge/ar/theta/arima/ets):
#           _lh_runners.run_lh_yu / run_lh_huang → direct-map per component
#         Skip-guard: if predictions_*.csv + eval_*.pkl exist → skip.
#         Per-model Plotly figure generated after all horizons complete.
#
#  Optional globals (set by the driver before exec'ing this file):
#     ARCH, PROTOCOL, HORIZONS, EPOCHS, TRAIN_KWARGS, FE_KEEP_QUANTILE, DIAGNOSE
#
#  READS    : df_features.pkl (log_price), cache/yu|huang_decomp_*.pkl
#             predictions_*_{hk}.csv  (for skip-guard check)
#  WRITES   : predictions_{PaperYu|PaperHuang|LHYu|LHHuang}_{ARCH}_{hk}.csv
#             eval_*.pkl, figures/trackA_*_{hk}.html|.png
#  COMPUTE  : GPU-HEAVY for NN archs; CPU-ONLY for extrapolators
#  PIPELINE : exec'd by A_yu_*.py / A_huang_*.py; Stage 2 Track A
#  RUNTIME  : ~7 min WOA-VMD + 3×3-5 min per horizon (Yu/NN)
#             ~15 min VMD+EMD + 15×3-5 min per horizon (Huang/NN)
# ════════════════════════════════════════════════════════════════════

import os as _os
from pathlib import Path as _Path

# ── Guard: ensure OUTPUT_DIR is always a Path object, never a plain str.
# C_compare.py / other analysis files set OUTPUT_DIR="outputs" (str), which
# breaks save_paper_predictions's `OUTPUT_DIR / "predictions_*.csv"` (str/str → TypeError).
if "OUTPUT_DIR" not in globals() or not isinstance(globals().get("OUTPUT_DIR"), _Path):
    OUTPUT_DIR = _Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

# ── bootstrap shared libs (idempotent) ──────────────────────────────
if "run_paper_yu" not in globals():
    exec(open("_paper_helpers.py", encoding="utf-8").read())          # decomposition + NN per-band + save
if "run_lh_yu" not in globals():
    exec(open("_lh_runners.py", encoding="utf-8").read())             # per-component extrapolators (+ bootstraps forecasters)
if "DLinearNet" not in globals() and _os.path.exists("_lh_sota_models.py"):
    exec(open("_lh_sota_models.py", encoding="utf-8").read())         # DLinearNet/NLinearNet/TiDE/FITS/SegRNN -> MODEL_REGISTRY
if _os.path.exists("_models_plus.py") and "FreEformer" not in globals():
    exec(open("_models_plus.py", encoding="utf-8").read())            # FEDformerW/FreEformer + improved configs

# ── resolve driver inputs ───────────────────────────────────────────
ARCH      = globals().get("ARCH", "PatchTST")
PROTOCOL  = globals().get("PROTOCOL", "yu").lower()
HORIZONS  = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])

# ── Speed profile (DDP_SPEED) — Track A's training funnels through
# _paper_helpers.train_band_model, which never read 02_model_shared's
# MAX_EPOCHS/BATCH_SIZE.  Inject the profile HERE so balanced/aggressive cut
# epochs + raise batch size for every per-band NN, exactly like Track B.
# A driver that explicitly sets EPOCHS / TRAIN_KWARGS still wins (setdefault).
_speed = _os.environ.get("DDP_SPEED", "full").lower()
_SPEED_EPOCHS   = {"balanced": 50, "aggressive": 35}
_SPEED_PATIENCE = {"balanced": 8,  "aggressive": 6}
try:
    import torch as _torch_chk
    _is_cuda = bool(_torch_chk.cuda.is_available())
except Exception:
    _is_cuda = False

# EPOCHS: explicit driver value wins; else the profile default; else paper 80.
if "EPOCHS" in globals():
    EPOCHS = globals()["EPOCHS"]
else:
    EPOCHS = _SPEED_EPOCHS.get(_speed, 80)

TRAIN_KWARGS = dict(globals().get("TRAIN_KWARGS", {}))
if _speed in ("balanced", "aggressive"):
    if _is_cuda:                                   # 256 only helps on GPU
        TRAIN_KWARGS.setdefault("batch_size", 256)
    TRAIN_KWARGS.setdefault("max_patience", _SPEED_PATIENCE[_speed])
_use_amp = (_speed in ("balanced", "aggressive")) and _is_cuda   # AMP in train_band_model

FE_KEEP_QUANTILE = globals().get("FE_KEEP_QUANTILE", None)
DIAGNOSE  = globals().get("DIAGNOSE", False)

_EXTRAPOLATORS = {"dlinear", "nlinear", "ridge", "ar", "theta", "arima", "ets", "select"}
_is_extrap = ARCH.lower() in _EXTRAPOLATORS

print("\n" + "#" * 64)
print(f"#  Track A (leaky {PROTOCOL.upper()})  —  ARCH={ARCH}  "
      f"({'extrapolator' if _is_extrap else 'neural'})  horizons={HORIZONS}")
if not _is_extrap:
    print(f"#  speed='{_speed}'  EPOCHS={EPOCHS}  "
          f"batch={TRAIN_KWARGS.get('batch_size', 64)}  "
          f"patience={TRAIN_KWARGS.get('max_patience', 10)}  AMP={_use_amp}")
print("#" * 64)

_results, _failures = {}, {}
for _hk in HORIZONS:
    # Resume guard: skip a (model × horizon) already on disk (saves GPU hours
    # after a disconnect). Name mirrors save_paper_predictions / run_lh_*.
    _save_name = (f"LH{'Huang' if PROTOCOL == 'huang' else 'Yu'}_{ARCH.lower()}"
                  if _is_extrap else
                  f"Paper{'Huang' if PROTOCOL == 'huang' else 'Yu'}_{ARCH}")
    if globals().get("already_done") and already_done(_save_name, _hk):
        print(f"  ⏭️  skip {_save_name} {_hk} (already done; set DDP_FORCE_RERUN=1 to redo)")
        continue
    try:
        if _is_extrap:
            method = ARCH.lower()
            if PROTOCOL == "huang":
                r = run_lh_huang(_hk, method=method, diagnose=DIAGNOSE,
                                 fe_keep_quantile=FE_KEEP_QUANTILE)
            else:
                r = run_lh_yu(_hk, method=method, diagnose=DIAGNOSE)
        else:
            if PROTOCOL == "huang":
                r = run_paper_huang(hk=_hk, model_name=ARCH, epochs=EPOCHS,
                                    train_kwargs=TRAIN_KWARGS or None,
                                    fe_keep_quantile=FE_KEEP_QUANTILE, diagnose=DIAGNOSE)
            else:
                r = run_paper_yu(hk=_hk, model_name=ARCH, epochs=EPOCHS,
                                 train_kwargs=TRAIN_KWARGS or None, diagnose=DIAGNOSE)
        _results[_hk] = r
    except Exception as _e:
        print(f"  ⚠️  {PROTOCOL}/{ARCH}/{_hk} FAILED: {_e}")
        _failures[_hk] = str(_e)

print(f"\n{'='*60}\n  Track A {PROTOCOL}/{ARCH} — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures:
    print("  Failures:", _failures)

# ── Per-model Plotly figures (generated right after all horizons finish) ──
# Saves a pred-vs-actual HTML+PNG per horizon so analysts can inspect Track A
# results immediately, without waiting for the final 06_analysis stage.
try:
    if "save_fig" in globals():
        import plotly.graph_objects as _pgo_pa
        from plotly.subplots import make_subplots as _pmsp_pa
        _save_prefix = (f"LH{'Huang' if PROTOCOL == 'huang' else 'Yu'}_{ARCH.lower()}"
                        if _is_extrap else
                        f"Paper{'Huang' if PROTOCOL == 'huang' else 'Yu'}_{ARCH}")
        for _hk_fig in list(_results.keys()):
            _pred_csv = _Path(globals().get("OUTPUT_DIR", _Path("outputs"))) / f"predictions_{_save_prefix}_{_hk_fig}.csv"
            if not _pred_csv.exists():
                continue
            import pandas as _pd_pa
            _pr = _pd_pa.read_csv(_pred_csv, index_col=0, parse_dates=True).sort_index()
            if _pr.empty or "actual" not in _pr.columns or "pred" not in _pr.columns:
                continue
            _ya = _pr["actual"].astype(float); _yp = _pr["pred"].astype(float)
            _mae_v = _results[_hk_fig].get(f"Overall_{_hk_fig}", {}).get("MAE", float("nan"))
            _fig_pa = _pmsp_pa(rows=2, cols=1,
                               subplot_titles=[f"{_save_prefix} {_hk_fig}  MAE=${_mae_v:.2f}",
                                               "Residuals"],
                               row_heights=[0.65, 0.35], vertical_spacing=0.10)
            _fig_pa.add_trace(_pgo_pa.Scatter(x=_pr.index.strftime("%Y-%m-%d"),
                                              y=_ya, name="Actual",
                                              line=dict(color="#1f3b73", width=1.2)), row=1, col=1)
            _fig_pa.add_trace(_pgo_pa.Scatter(x=_pr.index.strftime("%Y-%m-%d"),
                                              y=_yp, name="Predicted",
                                              line=dict(color="#d1495b", width=1.2, dash="dash")), row=1, col=1)
            if "add_crises" in globals():
                add_crises(_fig_pa, row=1, col=1)
            _fig_pa.add_trace(_pgo_pa.Bar(x=_pr.index.strftime("%Y-%m-%d"),
                                          y=(_ya - _yp), name="Residual",
                                          marker_color="#8d99ae", opacity=0.7), row=2, col=1)
            _fig_pa.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
            _fig_pa.update_layout(height=600,
                                  title=f"Track A ({PROTOCOL.upper()}) — {_save_prefix} {_hk_fig}")
            save_fig(_fig_pa, f"trackA_{_save_prefix}_{_hk_fig}")
            print(f"  📊 trackA_{_save_prefix}_{_hk_fig} figure saved")
except Exception as _pa_fig_err:
    print(f"  [warn] Track A per-model figures skipped: {_pa_fig_err}")
