# ════════════════════════════════════════════════════════════════════
#  FILE : 01_black_swan_baselines.py
#  ROLE : Stage 1 — crisis timeline + statistical forecast baselines
# ════════════════════════════════════════════════════════════════════
#  WHAT — (A) Plots annotated WTI price history with major black-swan
#         episodes shaded (GFC 2008, OPEC crash 2014, COVID 2020, etc.)
#         to provide narrative context for the forecast evaluation windows.
#         (B) Fits and evaluates four baseline models (Naive, Drift,
#         ARIMA, Holt-Winters ETS) on the same walk-forward splits used
#         by all Track B models — so leaderboard comparisons are apples-
#         to-apples. Saves metrics for 1d / 7d / 90d / 180d horizons.
#
#  WHY  — Baselines establish the floor: any model that cannot beat a
#         simple Naive carry-forward has no forecasting value. The crisis
#         plot is the first figure in the thesis results chapter — it sets
#         the stage for WHY oil forecasting is hard (sudden 50–80% drops
#         inside weeks) and which regimes the evaluation period covers.
#         Using the same WF splits as the NN models ensures baseline MAEs
#         in comparator_table.csv are not inflated by a favourable split.
#
#  HOW  — For each horizon hk and each WF split:
#           1. Compute Naive (last observed value), Drift (linear trend
#              extrapolated h steps), ARIMA(p,d,q) auto-fitted on window.
#           2. Evaluate MAE/RMSE/MAPE on the h-step-ahead actuals.
#           3. Aggregate into baselines_results.pkl (keyed by model × hk).
#         The crisis plot uses shaded vrect bands (add_crises from
#         _plotting.py if loaded, else falls back to matplotlib patches).
#
#  READS    : df_features.pkl, wti_raw.pkl  (Stage 0 outputs)
#  WRITES   : baselines_results.pkl, baseline_metrics.csv
#             figures/01_crisis_timeline.html|.png
#  COMPUTE  : CPU-ONLY; ARIMA auto-fitting ~2–5 min total
#  PIPELINE : Stage 1 — run after Stage 0; before all model stages
#  RUNTIME  : ~3–10 min (dominated by ARIMA auto-search)
# ════════════════════════════════════════════════════════════════════
import pickle, warnings, sys, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

CRISES = {
    "GFC":      ("2008-09-01","2009-06-30"),
    "OPEC_War": ("2014-06-01","2016-03-31"),
    "COVID":    ("2020-02-01","2020-09-30"),
}
NEGATIVE_PRICE_DATE = "2020-04-20"
CRISIS_COLORS = {"GFC":"#d62728","OPEC_War":"#ff7f0e","COVID":"#9467bd"}
PALETTE = {"actual":"#1f77b4","pred":"#e74c3c","interval":"#aec7e8"}

# ── LOAD PICKLES ─────────────────────────────────────────────
with open(OUTPUT_DIR/"df_features.pkl","rb")    as f: df            = pickle.load(f)
with open(OUTPUT_DIR/"wti_raw.pkl","rb")        as f: wti_raw       = pickle.load(f)
with open(OUTPUT_DIR/"extreme_events.pkl","rb") as f: extreme_events= pickle.load(f)
print(f"✅ Loaded: df {df.shape}, wti_raw {wti_raw.shape}, {len(extreme_events)} extreme events")

# ── BLACK SWAN OVERVIEW PLOT (Plotly) ─────────────────────────
# Full price history with every detected extreme event shaded; named crises
# (GFC / OPEC / COVID) get a distinct colour, unnamed events get grey bands.
# Guarded so a plotting hiccup logs a warning but never aborts the stage.
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo1
    _d1 = wti_raw.index.strftime("%Y-%m-%d")
    _fig1 = _pgo1.Figure()
    _fig1.add_trace(_pgo1.Scatter(x=_d1, y=wti_raw["price"], name="WTI Price",
                                  line=dict(color=PALETTE["actual"], width=0.9)))
    for _, ev in extreme_events.iterrows():
        _nm = ev.get("crisis_name")
        _col = CRISIS_COLORS.get(_nm, "#aaaaaa") if _nm else "#aaaaaa"
        _fig1.add_vrect(x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=_col, opacity=0.20, line_width=0)
    for _nm, (_s, _e) in CRISES.items():
        _fig1.add_vrect(x0=_s, x1=_e, fillcolor=CRISIS_COLORS[_nm], opacity=0.0,
                        line_width=0, annotation_text=_nm.replace("_", " "),
                        annotation_position="top left")
    _fig1.add_vline(x=NEGATIVE_PRICE_DATE, line_dash="dash", line_color="black", line_width=2)
    _fig1.add_trace(_pgo1.Scatter(x=[NEGATIVE_PRICE_DATE],
                                  y=[float(wti_raw.loc[NEGATIVE_PRICE_DATE, "price"])],
                                  mode="markers+text", name="Apr-20-2020  −$36.98",
                                  text=["−$36.98"], textposition="bottom right",
                                  marker=dict(color="black", size=9)))
    _fig1.update_layout(title="WTI Crude Oil — All Extreme Events Detected (named crises highlighted)",
                        yaxis_title="$/barrel", height=520)
    save_fig(_fig1, "02_black_swan_overview")
    print("✅ Figure saved: 02_black_swan_overview (html+png)")
except Exception as _e1:
    print(f"  [warn] black-swan overview figure skipped: {_e1}")

# ── BASELINES — WALK-FORWARD (same splits/dates as PatchTST) ──
# Earlier this file used a SINGLE fixed split (TRAIN_CUTOFF=2018). But PatchTST
# is walk-forward from 2007, so a fixed-split naive was scored on an easier,
# different test period → apples-to-oranges. We now retrain every baseline per
# walk-forward split and pool predictions over the SAME out-of-sample dates,
# making every MAE/RMSE directly comparable to PatchTST and the deep comparators.
print("\nTraining baseline models (walk-forward) …")

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA as smARIMA

# Walk-forward config — MUST match _forecast_template.py / 09_vmd_hybrid.py
WF_CONFIG = {
    "7d":  {"start_test": "2007-01-01", "freq_months": 12},
    "90d": {"start_test": "2007-01-01", "freq_months": 24},
    "180d":{"start_test": "2007-01-01", "freq_months": 24},
}
HORIZON_H = {"7d": 7, "90d": 90, "180d": 180}

def get_wf_splits(df, hk):
    cfg = WF_CONFIG[hk]
    starts = pd.date_range(cfg["start_test"], df.index[-1], freq=f"{cfg['freq_months']}MS")
    return [(df.index[0], ts, min(ts + pd.DateOffset(months=cfg["freq_months"]), df.index[-1]))
            for ts in starts]

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true.abs() > 1
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.any() else np.nan
    print(f"  {label:20s}  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")
    return {"label":label,"MAE":mae,"RMSE":rmse,"MAPE":mape}

def _score(pred_by_target, label, store):
    """pred_by_target: {target_date → predicted price}. Align to wti_raw actuals."""
    if not pred_by_target: return
    s = pd.Series(pred_by_target).sort_index()
    s.index = pd.to_datetime(s.index)
    actual = wti_raw["price"].reindex(s.index)
    ok = actual.notna() & np.isfinite(s.values)
    if ok.sum() > 0:
        store[label] = compute_metrics(actual[ok], s[ok], label.replace("_", " "))

with open(OUTPUT_DIR/"feat_cols.pkl","rb") as f: FEAT = pickle.load(f)
try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

baselines_results = {}
_arima_order = {}        # cache the auto-ARIMA order per horizon (fit once on first split)

def _dlinear_split(train_lp, test_lp, h, L=64, kernel=25):
    """Fit DLinear on a train log-price series, predict each test origin h-ahead."""
    tv = train_lp.values
    Xtr, ytr = [], []
    for i in range(L, len(tv)-h):
        Xtr.append(tv[i-L:i]); ytr.append(tv[i+h])
    if len(Xtr) < 100: return {}
    def _decomp(X):
        tr = np.apply_along_axis(lambda r: np.convolve(r, np.ones(kernel)/kernel, mode="same"), 1, X)
        return np.hstack([tr, X - tr])
    lr = LinearRegression().fit(_decomp(np.array(Xtr)), np.array(ytr))
    full = pd.concat([train_lp, test_lp])               # for building test windows w/ history
    fv, fidx = full.values, full.index
    out = {}
    for i in range(len(fv)):
        dt = fidx[i]
        if dt < test_lp.index[0] or i < L: continue     # only predict test origins
        win = fv[i-L:i]
        if len(win) < L or not np.all(np.isfinite(win)): continue
        pred_lp = float(lr.predict(_decomp(win.reshape(1, -1)))[0])
        out[dt + pd.tseries.offsets.BDay(h)] = float(np.exp(pred_lp))
    return out

for hk, h in HORIZON_H.items():
    print(f"\n  ── {hk} ({h}d) walk-forward baselines ──")
    splits   = get_wf_splits(df, hk)
    feat_cols = [c for c in FEAT[hk] if c in df.columns]
    tgt = f"target_{hk}"
    naive_p, dlin_p, gbm_p, garch_p, arima_p, ets_p = {}, {}, {}, {}, {}, {}

    for si, (tr_start, ts, te) in enumerate(splits):
        train_df = df[df.index <= ts]
        test_df  = df[(df.index > ts) & (df.index <= te)]
        if len(train_df) < 300 or len(test_df) < 5: continue
        tr_price = train_df["price"]

        # Naive persistence: forecast price h-ahead = current price
        for origin in test_df.index:
            naive_p[origin + pd.tseries.offsets.BDay(h)] = float(df.loc[origin, "price"])

        # DLinear (decomposition-linear; Zeng et al. 2023)
        try:    dlin_p.update(_dlinear_split(train_df["log_price"], test_df["log_price"], h))
        except Exception as e: print(f"    ⚠️ DLinear split{si}: {e}")

        # LightGBM on the SAME engineered features PatchTST uses
        if _HAS_LGB and tgt in df.columns:
            try:
                tr = train_df.dropna(subset=[tgt])
                if len(tr) >= 100:
                    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=42, verbose=-1)
                    gbm.fit(tr[feat_cols], tr[tgt])
                    pl = gbm.predict(test_df[feat_cols])
                    for origin, lr_hat in zip(test_df.index, pl):
                        gbm_p[origin + pd.tseries.offsets.BDay(h)] = float(df.loc[origin,"price"]*np.exp(lr_hat))
            except Exception as e: print(f"    ⚠️ LightGBM split{si}: {e}")

        # GARCH(1,1) drift-compounded point forecast
        if _HAS_ARCH:
            try:
                ret_tr = train_df["log_ret_1d"].dropna() * 100.0
                gres = arch_model(ret_tr, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
                mu_daily = gres.params.get("mu", 0.0) / 100.0
                for origin in test_df.index:
                    garch_p[origin + pd.tseries.offsets.BDay(h)] = float(df.loc[origin,"price"]*np.exp(mu_daily*h))
            except Exception as e: print(f"    ⚠️ GARCH split{si}: {e}")

        # ARIMA (7d only — expensive). Pick order once via auto_arima, reuse across splits.
        if hk == "7d":
            try:
                if hk not in _arima_order:
                    am = pm.auto_arima(tr_price.values, seasonal=False, stepwise=True,
                                       max_p=3, max_q=2, d=1, information_criterion="aic",
                                       error_action="ignore", suppress_warnings=True)
                    _arima_order[hk] = am.order
                    print(f"    Auto-ARIMA order ({hk}): {am.order}")
                order = _arima_order.get(hk, (2,1,2))
                fit = smARIMA(tr_price.values, order=order).fit()
                steps = len(test_df)
                fc = np.asarray(fit.forecast(steps))            # h-ahead path over test window
                for j, origin in enumerate(test_df.index):
                    k = min(j + h, steps - 1)
                    arima_p[origin + pd.tseries.offsets.BDay(h)] = float(fc[k])
            except Exception as e: print(f"    ⚠️ ARIMA split{si}: {e}")

        # ETS (7d, 90d) — damped additive trend
        if hk in ("7d", "90d"):
            try:
                em = ExponentialSmoothing(tr_price, trend="add", damped_trend=True).fit(optimized=True)
                steps = len(test_df)
                fc = np.asarray(em.forecast(steps))
                for j, origin in enumerate(test_df.index):
                    k = min(j + h, steps - 1)
                    ets_p[origin + pd.tseries.offsets.BDay(h)] = float(fc[k])
            except Exception as e: print(f"    ⚠️ ETS split{si}: {e}")

    _score(naive_p, f"Naive_{hk}",    baselines_results)
    _score(dlin_p,  f"DLinear_{hk}",  baselines_results)
    if gbm_p:   _score(gbm_p,   f"LightGBM_{hk}", baselines_results)
    if garch_p: _score(garch_p, f"GARCH_{hk}",    baselines_results)
    if arima_p: _score(arima_p, f"ARIMA_{hk}",    baselines_results)
    if ets_p:   _score(ets_p,   f"ETS_{hk}",      baselines_results)

print("\n✅ Baselines done (walk-forward, same OOS dates as PatchTST)")
baseline_df = pd.DataFrame(baselines_results).T
print(baseline_df.to_string())
baseline_df.to_csv(OUTPUT_DIR/"baseline_metrics.csv")

with open(OUTPUT_DIR/"baselines_results.pkl","wb") as f:
    pickle.dump(baselines_results, f)

print("\n✅ FILE 01 COMPLETE — baseline_metrics.csv + baselines_results.pkl saved")
