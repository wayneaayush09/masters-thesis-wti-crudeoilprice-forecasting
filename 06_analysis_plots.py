# ════════════════════════════════════════════════════════════════════
#  FILE : 06_analysis_plots.py
#  ROLE : Master analysis & reporting — business-grade Plotly figures
# ════════════════════════════════════════════════════════════════════
#  WHAT — Generates the complete set of thesis figures and tables:
#         timeline, baseline comparison, crisis recovery, error
#         distribution, master model comparison, risk metrics, regime
#         analysis, DM-test heatmap, animated ranking race, animated
#         forecast unfolding, and all summary CSVs.
#
#  WHY  — This is the single deliverable stage: ALL cross-model,
#         cross-horizon results are synthesized here once every model
#         has saved its predictions_*.csv / eval_*.pkl. Plotly gives
#         interactive HTML (for the thesis defense) + high-DPI PNG
#         (for the printed document) with no legend/text overlap.
#
#  HOW  — Loads all prediction CSVs + eval PKLs from OUTPUT_DIR,
#         computes metrics (MAE/RMSE/MAPE/DirAcc/Skill/CRPS/DM),
#         builds cross-model comp_df, then renders figures through
#         _plotting.py's save_fig / save_table / save_animation.
#         Animated figures use go.Frame + play/pause slider.
#
#  READS    : predictions_{hk}.csv, eval_{hk}.pkl (PatchTST, all models)
#             baselines_results.pkl, regimes.pkl, risk_evt_*.pkl,
#             wti_raw.pkl, wti_wins.pkl, wins_floor.pkl
#  WRITES   : figures/*/  (HTML + PNG for every figure)
#             tables/*/   (CSV for every table)
#             comparator_table.csv, risk_metrics.csv, dm_test_results.csv
#  COMPUTE  : CPU-ONLY
#  PIPELINE : Stage 7 — run AFTER all Track-A/B drivers complete
#  RUNTIME  : ~2–5 min (dominated by kaleido PNG exports)
# ════════════════════════════════════════════════════════════════════

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import chi2, t as _tdist

warnings.filterwarnings("ignore")

# ── Environment & plotting bootstrap ─────────────────────────────────
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)

# Bootstrap _plotting.py if not already in globals
if "save_fig" not in globals():
    exec(open("_plotting.py", encoding="utf-8").read())

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HORIZONS    = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
CRISES = {
    "GFC":      ("2008-09-01", "2009-06-30"),
    "OPEC_War": ("2014-06-01", "2016-03-31"),
    "COVID":    ("2020-02-01", "2020-09-30"),
}
NEGATIVE_PRICE_DATE = "2020-04-20"
HZ_COLORS = {"1d": "#27ae60", "7d": "#e74c3c", "90d": "#9b59b6", "180d": "#2980b9"}
TARGETS   = {"1d": 1.5, "7d": 2.5, "90d": 7.0, "180d": 14.0}

# ────────────────────────────────────────────────────────────────────
# SECTION A — Statistical helper functions (pure computation, no plots)
# ────────────────────────────────────────────────────────────────────

def crps_gaussian(y, mu, sigma):
    """CRPS for Gaussian predictive distribution (closed-form, lower=better)."""
    try:
        from scipy.stats import norm as _spnorm
        sigma = np.maximum(np.asarray(sigma, float), 1e-9)
        z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
        ok = np.isfinite(z)
        if ok.sum() == 0: return np.nan
        return float(np.nanmean(sigma[ok] * (
            z[ok]*(2*_spnorm.cdf(z[ok])-1) + 2*_spnorm.pdf(z[ok]) - 1/np.sqrt(np.pi))))
    except Exception:
        return np.nan

def pinball_loss(y, q_pred, tau):
    err = np.asarray(y, float) - np.asarray(q_pred, float)
    ok = np.isfinite(err)
    if ok.sum() == 0: return np.nan
    return float(np.mean(np.where(err[ok] >= 0, tau*err[ok], (tau-1)*err[ok])))

def qlike_loss(realized_var, predicted_var):
    h = np.asarray(realized_var, float); s = np.asarray(predicted_var, float)
    ok = (s > 0) & (h > 0) & np.isfinite(h) & np.isfinite(s)
    if ok.sum() == 0: return np.nan
    return float(np.nanmean(h[ok]/s[ok] - np.log(h[ok]/s[ok]) - 1))

def pesaran_timmermann(actual_dir, pred_dir):
    from scipy.stats import norm as _nm
    a = np.sign(np.asarray(actual_dir, float))
    p = np.sign(np.asarray(pred_dir, float))
    ok = np.isfinite(a) & np.isfinite(p) & (p != 0) & (a != 0)
    if ok.sum() < 10: return np.nan, np.nan
    a, p = a[ok], p[ok]; n = len(a)
    da = float(np.mean(a == p))
    py = float(np.mean(p > 0)); qy = float(np.mean(a > 0))
    e_da = py*qy + (1-py)*(1-qy)
    var_da = e_da*(1-e_da)/n
    if var_da <= 0: return da, np.nan
    z = (da - e_da) / np.sqrt(var_da)
    return da, float(2*(1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(z))))

def dq_test(breach_seq, alpha=0.01, n_lags=4):
    try:
        from scipy.stats import chi2 as _c2
        b = np.asarray(breach_seq, float); n = len(b)
        if n < n_lags + 15: return np.nan
        Y = b[n_lags:]
        X = np.column_stack([b[n_lags-k:n-k] for k in range(1, n_lags+1)]
                            + [np.ones(n - n_lags)])
        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ coef; n_obs = len(Y); p_x = X.shape[1]
        s2 = float(resid @ resid) / max(n_obs - p_x, 1)
        if s2 <= 0: return np.nan
        XtX_inv = np.linalg.pinv(X.T @ X)
        cov = s2 * XtX_inv
        c = coef[:n_lags]
        Ci = np.linalg.pinv(cov[:n_lags, :n_lags])
        wald = float(c @ Ci @ c)
        return float(1 - _c2.cdf(wald, df=n_lags))
    except Exception:
        return np.nan

def kupiec_pof(n, x, p):
    if n == 0: return np.nan, np.nan, np.nan
    pi = x / n
    if x == 0:
        lr = -2*np.log((1-p)**n)
    else:
        lr = -2*(np.log(((1-p)**(n-x))*(p**x)) - np.log(((1-pi)**(n-x))*(pi**x)))
    return pi, float(lr), float(1 - chi2.cdf(lr, 1))

def christoffersen_cc(breach_seq, p):
    b = np.asarray(breach_seq).astype(int); n = len(b)
    if n < 2: return np.nan
    x = int(b.sum())
    n00=n01=n10=n11=0
    for t in range(1, n):
        prev, cur = b[t-1], b[t]
        if   prev==0 and cur==0: n00+=1
        elif prev==0 and cur==1: n01+=1
        elif prev==1 and cur==0: n10+=1
        else:                    n11+=1
    pi  = (n01+n11)/max(n00+n01+n10+n11, 1)
    pi0 = n01/max(n00+n01, 1); pi1 = n11/max(n10+n11, 1)
    def _safe(v): return v if v > 0 else 1e-12
    ll_ind = (n00*np.log(_safe(1-pi))+n01*np.log(_safe(pi))
              +n10*np.log(_safe(1-pi))+n11*np.log(_safe(pi)))
    ll_dep = (n00*np.log(_safe(1-pi0))+n01*np.log(_safe(pi0))
              +n10*np.log(_safe(1-pi1))+n11*np.log(_safe(pi1)))
    lr_ind = -2*(ll_ind - ll_dep)
    _, lr_uc, _ = kupiec_pof(n, x, p)
    lr_cc = (lr_uc if lr_uc == lr_uc else 0.0) + max(lr_ind, 0.0)
    return float(1 - chi2.cdf(lr_cc, 2))

def diebold_mariano(e1, e2, h=1, loss="absolute"):
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = np.abs(e1) - np.abs(e2) if loss == "absolute" else e1**2 - e2**2
    d = d[np.isfinite(d)]; n = len(d)
    if n < 12: return np.nan, np.nan
    dbar = d.mean()
    lags = max(1, h - 1)
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[k:], d[:-k])[0,1] for k in range(1, lags+1)]
    nw_var = gamma0 + 2*sum(gammas)
    if nw_var <= 0: return np.nan, np.nan
    dm_stat = dbar / np.sqrt(nw_var / n)
    p_val   = 2*(1 - _tdist.cdf(abs(dm_stat), df=n-1))
    return float(dm_stat), float(p_val)

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true.abs() > 1
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.sum()>0 else np.nan
    return {"label": label, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

def normal_mask(dates):
    m = np.ones(len(dates), dtype=bool)
    for s, e in CRISES.values():
        m &= ~((dates >= s) & (dates <= e))
    return m

WF_TEST_START = {h: "2007-01-01" for h in HORIZONS}

# ────────────────────────────────────────────────────────────────────
# SECTION B — Data loading
# ────────────────────────────────────────────────────────────────────
with open(OUTPUT_DIR/"wti_raw.pkl",  "rb") as f: wti_raw  = pickle.load(f)
with open(OUTPUT_DIR/"wti_wins.pkl", "rb") as f: wti_wins = pickle.load(f)
try:
    with open(OUTPUT_DIR/"wins_floor.pkl", "rb") as f: wins_floor = pickle.load(f)
except Exception:
    wins_floor = wti_wins["price"].min()

baselines_results = {}
bp = OUTPUT_DIR/"baselines_results.pkl"
if bp.exists():
    with open(bp, "rb") as f: baselines_results = pickle.load(f)

regime_df = None
rp = OUTPUT_DIR/"regimes.pkl"
if rp.exists():
    with open(rp, "rb") as f: regime_df = pickle.load(f)

evt_results = {}
for hk in HORIZONS:
    ep = OUTPUT_DIR/f"risk_evt_{hk}.pkl"
    if ep.exists():
        with open(ep, "rb") as f: evt_results[hk] = pickle.load(f)

# Primary (PatchTST walk-forward) results
all_results, eval_summary, insample_results = {}, {}, {}
for hk in HORIZONS:
    pred_csv = OUTPUT_DIR/f"predictions_{hk}.csv"
    eval_pkl = OUTPUT_DIR/f"eval_{hk}.pkl"
    if pred_csv.exists():
        res = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
        all_results[hk] = res
        print(f"  ✅ PatchTST {hk}: {len(res)} prediction dates")
    else:
        print(f"  ⚠️  {hk}: predictions_{hk}.csv not found")
    if eval_pkl.exists():
        with open(eval_pkl, "rb") as f: partial = pickle.load(f)
        eval_summary.update(partial)
    is_csv = OUTPUT_DIR/f"predictions_insample_{hk}.csv"
    if is_csv.exists():
        try:
            is_res = pd.read_csv(is_csv, index_col=0, parse_dates=True)
            if not is_res.empty:
                insample_results[hk] = is_res.sort_index()
        except Exception:
            pass

print(f"\nHorizons with results: {list(all_results.keys())}")

# Backfill missing eval entries
for hk, res_df in all_results.items():
    if res_df.empty: continue
    y_true = res_df["actual"].astype(float); y_pred = res_df["pred"].astype(float)
    if f"Overall_{hk}" not in eval_summary:
        eval_summary[f"Overall_{hk}"] = compute_metrics(y_true, y_pred, f"Overall {hk}")
    nm = normal_mask(res_df.index)
    if nm.sum() > 10 and f"Normal_{hk}" not in eval_summary:
        eval_summary[f"Normal_{hk}"] = compute_metrics(y_true[nm], y_pred[nm], f"Normal {hk}")
    for cname, (cs, ce) in CRISES.items():
        cm = (res_df.index >= cs) & (res_df.index <= ce)
        if cm.sum() > 5 and f"{cname}_{hk}" not in eval_summary:
            eval_summary[f"{cname}_{hk}"] = compute_metrics(y_true[cm], y_pred[cm], f"{cname} {hk}")
        post_s = pd.Timestamp(ce) + pd.DateOffset(days=1)
        post_e = post_s + pd.DateOffset(months=12)
        pm = (res_df.index >= post_s) & (res_df.index <= post_e)
        if pm.sum() > 5 and f"{cname}_Post_{hk}" not in eval_summary:
            eval_summary[f"{cname}_Post_{hk}"] = compute_metrics(
                y_true[pm], y_pred[pm], f"{cname}+12mo {hk}")

def origin_price(dt):
    if dt in wti_raw.index: return float(wti_raw.loc[dt, "price"])
    pos = min(max(wti_raw.index.searchsorted(dt), 0), len(wti_raw)-1)
    return float(wti_raw["price"].iloc[pos])

# ────────────────────────────────────────────────────────────────────
# SECTION C — Cross-model comparator table (discover all eval_*.pkl)
# ────────────────────────────────────────────────────────────────────

def _discover_paper_models():
    found = set()
    for p in OUTPUT_DIR.glob("eval_Paper*_*.pkl"):
        stem = p.stem.replace("eval_", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            found.add(parts[0])
    return sorted(found)

def _discover_lh_models():
    found = set()
    for p in OUTPUT_DIR.glob("eval_LH*_*.pkl"):
        stem = p.stem.replace("eval_", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("1d","7d","90d","180d"):
            found.add(parts[0])
    return sorted(found)

_PAPER_ROWS = _discover_paper_models()
_LH_ROWS    = _discover_lh_models()
for _legacy in ("PaperYu", "PaperHuang"):
    if any((OUTPUT_DIR/f"eval_{_legacy}_{hk}.pkl").exists() for hk in HORIZONS):
        if _legacy not in _PAPER_ROWS: _PAPER_ROWS.append(_legacy)
print(f"  Discovered Paper models: {_PAPER_ROWS}")
print(f"  Discovered LH models:    {_LH_ROWS}")

DEEP_MODELS = [
    "TCN","TCNBiLSTM","TimeMixer","NHiTS","Informer","Autoformer",
    "iTransformer","TimeLLM","MultiPatchTST","TFT","CNNBiLSTMAttn",
    "FEDformer","FEDformerW","FreEformer","SegRNN","DLinear_wf","NLinear_wf",
    "Chronos","TimesFM","RAGForecaster",
] + _PAPER_ROWS + _LH_ROWS

deep_eval = {}
for m in DEEP_MODELS + ["VMD"]:
    for hk in HORIZONS:
        p = OUTPUT_DIR/f"eval_{m}_{hk}.pkl"
        if p.exists():
            with open(p, "rb") as f: deep_eval[(m, hk)] = pickle.load(f)

def metric_for(model, hk, metric):
    if model == "PatchTST":
        return eval_summary.get(f"Overall_{hk}", {}).get(metric, np.nan)
    if model in DEEP_MODELS or model == "VMD":
        return deep_eval.get((model, hk), {}).get(f"Overall_{hk}", {}).get(metric, np.nan)
    return baselines_results.get(f"{model}_{hk}", {}).get(metric, np.nan)

def _model_series(model, hk):
    if model == "PatchTST":
        return all_results.get(hk)
    suffix = f"_{model}" if model != "PatchTST" else ""
    cp = OUTPUT_DIR/f"predictions{suffix}_{hk}.csv"
    if not cp.exists():
        cp = OUTPUT_DIR/f"predictions_{model}_{hk}.csv"
    if cp.exists():
        try: return pd.read_csv(cp, index_col=0, parse_dates=True)
        except Exception: pass
    return None

def _dir_acc(res):
    if res is None or res.empty: return np.nan
    res = res.sort_index()
    actual = res["actual"].astype(float); pred = res["pred"].astype(float)
    if "pred_logret" in res.columns:
        pd_ = np.sign(res["pred_logret"].astype(float).values)
    else:
        pd_ = np.sign(pred.values - np.array([origin_price(d) for d in res.index]))
    ad = np.sign(actual.values - np.array([origin_price(d) for d in res.index]))
    v = pd_ != 0
    return float(np.mean(pd_[v] == ad[v]))*100 if v.sum() else np.nan

COMP_MODELS = [
    "Naive","ARIMA","ETS","DLinear","GARCH","LightGBM",
    "PatchTST","MultiPatchTST","TCN","TCNBiLSTM","TimeMixer","NHiTS",
    "Informer","Autoformer","iTransformer","TimeLLM","VMD",
    "TFT","CNNBiLSTMAttn","FEDformer","Chronos","TimesFM","RAGForecaster",
]
COMP_MODELS += [m for m in _PAPER_ROWS + _LH_ROWS if m not in COMP_MODELS]

comp_rows = []
for model in COMP_MODELS:
    row = {"Model": model}; has_any = False
    for hk in HORIZONS:
        mae = metric_for(model, hk, "MAE")
        if np.isfinite(mae if mae==mae else float('nan')): has_any = True
        row[f"MAE_{hk}"]    = round(mae, 2)  if mae == mae else np.nan
        rmse = metric_for(model, hk, "RMSE")
        row[f"RMSE_{hk}"]   = round(rmse, 2) if rmse == rmse else np.nan
        da = _dir_acc(_model_series(model, hk))
        row[f"DirAcc_{hk}"] = round(da, 1)   if da == da else np.nan
    if has_any:
        comp_rows.append(row)

if comp_rows:
    comp_df = pd.DataFrame(comp_rows)
    _naive_mae = {}
    for _hk in HORIZONS:
        _nm = metric_for("Naive", _hk, "MAE")
        _naive_mae[_hk] = _nm if _nm == _nm else np.nan
    for _hk in HORIZONS:
        comp_df[f"Skill_{_hk}"] = comp_df.apply(
            lambda r: round(1 - r[f"MAE_{_hk}"] / _naive_mae[_hk], 3)
            if (np.isfinite(_naive_mae.get(_hk, np.nan)) and
                r.get(f"MAE_{_hk}") == r.get(f"MAE_{_hk}"))
            else np.nan, axis=1)
    comp_df.to_csv(OUTPUT_DIR/"comparator_table.csv", index=False)
    print(f"  ✅ comparator_table.csv — {len(comp_df)} models × {len(HORIZONS)} horizons")
else:
    comp_df = pd.DataFrame()
    print("  ⚠️  No comparator rows found — run model stages first")

# ────────────────────────────────────────────────────────────────────
# SECTION D — Risk metrics computation
# ────────────────────────────────────────────────────────────────────
risk_metric_rows = []
avail_r = [h for h in HORIZONS if h in all_results and not all_results[h].empty]
for hk in avail_r:
    res    = all_results[hk].sort_index()
    actual = res["actual"].astype(float)
    pred   = res["pred"].astype(float)
    err    = actual - pred
    if "pred_logret" in res.columns:
        pred_dir = np.sign(res["pred_logret"].astype(float).values)
    else:
        pred_dir = np.sign(pred.values - np.array([origin_price(d) for d in res.index]))
    actual_dir = np.sign(actual.values - np.array([origin_price(d) for d in res.index]))
    valid      = pred_dir != 0
    dir_acc    = float(np.mean(pred_dir[valid] == actual_dir[valid])) if valid.sum() else np.nan
    cover = float(np.mean((actual >= res["lower"].astype(float)) &
                          (actual <= res["upper"].astype(float)))) if "lower" in res.columns else np.nan
    if hk in evt_results and evt_results[hk].get("abs_error"):
        var99 = evt_results[hk]["abs_error"]["VaR_99"]
        es99  = evt_results[hk]["abs_error"]["ES_99"]
    else:
        var99 = float(np.percentile(np.abs(err), 99)); es99 = np.nan
    breach_seq = (np.abs(err) > var99).astype(int).values
    breaches   = int(breach_seq.sum())
    br_rate, lr, pval = kupiec_pof(len(err), breaches, 0.01)
    cc_p = christoffersen_cc(breach_seq, 0.01)
    pt_da, pt_p = pesaran_timmermann(actual_dir, pred_dir)
    dq_p = dq_test(breach_seq)
    risk_metric_rows.append({
        "Horizon": hk,
        "Dir. Acc (%)": round(dir_acc*100, 1) if dir_acc == dir_acc else np.nan,
        "CI Coverage":  round(cover, 3) if cover == cover else np.nan,
        "VaR99 ($)":    round(var99, 2),
        "ES99 ($)":     round(es99, 2) if es99 == es99 else np.nan,
        "Breach Rate":  round(br_rate, 4) if br_rate == br_rate else np.nan,
        "Kupiec p":     round(pval, 3)   if pval == pval else np.nan,
        "Christ. p":    round(cc_p, 3)   if cc_p == cc_p else np.nan,
        "PT_DA (%)":    round(pt_da*100, 1) if pt_da == pt_da else np.nan,
        "PT_p":         round(pt_p, 3)   if pt_p == pt_p else np.nan,
        "DQ_p":         round(dq_p, 3)   if dq_p == dq_p else np.nan,
        "Shrinkage α":  round(eval_summary.get(f"alpha_shrink_{hk}", np.nan), 3),
    })

if risk_metric_rows:
    risk_metrics_df = pd.DataFrame(risk_metric_rows)
    risk_metrics_df.to_csv(OUTPUT_DIR/"risk_metrics.csv", index=False)
    print("  ✅ risk_metrics.csv")

# ════════════════════════════════════════════════════════════════════
# FIGURE 1 — All-horizon forecast timeline (interactive + static PNG)
# ════════════════════════════════════════════════════════════════════
avail_hz = [h for h in HORIZONS if h in all_results]
if avail_hz:
    fig1 = make_subplots(
        rows=len(avail_hz), cols=1,
        shared_xaxes=True,
        subplot_titles=[f"PatchTST {hk} — MAE=${eval_summary.get(f'Overall_{hk}',{}).get('MAE',float('nan')):.2f} "
                        f"(target ≤${TARGETS.get(hk,'?')})" for hk in avail_hz],
        vertical_spacing=0.06)
    for i, hk in enumerate(avail_hz, 1):
        res = all_results[hk].sort_index()
        if res.empty: continue
        dates  = res.index.strftime("%Y-%m-%d")
        y_true = res["actual"].astype(float)
        y_pred = res["pred"].astype(float)
        c = HZ_COLORS.get(hk, "#333")
        if "lower" in res.columns and "upper" in res.columns:
            fig1.add_trace(go.Scatter(
                x=list(dates)+list(dates[::-1]),
                y=list(res["upper"].astype(float))+list(res["lower"].astype(float)[::-1]),
                fill="toself", fillcolor=c, opacity=0.15,
                line=dict(width=0), name=f"10–90% CI ({hk})", showlegend=(i==1)),
                row=i, col=1)
        fig1.add_trace(go.Scatter(x=dates, y=y_true, name="Actual", line=dict(color="#1f3b73", width=1.2),
                                  showlegend=(i==1)), row=i, col=1)
        fig1.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted", line=dict(color=c, width=1.2, dash="dash"),
                                  showlegend=(i==1)), row=i, col=1)
        add_crises(fig1, row=i, col=1)
        fig1.add_vline(x=NEGATIVE_PRICE_DATE, line_dash="dot", line_color="black",
                       line_width=1, row=i, col=1)
        fig1.update_yaxes(title_text="$/barrel", row=i, col=1)
    fig1.update_layout(
        title="WTI Crude Oil — PatchTST Out-of-Sample Forecast (all horizons)",
        height=350*len(avail_hz))
    save_fig(fig1, "01_forecast_timeline")
    print("✅ Figure 01_forecast_timeline saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 1B — Full history: in-sample fitted + out-of-sample forecast
# ════════════════════════════════════════════════════════════════════
fh_avail = [h for h in HORIZONS if h in all_results or h in insample_results]
if fh_avail:
    fig1b = make_subplots(rows=len(fh_avail), cols=1, shared_xaxes=True,
                          subplot_titles=[f"Full History {hk}" for hk in fh_avail],
                          vertical_spacing=0.06)
    raw_dates = wti_raw.index.strftime("%Y-%m-%d")
    for i, hk in enumerate(fh_avail, 1):
        c = HZ_COLORS.get(hk, "#333")
        fig1b.add_trace(go.Scatter(x=raw_dates, y=wti_raw["price"],
                                   name="Actual (full)", line=dict(color="#1f3b73", width=1.0),
                                   showlegend=(i==1)), row=i, col=1)
        if hk in insample_results:
            is_res = insample_results[hk]
            fig1b.add_trace(go.Scatter(x=is_res.index.strftime("%Y-%m-%d"),
                                       y=is_res["pred"].astype(float),
                                       name="Fitted (in-sample)", line=dict(color="#2a9d8f", width=1.0, dash="dot"),
                                       showlegend=(i==1)), row=i, col=1)
        if hk in all_results:
            res = all_results[hk].sort_index()
            fig1b.add_trace(go.Scatter(x=res.index.strftime("%Y-%m-%d"), y=res["pred"].astype(float),
                                       name=f"Forecast OOS ({hk})", line=dict(color=c, width=1.2, dash="dash"),
                                       showlegend=(i==1)), row=i, col=1)
        add_crises(fig1b, row=i, col=1)
        fig1b.update_yaxes(title_text="$/barrel", row=i, col=1)
    fig1b.update_layout(title="WTI Crude Oil — Full 1986→Present History + Forecast",
                        height=350*len(fh_avail))
    save_fig(fig1b, "11_full_history")
    print("✅ Figure 11_full_history saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 2 — Baseline comparison (MAE grouped bar chart)
# ════════════════════════════════════════════════════════════════════
base_models = ["Naive", "ARIMA", "ETS", "DLinear", "GARCH", "LightGBM", "PatchTST"]
base_data = {m: [metric_for(m, hk, "MAE") for hk in HORIZONS] for m in base_models}
_base_has = {m for m in base_models if any(np.isfinite(v) for v in base_data[m] if v==v)}
if _base_has:
    fig2 = go.Figure()
    _colors2 = ["#8d99ae","#e9a23b","#6a4c93","#118ab2","#ef476f","#06d6a0","#d1495b"]
    for mi, m in enumerate(base_models):
        if m not in _base_has: continue
        fig2.add_trace(go.Bar(
            name=m, x=[str(h) for h in HORIZONS], y=base_data[m],
            marker_color=_colors2[mi % len(_colors2)], opacity=0.85,
            text=[f"{v:.2f}" if v==v else "" for v in base_data[m]],
            textposition="outside"))
    for hk in HORIZONS:
        nv = metric_for("Naive", hk, "MAE")
        if np.isfinite(nv if nv==nv else float("nan")):
            fig2.add_hline(y=nv, line_dash="dash", line_color="green",
                           annotation_text=f"Naive {hk}: {nv:.2f}", annotation_position="right")
    fig2.update_layout(
        title="Baseline vs PatchTST — MAE by Horizon ($/barrel)",
        barmode="group", xaxis_title="Horizon", yaxis_title="MAE ($/barrel)")
    save_fig(fig2, "02_baseline_comparison")
    print("✅ Figure 02_baseline_comparison saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 3 — Crisis recovery rolling 63-day MAE
# ════════════════════════════════════════════════════════════════════
avail_r3 = [h for h in HORIZONS if h in all_results and not all_results[h].empty]
if avail_r3:
    fig3 = go.Figure()
    for hk in avail_r3:
        res = all_results[hk].sort_index()
        err = (res["actual"].astype(float) - res["pred"].astype(float)).abs()
        rolling_mae = err.rolling(63, min_periods=10).mean()
        c = HZ_COLORS.get(hk, "#333")
        fig3.add_trace(go.Scatter(
            x=res.index.strftime("%Y-%m-%d"), y=rolling_mae,
            name=f"{hk}", line=dict(color=c, width=1.5)))
    add_crises(fig3)
    fig3.add_vline(x=NEGATIVE_PRICE_DATE, line_dash="dot", line_color="black")
    fig3.update_layout(
        title="Rolling 63-Day MAE Through Crises + 12-Month Recovery",
        xaxis_title="Date", yaxis_title="Rolling MAE ($/barrel)")
    save_fig(fig3, "03_crisis_recovery")
    print("✅ Figure 03_crisis_recovery saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-horizon 3-panel detail (full / GFC zoom / COVID zoom)
# ════════════════════════════════════════════════════════════════════
for hk in avail_r3:
    res  = all_results[hk].sort_index()
    y_t  = res["actual"].astype(float)
    y_p  = res["pred"].astype(float)
    mae_v = eval_summary.get(f"Overall_{hk}", {}).get("MAE", np.nan)
    c = HZ_COLORS.get(hk, "#333")
    windows = [
        ("Full test period", res.index.min().strftime("%Y-%m-%d"), res.index.max().strftime("%Y-%m-%d")),
        ("GFC 2008-09",       "2008-01-01", "2010-06-30"),
        ("COVID 2020",        "2019-10-01", "2021-06-30"),
    ]
    fig4 = make_subplots(rows=1, cols=3, subplot_titles=[w[0] for w in windows])
    for col_i, (label, s, e) in enumerate(windows, 1):
        mask = (res.index >= s) & (res.index <= e)
        sub  = res[mask]; sy_t = y_t[mask]; sy_p = y_p[mask]
        if sub.empty: continue
        if "lower" in sub.columns:
            fig4.add_trace(go.Scatter(
                x=list(sub.index.strftime("%Y-%m-%d"))+list(sub.index.strftime("%Y-%m-%d")[::-1]),
                y=list(sub["upper"].astype(float))+list(sub["lower"].astype(float)[::-1]),
                fill="toself", fillcolor=c, opacity=0.15, line_width=0,
                name="CI", showlegend=(col_i==1)), row=1, col=col_i)
        fig4.add_trace(go.Scatter(x=sub.index.strftime("%Y-%m-%d"), y=sy_t,
                                  name="Actual", line=dict(color="#1f3b73", width=1.2),
                                  showlegend=(col_i==1)), row=1, col=col_i)
        fig4.add_trace(go.Scatter(x=sub.index.strftime("%Y-%m-%d"), y=sy_p,
                                  name="Predicted", line=dict(color=c, width=1.2, dash="dash"),
                                  showlegend=(col_i==1)), row=1, col=col_i)
        add_crises(fig4, row=1, col=col_i)
    fig4.update_layout(title=f"Per-Horizon Detail — {hk} (MAE=${mae_v:.2f})")
    save_fig(fig4, f"04_per_horizon_{hk}")
print("✅ Figures 04_per_horizon_* saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 5 — Signed prediction error over time
# ════════════════════════════════════════════════════════════════════
if avail_r3:
    fig5 = make_subplots(rows=len(avail_r3), cols=1, shared_xaxes=True,
                         subplot_titles=[f"Signed Error {hk}" for hk in avail_r3],
                         vertical_spacing=0.08)
    for i, hk in enumerate(avail_r3, 1):
        res = all_results[hk].sort_index()
        err = res["actual"].astype(float) - res["pred"].astype(float)
        c = HZ_COLORS.get(hk, "#333")
        fig5.add_trace(go.Bar(x=res.index.strftime("%Y-%m-%d"), y=err,
                              name=hk, marker_color=c, opacity=0.7,
                              showlegend=True), row=i, col=1)
        fig5.add_hline(y=0, line_dash="dash", line_color="black", row=i, col=1)
        add_crises(fig5, row=i, col=1)
        fig5.update_yaxes(title_text="Error ($/bbl)", row=i, col=1)
    fig5.update_layout(title="Signed Prediction Error Over Time  (actual − predicted)",
                       height=280*len(avail_r3))
    save_fig(fig5, "05_error_over_time")
    print("✅ Figure 05_error_over_time saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 6 — Error distribution (violin + histogram per horizon)
# ════════════════════════════════════════════════════════════════════
if avail_r3:
    fig6 = go.Figure()
    for hk in avail_r3:
        res = all_results[hk].sort_index()
        err = (res["actual"].astype(float) - res["pred"].astype(float)).dropna().values
        c = HZ_COLORS.get(hk, "#333")
        fig6.add_trace(go.Violin(y=err, name=hk, box_visible=True, meanline_visible=True,
                                 line_color=c, fillcolor=c, opacity=0.6, points="outliers"))
    fig6.add_hline(y=0, line_dash="dash", line_color="black")
    fig6.update_layout(title="Error Distribution per Horizon  (actual − predicted)",
                       yaxis_title="Error ($/barrel)", violinmode="group")
    save_fig(fig6, "06_error_distribution")
    print("✅ Figure 06_error_distribution saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 7 — Horizon comparison go.Table (replaces matplotlib table fig)
# ════════════════════════════════════════════════════════════════════
tbl_rows = []
for hk in HORIZONS:
    ov = eval_summary.get(f"Overall_{hk}", {})
    nm = eval_summary.get(f"Normal_{hk}", {})
    tbl_rows.append({
        "Horizon": hk, "N": ov.get("label","")[:3],
        "MAE": f"{ov.get('MAE',float('nan')):.3f}",
        "RMSE": f"{ov.get('RMSE',float('nan')):.3f}",
        "MAPE%": f"{ov.get('MAPE',float('nan')):.1f}",
        "MAE (Normal)": f"{nm.get('MAE',float('nan')):.3f}",
        "Target MAE": f"≤{TARGETS.get(hk,'?')}",
        "Meets target": "✓" if ov.get("MAE",float("nan")) <= TARGETS.get(hk,999) else "✗",
    })
if tbl_rows:
    tbl_df = pd.DataFrame(tbl_rows)
    save_table(tbl_df, "07_horizon_comparison_table",
               title="Horizon Performance Summary — PatchTST (honest walk-forward)")
    print("✅ Table 07_horizon_comparison_table saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 8 — Risk metrics: directional accuracy + CI calibration
# ════════════════════════════════════════════════════════════════════
if risk_metric_rows:
    fig8 = make_subplots(rows=1, cols=2,
                         subplot_titles=["Directional Accuracy (%)", "Predictive-Interval Coverage"])
    hzs = [r["Horizon"] for r in risk_metric_rows]
    da  = [r["Dir. Acc (%)"] for r in risk_metric_rows]
    cov = [r["CI Coverage"]  for r in risk_metric_rows]
    cols8 = [HZ_COLORS.get(h, "#333") for h in hzs]

    fig8.add_trace(go.Bar(x=hzs, y=da, name="Dir. Acc %",
                          marker_color=cols8, text=[f"{v:.1f}%" if v==v else "" for v in da],
                          textposition="outside"), row=1, col=1)
    fig8.add_hline(y=50, line_dash="dash", line_color="#d1495b",
                   annotation_text="Coin-flip (50%)", row=1, col=1)

    fig8.add_trace(go.Bar(x=hzs, y=cov, name="CI Coverage",
                          marker_color=cols8, text=[f"{v:.0%}" if v==v else "" for v in cov],
                          textposition="outside"), row=1, col=2)
    fig8.add_hline(y=0.80, line_dash="dash", line_color="#2a9d8f",
                   annotation_text="Target 80%", row=1, col=2)

    fig8.update_layout(title="Risk-Aware Metrics — Directional Skill & Interval Calibration",
                       showlegend=False)
    fig8.update_yaxes(title_text="%", row=1, col=1)
    fig8.update_yaxes(title_text="Coverage", range=[0, 1.1], row=1, col=2)
    save_fig(fig8, "09_risk_metrics")

    save_table(risk_metrics_df, "09_risk_metrics_table",
               title="PatchTST Risk-Aware Metrics (Kupiec/Christoffersen/PT/DQ tests)")
    print("✅ Figure 09_risk_metrics + table saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 9 — Regime-conditional MAE (HMM regimes heatmap)
# ════════════════════════════════════════════════════════════════════
if regime_df is not None:
    _reg_col = "regime" if "regime" in regime_df.columns else regime_df.columns[0]
    _n_reg   = int(regime_df[_reg_col].nunique())
    reg_rows_9 = []
    for hk in avail_r3:
        res = all_results[hk].sort_index()
        err = (res["actual"].astype(float) - res["pred"].astype(float)).abs()
        reg_aligned = regime_df[_reg_col].reindex(res.index, method="ffill")
        for reg in sorted(regime_df[_reg_col].unique()):
            m = reg_aligned == reg
            if m.sum() > 5:
                reg_rows_9.append({"Horizon": hk, "Regime": f"Reg {reg}",
                                   "MAE": round(float(err[m].mean()), 3)})
    if reg_rows_9:
        reg9_df = pd.DataFrame(reg_rows_9).pivot(index="Horizon", columns="Regime", values="MAE")
        fig9 = go.Figure(go.Heatmap(
            z=reg9_df.values.tolist(), x=list(reg9_df.columns), y=list(reg9_df.index),
            colorscale="Blues", text=[[f"{v:.2f}" for v in row] for row in reg9_df.values],
            texttemplate="%{text}", colorbar=dict(title="MAE ($/bbl)")))
        fig9.update_layout(title="Regime-Conditional MAE — HMM Market States",
                           xaxis_title="HMM Regime", yaxis_title="Horizon")
        save_fig(fig9, "10_regime_mae")
        save_table(reg9_df.reset_index(), "10_regime_mae_table",
                   title="MAE by HMM Regime × Horizon")
        print("✅ Figure 10_regime_mae saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 12 — Master model comparison (horizontal bar chart)
# ════════════════════════════════════════════════════════════════════
if not comp_df.empty:
    fig12 = make_subplots(rows=1, cols=len(HORIZONS),
                          subplot_titles=[f"{hk} MAE" for hk in HORIZONS])
    for col_i, hk in enumerate(HORIZONS, 1):
        col_name = f"MAE_{hk}"
        if col_name not in comp_df.columns: continue
        sub = comp_df[["Model", col_name]].dropna().sort_values(col_name)
        nv  = metric_for("Naive", hk, "MAE")
        _bar_colors = []
        for m in sub["Model"]:
            if m == "Naive":       _bar_colors.append("#2a9d8f")
            elif m == "PatchTST":  _bar_colors.append("#d1495b")
            elif "Paper" in str(m) or "LH" in str(m): _bar_colors.append("#e9a23b")
            else:                  _bar_colors.append("#8d99ae")
        fig12.add_trace(go.Bar(
            x=sub[col_name], y=sub["Model"],
            orientation="h", name=hk,
            marker_color=_bar_colors, opacity=0.85,
            text=[f"${v:.2f}" for v in sub[col_name]],
            textposition="outside", showlegend=False), row=1, col=col_i)
        if np.isfinite(nv if nv==nv else float("nan")):
            fig12.add_vline(x=nv, line_dash="dash", line_color="#2a9d8f",
                            annotation_text=f"Naive {nv:.1f}", row=1, col=col_i)
    fig12.update_layout(
        title="Master Model Comparison — Point MAE $/bbl on Identical Walk-Forward OOS Dates<br>"
              "<sub>Blue=Naive  Red=PatchTST  Orange=Decomp/LH  Grey=Deep NNs</sub>",
        height=max(600, 25*len(comp_rows)))
    save_fig(fig12, "12_model_comparison")
    print("✅ Figure 12_model_comparison saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 14 — Skill Score by Horizon (key thesis figure)
# ════════════════════════════════════════════════════════════════════
if not comp_df.empty:
    _plot_models14 = [m for m in ["PatchTST","MultiPatchTST","TCN","TimeMixer","NHiTS",
                                  "Informer","Autoformer","iTransformer","DLinear","VMD",
                                  "Chronos","TimesFM"]
                      if m in comp_df["Model"].values]
    if _plot_models14:
        fig14 = go.Figure()
        _palette14 = ["#d1495b","#e9a23b","#1f3b73","#2a9d8f","#6a4c93",
                      "#118ab2","#ef476f","#06d6a0","#8d99ae","#073b4c","#ffd166","#06d6a0"]
        for mi, m in enumerate(_plot_models14):
            row14 = comp_df[comp_df["Model"] == m].iloc[0]
            skills = [float(row14.get(f"Skill_{hk}", np.nan)) for hk in HORIZONS]
            lw = 2.5 if m in ("PatchTST","MultiPatchTST") else 1.5
            dash = "solid" if m in ("PatchTST","MultiPatchTST") else "dash"
            fig14.add_trace(go.Scatter(
                x=[str(h) for h in HORIZONS], y=skills, name=m, mode="lines+markers",
                line=dict(color=_palette14[mi%len(_palette14)], width=lw, dash=dash),
                marker=dict(size=8)))
        fig14.add_hline(y=0, line_dash="dot", line_color="black",
                        annotation_text="Naive baseline (Skill=0)")
        fig14.update_layout(
            title="Skill Score by Horizon  (1 − MAE_model / MAE_naive)<br>"
                  "<sub>Positive = beats naive | Flat slope = multi-horizon robustness</sub>",
            xaxis_title="Forecast Horizon", yaxis_title="Skill Score")
        save_fig(fig14, "14_skill_score_horizon")
        print("✅ Figure 14_skill_score_horizon saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 15 — DM-test significance heatmap (vs Naive baseline)
# ════════════════════════════════════════════════════════════════════
_dm_rows = []
if not comp_df.empty:
    naive_errs = {}
    for hk in HORIZONS:
        ns = _model_series("Naive", hk)
        if ns is not None and not ns.empty:
            naive_errs[hk] = (ns["actual"].astype(float) - ns["pred"].astype(float)).dropna()

    _dm_models = [m for m in comp_df["Model"].tolist()
                  if m != "Naive" and any(_model_series(m, hk) is not None for hk in HORIZONS)]
    _hk_ints = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
    for m in _dm_models[:20]:
        row_dm = {"Model": m}
        for hk in HORIZONS:
            if hk not in naive_errs:
                row_dm[hk] = np.nan; continue
            ms = _model_series(m, hk)
            if ms is None or ms.empty:
                row_dm[hk] = np.nan; continue
            me = (ms["actual"].astype(float) - ms["pred"].astype(float)).dropna()
            common = naive_errs[hk].index.intersection(me.index)
            if len(common) < 12:
                row_dm[hk] = np.nan; continue
            _, pv = diebold_mariano(naive_errs[hk].reindex(common).values,
                                    me.reindex(common).values, h=_hk_ints.get(hk, 1))
            row_dm[hk] = round(pv, 4) if pv == pv else np.nan
        _dm_rows.append(row_dm)

if _dm_rows:
    dm_df = pd.DataFrame(_dm_rows)
    dm_df.to_csv(OUTPUT_DIR/"dm_test_results.csv", index=False)
    dm_pivot = dm_df.set_index("Model")[[h for h in HORIZONS if h in dm_df.columns]]
    # Heatmap: p < 0.05 = significant improvement over naive (green)
    _z_vals = dm_pivot.values.tolist()
    _text_vals = [[f"p={v:.3f}" if v==v else "n/a" for v in row] for row in dm_pivot.values]
    _colors_dm = [[("#2a9d8f" if (v==v and v < 0.05) else
                    "#d1495b"  if (v==v and v > 0.50) else
                    "#e9a23b") for v in row] for row in dm_pivot.values]
    fig15 = go.Figure(go.Heatmap(
        z=dm_pivot.values.tolist(),
        x=[str(h) for h in HORIZONS if h in dm_pivot.columns],
        y=list(dm_pivot.index),
        colorscale=[[0,"#2a9d8f"],[0.05,"#f0f4ff"],[1,"#d1495b"]],
        text=_text_vals, texttemplate="%{text}",
        zmid=0.05, zmin=0, zmax=1,
        colorbar=dict(title="DM p-value")))
    fig15.update_layout(
        title="Diebold-Mariano Test p-values vs Naive  (green p<0.05 = sig. better than naive)",
        xaxis_title="Horizon", yaxis_title="Model",
        height=max(400, 30*len(_dm_rows)))
    save_fig(fig15, "15_dm_test_heatmap")
    save_table(dm_df, "15_dm_test_results", title="DM test p-values (vs Naive)")
    print("✅ Figure 15_dm_test_heatmap saved")

# ════════════════════════════════════════════════════════════════════
# FIGURE 16 — Cross-model × horizon MAE comparison table (go.Table)
# ════════════════════════════════════════════════════════════════════
if not comp_df.empty:
    _cols_show = ["Model"] + [f"MAE_{h}" for h in HORIZONS] + [f"Skill_{h}" for h in HORIZONS]
    _cols_show = [c for c in _cols_show if c in comp_df.columns]
    save_table(comp_df[_cols_show], "16_master_comparator",
               title="Master Model Comparator — MAE + Skill Score per Horizon (honest WF OOS)")
    print("✅ Table 16_master_comparator saved")

# ════════════════════════════════════════════════════════════════════
# ANIMATED FIGURE A — Forecast Unfolding (pred-vs-actual sweeping
#   through the test period; crisis behaviour plays out live)
# ════════════════════════════════════════════════════════════════════
for hk in avail_r3:
    res = all_results[hk].sort_index()
    if len(res) < 30: continue
    dates_str = res.index.strftime("%Y-%m-%d").tolist()
    y_true    = res["actual"].astype(float).tolist()
    y_pred    = res["pred"].astype(float).tolist()
    step      = max(1, len(dates_str) // 60)   # ~60 frames

    _frames = []
    for fi in range(step, len(dates_str)+1, step):
        _frames.append(go.Frame(
            data=[
                go.Scatter(x=dates_str[:fi], y=y_true[:fi], name="Actual",
                           line=dict(color="#1f3b73", width=1.5)),
                go.Scatter(x=dates_str[:fi], y=y_pred[:fi], name="Predicted",
                           line=dict(color=HZ_COLORS.get(hk,"#d1495b"), width=1.5, dash="dash")),
            ],
            name=str(fi)))

    mae_v = eval_summary.get(f"Overall_{hk}", {}).get("MAE", float("nan"))
    fig_anim = go.Figure(
        data=[
            go.Scatter(x=dates_str[:step], y=y_true[:step], name="Actual",
                       line=dict(color="#1f3b73", width=1.5)),
            go.Scatter(x=dates_str[:step], y=y_pred[:step], name="Predicted",
                       line=dict(color=HZ_COLORS.get(hk,"#d1495b"), width=1.5, dash="dash")),
        ],
        layout=go.Layout(
            title=f"WTI Crude Oil — PatchTST {hk} Forecast Unfolding  (MAE=${mae_v:.2f})",
            xaxis_title="Date", yaxis_title="$/barrel",
            updatemenus=[dict(type="buttons", showactive=False,
                              buttons=[
                                  dict(label="▶ Play",
                                       method="animate",
                                       args=[None, dict(frame=dict(duration=80,redraw=True),
                                                        fromcurrent=True)]),
                                  dict(label="⏸ Pause",
                                       method="animate",
                                       args=[[None], dict(frame=dict(duration=0,redraw=False),
                                                          mode="immediate")]),
                              ])],
            sliders=[dict(steps=[dict(args=[[f.name], dict(mode="immediate",
                                                           frame=dict(duration=80,redraw=True))],
                                     method="animate", label=dates_str[min(int(f.name)-1,
                                                                           len(dates_str)-1)])
                                 for f in _frames],
                          x=0, xanchor="left", y=0, yanchor="top", len=0.9)]),
        frames=_frames)
    add_crises(fig_anim)
    save_animation(fig_anim, f"anim_forecast_unfolding_{hk}", gif=False)
    print(f"✅ Animation anim_forecast_unfolding_{hk} saved")

# ════════════════════════════════════════════════════════════════════
# ANIMATED FIGURE B — Model Ranking Race
#   Bars re-order across horizons 1d→7d→90d→180d (one frame per
#   horizon) showing which model leads at each timescale.
# ════════════════════════════════════════════════════════════════════
if not comp_df.empty and len(HORIZONS) >= 2:
    _race_models = [m for m in comp_df["Model"].tolist()
                    if any(np.isfinite(comp_df.loc[comp_df["Model"]==m, f"MAE_{h}"].values[0])
                           for h in HORIZONS
                           if f"MAE_{h}" in comp_df.columns
                           and len(comp_df.loc[comp_df["Model"]==m, f"MAE_{h}"].values) > 0)]
    _race_models = _race_models[:20]

    def _race_bar_colors(models, hk):
        nv = metric_for("Naive", hk, "MAE")
        cols = []
        for m in models:
            mv = metric_for(m, hk, "MAE")
            if m == "Naive":                    cols.append("#2a9d8f")
            elif m == "PatchTST":               cols.append("#d1495b")
            elif "Paper" in str(m) or "LH" in str(m): cols.append("#e9a23b")
            elif np.isfinite(mv if mv==mv else float("nan")) and np.isfinite(nv if nv==nv else float("nan")) and mv < nv:
                cols.append("#06d6a0")
            else:                               cols.append("#8d99ae")
        return cols

    _race_frames = []
    for hk in HORIZONS:
        col_n = f"MAE_{hk}"
        if col_n not in comp_df.columns: continue
        _sub = comp_df[comp_df["Model"].isin(_race_models)][["Model", col_n]].dropna()
        _sub = _sub.sort_values(col_n)
        _sub_models = _sub["Model"].tolist()
        _sub_vals   = _sub[col_n].tolist()
        _race_frames.append(go.Frame(
            data=[go.Bar(x=_sub_vals, y=_sub_models, orientation="h",
                         marker_color=_race_bar_colors(_sub_models, hk),
                         text=[f"${v:.2f}" for v in _sub_vals],
                         textposition="outside")],
            layout=go.Layout(
                title_text=f"Model Ranking Race — MAE at {hk} ($/barrel)"),
            name=hk))

    if _race_frames:
        _init_hk = HORIZONS[0]
        _init_col = f"MAE_{_init_hk}"
        _init_sub = comp_df[comp_df["Model"].isin(_race_models)][["Model", _init_col]].dropna().sort_values(_init_col)
        fig_race = go.Figure(
            data=[go.Bar(x=_init_sub[_init_col].tolist(), y=_init_sub["Model"].tolist(),
                         orientation="h",
                         marker_color=_race_bar_colors(_init_sub["Model"].tolist(), _init_hk))],
            layout=go.Layout(
                title=f"Model Ranking Race — MAE ($/barrel)  [frame = horizon]",
                xaxis_title="MAE ($/barrel)", yaxis_title="Model",
                updatemenus=[dict(type="buttons", showactive=False,
                                  buttons=[
                                      dict(label="▶ Play",
                                           method="animate",
                                           args=[None, dict(frame=dict(duration=1200, redraw=True),
                                                            fromcurrent=True, transition=dict(duration=600))]),
                                      dict(label="⏸ Pause",
                                           method="animate",
                                           args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                              mode="immediate")]),
                                  ])],
                sliders=[dict(steps=[dict(args=[[f.name],
                                               dict(mode="immediate",
                                                    frame=dict(duration=1200,redraw=True),
                                                    transition=dict(duration=600))],
                                          method="animate", label=f.name)
                                     for f in _race_frames],
                              x=0, xanchor="left", y=0, yanchor="top", len=0.9)]),
            frames=_race_frames)
        save_animation(fig_race, "anim_model_ranking_race", gif=False)
        print("✅ Animation anim_model_ranking_race saved")

# ════════════════════════════════════════════════════════════════════
# FINAL SUMMARY — print headline numbers
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  THESIS HEADLINE RESULTS")
print("="*70)
for hk in HORIZONS:
    ov = eval_summary.get(f"Overall_{hk}", {})
    mae = ov.get("MAE", float("nan"))
    target = TARGETS.get(hk, float("nan"))
    status = "✅ MEETS TARGET" if mae <= target else "⚠️  above target"
    print(f"  {hk:5s}  MAE=${mae:.3f}  RMSE=${ov.get('RMSE',float('nan')):.3f}"
          f"  MAPE={ov.get('MAPE',float('nan')):.1f}%  target≤${target}  {status}")
if not comp_df.empty:
    print("\n  Best model per horizon by MAE:")
    for hk in HORIZONS:
        col_n = f"MAE_{hk}"
        if col_n not in comp_df.columns: continue
        best = comp_df.loc[comp_df[col_n].idxmin(), ["Model", col_n]].to_dict() if comp_df[col_n].notna().any() else {}
        print(f"    {hk}: {best.get('Model','?')}  MAE=${best.get(col_n, float('nan')):.3f}")
print("="*70)
print(f"\n✅ 06_analysis_plots.py complete — all figures in {_FIG_DIR}")
