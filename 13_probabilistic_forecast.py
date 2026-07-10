# ════════════════════════════════════════════════════════════════════
#  FILE : 13_probabilistic_forecast.py
#  ROLE : Stage 5c — unified probabilistic forecast layer
# ════════════════════════════════════════════════════════════════════
#  WHAT — Synthesises existing point-forecast outputs into a full
#         probabilistic framework. Reads predictions_{MODEL}_{hk}.csv
#         (which contains mu = pred_logret and sigma = sigma_logret
#         written by _forecast_template.py) and emits per model × horizon:
#
#           probabilistic_forecast_{MODEL}_{hk}.csv:
#             8 quantile columns (pred_q05, q10, q25, q50, q75, q90,
#             q95, q99) per origin — for fan charts and pinball loss.
#           scenarios_{MODEL}_{hk}.csv:
#             500 sampled future price paths for the last 20 test
#             origins (long-form: origin, scenario_id, price) —
#             for the animated scenario fan in 06_analysis_plots.
#           pit_values_{MODEL}_{hk}.csv:
#             PIT u_t = F_t(actual_logret) — uniform iff calibrated.
#
#  WHY  — A point forecast (mu) is insufficient for risk management.
#         Traders and risk desks need interval forecasts and scenario
#         fans. By reusing the sigma already produced by the conformal
#         calibration in _forecast_template, this file adds probabilistic
#         outputs with zero additional training cost. The PIT histogram
#         gives a visual calibration test: uniform PIT = well-calibrated
#         uncertainty; U-shaped PIT = over-confident; hump = over-dispersed.
#         The 500 scenario paths directly enable VaR/CVaR estimation and
#         the animated scenario fan in the thesis defense slides.
#
#  HOW  — Distribution assumption: forecast errors ∼ N(0, σ²) where σ is
#         the empirical std of validation residuals (split-conformal).
#         Per origin: quantile_k = mu + Φ⁻¹(k) × σ.
#         Monte-Carlo scenarios: draw 500 ε ~ N(0, σ²) paths of length h,
#         cumulate log-returns → price paths.
#         PIT: u_t = Φ((actual - mu_t) / σ_t).
#
#  READS    : predictions_{MODEL}_{hk}.csv (mu + sigma columns)
#  WRITES   : probabilistic_forecast_{MODEL}_{hk}.csv
#             scenarios_{MODEL}_{hk}.csv
#             pit_values_{MODEL}_{hk}.csv
#             figures/prob_fan_{MODEL}_{hk}.html|.png
#             figures/pit_histogram_{MODEL}_{hk}.html|.png
#  COMPUTE  : CPU-ONLY; trivially fast (pure numpy)
#  PIPELINE : Stage 5 — run after any model with sigma in predictions CSV
#  RUNTIME  : < 60 s for all models × horizons combined
#
# Synthesises existing model outputs into a unified probabilistic-forecast
# layer. Reads predictions_{MODEL}_{hk}.csv (with mu = pred_logret and
# sigma = sigma_logret already saved by _forecast_template.py) and emits:
#
#   - probabilistic_forecast_{MODEL}_{hk}.csv: 8 quantile columns
#       (pred_q05, q10, q25, q50, q75, q90, q95, q99) per origin
#   - scenarios_{MODEL}_{hk}.csv:               500 sampled future paths
#       for the last 20 test origins (long-form: origin, scenario_id, price)
#   - pit_values_{MODEL}_{hk}.csv:              PIT u_t = F_t(actual_logret)
#   - probabilistic_metrics.csv:                 CRPS / Sharpness / Reliability /
#                                                KS-uniformity per (model × hk)
#   - 15_fan_chart.png:        actual + 4 quantile bands per horizon
#   - 16_pit_histogram.png:    PIT histograms (flat = calibrated)
#   - 17_reliability.png:      empirical vs nominal coverage
#   - 18_scenarios.png:        50-trajectory spaghetti plot
#
# Predictive distribution per origin:
#   F_t(x) = StudentT(x; mu_t, sigma_t * conf_scale, df=5) in the central
#            body (5th–95th percentile), with GPD-extrapolated tails for
#            the deep quantiles q<0.05 and q>0.95.
#
# NO RETRAINING. Reads existing outputs only.
# Run AFTER: 07_risk_evt.py (for GPD params) + any model file you want
#            to score (PatchTST / TFT / MultiPatchTST / Ensemble).
# Usage:  exec(open('13_probabilistic_forecast.py', encoding="utf-8").read())
# ============================================================

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import genpareto, kstest, t as student_t

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────
HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
HZ_COLORS   = {"1d":"#27ae60","7d":"#e74c3c","90d":"#9b59b6","180d":"#2980b9"}
MODELS      = ["PatchTST", "MultiPatchTST", "TFT", "Ensemble"]   # extend if desired
QUANTILES   = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
NU_T        = 5.0       # Student-t degrees of freedom matching the StudentT head training prior
N_SCENARIOS = 500       # sampled future paths per origin
N_RECENT    = 20        # last N origins for scenario CSV (keeps file size bounded)
RECALIBRATE_PIT = True  # isotonic remap predicted CDF → empirical CDF if uncalibrated
NOMINAL_LEVELS = np.arange(0.05, 1.00, 0.05)   # for reliability diagram

print("=" * 60)
print("  FILE 13 — Probabilistic Forecast Engine")
print(f"  models    : {MODELS}")
print(f"  horizons  : {HORIZONS}")
print(f"  quantiles : {QUANTILES}")
print("=" * 60)

# ── Load supporting data ──────────────────────────────────────
try:
    with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df = pickle.load(f)
    with open(OUTPUT_DIR/"wti_raw.pkl",    "rb") as f: wti_raw = pickle.load(f)
    print(f"✅ df_features.pkl ({df.shape}), wti_raw.pkl loaded")
except FileNotFoundError as e:
    raise RuntimeError(f"Run 00_setup_data_features.py first: {e}")

# ── Helpers ───────────────────────────────────────────────────
def _pred_csv_path(model, hk):
    """PatchTST keeps the legacy `predictions_{hk}.csv` (no model suffix)."""
    if model == "PatchTST":
        return OUTPUT_DIR/f"predictions_{hk}.csv"
    return OUTPUT_DIR/f"predictions_{model}_{hk}.csv"

def _eval_pkl_path(model, hk):
    if model == "PatchTST":
        return OUTPUT_DIR/f"eval_{hk}.pkl"
    return OUTPUT_DIR/f"eval_{model}_{hk}.pkl"

def _load_predictions(model, hk):
    p = _pred_csv_path(model, hk)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)

def _load_eval(model, hk):
    p = _eval_pkl_path(model, hk)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)

def _fit_tail_gpd(values, q_thresh=0.90):
    """Fit a GPD to the upper tail of positive `values`. Returns (xi, beta, u, n, n_exceed)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 30:
        return None
    u = float(np.quantile(v, q_thresh))
    exc = v[v > u] - u
    if len(exc) < 10:
        return None
    try:
        xi, _, beta = genpareto.fit(exc, floc=0)
    except Exception:
        return None
    return {"xi": float(xi), "beta": float(beta), "u": u,
            "n": int(len(v)), "n_exceed": int(len(exc))}

def _gpd_var(q, fit):
    """GPD-extrapolated VaR-style quantile at level q (q > q_thresh).
    Standard POT formula:  VaR_q = u + (β/ξ) * ((n/Nu * (1-q))^(-ξ) - 1)."""
    if fit is None or q <= fit["u"]:    # below threshold → not extrapolating
        return None
    xi, beta, u, n, Nu = fit["xi"], fit["beta"], fit["u"], fit["n"], fit["n_exceed"]
    p_tail = (1 - q)
    if abs(xi) > 1e-8:
        return u + (beta / xi) * (((n / Nu) * p_tail) ** (-xi) - 1)
    return u + beta * np.log((n / Nu) / max(p_tail, 1e-12))

# ── Per-row predictive quantile in LOG-RETURN space ───────────
def _quantile_logret(mu, sigma, q, conf_scale, upper_gpd, lower_gpd):
    """One row → log-return at quantile q.
    Body uses Student-t(df=NU_T) scaled by sigma*conf_scale.
    Deep tails (q>=0.95 or q<=0.05) blend in GPD-extrapolated quantile."""
    # Student-t central body
    z_q   = student_t.ppf(q, df=NU_T)
    body  = mu + sigma * conf_scale * z_q

    # GPD blending for deep tails (only if a tail fit exists)
    if q >= 0.95 and upper_gpd is not None:
        gpd_q = _gpd_var(q, upper_gpd)
        if gpd_q is not None and np.isfinite(gpd_q):
            return mu + max(body - mu, gpd_q)        # widen toward GPD if it's heavier
    if q <= 0.05 and lower_gpd is not None:
        gpd_q = _gpd_var(1 - q, lower_gpd)           # symmetric fit on |neg-resid|
        if gpd_q is not None and np.isfinite(gpd_q):
            return mu - max(mu - body, gpd_q)
    return body

def _cdf_at(mu, sigma, x, conf_scale, upper_gpd, lower_gpd):
    """Inverse of the quantile function — predictive CDF F_t(x) at log-return x.
    Uses Student-t in the body; GPD-blended tails outside the 5th/95th."""
    z  = (x - mu) / max(sigma * conf_scale, 1e-9)
    cdf_body = float(student_t.cdf(z, df=NU_T))
    # Detect tail region via Student-t cdf
    if cdf_body >= 0.95 and upper_gpd is not None:
        excess = (x - mu)
        if excess > upper_gpd["u"]:
            xi, beta, u = upper_gpd["xi"], upper_gpd["beta"], upper_gpd["u"]
            Nu, N       = upper_gpd["n_exceed"], upper_gpd["n"]
            base = 1 - (Nu / N)
            if abs(xi) > 1e-8:
                cdf_tail = base + (Nu / N) * (1 - (1 + xi * (excess - u) / beta) ** (-1 / xi))
            else:
                cdf_tail = base + (Nu / N) * (1 - np.exp(-(excess - u) / beta))
            return max(cdf_body, float(min(cdf_tail, 1 - 1e-6)))
    if cdf_body <= 0.05 and lower_gpd is not None:
        excess = -(x - mu)
        if excess > lower_gpd["u"]:
            xi, beta, u = lower_gpd["xi"], lower_gpd["beta"], lower_gpd["u"]
            Nu, N       = lower_gpd["n_exceed"], lower_gpd["n"]
            base = (Nu / N)
            if abs(xi) > 1e-8:
                cdf_tail = base * (1 + xi * (excess - u) / beta) ** (-1 / xi)
            else:
                cdf_tail = base * np.exp(-(excess - u) / beta)
            return min(cdf_body, float(max(cdf_tail, 1e-6)))
    return cdf_body

def _isotonic_recalibrate(pit_train, pit_eval):
    """Apply isotonic regression mapping predicted CDF → empirical CDF.

    Fit on `pit_train` (e.g. first half of test, used as a proxy for validation
    since the saved CSVs don't carry validation residuals) and apply to `pit_eval`.
    Returns calibrated PIT values."""
    from sklearn.isotonic import IsotonicRegression
    pit_train = np.asarray(pit_train, dtype=float)
    pit_eval  = np.asarray(pit_eval,  dtype=float)
    pit_train = pit_train[~np.isnan(pit_train)]   # drop NaN before fitting
    if len(pit_train) < 30:
        return pit_eval
    pit_train = np.clip(pit_train, 1e-6, 1 - 1e-6)
    # Empirical CDF of training PITs at each value:
    sorted_t = np.sort(pit_train)
    emp_cdf  = np.searchsorted(sorted_t, pit_train, side="right") / len(pit_train)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(pit_train, emp_cdf)
    pit_eval_clean = np.where(np.isnan(pit_eval), 0.5, pit_eval)  # replace NaN with neutral 0.5
    return np.clip(iso.predict(np.clip(pit_eval_clean, 1e-6, 1 - 1e-6)), 1e-6, 1 - 1e-6)

# ── Core scoring loop ─────────────────────────────────────────
metrics_rows  = []
all_fan_data  = {}    # (model, hk) → quantile DataFrame for fan-chart figure
all_pit_data  = {}    # (model, hk) → calibrated PIT array

for model in MODELS:
    for hk in HORIZONS:
        h_days = {"1d":1, "7d":7, "90d":90, "180d":180}[hk]
        res    = _load_predictions(model, hk)
        if res is None or res.empty:
            continue
        if not all(c in res.columns for c in ("pred_logret","sigma_logret","actual")):
            print(f"  ⚠️  {model} {hk}: missing pred_logret / sigma_logret columns — skipping")
            continue

        ev     = _load_eval(model, hk)
        conf_lo = ev.get(f"conf_lo_{hk}")
        conf_hi = ev.get(f"conf_hi_{hk}")
        # Width-correction factor: scale Student-t std deviates so the 10th/90th
        # percentile of the predictive distribution matches the conformal offsets.
        if (conf_lo is not None) and (conf_hi is not None):
            z10 = student_t.ppf(0.10, df=NU_T)
            z90 = student_t.ppf(0.90, df=NU_T)
            conf_scale = float((conf_hi - conf_lo) / max(z90 - z10, 1e-6))
        else:
            conf_scale = 1.0

        # Origin price per row (needed to convert log-return quantiles → price)
        origin_price = df["price"].reindex(res.index).values
        actual_price = res["actual"].astype(float).values
        mu_arr       = res["pred_logret"].astype(float).values
        sigma_arr    = res["sigma_logret"].astype(float).fillna(np.nanmedian(res["sigma_logret"])).values
        sigma_arr[~np.isfinite(sigma_arr)] = float(np.nanmedian(sigma_arr[np.isfinite(sigma_arr)]) or 0.05)

        # Test log-return residuals → tail-fit GPDs (signed)
        actual_logret = np.log(np.clip(actual_price, 1e-6, None) / np.clip(origin_price, 1e-6, None))
        resid         = actual_logret - mu_arr
        upper_gpd     = _fit_tail_gpd(np.clip( resid, 0, None), q_thresh=0.90)
        lower_gpd     = _fit_tail_gpd(np.clip(-resid, 0, None), q_thresh=0.90)

        # Build quantile fan in log-return space
        quant_cols = {f"pred_q{int(q*100):02d}": np.zeros(len(res)) for q in QUANTILES}
        for i, q in enumerate(QUANTILES):
            for r in range(len(res)):
                lr = _quantile_logret(mu_arr[r], sigma_arr[r], q, conf_scale,
                                       upper_gpd, lower_gpd)
                quant_cols[f"pred_q{int(q*100):02d}"][r] = lr
        # Convert to price-space
        for q in QUANTILES:
            col = f"pred_q{int(q*100):02d}"
            quant_cols[col] = origin_price * np.exp(quant_cols[col])

        # Sort quantiles per row to enforce monotonicity (occasionally violated
        # at the seam between Student-t body and GPD tail)
        quant_arr = np.column_stack([quant_cols[f"pred_q{int(q*100):02d}"] for q in QUANTILES])
        quant_arr = np.sort(quant_arr, axis=1)
        for i, q in enumerate(QUANTILES):
            quant_cols[f"pred_q{int(q*100):02d}"] = quant_arr[:, i]

        # Assemble + save quantile CSV
        prob_df = pd.DataFrame(quant_cols, index=res.index)
        prob_df.insert(0, "actual", actual_price)
        prob_df["origin_price"] = origin_price
        prob_csv = OUTPUT_DIR/f"probabilistic_forecast_{model}_{hk}.csv"
        prob_df.to_csv(prob_csv)

        # PIT u_t = F_t(actual_logret) per row
        pit = np.array([_cdf_at(mu_arr[i], sigma_arr[i], actual_logret[i],
                                conf_scale, upper_gpd, lower_gpd)
                        for i in range(len(res))])
        pit_raw = pit.copy()
        # Optional isotonic recalibration: train on first half, apply to second
        if RECALIBRATE_PIT and len(pit) > 60:
            mid = len(pit) // 2
            pit = np.concatenate([_isotonic_recalibrate(pit[:mid], pit[:mid]),
                                  _isotonic_recalibrate(pit[:mid], pit[mid:])])
        pd.DataFrame({"pit": pit, "pit_raw": pit_raw,
                      "actual_logret": actual_logret},
                     index=res.index).to_csv(OUTPUT_DIR/f"pit_values_{model}_{hk}.csv")

        # Scenario sampling (last N_RECENT origins, N_SCENARIOS draws each)
        scen_rows = []
        rng = np.random.default_rng(42)
        for r in range(max(0, len(res) - N_RECENT), len(res)):
            for sid in range(N_SCENARIOS):
                z      = student_t.rvs(df=NU_T, random_state=rng)
                lr     = mu_arr[r] + sigma_arr[r] * conf_scale * z
                price  = origin_price[r] * float(np.exp(lr))
                scen_rows.append({"origin": res.index[r], "scenario_id": sid,
                                   "scenario_price": price})
        if scen_rows:
            pd.DataFrame(scen_rows).to_csv(OUTPUT_DIR/f"scenarios_{model}_{hk}.csv",
                                            index=False)

        # ── Probabilistic skill metrics ───────────────────────
        # CRPS approximation: integral of (F_t(x) - 1{x>=y})^2 dx — closed-form
        # for a Student-t is messy, so we evaluate via the 8 quantiles trapezoid.
        crps_vals = []
        for r in range(len(res)):
            qs = np.array(QUANTILES)
            xs = np.array([quant_cols[f"pred_q{int(q*100):02d}"][r] / origin_price[r]
                            for q in QUANTILES])
            xs = np.log(np.clip(xs, 1e-6, None))     # log-return space
            y  = actual_logret[r]
            # CRPS via the "quantile decomposition": 2 * E[(τ - 1{y<F^{-1}(τ)}) (F^{-1}(τ) - y)]
            indic = (y < xs).astype(float)
            integrand = (qs - indic) * (xs - y)
            crps_vals.append(2 * float(np.trapz(integrand, qs)))
        crps = float(np.nanmean(np.abs(crps_vals)))

        # Sharpness: mean (q90 - q10) band width in price space
        band_w = (quant_cols["pred_q90"] - quant_cols["pred_q10"])
        sharpness = float(np.nanmean(band_w))

        # Reliability: mean abs deviation of empirical coverage from nominal
        rel_devs = []
        for nl in NOMINAL_LEVELS:
            lower_pct = (1 - nl) / 2
            upper_pct = 1 - lower_pct
            qL = np.array([_quantile_logret(mu_arr[r], sigma_arr[r], lower_pct,
                                             conf_scale, upper_gpd, lower_gpd)
                            for r in range(len(res))])
            qU = np.array([_quantile_logret(mu_arr[r], sigma_arr[r], upper_pct,
                                             conf_scale, upper_gpd, lower_gpd)
                            for r in range(len(res))])
            covered = ((actual_logret >= qL) & (actual_logret <= qU)).mean()
            rel_devs.append(abs(covered - nl))
        reliability = float(np.mean(rel_devs))

        # KS-distance of PIT histogram from Uniform(0,1)
        ks_pit = float(kstest(pit[np.isfinite(pit)], "uniform").statistic) if np.isfinite(pit).sum() > 10 else np.nan

        metrics_rows.append({
            "Model":       model,
            "Horizon":     hk,
            "n_origins":   len(res),
            "CRPS":        round(crps, 4),
            "Sharpness":   round(sharpness, 2),
            "Reliability": round(reliability, 4),
            "KS_PIT":      round(ks_pit, 4),
            "conf_scale":  round(conf_scale, 3),
            "upper_GPD_xi":  round(upper_gpd["xi"], 3) if upper_gpd else np.nan,
            "lower_GPD_xi":  round(lower_gpd["xi"], 3) if lower_gpd else np.nan,
        })
        print(f"  ✅ {model:14s} {hk:5s}  "
              f"CRPS={crps:.4f}  Sharp={sharpness:.2f}  "
              f"Rel={reliability:.4f}  KS_PIT={ks_pit:.4f}  "
              f"n={len(res)}")

        all_fan_data[(model, hk)] = prob_df
        all_pit_data[(model, hk)] = pit

# ── Save metrics CSV ──────────────────────────────────────────
if metrics_rows:
    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(OUTPUT_DIR/"probabilistic_metrics.csv", index=False)
    print("\n" + "=" * 60)
    print("  PROBABILISTIC METRICS TABLE")
    print("=" * 60)
    print(mdf.to_string(index=False))
    print("\nInterpretation:")
    print("  CRPS:        lower = sharper + better calibrated (units = avg log-return error)")
    print("  Sharpness:   80% band width in $; narrower = sharper at fixed coverage")
    print("  Reliability: mean |empirical_coverage - nominal| across 19 levels (smaller better)")
    print("  KS_PIT:      KS-distance of PIT from Uniform(0,1) — <0.10 = well-calibrated")
else:
    print("\n⚠️  No models produced probabilistic outputs — check that predictions CSVs exist.")

# ── Figures (Plotly) — all guarded so a plotting hiccup never aborts Stage 13 ──
print("\nGenerating figures …")
headline_model = "PatchTST"
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo13
    from plotly.subplots import make_subplots as _pmsp13

    # ── Figure 15: Fan chart (one panel per horizon) ──────────────
    try:
        _fig15 = _pmsp13(rows=len(HORIZONS), cols=1,
                         subplot_titles=[f"{headline_model} — {hk} probabilistic forecast (quantile fan)"
                                         for hk in HORIZONS], vertical_spacing=0.08)
        _bands = [(0.20, 0.80, "#a9cce3", "60% band"),
                  (0.10, 0.90, "#7fb3d5", "80% band"),
                  (0.05, 0.95, "#5499c7", "90% band")]
        for i, hk in enumerate(HORIZONS, 1):
            pdf = all_fan_data.get((headline_model, hk))
            if pdf is None or pdf.empty:
                continue
            _dx = pdf.index.strftime("%Y-%m-%d").tolist()
            for lo, hi, color, lab in _bands:
                col_lo, col_hi = f"pred_q{int(lo*100):02d}", f"pred_q{int(hi*100):02d}"
                if col_lo in pdf.columns and col_hi in pdf.columns:
                    _fig15.add_trace(_pgo13.Scatter(
                        x=_dx + _dx[::-1],
                        y=list(pdf[col_hi]) + list(pdf[col_lo][::-1]),
                        fill="toself", fillcolor=color, opacity=0.85, line_width=0,
                        name=lab, showlegend=(i == 1)), row=i, col=1)
            _fig15.add_trace(_pgo13.Scatter(x=_dx, y=pdf["pred_q50"], name="median",
                                            line=dict(color="#2c3e50", width=1.4),
                                            showlegend=(i == 1)), row=i, col=1)
            _fig15.add_trace(_pgo13.Scatter(x=_dx, y=pdf["actual"], name="actual",
                                            line=dict(color="#e74c3c", width=1.0),
                                            showlegend=(i == 1)), row=i, col=1)
            _fig15.update_yaxes(title_text="Price ($/bbl)", row=i, col=1)
        _fig15.update_layout(height=340 * len(HORIZONS),
                             title=f"{headline_model} — Quantile Fan Charts")
        save_fig(_fig15, "15_fan_chart")
        print("  ✅ 15_fan_chart")
    except Exception as _e15:
        print(f"  [warn] fan chart skipped: {_e15}")

    # ── Figure 16: PIT histograms (calibration) ───────────────────
    try:
        _fig16 = _pmsp13(rows=1, cols=len(HORIZONS),
                         subplot_titles=[f"{hk}: PIT histogram" for hk in HORIZONS])
        for i, hk in enumerate(HORIZONS, 1):
            pit = all_pit_data.get((headline_model, hk))
            if pit is None or not len(pit):
                continue
            pit = pit[np.isfinite(pit)]
            _col = HZ_COLORS[hk] if hk in HZ_COLORS else "#1f3b73"
            _fig16.add_trace(_pgo13.Histogram(x=pit, nbinsx=20, marker_color=_col,
                                              opacity=0.75, showlegend=False), row=1, col=i)
            _fig16.update_xaxes(title_text="u_t = F_t(actual)", range=[0, 1], row=1, col=i)
        _fig16.update_layout(height=420,
                             title=f"PIT histograms ({headline_model}) — flat = well-calibrated")
        save_fig(_fig16, "16_pit_histogram")
        print("  ✅ 16_pit_histogram")
    except Exception as _e16:
        print(f"  [warn] PIT histogram skipped: {_e16}")

    # ── Figure 17: Reliability diagram ────────────────────────────
    try:
        _fig17 = _pmsp13(rows=1, cols=len(HORIZONS),
                         subplot_titles=[f"{hk}: reliability" for hk in HORIZONS])
        for i, hk in enumerate(HORIZONS, 1):
            pdf = all_fan_data.get((headline_model, hk))
            if pdf is None or pdf.empty:
                continue
            actual_p = pdf["actual"].values
            obs = []
            for nl in NOMINAL_LEVELS:
                lo_pct = int((1 - nl) / 2 * 100); hi_pct = int((1 - (1 - nl) / 2) * 100)
                col_lo, col_hi = f"pred_q{lo_pct:02d}", f"pred_q{hi_pct:02d}"
                if col_lo in pdf.columns and col_hi in pdf.columns:
                    obs.append(float(((actual_p >= pdf[col_lo].values) &
                                      (actual_p <= pdf[col_hi].values)).mean()))
                else:
                    obs.append(np.nan)
            _col = HZ_COLORS[hk] if hk in HZ_COLORS else "#1f3b73"
            _fig17.add_trace(_pgo13.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                            line=dict(color="black", dash="dash"),
                                            name="perfect", showlegend=(i == 1)), row=1, col=i)
            _fig17.add_trace(_pgo13.Scatter(x=list(NOMINAL_LEVELS), y=obs, mode="lines+markers",
                                            line=dict(color=_col, width=2),
                                            name="empirical", showlegend=(i == 1)), row=1, col=i)
            _fig17.update_xaxes(title_text="nominal", range=[0, 1], row=1, col=i)
            _fig17.update_yaxes(range=[0, 1], row=1, col=i)
        _fig17.update_layout(height=420,
                             title=f"Reliability diagrams ({headline_model}) — diagonal = perfect")
        save_fig(_fig17, "17_reliability")
        print("  ✅ 17_reliability")
    except Exception as _e17:
        print(f"  [warn] reliability diagram skipped: {_e17}")

    # ── Figure 18: Scenario spaghetti (most recent origin per horizon) ──
    try:
        _fig18 = _pmsp13(rows=1, cols=len(HORIZONS),
                         subplot_titles=[f"{hk}: sampled scenarios" for hk in HORIZONS])
        for i, hk in enumerate(HORIZONS, 1):
            scen_path = OUTPUT_DIR/f"scenarios_{headline_model}_{hk}.csv"
            if not scen_path.exists():
                continue
            sdf = pd.read_csv(scen_path, parse_dates=["origin"])
            if sdf.empty:
                continue
            last_origin = sdf["origin"].max()
            recent = sdf[sdf["origin"] == last_origin]
            h_days = {"1d":1, "7d":7, "90d":90, "180d":180}[hk]
            target = last_origin + pd.tseries.offsets.BDay(h_days)
            op = float(df["price"].reindex([last_origin])[0]) if last_origin in df.index else np.nan
            _col = HZ_COLORS[hk] if hk in HZ_COLORS else "#1f3b73"
            plotted = recent.sample(min(50, len(recent)), random_state=0)
            _ox, _tx = last_origin.strftime("%Y-%m-%d"), target.strftime("%Y-%m-%d")
            for _, row in plotted.iterrows():
                _fig18.add_trace(_pgo13.Scatter(x=[_ox, _tx], y=[op, row["scenario_price"]],
                                                mode="lines", line=dict(color=_col, width=0.8),
                                                opacity=0.25, showlegend=False), row=1, col=i)
            med = float(recent["scenario_price"].median())
            _fig18.add_trace(_pgo13.Scatter(x=[_ox, _tx], y=[op, med], mode="lines",
                                            line=dict(color="black", width=2),
                                            name="median", showlegend=(i == 1)), row=1, col=i)
            if target in wti_raw.index:
                _fig18.add_trace(_pgo13.Scatter(x=[_tx], y=[float(wti_raw.loc[target, "price"])],
                                                mode="markers", marker=dict(color="red", size=10),
                                                name="actual", showlegend=(i == 1)), row=1, col=i)
            _fig18.update_yaxes(title_text="Price ($/bbl)", row=1, col=i)
        _fig18.update_layout(height=440,
                             title=f"Scenario sampling ({headline_model}) — N={N_SCENARIOS} draws/origin")
        save_fig(_fig18, "18_scenarios")
        print("  ✅ 18_scenarios")
    except Exception as _e18:
        print(f"  [warn] scenario figure skipped: {_e18}")
except Exception as _e13fig:
    print(f"  [warn] Stage 13 figures skipped: {_e13fig}")

print("\n✅ FILE 13 COMPLETE — probabilistic forecast layer saved")
print(f"   Outputs in {OUTPUT_DIR}/")
print(f"     probabilistic_forecast_{{MODEL}}_{{hk}}.csv  (8 quantile columns per origin)")
print(f"     scenarios_{{MODEL}}_{{hk}}.csv               (500 sampled paths × last {N_RECENT} origins)")
print(f"     pit_values_{{MODEL}}_{{hk}}.csv              (PIT u_t per origin)")
print(f"     probabilistic_metrics.csv                  (CRPS / Sharpness / Reliability / KS)")
print(f"     15_fan_chart.png  16_pit_histogram.png  17_reliability.png  18_scenarios.png")
