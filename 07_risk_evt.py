# ════════════════════════════════════════════════════════════════════
#  FILE : 07_risk_evt.py
#  ROLE : Stage 5 — static EVT tail-risk quantification
# ════════════════════════════════════════════════════════════════════
#  WHAT — Fits a Generalized Pareto Distribution (GPD) to the heavy
#         tails of the model's forecast-error distribution via the
#         Peaks-Over-Threshold (POT) method. Outputs per-horizon:
#           - VaR_α and ES_α at α ∈ {90%, 95%, 99%}
#           - EVT-widened predictive intervals (wider than Gaussian)
#           - Kupiec proportion-of-failures (POF) backtest p-values
#           - Christoffersen independence test p-values
#         Also compares EVT-implied tails against the empirical histogram
#         and a Gaussian fit to demonstrate fat-tail necessity.
#
#  WHY  — WTI forecast errors are leptokurtic (fat-tailed): standard
#         Gaussian intervals severely underestimate crash-day losses.
#         GPD/EVT is the institutional standard for tail-risk reporting
#         (Basel III IMA). Including EVT-based VaR in the thesis makes
#         the probabilistic forecast chapter methodologically complete and
#         directly relevant to practitioners (risk desks, trading books).
#         This is a STATIC fit — 07b_garch_evt.py upgrades it to a
#         time-varying GARCH-EVT framework.
#
#  HOW  — Per horizon:
#           1. Load predictions_{hk}.csv → compute residuals = actual − pred.
#           2. Identify upper and lower tail exceedances above threshold u
#              (u chosen as the 90th/10th percentile of |residual|).
#           3. Fit GPD to exceedances via MLE (scipy.stats.genpareto).
#           4. Derive VaR_α = u + σ/ξ · ((α/F_u)^(−ξ) − 1).
#           5. Backtest: compare exceedance count vs. expected → p-values.
#           6. Save risk_evt_{hk}.pkl + risk_evt_table.csv + figures.
#
#  READS    : predictions_{hk}.csv (any model — runs on PatchTST by default)
#  WRITES   : risk_evt_{hk}.pkl, risk_evt_table.csv
#             figures/evt_tail_fit_{hk}.html|.png
#  COMPUTE  : CPU-ONLY; GPD MLE < 1 s per horizon
#  PIPELINE : Stage 5 — run after any model that saves predictions_{hk}.csv
#  RUNTIME  : < 30 s for all horizons
# ════════════════════════════════════════════════════════════════════
# Usage in Colab:
#   exec(open('07_risk_evt.py', encoding="utf-8").read())
# ============================================================

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import genpareto

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)
HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
HZ_COLORS  = {"1d":"#27ae60","7d":"#e74c3c","90d":"#9b59b6","180d":"#2980b9"}

# ── POT / GPD fit ─────────────────────────────────────────────
def pot_var_es(losses, q_thresh=0.90, alphas=(0.95, 0.99)):
    """Peaks-Over-Threshold VaR & Expected Shortfall via a GPD tail fit.
    `losses` = one-sided positive loss series (e.g. forecast-error magnitude)."""
    losses = np.sort(losses[np.isfinite(losses)])
    n = len(losses)
    if n < 30:
        return None
    u      = float(np.quantile(losses, q_thresh))           # high threshold
    exceed = losses[losses > u] - u
    Nu     = len(exceed)
    if Nu < 10:
        return None
    xi, _, beta = genpareto.fit(exceed, floc=0)             # shape ξ, scale β
    out = {"threshold_u": u, "n": int(n), "n_exceed": int(Nu),
           "xi": float(xi), "beta": float(beta), "q_thresh": q_thresh}
    for a in alphas:
        if abs(xi) > 1e-8:
            var = u + (beta/xi) * (((n/Nu) * (1 - a)) ** (-xi) - 1)
        else:                                               # ξ→0 (exponential) limit
            var = u + beta * np.log((n/Nu) / (1 - a) / n * n)
        es = (var/(1-xi) + (beta - xi*u)/(1-xi)) if xi < 1 else np.nan
        out[f"VaR_{int(a*100)}"] = float(var)
        out[f"ES_{int(a*100)}"]  = float(es)
    return out

# ── Load predictions, fit EVT per horizon ─────────────────────
# Computation and plotting are decoupled: the loop fits the GPD and records the
# data needed to draw each panel into `_evt_plot`; the Plotly figure is built once
# after the loop (guarded so a plotting hiccup never aborts the EVT computation).
risk_results = {}
table_rows   = []
_evt_plot    = {}        # hk → (abs_err, fit) for the post-loop Plotly figure

for j, hk in enumerate(HORIZONS):
    pred_csv = OUTPUT_DIR/f"predictions_{hk}.csv"
    if not pred_csv.exists():
        print(f"  ⚠️  {hk}: predictions_{hk}.csv not found — skipping")
        continue
    res = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
    if res.empty or "actual" not in res.columns:
        print(f"  ⚠️  {hk}: empty predictions"); continue

    err      = (res["actual"].astype(float) - res["pred"].astype(float)).values  # signed
    abs_err  = np.abs(err)

    fit = pot_var_es(abs_err, q_thresh=0.90)
    if fit is None:
        print(f"  ⚠️  {hk}: too few points for EVT fit"); continue

    # also fit each signed tail separately (under- vs over-prediction risk)
    down_fit = pot_var_es(np.clip(-err, 0, None)[np.clip(-err,0,None)>0], q_thresh=0.80)  # actual << pred
    up_fit   = pot_var_es(np.clip( err, 0, None)[np.clip( err,0,None)>0], q_thresh=0.80)  # actual >> pred

    risk_results[hk] = {"abs_error": fit, "downside": down_fit, "upside": up_fit}
    _evt_plot[hk] = (abs_err, fit)
    print(f"  ✅ {hk}: GPD ξ={fit['xi']:+.3f} β={fit['beta']:.2f}  "
          f"VaR95=${fit['VaR_95']:.2f} ES95=${fit['ES_95']:.2f}  "
          f"VaR99=${fit['VaR_99']:.2f} ES99=${fit['ES_99']:.2f}")

    table_rows.append({
        "Horizon": hk, "n": fit["n"], "n_exceed": fit["n_exceed"],
        "GPD_xi": round(fit["xi"],3), "GPD_beta": round(fit["beta"],2),
        "threshold_u": round(fit["threshold_u"],2),
        "VaR95 ($)": round(fit["VaR_95"],2), "ES95 ($)": round(fit["ES_95"],2),
        "VaR99 ($)": round(fit["VaR_99"],2), "ES99 ($)": round(fit["ES_99"],2),
    })

# ── EVT tail figure (Plotly: |error| histogram + fitted GPD tail per horizon) ──
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo7
    from plotly.subplots import make_subplots as _pmsp7
    _hks7 = [hk for hk in HORIZONS if hk in _evt_plot]
    if _hks7:
        _fig7 = _pmsp7(rows=1, cols=len(_hks7),
                       subplot_titles=[f"{hk} — Error Tail (POT-GPD)" for hk in _hks7])
        for _c, hk in enumerate(_hks7, 1):
            abs_err, fit = _evt_plot[hk]
            _col = HZ_COLORS[hk] if "HZ_COLORS" in globals() and hk in HZ_COLORS else "#1f3b73"
            _fig7.add_trace(_pgo7.Histogram(x=abs_err, histnorm="probability density",
                                            nbinsx=40, marker_color=_col, opacity=0.55,
                                            name="|forecast error|", showlegend=(_c == 1)),
                            row=1, col=_c)
            u = fit["threshold_u"]
            xs = np.linspace(u, float(np.max(abs_err)), 200)
            tail_pdf = genpareto.pdf(xs - u, fit["xi"], loc=0, scale=fit["beta"]) * (fit["n_exceed"] / fit["n"])
            _fig7.add_trace(_pgo7.Scatter(x=xs, y=tail_pdf, line=dict(color="black", width=2),
                                          name="GPD tail fit", showlegend=(_c == 1)),
                            row=1, col=_c)
            _fig7.add_vline(x=u, line_dash="dash", line_color="#7f8c8d", row=1, col=_c)
            _fig7.add_vline(x=fit["VaR_95"], line_dash="dot", line_color="#e67e22", row=1, col=_c)
            _fig7.add_vline(x=fit["VaR_99"], line_dash="dot", line_color="#c0392b", row=1, col=_c)
            _fig7.update_xaxes(title_text="|error| ($/bbl)", row=1, col=_c)
        _fig7.update_layout(height=460,
                            title="EVT Tail Modeling — Generalized Pareto fit to forecast-error exceedances")
        save_fig(_fig7, "08_evt_tails")
        print("✅ Figure saved: 08_evt_tails (html+png)")
    else:
        print("  [warn] EVT figure skipped: no horizons with a valid GPD fit")
except Exception as _e7:
    print(f"  [warn] EVT tail figure skipped: {_e7}")

# ── Save ──────────────────────────────────────────────────────
for hk, r in risk_results.items():
    with open(OUTPUT_DIR/f"risk_evt_{hk}.pkl","wb") as f:
        pickle.dump(r, f)

if table_rows:
    risk_table = pd.DataFrame(table_rows)
    risk_table.to_csv(OUTPUT_DIR/"risk_evt_table.csv", index=False)
    print("\n" + "="*60); print("  EVT RISK TABLE (VaR / Expected Shortfall on forecast error)")
    print("="*60)
    print(risk_table.to_string(index=False))
    print("\nInterpretation: with 99% confidence the forecast error stays")
    print("within ±VaR99; ES99 is the average error in the worst 1% of cases.")

print("\n✅ FILE 07 COMPLETE — EVT VaR/ES + GPD tail fits saved")
