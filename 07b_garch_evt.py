# ════════════════════════════════════════════════════════════════════
#  FILE : 07b_garch_evt.py
#  ROLE : Stage 5b — conditional (time-varying) GARCH-EVT tail risk
# ════════════════════════════════════════════════════════════════════
#  WHAT — Upgrades the static GPD-POT from 07_risk_evt.py to the
#         McNeil-Frey (2000) GARCH-EVT framework:
#           1. GARCH(1,1) on log-returns → conditional volatility σ_t.
#           2. Standardised residuals z_t = ε_t / σ_t (approximately i.i.d.).
#           3. POT-GPD fit on the tails of z_t.
#           4. Dynamic VaR_t = σ_t × GPD_quantile(α) + μ_t.
#         Backtests: Kupiec POF, Christoffersen CC (independence).
#         Also saves σ_t as a feature column for downstream models.
#
#  WHY  — Static EVT (07_risk_evt) ignores volatility clustering: it
#         assigns the same VaR to a calm day and a volatile one.
#         GARCH-EVT is the industry standard (Basel III IMA) because it
#         captures both clustering (GARCH) and fat tails (GPD) — the two
#         dominant features of crude-oil return risk. The dynamic σ_t
#         output also feeds back into _forecast_template.py as a
#         conditioning feature, closing the loop between risk and prediction.
#
#  HOW  — Per horizon hk:
#           1. Load actual log-returns from df_features.pkl (training window).
#           2. Fit GARCH(1,1) via arch library (or manual recursion fallback).
#           3. Compute standardised residuals z_t = r_t / σ_t.
#           4. POT threshold u = 90th percentile of |z_t|.
#           5. Fit GPD to exceedances; derive dynamic VaR_t, ES_t.
#           6. Backtest on out-of-sample period; record p-values.
#           7. Save garch_evt_{hk}.pkl, garch_evt_table.csv.
#
#  READS    : df_features.pkl (log_return), predictions_{hk}.csv
#  WRITES   : garch_evt_{hk}.pkl (VaR/ES time series + backtest stats)
#             garch_evt_table.csv
#             figures/garch_evt_{hk}.html|.png
#  COMPUTE  : CPU-ONLY; GARCH(1,1) fit < 5 s per horizon
#  PIPELINE : Stage 5 — run alongside 07_risk_evt; before Stage 13
#  RUNTIME  : < 60 s for all horizons
# ════════════════════════════════════════════════════════════════════

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))

if "compute_garch_evt" not in globals():
    exec(open("_garch_evt.py", encoding="utf-8").read())

HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
TRAIN_FRAC = 0.7   # match walk-forward start year (2007)
QUANTILES  = (0.95, 0.99)

print("=" * 60)
print("  Dynamic GARCH-EVT (McNeil-Frey) — conditional VaR/ES")
print("=" * 60)

with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
    df = pickle.load(f)

returns = df["log_ret_1d"].fillna(0).values.astype(float)   # 1-day log returns

rows = []
garch_evts = {}
for hk in HORIZONS:
    h = {"1d":1,"7d":7,"90d":90,"180d":180}[hk]
    # h-day log-return
    ret_h = df["log_price"].diff(h).fillna(0).values.astype(float)

    print(f"\n  Horizon {hk} (h={h}):")
    ev = compute_garch_evt(ret_h, quantiles=QUANTILES, train_frac=TRAIN_FRAC)
    garch_evts[hk] = ev

    n_test = int(len(ret_h) * (1 - TRAIN_FRAC))
    test_ret = ret_h[-n_test:]

    row = {"horizon": hk, "GARCH_xi": round(ev["xi"],4), "GARCH_beta": round(ev["beta"],4),
           "threshold_u": round(ev["threshold_u"],4), "pct_exceed": round(ev["pct_exceed"]*100,2)}

    for q in QUANTILES:
        qi = int(q*100)
        dvar = ev[f"dynamic_VaR_{qi}"][-n_test:] if len(ev[f"dynamic_VaR_{qi}"]) >= n_test else ev[f"dynamic_VaR_{qi}"]
        lr_k, pv_k = kupiec_test(test_ret, dvar, q)
        lr_c, pv_c = christoffersen_test(test_ret, dvar, q)
        row[f"dynamic_VaR{qi}(avg)"] = round(float(np.nanmean(np.abs(dvar))),4)
        row[f"static_VaR{qi}"]       = round(ev[f"static_VaR_{qi}"],4) if f"static_VaR_{qi}" in ev else float("nan")
        row[f"Kupiec_p{qi}"]         = round(pv_k,3) if pv_k is not None else float("nan")
        row[f"Christoff_p{qi}"]      = round(pv_c,3) if pv_c is not None else float("nan")
        ku_pass = "PASS" if (pv_k is not None and not np.isnan(pv_k) and pv_k > 0.05) else "FAIL"
        ch_pass = "PASS" if (pv_c is not None and not np.isnan(pv_c) and pv_c > 0.05) else "FAIL"
        print(f"    VaR{qi}%: dynamic(avg)={row[f'dynamic_VaR{qi}(avg)']:.4f}  "
              f"static={row.get(f'static_VaR{qi}',float('nan')):.4f}  "
              f"Kupiec p={row[f'Kupiec_p{qi}']:.3f} [{ku_pass}]  "
              f"Christoff p={row[f'Christoff_p{qi}']:.3f} [{ch_pass}]")

    rows.append(row)
    with open(OUTPUT_DIR / f"garch_evt_{hk}.pkl", "wb") as f2:
        pickle.dump(ev, f2)

    # also save dynamic sigma as a feature-augmentation series
    sig_col = pd.Series(ev["sigma"], index=df.index[:len(ev["sigma"])],
                        name=f"garch_sigma_{hk}")
    ev["sigma_series"] = sig_col  # for feature import

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "garch_evt_table.csv", index=False)
print(f"\n  ✅ Saved garch_evt_{{1d..180d}}.pkl + garch_evt_table.csv")
print("  Usage in _fusion.py: replace static evt_clip with dynamic VaR_t bounds.")
print("  Usage in 13_probabilistic_forecast: replace static GPD with garch_evt_{hk}.pkl.")
print("\n  NOTE: dynamic VaR passes Kupiec if VaR model is well-calibrated (p > 0.05).")
print("        Christoffersen tests if violations cluster (p > 0.05 = no clustering).")
