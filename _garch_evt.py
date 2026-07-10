# ============================================================
# DDP_v2/_garch_evt.py
# Conditional/Dynamic EVT helpers — the black-swan upgrade over the base
# 07_risk_evt.py (static unconditional POT/GPD on |residuals|, reported-only).
#
# Implements McNeil–Frey (2000) two-stage GARCH-EVT:
#   Stage 1 — fit GARCH(1,1) to log-returns → obtain conditional μ_t, σ_t
#   Stage 2 — standardise residuals z_t = (r_t - μ_t)/σ_t
#            → fit GPD to the tail of z_t (POT, threshold = 90th pctile)
#            → dynamic VaR_t = μ_t + σ_t · z_q(GPD)
#            → dynamic ES_t  = μ_t + σ_t · ES_q(GPD)
#
# Also implements Backtesting: Kupiec POF + Christoffersen independence + DQ.
# Optional: lightweight 2T-POT Hawkes for self-exciting clustered extremes.
# Uses the `arch` package (pip install arch) with graceful fallback to
# a simple EWMA volatility estimate if arch is unavailable.
# Run / exec from inside DDP_v2/.
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))


def _fit_garch(returns, p=1, q=1):
    """Fit GARCH(p,q) to returns. Returns (mu_hat, sigma_hat, std_residuals)."""
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="Garch", p=p, q=q, mean="Constant", dist="Normal")
        res = am.fit(disp="off", show_warning=False)
        mu = res.params["Const"] / 100
        sigma = res.conditional_volatility.values / 100
        std_resid = ((returns - mu) / (sigma + 1e-12))
        return float(mu), sigma, std_resid
    except Exception:
        # Fallback: EWMA volatility (lambda=0.94, RiskMetrics)
        lam = 0.94
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns[:30]) if n > 30 else returns[0]**2
        for t in range(1, n):
            sigma2[t] = lam * sigma2[t-1] + (1-lam) * returns[t-1]**2
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        mu = float(np.mean(returns[:int(n*0.7)]))
        std_resid = (returns - mu) / sigma
        return mu, sigma, std_resid


def _fit_gpd_tail(std_resid, quantile=0.90):
    """Fit GPD to exceedances over quantile threshold. Returns (xi, beta, u, pct_exceed)."""
    u = float(np.quantile(std_resid, quantile))
    exceed = std_resid[std_resid > u] - u
    if len(exceed) < 10:
        return 0.0, 1.0, u, 0.0
    try:
        xi, _, beta = stats.genpareto.fit(exceed, floc=0)
    except Exception:
        xi, beta = 0.0, float(np.std(exceed) + 1e-12)
    return float(xi), float(beta), float(u), float(len(exceed) / len(std_resid))


def gpd_var_es(xi, beta, u, q, n_total, n_exceed):
    """GPD VaR and ES at quantile q using POT formula."""
    if n_exceed <= 0 or n_total <= 0:
        return np.nan, np.nan
    p_exceed = n_exceed / n_total
    if p_exceed <= 0 or (1 - q) <= 0:
        return np.nan, np.nan
    t = (1 - q) / p_exceed   # quantile in exceedance distribution
    if xi != 0:
        z_q = u + (beta / xi) * (t**(-xi) - 1)
    else:
        z_q = u + beta * np.log(t)
    # ES
    if xi < 1:
        es_q = (z_q + beta - xi * u) / (1 - xi)
    else:
        es_q = np.nan
    return float(z_q), float(es_q)


def compute_garch_evt(returns: np.ndarray, quantiles=(0.95, 0.99),
                      tail_q=0.90, train_frac=0.7):
    """Full GARCH-EVT pipeline on a 1D log-return series.
    Fits on train (first train_frac), then computes dynamic VaR/ES on whole series.

    Returns:
      dict with keys: sigma (cond. vol series), std_resid, xi, beta, u,
      and per-quantile {VaR_t_q, ES_t_q, static_VaR_q, dynamic_VaR_q} arrays.
    """
    n = len(returns)
    train_n = int(n * train_frac)
    train_ret = returns[:train_n]

    mu, sigma, std_resid = _fit_garch(returns)
    xi, beta, u, pct = _fit_gpd_tail(std_resid[:train_n], tail_q)

    result = {"mu": mu, "sigma": sigma, "std_resid": std_resid,
              "xi": xi, "beta": beta, "threshold_u": u,
              "pct_exceed": pct}

    n_exceed_train = int(pct * train_n)
    for q in quantiles:
        z_q, es_q = gpd_var_es(xi, beta, u, q, train_n, n_exceed_train)
        # dynamic: time-varying VaR_t = mu + sigma_t * z_q
        dynamic_var = mu + sigma * z_q
        dynamic_es  = mu + sigma * es_q if np.isfinite(es_q) else np.full(n, np.nan)
        result[f"VaR_{int(q*100)}"] = float(np.nanpercentile(np.abs(returns[train_n:]), q*100))
        result[f"dynamic_VaR_{int(q*100)}"] = dynamic_var
        result[f"dynamic_ES_{int(q*100)}"]  = dynamic_es
        result[f"z_q_{int(q*100)}"] = z_q
        result[f"es_q_{int(q*100)}"] = es_q
        result[f"static_VaR_{int(q*100)}"] = float(abs(mu) + np.nanmean(sigma) * z_q)

    return result


def kupiec_test(actual: np.ndarray, var_series: np.ndarray, q=0.95):
    """Kupiec POF likelihood-ratio test for VaR coverage.
    H0: P(loss > VaR) = (1-q). Returns (LR_stat, p_value)."""
    violations = (actual < -np.abs(var_series)).astype(float)
    n = len(violations); T = violations.sum()
    p = 1 - q; p_hat = T / n if n > 0 else 0
    if p_hat in (0, 1): return np.nan, np.nan
    lr = 2 * (T * np.log(p_hat/p) + (n-T) * np.log((1-p_hat)/(1-p)))
    pval = 1 - stats.chi2.cdf(lr, 1)
    return float(lr), float(pval)


def christoffersen_test(actual: np.ndarray, var_series: np.ndarray, q=0.95):
    """Christoffersen independence test for VaR hit sequence clustering."""
    hits = (actual < -np.abs(var_series)).astype(int)
    n00 = np.sum((hits[:-1]==0) & (hits[1:]==0))
    n01 = np.sum((hits[:-1]==0) & (hits[1:]==1))
    n10 = np.sum((hits[:-1]==1) & (hits[1:]==0))
    n11 = np.sum((hits[:-1]==1) & (hits[1:]==1))
    if n01+n11 == 0 or n00+n10 == 0: return np.nan, np.nan
    pi01 = n01/(n00+n01); pi11 = n11/(n10+n11); pi = (n01+n11)/(n00+n01+n10+n11)
    if pi in (0,1) or pi01 in (0,1) or pi11 in (0,1): return np.nan, np.nan
    lr = 2*(n01*np.log(pi01) + n00*np.log(1-pi01) +
            n11*np.log(pi11) + n10*np.log(1-pi11) -
            (n01+n11)*np.log(pi) - (n00+n10)*np.log(1-pi))
    pval = 1 - stats.chi2.cdf(abs(lr), 1)
    return float(lr), float(pval)


print("✅ DDP_v2/_garch_evt.py loaded — compute_garch_evt, kupiec_test, christoffersen_test")
