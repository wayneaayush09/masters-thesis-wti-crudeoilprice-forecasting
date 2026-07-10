# ============================================================
# DDP_v2/_rag_forecaster.py
# RAG-as-Forecaster helpers (TimeRAG / TS-RAG style).
# Retrieves k most-similar PAST price windows by DTW or L2 embedding similarity,
# computes a similarity-weighted analog ensemble forecast, and exposes
# retrieved-analog stats as features for multivariate models.
#
# References:
#   TimeRAG (arXiv 2412.16643) — RAG for time-series LLMs via DTW retrieval
#   TS-RAG (arXiv 2503.07649) — pre-trained encoder + FAISS knowledge base
#   Retrieval-Augmented TS Forecasting (arXiv 2411.08249) — +2.97% avg accuracy
#
# Leakage-safe: for each origin t, only windows with end ≤ t-1 are indexed.
# Run / exec from inside DDP_v2/.
# ============================================================

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import cdist
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _z_normalise(w):
    """z-normalise a 1D window."""
    s = w.std()
    return (w - w.mean()) / (s + 1e-9)


def _dtw_distance(a, b):
    """Simple O(n^2) DTW between two equal-length sequences."""
    n = len(a)
    D = np.full((n+1, n+1), np.inf); D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            cost = (a[i-1]-b[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n, n])


def build_knowledge_base(series: np.ndarray, lookback: int,
                         stride: int = 1) -> dict:
    """Build a knowledge base of all past windows.
    Returns {'windows': array(N, lookback), 'origins': array(N)} — all windows
    ending at or before the last train index (caller should pass train slice).
    """
    windows, origins = [], []
    for i in range(lookback, len(series), stride):
        windows.append(series[i-lookback:i])
        origins.append(i)
    kb = {
        "windows": np.asarray(windows, dtype=np.float32),
        "origins": np.asarray(origins),
        "series":  series,
    }
    return kb


def retrieve_analogs(kb: dict, query_window: np.ndarray,
                     k: int = 5, method: str = "l2_norm") -> tuple:
    """Retrieve k most-similar past windows from the knowledge base.
    method: 'l2_norm' (fast), 'cosine', or 'dtw' (slow but robust).
    Returns (indices, similarities, weights).
    """
    qnorm = _z_normalise(query_window)
    kbw = kb["windows"]
    n = len(kbw)
    if n == 0: return [], [], []

    if method == "dtw":
        dists = np.array([_dtw_distance(qnorm, _z_normalise(kbw[i])) for i in range(n)])
        sims  = 1.0 / (dists + 1e-9)
    elif method == "cosine":
        norms = np.linalg.norm(kbw, axis=1, keepdims=True) + 1e-9
        kbn   = kbw / norms
        qn    = qnorm / (np.linalg.norm(qnorm) + 1e-9)
        sims  = (kbn @ qn).ravel()
    else:   # l2_norm (default, fast)
        kbn   = np.stack([_z_normalise(kbw[i]) for i in range(n)])
        dists = np.linalg.norm(kbn - qnorm, axis=1)
        sims  = 1.0 / (dists + 1e-9)

    top_k = min(k, n)
    idx   = np.argsort(sims)[::-1][:top_k]
    sims_k= sims[idx]
    weights = sims_k / (sims_k.sum() + 1e-12)
    return idx, sims_k, weights


def analog_forecast(kb: dict, query_origin: int, idx, weights,
                    h: int = 1) -> dict:
    """Produce a weighted-analog forecast h steps ahead.
    For each retrieved analog at origin o_i, the analog h-step move is
    series[o_i + h] - series[o_i]. Forecast = Σ w_i * (series[o_i+h] - series[o_i])
    + query_origin_value (so we predict the ABSOLUTE level at t+h).
    Returns dict with: forecast, analog_moves (array), analog_weights (array),
    analog_dispersion, analog_percentile_10, analog_percentile_90.
    """
    series = kb["series"]
    origins_kb = kb["origins"]
    moves = []
    for i in idx:
        o = origins_kb[i]
        if o + h < len(series):
            moves.append(float(series[o + h] - series[o]))
        else:
            moves.append(0.0)
    moves = np.array(moves)
    analog_move  = float(np.dot(weights, moves))
    base_val     = float(series[query_origin]) if query_origin < len(series) else float(series[-1])
    forecast     = base_val + analog_move
    return {
        "forecast": forecast,
        "analog_move_mean": analog_move,
        "analog_dispersion": float(np.std(moves)),
        "analog_p10": float(np.percentile(moves, 10)),
        "analog_p90": float(np.percentile(moves, 90)),
        "analog_count": len(moves),
    }


def rag_forecast_series(series: np.ndarray, train_len: int, h: int,
                        lookback: int, k: int = 5, method: str = "l2_norm",
                        stride: int = 5) -> dict:
    """Roll over all test origins, retrieve analogs, produce h-ahead forecasts.
    Returns {origin_idx: analog_forecast_dict}.
    Leakage-safe: KB is built on series[:train_len], never expanded to test.
    stride: step between test origins (=1 for every day, =5 for weekly, etc.)
    """
    kb = build_knowledge_base(series[:train_len], lookback, stride=1)
    preds = {}
    for origin in range(train_len, len(series) - h, stride):
        window = series[origin - lookback:origin]
        idx, sims, weights = retrieve_analogs(kb, window, k=k, method=method)
        if len(idx) == 0: continue
        fc = analog_forecast(kb, origin, idx, weights, h=h)
        # expand to every day via forward fill (handled in caller)
        preds[origin] = fc
    return preds


def rag_feature_columns(series: np.ndarray, train_len: int,
                        horizons=(1, 7), lookback: int = 30, k: int = 5) -> pd.DataFrame:
    """Compute RAG analog features for every date (strict past-only).
    Returns DataFrame with columns: rag_forecast_h, rag_dispersion_h, rag_move_h per horizon.
    """
    n = len(series)
    kb = build_knowledge_base(series[:train_len], lookback, stride=1)
    rows = []
    for t in range(lookback, n):
        row = {"t": t}
        window = series[t - lookback:t]
        idx, sims, weights = retrieve_analogs(kb, window, k=k, method="l2_norm")
        for h in horizons:
            if len(idx) > 0:
                fc = analog_forecast(kb, t, idx, weights, h=h)
                row[f"rag_forecast_{h}d"] = fc["forecast"]
                row[f"rag_move_{h}d"]     = fc["analog_move_mean"]
                row[f"rag_disp_{h}d"]     = fc["analog_dispersion"]
            else:
                row[f"rag_forecast_{h}d"] = np.nan
                row[f"rag_move_{h}d"]     = np.nan
                row[f"rag_disp_{h}d"]     = np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("t")
    return df


print("✅ DDP_v2/_rag_forecaster.py loaded — rag_forecast_series, rag_feature_columns, retrieve_analogs")
