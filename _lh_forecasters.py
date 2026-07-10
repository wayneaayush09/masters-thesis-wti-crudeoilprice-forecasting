# ════════════════════════════════════════════════════════════════════
#  FILE : _lh_forecasters.py
#  ROLE : Per-component extrapolators for the Track A decomposition pipeline
# ════════════════════════════════════════════════════════════════════
#  WHAT — Provides `forecast_component` and `select_best_component`:
#         single-function entry points that take one VMD/EMD band (full
#         array, log-price scale) + a train_len/horizon/lookback and return
#         a dict {target_idx → predicted_band_value}. Methods available:
#         dlinear, nlinear, ridge, ar, theta, arima, ets, select.
#         Also exposes `select_best_component` which grid-searches methods
#         on the validation window and picks the lowest-MAE one.
#
#  WHY  — The paper's whole-series VMD/EMD components are smooth and
#         future-informed (lookahead leakage by design). The right predictor
#         for such a component is a DIRECT EXTRAPOLATOR (linear map, AR,
#         Theta) — not a mean-reverting neural net. These forecasters run
#         in numpy/sklearn with zero GPU cost, producing results at ~1/100th
#         the compute time of a PatchTST per-band training pass.
#         They are the engine behind _lh_runners.run_lh_yu/huang and are
#         also usable standalone for the `LHStandalone` (no-decomp) baseline.
#
#  HOW  — Unified interface: every method routes through `forecast_component`.
#         For each origin t in [train_len, N-h]:
#           1. Extract lookback window: comp[t-lookback : t].
#           2. Fit the model (OLS / AR / statsmodels / etc.) on this window.
#           3. Predict h steps ahead; record as {t+h: prediction}.
#         `select_best_component` runs all methods on a held-out validation
#         slice and returns the one with lowest MAE (+ the full score dict).
#         `fuzzy_entropy` is also re-exposed for use in _lh_runners'
#         fe_keep_quantile filtering.
#
#  READS    : (no files — pure in-memory computation)
#  WRITES   : (no files — returns Python dicts to caller)
#  COMPUTE  : CPU-ONLY; AR/DLinear/Ridge/Theta < 1 min per horizon
#             ARIMA/ETS: ~5–15 min per horizon (auto-fitting overhead)
#  PIPELINE : exec'd by _lh_runners.py (Stage 2 Track A); loaded via globals
#  RUNTIME  : < 1 s per band per origin (DLinear/Ridge); < 1 s (AR/Theta)
# ════════════════════════════════════════════════════════════════════

import numpy as np

try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None


# ── design-matrix helpers ────────────────────────────────────
def _train_xy(series, lookback, h, train_len):
    """Direct (window -> value@h) supervised pairs from the TRAIN slice only."""
    X, y = [], []
    for i in range(lookback, train_len - h):
        X.append(series[i - lookback:i]); y.append(series[i + h])
    if not X:
        return np.empty((0, lookback)), np.empty((0,))
    return np.asarray(X, float), np.asarray(y, float)


def _test_windows(series, lookback, h, train_len):
    """Windows + target indices for every test origin."""
    origins = range(train_len, len(series) - h)
    X = np.asarray([series[o - lookback:o] for o in origins], float)
    tgt = np.asarray([o + h for o in origins])
    return X, tgt


def _moving_avg_rows(X, k=25):
    """Row-wise centered moving average (trend) of each lookback window."""
    if k >= X.shape[1]:
        k = max(3, X.shape[1] // 2)
    kernel = np.ones(k) / k
    pad = k // 2
    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="edge")
    trend = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, Xp)
    return trend[:, :X.shape[1]]


# ── M1: linear direct maps (DLinear / NLinear / plain Ridge) ──
def _ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
    if Ridge is not None:
        m = Ridge(alpha=alpha).fit(Xtr, ytr); return m.predict(Xte)
    # numpy least-squares fallback
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef


def dlinear_component(comp, train_len, h, lookback, **kw):
    """DLinear: split each window into trend(moving-avg)+seasonal, linear map each, sum."""
    Xtr, ytr = _train_xy(comp, lookback, h, train_len)
    if len(Xtr) < lookback + 5:
        return {}
    Xte, tgt = _test_windows(comp, lookback, h, train_len)
    Ttr, Tte = _moving_avg_rows(Xtr), _moving_avg_rows(Xte)
    Str, Ste = Xtr - Ttr, Xte - Tte
    pred = _ridge_fit_predict(Ttr, ytr, Tte) * 0.0  # placeholder shape
    # trend maps to y; seasonal maps to residual-of-trend-fit
    yt = _ridge_fit_predict(Ttr, ytr, Tte)          # trend contribution -> full target
    # seasonal contributes the deviation; fit seasonal->(y - trend_fit_on_train)
    ytr_trendfit = _ridge_fit_predict(Ttr, ytr, Ttr)
    ys = _ridge_fit_predict(Str, ytr - ytr_trendfit, Ste)
    pred = yt + ys
    return {int(t): float(p) for t, p in zip(tgt, pred)}


def nlinear_component(comp, train_len, h, lookback, **kw):
    """NLinear: subtract last value of window, linear map the residual, add it back.
    Removes the level/trend so the map only learns the h-step *change* -> no cap."""
    Xtr, ytr = _train_xy(comp, lookback, h, train_len)
    if len(Xtr) < lookback + 5:
        return {}
    Xte, tgt = _test_windows(comp, lookback, h, train_len)
    last_tr, last_te = Xtr[:, -1:], Xte[:, -1:]
    pred = _ridge_fit_predict(Xtr - last_tr, ytr - last_tr.ravel(), Xte - last_te) + last_te.ravel()
    return {int(t): float(p) for t, p in zip(tgt, pred)}


def ridge_component(comp, train_len, h, lookback, alpha=1.0, **kw):
    """Plain direct linear map window->value@h."""
    Xtr, ytr = _train_xy(comp, lookback, h, train_len)
    if len(Xtr) < lookback + 5:
        return {}
    Xte, tgt = _test_windows(comp, lookback, h, train_len)
    pred = _ridge_fit_predict(Xtr, ytr, Xte, alpha=alpha)
    return {int(t): float(p) for t, p in zip(tgt, pred)}


# ── M2: classical extrapolators ──────────────────────────────
def ar_component(comp, train_len, h, lookback, p=10, **kw):
    """AR(p) fit on the train slice, recursive h-step forecast from each origin."""
    s = np.asarray(comp, float)
    # fit AR(p) with intercept on train
    X, y = [], []
    for i in range(p, train_len):
        X.append(s[i - p:i]); y.append(s[i])
    if len(X) < p + 5:
        return {}
    A = np.hstack([np.asarray(X), np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    w, b = coef[:-1], coef[-1]
    preds = {}
    for o in range(train_len, len(s) - h):
        hist = list(s[o - p:o])
        for _ in range(h):
            nxt = float(np.dot(w, hist[-p:]) + b)
            hist.append(nxt)
        preds[o + h] = hist[-1]
    return preds


def theta_component(comp, train_len, h, lookback, **kw):
    """Lightweight Theta: blend a linear-trend extrapolation of the lookback window
    with a simple-exponential-smoothing level. Neither mean-reverts."""
    s = np.asarray(comp, float)
    preds = {}
    t = np.arange(lookback)
    tbar = t.mean(); tden = ((t - tbar) ** 2).sum() + 1e-12
    alpha = 0.2
    for o in range(train_len, len(s) - h):
        w = s[o - lookback:o]
        # linear trend (theta=0 line)
        b = ((t - tbar) * (w - w.mean())).sum() / tden
        a = w.mean() - b * tbar
        trend_fc = a + b * (lookback - 1 + h)
        # SES level (theta=2 line proxy)
        lvl = w[0]
        for x in w[1:]:
            lvl = alpha * x + (1 - alpha) * lvl
        ses_fc = lvl + 0.5 * b * h
        preds[o + h] = float(0.5 * trend_fc + 0.5 * ses_fc)
    return preds


def _statsmodels_component(comp, train_len, h, lookback, kind="arima", stride=10, fit_timeout=60, **kw):
    """ARIMA / ETS via statsmodels, strided over test origins (forward-filled).
    Strided because per-origin refit is O(N) -> stride keeps it tractable.
    fit_timeout: seconds per fit before skipping (prevents hangs on pathological series)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
    try:
        if kind == "arima":
            from statsmodels.tsa.arima.model import ARIMA as _M
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as _M
    except Exception:
        return {}
    s = np.asarray(comp, float)
    origins = list(range(train_len, len(s) - h))
    sampled = origins[::stride]
    raw = {}
    _skipped = 0
    for o in sampled:
        hist = s[max(0, o - 1000):o]   # cap history for speed
        def _fit(_hist=hist):
            if kind == "arima":
                fit = _M(_hist, order=(5, 1, 0)).fit(method_kwargs={"warn_convergence": False})
                return float(fit.forecast(h)[-1])
            else:
                fit = _M(_hist, trend="add", damped_trend=True,
                         initialization_method="estimated").fit()
                return float(fit.forecast(h)[-1])
        try:
            with ThreadPoolExecutor(max_workers=1) as _ex:
                fc = _ex.submit(_fit).result(timeout=fit_timeout)
            raw[o] = fc
        except _FutureTimeout:
            raw[o] = float(hist[-1]); _skipped += 1
        except Exception:
            raw[o] = float(hist[-1])
    if _skipped:
        print(f"    [warn] {kind}: {_skipped}/{len(sampled)} fits timed out (>{fit_timeout}s), used last value")
    # forward-fill to every origin
    preds, last = {}, None
    si = 0
    for o in origins:
        if si < len(sampled) and o >= sampled[si]:
            last = raw.get(sampled[si], last); si += 1
        if last is not None:
            preds[o + h] = last
    return preds


def arima_component(comp, train_len, h, lookback, stride=10, **kw):
    return _statsmodels_component(comp, train_len, h, lookback, kind="arima", stride=stride)


def ets_component(comp, train_len, h, lookback, stride=10, **kw):
    return _statsmodels_component(comp, train_len, h, lookback, kind="ets", stride=stride)


# ── dispatcher + per-component model selection (M3) ──────────
_METHODS = {
    "dlinear": dlinear_component, "nlinear": nlinear_component, "ridge": ridge_component,
    "ar": ar_component, "theta": theta_component,
    "arima": arima_component, "ets": ets_component,
}


def forecast_component(comp, train_len, h, lookback, method="dlinear", **kw):
    comp = np.asarray(comp, float)
    fn = _METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown method '{method}'. choices: {list(_METHODS)}")
    return fn(comp, train_len, h, lookback, **kw)


def _val_mae(comp, train_len, h, lookback, method, val_frac=0.1, **kw):
    """Fit on the first (1-val_frac) of TRAIN, score band-MAE on the val tail of TRAIN."""
    comp = np.asarray(comp, float)
    inner = int(train_len * (1 - val_frac))
    if inner < lookback + h + 5:
        return np.inf
    fn = _METHODS[method]
    # predict the val region of train: treat `inner` as the inner train_len, but stop at train_len
    full = comp[:train_len]
    preds = fn(full, inner, h, lookback, **kw)
    errs = [abs(preds[k] - full[k]) for k in preds if k < train_len]
    return float(np.mean(errs)) if errs else np.inf


def select_best_component(comp, train_len, h, lookback,
                          methods=("dlinear", "nlinear", "ridge", "ar", "theta"),
                          val_frac=0.1, **kw):
    """Pick the per-component method with the lowest validation band-MAE, then
    refit on the full train and predict the test set. Returns (preds, chosen)."""
    scores = {m: _val_mae(comp, train_len, h, lookback, m, val_frac=val_frac, **kw)
              for m in methods}
    chosen = min(scores, key=scores.get)
    preds = forecast_component(comp, train_len, h, lookback, method=chosen, **kw)
    return preds, chosen, scores


print("✅ longhorizon_v2/_lh_forecasters.py loaded")
