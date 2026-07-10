# ════════════════════════════════════════════════════════════════════
#  FILE : _lh_runners.py
#  ROLE : LH (long-horizon) extrapolator runners for Track A components
# ════════════════════════════════════════════════════════════════════
#  WHAT — Provides run_lh_yu / run_lh_huang / run_lh_standalone:
#         top-level functions that reuse the whole-series WOA-VMD-FE /
#         VMD-EMD decompositions from _paper_helpers and swap the per-band
#         NN predictor for fast linear extrapolators (DLinear, NLinear,
#         Ridge, AR, Theta, ARIMA). Also wires disk-backed decomposition
#         caches for resume across runtime disconnects.
#
#  WHY  — The original _paper_helpers runs the NN (PatchTST) on each
#         decomposed band. For classical methods (AR, Theta), we want the
#         SAME whole-series decomposition but a much faster per-component
#         predictor (seconds vs minutes). _lh_runners reuses the cached
#         decomposition and swaps only the predictor function.
#         This lets A_components_linear/classical/hybrid produce apples-to-
#         apples comparisons with the NN Track A results at near-zero cost.
#
#  HOW  — Self-bootstraps _paper_helpers + _lh_forecasters into globals.
#         run_lh_yu:
#           1. Load log_price from df_features.pkl.
#           2. Check disk cache (cache/yu_decomp_{hk}.pkl) then in-memory.
#           3. Compute WOA-VMD-FE if neither exists; save to disk.
#           4. Call _predict_bands with the chosen extrapolator method.
#           5. save_paper_predictions → predictions_LHYu_{method}_{hk}.csv.
#         run_lh_huang: same but VMD+EMD, ~15 components.
#         run_lh_standalone: no decomposition, extrapolate raw log-price
#           directly (honest/leaky=False baseline).
#
#  READS    : df_features.pkl (log_price), cache/yu|huang_decomp_*.pkl
#  WRITES   : predictions_LHYu_{method}_{hk}.csv (+ LHHuang, LHStandalone)
#             eval_LHYu_{method}_{hk}.pkl
#             cache/yu_decomp_{hk}.pkl (if not already saved)
#  COMPUTE  : CPU-ONLY for AR/Theta/DLinear; slow CPU for ARIMA/ETS
#  PIPELINE : exec'd by A_components_linear/classical/hybrid.py (Stage 2)
#  RUNTIME  : ~1 min (AR/Theta per horizon)  ~15 min (ARIMA/ETS per horizon)
# ════════════════════════════════════════════════════════════════════

import pickle, time
import numpy as np

import os as _os
if "save_paper_predictions" not in globals():
    exec(open("_paper_helpers.py", encoding="utf-8").read())
if "forecast_component" not in globals():
    # self-contained DDP_v2: forecasters are flat here; fall back to the longhorizon_v2 path
    _fc = "_lh_forecasters.py" if _os.path.exists("_lh_forecasters.py") else "longhorizon_v2/_lh_forecasters.py"
    exec(open(_fc, encoding="utf-8").read())

_LH_LOOKBACK = {"1d": 30, "7d": 60, "90d": 180, "180d": 360}
_HORIZONS_H  = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}


def _load_logprice():
    with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
        df = pickle.load(f)
    return df["log_price"].values.astype(float), df.index


def _predict_bands(comps, names, train_len, h, lookback, method,
                   diagnose=False, sel_methods=None, **fc_kw):
    """Predict every component, sum into {target_idx: summed_band_value}.
    Returns (band_preds_by_origin, chosen_methods)."""
    band_preds, chosen = {}, {}
    for name, comp in zip(names, comps):
        comp = np.asarray(comp, float)
        if method == "select":
            preds, pick, _ = select_best_component(
                comp, train_len, h, lookback,
                methods=sel_methods or ("dlinear", "nlinear", "ridge", "ar", "theta"),
                **fc_kw)
            chosen[name] = pick
        else:
            preds = forecast_component(comp, train_len, h, lookback, method=method, **fc_kw)
            chosen[name] = method
        if diagnose and preds:
            idx = [k for k in preds if k < len(comp)]
            mae = float(np.mean([abs(preds[k] - comp[k]) for k in idx])) if idx else float("nan")
            print(f"    [diag] {name}: pred={chosen[name]:<8} band-MAE={mae:.5f} (n={len(idx)})")
        for k, v in preds.items():
            band_preds[k] = band_preds.get(k, 0.0) + v
    return band_preds, chosen


def run_lh_yu(hk, method="dlinear", model_tag=None, split_ratio=0.7,
              diagnose=False, sel_methods=None, **fc_kw):
    h = _HORIZONS_H[hk]; lookback = _LH_LOOKBACK[hk]
    tag = model_tag or method
    print("=" * 60)
    print(f"  LH-Yu (leaky WOA-VMD-FE) + [{method}]  —  {hk} (h={h})")
    print("=" * 60)
    log_price, dates = _load_logprice()
    # Try disk cache first — run_lh_yu is called independently of run_paper_yu so the
    # in-memory cache (_YU_DECOMP_CACHE) may be empty even if a disk pkl exists.
    # Same pattern as run_paper_yu in _paper_helpers.py; they share the same globals dict.
    if hk not in _YU_DECOMP_CACHE and globals().get("cache_load"):
        _disk = cache_load(f"yu_decomp_{hk}")
        if _disk is not None:
            _YU_DECOMP_CACHE[hk] = _disk
            print("  [Yu] loaded decomposition from disk cache (no recompute)")
    if hk in _YU_DECOMP_CACHE:
        (lf, mf, hf), (K_star, A_star) = _YU_DECOMP_CACHE[hk]
        print(f"  [Yu] reusing cached decomposition (K={K_star}, α={A_star})")
    else:
        t0 = time.time()
        (lf, mf, hf), (K_star, A_star) = whole_series_woa_vmd_fe(log_price)
        print(f"  [Yu] decomposition in {time.time()-t0:.1f}s")
        _YU_DECOMP_CACHE[hk] = ((lf, mf, hf), (K_star, A_star))
        if globals().get("cache_save"):
            cache_save(f"yu_decomp_{hk}", _YU_DECOMP_CACHE[hk])
    train_len = int(len(log_price) * split_ratio)
    band_preds, chosen = _predict_bands([lf, mf, hf], ["LF", "MF", "HF"],
                                        train_len, h, lookback, method,
                                        diagnose=diagnose, sel_methods=sel_methods, **fc_kw)
    return save_paper_predictions(band_preds, log_price, dates, h=h, hk=hk,
                                  model_name=f"LHYu_{tag}",
                                  extra_meta={"leaky": True, "method": method,
                                              "arch": f"LH_{tag}", "chosen": chosen,
                                              "K_star": K_star, "alpha_star": A_star,
                                              "protocol": "leaky_whole_series_decomposition"})


def run_lh_huang(hk, method="dlinear", model_tag=None, split_ratio=0.7, K=3, alpha=2000,
                 diagnose=False, sel_methods=None, fe_keep_quantile=None, **fc_kw):
    h = _HORIZONS_H[hk]; lookback = _LH_LOOKBACK[hk]
    tag = model_tag or method
    print("=" * 60)
    print(f"  LH-Huang (leaky VMD-EMD) + [{method}]  —  {hk} (h={h})")
    print("=" * 60)
    log_price, dates = _load_logprice()
    ckey = (hk, K, alpha)
    # Disk-cache check — same pattern as run_paper_huang in _paper_helpers.py.
    _disk_key = f"huang_decomp_{hk}_{K}_{alpha}"
    if ckey not in _HUANG_DECOMP_CACHE and globals().get("cache_load"):
        _disk = cache_load(_disk_key)
        if _disk is not None:
            _HUANG_DECOMP_CACHE[ckey] = _disk
            print("  [Huang] loaded decomposition from disk cache (no recompute)")
    if ckey in _HUANG_DECOMP_CACHE:
        components = _HUANG_DECOMP_CACHE[ckey]
        print(f"  [Huang] reusing cached decomposition ({len(components)} comps)")
    else:
        t0 = time.time()
        components = whole_series_vmd_emd(log_price, K=K, alpha=alpha)
        print(f"  [Huang] decomposition in {time.time()-t0:.1f}s ({len(components)} comps)")
        _HUANG_DECOMP_CACHE[ckey] = components
        if globals().get("cache_save"):
            cache_save(_disk_key, components)
    train_len = int(len(log_price) * split_ratio)
    # pad components to full length
    comps = []
    for c in components:
        c = np.asarray(c, float)
        if len(c) < len(log_price):
            c = np.concatenate([[c[0]], c])
        comps.append(c)
    names = [f"c{i+1:02d}" for i in range(len(comps))]

    # optional FE-drop: model only low-FE comps, predict high-FE comps as train-mean
    keep = set(range(len(comps)))
    if fe_keep_quantile is not None and 0 < fe_keep_quantile < 1:
        fes = np.array([fuzzy_entropy(c[:train_len]) for c in comps])
        fes = np.nan_to_num(fes, nan=np.nanmax(fes))
        thr = np.quantile(fes, fe_keep_quantile)
        keep = {i for i, fe in enumerate(fes) if fe <= thr}
        print(f"  [Huang] FE-drop: modeling {len(keep)}/{len(comps)} comps (FE<={thr:.4f})")

    band_preds, chosen = {}, {}
    for i, (name, comp) in enumerate(zip(names, comps)):
        if i in keep:
            sub, ch = _predict_bands([comp], [name], train_len, h, lookback, method,
                                     diagnose=diagnose, sel_methods=sel_methods, **fc_kw)
            chosen[name] = ch[name]
        else:
            cmean = float(np.mean(comp[:train_len]))
            sub = {o + h: cmean for o in range(train_len, len(comp) - h)}
            chosen[name] = "mean"
        for k, v in sub.items():
            band_preds[k] = band_preds.get(k, 0.0) + v
    return save_paper_predictions(band_preds, log_price, dates, h=h, hk=hk,
                                  model_name=f"LHHuang_{tag}",
                                  extra_meta={"leaky": True, "method": method,
                                              "arch": f"LH_{tag}", "chosen": chosen,
                                              "K": K, "alpha": alpha,
                                              "fe_keep_quantile": fe_keep_quantile,
                                              "protocol": "leaky_whole_series_decomposition"})


def run_lh_standalone(hk, method="dlinear", model_tag=None, split_ratio=0.7, **fc_kw):
    """No decomposition: direct extrapolator on raw log-price. Honest (leaky=False)."""
    h = _HORIZONS_H[hk]; lookback = _LH_LOOKBACK[hk]
    tag = model_tag or method
    print("=" * 60)
    print(f"  LH-Standalone (raw log-price, NO decomposition) + [{method}]  —  {hk} (h={h})")
    print("=" * 60)
    log_price, dates = _load_logprice()
    train_len = int(len(log_price) * split_ratio)
    preds = forecast_component(log_price, train_len, h, lookback, method=method, **fc_kw)
    return save_paper_predictions(preds, log_price, dates, h=h, hk=hk,
                                  model_name=f"LHStandalone_{tag}",
                                  extra_meta={"leaky": False, "method": method,
                                              "arch": f"LHstd_{tag}",
                                              "protocol": "standalone_no_decomposition"})


print("✅ longhorizon_v2/_lh_runners.py loaded")
