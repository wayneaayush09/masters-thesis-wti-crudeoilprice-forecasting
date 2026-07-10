# ============================================================
# DDP_v2/_fusion.py
# Post-hoc fusion of text/RAG/agent/EVT signals into point forecasts.
# Currently text sentiment (10_text_features.py) and the agentic RAG direction
# (12_agentic_rag.py) are merged as *features* but never fused INTO the final
# price forecast.  This module does that.
#
# Three fusion strategies (all optional, composable):
#   1. sentiment_shift(preds_df, features_df, alpha=0.15)
#      → Shift the log-return prediction by alpha × finbert_sent × sigma
#        (scale by the model's own uncertainty so small alpha is safe).
#   2. agent_ensemble(preds_df, agent_df, beta=0.1)
#      → On anomaly dates where the agent fires, blend: final = (1-β)×model + β×agent_direction
#        weighted by agent hit-rate on recent 50 anomalies (self-calibrating).
#   3. evt_clip(preds_df, evt_pkl_path, coverage=0.95)
#      → Clip log-return predictions to [VaR_5, VaR_95] from the fitted GPD tail,
#        avoiding implausible point forecasts beyond EVT-estimated extremes.
#
# Usage:
#   exec(open('_fusion.py', encoding="utf-8").read())
#   preds_fused = fuse_signals(preds_df, hk="7d")
# Run from inside DDP_v2/.
# ============================================================

import os, pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))


def _load_pkl(p, default=None):
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default


def _load_text_features():
    p = OUTPUT_DIR / "text_features_daily.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _load_agent_signals():
    p = OUTPUT_DIR / "agentic_rag_signal.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def sentiment_shift(preds_df: pd.DataFrame, hk: str,
                    alpha: float = 0.15,
                    sent_col: str = "crudebert_sent") -> pd.DataFrame:
    """Shift model log-return prediction by alpha × sentiment × |pred_sigma|.
    If crudebert_sent unavailable, falls back to finbert_sent, then gdelt_tone.
    Args:
        preds_df: predictions CSV as loaded (cols: pred, actual, pred_logret, sigma_logret, …)
        hk: horizon key ("7d" etc.) — used only for logging
        alpha: shift strength [0, 1]; 0.15 is conservative
    Returns: copy of preds_df with pred_logret / pred (price) adjusted.
    """
    txt = _load_text_features()
    if txt is None:
        return preds_df
    # choose best available sentiment column
    for col in [sent_col, "finbert_sent", "cb_polarity", "gdelt_tone"]:
        if col in txt.columns:
            sent_col = col; break
    else:
        return preds_df

    out = preds_df.copy()
    out.index = pd.to_datetime(out.index)
    joined = out.join(txt[[sent_col]].rename(columns={sent_col: "_sent"}), how="left")
    joined["_sent"] = joined["_sent"].fillna(0).clip(-1, 1)

    if "sigma_logret" in out.columns:
        sigma = out["sigma_logret"].fillna(out["sigma_logret"].median()).clip(lower=0.001)
    else:
        sigma = pd.Series(0.02, index=out.index)  # 2% default uncertainty

    shift = alpha * joined["_sent"] * sigma
    if "pred_logret" in out.columns:
        out["pred_logret"] = out["pred_logret"] + shift.values
        # recompute price from shifted logret if origin_price available
        if "origin_price" in out.columns:
            out["pred"] = out["origin_price"] * np.exp(out["pred_logret"])
        else:
            # adjust price multiplicatively
            out["pred"] = out["pred"] * np.exp(shift.values)
    else:
        out["pred"] = out["pred"] * np.exp(shift.values)

    n_adj = (shift.abs() > 1e-6).sum()
    print(f"  [fusion/sentiment] {hk}: adjusted {n_adj}/{len(out)} predictions "
          f"(col={sent_col}, alpha={alpha})")
    return out


def agent_ensemble(preds_df: pd.DataFrame, hk: str,
                   beta: float = 0.10,
                   min_hitrate: float = 0.52) -> pd.DataFrame:
    """On flagged anomaly dates where agentic_rag_signal.csv fired, blend
    the model forecast with the agent direction signal:
        final_logret = (1-w) * model_logret + w * agent_logret
    where w = beta × (agent_hitrate - 0.5) / 0.5 → 0 if hitrate <= 0.5.
    The agent direction is converted to log-return via direction × 0.01 (1% move).
    """
    agent = _load_agent_signals()
    if agent is None or "direction" not in agent.columns:
        return preds_df

    out = preds_df.copy()
    out.index = pd.to_datetime(out.index)

    # compute recent hit-rate (last 50 anomalies)
    if "correct" in agent.columns:
        recent = agent.dropna(subset=["correct"]).tail(50)
        hitrate = float(recent["correct"].mean()) if len(recent) > 0 else 0.5
    else:
        hitrate = 0.5

    if hitrate <= min_hitrate:
        print(f"  [fusion/agent] {hk}: agent hitrate={hitrate:.3f} below min {min_hitrate}, skipping.")
        return out

    w = beta * (hitrate - 0.5) / 0.5

    # get agent dates in the forecast index
    common = out.index.intersection(agent.index)
    if len(common) == 0:
        return out

    agent_logret = agent.loc[common, "direction"] * 0.01  # 1% per unit direction
    model_col = "pred_logret" if "pred_logret" in out.columns else None

    n_adj = 0
    for dt in common:
        if model_col:
            ml = float(out.at[dt, model_col])
            al = float(agent_logret.at[dt])
            out.at[dt, model_col] = (1 - w) * ml + w * al
        # adjust price
        if "origin_price" in out.columns:
            out.at[dt, "pred"] = float(out.at[dt, "origin_price"]) * np.exp(out.at[dt, model_col] if model_col else 0)
        n_adj += 1

    print(f"  [fusion/agent] {hk}: blended {n_adj} anomaly dates "
          f"(hitrate={hitrate:.3f}, w={w:.3f})")
    return out


def evt_clip(preds_df: pd.DataFrame, hk: str) -> pd.DataFrame:
    """Clip log-return predictions to [VaR_5, VaR_95] from the fitted GPD.
    Prevents implausible point forecasts in edge cases.
    """
    evt_path = OUTPUT_DIR / f"risk_evt_{hk}.pkl"
    evt = _load_pkl(evt_path)
    if evt is None or "VaR_95" not in evt:
        return preds_df

    var95 = float(evt["VaR_95"])   # positive → upper bound
    var5  = -var95                  # symmetric clip (GPD is one-sided so approximate)

    out = preds_df.copy()
    if "pred_logret" in out.columns:
        n_hi = (out["pred_logret"] > var95).sum()
        n_lo = (out["pred_logret"] < var5).sum()
        out["pred_logret"] = out["pred_logret"].clip(lower=var5, upper=var95)
        if "origin_price" in out.columns:
            out["pred"] = out["origin_price"] * np.exp(out["pred_logret"])
        if n_hi + n_lo > 0:
            print(f"  [fusion/evt_clip] {hk}: clipped {n_hi} upper + {n_lo} lower predictions "
                  f"(VaR_95=±{var95:.4f})")
    return out


def fuse_signals(preds_df: pd.DataFrame, hk: str,
                 use_sentiment: bool = True,
                 use_agent: bool = True,
                 use_evt_clip: bool = True,
                 alpha: float = 0.15,
                 beta: float = 0.10) -> pd.DataFrame:
    """Apply all enabled fusion steps in order: sentiment → agent → evt_clip.
    Each step is additive and gracefully degrades if the required input file is missing.
    """
    print(f"\n[fusion] {hk}: applying signal fusion pipeline …")
    df = preds_df
    if use_sentiment:
        df = sentiment_shift(df, hk, alpha=alpha)
    if use_agent:
        df = agent_ensemble(df, hk, beta=beta)
    if use_evt_clip:
        df = evt_clip(df, hk)
    return df


def fuse_and_save(model_name: str, hk: str, **kwargs) -> pd.DataFrame:
    """Load predictions_{model_name}_{hk}.csv, fuse, overwrite with fused version,
    and save the fused copy as predictions_{model_name}_fused_{hk}.csv.
    PatchTST uses no model suffix (predictions_{hk}.csv); all others use _{model_name}.
    """
    _sfx = "" if model_name == "PatchTST" else f"_{model_name}"
    p = OUTPUT_DIR / f"predictions{_sfx}_{hk}.csv"
    if not p.exists():
        print(f"  [fusion] {p} not found — skipping.")
        return None
    preds = pd.read_csv(p, index_col=0, parse_dates=True)
    fused = fuse_signals(preds, hk, **kwargs)
    out_p = OUTPUT_DIR / f"predictions{_sfx}_fused_{hk}.csv"
    fused.to_csv(out_p)
    print(f"  [fusion] saved {out_p}")
    return fused


print("✅ DDP_v2/_fusion.py loaded  (fuse_signals, fuse_and_save, sentiment_shift, agent_ensemble, evt_clip)")
