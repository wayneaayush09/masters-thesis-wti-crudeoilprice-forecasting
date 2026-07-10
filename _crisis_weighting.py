# ============================================================
# DDP_v2/_crisis_weighting.py
# Crisis-stratified sample weighting + regime-conditional helpers.
# The HONEST walk-forward pipeline (Track B) currently treats all training
# dates equally. GFC/OPEC_War/COVID are in-sample for early splits and
# under-represented. This module provides:
#   1. crisis_sample_weights(df, hk) → per-row weights for weighted sampler
#   2. get_crisis_flags(df) → binary bool series for crisis periods
#   3. regime_conditional_split(df, regime_col) → dict of per-regime dataframes
# These are ADDITIVES: the base _forecast_template.py uses them optionally.
# Run from inside DDP_v2/.
# ============================================================

import numpy as np
import pandas as pd

# ── crisis period definitions (same as base pipeline) ────────
CRISIS_WINDOWS = {
    "GFC":       ("2008-09-01", "2009-06-30"),
    "OPEC_War":  ("2014-06-01", "2016-03-31"),
    "COVID":     ("2020-02-01", "2020-09-30"),
}

# Multipliers applied to training sample weights during each crisis window.
CRISIS_WEIGHT  = 3.0    # crisis dates get 3× weight
VOLATILE_WEIGHT = 1.5   # HMM regime_vol_rank > 0.66 gets 1.5×
NORMAL_WEIGHT  = 1.0    # background weight

def get_crisis_flags(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True on dates inside any defined crisis window."""
    flags = pd.Series(False, index=df.index)
    for name, (s, e) in CRISIS_WINDOWS.items():
        flags |= (df.index >= s) & (df.index <= e)
    return flags


def crisis_sample_weights(df: pd.DataFrame, hk: str,
                          crisis_w: float = CRISIS_WEIGHT,
                          volatile_w: float = VOLATILE_WEIGHT) -> np.ndarray:
    """Per-sample weight array (shape = len(df)) for use in WeightedRandomSampler.
    Weights are un-normalized (the DataLoader normalizes implicitly).

    Strategy:
      • Dates in a crisis window → crisis_w
      • Dates with HMM regime_vol_rank > 0.66 (volatile/crisis regime) → volatile_w
      • Everything else → 1.0
    At long horizons (90d/180d) the long-lookback windows overlap crises more;
    upweighting helps the model see more crisis-level dynamics.
    """
    n = len(df)
    w = np.ones(n, dtype=np.float32)

    # volatile regime
    if "regime_vol_rank" in df.columns:
        vol_mask = df["regime_vol_rank"].fillna(0).values > 0.66
        w[vol_mask] = volatile_w

    # crisis windows (override volatile weight)
    for name, (s, e) in CRISIS_WINDOWS.items():
        mask = (df.index >= s) & (df.index <= e)
        w[mask.values] = crisis_w

    return w


def regime_conditional_split(df: pd.DataFrame,
                              regime_col: str = "regime_state",
                              n_states: int = 4) -> dict:
    """Return a dict {regime_state: subset_df} for separate-head training.
    States 0–3 per the base HMM (0=calm, 3=crisis).
    Missing regime → falls back to the full df under key 'all'.
    """
    if regime_col not in df.columns:
        return {"all": df}
    splits = {}
    for s in range(n_states):
        sub = df[df[regime_col] == s]
        if len(sub) >= 30:   # need at least 30 rows to train
            splits[s] = sub
    if not splits:
        splits["all"] = df
    return splits


def augment_features_with_crisis(df: pd.DataFrame) -> pd.DataFrame:
    """Add crisis-interaction features to the feature panel (additive to _features_plus):
    - shock_memory: exponentially decayed magnitude of past large moves (λ=30d)
    - vol_crisis_interact: vol_30d × crisis_flag (captures 'bad news in stressed market')
    - regime_crisis_interact: regime_vol_rank × crisis_flag
    These help models distinguish 'volatile but recovering' from 'volatile and crashing'.
    """
    df = df.copy()
    if "log_ret_1d" in df.columns:
        returns = df["log_ret_1d"].fillna(0).values
        shock_mem = np.zeros(len(returns))
        for i in range(1, len(returns)):
            shock_mem[i] = abs(returns[i]) + 0.97 * shock_mem[i - 1]  # λ=30d half-life
        df["shock_memory_30d"] = shock_mem.astype(np.float32)

    crisis_flag = get_crisis_flags(df).astype(float)
    if "vol_30d" in df.columns:
        df["vol_crisis_interact"] = (df["vol_30d"].fillna(0) * crisis_flag).astype(np.float32)
    if "regime_vol_rank" in df.columns:
        df["regime_crisis_interact"] = (df["regime_vol_rank"].fillna(0) * crisis_flag).astype(np.float32)

    return df


print("✅ DDP_v2/_crisis_weighting.py loaded")
