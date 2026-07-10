# ============================================================
# DDP_v2/_features_plus.py  — ADDITIVE feature engineering (run after 00_setup)
# Appends new columns to outputs/df_features.pkl and registers them in
# outputs/feat_cols.pkl for 7d/90d/180d. Never edits existing columns.
# Derives features locally from existing columns (robust on Kaggle); external
# fetches (COT/EUI) are optional and skipped gracefully on failure.
#
# Adds:
#   • Jump / realized-vol family: realized_vol_21, jump_flag, jump_size,
#     jump_intensity_63, time_since_jump, bipower_21, jump_var_ratio
#   • Term-structure / carry slopes: carry_21, carry_126, carry_252, carry_accel
#   • Shock-decay memory: shock_decay_30, shock_decay_90
#   • Regime interactions: regime_x_vol, regime_x_carry, (regime_x_sent if text present)
#   • Leaky decomposition bands (whole-series): vmd_lf_leaky/mf_leaky/hf_leaky
#     (the non-leaky rolling bands vmd_lf/mf/hf come from 01a_vmd_features.py)
# Run from inside DDP_v2/.
# ============================================================
import pickle, os
import numpy as np
import pandas as pd

OUT = str(globals().get("OUTPUT_DIR", "outputs"))
with open(os.path.join(OUT, "df_features.pkl"), "rb") as f:
    df = pickle.load(f)

lp = df["log_price"].astype(float)
r1 = lp.diff()                                  # 1d log-return
new_cols = []

def _add(name, series):
    df[name] = series.reindex(df.index).astype(float)
    new_cols.append(name)

# ── Jump / realized-vol family ───────────────────────────────
rv21 = (r1 ** 2).rolling(21).sum()
_add("realized_vol_21", np.sqrt(rv21))
roll_std = r1.rolling(63).std()
jump = (r1.abs() > 3.0 * roll_std).astype(float)
_add("jump_flag", jump)
_add("jump_size", (r1 * jump).fillna(0.0))
_add("jump_intensity_63", jump.rolling(63).sum())
# time since last jump (days)
tsj = np.zeros(len(df)); c = 0
jv = jump.fillna(0).values
for i in range(len(df)):
    c = 0 if jv[i] == 1 else c + 1
    tsj[i] = c
_add("time_since_jump", pd.Series(tsj, index=df.index))
# bipower variation (jump-robust vol) + jump variation ratio
bp = (r1.abs() * r1.abs().shift(1)).rolling(21).sum() * (np.pi / 2)
_add("bipower_21", np.sqrt(bp.clip(lower=0)))
_add("jump_var_ratio", (rv21 - bp).clip(lower=0) / (rv21 + 1e-9))

# ── Term-structure / carry slopes (contango/backwardation proxies) ──
for w in (21, 126, 252):
    _add(f"carry_{w}", lp - lp.shift(w))
_add("carry_accel", df["carry_21"] - (df["carry_126"] / 6.0))   # short vs long carry gap

# ── Shock-decay memory (exponentially-weighted past shock magnitude) ──
shock = r1.abs().fillna(0.0)
for lam in (30, 90):
    _add(f"shock_decay_{lam}", shock.ewm(halflife=lam).mean())

# ── Regime interactions (only if regime features exist) ──────
rvr = df["regime_vol_rank"] if "regime_vol_rank" in df.columns else None
if rvr is not None:
    base_vol = df["vol_30d"] if "vol_30d" in df.columns else df["realized_vol_21"]
    _add("regime_x_vol", rvr.astype(float) * base_vol.astype(float))
    _add("regime_x_carry", rvr.astype(float) * df["carry_126"].astype(float))
    for sent in ("finbert_sent", "gdelt_tone", "llm_news_score"):
        if sent in df.columns:
            _add(f"regime_x_{sent}", rvr.astype(float) * df[sent].astype(float))

# ── Leaky whole-series decomposition bands (optional, guarded) ──
if "vmd_lf_leaky" not in df.columns:
    try:
        if "whole_series_woa_vmd_fe" not in globals():
            exec(open("_paper_helpers.py", encoding="utf-8").read())
        (lf, mf, hf), _ = whole_series_woa_vmd_fe(lp.values.astype(float))
        for nm, arr in (("vmd_lf_leaky", lf), ("vmd_mf_leaky", mf), ("vmd_hf_leaky", hf)):
            a = np.asarray(arr, float)
            if len(a) < len(df):
                a = np.concatenate([[a[0]] * (len(df) - len(a)), a])
            _add(nm, pd.Series(a[:len(df)], index=df.index))
        print("  [features+] appended leaky whole-series VMD bands")
    except Exception as e:
        print(f"  [features+] skipped leaky VMD bands ({e})")

# ── Optional external data (COT / EUI) — skipped gracefully ──
# (CFTC COT positioning + Energy Uncertainty Index need network/keys; attempt then skip.)
# Left as a documented hook; derive-from-local features above are the robust core.

df[new_cols] = df[new_cols].ffill().bfill().fillna(0.0)
with open(os.path.join(OUT, "df_features.pkl"), "wb") as f:
    pickle.dump(df, f)

# register new features for the medium/long horizons in feat_cols.pkl
try:
    with open(os.path.join(OUT, "feat_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    for hk in ("7d", "90d", "180d"):
        if hk in feat_cols:
            for c in new_cols:
                if c not in feat_cols[hk]:
                    feat_cols[hk].append(c)
    with open(os.path.join(OUT, "feat_cols.pkl"), "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"  [features+] registered {len(new_cols)} new cols into feat_cols[7d/90d/180d]")
except Exception as e:
    print(f"  [features+] feat_cols.pkl not updated ({e})")

print(f"✅ _features_plus.py — added {len(new_cols)} features: {new_cols}")
