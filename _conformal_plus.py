# ============================================================
# DDP_v2/_conformal_plus.py
# Upgraded uncertainty estimation — beyond the fixed split-conformal in
# _forecast_template.py.
#
# Provides:
#   1. CQRInterval — Conformalized Quantile Regression:
#      trains quantile head, then calibrates lower/upper on validation,
#      giving sharper + adaptively heteroscedastic intervals.
#   2. HeteroscedasticConformal — scales conformal offsets by predicted σ
#      (if σ_head available from StudentT), producing wider bands when
#      the model is uncertain and narrower when confident.
#   3. MondrianConformal — per-regime (HMM state) conformal calibration;
#      splits calibration set by regime → regime-specific widths.
#   4. calibrate_and_save(model_name, hk) — loads an existing predictions CSV,
#      applies the best-available method, saves fused intervals.
# Run / exec from inside DDP_v2/.
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))


# ── 1. Conformalized Quantile Regression (CQR) ────────────────────────────────
class CQRInterval:
    """Split-CQR: calibrate {lower, upper} conformity scores from validation set.
    Given val residuals + lower/upper quantile predictions, outputs adjusted bands."""

    def __init__(self, alpha=0.10):
        self.alpha   = alpha    # marginal miscoverage rate (0.10 → 90% band)
        self.q_lo    = None
        self.q_hi    = None

    def calibrate(self, val_actual: np.ndarray,
                  val_pred_lo: np.ndarray, val_pred_hi: np.ndarray):
        """Compute conformity scores on val set.
        Score = max(pred_lo - actual, actual - pred_hi) — positive if not covered."""
        scores = np.maximum(val_pred_lo - val_actual, val_actual - val_pred_hi)
        n = len(scores)
        q = np.quantile(scores, 1 - self.alpha + 1/(n+1), interpolation="higher")
        self.q_lo = float(q)
        self.q_hi = float(q)

    def predict(self, pred_lo: np.ndarray, pred_hi: np.ndarray):
        """Apply calibrated offset to test quantile predictions."""
        if self.q_lo is None:
            return pred_lo, pred_hi
        return pred_lo - self.q_lo, pred_hi + self.q_hi


# ── 2. Heteroscedastic Conformal ──────────────────────────────────────────────
class HeteroscedasticConformal:
    """σ-scaled conformal: conformity score = |residual| / σ_pred.
    Calibrates a single scalar q on val, then apply val: [pred-q*σ, pred+q*σ].
    Gives narrower bands when the model is confident, wider when uncertain."""

    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self.q = None

    def calibrate(self, val_actual: np.ndarray, val_pred: np.ndarray,
                  val_sigma: np.ndarray):
        """val_sigma: predicted std (from StudentT head or GARCH-EVT σ)."""
        sigma = np.abs(val_sigma) + 1e-9
        scores = np.abs(val_actual - val_pred) / sigma
        n = len(scores)
        self.q = float(np.quantile(scores, self.coverage + 1/(n+1)))

    def predict(self, pred: np.ndarray, sigma: np.ndarray):
        if self.q is None: return pred, pred
        w = self.q * np.abs(sigma)
        return pred - w, pred + w


# ── 3. Mondrian (per-regime) Conformal ───────────────────────────────────────
class MondrianConformal:
    """Regime-conditional conformal calibration.
    Calibrates separate quantiles per HMM regime (0=calm, 1=trend, 2=vol, 3=crisis).
    Inference routes each test date to its predicted regime."""

    def __init__(self, coverage=0.90, n_regimes=4):
        self.coverage  = coverage
        self.n_regimes = n_regimes
        self.qs        = {}   # regime → calibrated quantile

    def calibrate(self, val_actual: np.ndarray, val_pred: np.ndarray,
                  val_regime: np.ndarray):
        scores = np.abs(val_actual - val_pred)
        for r in range(self.n_regimes):
            mask = (val_regime == r)
            if mask.sum() < 5:
                self.qs[r] = float(np.quantile(scores, self.coverage))
            else:
                self.qs[r] = float(np.quantile(scores[mask], self.coverage))

    def predict(self, pred: np.ndarray, regime: np.ndarray):
        lo = np.zeros_like(pred); hi = np.zeros_like(pred)
        for r in range(self.n_regimes):
            q = self.qs.get(r, self.qs.get(0, 0.0))
            mask = (regime == r)
            lo[mask] = pred[mask] - q
            hi[mask] = pred[mask] + q
        return lo, hi


# ── 4. Calibrate and save helper ──────────────────────────────────────────────
def calibrate_and_save(model_name: str, hk: str, method: str = "heteroscedastic",
                       val_frac: float = 0.25, coverage: float = 0.90):
    """Load predictions_{model_name}_{hk}.csv, apply upgraded conformal calibration,
    overwrite lower/upper columns, save as predictions_{model_name}_cqr_{hk}.csv.

    method: 'split' (standard), 'cqr', 'heteroscedastic', 'mondrian'
    """
    pred_path = OUTPUT_DIR / f"predictions_{model_name}_{hk}.csv"
    if not pred_path.exists():
        print(f"  [conformal+] {pred_path} not found — skipping.")
        return

    df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    if "actual" not in df.columns or "pred" not in df.columns:
        return

    n = len(df); n_val = max(10, int(n * val_frac))
    val_df, test_df = df.iloc[:n_val], df.iloc[n_val:]

    actual_val = val_df["actual"].values
    pred_val   = val_df["pred"].values

    if method == "heteroscedastic" and "sigma_logret" in df.columns:
        cal = HeteroscedasticConformal(coverage=coverage)
        # convert log-return sigma to price sigma proxy
        origin_p = df.get("origin_price", df["pred"])
        sigma_val = (val_df["sigma_logret"].fillna(0.02).values *
                     origin_p.iloc[:n_val].values)
        cal.calibrate(actual_val, pred_val, sigma_val)
        sigma_test = (test_df["sigma_logret"].fillna(0.02).values *
                      origin_p.iloc[n_val:].values)
        lo, hi = cal.predict(test_df["pred"].values, sigma_test)
    elif method == "mondrian":
        # Load regime from df_features
        try:
            with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
                feat = pickle.load(f)
            regime_val  = feat.loc[val_df.index,  "regime_state"].fillna(0).astype(int).values if "regime_state" in feat else np.zeros(n_val, int)
            regime_test = feat.loc[test_df.index, "regime_state"].fillna(0).astype(int).values if "regime_state" in feat else np.zeros(n-n_val, int)
        except Exception:
            regime_val = np.zeros(n_val, int); regime_test = np.zeros(n-n_val, int)
        cal = MondrianConformal(coverage=coverage)
        cal.calibrate(actual_val, pred_val, regime_val)
        lo, hi = cal.predict(test_df["pred"].values, regime_test)
    else:
        # CQR or standard split
        lo_val = val_df.get("lower", val_df["pred"]).values
        hi_val = val_df.get("upper", val_df["pred"]).values
        cal = CQRInterval(alpha=1-coverage)
        cal.calibrate(actual_val, lo_val, hi_val)
        lo_test = test_df.get("lower", test_df["pred"]).values
        hi_test = test_df.get("upper", test_df["pred"]).values
        lo, hi = cal.predict(lo_test, hi_test)

    out = test_df.copy()
    out["lower"] = lo; out["upper"] = hi
    out_path = OUTPUT_DIR / f"predictions_{model_name}_{method}_{hk}.csv"
    out.to_csv(out_path)
    # coverage
    coverage_actual = float(np.mean((out["actual"] >= lo) & (out["actual"] <= hi)))
    print(f"  [conformal+] {model_name} {hk} {method}: "
          f"actual coverage={coverage_actual:.3f} (target={coverage:.2f})")
    return out


print("✅ DDP_v2/_conformal_plus.py loaded — CQRInterval, HeteroscedasticConformal, MondrianConformal, calibrate_and_save")
