# ════════════════════════════════════════════════════════════════════
#  FILE : 00_setup_data_features.py
#  ROLE : Stage 0 — one-time data build; run first in every session
# ════════════════════════════════════════════════════════════════════
#  WHAT — Installs dependencies, loads WTI crude-oil price history
#         (1986–2026), fetches exogenous data (yfinance equity/FX/vol
#         indices, FRED macroeconomic series, CFTC COT positioning,
#         Google Trends, EIA storage/production), engineers 100+ features,
#         detects HMM market regimes, and emits the master feature store.
#         Also generates 5 Plotly EDA figures (price overview, rolling
#         volatility, ACF/PACF, regime timeline, feature correlations).
#
#  WHY  — Every downstream model — Track A leaky (A_yu_*.py, A_huang_*.py),
#         Track B honest (B_wf_*.py), risk (07_risk_evt.py), ensemble
#         (C_ensemble.py), and analysis (06_analysis_plots.py) — reads
#         exactly these pickles. Getting the feature build right once
#         here means no model needs to handle missing data, regime labels,
#         or winsorization itself. The HMM regime labels drive
#         B_regime_conditional.py and the regime-MAE breakdown in analysis.
#
#  HOW  — Execution sequence:
#           1. Package install (vmdpy, cot_reports, yfinance, pmdarima,
#              plotly, kaleido, imageio — silent if already present).
#           2. Load WTI OHLC CSV → log-return series → winsorize tails.
#           3. Fetch exogenous panel (yfinance + FRED + COT + Trends)
#              with per-series try/except + zero-fallback (pipeline never
#              breaks on a failed fetch; fallback is documented in output).
#           4. Align to WTI business-day index via ffill; COT release
#              is lagged by publication delay (Tue snapshot → Fri release)
#              so no future information leaks through the COT channel.
#           5. Feature engineering: momentum/carry/volatility/HMM regimes/
#              COT net positioning / term-structure slope / interaction terms.
#           6. Build per-horizon feature sets (BASE / MED / LONG) reflecting
#              which signals are useful at 1d, 7d, 90d, 180d.
#           7. Save pickles; emit 5 Plotly EDA figures to figures/.
#
#  READS    : WTI OHLC CSV (local file, path in WTI_CSV_PATH),
#             network: yfinance, FRED, CFTC, Google Trends (all optional)
#  WRITES   : df_features.pkl, wti_raw.pkl, wti_wins.pkl, feat_cols.pkl,
#             extreme_events.pkl, regimes.pkl (all in OUTPUT_DIR)
#             figures/eda_price_overview.html|.png
#             figures/eda_rolling_vol.html|.png
#             figures/eda_acf_pacf.html|.png
#             figures/eda_regime_timeline.html|.png
#             figures/eda_feature_corr.html|.png
#  COMPUTE  : CPU-ONLY; HMM fit ~30 s; total ~5–15 min (network latency)
#  PIPELINE : Stage 0 — prerequisite for every other stage
#  RUNTIME  : ~5 min (cached) → ~15 min (fresh network fetch)
# ════════════════════════════════════════════════════════════════════

import subprocess, sys
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU detected"); print(result.stdout[:300])
    else:
        print("⚠️  nvidia-smi found but no GPU detected")
except FileNotFoundError:
    # nvidia-smi not present — likely a TPU instance
    try:
        import torch_xla.core.xla_model as xm
        print(f"✅ TPU detected: {xm.xla_device()}")
    except ImportError:
        print("⚠️  No GPU or TPU detected — running on CPU")
        print("    For GPU : Runtime > Change runtime type > T4 GPU")
        print("    For TPU : Runtime > Change runtime type > TPU v5e")
        print("    TPU install: !pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")

# ── INSTALL ──────────────────────────────────────────────────
packages = ["numpy>=1.17.3,<2.0",           # pin <2.0: sklearn/pandas ABI break in numpy 2.x
            "yfinance>=0.2.55","fredapi",
            "pmdarima","ta","statsmodels","scikit-learn",
            "requests","openpyxl","xlrd",
            "hmmlearn","arch","lightgbm",   # regime detection + strong baselines
            "vmdpy",                         # VMD decomposition hybrid (file 09)
            "transformers","sentence-transformers","faiss-cpu",  # LLM/RAG track (10/11/12)
            "pytrends",                      # Google Trends exogenous signal (B3)
            "PyEMD",                         # CEEMDAN secondary decomposition (file 09 TIER C)
            "cot_reports",                   # CFTC Commitment-of-Traders positioning (B6)
            "optuna"]                        # Hyperparameter sweep (file 16, Addendum 14)
for pkg in packages:
    subprocess.run([sys.executable,"-m","pip","install","-q",pkg], check=False)
print("✅ Packages ready")

# ── FIX: pytrends uses urllib3 Retry(method_whitelist=...) which was renamed to
# allowed_methods in urllib3 2.0+. Monkey-patch to accept both names. ──────────
try:
    from urllib3.util.retry import Retry as _Retry
    _orig_retry_init = _Retry.__init__
    def _patched_retry_init(self, *args, method_whitelist=None, allowed_methods=None, **kwargs):
        # merge: if old-style kwarg received, convert to new-style
        if method_whitelist is not None and allowed_methods is None:
            allowed_methods = method_whitelist
        if allowed_methods is not None:
            _orig_retry_init(self, *args, allowed_methods=allowed_methods, **kwargs)
        else:
            _orig_retry_init(self, *args, **kwargs)
    _patched_retry_init._patched_by_ddp = True
    if not getattr(_orig_retry_init, '_patched_by_ddp', False):
        _Retry.__init__ = _patched_retry_init
    print("✅ urllib3 Retry compatibility patch applied (fixes pytrends method_whitelist error)")
except Exception as _pe:
    print(f"  ⚠️  urllib3 patch failed ({_pe}) — Google Trends may still error; using zero fallback")

# ── IMPORTS ──────────────────────────────────────────────────
import os, warnings, math, random, io, pickle
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch as _torch
    _TORCH_OK = True
except Exception as _te:
    _torch = None
    _TORCH_OK = False
    print(f"  ⚠️  torch import failed ({_te}); seeding will use numpy/random only")
import yfinance as yf
# pandas_datareader is broken on newer pandas (deprecate_kwarg signature change).
# We use fredapi directly for all FRED fetches instead.
try:
    import pandas_datareader.data as web
    _PDR_OK = True
except Exception:
    _PDR_OK = False

# fredapi wrapper — used as a drop-in replacement for web.DataReader(id, "fred", ...)
try:
    from fredapi import Fred as _Fred
    _fred_client = _Fred()          # uses FRED_API_KEY env-var or anonymous (rate-limited)
    _FRED_OK = True
except Exception:
    _FRED_OK = False

def _fred_fetch(series_id, start, end):
    """Fetch a FRED series as a named pd.Series. Tries fredapi first, then pdr."""
    if _FRED_OK:
        try:
            s = _fred_client.get_series(series_id, observation_start=start, observation_end=end)
            s.name = series_id
            return s.dropna()
        except Exception:
            pass
    if _PDR_OK:
        try:
            raw = web.DataReader(series_id, "fred", start, end)
            s = (raw[series_id] if series_id in raw.columns else raw.iloc[:, 0]).dropna()
            s.name = series_id
            return s
        except Exception:
            pass
    return pd.Series(dtype=float, name=series_id)
import requests
import ta
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    if _TORCH_OK:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available(): _torch.cuda.manual_seed_all(seed)
set_seed(42)

# ── CONFIG ───────────────────────────────────────────────────
# Search for the CSV in cwd first, then the directory of this file.
_csv_name = "Cushing_OK_WTI_Spot_Price_FOB.csv"
_csv_candidates = [Path(_csv_name), Path(__file__).parent / _csv_name if "__file__" in globals() else None]
WTI_CSV = next((p for p in _csv_candidates if p is not None and p.exists()), None)
if WTI_CSV is None:
    raise FileNotFoundError(
        f"WTI CSV '{_csv_name}' not found in cwd ({Path.cwd()}) or script directory. "
        "Make sure the notebook is run from the DDP_v2 folder."
    )
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

HORIZONS = {
    "7d":  {"h": 7,   "lookback": 60,  "patch": 8,  "layers": 3, "d_model": 128, "loss": "mse",      "n_quantiles": 1},
    "90d": {"h": 90,  "lookback": 360, "patch": 16, "layers": 4, "d_model": 128, "loss": "huber",    "n_quantiles": 1},
    "180d":{"h": 180, "lookback": 360, "patch": 16, "layers": 3, "d_model": 128, "loss": "studentt", "n_quantiles": 1},
}
N_SEEDS = 5; N_CONFIGS = 3; BATCH_SIZE = 64; MAX_EPOCHS = 80; LR = 3e-4; PATIENCE = 10
WINS_PCTILE = 0.1
CRISES = {
    "GFC":      ("2008-09-01","2009-06-30"),
    "OPEC_War": ("2014-06-01","2016-03-31"),
    "COVID":    ("2020-02-01","2020-09-30"),
}
NEGATIVE_PRICE_DATE = "2020-04-20"
CRISIS_COLORS = {"GFC":"#d62728","OPEC_War":"#ff7f0e","COVID":"#9467bd"}
PALETTE = {"actual":"#1f77b4","pred":"#e74c3c","interval":"#aec7e8"}
print("✅ Config loaded")

# ── LOAD WTI ─────────────────────────────────────────────────
def load_wti(path):
    df = pd.read_csv(path, skiprows=4, names=["date","price"],
                     parse_dates=["date"], dayfirst=False)
    df = df.dropna(subset=["date","price"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date"); df.index.name = "date"
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna()

wti_raw = load_wti(WTI_CSV)
print(f"WTI: {wti_raw.index[0].date()} → {wti_raw.index[-1].date()}  ({len(wti_raw):,} rows)")
print(f"Negative prices: {(wti_raw['price']<0).sum()} days  |  Min: ${wti_raw['price'].min():.2f}")

# ── WINSORIZATION ────────────────────────────────────────────
wins_floor = np.percentile(wti_raw["price"], WINS_PCTILE)
print(f"Winsorization floor ({WINS_PCTILE}th pctile): ${wins_floor:.2f}")
wti_wins = wti_raw.copy()
wti_wins["price_orig"]    = wti_wins["price"]
wti_wins["price"]         = wti_wins["price"].clip(lower=wins_floor)
wti_wins["is_winsorized"] = (wti_wins["price_orig"] < wins_floor).astype(int)
print(wti_raw[wti_raw["price"]<0].to_string())
print(f"After winsorization — min price: ${wti_wins['price'].min():.2f}")

# ── EDA PLOT (Plotly — price overview + winsorization zoom) ──────────
# Two stacked panels: (1) full 1986–2026 history with crisis shading and the
# Apr-2020 negative-price marker; (2) a 2019–2022 zoom showing the winsorization
# clip. Wrapped so a plotting hiccup logs a warning but never aborts Stage 0.
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo0
    from plotly.subplots import make_subplots as _pmsp0
    _d_all = wti_raw.index.strftime("%Y-%m-%d")
    _fig0 = _pmsp0(rows=2, cols=1,
                   subplot_titles=["WTI Spot Price 1986–2026 (original, crises shaded)",
                                   "Winsorization Effect: 2019–2022 Zoom"],
                   row_heights=[0.6, 0.4], vertical_spacing=0.12)
    # Panel 1 — full history
    _fig0.add_trace(_pgo0.Scatter(x=_d_all, y=wti_raw["price"], name="WTI Spot Price",
                                  line=dict(color=PALETTE["actual"], width=0.9)), row=1, col=1)
    for _nm, (_s, _e) in CRISES.items():
        _fig0.add_vrect(x0=_s, x1=_e, fillcolor=CRISIS_COLORS[_nm], opacity=0.15,
                        line_width=0, annotation_text=_nm.replace("_", " "),
                        annotation_position="top left", row=1, col=1)
    _fig0.add_vline(x=NEGATIVE_PRICE_DATE, line_dash="dash", line_color="black",
                    line_width=1.2, row=1, col=1)
    # Panel 2 — winsorization zoom
    _mask0 = (wti_raw.index >= "2019-01-01") & (wti_raw.index <= "2022-01-01")
    _dz = wti_raw.index[_mask0].strftime("%Y-%m-%d")
    _fig0.add_trace(_pgo0.Scatter(x=_dz, y=wti_raw["price"][_mask0], name="Original",
                                  line=dict(color=PALETTE["actual"], width=1.4)), row=2, col=1)
    _fig0.add_trace(_pgo0.Scatter(x=_dz, y=wti_wins["price"][_mask0], name="Winsorized",
                                  line=dict(color="#e74c3c", width=1.4, dash="dash")), row=2, col=1)
    _fig0.add_hline(y=wins_floor, line_dash="dot", line_color="gray",
                    annotation_text=f"Floor ${wins_floor:.1f}", row=2, col=1)
    _fig0.update_yaxes(title_text="$/barrel", row=1, col=1)
    _fig0.update_yaxes(title_text="$/barrel", row=2, col=1)
    _fig0.update_layout(height=820, title="WTI Crude Oil — Price Overview & Winsorization")
    save_fig(_fig0, "01_wti_overview")
    print("✅ Figure saved: 01_wti_overview (html+png)")
except Exception as _e0:
    print(f"  [warn] WTI overview figure skipped: {_e0}")

# ── EXOGENOUS DATA ────────────────────────────────────────────
START_DATE = "1986-01-01"
END_DATE   = wti_raw.index[-1].strftime("%Y-%m-%d")

def safe_yf_download(ticker, start, end, col="Close"):
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty: return pd.Series(dtype=float, name=ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            matches = [(c,t) for c,t in raw.columns if c==col]
            s = raw[matches[0]] if matches else raw.iloc[:,0]
        else:
            s = raw[col] if col in raw.columns else raw.iloc[:,0]
        s = s.squeeze(); s.index = pd.to_datetime(s.index); s = s.dropna()
        if s.empty: return pd.Series(dtype=float, name=ticker)
        s.name = ticker
        bdays = pd.bdate_range(s.index[0], s.index[-1])
        return s.reindex(bdays).ffill()
    except Exception as e:
        print(f"  ⚠️  {ticker}: {e}"); return pd.Series(dtype=float, name=ticker)

def _fmt(s, label):
    d = s.dropna()
    if d.empty: return f"  {label:<12} ⚠️  empty"
    return f"  {label:<12} {d.index[0].date()} → {d.index[-1].date()}  ({len(d):,} pts)"

print("Fetching yfinance …")
vix = safe_yf_download("^VIX", START_DATE, END_DATE)
if vix.dropna().empty:
    vix = _fred_fetch("VIXCLS", START_DATE, END_DATE)
vix.name = "vix"; print(_fmt(vix,"VIX:"))

sp500 = safe_yf_download("^GSPC", START_DATE, END_DATE)
if sp500.dropna().empty:
    sp500 = _fred_fetch("SP500", START_DATE, END_DATE)
sp500.name = "sp500"; print(_fmt(sp500,"S&P 500:"))

brent_yf = safe_yf_download("BZ=F", START_DATE, END_DATE); brent_yf.name = "brent"; print(_fmt(brent_yf,"Brent:"))
# (Brent has implicit FRED fallback via brent_fred at the merge step below.)

natgas = safe_yf_download("NG=F", START_DATE, END_DATE)
if natgas.dropna().empty:
    natgas = _fred_fetch("DHHNGSP", START_DATE, END_DATE)
natgas.name = "natgas"; print(_fmt(natgas,"Nat Gas:"))

gold = safe_yf_download("GC=F", START_DATE, END_DATE)
if gold.dropna().empty:
    gold = _fred_fetch("GOLDAMGBD228NLBM", START_DATE, END_DATE)
gold.name = "gold"; print(_fmt(gold,"Gold:"))

dxy = safe_yf_download("DX-Y.NYB", START_DATE, END_DATE)
if dxy.dropna().empty:
    # DTWEXBGS = Broad Trade-Weighted USD (proxy; ~0.95 monthly corr with DXY 6-currency basket)
    dxy = _fred_fetch("DTWEXBGS", START_DATE, END_DATE)
dxy.name = "dxy"; print(_fmt(dxy,"DXY:"))

t10y = safe_yf_download("^TNX", START_DATE, END_DATE)
if t10y.dropna().empty:
    t10y = _fred_fetch("DGS10", START_DATE, END_DATE)
t10y.name = "t10y"; print(_fmt(t10y,"T10Y:"))

print("\nFetching FRED …")
FRED_SERIES = {
    "brent_fred":     "DCOILBRENTEU",
    "wti_fred":       "DCOILWTICO",
    "indpro":         "INDPRO",
    "unrate":         "UNRATE",
    "cpi":            "CPIAUCSL",
    "fedfunds":       "FEDFUNDS",
    "yield_spread":   "T10Y2Y",
    "usdeur":         "DEXUSEU",
    # ADD-10: list-of-IDs → fetcher tries each in order, keeps first non-empty.
    # Extended fallback list — FRED series IDs occasionally change/deprecate.
    # Cushing, OK crude oil stocks: WCESTUS1 (primary), WCSSTUS1, WTTSTUS1 fallbacks.
    # US crude oil ending stocks:   WCESTUS0 (primary), WCRSTUS1, W_EPC0_SAX_YCUOK_MBBL fallbacks.
    "cushing_stocks": ["WTTSTUS1", "WCESTUS1", "WCSSTUS2", "WCRFPUS2"],
    "crude_stocks":   ["WCESTUS0", "WCRSTUS1", "MTESTUS1", "DCOILBRENTEU"],
    "gdp":            "GDP",
}
fred_data = {}
for name, sid in FRED_SERIES.items():
    sids = sid if isinstance(sid, list) else [sid]
    s = pd.Series(dtype=float, name=name)
    last_err = None
    for _try in sids:
        try:
            cand = _fred_fetch(_try, START_DATE, END_DATE)
            if not cand.empty:
                s = cand; s.name = name
                if len(sids) > 1:
                    print(f"  ✅ {name} (via {_try}): {s.index[0].date()} → {s.index[-1].date()}")
                else:
                    print(f"  ✅ {name} ({_try}): {s.index[0].date()} → {s.index[-1].date()}")
                break
        except Exception as e:
            last_err = e
            continue
    if s.empty:
        print(f"  ⚠️  {name}: all IDs failed ({last_err})")
    fred_data[name] = s

print("\nFetching Baker Hughes Rig Count …")
# Primary: FRED series OILRIGS (Baker Hughes US rig count via EIA, no timeout risk)
# Fallback: Baker Hughes static file (one attempt, 30s timeout)
rig_count = pd.Series(dtype=float, name="rig_count")
for _rig_id in ["OILRIGS", "WRLURI2US"]:
    try:
        _rc = _fred_fetch(_rig_id, START_DATE, END_DATE)
        if not _rc.dropna().empty:
            rig_count = _rc.rename("rig_count")
            print(f"  ✅ Rig count (FRED/{_rig_id}): {rig_count.dropna().index[0].date()} → {rig_count.dropna().index[-1].date()}")
            break
    except Exception:
        continue
if rig_count.empty:
    # One-shot fallback to Baker Hughes website (may timeout on Kaggle/restricted networks)
    BH_URL = "https://rigcount.bakerhughes.com/static-files/7b91b020-a6a1-4a97-b9a9-08e2bfa85dc8"
    try:
        resp = requests.get(BH_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            xls = pd.ExcelFile(io.BytesIO(resp.content))
            rc  = pd.read_excel(xls, sheet_name=0, header=5)
            date_col = rc.columns[0]
            us_cols  = [c for c in rc.columns if "united states" in str(c).lower() or "u.s." in str(c).lower()]
            rig_count = rc.set_index(date_col)[us_cols[0] if us_cols else rc.columns[1]].dropna()
            rig_count.index = pd.to_datetime(rig_count.index, errors="coerce")
            rig_count = rig_count.dropna(); rig_count.name = "rig_count"
            if not rig_count.empty:
                print(f"  ✅ Rig count (Baker Hughes URL): {rig_count.index[0].date()} → {rig_count.index[-1].date()}")
    except Exception as e:
        print(f"  ⚠️  Rig count BH URL failed ({e})")
if rig_count.empty:
    print("  ⚠️  Rig count: all attempts failed — rig_count_norm will be 0 (non-critical feature)")

print("\nFetching EPU …")
# URL: https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv
epu = pd.Series(dtype=float, name="epu")
try:
    epu_df  = pd.read_csv("https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv")
    idx_col = [c for c in epu_df.columns if "index" in c.lower() or "uncertainty" in c.lower()]
    if not idx_col: idx_col = [epu_df.columns[-1]]
    epu_df["date"] = pd.to_datetime(epu_df[["Year","Month"]].assign(Day=1))
    epu = epu_df.set_index("date")[idx_col[0]].dropna(); epu.name = "epu"
    print(f"  ✅ EPU: {epu.index[0].date()} → {epu.index[-1].date()}" if not epu.empty else "  ⚠️  empty")
except Exception as e: print(f"  ⚠️  EPU: {e}")

print("\nFetching GPR …")
# Primary : https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
# Fallback: https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.csv
gpr = pd.Series(dtype=float, name="gpr")
for url, reader in [("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", pd.read_excel),
                    ("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.csv",  pd.read_csv)]:
    try:
        gpr_df  = reader(url, index_col=0)
        gpr_col = [c for c in gpr_df.columns if str(c).lower().strip() in ("gprd","gpr","gpr_index")]
        if not gpr_col: gpr_col = [gpr_df.select_dtypes(include="number").columns[0]]
        gpr = gpr_df[gpr_col[0]].dropna()
        gpr.index = pd.to_datetime(gpr.index, errors="coerce"); gpr = gpr.dropna(); gpr.name = "gpr"
        print(f"  ✅ GPR: {gpr.index[0].date()} → {gpr.index[-1].date()}" if not gpr.empty else "  ⚠️  empty"); break
    except Exception as e: print(f"  ⚠️  GPR ({url.split('/')[-1]}): {e}")

print("\nFetching OPEC production proxy (FRED) …")
# ADD-10: PAPR_OPEC was retired on FRED. Try chain of candidate IDs;
# if all fail, zero-fill gracefully (pipeline handles missing macros).
opec_prod = pd.Series(dtype=float, name="opec_prod")
last_err = None
for _try in ["PAPR_OPEC", "POILAPSPINDEXM"]:
    try:
        cand = _fred_fetch(_try, START_DATE, END_DATE)
        if not cand.empty:
            opec_prod = cand; opec_prod.name = "opec_prod"
            print(f"  ✅ OPEC (via {_try}): {opec_prod.index[0].date()} → {opec_prod.index[-1].date()}")
            break
    except Exception as e:
        last_err = e; continue
if opec_prod.empty:
    print(f"  ⚠️  OPEC: zero fallback (all FRED IDs unavailable; last err: {last_err})")

print("\nFetching OVX (CBOE Crude Oil Volatility Index — FRED OVXCLS) …")
# B1: OVX — Liang & Lv 2020, J. Futures Markets: +10-20% RMSE reduction when OVX added to VIX
ovx_s = pd.Series(dtype=float, name="ovx")
try:
    ovx_s = _fred_fetch("OVXCLS", START_DATE, END_DATE)
    ovx_s.name = "ovx"
    if not ovx_s.empty:
        print(f"  ✅ OVX: {ovx_s.index[0].date()} → {ovx_s.index[-1].date()}  ({len(ovx_s):,} pts)")
    else:
        print("  ⚠️  OVX (OVXCLS): empty — zero fallback")
except Exception as e:
    print(f"  ⚠️  OVX (OVXCLS): {e} — zero fallback")

print("\nFetching Google Trends (pytrends) …")
# B3: Google Trends — Yang et al. arXiv 2002.09656 & Elshendy et al. Energy 2018
import time as _time
_gt_cols = ["gt_oil_price", "gt_OPEC", "gt_WTI"]
_gt_data = {c: pd.Series(dtype=float) for c in _gt_cols}
try:
    from pytrends.request import TrendReq as _TrendReq
    _pt = _TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)
    _kw_map = [("gt_oil_price","oil price"), ("gt_OPEC","OPEC"), ("gt_WTI","WTI")]
    for _col, _kw_raw in _kw_map:
        try:
            _pt.build_payload([_kw_raw], timeframe=f"{START_DATE[:10]} {END_DATE[:10]}", geo="")
            _gt_raw = _pt.interest_over_time()
            if _kw_raw in _gt_raw.columns:
                _gt_data[_col] = _gt_raw[_kw_raw].astype(float)
                print(f"  ✅ Google Trends '{_kw_raw}'")
            _time.sleep(2)   # be polite to the free endpoint
        except Exception as _e2:
            print(f"  ⚠️  Google Trends '{_kw_raw}': {_e2} — zero fallback")
except ImportError:
    print("  ⚠️  pytrends not installed — Google Trends will use zero fallback")
except Exception as _e:
    print(f"  ⚠️  Google Trends error ({_e}) — zero fallback")

# ── B6: CFTC Commitment-of-Traders (managed-money positioning) ───────────────
# Net speculative positioning is a well-cited mid/long-horizon WTI predictor
# (Kang et al. 2020; Sanders et al.). The disaggregated COT report (2009+)
# gives managed-money longs/shorts + open interest for NYMEX WTI crude.
#
# ⚠️  LEAKAGE DISCIPLINE: each weekly report is a TUESDAY snapshot RELEASED the
# following FRIDAY (~3 business days later). We therefore index each report by
# its RELEASE date (snapshot + 3 bdays), so a given business day only ever sees
# reports that were already public. resample_bd(..., "ffill") at the merge step
# then carries the latest *released* report forward. Same discipline as the
# news features (10_text_features.py) and rolling VMD (09_vmd_hybrid.py).
print("\nFetching CFTC Commitment-of-Traders (managed-money WTI positioning) …")
cot_mm_long  = pd.Series(dtype=float, name="cot_mm_long")
cot_mm_short = pd.Series(dtype=float, name="cot_mm_short")
cot_oi       = pd.Series(dtype=float, name="cot_oi")
try:
    import cot_reports as _cot
    _raw = _cot.cot_all(cot_report_type="disaggregated_fut")
    # locate the crude-oil WTI / NYMEX market rows (column name varies by vintage)
    _mkt_col = next((c for c in _raw.columns if "Market_and_Exchange" in c), None)
    _date_col = next((c for c in ["Report_Date_as_YYYY-MM-DD", "Report_Date_as_YYYY_MM_DD",
                                  "As_of_Date_in_Form_YYMMDD", "As_of_Date_In_Form_YYMMDD"]
                      if c in _raw.columns), None)
    if _mkt_col and _date_col:
        _name = _raw[_mkt_col].astype(str).str.upper()
        _is_wti = _name.str.contains("CRUDE OIL") & (
            _name.str.contains("WTI") | _name.str.contains("NEW YORK MERCANTILE") | _name.str.contains("NYMEX"))
        _w = _raw[_is_wti].copy()
        # parse report (snapshot) date
        if "YYMMDD" in _date_col:
            _dt = pd.to_datetime(_w[_date_col].astype(str).str.zfill(6), format="%y%m%d", errors="coerce")
        else:
            _dt = pd.to_datetime(_w[_date_col], errors="coerce")
        _w = _w.assign(_snap=_dt).dropna(subset=["_snap"]).sort_values("_snap")
        _long_col  = next((c for c in _w.columns if "M_Money_Positions_Long"  in c), None)
        _short_col = next((c for c in _w.columns if "M_Money_Positions_Short" in c), None)
        _oi_col    = next((c for c in _w.columns if c.startswith("Open_Interest_All")), None)
        if _long_col and _short_col and _oi_col:
            _w = _w.drop_duplicates(subset=["_snap"], keep="last").set_index("_snap")
            # LEAKAGE-SAFE: shift snapshot → public release date (+3 business days)
            _release_idx = _w.index + pd.tseries.offsets.BDay(3)
            cot_mm_long  = pd.Series(pd.to_numeric(_w[_long_col],  errors="coerce").values, index=_release_idx, name="cot_mm_long").dropna()
            cot_mm_short = pd.Series(pd.to_numeric(_w[_short_col], errors="coerce").values, index=_release_idx, name="cot_mm_short").dropna()
            cot_oi       = pd.Series(pd.to_numeric(_w[_oi_col],    errors="coerce").values, index=_release_idx, name="cot_oi").dropna()
            print(f"  ✅ COT WTI: {cot_mm_long.index[0].date()} → {cot_mm_long.index[-1].date()}  "
                  f"({len(cot_mm_long):,} weekly reports, release-lagged +3bd)")
        else:
            print("  ⚠️  COT: managed-money / open-interest columns not found — zero fallback")
    else:
        print("  ⚠️  COT: market/date columns not found — zero fallback")
except Exception as _e:
    print(f"  ⚠️  COT fetch failed ({_e}) — zero fallback (pipeline unaffected)")

# ── B6b: Futures term-structure slope (front vs deferred) — best-effort ──────
# True contango/backwardation. Complements the cheap 63d realized_carry proxy.
# Best-effort: tries candidate FRED/yfinance series; if none resolve, ts_slope
# is simply not built and select_features() drops it (realized_carry remains).
print("Fetching WTI futures term structure (front vs deferred) …")
wti_fut_c1 = pd.Series(dtype=float, name="wti_fut_c1")
wti_fut_c4 = pd.Series(dtype=float, name="wti_fut_c4")
try:
    # contract 1 (front): WTI Cushing daily (EIA contract-1 on FRED)
    wti_fut_c1 = _fred_fetch("DCOILWTICO", START_DATE, END_DATE)
    if wti_fut_c1.dropna().empty:
        wti_fut_c1 = safe_yf_download("CL=F", START_DATE, END_DATE)
    wti_fut_c1.name = "wti_fut_c1"
    # deferred contract candidates (vary by source/vintage; first non-empty wins)
    for _cid in ("MCOILWTICO", "WCOILWTICO"):
        _c4 = _fred_fetch(_cid, START_DATE, END_DATE)
        if not _c4.dropna().empty:
            wti_fut_c4 = _c4; break
    wti_fut_c4.name = "wti_fut_c4"
    if not wti_fut_c1.dropna().empty:
        print(_fmt(wti_fut_c1, "WTI fut c1:"))
    if not wti_fut_c4.dropna().empty:
        print(_fmt(wti_fut_c4, "WTI fut c4:"))
    else:
        print("  ℹ️  No deferred WTI futures series resolved — ts_slope will fall back to realized_carry proxy")
except Exception as _e:
    print(f"  ⚠️  Futures term-structure fetch failed ({_e}) — using realized_carry proxy")

print("\n✅ All exogenous data fetched")

# ── MERGE ────────────────────────────────────────────────────
def resample_bd(s, method="ffill"):
    if s.empty: return s
    s = s[~s.index.duplicated(keep="first")].sort_index()
    idx = pd.bdate_range(s.index[0], s.index[-1])
    s = s.reindex(idx)
    return s.ffill() if method=="ffill" else s.interpolate(method="time").ffill()

series_dict = {
    "vix":          resample_bd(vix),
    "sp500":        resample_bd(sp500),
    "brent":        resample_bd(brent_yf if not brent_yf.empty else fred_data.get("brent_fred", pd.Series(dtype=float))),
    "natgas":       resample_bd(natgas),
    "gold":         resample_bd(gold),
    "dxy":          resample_bd(dxy),
    "t10y":         resample_bd(t10y),
    "brent_fred":   resample_bd(fred_data.get("brent_fred", pd.Series(dtype=float))),
    "indpro":       resample_bd(fred_data.get("indpro",  pd.Series(dtype=float)), "interpolate"),
    "unrate":       resample_bd(fred_data.get("unrate",  pd.Series(dtype=float))),
    "cpi":          resample_bd(fred_data.get("cpi",     pd.Series(dtype=float)), "interpolate"),
    "fedfunds":     resample_bd(fred_data.get("fedfunds",pd.Series(dtype=float))),
    "yield_spread": resample_bd(fred_data.get("yield_spread", pd.Series(dtype=float))),
    "usdeur":       resample_bd(fred_data.get("usdeur",  pd.Series(dtype=float))),
    "cushing_stocks":resample_bd(fred_data.get("cushing_stocks", pd.Series(dtype=float))),
    "crude_stocks": resample_bd(fred_data.get("crude_stocks",    pd.Series(dtype=float))),
    "gdp":          resample_bd(fred_data.get("gdp",     pd.Series(dtype=float)), "interpolate"),
    "opec_prod":    resample_bd(opec_prod),
    "rig_count":    resample_bd(rig_count),
    "epu":          resample_bd(epu),
    "gpr":          resample_bd(gpr),
    # Tier B additions
    "ovx":          resample_bd(ovx_s),              # B1: oil VIX
    "gt_oil_price": resample_bd(_gt_data.get("gt_oil_price", pd.Series(dtype=float))),  # B3
    "gt_OPEC":      resample_bd(_gt_data.get("gt_OPEC",      pd.Series(dtype=float))),
    "gt_WTI":       resample_bd(_gt_data.get("gt_WTI",       pd.Series(dtype=float))),
    # B6: CFTC COT positioning (release-lagged) + futures term structure
    "cot_mm_long":  resample_bd(cot_mm_long),
    "cot_mm_short": resample_bd(cot_mm_short),
    "cot_oi":       resample_bd(cot_oi),
    "wti_fut_c1":   resample_bd(wti_fut_c1),
    "wti_fut_c4":   resample_bd(wti_fut_c4),
}

master = wti_wins.copy()
for name, s in series_dict.items():
    master = master.join(s.rename(name), how="left") if not s.empty else master.assign(**{name: np.nan})
master = master.ffill().bfill().fillna(0)
print(f"Master shape: {master.shape}")

# ── FEATURE ENGINEERING ──────────────────────────────────────
df = master.copy()
p  = df["price"]

df["log_price"]   = np.log(p.clip(lower=1e-6))
df["log_ret_1d"]  = df["log_price"].diff(1)
df["log_ret_5d"]  = df["log_price"].diff(5)
df["log_ret_21d"] = df["log_price"].diff(21)

df["vol_7d"]  = df["log_ret_1d"].rolling(7).std()
df["vol_30d"] = df["log_ret_1d"].rolling(30).std()
df["vol_90d"] = df["log_ret_1d"].rolling(90).std()
df["vol_ratio_7_30"] = df["vol_7d"] / (df["vol_30d"] + 1e-8)

for w in [5,20,50,200]:
    df[f"ma_{w}d"]       = p.rolling(w).mean()
    df[f"ma_ratio_{w}d"] = p / (df[f"ma_{w}d"] + 1e-8)

df["rsi_14"]      = ta.momentum.RSIIndicator(close=p, window=14).rsi()
macd_obj          = ta.trend.MACD(close=p, window_slow=26, window_fast=12, window_sign=9)
df["macd"]        = macd_obj.macd()
df["macd_signal"] = macd_obj.macd_signal()
df["macd_diff"]   = macd_obj.macd_diff()
bb = ta.volatility.BollingerBands(close=p, window=20, window_dev=2)
df["bb_upper"] = bb.bollinger_hband(); df["bb_lower"] = bb.bollinger_lband()
df["bb_width"] = (df["bb_upper"]-df["bb_lower"])/(df["ma_20d"]+1e-8)
df["bb_pct"]   = bb.bollinger_pband()
df["atr_14"]   = ta.volatility.AverageTrueRange(high=p*1.005, low=p*0.995, close=p, window=14).average_true_range()
df["cci_20"]   = ta.trend.CCIIndicator(high=p*1.005, low=p*0.995, close=p, window=20).cci()

brent_col = "brent_fred" if df["brent_fred"].abs().sum()>0 else "brent"
df["brent_wti_spread"] = df[brent_col] - df["price"]
df["gas_crude_ratio"]  = df["natgas"] / (p + 1e-8)
df["gold_oil_ratio"]   = df["gold"]   / (p + 1e-8)

df["vix_level"]  = df["vix"]
df["vix_ma21"]   = df["vix"].rolling(21).mean()
df["vix_pctile"] = df["vix"].rolling(252).rank(pct=True)
df["crisis_flag"]= (df["vix"]>30).astype(int)
df["extreme_price_flag"] = df["is_winsorized"]

# ── Raw-shock features (from pre-winsorization price) ─────────
# Winsorization floors sub-$10.95 prices upward (required for log_price),
# but the model needs to see how large the shock was. price_orig preserves
# the raw value (including the COVID negative print at -$36.98).
_d = df["price_orig"].diff()
df["raw_ret_signed_log"] = np.sign(_d) * np.log1p(_d.abs())  # sign-preserving log magnitude
df["raw_dd_below_floor"] = (wins_floor - df["price_orig"]).clip(lower=0)  # 0 normally; >0 on extreme lows

df["cushing_chg_7d"] = df["cushing_stocks"].diff(7)
df["crude_chg_7d"]   = df["crude_stocks"].diff(7)

# ── B1: OVX-derived features (oil-specific volatility signal) ─────────────────
# 10-20% RMSE lift documented in Liang & Lv 2020 J. Futures Markets
df["ovx_chg_5d"]    = df["ovx"].pct_change(5).fillna(0)
df["ovx_minus_vix"] = (df["ovx"] - df["vix_level"]).fillna(0)   # oil-specific tail premium

# ── B2: Futures curve slope / carry proxy ────────────────────────────────────
# Contango vs backwardation is one of the strongest mid-horizon WTI predictors.
# Carry proxy = 63-day log-price change (captures same regime info cheaply).
df["realized_carry"] = (df["log_price"] - df["log_price"].shift(63)).fillna(0)

# ── B6: COT positioning + true term-structure features (bounded/stationary) ───
# Net managed-money positioning relative to open interest, its 252d z-score
# (regime-relative extreme), and weekly momentum. All release-lagged upstream
# (leakage-safe). ts_slope is the true front-vs-deferred log spread when a
# deferred futures series resolved; otherwise it falls back to realized_carry.
if {"cot_mm_long", "cot_mm_short", "cot_oi"}.issubset(df.columns) and df["cot_oi"].abs().sum() > 0:
    _oi = df["cot_oi"].replace(0, np.nan)
    df["cot_mm_net"]        = ((df["cot_mm_long"] - df["cot_mm_short"]) / _oi).fillna(0)
    _mean = df["cot_mm_net"].rolling(252, min_periods=21).mean()
    _std  = df["cot_mm_net"].rolling(252, min_periods=21).std()
    df["cot_mm_net_z"]      = ((df["cot_mm_net"] - _mean) / (_std + 1e-8)).fillna(0)
    df["cot_mm_net_chg_5d"] = df["cot_mm_net"].diff(5).fillna(0)
else:
    df["cot_mm_net"] = df["cot_mm_net_z"] = df["cot_mm_net_chg_5d"] = 0.0

if {"wti_fut_c1", "wti_fut_c4"}.issubset(df.columns) and df["wti_fut_c4"].abs().sum() > 0:
    _c1 = df["wti_fut_c1"].clip(lower=1e-6); _c4 = df["wti_fut_c4"].clip(lower=1e-6)
    df["ts_slope"] = (np.log(_c1) - np.log(_c4)).fillna(0)   # >0 backwardation, <0 contango
else:
    df["ts_slope"] = df["realized_carry"]                    # graceful fallback to the proxy

# ── B5: Explicit shock event dummies + OPEC decay ────────────────────────────
# One-hot dummies for the three most market-disrupting events; the rolling
# crisis_flag catches nearby days but not the exact shock day.
df["event_covid_neg"]  = (df.index == pd.Timestamp("2020-04-20")).astype(float)
df["event_russia_inv"] = (((df.index >= pd.Timestamp("2022-02-24")) &
                            (df.index <= pd.Timestamp("2022-03-15")))).astype(float)
df["event_opec_war"]   = (((df.index >= pd.Timestamp("2014-06-01")) &
                            (df.index <= pd.Timestamp("2016-03-31")))).astype(float)

# ── B3: Google Trends features (fill zeros if fetch failed) ─────────────────
for _gt_col in ["gt_oil_price", "gt_OPEC", "gt_WTI"]:
    if _gt_col in df.columns:
        df[_gt_col] = df[_gt_col].fillna(0.0)
    else:
        df[_gt_col] = 0.0

df["z_score_90d"]  = (p-p.rolling(90).mean()) / (p.rolling(90).std()+1e-8)
df["z_score_252d"] = (p-p.rolling(252).mean())/ (p.rolling(252).std()+1e-8)

df["day_of_week"] = df.index.dayofweek; df["month"] = df.index.month
df["quarter"]     = df.index.quarter;   df["year"]  = df.index.year
df["month_sin"] = np.sin(2*np.pi*df["month"]/12); df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/5); df["dow_cos"] = np.cos(2*np.pi*df["day_of_week"]/5)
df["opec_month"]= df["month"].isin({1,3,5,6,9,11,12}).astype(int)
# OPEC meeting decay: heightened market attention in the week around OPEC months
# (must come AFTER opec_month is defined above)
df["opec_decay_5d"] = df["opec_month"].rolling(5, min_periods=1).max().fillna(0)

df["cpi_yoy"]    = df["cpi"].pct_change(252)
df["indpro_yoy"] = df["indpro"].pct_change(252)
df["gdp_qoq"]    = df["gdp"].pct_change(63)

for col in ["sp500","gold","natgas","dxy"]:
    if df[col].abs().sum()>0:
        df[f"{col}_norm"] = df[col]/(df[col].rolling(252).mean()+1e-8)
df["rig_count_norm"] = df["rig_count"]/(df["rig_count"].rolling(52*5).mean()+1e-8)

df = df.replace([np.inf,-np.inf], np.nan).ffill().fillna(0)
print(f"Feature engineering done. Shape: {df.shape}")

# ── HMM REGIME DETECTION ─────────────────────────────────────
# Fit a Gaussian HMM on [log_ret_1d, vol_30d, vix] → data-driven latent
# market regimes (calm / trending / volatile / crisis). Replaces the crude
# VIX>30 flag with regime states + posterior probabilities as features.
# States are re-labelled by ascending volatility so regime_vol_rank ∈ [0,1]
# is an interpretable severity score (0 = calmest, 1 = most turbulent).
try:
    from hmmlearn.hmm import GaussianHMM
    N_REGIMES = 4
    hmm_feats = df[["log_ret_1d","vol_30d","vix_level"]].fillna(0).values.astype(float)
    hmm_X     = (hmm_feats - hmm_feats.mean(0)) / (hmm_feats.std(0) + 1e-8)
    set_seed(42)
    hmm = GaussianHMM(n_components=N_REGIMES, covariance_type="diag",
                      n_iter=200, random_state=42)
    hmm.fit(hmm_X)
    states = hmm.predict(hmm_X)
    post   = hmm.predict_proba(hmm_X)
    # order states by mean (standardized) volatility → severity ranking
    vol_by_state = [hmm_X[states==s][:,1].mean() if (states==s).any() else -np.inf
                    for s in range(N_REGIMES)]
    order   = list(np.argsort(vol_by_state))          # ascending volatility
    relabel = {old:new for new,old in enumerate(order)}
    df["regime_state"]    = [relabel[s] for s in states]
    df["regime_vol_rank"] = df["regime_state"] / (N_REGIMES - 1)   # 0..1 severity
    for new_idx, old in enumerate(order):
        df[f"regime_prob_{new_idx}"] = post[:, old]
    REGIME_FEATURES = ["regime_vol_rank"] + [f"regime_prob_{i}" for i in range(N_REGIMES)]
    print(f"✅ HMM regimes fitted ({N_REGIMES} states). "
          f"Counts: {pd.Series(df['regime_state']).value_counts().sort_index().to_dict()}")
except Exception as e:
    print(f"⚠️  HMM regime detection failed ({e}); falling back to crisis_flag")
    df["regime_vol_rank"] = df["crisis_flag"].astype(float)
    REGIME_FEATURES = ["regime_vol_rank"]

df = df.replace([np.inf,-np.inf], np.nan).fillna(0)

# ── MERGE TEXT / NEWS CONTEXT FEATURES (from 10_text_features.py) ──
# Leakage-safe daily GDELT tone/volume + event intensities + RAG retrieval
# + FinBERT sentiment. Skips gracefully if file 10 wasn't run.
TEXT_FEATURES = []
_tf_path = OUTPUT_DIR/"text_features_daily.csv"
if _tf_path.exists():
    try:
        tf = pd.read_csv(_tf_path, index_col=0, parse_dates=True)
        tf = tf[~tf.index.duplicated(keep="first")].sort_index()
        df = df.join(tf.reindex(df.index).ffill(), how="left")
        TEXT_FEATURES = [c for c in tf.columns if c in df.columns]
        df[TEXT_FEATURES] = df[TEXT_FEATURES].fillna(0.0)
        print(f"✅ Merged {len(TEXT_FEATURES)} text/news features: {TEXT_FEATURES}")
    except Exception as e:
        print(f"⚠️  Could not merge text_features_daily.csv ({e}); continuing without")
else:
    print("ℹ️  text_features_daily.csv not found — run 10_text_features.py first to add "
          "GDELT/FinBERT/RAG features (pipeline still works without them)")

df = df.replace([np.inf,-np.inf], np.nan).fillna(0)

# ── FEATURE SETS PER HORIZON ─────────────────────────────────
BASE_FEATURES = [
    "log_price","log_ret_1d","log_ret_5d","vol_7d","vol_30d","vol_ratio_7_30",
    "ma_ratio_5d","ma_ratio_20d","rsi_14","macd_diff","bb_pct","bb_width",
    "vix_level","crisis_flag","extreme_price_flag","brent_wti_spread","gold_oil_ratio",
    "month_sin","month_cos","dow_sin","dow_cos","opec_month","z_score_90d",
    "raw_ret_signed_log","raw_dd_below_floor",   # shock magnitude from pre-winsorization price
    # Tier B additions (B1 OVX, B2 carry, B3 Google Trends, B5 event dummies)
    "ovx","ovx_chg_5d","ovx_minus_vix",          # B1: oil-specific volatility
    "realized_carry",                             # B2: 63d carry proxy (contango/backwardation)
    "gt_oil_price","gt_OPEC","gt_WTI",            # B3: public attention signals
    "event_covid_neg","event_russia_inv","event_opec_war","opec_decay_5d",  # B5: event dummies
]
MED_FEATURES  = BASE_FEATURES + [
    "log_ret_21d","vol_90d","ma_ratio_50d","vix_pctile","vix_ma21","yield_spread",
    "cushing_chg_7d","crude_chg_7d","sp500_norm","dxy","epu","gpr",
    "cci_20","atr_14","rig_count_norm","opec_prod",
    # B6: CFTC managed-money positioning + true term-structure slope (mid/long horizon)
    "cot_mm_net","cot_mm_net_z","cot_mm_net_chg_5d","ts_slope",
]
LONG_FEATURES = MED_FEATURES + [
    "ma_ratio_200d","z_score_252d","cpi_yoy","indpro_yoy","gdp_qoq",
    "fedfunds","usdeur","gas_crude_ratio","gold_norm","macd","macd_signal",
]

def select_features(fl, df):
    return [f for f in dict.fromkeys(fl) if f in df.columns]

# Append HMM regime features to every horizon (regime context helps all horizons)
BASE_FEATURES = BASE_FEATURES + REGIME_FEATURES
MED_FEATURES  = MED_FEATURES  + REGIME_FEATURES
LONG_FEATURES = LONG_FEATURES + REGIME_FEATURES

# Append text/news context features (GDELT/FinBERT/RAG) to every horizon
if TEXT_FEATURES:
    BASE_FEATURES = BASE_FEATURES + TEXT_FEATURES
    MED_FEATURES  = MED_FEATURES  + TEXT_FEATURES
    LONG_FEATURES = LONG_FEATURES + TEXT_FEATURES

FEAT = {
    "7d":   select_features(BASE_FEATURES, df),
    "90d":  select_features(MED_FEATURES,  df),
    "180d": select_features(LONG_FEATURES, df),
}
# ADD-14: 1-day horizon reuses the 7d feature set (short-horizon, same exogenous mix).
FEAT["1d"] = list(FEAT["7d"])
for k,v in FEAT.items(): print(f"  {k} → {len(v)} features")

TARGET_COLS = {"1d":"target_1d","7d":"target_7d","90d":"target_90d","180d":"target_180d"}
df["target_1d"]   = df["log_price"].shift(-1)   - df["log_price"]
df["target_7d"]   = df["log_price"].shift(-7)   - df["log_price"]
df["target_90d"]  = df["log_price"].shift(-90)  - df["log_price"]
df["target_180d"] = df["log_price"].shift(-180) - df["log_price"]
print("✅ Feature sets defined")

# ── BLACK SWAN DETECTION ──────────────────────────────────────
def detect_extreme_events(price_series, z_window=252, z_thresh=2.5, min_gap_days=90):
    roll_mean = price_series.rolling(z_window).mean()
    roll_std  = price_series.rolling(z_window).std()
    z = (price_series - roll_mean) / (roll_std + 1e-8)
    ret = price_series.pct_change()
    large_move = ret.abs() > ret.rolling(z_window).std() * 3
    extreme = (z.abs() > z_thresh) | large_move
    events, in_event, event_start = [], False, None
    last_end = pd.Timestamp("1900-01-01")
    for dt, is_ex in extreme.items():
        if is_ex and not in_event:
            if (dt - last_end).days > min_gap_days: in_event = True; event_start = dt
        elif not is_ex and in_event:
            in_event = False; w = price_series.loc[event_start:dt]
            events.append({"start":event_start,"end":dt,
                           "peak_date": w.idxmin() if w.min()<0 else w.idxmax(),
                           "peak_price": w.min() if w.min()<0 else w.max(),
                           "pct_change":(w.iloc[-1]-w.iloc[0])/(abs(w.iloc[0])+1e-8)*100,
                           "duration_days":(dt-event_start).days})
            last_end = dt
    return pd.DataFrame(events)

extreme_events = detect_extreme_events(wti_raw["price"])
def tag_named_crises(ev_df, crises):
    labels = []
    for _, row in ev_df.iterrows():
        matched = next((n for n,(cs,ce) in crises.items()
                       if not (row["end"]<pd.Timestamp(cs) or row["start"]>pd.Timestamp(ce))), None)
        labels.append(matched)
    ev_df = ev_df.copy(); ev_df["crisis_name"] = labels; return ev_df

extreme_events = tag_named_crises(extreme_events, CRISES)
print(f"Detected {len(extreme_events)} extreme events")
print(extreme_events[["start","end","crisis_name","pct_change"]].to_string())

# ── PICKLE SAVE ───────────────────────────────────────────────
import pickle
with open(OUTPUT_DIR/"df_features.pkl",   "wb") as f: pickle.dump(df,             f)
with open(OUTPUT_DIR/"wti_raw.pkl",       "wb") as f: pickle.dump(wti_raw,        f)
with open(OUTPUT_DIR/"wti_wins.pkl",      "wb") as f: pickle.dump(wti_wins,       f)
with open(OUTPUT_DIR/"feat_cols.pkl",     "wb") as f: pickle.dump(FEAT,           f)
with open(OUTPUT_DIR/"extreme_events.pkl","wb") as f: pickle.dump(extreme_events, f)
with open(OUTPUT_DIR/"wins_floor.pkl",    "wb") as f: pickle.dump(wins_floor,     f)
# Lightweight regime series for regime-conditional evaluation in 06_
if "regime_state" in df.columns:
    regime_df = df[["regime_state","regime_vol_rank"]].copy()
    with open(OUTPUT_DIR/"regimes.pkl","wb") as f: pickle.dump(regime_df, f)

# ── EDA FIGURES (generated here so analysts see the data before training) ──
# Bootstrap _plotting.py if not already loaded (exec'd by orchestrators).
# Guarded: if _plotting.py is unavailable (rare), skip silently.
try:
    if "save_fig" not in globals():
        exec(open("_plotting.py", encoding="utf-8").read())
    import plotly.graph_objects as _pgo
    from plotly.subplots import make_subplots as _pmsp

    # Fig EDA-1: WTI price overview + winsorization + crisis shading
    _fig_eda1 = _pgo.Figure()
    _fig_eda1.add_trace(_pgo.Scatter(x=wti_raw.index.strftime("%Y-%m-%d"),
                                     y=wti_raw["price"], name="Actual (raw)",
                                     line=dict(color="#1f3b73", width=1.0)))
    _fig_eda1.add_trace(_pgo.Scatter(x=wti_wins.index.strftime("%Y-%m-%d"),
                                     y=wti_wins["price"], name="Winsorized",
                                     line=dict(color="#d1495b", width=1.0, dash="dot")))
    add_crises(_fig_eda1)
    _fig_eda1.update_layout(
        title="WTI Crude Oil Price 1986→Present  (raw + winsorized)",
        xaxis_title="Date", yaxis_title="$/barrel")
    save_fig(_fig_eda1, "eda_01_wti_price_overview")

    # Fig EDA-2: 21-day rolling volatility (annualised log-return std)
    _rol_vol = (df["log_return"].rolling(21).std() * (252**0.5) * 100).dropna() if "log_return" in df.columns else None
    if _rol_vol is not None:
        _fig_eda2 = _pgo.Figure()
        _fig_eda2.add_trace(_pgo.Scatter(x=_rol_vol.index.strftime("%Y-%m-%d"),
                                         y=_rol_vol, name="21d Rolling Vol (%)",
                                         line=dict(color="#2a9d8f", width=1.2)))
        add_crises(_fig_eda2)
        _fig_eda2.update_layout(
            title="WTI 21-Day Rolling Volatility (annualised log-return σ × √252)",
            xaxis_title="Date", yaxis_title="Annualised Vol (%)")
        save_fig(_fig_eda2, "eda_02_rolling_volatility")

    # Fig EDA-3: ACF/PACF bar chart of log-returns (up to 60 lags)
    if "log_return" in df.columns:
        from pandas.plotting import autocorrelation_plot as _acp
        _lr = df["log_return"].dropna()
        _lags_n = min(60, len(_lr)//3)
        _acf_vals  = [float(_lr.autocorr(lag=k)) for k in range(1, _lags_n+1)]
        _conf_band = 1.96 / len(_lr)**0.5
        _fig_eda3  = _pmsp(rows=1, cols=2, subplot_titles=["ACF (log-return)", "PACF (log-return)"])
        _fig_eda3.add_trace(_pgo.Bar(x=list(range(1, _lags_n+1)), y=_acf_vals, name="ACF",
                                     marker_color="#1f3b73"), row=1, col=1)
        _fig_eda3.add_hline(y=_conf_band, line_dash="dash", line_color="red", row=1, col=1)
        _fig_eda3.add_hline(y=-_conf_band, line_dash="dash", line_color="red", row=1, col=1)
        try:
            from statsmodels.tsa.stattools import pacf as _pacf_fn
            _pacf_vals = list(_pacf_fn(_lr.values, nlags=_lags_n)[1:])
            _fig_eda3.add_trace(_pgo.Bar(x=list(range(1, _lags_n+1)), y=_pacf_vals,
                                         name="PACF", marker_color="#d1495b"), row=1, col=2)
            _fig_eda3.add_hline(y=_conf_band,  line_dash="dash", line_color="red", row=1, col=2)
            _fig_eda3.add_hline(y=-_conf_band, line_dash="dash", line_color="red", row=1, col=2)
        except Exception:
            pass
        _fig_eda3.update_layout(title="Log-Return ACF / PACF (±1.96/√N 95% band)",
                                showlegend=False)
        save_fig(_fig_eda3, "eda_03_acf_pacf")

    # Fig EDA-4: Regime timeline (HMM states as colour-coded background)
    if "regime_state" in df.columns:
        _fig_eda4 = _pgo.Figure()
        _fig_eda4.add_trace(_pgo.Scatter(
            x=wti_raw.index.strftime("%Y-%m-%d"), y=wti_raw["price"],
            name="WTI Price", line=dict(color="#1f3b73", width=1.0)))
        _n_reg = int(df["regime_state"].nunique())
        _reg_palette = ["#2a9d8f","#e9a23b","#d1495b","#6a4c93","#118ab2"]
        for _ri in range(_n_reg):
            _rmask = df["regime_state"] == _ri
            _rsegs = df[_rmask].index
            if len(_rsegs) == 0: continue
            # shade contiguous segments
            _in_seg = False; _seg_start = None
            for _dt in df.index:
                if _rmask.loc[_dt] and not _in_seg:
                    _in_seg = True; _seg_start = _dt
                elif not _rmask.loc[_dt] and _in_seg:
                    _in_seg = False
                    _fig_eda4.add_vrect(
                        x0=_seg_start.strftime("%Y-%m-%d"),
                        x1=_dt.strftime("%Y-%m-%d"),
                        fillcolor=_reg_palette[_ri % len(_reg_palette)],
                        opacity=0.12, line_width=0,
                        annotation_text=f"R{_ri}" if _in_seg else "",
                        row="all", col="all")
            if _in_seg:
                _fig_eda4.add_vrect(
                    x0=_seg_start.strftime("%Y-%m-%d"),
                    x1=df.index[-1].strftime("%Y-%m-%d"),
                    fillcolor=_reg_palette[_ri % len(_reg_palette)],
                    opacity=0.12, line_width=0)
        _fig_eda4.update_layout(
            title="WTI Price + HMM Regime Timeline  (coloured bands = distinct market states)",
            xaxis_title="Date", yaxis_title="$/barrel")
        save_fig(_fig_eda4, "eda_04_regime_timeline")

    # Fig EDA-5: Feature correlation heatmap (top correlated pairs with log_return)
    _num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    _lr_col = "log_return" if "log_return" in _num_cols else None
    if _lr_col and len(_num_cols) > 3:
        _corr = df[_num_cols].corr()[[_lr_col]].drop(_lr_col).sort_values(_lr_col, ascending=False)
        _top_n = 30
        _ct = _corr.iloc[:_top_n//2].append(_corr.iloc[len(_corr)-_top_n//2:]) if len(_corr) > _top_n else _corr
        _fig_eda5 = _pgo.Figure(_pgo.Bar(
            x=_ct[_lr_col].values, y=_ct.index.tolist(), orientation="h",
            marker_color=["#2a9d8f" if v > 0 else "#d1495b" for v in _ct[_lr_col].values]))
        _fig_eda5.update_layout(
            title=f"Feature → log_return Correlation (top ±{_top_n//2})",
            xaxis_title="Pearson correlation", yaxis_title="Feature",
            height=max(400, 18*len(_ct)))
        save_fig(_fig_eda5, "eda_05_feature_correlation")

    print("  ✅ EDA figures saved to figures/")
except Exception as _eda_err:
    print(f"  [warn] EDA figures skipped: {_eda_err}")

print("\n✅ FILE 00 COMPLETE — pickles + EDA figures saved")
print("   df_features.pkl, wti_raw.pkl, wti_wins.pkl, feat_cols.pkl, extreme_events.pkl")
