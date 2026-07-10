# ============================================================
# DDP_v2/_llm_news.py  — LLM news-impact module (Google Gemini, free tier)
# Type free-text world news -> Gemini returns a STRUCTURED impact score ->
# map to a log-return shock -> adjust the base price forecast + scenario fan.
#
# Three capabilities (helpers here; interactive driver = D_llm_news_scenario.py):
#   1) score_news(text)                  -> {direction, magnitude_pct, horizon_days, confidence, factors}
#   2) news_to_logret_shock(score, hk)   -> calibrated/vol-scaled log-return shock
#   3) calibrate_from_history(...)       -> fit score->realized-move mapping (backtest)
#
# Backend: Gemini via `google-generativeai`, GOOGLE_API_KEY from env/Kaggle secret.
# Graceful fallback to a deterministic heuristic if no key / package.
# Leakage-safe: interactive = user hypothetical; feature/backtest = strictly past-dated.
# Run / exec from inside DDP_v2/.
# ============================================================
import os, json, pickle, re
import numpy as np
import pandas as pd

OUT = "outputs"
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

_gemini = None
if _API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_API_KEY)
        _gemini = genai.GenerativeModel(_GEMINI_MODEL)
        print(f"✅ Gemini ready ({_GEMINI_MODEL})")
    except Exception as e:
        print(f"⚠️  Gemini init failed ({e}) — using heuristic fallback")
else:
    print("⚠️  No GOOGLE_API_KEY — LLM news module uses deterministic heuristic fallback")

_PROMPT = """You are a crude-oil (WTI) market analyst. Given a news statement, estimate its
impact on the WTI crude oil price. Respond ONLY with compact JSON, no prose:
{{"direction": <float -1..1, negative=bearish>,
  "magnitude_pct": <float, expected absolute % price move>,
  "horizon_days": <int, days over which the move plays out>,
  "confidence": <float 0..1>,
  "factors": [<short strings>]}}
News: "{news}"
JSON:"""

_HEUR_WORDS = {  # crude lexicon fallback (direction, magnitude%)
    "attack": (1.0, 8), "war": (1.0, 7), "sanction": (0.8, 6), "outage": (0.9, 6),
    "cut": (0.7, 5), "opec": (0.6, 4), "hurricane": (0.7, 5), "shortage": (0.8, 6),
    "recession": (-0.9, 7), "glut": (-0.8, 6), "oversupply": (-0.8, 6),
    "demand fall": (-0.7, 5), "ceasefire": (-0.6, 4), "deal": (-0.4, 3),
    "growth": (0.5, 3), "build": (-0.4, 3), "draw": (0.4, 3),
}


def _heuristic_score(text):
    t = text.lower(); d, m = 0.0, 2.0
    for w, (dd, mm) in _HEUR_WORDS.items():
        if w in t:
            d += dd; m = max(m, mm)
    d = float(np.clip(d, -1, 1))
    return {"direction": d, "magnitude_pct": float(m), "horizon_days": 7,
            "confidence": 0.4, "factors": ["heuristic"], "_source": "heuristic"}


def score_news(text):
    """Return a structured impact score for a free-text news string."""
    if _gemini is not None:
        try:
            resp = _gemini.generate_content(_PROMPT.format(news=text.replace('"', "'")))
            raw = resp.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            js = json.loads(m.group(0) if m else raw)
            return {"direction": float(np.clip(js.get("direction", 0), -1, 1)),
                    "magnitude_pct": float(abs(js.get("magnitude_pct", 2.0))),
                    "horizon_days": int(js.get("horizon_days", 7)),
                    "confidence": float(np.clip(js.get("confidence", 0.5), 0, 1)),
                    "factors": list(js.get("factors", []))[:6], "_source": "gemini"}
        except Exception as e:
            print(f"  [llm_news] Gemini call failed ({e}) — heuristic fallback")
    return _heuristic_score(text)


def _latest_vol():
    """Annualized-ish daily vol proxy from df_features (realized_vol_21 or vol_30d)."""
    try:
        with open(os.path.join(OUT, "df_features.pkl"), "rb") as f:
            df = pickle.load(f)
        for c in ("realized_vol_21", "vol_30d", "vol_90d"):
            if c in df.columns and np.isfinite(df[c].iloc[-1]):
                return float(df[c].iloc[-1]) or 0.02
    except Exception:
        pass
    return 0.02


# optional calibration mapping (score_signed -> realized log-move), fit by calibrate_from_history
_CALIB = None
_HK_DAYS = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}


def news_to_logret_shock(score, hk="7d"):
    """Map a structured score to a horizon-hk log-return shock.
    Calibrated if calibrate_from_history() has run; else vol-scaled heuristic."""
    signed = score["direction"] * (score["magnitude_pct"] / 100.0) * score["confidence"]
    if _CALIB is not None:
        a, b = _CALIB
        base = a + b * signed                          # calibrated total move
    else:
        base = signed                                  # heuristic: % move as log-approx
    # scale from the news' own horizon to the requested forecast horizon (sqrt-time, capped)
    nd = max(1, score.get("horizon_days", 7)); hd = _HK_DAYS.get(hk, 7)
    scale = min(np.sqrt(hd / nd), 2.0) if hd >= nd else (hd / nd)
    return float(base * scale)


def calibrate_from_history(events=None, hk="7d"):
    """Fit a 1-D regression: realized hk log-move ~ signed news score, over past events.
    `events` = list of (date_str, news_text). If None, derive proxy events from GDELT
    tone spikes in df_features (strictly past-dated). Stores _CALIB and writes a report."""
    global _CALIB
    with open(os.path.join(OUT, "df_features.pkl"), "rb") as f:
        df = pickle.load(f)
    lp = df["log_price"].astype(float); hd = _HK_DAYS.get(hk, 7)
    xs, ys = [], []
    if events:
        for date_str, text in events:
            try:
                i = df.index.get_indexer([pd.Timestamp(date_str)], method="nearest")[0]
                if i + hd >= len(df):
                    continue
                s = score_news(text)
                xs.append(s["direction"] * s["magnitude_pct"] / 100.0 * s["confidence"])
                ys.append(float(lp.iloc[i + hd] - lp.iloc[i]))
            except Exception:
                continue
    else:
        # proxy: GDELT tone spikes as pseudo-events (sign of tone = direction)
        tone = df["gdelt_tone"] if "gdelt_tone" in df.columns else None
        if tone is not None:
            z = (tone - tone.rolling(126).mean()) / (tone.rolling(126).std() + 1e-9)
            idx = np.where(np.abs(z.values) > 1.5)[0]
            for i in idx:
                if i + hd >= len(df):
                    continue
                xs.append(float(np.sign(z.values[i]) * min(abs(z.values[i]) / 3, 0.08)))
                ys.append(float(lp.iloc[i + hd] - lp.iloc[i]))
    if len(xs) >= 20:
        xs = np.asarray(xs); ys = np.asarray(ys)
        b = float(np.cov(xs, ys)[0, 1] / (np.var(xs) + 1e-12)); a = float(ys.mean() - b * xs.mean())
        _CALIB = (a, b)
        hit = float(np.mean(np.sign(xs) == np.sign(ys)))
        rep = pd.DataFrame([{"hk": hk, "n_events": len(xs), "intercept": a, "slope": b,
                             "direction_hit_rate": round(hit, 3)}])
        rep.to_csv(os.path.join(OUT, "llm_news_calibration.csv"), index=False)
        print(f"  [llm_news] calibrated hk={hk}: slope={b:.4f} hit-rate={hit:.2%} (n={len(xs)})")
    else:
        print(f"  [llm_news] not enough events to calibrate (n={len(xs)}) — using heuristic")
    return _CALIB


print("✅ DDP_v2/_llm_news.py loaded — score_news / news_to_logret_shock / calibrate_from_history")
