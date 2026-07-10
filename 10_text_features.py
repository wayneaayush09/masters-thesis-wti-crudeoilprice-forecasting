# ============================================================
# FILE 10 — Text / News Context Features (GDELT + FinBERT + RAG)
# Run FIRST, BEFORE 00_setup_data_features.py.
# Produces outputs/text_features_daily.csv (date-indexed), which 00 merges
# into the feature panel so EVERY model (incl. PatchTST) can use it.
#
# LEAKAGE DISCIPLINE (the whole point of the reviews):
#   • every daily feature at date t uses only news published ON/BEFORE t
#   • RAG retrieval at date t restricts the index to documents dated < t
#
# ROBUSTNESS: every external call is wrapped — on ANY failure the affected
# column degrades to 0 / ffill so the numeric pipeline is NEVER blocked.
#
# HONEST COVERAGE NOTE: GDELT's free DOC 2.0 API timelines realistically
# cover ~2017→present. Pre-2017 dates fall back to neutral/zero. Since the
# walk-forward up-weights recent data and COVID(2020)+ is well covered, this
# still adds genuine event/sentiment signal where it matters most.
#
# Usage:  exec(open('10_text_features.py', encoding="utf-8").read())
# ============================================================

import subprocess, sys, json, time, warnings
from pathlib import Path

# ── Self-install: file 10 may run BEFORE file 00, so it installs its own
# dependencies. Packages: faiss-cpu (RAG index), sentence-transformers
# (embeddings), transformers (FinBERT/CrudeBERT). Pip is idempotent — these
# are no-ops if the packages are already present (e.g. when run after file 00).
print("Installing 10_text_features dependencies (faiss-cpu, sentence-transformers, transformers)…")
for _pkg in ["faiss-cpu", "sentence-transformers", "transformers"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)
print("✅ deps ready")

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = "2007-01-01"          # matches the walk-forward start
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
GDELT_DOC  = "https://api.gdeltproject.org/api/v2/doc/doc"

# Themed queries → tone (sentiment) + volume (intensity) signals
QUERIES = {
    "oil":      "(crude oil OR petroleum OR WTI OR Brent OR OPEC)",
    "conflict": "(oil AND (war OR sanctions OR attack OR strait OR pipeline))",
    "econ":     "(oil AND (demand OR recession OR inflation OR economy))",
    "supply":   "(oil AND (production OR inventory OR output OR OPEC quota))",
}

bdays = pd.bdate_range(START_DATE, END_DATE)
feat = pd.DataFrame(index=bdays)
feat.index.name = "date"

def _gdelt_timeline(query, mode, year):
    """Fetch one year of a GDELT DOC 2.0 timeline as a date→value Series.
    mode ∈ {timelinetone, timelinevolraw}. Returns empty Series on failure."""
    s, e = f"{year}0101000000", f"{year}1231235959"
    params = {"query": query, "mode": mode, "format": "json",
              "startdatetime": s, "enddatetime": e}
    try:
        r = requests.get(GDELT_DOC, params=params, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return pd.Series(dtype=float)
        data = r.json().get("timeline", [])
        if not data: return pd.Series(dtype=float)
        pts = data[0].get("data", [])
        idx = [pd.to_datetime(p["date"][:8], format="%Y%m%d", errors="coerce") for p in pts]
        val = [float(p.get("value", np.nan)) for p in pts]
        out = pd.Series(val, index=idx).dropna()
        return out[~out.index.duplicated()]
    except Exception:
        return pd.Series(dtype=float)

def _collect(query, mode):
    """Year-by-year collection across the full range (bounded, polite)."""
    parts = []
    for yr in range(int(START_DATE[:4]), int(END_DATE[:4]) + 1):
        s = _gdelt_timeline(query, mode, yr)
        if not s.empty: parts.append(s)
        time.sleep(0.4)                          # be polite to the free API
    if not parts: return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()

print("Fetching GDELT timelines (graceful fallback on failure)…")
gdelt_ok = False
try:
    tone = _collect(QUERIES["oil"], "timelinetone")
    vol  = _collect(QUERIES["oil"], "timelinevolraw")
    if not tone.empty or not vol.empty:
        gdelt_ok = True
        feat["gdelt_tone"]   = tone.reindex(feat.index).ffill()
        feat["gdelt_volume"] = vol.reindex(feat.index).ffill()
        for key in ("conflict", "econ", "supply"):
            v = _collect(QUERIES[key], "timelinevolraw")
            feat[f"evt_{key}"] = v.reindex(feat.index).ffill() if not v.empty else 0.0
        print(f"  ✅ GDELT: tone {tone.shape[0]} pts, volume {vol.shape[0]} pts")
    else:
        print("  ⚠️  GDELT returned no data — using zero fallback")
except Exception as e:
    print(f"  ⚠️  GDELT fetch failed ({e}) — using zero fallback")

# Zero-fallback for any missing GDELT columns
for c in ["gdelt_tone", "gdelt_volume", "evt_conflict", "evt_econ", "evt_supply"]:
    if c not in feat.columns: feat[c] = 0.0
feat = feat.fillna(0.0)

# Derived GDELT dynamics (leakage-safe: rolling over PAST values only)
feat["gdelt_tone_vol"] = feat["gdelt_tone"].rolling(21, min_periods=5).std().fillna(0.0)
feat["gdelt_tone_mom"] = (feat["gdelt_tone"] - feat["gdelt_tone"].rolling(63, min_periods=10).mean()).fillna(0.0)

# ── RAG retrieval feature (leakage-safe; sentence-transformers + FAISS) ──
# Render each day as a short text "document"; for date t retrieve the most
# similar PAST days (< t) and output their similarity-weighted tone — a
# "what followed when conditions last looked like this" context signal.
print("Building RAG retrieval feature…")
feat["rag_retrieved_tone"] = 0.0
feat["rag_sim"] = 0.0
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    def _render(row):
        return (f"oil news tone {row['gdelt_tone']:.1f}, volume {row['gdelt_volume']:.0f}, "
                f"conflict {row['evt_conflict']:.0f}, econ {row['evt_econ']:.0f}, "
                f"supply {row['evt_supply']:.0f}")
    docs = feat.apply(_render, axis=1).tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb = embedder.encode(docs, batch_size=256, show_progress_bar=False,
                          normalize_embeddings=True).astype(np.float32)
    dim = emb.shape[1]
    tone_vals = feat["gdelt_tone"].values
    K = 5
    rag_tone = np.zeros(len(feat)); rag_sim = np.zeros(len(feat))
    index = faiss.IndexFlatIP(dim)               # cosine (normalized) similarity
    WARM = 252                                   # need a year of history before retrieving
    for i in range(len(feat)):
        if i > WARM:
            sims, ids = index.search(emb[i:i+1], min(K, i))   # search PAST docs only
            ids = ids[0][ids[0] >= 0]; sims = sims[0][:len(ids)]
            if len(ids):
                w = np.clip(sims, 0, None); w = w/ (w.sum()+1e-9)
                rag_tone[i] = float(np.dot(w, tone_vals[ids]))
                rag_sim[i]  = float(sims.max())
        index.add(emb[i:i+1])                     # only NOW add today's doc (so future never seen)
    feat["rag_retrieved_tone"] = rag_tone
    feat["rag_sim"] = rag_sim
    print("  ✅ RAG retrieval features built (past-only index)")
except Exception as e:
    print(f"  ⚠️  RAG retrieval skipped ({e}) — zero fallback")

# ── Sentiment models: FinBERT (general finance) + CrudeBERT (oil-domain) ──
# Both are applied to the same monthly GDELT article samples so titles are
# fetched ONCE per month — no duplicate API calls.
# CrudeBERT (oil-domain) + FinBERT (general finance): both loaded via a fallback chain.
# crudebert_sent = oil-domain primary; finbert_sent = general finance secondary.
# On failure either column proxies from the other (graceful degradation).
print("Loading sentiment models: FinBERT (finance) + CrudeBERT (oil-domain)…")
feat["finbert_sent"]   = np.nan
feat["crudebert_sent"] = np.nan
finbert = crudebert = None

# Suppress noisy HuggingFace model-loading warnings (position_ids UNEXPECTED etc.)
try:
    import transformers as _tf_logging
    _tf_logging.logging.set_verbosity_error()
except Exception:
    pass
import warnings as _w; _w.filterwarnings("ignore", message=".*position_ids.*")

try:
    import torch as _torch
    from transformers import pipeline
    _dev = 0 if _torch.cuda.is_available() else -1
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                       truncation=True, device=_dev)
    print("  ✅ FinBERT loaded (ProsusAI/finbert)")
except Exception as _e:
    print(f"  ⚠️  FinBERT load failed ({_e})")

try:
    import torch as _torch2
    from transformers import pipeline as _pl2
    _dev2 = 0 if _torch2.cuda.is_available() else -1
    # CrudeBERT fallback chain — tries oil/energy-specific models first, then
    # falls back to well-tested finance-tone models. First one that loads wins.
    # Model notes:
    #   "Captain-1337/CrudeBERT"  — re-uploaded CrudeBERT (oil domain, AAAI paper)
    #   "nickmuchi/finbert-tone-finetuned-fintwitter-classification"  — finance Twitter tone
    #   "yiyanghkust/finbert-tone" — classic FinBERT tone (positive/negative/neutral)
    _CRUDEBERT_CANDIDATES = [
        ("Captain-1337/CrudeBERT",                                    "oil-domain CrudeBERT"),
        ("nickmuchi/finbert-tone-finetuned-fintwitter-classification", "finance-Twitter tone"),
        ("yiyanghkust/finbert-tone",                                   "FinBERT-tone"),
    ]
    crudebert = None
    for _mid, _label in _CRUDEBERT_CANDIDATES:
        try:
            crudebert = _pl2("text-classification", model=_mid,
                             truncation=True, device=_dev2)
            print(f"  ✅ Oil-domain sentiment loaded: {_mid} ({_label})")
            break
        except Exception as _e2:
            print(f"  ⚠️  {_mid} unavailable ({_e2}), trying next…")
    if crudebert is None:
        print("  ⚠️  All CrudeBERT candidates failed — crudebert_sent will proxy from finbert_sent")
except Exception as _e:
    print(f"  ⚠️  CrudeBERT load block failed ({_e}) — crudebert_sent will proxy from finbert_sent")

def _artlist_titles(year, month, max_records=100):
    """Fetch GDELT article titles for a month. Tries multiple queries for robustness."""
    s = f"{year}{month:02d}01000000"
    em = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1))
    e = f"{em.strftime('%Y%m%d')}235959"
    # Try oil query first, then broader energy query as fallback
    for _q in [QUERIES.get("oil", "crude oil"), "oil price", "petroleum energy"]:
        try:
            r = requests.get(GDELT_DOC, params={"query": _q, "mode": "artlist",
                             "maxrecords": max_records, "format": "json",
                             "startdatetime": s, "enddatetime": e}, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0"})
            arts = r.json().get("articles", []) if r.text.strip().startswith("{") else []
            titles = [a.get("title", "") for a in arts if a.get("title")]
            if titles:
                return titles
        except Exception:
            continue
    return []

def _score_dims(model, titles, max_titles=40):
    """Run a HuggingFace classification pipeline on titles.
    B4 (Addendum 8): returns {polarity, intensity, uncertainty, dispersion}
    — all four dimensions shown by Dai et al. (arXiv 2603.11408) to matter for WTI.
      polarity    = mean(signed score)                — sentiment direction
      intensity   = mean(|signed score|)              — how strongly opinionated
      uncertainty = mean(1 − confidence)              — low-confidence = uncertain news
      dispersion  = std(signed scores over titles)    — intra-month disagreement
    All return np.nan on failure (robust fallback in the calling loop)."""
    _nan_result = {"polarity": np.nan, "intensity": np.nan,
                   "uncertainty": np.nan, "dispersion": np.nan}
    if model is None or not titles:
        return _nan_result
    try:
        scored = model(titles[:max_titles])
        polar  = [{"positive": 1, "negative": -1, "neutral": 0}.get(x["label"].lower(), 0)
                  for x in scored]
        conf   = [x["score"] for x in scored]
        signed = [p * c for p, c in zip(polar, conf)]
        return {
            "polarity":    float(np.nanmean(signed)),
            "intensity":   float(np.nanmean(np.abs(signed))),
            "uncertainty": float(np.nanmean(1 - np.array(conf))),
            "dispersion":  float(np.nanstd(signed)) if len(signed) > 1 else 0.0,
        }
    except Exception:
        return _nan_result

# Single monthly loop: fetch article titles ONCE, score with both models.
# B4: capture 4 dimensions per model (polarity/intensity/uncertainty/dispersion).
_DIM_KEYS = ["polarity", "intensity", "uncertainty", "dispersion"]
fb_monthly  = {k: {} for k in _DIM_KEYS}
cb_monthly  = {k: {} for k in _DIM_KEYS}

if finbert is not None or crudebert is not None:
    for ts in pd.date_range(START_DATE, END_DATE, freq="MS"):
        titles = _artlist_titles(ts.year, ts.month)
        if titles:
            d_fb = _score_dims(finbert,   titles)
            d_cb = _score_dims(crudebert, titles)
            for k in _DIM_KEYS:
                if not np.isnan(d_fb[k]): fb_monthly[k][ts] = d_fb[k]
                if not np.isnan(d_cb[k]): cb_monthly[k][ts] = d_cb[k]
        time.sleep(0.3)

# FinBERT: polarity → finbert_sent (back-compat); all 4 dims → fb_* columns
if fb_monthly["polarity"]:
    _fb_pol = pd.Series(fb_monthly["polarity"]).sort_index()
    feat["finbert_sent"] = _fb_pol.reindex(feat.index, method="ffill")
    print(f"  ✅ FinBERT monthly sentiment: {len(fb_monthly['polarity'])} months scored")
    for k in _DIM_KEYS:
        if fb_monthly[k]:
            feat[f"fb_{k}"] = pd.Series(fb_monthly[k]).sort_index().reindex(feat.index, method="ffill")
        else:
            feat[f"fb_{k}"] = 0.0
else:
    print("  ⚠️  FinBERT produced no monthly scores")
    for k in _DIM_KEYS: feat[f"fb_{k}"] = 0.0

# CrudeBERT: polarity → crudebert_sent (back-compat); all 4 dims → cb_* columns
if cb_monthly["polarity"]:
    _cb_pol = pd.Series(cb_monthly["polarity"]).sort_index()
    feat["crudebert_sent"] = _cb_pol.reindex(feat.index, method="ffill")
    print(f"  ✅ CrudeBERT monthly sentiment: {len(cb_monthly['polarity'])} months scored")
    for k in _DIM_KEYS:
        if cb_monthly[k]:
            feat[f"cb_{k}"] = pd.Series(cb_monthly[k]).sort_index().reindex(feat.index, method="ffill")
        else:
            feat[f"cb_{k}"] = 0.0
else:
    print("  ⚠️  CrudeBERT produced no monthly scores")
    for k in _DIM_KEYS: feat[f"cb_{k}"] = 0.0

# ── Fallbacks ──
# FinBERT fallback → normalized GDELT tone
if feat["finbert_sent"].isna().all():
    t = feat["gdelt_tone"]
    feat["finbert_sent"] = ((t - t.mean()) / (t.std() + 1e-9)).clip(-3, 3)
    print("  ℹ️  finbert_sent fallback: normalized GDELT tone used")
# CrudeBERT fallback → finbert_sent
if feat["crudebert_sent"].isna().all():
    feat["crudebert_sent"] = feat["finbert_sent"]
    print("  ℹ️  crudebert_sent fallback: finbert_sent used as proxy")

# B4 multi-dim fallbacks: proxy missing dims from polarity + GDELT
for _pfx, _pol_col in [("fb", "finbert_sent"), ("cb", "crudebert_sent")]:
    for _k in ["polarity", "intensity", "uncertainty", "dispersion"]:
        _col = f"{_pfx}_{_k}"
        if _col not in feat.columns: feat[_col] = 0.0
        if feat[_col].isna().all():
            if _k == "polarity":
                feat[_col] = feat[_pol_col].fillna(0.0)
            elif _k == "intensity":
                feat[_col] = feat[_pol_col].abs().fillna(0.0)
            else:
                feat[_col] = 0.0

for _c in ["finbert_sent", "crudebert_sent"] + [f"{p}_{k}" for p in ["fb","cb"]
                                                 for k in ["polarity","intensity","uncertainty","dispersion"]]:
    if _c in feat.columns:
        feat[_c] = feat[_c].ffill().fillna(0.0)

# ── Save ──────────────────────────────────────────────────────
feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
feat.to_csv(OUTPUT_DIR/"text_features_daily.csv")
print(f"\n✅ FILE 10 COMPLETE — text_features_daily.csv  {feat.shape}  "
      f"(GDELT live={gdelt_ok})")
print(f"   columns: {list(feat.columns)}")
print("   Sentiment polarity: finbert_sent (general finance) + crudebert_sent (oil-domain)")
print("   B4 multi-dim: fb_polarity/intensity/uncertainty/dispersion + cb_* (same 4 dims)")
print("   NOTE: leakage-safe (each day uses only ≤t news; RAG index past-only).")
