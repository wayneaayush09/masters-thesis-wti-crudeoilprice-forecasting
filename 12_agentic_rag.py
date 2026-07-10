# ============================================================
# FILE 12 — Agentic RAG Directional/Risk Signal
# A ReAct-style loop using a LOCAL small instruct model: on anomaly origins
# (flagged by the HMM regime / extreme_price_flag), retrieve the most similar
# PAST news-context days (< t, leakage-safe) from a FAISS index and synthesize
# a structured signal {direction ∈ [-1,1], risk ∈ [0,1], rationale}.
#
# HONEST SCOPING: this is a simplified, reproducible agent (local model with a
# deterministic fallback), evaluated as a DIRECTIONAL OVERLAY — does its up/down
# call beat a coin flip on flagged dates? It is NOT silently fused into every
# model's point forecast.
#
# Run AFTER 10 (text features) + 00. Saves agentic_rag_signal.csv.
# Usage:  exec(open('12_agentic_rag.py', encoding="utf-8").read())
# ============================================================

import pickle, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df      = pickle.load(f)
with open(OUTPUT_DIR/"wti_raw.pkl",    "rb") as f: wti_raw = pickle.load(f)

EVAL_H      = 7        # evaluate the agent's direction call on the 7-day move
MAX_ANOMALY = 200      # cap LLM calls (most recent flagged origins)
TOPK        = 5
INSTRUCT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small, local; fallback if unavailable

# ── Identify anomaly origins (where context should matter most) ──
flag = pd.Series(False, index=df.index)
if "extreme_price_flag" in df: flag |= df["extreme_price_flag"].astype(bool)
if "regime_vol_rank" in df:    flag |= (df["regime_vol_rank"] >= 0.66)
if "log_ret_5d" in df:         flag |= (df["log_ret_5d"].abs() > df["log_ret_5d"].std()*2)
anomaly_dates = df.index[flag.values]
anomaly_dates = anomaly_dates[anomaly_dates >= "2007-01-01"][-MAX_ANOMALY:]
print(f"Agentic RAG: {len(anomaly_dates)} anomaly origins (capped {MAX_ANOMALY})")

# ── Build leakage-safe retrieval over daily news-context docs ────
TEXT_COLS = [c for c in ["gdelt_tone","gdelt_volume","evt_conflict","evt_econ",
                         "evt_supply","gdelt_tone_vol","gdelt_tone_mom",
                         "rag_retrieved_tone","finbert_sent"] if c in df.columns]
def render(dt):
    if not TEXT_COLS: return f"date {dt.date()}: no news context available"
    r = df.loc[dt, TEXT_COLS]
    return ("; ".join(f"{c}={float(r[c]):.2f}" for c in TEXT_COLS))

embedder = faiss_index = emb_all = None
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [render(dt) for dt in df.index]
    emb_all = embedder.encode(docs, batch_size=256, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
    print("  ✅ retrieval embeddings ready")
except Exception as e:
    print(f"  ⚠️  retrieval unavailable ({e}) — agent will use current context only")

def retrieve_past(dt, k=TOPK):
    """Top-k most similar PAST days (strictly before dt)."""
    if emb_all is None: return []
    pos = df.index.get_loc(dt)
    if pos < 30: return []
    import faiss
    idx = faiss.IndexFlatIP(emb_all.shape[1])
    idx.add(emb_all[:pos])                       # only days strictly before dt
    sims, ids = idx.search(emb_all[pos:pos+1], min(k, pos))
    return [(df.index[j], float(s)) for j, s in zip(ids[0], sims[0]) if j >= 0]

# ── Local instruct model (graceful fallback to heuristic) ────────
generator = None
try:
    import torch
    from transformers import pipeline
    generator = pipeline("text-generation", model=INSTRUCT_MODEL,
                         torch_dtype="auto", device_map="auto",
                         max_new_tokens=120, do_sample=False)
    print(f"  ✅ instruct model loaded: {INSTRUCT_MODEL}")
except Exception as e:
    print(f"  ⚠️  instruct model unavailable ({e}) — using deterministic heuristic agent")

def heuristic_signal(dt, retrieved):
    """Deterministic fallback: direction from tone momentum, risk from tone vol."""
    tone = float(df.loc[dt, "gdelt_tone"]) if "gdelt_tone" in df else 0.0
    mom  = float(df.loc[dt, "gdelt_tone_mom"]) if "gdelt_tone_mom" in df else 0.0
    vol  = float(df.loc[dt, "gdelt_tone_vol"]) if "gdelt_tone_vol" in df else 0.0
    direction = float(np.tanh(0.5*np.sign(tone) + 0.5*np.sign(mom)))
    risk = float(np.clip(abs(vol)/ (abs(vol)+1) + 0.3, 0, 1))
    return direction, risk, "heuristic: tone/momentum sign, risk from tone volatility"

def llm_signal(dt, retrieved):
    ctx = render(dt)
    past = "; ".join(f"{d.date()}(sim{ s:.2f})" for d, s in retrieved) or "none"
    prompt = (
        "You are an oil-market analyst. Based ONLY on the news-context features and the "
        "most similar past days, output a JSON object with keys: direction (a number in "
        "[-1,1], + means price likely up over the next week), risk (0..1), rationale (short).\n"
        f"Today context: {ctx}\nSimilar past days: {past}\n"
        "Respond with ONLY the JSON object.")
    try:
        out = generator(prompt)[0]["generated_text"][len(prompt):]
        start, end = out.find("{"), out.rfind("}")
        obj = json.loads(out[start:end+1])
        d = float(np.clip(obj.get("direction", 0.0), -1, 1))
        r = float(np.clip(obj.get("risk", 0.5), 0, 1))
        return d, r, str(obj.get("rationale", ""))[:160]
    except Exception:
        return heuristic_signal(dt, retrieved)     # fall back per-call

# ── Run the agent over anomaly origins ───────────────────────────
rows = []
for dt in anomaly_dates:
    retrieved = retrieve_past(dt)
    if generator is not None:
        d, r, why = llm_signal(dt, retrieved)
    else:
        d, r, why = heuristic_signal(dt, retrieved)
    tgt = dt + pd.tseries.offsets.BDay(EVAL_H)
    realized = np.nan
    if dt in wti_raw.index and tgt in wti_raw.index:
        realized = float(np.sign(wti_raw.loc[tgt,"price"] - wti_raw.loc[dt,"price"]))
    rows.append({"date": dt, "direction": round(d,3), "risk": round(r,3),
                 "realized_dir": realized,
                 "correct": (np.sign(d) == realized) if (d != 0 and realized==realized) else np.nan,
                 "rationale": why})

sig = pd.DataFrame(rows).set_index("date")
sig.to_csv(OUTPUT_DIR/"agentic_rag_signal.csv")

valid = sig["correct"].dropna()
hit = float(valid.mean())*100 if len(valid) else np.nan
mode = "LLM" if generator is not None else "heuristic"
print(f"\n✅ FILE 12 COMPLETE — agentic_rag_signal.csv ({len(sig)} signals, {mode} agent)")
if hit == hit:
    print(f"   Directional hit-rate on flagged {EVAL_H}d moves: {hit:.1f}%  (vs 50% coin-flip)")
    print(f"   n={len(valid)} evaluated. >50% suggests the news/agent overlay carries directional info.")
print("   NOTE: leakage-safe (retrieval index strictly < origin date); reported as an overlay,")
print("   NOT fused into model point forecasts.")
