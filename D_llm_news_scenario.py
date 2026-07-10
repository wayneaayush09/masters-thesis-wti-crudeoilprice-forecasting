# ============================================================
# DDP_v2/D_llm_news_scenario.py  — interactive news what-if tool
# Usage (from inside DDP_v2/, after exec):
#     news_scenario("attack on Saudi crude reserves", hk="7d")
#     news_scenario("strong economic growth in Saudi Arabia", hk="90d")
# Returns the structured score, the base price, the news-adjusted price, a
# confidence fan, and renders a scenario chart. Leakage-safe (hypothetical input).
# ============================================================
import os, pickle
import numpy as np
import pandas as pd

if "score_news" not in globals():
    exec(open("_llm_news.py", encoding="utf-8").read())

OUT = "outputs"


def _base_price_and_forecast(hk):
    """Latest spot price + best available base forecast for horizon hk (if any)."""
    with open(os.path.join(OUT, "df_features.pkl"), "rb") as f:
        df = pickle.load(f)
    spot = float(np.exp(df["log_price"].iloc[-1]))
    base_fc = spot   # default: random-walk base
    # use a model forecast if a predictions file exists for hk (prefer hybrid/linear/PatchTST)
    import glob
    prefer = ["LHYu_hybrid", "LHYu_dlinear", "PaperHuang_PatchTST", "predictions"]
    cands = glob.glob(os.path.join(OUT, f"predictions_*_{hk}.csv")) + \
            glob.glob(os.path.join(OUT, f"predictions_{hk}.csv"))
    if cands:
        def _rank(p):
            b = os.path.basename(p)
            return next((i for i, k in enumerate(prefer) if k in b), len(prefer))
        cands.sort(key=_rank)
        try:
            d = pd.read_csv(cands[0])
            if "pred" in d.columns and len(d):
                base_fc = float(d["pred"].iloc[-1])
        except Exception:
            pass
    return spot, base_fc


def news_scenario(text, hk="7d", show=True):
    score = score_news(text)
    shock = news_to_logret_shock(score, hk=hk)
    spot, base_fc = _base_price_and_forecast(hk)
    adj_fc = base_fc * float(np.exp(shock))
    # confidence fan: wider when confidence is low
    spread = abs(shock) * (1.5 - score["confidence"]) + 0.02
    lo, hi = adj_fc * np.exp(-spread), adj_fc * np.exp(spread)
    out = {"news": text, "horizon": hk, "score": score,
           "logret_shock": round(shock, 4), "spot": round(spot, 2),
           "base_forecast": round(base_fc, 2), "news_adjusted_forecast": round(adj_fc, 2),
           "fan_low": round(lo, 2), "fan_high": round(hi, 2),
           "pct_vs_base": round(100 * (adj_fc / base_fc - 1), 2)}
    if show:
        print("\n" + "=" * 60)
        print(f"  NEWS SCENARIO ({score.get('_source','?')})  —  horizon {hk}")
        print("=" * 60)
        print(f'  news        : "{text}"')
        print(f"  direction   : {score['direction']:+.2f}   magnitude: {score['magnitude_pct']:.1f}%"
              f"   conf: {score['confidence']:.2f}   factors: {score.get('factors')}")
        print(f"  spot        : ${spot:.2f}")
        print(f"  base fc     : ${base_fc:.2f}")
        print(f"  NEWS-adj fc : ${adj_fc:.2f}   ({out['pct_vs_base']:+.2f}% vs base)")
        print(f"  fan 80%     : [${lo:.2f}, ${hi:.2f}]")
        # optional chart (Plotly bar: spot / base / news-adjusted + 80% fan error bar)
        try:
            if "save_fig" not in globals():
                exec(open("_plotting.py", encoding="utf-8").read())
            import plotly.graph_objects as _pgoD
            _figD = _pgoD.Figure()
            _figD.add_trace(_pgoD.Bar(x=["spot", "base fc", "news-adj fc"],
                                      y=[spot, base_fc, adj_fc],
                                      marker_color=["#888", "#3b82f6", "#ef4444"]))
            _figD.add_trace(_pgoD.Scatter(x=["news-adj fc"], y=[adj_fc], mode="markers",
                                          marker=dict(color="black", size=8),
                                          error_y=dict(type="data", symmetric=False,
                                                       array=[hi - adj_fc], arrayminus=[adj_fc - lo]),
                                          showlegend=False))
            _figD.update_layout(title=f"News scenario @ {hk}: {text[:48]}",
                                yaxis_title="WTI price ($)")
            save_fig(_figD, "llm_news_scenario")
            print("  chart       : figures/llm_news_scenario (html+png)")
        except Exception:
            pass
    # append to a log csv
    try:
        row = {k: (v if not isinstance(v, dict) else None) for k, v in out.items()}
        row.update({"direction": score["direction"], "magnitude_pct": score["magnitude_pct"],
                    "confidence": score["confidence"]})
        p = os.path.join(OUT, "llm_news_scenarios.csv")
        pd.DataFrame([row]).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
    except Exception:
        pass
    return out


print("✅ D_llm_news_scenario.py loaded — call: news_scenario(\"<headline>\", hk=\"7d\")")
# demo (no-op safe even without API key — uses heuristic):
if globals().get("LLM_NEWS_DEMO", False):
    news_scenario("attack on Saudi crude oil reserves", hk="7d")
