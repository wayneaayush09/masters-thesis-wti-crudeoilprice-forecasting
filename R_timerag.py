# ════════════════════════════════════════════════════════════════════
#  FILE : R_timerag.py
#  ROLE : Track B — TimeRAG analog-retrieval forecaster (leakage-free)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Implements a Retrieval-Augmented Generation (RAG) approach to
#         time-series forecasting: for each test origin, it retrieves
#         the k=5 most historically-similar past price windows from a
#         training-only knowledge base, then produces a similarity-weighted
#         "analog ensemble" forecast.  Tagged leaky=False.
#         Also generates RAG feature columns (analog mean, dispersion,
#         similarity scores) appended to df_features for downstream models.
#
#  WHY  — Analog forecasting ("find similar past episodes and see what
#         happened next") is a non-parametric, interpretable approach that
#         requires zero training time.  For WTI specifically, it captures
#         structural recurrences (OPEC shock patterns, demand-collapse shapes)
#         that a NN must learn from scratch.  TimeRAG also produces an
#         uncertainty estimate (dispersion of the k retrieved futures) at
#         no extra cost.  As a CPU-only method with zero hyperparameters,
#         it is an ideal ablation baseline: "how much value does the NN
#         architecture add over pure pattern matching?"
#
#  HOW  — Leakage-safe construction:
#           1. Build knowledge base from training window only
#              (log_price[:train_len] as non-overlapping segments of length L).
#           2. At test origin t: z-normalise log_price[t-L:t],
#              compute L2 distance to all KB entries, retrieve top-k.
#           3. For each analog: look at its "next h steps" in the training
#              data; weight by exp(-distance²); average → forecast.
#           4. Append rag_analog_mean, rag_analog_dispersion, rag_sim_score
#              to df_features (these become features for B_wf_* models).
#
#  READS    : df_features.pkl (log_price)
#  WRITES   : predictions_RAGForecaster_{hk}.csv
#             eval_RAGForecaster_{hk}.pkl
#             (updates df_features.pkl with RAG feature columns)
#  COMPUTE  : CPU-ONLY; O(N_test × N_KB × L) distance search — fast
#  PIPELINE : Stage 3 Track B — run after Stage 0; before B_wf_*.py
#             (so RAG features are available as inputs to B_wf models)
#  RUNTIME  : < 5 min per horizon on CPU (FAISS or numpy L2)
# ════════════════════════════════════════════════════════════════════

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))

if "rag_forecast_series" not in globals():
    exec(open("_rag_forecaster.py", encoding="utf-8").read())
if "save_paper_predictions" not in globals():
    exec(open("_paper_helpers.py", encoding="utf-8").read())

HORIZONS    = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
K_NEIGHBORS = globals().get("K_NEIGHBORS", 5)
STRIDE      = globals().get("RAG_STRIDE", 3)   # predict every N days (speed)
SPLIT_RATIO = 0.7
METHOD      = "l2_norm"
_H          = {"1d":1,"7d":7,"90d":90,"180d":180}
_LB         = {"1d":30,"7d":60,"90d":180,"180d":360}

print("="*60)
print("  TimeRAG analog-retrieval forecaster")
print(f"  k={K_NEIGHBORS}  stride={STRIDE}  method={METHOD}")
print("="*60)

with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
    df = pickle.load(f)

log_price = df["log_price"].values.astype(float)
dates     = df.index
train_len = int(len(log_price) * SPLIT_RATIO)

# ── Step 1: compute RAG features for multivariate models ──────
print("\n[RAG] Computing analog feature columns (rag_forecast/move/disp)…")
try:
    rag_feats = rag_feature_columns(log_price, train_len,
                                    horizons=(1,7,90,180), lookback=30, k=K_NEIGHBORS)
    # Align to df index
    rag_feats.index = dates[rag_feats.index]
    for col in rag_feats.columns:
        df[col] = rag_feats[col].reindex(df.index)
    with open(OUTPUT_DIR / "df_features.pkl", "wb") as f2:
        pickle.dump(df, f2)
    print(f"  ✅ Added {len(rag_feats.columns)} RAG feature columns to df_features.pkl")
except Exception as e:
    print(f"  ⚠️  RAG feature computation failed ({e}); continuing with forecasts only")

# ── Step 2: per-horizon analog-ensemble forecast ───────────────
_results, _failures = {}, {}
for hk in HORIZONS:
    h  = _H[hk]; lb = _LB[hk]
    print(f"\n[RAG] Horizon {hk} (h={h}, lookback={lb})…")
    try:
        preds_dict = rag_forecast_series(log_price, train_len, h=h, lookback=lb,
                                         k=K_NEIGHBORS, method=METHOD, stride=STRIDE)
        # preds_dict[origin_idx] = {forecast: log_price_pred, …}
        # Convert to {target_idx: log_price} for save_paper_predictions
        lp_preds = {origin + h: v["forecast"] for origin, v in preds_dict.items()}
        if not lp_preds:
            print(f"  ⚠️  No predictions generated for {hk}"); continue

        # Also save analog features per origin
        feat_df = pd.DataFrame([
            {"origin": dates[o], "rag_analog_move": v["analog_move_mean"],
             "rag_dispersion": v["analog_dispersion"],
             "rag_p10": v["analog_p10"], "rag_p90": v["analog_p90"]}
            for o, v in preds_dict.items() if o < len(dates)
        ]).set_index("origin")
        feat_df.to_csv(OUTPUT_DIR / f"rag_features_{hk}.csv")

        ev = save_paper_predictions(lp_preds, log_price, dates, h=h, hk=hk,
                                    model_name="RAGForecaster",
                                    extra_meta={"leaky": False,
                                                "protocol": "rag_analog_retrieval",
                                                "arch": "RAGForecaster",
                                                "k_neighbors": K_NEIGHBORS,
                                                "method": METHOD})
        _results[hk] = ev
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ⚠️  RAGForecaster/{hk} FAILED: {e}")
        _failures[hk] = str(e)

print(f"\n{'='*60}\n  R_timerag complete — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
print("  Outputs: predictions_RAGForecaster_{hk}.csv, rag_features_{hk}.csv")
print("  RAG feature columns added to df_features.pkl (rag_forecast_*d, rag_move_*d, rag_disp_*d)")
print("  Run C_compare.py to rank RAGForecaster vs neural models.")
if _failures: print("  Failures:", _failures)
