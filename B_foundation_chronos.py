# ════════════════════════════════════════════════════════════════════
#  FILE : B_foundation_chronos.py
#  ROLE : Track B — Amazon Chronos-Bolt zero-shot foundation model
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs Amazon Chronos-Bolt in zero-shot mode on the raw WTI
#         log-price series for all TRACK_HORIZONS.  No fine-tuning;
#         the model generates predictive distributions from its pre-
#         trained weights.  Tagged leaky=False (honest walk-forward:
#         each prediction sees only past data up to the origin date).
#
#  WHY  — Foundation models (Chronos, TimesFM) are trained on billions
#         of diverse time-series and generalise to new domains zero-shot.
#         Including Chronos-Bolt quantifies the "free lunch" question:
#         does a pre-trained foundation model — with ZERO task-specific
#         training — outperform specialised NN models trained for hundreds
#         of epochs on WTI?  The answer frames the foundation-model
#         contribution in the thesis.  Chronos also provides calibrated
#         predictive quantiles, supporting the probabilistic chapter.
#
#  HOW  — Calls run_foundation_standalone(_hk, model="chronos", stride=STRIDE)
#         from _foundation.py which:
#           1. Lazy-loads the Chronos-Bolt model (huggingface/amazon).
#           2. At each test origin (sampled every STRIDE days for speed),
#              passes log_price[t-lookback:t] as the context window.
#           3. Samples 500 quantile paths → median as point forecast.
#           4. Saves predictions_Chronos_{hk}.csv + eval_Chronos_{hk}.pkl.
#         STRIDE controls evaluation density vs. compute cost (default 10).
#
#  READS    : df_features.pkl (log_price, dates)
#  WRITES   : predictions_Chronos_{hk}.csv, eval_Chronos_{hk}.pkl
#  COMPUTE  : GPU-OPTIONAL (CPU feasible but slow); model ~290M params
#  PIPELINE : Stage 3 Track B — parallel with B_wf_*.py
#  RUNTIME  : ~5–15 min per horizon on GPU (depends on STRIDE)
# ════════════════════════════════════════════════════════════════════
# Track B — Amazon Chronos-Bolt zero-shot, standalone on raw log-price (honest).
# Run from inside DDP_v2/.
if "run_foundation_standalone" not in globals():
    exec(open("_foundation.py", encoding="utf-8").read())
for _hk in globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"]):
    try:
        run_foundation_standalone(_hk, model="chronos", stride=globals().get("STRIDE", 10))
    except Exception as _e:
        print(f"  ⚠️  chronos/{_hk} FAILED: {_e}")
