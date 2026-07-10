# ════════════════════════════════════════════════════════════════════
#  FILE : B_foundation_timesfm.py
#  ROLE : Track B — Google TimesFM zero-shot foundation model
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs Google TimesFM in zero-shot mode on the raw WTI log-price
#         series for all TRACK_HORIZONS.  No fine-tuning.  Tagged leaky=False
#         (each prediction uses only past data up to the origin date).
#
#  WHY  — TimesFM (Das et al. 2024) is pre-trained on 100B time-series
#         points from Google's internal data and Google Trends.  Paired
#         with Chronos, it provides a second independent foundation-model
#         data point for the zero-shot benchmarking chapter.  The two
#         foundation models are likely to agree where the signal is
#         strongest and disagree at crisis boundaries — that disagreement
#         itself is a forecasting signal (high ensemble variance → high
#         uncertainty).  Comparing Chronos vs TimesFM vs best Track B NN
#         answers "does model scale or task-specific training matter more?"
#
#  HOW  — Calls run_foundation_standalone(_hk, model="timesfm", stride=STRIDE)
#         from _foundation.py which:
#           1. Lazy-loads the TimesFM checkpoint (pip install timesfm or
#              huggingface model card).
#           2. At each test origin (every STRIDE days), passes the lookback
#              window to the model and reads back point forecast + intervals.
#           3. Saves predictions_TimesFM_{hk}.csv + eval_TimesFM_{hk}.pkl.
#         STRIDE=10 by default; set STRIDE=1 for full-density evaluation
#         at the cost of ~10× runtime.
#
#  READS    : df_features.pkl (log_price, dates)
#  WRITES   : predictions_TimesFM_{hk}.csv, eval_TimesFM_{hk}.pkl
#  COMPUTE  : GPU-OPTIONAL (200M params; CPU feasible for quick eval)
#  PIPELINE : Stage 3 Track B — parallel with B_wf_*.py and Chronos
#  RUNTIME  : ~5–15 min per horizon on GPU (STRIDE=10)
# ════════════════════════════════════════════════════════════════════
# Track B — Google TimesFM zero-shot, standalone on raw log-price (honest).
# Run from inside DDP_v2/.
if "run_foundation_standalone" not in globals():
    exec(open("_foundation.py", encoding="utf-8").read())
for _hk in globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"]):
    try:
        run_foundation_standalone(_hk, model="timesfm", stride=globals().get("STRIDE", 10))
    except Exception as _e:
        print(f"  ⚠️  timesfm/{_hk} FAILED: {_e}")
