# ════════════════════════════════════════════════════════════════════
#  FILE : B_wf_timemixer.py
#  ROLE : Track B honest walk-forward driver — TimeMixer
# ════════════════════════════════════════════════════════════════════
#  WHAT — One-line driver: sets MODEL_NAME="TimeMixer" then execs
#         _run_wf_arch.py which trains TimeMixer over the full set of
#         TRACK_HORIZONS using the honest walk-forward protocol.
#
#  WHY  — Track B is the DEFENSIBLE thesis benchmark: only past data
#         is visible at each forecast origin (no lookahead leakage).
#         These MAEs are what the thesis actually claims. Tagged leaky=False.
#
#  HOW  — _run_wf_arch.py wraps _features_plus, _crisis_weighting,
#         _fusion, and _forecast_template in a skip-guarded horizon loop.
#         Resume: already-done (model x horizon) pairs are skipped.
#
#  READS    : df_features.pkl, feat_cols.pkl, wti_raw.pkl
#  WRITES   : predictions_TimeMixer_{hk}.csv, eval_TimeMixer_{hk}.pkl
#             figures/model_TimeMixer_{hk}.html|.png (per-horizon)
#  COMPUTE  : GPU-HEAVY  (run on Colab / Kaggle A100)
#  PIPELINE : Stage 3 Track B; Stage 0 artifacts must exist
#  RUNTIME  : ~3-5 min per horizon on GPU  (4 horizons ≈ 15-20 min)
# ════════════════════════════════════════════════════════════════════
MODEL_NAME = "TimeMixer"
exec(open("_run_wf_arch.py", encoding="utf-8").read())