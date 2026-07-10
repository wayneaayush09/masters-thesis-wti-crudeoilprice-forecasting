# ════════════════════════════════════════════════════════════════════
#  FILE : run_local_cpu.py
#  ROLE : Local CPU prep — build everything that needs NO GPU
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs only the CPU_ONLY stages: data + features, the rolling and
#         whole-series decompositions (cached to disk), the classical /
#         linear extrapolators, EVT/GARCH risk, and the analysis/compare
#         steps. Writes shared inputs to results/shared/ and per-track
#         CPU results to results/all/.
#
#  WHY  — GPU hours on Colab/Kaggle are scarce. None of this work needs a
#         GPU, yet re-doing it in a GPU session burns paid time. Run it
#         locally for free; sync results/ to Drive; the GPU session then
#         finds df_features.pkl + the decomposition caches already there
#         and starts straight into neural-net training.
#
#  HOW  — Uses compute_manifest.cpu_only_files() as the source of truth
#         for what is safe to run here, ordered by dependency. Builds all
#         four horizons so BOTH the short and long GPU tracks can reuse
#         the shared data. Hands off to _orchestrate.py.
#
#  USAGE :  os.environ['DDP_OUTPUT_DIR'] = r'C:\\...\\DDP\\results'
#           exec(open('run_local_cpu.py', encoding="utf-8").read())     # from inside DDP_v2/
#  COMPUTE  : CPU-ONLY (runs on your laptop)           RUNTIME : ~1-2 h first build
# ════════════════════════════════════════════════════════════════════

import os
os.environ.setdefault("DDP_TRACK", "all")   # CPU artifacts go to results/all/
HORIZONS = ["1d", "7d", "90d", "180d"]       # build all so both tracks can reuse

exec(open("_env.py", encoding="utf-8").read())
exec(open("_plotting.py", encoding="utf-8").read())

# ── CPU-only stages, ordered by dependency ───────────────────────────
# (Data/decomposition first — these create the shared artifacts the GPU
#  session reuses; then the CPU forecasters; then risk + analysis.)
STAGES = [
    "00_setup_data_features.py",   # → shared/df_features.pkl, feat_cols.pkl, …
    "01a_vmd_features.py",         # rolling WOA-VMD feature columns (slow, CPU)
    "01_black_swan_baselines.py",  # Naive/ARIMA/ETS baselines
    "10_text_features.py",         # GDELT/FinBERT (CPU-tolerable; needs network)
    "09_vmd_hybrid.py",            # rolling VMD + AR (honest hybrid)
    "A_components_linear.py",      # builds + caches whole-series decompositions
    "A_components_classical.py",
    "A_components_hybrid.py",
    "B_honest_decomp.py",
    "R_timerag.py",
    "07_risk_evt.py",
    "07b_garch_evt.py",
    "13_probabilistic_forecast.py",
    "C_ensemble.py",
    "C_compare.py",
    "06_analysis_plots.py",
    "17_audit_pipeline.py",
]

exec(open("_orchestrate.py", encoding="utf-8").read())

print("\nℹ️  Local CPU prep complete. Sync the results/ folder to Drive, then run "
      "run_short.py / run_long.py on Colab/Kaggle — the GPU session will reuse "
      "shared/df_features.pkl and the cached decompositions (no GPU time on prep).")
