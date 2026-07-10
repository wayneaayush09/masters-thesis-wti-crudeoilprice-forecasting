# ════════════════════════════════════════════════════════════════════
#  FILE : run_phase3.py
#  ROLE : Phase 3 consolidation — risk + probabilistic + ensemble +
#         compare + analysis plots + audit  (CPU-only, ALL horizons)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs the cross-model / cross-horizon analysis over EVERY
#         prediction present in DDP_OUTPUT_DIR — Track B (fresh, from the
#         four run_trackB.py sessions) plus Track A (LH extrapolators +
#         the Paper neural files copied in from the ddp-results1 dataset).
#         Produces the master comparator, skill-score, DM-test heatmap,
#         ranking-race animation, ensemble, risk/EVT and probabilistic-fan
#         figures + tables into figures/all/ and tables/all/.
#
#  WHY  — Every stage here is CPU-only (see COMPUTE_MANIFEST.md) → run it
#         LOCALLY for free, no GPU hour spent. Run AFTER the four per-horizon
#         Track B sessions have synced into your results folder (and after
#         Track A's predictions_Paper* / eval_Paper* are copied in, if you
#         want the leaky benchmark bars in the figures).
#
#  HOW  —   os.environ['DDP_OUTPUT_DIR'] = r'<results folder>'
#           exec(open('run_phase3.py', encoding='utf-8').read())
#         DDP_TRACK is forced to 'all' so the comparison spans 1d/7d/90d/180d.
#         FORCE_RERUN is forced on so the plots always rebuild over the latest
#         prediction set (these stages may have been logged 'done' by an
#         earlier run); they are cheap and idempotent.
#  COMPUTE: CPU-ONLY.  RUNTIME: ~10-30 min.
# ════════════════════════════════════════════════════════════════════

import os
assert os.environ.get("DDP_OUTPUT_DIR"), (
    "Set os.environ['DDP_OUTPUT_DIR'] to your results folder (the one the "
    "per-horizon Track B runs wrote to) BEFORE running.")
os.environ["DDP_TRACK"]       = "all"   # full cross-horizon comparison
os.environ["DDP_FORCE_RERUN"] = "1"     # always rebuild plots over the current predictions

exec(open("_env.py",      encoding="utf-8").read())
exec(open("_plotting.py", encoding="utf-8").read())

STAGES = [
    "07_risk_evt.py",
    "07b_garch_evt.py",
    "13_probabilistic_forecast.py",
    "C_ensemble.py",
    "C_compare.py",
    "06_analysis_plots.py",
    "17_audit_pipeline.py",
]

print(f"\n>>> Phase 3 consolidation  |  horizons={TRACK_HORIZONS}  |  out={OUTPUT_DIR}")
exec(open("_orchestrate.py", encoding="utf-8").read())

# Reset the force flag so a later import in the same kernel behaves normally.
os.environ["DDP_FORCE_RERUN"] = "0"
