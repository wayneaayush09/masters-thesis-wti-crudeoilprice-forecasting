# ════════════════════════════════════════════════════════════════════
#  FILE : run_long.py
#  ROLE : Orchestrator for the LONG horizon track  (90-day + 180-day)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs the full long-horizon pipeline for horizons 90d & 180d
#         only, writing to results/long/.
#
#  WHY  — Mirror of run_short.py for the hard long horizons (kept fully
#         separate, no overlap). Long horizons need the COT / term-
#         structure features (see the COT plan) and longer training.
#
#  HOW  — Identical engine to run_short.py; only HORIZONS and the track
#         differ. Hands off to _orchestrate.py.
#
#  USAGE :  os.environ['DDP_OUTPUT_DIR'] = '<drive-or-local-path>'
#           exec(open('run_long.py', encoding="utf-8").read())          # from inside DDP_v2/
#  COMPUTE  : mixed (GPU stages need Colab/Kaggle)     RUNTIME : hours
# ════════════════════════════════════════════════════════════════════

import os
os.environ["DDP_TRACK"] = "long"           # → results/long/
HORIZONS = ["90d", "180d"]

exec(open("_env.py", encoding="utf-8").read())
exec(open("_plotting.py", encoding="utf-8").read())

# ── Stages, in dependency order (same set as short; horizons differ). ─
STAGES = [
    # ── data + features ───────────────────────────────────────────────────
    "00_setup_data_features.py",
    "01a_vmd_features.py",
    "01_black_swan_baselines.py",

    # ── Track A leaky: CPU extrapolators ─────────────────────────────────
    "A_components_linear.py",
    "A_components_classical.py",
    "A_components_hybrid.py",

    # ── Track A leaky: Yu protocol (WOA-VMD-FE) — all 18 architectures ──
    "A_yu_patchtst.py",
    "A_yu_itransformer.py",
    "A_yu_autoformer.py",
    "A_yu_informer.py",
    "A_yu_fedformer.py",
    "A_yu_fedformerw.py",
    "A_yu_freeformer.py",
    "A_yu_tft.py",
    "A_yu_timemixer.py",
    "A_yu_nhits.py",
    "A_yu_tide.py",
    "A_yu_tcn.py",
    "A_yu_tcnbilstm.py",
    "A_yu_cnnbilstmattn.py",
    "A_yu_segrnn.py",
    "A_yu_dlinearnet.py",
    "A_yu_nlinearnet.py",
    "A_yu_fits.py",

    # ── Track A leaky: Huang protocol (VMD-EMD) — all 18 architectures ──
    "A_huang_patchtst.py",
    "A_huang_itransformer.py",
    "A_huang_autoformer.py",
    "A_huang_informer.py",
    "A_huang_fedformer.py",
    "A_huang_fedformerw.py",
    "A_huang_freeformer.py",
    "A_huang_tft.py",
    "A_huang_timemixer.py",
    "A_huang_nhits.py",
    "A_huang_tide.py",
    "A_huang_tcn.py",
    "A_huang_tcnbilstm.py",
    "A_huang_cnnbilstmattn.py",
    "A_huang_segrnn.py",
    "A_huang_dlinearnet.py",
    "A_huang_nlinearnet.py",
    "A_huang_fits.py",

    # ── Track B honest: leakage-free CPU models ───────────────────────────
    "09_vmd_hybrid.py",
    "B_honest_decomp.py",

    # ── Track B honest: walk-forward NNs — all 17 architectures ──────────
    "B_wf_patchtst.py",
    "B_wf_itransformer.py",
    "B_wf_autoformer.py",
    "B_wf_informer.py",
    "B_wf_fedformer.py",
    "B_wf_fedformerw.py",
    "B_wf_freeformer.py",
    "B_wf_tft.py",
    "B_wf_timemixer.py",
    "B_wf_nhits.py",
    "B_wf_tide.py",
    "B_wf_tcn.py",
    "B_wf_tcnbilstm.py",
    "B_wf_cnnbilstmattn.py",
    "B_wf_segrnn.py",
    "B_wf_dlinear.py",
    "B_wf_nlinear.py",

    # ── Track B special ───────────────────────────────────────────────────
    "B_regime_conditional.py",

    # ── Foundation models (zero-shot, no training) ────────────────────────
    "B_foundation_chronos.py",
    "B_foundation_timesfm.py",

    # ── Risk + probabilistic ──────────────────────────────────────────────
    "07_risk_evt.py",
    "07b_garch_evt.py",
    "13_probabilistic_forecast.py",

    # ── Combine + compare + report ────────────────────────────────────────
    "C_ensemble.py",
    "C_compare.py",
    "06_analysis_plots.py",
    "17_audit_pipeline.py",
]

exec(open("_orchestrate.py", encoding="utf-8").read())
