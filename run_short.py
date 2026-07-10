# ════════════════════════════════════════════════════════════════════
#  FILE : run_short.py
#  ROLE : Orchestrator for the SHORT horizon track  (1-day + 7-day)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs the full short-horizon pipeline (Track A leaky, Track B
#         honest, foundation, risk, ensemble, compare, report) for
#         horizons 1d & 7d only, writing to results/short/.
#
#  WHY  — You asked for 1d+7d work kept completely separate from 90d+
#         180d. Same shared codebase, two thin orchestrators: this one
#         pins HORIZONS=['1d','7d'] and DDP_TRACK='short' so outputs land
#         in their own tree and never overlap with the long track.
#
#  HOW  — Sets the env/track + HORIZONS, loads the infra (_env, _plotting),
#         defines STAGES, and hands off to _orchestrate.py. Re-running
#         after a disconnect skips finished (model×horizon) pairs.
#
#  USAGE (after setting your storage target, see _env.py):
#         os.environ['DDP_OUTPUT_DIR'] = '<drive-or-local-path>'
#         exec(open('run_short.py', encoding="utf-8").read())          # from inside DDP_v2/
#  COMPUTE  : mixed (GPU stages need Colab/Kaggle)    RUNTIME : hours
# ════════════════════════════════════════════════════════════════════

import os
os.environ["DDP_TRACK"] = "short"          # → results/short/
HORIZONS = ["1d", "7d"]                     # the orchestrator's horizon choice

exec(open("_env.py", encoding="utf-8").read())                # OUTPUT_DIR, DATA_DIR, already_done, …
exec(open("_plotting.py", encoding="utf-8").read())           # Plotly theme + save_fig/table/animation

# ── Stages, in dependency order. Comment out what you don't need. ─────
# (00 + features are usually built once by run_local_cpu.py and shared;
#  keep them here so a standalone short run also works.)
STAGES = [
    # ── data + features (shared; skipped fast if already built) ──────────
    "00_setup_data_features.py",
    "01a_vmd_features.py",
    "01_black_swan_baselines.py",

    # ── Track A leaky: CPU extrapolators ─────────────────────────────────
    "A_components_linear.py",
    "A_components_classical.py",
    "A_components_hybrid.py",

    # ── Track A leaky: Yu protocol (WOA-VMD-FE) — all 18 architectures ──
    # WOA-VMD decomposition is computed once and cached; all subsequent
    # A_yu_* files reuse the cache (~1s) rather than recomputing (~15 min).
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
