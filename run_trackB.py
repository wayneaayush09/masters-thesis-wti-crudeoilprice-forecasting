# ════════════════════════════════════════════════════════════════════
#  FILE : run_trackB.py
#  ROLE : Per-horizon GPU runner — Track B honest walk-forward ONLY
# ════════════════════════════════════════════════════════════════════
#  WHAT — Trains the 17 Track-B (honest walk-forward) neural models for a
#         SINGLE horizon chosen by the DDP_TRACK env var
#         ('1d' | '7d' | '90d' | '180d'). Nothing else runs here: Track A,
#         risk, ensemble, compare and the analysis plots are handled by
#         run_phase3.py (CPU, local) so this GPU session does only GPU work.
#
#  WHY  — You asked to separate 1d/7d/90d/180d into independent, resumable
#         Colab sessions and to stop spending GPU on the already-complete
#         Track A. One horizon per session = a small, disconnect-resilient
#         unit. All four safely share one DDP_results root (horizon-suffixed
#         filenames + track-aware resume), so df_features is stored once and
#         the final compare sees every horizon.
#
#  HOW  — The CALLER sets the environment BEFORE exec'ing this file:
#           os.environ['DDP_OUTPUT_DIR'] = '<persistent path>'   # e.g. a Drive folder
#           os.environ['DDP_TRACK']      = '1d'                  # ONE horizon
#           os.environ['DDP_SPEED']      = 'balanced'
#           N_SEEDS        = 2                                    # the driver reads this global
#           WF_FREQ_MONTHS = {'1d':24,'7d':24,'90d':36,'180d':36}
#         Then:  exec(open('run_trackB.py', encoding='utf-8').read())
#
#  RESUME — re-running after a disconnect skips finished (model×horizon)
#           pairs via already_done(), resumes the in-progress model from its
#           last per-fold checkpoint, and skips done stages (track-aware).
#           Because DDP_OUTPUT_DIR points at Drive, all of that survives.
#  COMPUTE: GPU-HEAVY (Colab/Kaggle).  RUNTIME: ~30-70 min/horizon (balanced)
# ════════════════════════════════════════════════════════════════════

import os
assert os.environ.get("DDP_OUTPUT_DIR"), (
    "Set os.environ['DDP_OUTPUT_DIR'] to a PERSISTENT path (e.g. a Google Drive "
    "folder) BEFORE running — otherwise results are written to disposable disk.")
os.environ.setdefault("DDP_TRACK", "1d")          # one of: 1d | 7d | 90d | 180d
os.environ.setdefault("DDP_SPEED", "balanced")

exec(open("_env.py",      encoding="utf-8").read())   # OUTPUT_DIR, TRACK_HORIZONS, already_done, …
exec(open("_plotting.py", encoding="utf-8").read())   # Plotly theme + save_fig/table/animation

# ── Track B honest walk-forward — all 17 architectures, THIS horizon only ──
# Per-model Plotly figures (forecast-vs-actual + residuals) are emitted by
# _run_wf_arch.py immediately after each model completes, so you can inspect
# results as they land instead of waiting for the end.
STAGES = [
    "B_wf_patchtst.py",     "B_wf_itransformer.py", "B_wf_autoformer.py",
    "B_wf_informer.py",     "B_wf_fedformer.py",    "B_wf_fedformerw.py",
    "B_wf_freeformer.py",   "B_wf_tft.py",          "B_wf_timemixer.py",
    "B_wf_nhits.py",        "B_wf_tide.py",         "B_wf_tcn.py",
    "B_wf_tcnbilstm.py",    "B_wf_cnnbilstmattn.py","B_wf_segrnn.py",
    "B_wf_dlinear.py",      "B_wf_nlinear.py",
]

print(f"\n>>> Track B per-horizon run  |  horizon(s)={TRACK_HORIZONS}  "
      f"|  speed={os.environ['DDP_SPEED']}  |  seeds={globals().get('N_SEEDS', 3)}  "
      f"|  out={OUTPUT_DIR}")
exec(open("_orchestrate.py", encoding="utf-8").read())
