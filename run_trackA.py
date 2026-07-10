# ════════════════════════════════════════════════════════════════════
#  FILE : run_trackA.py
#  ROLE : Per-horizon GPU runner — Track A LEAKY paper reproductions ONLY
# ════════════════════════════════════════════════════════════════════
#  WHAT — Trains the Track-A leaky paper reproductions (PaperYu_* +
#         PaperHuang_*, 18 architectures × 2 protocols = 36 drivers) for a
#         SINGLE horizon chosen by the DDP_TRACK env var
#         ('1d' | '7d' | '90d' | '180d'). Nothing else runs here: Track B,
#         risk, ensemble, compare and the analysis plots are handled by
#         run_trackB.py / run_phase3.py so this GPU session does only Track-A
#         GPU work.
#
#  WHY  — Track A never had the per-horizon + speed + resume treatment that
#         Track B got. This runner gives it the same: one horizon per session
#         (small, disconnect-resilient), the DDP_SPEED profile reaching the
#         per-band trainer (AMP + batch 256 + 50-epoch early stop on
#         'balanced'), and per-band/per-component checkpoints so a mid-arch
#         disconnect costs only the band/component in flight.
#
#  HOW  — The CALLER sets the environment BEFORE exec'ing this file:
#           os.environ['DDP_OUTPUT_DIR'] = '<persistent path>'   # Kaggle working / Drive
#           os.environ['DDP_TRACK']      = '1d'                  # ONE horizon
#           os.environ['DDP_SPEED']      = 'balanced'
#         Then:  exec(open('run_trackA.py', encoding='utf-8').read())
#
#  RESUME — re-running after a disconnect skips finished (arch×horizon) pairs
#           via already_done(), resumes the in-progress arch from its last
#           per-band/per-component checkpoint, skips done stages (track-aware),
#           and reloads the whole-series WOA-VMD / VMD-EMD decomposition from
#           the disk cache in <1 s. Because DDP_OUTPUT_DIR is persistent, all
#           of that survives a session teardown.
#  COMPUTE: GPU-HEAVY (Kaggle/Colab).  RUNTIME: ~3 h (1d/7d) to ~6 h (180d)
#           for all 36 drivers on 'balanced'; far less when archs already exist.
# ════════════════════════════════════════════════════════════════════

import os
assert os.environ.get("DDP_OUTPUT_DIR"), (
    "Set os.environ['DDP_OUTPUT_DIR'] to a PERSISTENT path (e.g. Kaggle "
    "/kaggle/working/DDP_results or a Google Drive folder) BEFORE running — "
    "otherwise results are written to disposable disk.")
os.environ.setdefault("DDP_TRACK", "1d")          # one of: 1d | 7d | 90d | 180d
os.environ.setdefault("DDP_SPEED", "balanced")

exec(open("_env.py",      encoding="utf-8").read())   # OUTPUT_DIR, TRACK_HORIZONS, already_done, …
exec(open("_plotting.py", encoding="utf-8").read())   # Plotly theme + save_fig/table/animation

# ── Track A leaky reproductions — all 18 archs × {Yu, Huang}, THIS horizon ──
# Ordered light → heavy so the cheap archs bank results first and the costly
# Yu (WOA-VMD ~7 min) / Huang (VMD+EMD ~15 min) decompositions are cached early
# in the session. Each arch runs both protocols (Yu = 3 bands, Huang = ~15
# components) so its leakage-premium pair completes together. Per-arch Plotly
# figures emit from _run_paper_arch.py as each finishes.
_ARCH_ORDER = [
    "dlinearnet", "nlinearnet", "fits", "tide", "segrnn",                       # light
    "patchtst", "tcn", "tcnbilstm", "nhits", "timemixer", "cnnbilstmattn",      # medium
    "itransformer",
    "autoformer", "informer", "fedformer", "fedformerw", "freeformer", "tft",   # heavy
]
STAGES = [f"A_{proto}_{arch}.py"
          for arch in _ARCH_ORDER
          for proto in ("yu", "huang")]

print(f"\n>>> Track A per-horizon run  |  horizon(s)={TRACK_HORIZONS}  "
      f"|  speed={os.environ['DDP_SPEED']}  |  {len(STAGES)} drivers  "
      f"|  out={OUTPUT_DIR}")
exec(open("_orchestrate.py", encoding="utf-8").read())
