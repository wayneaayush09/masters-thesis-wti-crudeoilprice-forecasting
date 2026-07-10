# ════════════════════════════════════════════════════════════════════
#  FILE : A_yu_fedformerw.py
#  ROLE : Track A leaky Yu-protocol driver — FEDformerW
# ════════════════════════════════════════════════════════════════════
#  WHAT — Sets ARCH="FEDformerW" PROTOCOL="yu" then execs _run_paper_arch.py.
#         Runs WOA-VMD-FE whole-series decomposition (3 bands: LF/MF/HF)
#         followed by per-band FEDformerW training. Tagged leaky=True.
#
#  WHY  — Reproduces the Yu 2025 paper protocol. Whole-series decomp
#         BEFORE the 70/30 split introduces lookahead leakage by design.
#         These numbers benchmark against published results (NOT defensible
#         thesis claims). Run both tracks to quantify the leakage premium.
#
#  HOW  — One-line driver: set ARCH + PROTOCOL, delegate to _run_paper_arch
#         -> _paper_helpers.run_paper_yu -> train one NN per band, sum.
#         Disk-cached WOA-VMD decomposition (~7 min) reused across archs.
#
#  READS    : df_features.pkl (log_price), cache/yu_decomp_{hk}.pkl
#  WRITES   : predictions_PaperYu_FEDformerW_{hk}.csv, eval_PaperYu_FEDformerW_{hk}.pkl
#             figures/trackA_PaperYu_FEDformerW_{hk}.html|.png
#  COMPUTE  : GPU-HEAVY  (run on Colab / Kaggle A100)
#  PIPELINE : Stage 2 Track A; Stage 0 must complete first
#  RUNTIME  : ~7 min WOA-VMD (cached) + 3x3-5 min NN training per horizon
# ════════════════════════════════════════════════════════════════════
ARCH = "FEDformerW"
PROTOCOL = "yu"
exec(open("_run_paper_arch.py", encoding="utf-8").read())