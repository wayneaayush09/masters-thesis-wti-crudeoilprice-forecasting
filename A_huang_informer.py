# ════════════════════════════════════════════════════════════════════
#  FILE : A_huang_informer.py
#  ROLE : Track A leaky Huang-protocol driver — Informer
# ════════════════════════════════════════════════════════════════════
#  WHAT — Sets ARCH="Informer" PROTOCOL="huang" then execs _run_paper_arch.py.
#         Runs VMD+EMD quadratic decomposition (~15 components) followed
#         by per-component Informer training. Tagged leaky=True.
#
#  WHY  — Reproduces the Huang 2024 benchmark (MAE 3.45 at 180d). Also
#         leaky (whole-series before split) but with a different leakage
#         source than Yu (EMD on VMD residual). Used to cross-validate
#         the Yu leakage premium and benchmark against published targets.
#
#  HOW  — One-line driver: set ARCH + PROTOCOL="huang", delegate to
#         _run_paper_arch -> _paper_helpers.run_paper_huang -> VMD+EMD
#         decomp (disk-cached), train one NN per component, sum bands.
#
#  READS    : df_features.pkl (log_price), cache/huang_decomp_{hk}_*.pkl
#  WRITES   : predictions_PaperHuang_Informer_{hk}.csv
#             eval_PaperHuang_Informer_{hk}.pkl
#             figures/trackA_PaperHuang_Informer_{hk}.html|.png
#  COMPUTE  : GPU-HEAVY  (run on Colab / Kaggle A100)
#  PIPELINE : Stage 2 Track A; Stage 0 must complete first
#  RUNTIME  : ~15 min VMD+EMD (cached) + 15x3-5 min NN per horizon
# ════════════════════════════════════════════════════════════════════
ARCH = "Informer"
PROTOCOL = "huang"
exec(open("_run_paper_arch.py", encoding="utf-8").read())