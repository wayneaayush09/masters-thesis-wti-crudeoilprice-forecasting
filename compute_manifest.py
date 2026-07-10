# ════════════════════════════════════════════════════════════════════
#  FILE : compute_manifest.py
#  ROLE : GPU-vs-CPU routing map for every driver
# ════════════════════════════════════════════════════════════════════
#  WHAT — Classifies each runnable file as CPU_ONLY, GPU_HEAVY, or
#         GPU_OPTIONAL, and (when run) writes COMPUTE_MANIFEST.md.
#
#  WHY  — Colab/Kaggle GPU hours are scarce. Anything that doesn't train
#         a neural net (data build, decomposition, AR/Theta/ARIMA, EVT/
#         GARCH, ensembling, analysis) can run for free on the local
#         machine; its artifacts are then synced to Drive so the GPU
#         session starts straight into NN training. This file is the
#         authoritative routing table that run_local_cpu.py consults.
#
#  HOW  — A dict literal (hand-verified from the source). CPU_ONLY files
#         are safe to run on the local CPU; GPU_HEAVY need a cloud GPU;
#         GPU_OPTIONAL (foundation models, LLM/text) run either place but
#         are far faster on GPU.
#
#  READS    : nothing      WRITES : COMPUTE_MANIFEST.md (when run directly)
#  COMPUTE  : CPU-ONLY     PIPELINE : reference/utility (not a stage)
# ════════════════════════════════════════════════════════════════════

COMPUTE = {
    # ── CPU_ONLY — run locally, sync artifacts to the GPU session ────
    "00_setup_data_features.py":  "CPU_ONLY",
    "01_black_swan_baselines.py": "CPU_ONLY",
    "01a_vmd_features.py":        "CPU_ONLY",   # slow but pure-CPU rolling VMD
    "09_vmd_hybrid.py":           "CPU_ONLY",
    "A_components_linear.py":     "CPU_ONLY",
    "A_components_classical.py":  "CPU_ONLY",
    "A_components_hybrid.py":     "CPU_ONLY",
    "B_honest_decomp.py":         "CPU_ONLY",
    "07_risk_evt.py":             "CPU_ONLY",
    "07b_garch_evt.py":           "CPU_ONLY",
    "13_probabilistic_forecast.py": "CPU_ONLY",
    "R_timerag.py":               "CPU_ONLY",
    "C_ensemble.py":              "CPU_ONLY",
    "C_compare.py":               "CPU_ONLY",
    "06_analysis_plots.py":       "CPU_ONLY",
    "17_audit_pipeline.py":       "CPU_ONLY",
    "T_thesis_compare.py":        "CPU_ONLY",
    # the linear extrapolator NN-variants are non-trained (closed form)
    "A_huang_dlinearnet.py":      "CPU_ONLY",
    "A_huang_nlinearnet.py":      "CPU_ONLY",
    "A_yu_dlinearnet.py":         "CPU_ONLY",
    "A_yu_nlinearnet.py":         "CPU_ONLY",
    "B_wf_dlinear.py":            "CPU_ONLY",
    "B_wf_nlinear.py":            "CPU_ONLY",

    # ── GPU_OPTIONAL — either place (pretrained inference, no backprop) ─
    "B_foundation_chronos.py":    "GPU_OPTIONAL",
    "B_foundation_timesfm.py":    "GPU_OPTIONAL",
    "10_text_features.py":        "GPU_OPTIONAL",
    "12_agentic_rag.py":          "GPU_OPTIONAL",
    "D_llm_news_scenario.py":     "GPU_OPTIONAL",

    # ── GPU_HEAVY — need a cloud GPU (neural-net training) ───────────
    "16_hyperopt.py":             "GPU_HEAVY",
    "B_regime_conditional.py":    "GPU_HEAVY",
    "T_thesis_honest.py":         "GPU_HEAVY",
    "T_thesis_leaky_repro.py":    "GPU_HEAVY",
    "E_uni_vs_multi.py":          "GPU_HEAVY",
    "F_fedformer_official.py":    "GPU_HEAVY",
    "F_fedformer_sweep.py":       "GPU_HEAVY",
    # all remaining A_yu_* / A_huang_* (transformer/RNN archs) and the
    # B_wf_* trainers are GPU_HEAVY; see compute_tier() fallback below.
}


def compute_tier(filename):
    """Return the compute tier for a driver, inferring GPU_HEAVY for any
    transformer/RNN A_*/B_wf_* arch not explicitly listed as CPU_ONLY."""
    if filename in COMPUTE:
        return COMPUTE[filename]
    if filename.startswith(("A_yu_", "A_huang_", "B_wf_")):
        return "GPU_HEAVY"          # arch trainers default to GPU
    return "UNKNOWN"


def cpu_only_files():
    """List of files safe to run on the local machine."""
    return [f for f, t in COMPUTE.items() if t == "CPU_ONLY"]


def _write_markdown():
    """Emit COMPUTE_MANIFEST.md grouped by tier (run this file directly)."""
    from pathlib import Path
    groups = {"CPU_ONLY": [], "GPU_OPTIONAL": [], "GPU_HEAVY": []}
    for f, t in sorted(COMPUTE.items()):
        groups.setdefault(t, []).append(f)
    lines = ["# DDP_v2 — Compute Routing Manifest\n",
             "Where to run each file. **CPU_ONLY** → run locally and sync the",
             "`results/` folder to Drive; the GPU session then reuses those",
             "artifacts (no GPU time on data prep / decomposition).\n"]
    titles = {"CPU_ONLY": "🟢 CPU_ONLY — run locally (preserve GPU hours)",
              "GPU_OPTIONAL": "🟡 GPU_OPTIONAL — either (pretrained inference)",
              "GPU_HEAVY": "🔴 GPU_HEAVY — Colab/Kaggle GPU required"}
    for tier in ("CPU_ONLY", "GPU_OPTIONAL", "GPU_HEAVY"):
        lines.append(f"\n## {titles[tier]}\n")
        lines += [f"- `{f}`" for f in groups.get(tier, [])]
    lines += ["\n## Note",
              "All `A_yu_*` / `A_huang_*` transformer & RNN archs and `B_wf_*`",
              "trainers not listed above are **GPU_HEAVY** (see `compute_tier()`)."]
    Path("COMPUTE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print("✅ wrote COMPUTE_MANIFEST.md")


if __name__ == "__main__":
    _write_markdown()
