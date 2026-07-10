# DDP_v2 — Compute Routing Manifest

Where to run each file. **CPU_ONLY** → run locally and sync the
`results/` folder to Drive; the GPU session then reuses those
artifacts (no GPU time on data prep / decomposition).


## 🟢 CPU_ONLY — run locally (preserve GPU hours)

- `00_setup_data_features.py`
- `01_black_swan_baselines.py`
- `01a_vmd_features.py`
- `06_analysis_plots.py`
- `07_risk_evt.py`
- `07b_garch_evt.py`
- `09_vmd_hybrid.py`
- `13_probabilistic_forecast.py`
- `17_audit_pipeline.py`
- `A_components_classical.py`
- `A_components_hybrid.py`
- `A_components_linear.py`
- `A_huang_dlinearnet.py`
- `A_huang_nlinearnet.py`
- `A_yu_dlinearnet.py`
- `A_yu_nlinearnet.py`
- `B_honest_decomp.py`
- `B_wf_dlinear.py`
- `B_wf_nlinear.py`
- `C_compare.py`
- `C_ensemble.py`
- `R_timerag.py`
- `T_thesis_compare.py`

## 🟡 GPU_OPTIONAL — either (pretrained inference)

- `10_text_features.py`
- `12_agentic_rag.py`
- `B_foundation_chronos.py`
- `B_foundation_timesfm.py`
- `D_llm_news_scenario.py`

## 🔴 GPU_HEAVY — Colab/Kaggle GPU required

- `16_hyperopt.py`
- `B_regime_conditional.py`
- `E_uni_vs_multi.py`
- `F_fedformer_official.py`
- `F_fedformer_sweep.py`
- `T_thesis_honest.py`
- `T_thesis_leaky_repro.py`

## Note
All `A_yu_*` / `A_huang_*` transformer & RNN archs and `B_wf_*`
trainers not listed above are **GPU_HEAVY** (see `compute_tier()`).