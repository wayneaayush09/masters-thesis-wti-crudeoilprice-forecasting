# DDP_v2 — Glossary

Recurring terms used across the codebase, defined once so per-file comments can stay concise.

## Protocols & tracks
- **Track A (leaky)** — Reproduces published papers by decomposing the **whole series** (train + test together) *before* the train/test split. This lets future information leak into the features ("lookahead leakage by design", Xu & Niu 2024). Reproduces headline benchmark numbers but is **not** a real out-of-sample forecast. Tagged `leaky=True`.
- **Track B (honest)** — Walk-forward: at each test origin only past data is visible, and features are derived only from the past. These are the **defensible** thesis numbers. Tagged `leaky=False`.
- **Yu protocol** — `whole_series_woa_vmd_fe`: WOA-tuned VMD into 3 fuzzy-entropy bands (LF/MF/HF), then forecast each band.
- **Huang protocol** — `whole_series_vmd_emd`: VMD followed by EMD on the residual ("quadratic"/double decomposition), ~15 components.
- **Leakage premium** — The MAE/RMSE gap between Track A (leaky) and Track B (honest); quantifies how much a published score is inflated by decomposing across the split.

## Decomposition
- **VMD** — Variational Mode Decomposition (Dragomiretskiy & Zosso 2014); splits a signal into K band-limited modes.
- **WOA** — Whale Optimization Algorithm; here a bounded search over VMD's (K, α) hyper-parameters.
- **EMD / CEEMDAN** — Empirical Mode Decomposition (and its noise-assisted ensemble variant); data-driven split into intrinsic mode functions (IMFs).
- **Fuzzy entropy (FE)** — Regularity measure (Chen et al. 2007); lower = smoother/more predictable. Used to rank/cluster modes.
- **Rolling vs whole-series decomposition** — Rolling = re-decompose only the trailing past window at each origin (leakage-free, slower). Whole-series = decompose once over everything (leaky, fast).

## Models
- **PatchTST** — Patch-based, channel-independent transformer for long-horizon forecasting.
- **RevIN** — Reversible Instance Normalization; per-window normalize then de-normalize to handle distribution shift.
- **DLinear / NLinear** — Linear forecasters (trend+seasonal decomposition / last-value normalization); strong, cheap baselines.
- **Foundation models** — Chronos-Bolt (Amazon), TimesFM (Google); pretrained, used zero-shot (no training).

## Evaluation & horizons
- **Horizon (hk)** — Forecast lead time: `1d`, `7d`, `90d`, `180d`. **Short track** = 1d+7d; **long track** = 90d+180d.
- **Walk-forward (WF)** — Expanding/rolling-origin evaluation across multiple chronological splits ("folds").
- **MAE / RMSE / MAPE** — Mean abs error / root mean sq error / mean abs % error (reported in $/barrel unless noted).
- **DM / Clark-West test** — Diebold-Mariano (equal predictive accuracy) and Clark-West (nested-model) significance tests.

## Risk & uncertainty
- **EVT / POT / GPD** — Extreme Value Theory; Peaks-Over-Threshold fit of a Generalized Pareto Distribution to error tails → VaR/ES.
- **GARCH-EVT** — Conditional volatility (GARCH) + EVT tails → dynamic, time-varying VaR/ES (McNeil-Frey).
- **Conformal (CQR / Mondrian)** — Distribution-free prediction intervals with finite-sample coverage; Mondrian = per-regime calibration.
- **PIT / CRPS** — Probability Integral Transform (calibration check) / Continuous Ranked Probability Score (probabilistic accuracy).

## Features & data
- **COT** — CFTC Commitment-of-Traders; weekly managed-money long/short positioning. Release-lagged to stay leakage-safe.
- **Term-structure slope** — log(front futures) − log(deferred); contango (<0) vs backwardation (>0).
- **HMM regime** — Hidden-Markov-Model market state (calm → crisis), ranked by volatility into `regime_vol_rank ∈ [0,1]`.

## Infrastructure (this restructure)
- **`DDP_OUTPUT_DIR`** — Env var: the persistent results root (Drive / local). **`DDP_TRACK`** — `short`/`long`/`all` sub-folder.
- **`already_done(name, hk)`** — Skip guard: True if a model×horizon already has its CSV + eval pkl (unless `DDP_FORCE_RERUN=1`).
- **Decomposition cache** — On-disk `cache/*.pkl` of whole-series VMD/EMD so a reconnected session skips the ~7-min recompute.
