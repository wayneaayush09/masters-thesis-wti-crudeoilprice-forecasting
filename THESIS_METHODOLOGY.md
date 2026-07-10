# DDP_v2 — Thesis Methodology Reference
## Crude Oil Price Forecasting: Methodology, Logic, and Execution

> **How to use this document:** This is the master reference for the DDP_v2 pipeline. Each chapter maps directly to a group of pipeline files and explains the *why* behind every design decision. Read Chapter 1 first (framework); then jump to whichever chapter you need for writing a specific thesis section.

---

## Table of Contents

1. [Research Problem & Design Philosophy](#1-research-problem--design-philosophy)
2. [Dataset & Data Sources](#2-dataset--data-sources)
3. [Preprocessing & Winsorization](#3-preprocessing--winsorization)
4. [Feature Engineering (100+ Features, Horizon-Aware)](#4-feature-engineering)
5. [Track A — Leaky Paper Reproductions (VMD/EMD Decomposition)](#5-track-a--leaky-paper-reproductions)
6. [Track B — Honest Walk-Forward (Leakage-Free)](#6-track-b--honest-walk-forward)
7. [Neural Architectures Deployed](#7-neural-architectures-deployed)
8. [Probabilistic & Tail-Risk Framework](#8-probabilistic--tail-risk-framework)
9. [Ensemble & Comparison Framework](#9-ensemble--comparison-framework)
10. [Implementation Architecture & Reproducibility](#10-implementation-architecture--reproducibility)
11. [Summary of Design Decisions](#11-summary-of-design-decisions)
12. [References](#12-references)

---

## 1. Research Problem & Design Philosophy

### 1.1 Why crude oil price forecasting is hard

WTI crude oil is one of the most difficult financial series to forecast for three compounding reasons:

1. **Fat-tailed shocks.** Daily WTI log-returns have kurtosis ~7–12 (Gaussian = 3). A model calibrated on the calm 2010–2014 period will catastrophically mis-price a COVID-style crash. Standard MSE loss and Gaussian intervals both underestimate tail probabilities by orders of magnitude.

2. **Regime non-stationarity.** The data-generating process shifts between regimes — OPEC cartel discipline, shale-supply disruptions, demand destruction crises, supply-chain dislocations — with no warning and no single model structure valid across all regimes.

3. **External shock dominance at long horizons.** At 90-day and 180-day horizons, macroeconomic policy decisions, geopolitical events, and speculative positioning dynamics overwhelm any technical pattern in historical prices. This makes long-horizon forecasting simultaneously the most practically valuable and methodologically challenging sub-problem.

### 1.2 The central flaw in the published literature

Almost all published "state-of-the-art" WTI forecasting results share a methodological flaw: **lookahead leakage introduced at the decomposition stage**.

Decomposition-based pipelines (VMD, EMD, CEEMDAN) are applied to the *entire* price series — train and test combined — before the train/test split. Every extracted mode therefore encodes future test-period information. When a neural network is subsequently trained on these modes, it learns patterns that can only exist because it has indirectly seen the future.

The result is a systematic inflation of out-of-sample performance. **Xu & Niu (2024, *Energy Economics* 128)** quantify this effect and label it "decomposition lookahead bias." This thesis reproduces and quantifies it empirically.

### 1.3 The dual-track design

The pipeline is deliberately split into two parallel tracks:

| Track | Name | Decomposition | Leakage | Purpose |
|---|---|---|---|---|
| **A** | Leaky (paper-faithful) | Whole series before split | **Yes** | Reproduce published numbers; contest benchmarks |
| **B** | Honest (walk-forward) | Rolling window, past data only | **No** | Defensible thesis claims |

The **leakage premium** = Track A MAE − Track B MAE for the same architecture. A large positive premium means the published number is inflated; a near-zero premium means decomposition provides genuine signal. This comparison is a primary thesis contribution.

### 1.4 Forecast horizons and their practical meaning

| Horizon | h (calendar days) | Use case | Primary stakeholders |
|---|---|---|---|
| 1d | 1 trading day | Algorithmic/desk trading | Prop desks, market-makers |
| 7d | 5 trading days | Tactical position management | Commodity funds, hedgers |
| 90d | ~63 trading days | Procurement, medium-term hedging | Refiners, airlines, industrial users |
| 180d | ~126 trading days | Capital planning, long-term supply contracts | National oil companies, majors |

The **Huang et al. (2024)** benchmark focuses on 90d and 180d: MAE = 3.45 $/bbl, MAPE = 7.84%. Beating this benchmark with a leakage-free model is the headline empirical claim.

---

## 2. Dataset & Data Sources

**File:** `00_setup_data_features.py`

### 2.1 Primary series

- **WTI crude oil spot price FOB (Cushing, OK)**: daily, 1986-01-01 to present
- Source: EIA CSV (`Cushing_OK_WTI_Spot_Price_FOB.csv`)
- 40-year span captures all major oil market cycles: 1990 Gulf War spike, 1998 Asian crisis collapse, 2008 GFC crash, 2014–16 OPEC price war, 2020 COVID demand destruction (historic negative print), 2021–22 supply crunch

### 2.2 Exogenous data categories

Every category has a 2–3 source fallback chain so network failures never crash the pipeline (returns zeros silently).

| Category | Key Series | Primary Source | Rationale |
|---|---|---|---|
| **Equity & FX** | S&P 500, Brent, Natural Gas, Gold, USD Index, 10Y Treasury | yfinance | Cross-asset correlation, risk-on/off signals |
| **Volatility** | VIX (^VIX), OVX (OVXCLS) | yfinance, FRED | Liang & Lv (2020): OVX adds 10–20% RMSE reduction over VIX alone |
| **Macroeconomic** | IndPro, CPI, GDP, Fed Funds Rate, 10Y-2Y spread, USD-EUR | FRED | Structural demand drivers; YoY/QoQ rates for stationarity |
| **Supply/Inventory** | Cushing crude stocks, US crude ending stocks, Baker Hughes rig count | FRED, bakerhughes.com | Inventory = immediate demand pressure; rig count = 6-month supply lead |
| **Policy uncertainty** | US EPU index, Caldara-Iacoviello GPR | EPU website, Iacoviello website | Political shocks propagate through commodity markets within days |
| **OPEC production** | PAPR_OPEC | FRED | Cartel supply decisions are the dominant structural driver at 90d+ |
| **Public attention** | "oil price", "OPEC", "WTI" Google Trends | pytrends | Yang et al. (arXiv 2002.09656), Elshendy et al. (*Energy* 2018): search volume nowcasts demand |
| **CFTC positioning** | Managed-money longs/shorts/open interest | cot_reports library | Kang et al. (2020): speculative positioning is a strong medium-horizon predictor |
| **Futures term structure** | WTI front contract (C1), deferred (C4) | FRED, yfinance | Contango (C4 > C1) signals storage glut; backwardation signals scarcity |
| **Text/sentiment** | GDELT tone, FinBERT entity sentiment | Stage 10 (`10_text_features.py`) | Directional signals from news media |

### 2.3 CFTC COT leakage discipline

CFTC releases each Tuesday snapshot on the following Friday (+3 business days). Every COT value in the feature set is **indexed by its release date**, not its snapshot date. Forward-filled to business days. This prevents any business day from seeing an unreleased report — a necessary condition for the honest walk-forward track to be truly leakage-free.

---

## 3. Preprocessing & Winsorization

**File:** `00_setup_data_features.py`

### 3.1 The negative-price problem

On 2020-04-20, WTI futures settled at −$36.98/bbl — the first negative close in the contract's 37-year history. Taking log(−36.98) is undefined; more practically, a single −∞ gradient in the first training batch corrupts an entire neural network's weights irreparably.

### 3.2 Winsorization approach

- **Threshold**: 0.1th percentile of raw WTI prices ≈ $10.95/bbl (for 1986–2026 data)
- **Clipping**: prices below the threshold are replaced by the threshold value
- **Why 0.1th, not 1st percentile?** The 1st percentile is ~$17/bbl — that would winsorize the 1998 Asian crisis low and the 2002 post-9/11 period, both of which are real market events carrying genuine information. The 0.1th percentile clips only extreme tail events not representable in a neural network's gradient regime.

### 3.3 Preserving the shock signal

Winsorization is *not* information deletion. The shock magnitude is preserved in dedicated features:

| Feature | Formula | Captures |
|---|---|---|
| `price_orig` | Raw un-clipped price | Full history including negatives |
| `raw_ret_signed_log` | sign(Δprice_orig) × log1p(|Δprice_orig|) | Signed log-magnitude of shock |
| `raw_dd_below_floor` | clip(wins_floor − price_orig, 0) | How far below floor (>0 only on extremes) |
| `is_winsorized` | Binary flag | Marks affected dates for model attention |

### 3.4 Business-day alignment

All exogenous series are resampled from their native frequency (daily, weekly, monthly, quarterly) to the WTI business-day calendar using **forward-fill** (no interpolation). This respects the real-world information availability: a monthly GDP release is only known on its release date, held forward until the next release.

---

## 4. Feature Engineering

**File:** `00_setup_data_features.py`, `01a_vmd_features.py`

### 4.1 Feature categories

Over 100 features are engineered across 15 categories. Each has a methodological justification:

**A. Price transformation & returns**
- `log_price`, `log_ret_1d/5d/21d` — stabilize variance; avoid level-based non-stationarity
- `raw_ret_signed_log`, `raw_dd_below_floor` — preserve shock information after winsorization

**B. Volatility (rolling window)**
- `vol_7d/30d/90d`, `vol_ratio_7_30` (spike detector: >1 = recent stress), `atr_14`
- Multi-scale: captures both short-term jumps (7d) and medium-regime shifts (90d)

**C. Momentum & trend (technical indicators)**
- `rsi_14` (overbought/oversold), `macd/macd_signal/macd_diff` (momentum crossover), `cci_20` (cycle turning points)
- Widely used by practitioner oil traders; provide signal in 1d–7d range

**D. Mean reversion (moving averages)**
- `ma_5/20/50/200` and their ratios to current price
- MA deviations measure distance from equilibrium — a mean-reversion signal

**E. Bollinger bands**
- `bb_upper/lower` (±2σ bands), `bb_width` (normalized volatility), `bb_pct` (position within band 0–1)
- Captures whether price is in compression (low bb_width) or expansion (high bb_width)

**F. Cross-asset relationships**
- `brent_wti_spread` — regional supply/pipeline arbitrage signal
- `gold_oil_ratio` — commodity risk-off correlation
- `gas_crude_ratio` — energy complex co-movement

**G. Market stress (real-time)**
- `vix_level`, `vix_ma21`, `vix_pctile` (252-day percentile rank), `crisis_flag` (VIX > 30)
- Regime-relative extremeness: a VIX of 30 means different things in 2008 vs. 2017

**H. Z-score normalization**
- `z_score_90d`, `z_score_252d` — how many standard deviations from recent history
- Provides regime-relative context; same price level can be cheap or expensive depending on regime

**I. Temporal / seasonal**
- `month_sin/cos`, `dow_sin/cos` — circular encoding avoids false proximity of Dec/Jan or Mon/Fri
- `opec_month`, `opec_decay_5d` — OPEC meets in odd months; market attention heightened ±2 days

**J. CFTC positioning (Tier B6 — medium/long horizons only)**
- `cot_mm_net = (mm_long − mm_short) / open_interest` — bounded net positioning ratio
- `cot_mm_net_z` — 252-day z-score (extreme positioning detector)
- `cot_mm_net_chg_5d` — weekly positioning momentum
- Only in MED/LONG features (90d/180d): positioning dynamics are noise at 1d/7d

**K. Futures term structure (Tier B2)**
- `ts_slope = log(WTI_C1) − log(WTI_C4)` — true contango/backwardation
- Positive = backwardation (storage-constrained, bullish structure)
- Negative = contango (oversupply, bearish structure)
- Replaces the proxy `realized_carry = log_price(t) − log_price(t−63)`

**L. Event dummies (Tier B5)**
- COVID negative price (2020-04-20), Russia invasion (2022-02-24 to 2022-03-15), OPEC war (2014-06-01 to 2016-03-31)
- Hard-coded flags for unprecedented events that no rolling feature captures

### 4.2 Horizon-specific feature sets

Different signals matter at different timescales. The pipeline assigns features to horizons:

| Feature Set | Horizon | # Features | Added vs. shorter |
|---|---|---|---|
| BASE | 1d, 7d | ~48 | Short technicals, tactical sentiment, OVX, event dummies |
| MED | 90d | ~61 | BASE + inventory, CFTC positioning, EPU/GPR, intermediate macro |
| LONG | 180d | ~69 | MED + CPI YoY, IndPro YoY, GDP QoQ, Fed Funds, yield spread, long-term MAs |

**Key insight:** adding 90d/180d-only features to short models adds noise; excluding them from long models loses signal. The split is economically motivated — macro variables move on quarters/years, not days.

### 4.3 HMM regime detection

Rather than a binary crisis flag (VIX > 30), a **4-state Gaussian HMM** (hmmlearn) is trained on three series: [log_ret_1d, vol_30d, vix_level]. The 4 states capture calm, trending, volatile, and crisis-like regimes. States are re-ordered by ascending mean volatility so state 0 = calmest, state 3 = most turbulent.

The posterior regime probabilities (`regime_prob_0` through `regime_prob_3`) are added as features to **every** horizon. These are richer than a hard label because the model sees the uncertainty at regime boundaries.

### 4.4 Target variables

All targets are forward log-returns (not price levels):

```
target_{hk} = log_price(t + h) − log_price(t)
```

Log-returns are more stationary, more normally-distributed in calm periods, and make inverse-transformation straightforward: `price_hat = price_t × exp(logret_hat)`.

---

## 5. Track A — Leaky Paper Reproductions

**Files:** `_paper_helpers.py`, `_lh_forecasters.py`, `A_components_*.py`, `A_yu_*.py`, `A_huang_*.py`

### 5.1 Why reproduce leaky pipelines?

Publishing honest numbers alone would not constitute a contribution — one could simply argue that the honest model is weaker. By *first reproducing the published numbers verbatim* and *then running the honest equivalent*, the thesis makes three claims simultaneously:
1. The published numbers are correct under the published protocol.
2. The protocol contains lookahead leakage.
3. The leakage premium = (Track A MAE) − (Track B MAE) quantifies how much the published advantage is real vs. artificial.

### 5.2 Yu et al. (2025) WOA-VMD-FE protocol

**Step-by-step:**

**Step 1 — Whale Optimization Algorithm (WOA) hyperparameter search**
- WOA is a population-based metaheuristic (pop=10, iters=10)
- Searches over K ∈ {3,4,5,6,7} modes × α ∈ {500,1000,2000,3000,5000} bandwidth = 25 candidates
- Fitness function: **fuzzy entropy of the VMD residual**
  - Lower fuzzy entropy = smoother, more predictable residual = better decomposition
  - Fuzzy entropy (Chen et al. 2007): `FE = log(φ(m)) − log(φ(m+1))` where φ(·) is mean fuzzy membership across m-embeddings
- Returns optimal (K*, α*); defaults to (5, 2000) if no improvement found
- **Compute cost:** ~7 minutes per horizon on CPU

**Step 2 — VMD decomposition**
- `VMD(log_price[full_series], K*, α*, tau=0.0)` → K* variational modes u_k(t)
- Applied to the *entire* 1986–2026 series before train/test split (**this is the leakage**)

**Step 3 — Fuzzy entropy clustering**
- Compute FE per mode
- LF (low-frequency) band = modes with FE ≤ 33rd percentile — most predictable
- MF (mid-frequency) band = modes with FE in 33rd–67th percentile
- HF (high-frequency) band = modes with FE > 67th percentile — noisiest
- Rationale: low-FE modes are smooth trend signals; high-FE modes are near-random noise

**Step 4 — Per-band neural network training**
- MinMaxScaler fitted on **train slice only**, applied to band → scale [0,1]
- RevIN instance normalization is **disabled** (Identity) because band-value prediction requires level information; RevIN would strip the level and cause convergence to train-mean
- Separate NN (PatchTST, iTransformer, etc.) per band, trained on 70% split
- Predictions inverse-scaled back to band space

**Step 5 — Reconstruction**
- `log_price_hat(t+h) = LF_hat(t+h) + MF_hat(t+h) + HF_hat(t+h)`
- `price_hat = price(t) × exp(log_price_hat(t+h) − log_price(t))`

### 5.3 Huang et al. (2024) VMD-EMD protocol

- **VMD** with fixed K=3, α=2000 (no WOA search) → 3 Variational Mode Functions (VMFs)
- **EMD secondary**: EMD applied to the VMD residual → ~15 Intrinsic Mode Functions (IMFs)
- Total ~18 components, each gets a separate NN
- Optional: skip high-fuzzy-entropy components (use train-mean constant); controlled by `fe_keep_quantile`
- Applied to full series → same lookahead leakage as Yu protocol

### 5.4 Per-component CPU forecasters

When NNs are not available (local CPU phase), these extrapolators are used:

| Method | Mechanism | Best for |
|---|---|---|
| Ridge | Windowed supervised linear regression | Smooth, trend-heavy LF bands |
| DLinear | Decompose window into trend + seasonal; fit two linear maps | Bands with clear trend + cycle |
| NLinear | Subtract last value; predict delta; add back | Trend-carrying bands at long horizons |
| AR(p=10) | Fit AR with intercept; iterate h steps | Weakly autocorrelated components |
| Theta | Blend linear trend + SES (α=0.2); equal weight | Noisy components needing stability |
| ARIMA(5,1,0) | Full ARIMA (slow; strided every 10 origins) | Seasonal + trend; when fitting budget allows |
| ETS (additive, damped) | Holt-Winters with damping | Seasonal + damped trend |
| Auto-select | Run all on validation slice; pick best MAE | Adaptive per band |

### 5.5 Disk-backed decomposition cache

WOA-VMD costs ~7 minutes per horizon. The result is saved to `cache/yu_decomp_{hk}.pkl` after the first run. On every subsequent session (including all Colab GPU sessions), the decomposition loads from disk in <1 second. This is the key bridge between Phase 1 (local CPU prep) and Phase 2 (cloud GPU training): all 40+ NN architectures share one set of cached decompositions.

---

## 6. Track B — Honest Walk-Forward

**Files:** `_run_wf_arch.py`, `_forecast_template.py`, `09_vmd_hybrid.py`, `B_honest_decomp.py`, `B_wf_*.py`

### 6.1 The honest constraint

At every test origin t, the model sees **only data strictly before t**:
- Features: computed from price and exogenous data up to t−1
- Decomposition (if any): VMD applied to a rolling window ending at t

No future information enters at any point. This is the standard academic definition of a valid out-of-sample test.

### 6.2 Rolling VMD hybrid (09_vmd_hybrid.py)

The leakage-free version of the Track A decomposition approach:

```
For each test origin t (stride = every 5 business days):
    window = log_price[t − 360 : t]     ← 360-day trailing window (past only)
    VMD(window, K=5, α=2000) → 5 modes
    AR(p=10).fit(mode_train) and forecast h steps per mode
    log_price_hat(t+h) = sum of 5 mode forecasts
    price_hat = price(t) × exp(log_price_hat − log_price(t))
```

- Stride = 5 business days reduces compute: ~400 VMD calls per horizon vs. 2000 without stride
- Optional: CEEMDAN secondary decomposition on noisiest mode (Wang et al. 2025) if PyEMD available

### 6.3 Walk-forward neural network split mechanics

| Parameter | 1d | 7d | 90d | 180d |
|---|---|---|---|---|
| Test start date | 2007-01-01 | 2007-01-01 | 2007-01-01 | 2007-01-01 |
| Test window size | 12 months | 12 months | 24 months | 24 months |
| Approximate folds | ~18 | ~18 | ~9 | ~9 |
| Min training samples | lookback + h + 50 | — | — | — |

**Why 2007-01-01 as test start?** This ensures the out-of-sample period contains all three major crises: GFC (2008-09), OPEC War (2014-16), and COVID (2020). A test period without crises would produce artificially optimistic MAE/RMSE numbers.

Each fold uses an **expanding training window** (all data from 1986 to fold start) — models accumulate more training data as the walk-forward progresses.

### 6.4 Per-horizon architecture choices

| Horizon | lookback | patch size | d_model | layers | dropout | loss function |
|---|---|---|---|---|---|---|
| 1d | 60 days | 8 tokens | 128 | 3 | 0.15 | MSE |
| 7d | 96 days | 8 tokens | 128 | 3 | 0.20 | MSE |
| 90d | 336 days | 16 tokens | 128 | 4 | 0.25 | Student-t NLL |
| 180d | 360 days | 16 tokens | 128 | 3 | 0.25 | Student-t NLL |

**Loss function rationale:**
- **MSE (1d/7d):** Returns at short horizons are approximately Gaussian in calm periods. MSE provides clean gradients and fast convergence.
- **Student-t NLL (90d/180d):** Log-returns at long horizons are fat-tailed. The model outputs two channels — predicted mean μ and predicted scale log(σ) — and minimizes negative Student-t log-likelihood. This forces the model to learn uncertainty calibration as part of the training objective. The same σ output is recycled for conformal prediction intervals.

**Why degrees of freedom = 5?** df=5 gives tails roughly matching empirical WTI kurtosis at long horizons while maintaining finite variance (df > 2) and finite kurtosis (df > 4).

### 6.5 Normalization (RobustScaler)

Each training fold: fit `sklearn.preprocessing.RobustScaler` on training rows only. The scaler uses the **median and IQR** instead of mean and std — this makes it robust to the 5–10 extreme crash days present in every training fold. The fitted scaler is applied to validation and test rows (no refitting), and inverse-transform converts predictions back to log-return space.

### 6.6 Seed aggregation and interval estimation

- `N_SEEDS = 3` per fold: train each fold 3 times with different random seeds
- Aggregate across seeds: **median** point forecast (robust to outlier seeds)
- Seed-level spread → inner PI bound: 10th/90th percentile of price across 3 seed predictions
- Combined with conformal calibration (see Chapter 8) for final intervals

### 6.7 Per-fold checkpointing

`checkpoints/{MODEL}_{hk}_fold{NN}.pkl` is written after each fold completes. On reconnect, completed folds are loaded and the next fold starts immediately. This prevents full restart after GPU session disconnect — critical for 90d/180d where each fold takes ~5 minutes on GPU.

---

## 7. Neural Architectures Deployed

**Files:** `02_model_shared.py`, `_lh_sota_models.py`, `_fedformer_plus.py`, `_models_plus.py`

All architectures are registered in `MODEL_REGISTRY` in `02_model_shared.py` and trained via the shared walk-forward loop (`_forecast_template.py`) or the Track A template (`_run_paper_arch.py`).

| Architecture | Family | Key Innovation | WTI-specific advantage |
|---|---|---|---|
| **PatchTST** | Transformer | Patch-level token embedding, channel-independent | Multi-scale: 8-token patches capture both day-level and week-level patterns |
| **iTransformer** | Transformer | Inverted architecture: embed per-variable, attend across variables | Naturally handles the 60–70 multivariate features |
| **Autoformer** | Transformer | Autocorrelation-based attention | Seasonal decomposition built into attention mechanism |
| **Informer** | Transformer | ProbSparse attention O(n log n) | Efficient on long 360-day lookbacks |
| **FEDformer** | Transformer | Frequency-enhanced decomposition + attention | Captures spectral structure of oil price cycles |
| **FreEformer** | Transformer | Learnable frequency filters in attention | Adaptive bandpass; model learns relevant frequency ranges |
| **TFT** | Transformer | Gated skip-connections + variable selection | Interpretable importance weights; handles mixed-frequency exogenous features |
| **NHiTS** | MLP | Hierarchical interpolation, multi-rate pooling | Efficient direct multi-horizon forecasting without autoregressive rollout |
| **TimeMixer** | MLP | Temporal mixing via pooling + decomposition blocks | Fast non-attention baseline; strong at 1d/7d |
| **TiDE** | MLP | Dense encoder-decoder with residual | Dense linear baseline; surprisingly competitive at all horizons |
| **DLinear/NLinear** | Linear | Trend-seasonal split / last-value normalization | Interpretable; strong baseline for 1d/7d; resistant to overfitting |
| **TCN** | CNN | Dilated causal convolutions | Long effective receptive field without RNN gradient issues |
| **TCN-BiLSTM** | CNN + RNN | CNN local features + LSTM sequential memory | Hybrid: local pattern detection + long-range sequential dependency |
| **SegRNN** | RNN | Segment-based recurrence | Better gradient flow than vanilla LSTM; handles 360-day lookbacks |
| **TimeRAG** | Retrieval | k=5 nearest historical analog windows | Zero-training; interpretable; identifies past crises similar to present |
| **Chronos-Bolt** | Foundation | Amazon zero-shot pre-trained model | Tests generalization: can a model trained on 100K+ series forecast WTI without finetuning? |
| **TimesFM** | Foundation | Google zero-shot pre-trained model | Same zero-shot benchmark; different pretraining corpus |

**Crisis-stratified sample weighting (`_crisis_weighting.py`):**
GFC, OPEC War, and COVID dates receive **3× sample weight** during training. Without this, a model trained on 37 years of data will implicitly minimize loss by ignoring the ~300 crisis days (0.8% of the data) — exactly the days the thesis focuses on.

---

## 8. Probabilistic & Tail-Risk Framework

**Files:** `13_probabilistic_forecast.py`, `07_risk_evt.py`, `07b_garch_evt.py`, `_conformal_plus.py`

### 8.1 Predictive intervals (split-conformal calibration)

Rather than assuming a specific distribution for errors, conformal prediction gives **distribution-free** coverage guarantees.

**Method:**
1. Collect validation-fold residuals: `r_t = actual_logret − predicted_logret`
2. For Student-t models: standardize by predicted scale: `z_t = r_t / σ_t`
3. Interval at test origin: `[predicted ± σ × quantile(z, 10%), predicted ± σ × quantile(z, 90%)]`
4. `lower_price = price_t × exp(lo_logret)` and symmetrically for upper

This provides approximately 80% empirical coverage without any distributional assumption. The only requirement is exchangeability of the calibration and test residuals — a reasonable assumption within each walk-forward fold.

### 8.2 Full probabilistic output (13_probabilistic_forecast.py)

For each horizon, the stage synthesizes all available model predictions into:

- **8-quantile fan chart:** {5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%} — using pinball loss-weighted quantile regression across model ensemble
- **500 Monte Carlo scenario paths:** bootstrapped from empirical residual distribution; each path represents a plausible future trajectory
- **PIT histogram:** `u_t = F_t(actual_logret)` — if model is well-calibrated, u_t ~ Uniform(0,1)
  - Kolmogorov-Smirnov test: ks_pit < 0.10 → well-calibrated
  - U-shaped histogram → over-confident; humped → over-dispersed
- **Reliability diagram:** plot nominal coverage (x-axis) vs. empirical coverage (y-axis) across 19 levels; perfect calibration = 45° line
- **CRPS (Continuous Ranked Probability Score):** proper scoring rule combining sharpness and calibration in a single number

### 8.3 GPD/EVT tail-risk (07_risk_evt.py)

Standard Gaussian intervals severely underestimate crash-day losses. GPD/EVT provides statistically rigorous tail quantification.

**Peaks-Over-Threshold (POT) method:**
1. Threshold `u` = 90th percentile of |forecast error|
2. Exceedances: `e_i = |error_i| − u` for all |error_i| > u
3. Fit GPD(ξ, β) via maximum likelihood to exceedances

**VaR formula:**
```
VaR_α = u + (β/ξ) × [ (n/Nu × (1−α))^(−ξ) − 1 ]
```
where n = total sample size, Nu = number of exceedances above u, ξ = GPD shape (>0 = fat tail), β = GPD scale.

**Expected Shortfall:**
```
ES_α = VaR_α / (1−ξ) + (β − ξu) / (1−ξ)
```
ES is the average loss conditional on exceeding VaR — the metric required by Basel III IMA for banks' internal models.

**Backtests:**
- **Kupiec POF test:** likelihood ratio test that exceedance count matches expected frequency; p > 0.05 → well-calibrated
- **Christoffersen independence test:** checks whether exceedances cluster (volatility bunching); p > 0.05 → violations are random (good)

Reports VaR and ES at α ∈ {90%, 95%, 99%} for each horizon.

### 8.4 GARCH(1,1)-EVT (07b_garch_evt.py)

The static GPD fit treats all days equally. GARCH-EVT makes VaR **time-varying** — wider on volatile days, tighter on calm days. This is the McNeil-Frey (2000) framework, standard in institutional risk management.

**Pipeline:**
1. Fit GARCH(1,1): `σ_t² = ω + α ε²_{t-1} + β σ²_{t-1}`
2. Standardize residuals: `z_t = ε_t / σ_t` (removes clustering → approx i.i.d.)
3. POT-GPD on z_t (static fit on i.i.d. standardized residuals)
4. Dynamic VaR: `VaR_t = σ_t × GPD_quantile(α)` — scale by today's conditional volatility

The dynamic σ_t is also saved as a feature (`garch_sigma_{hk}`) for downstream models.

---

## 9. Ensemble & Comparison Framework

**Files:** `C_ensemble.py`, `C_compare.py`

### 9.1 SLSQP ensemble weighting

With 40+ models producing predictions, simple averaging wastes information. SLSQP finds the optimal convex combination:

```
minimize   MAE( Σ w_i × pred_i, actual )   over validation set
subject to w_i ≥ 0   (non-negative)
           Σ w_i = 1  (probability distribution)
```

Solver: `scipy.optimize.minimize(method="SLSQP")`, converges in <1 second for ≤50 models. Three ensemble variants are computed:
1. **SLSQP (MAE objective):** optimal non-negative convex combination
2. **Simple average:** equal weights = (1/n, ..., 1/n)
3. **Ridge stacking:** meta-learner Ridge(α=10) trained on validation predictions

Best of the 3 selected by validation MAE. The ensemble outputs its own `predictions_Ensemble_DDP_v2_{hk}.csv`.

### 9.2 Model comparison and benchmark

`C_compare.py` scans all `eval_*.pkl` files in the output directory and computes a ranked leaderboard per (horizon × leaky/honest) stratum:

| Column | Description |
|---|---|
| MAE | Primary ranking metric |
| RMSE | Quadratic penalty; sensitive to crash days |
| MAPE | Scale-independent; used for Huang benchmark comparison |
| leaky | True = Track A (whole-series decomp); False = Track B (honest) |
| beats_bench | True if MAE < 3.45 AND MAPE < 7.84% at 90d/180d |
| leakage_premium | Track A MAE − Track B MAE (positive = inflation from leakage) |

The `leakage_premium` column is the central empirical contribution — it directly quantifies how much each architecture's published performance depends on the methodological flaw.

### 9.3 Diebold-Mariano statistical significance

To establish that improvements over the Huang benchmark are statistically significant and not due to chance:
- Compute loss differentials: `d_t = |actual_t − pred_model_t| − |actual_t − pred_benchmark_t|`
- DM statistic: `S = mean(d) / sqrt(LRV(d))` where LRV = long-run variance (Newey-West HAC)
- Null hypothesis H₀: equal predictive accuracy; reject if |S| > 1.96 (5% level)
- Applied at 90d and 180d for each Track B model vs. Huang benchmark

---

## 10. Implementation Architecture & Reproducibility

**Files:** `_env.py`, `run_local_cpu.py`, `run_short.py`, `run_long.py`, `run_pipeline.py`

### 10.1 Exec-chaining design

All files run in a single Python process via `exec(open('file.py', encoding='utf-8').read())`. Globals persist across stages — no redundant reloads of the full 40-year dataset. Helper files (`_paper_helpers.py`, `_plotting.py`) check `if "save_fig" not in globals()` before loading to avoid double-initialization.

**Why not import statements?** The pipeline was designed for interactive Jupyter/Colab execution where `import` semantics are inconsistent across cells and session restarts. `exec()` with explicit globals sharing replicates a single-process environment even in notebook cells.

### 10.2 Three-phase workflow

```
Phase 1 — LOCAL CPU (free, ~3–4 hours once)
  run_local_cpu.py
  → 00_setup_data_features.py   (data + features)
  → 01a_vmd_features.py         (rolling VMD columns)
  → 01_black_swan_baselines.py  (naive/ARIMA/ETS baselines)
  → A_components_linear/classical/hybrid.py  (CPU Track A extrapolators)
  → B_honest_decomp.py, 09_vmd_hybrid.py    (CPU Track B)
  → 07_risk_evt.py, 07b_garch_evt.py        (EVT risk)
  Writes: df_features.pkl, feat_cols.pkl, cache/yu_decomp_*.pkl
  
       ↓ sync results/ to Google Drive
       
Phase 2 — CLOUD GPU (Colab/Kaggle, ~3–6 hours)
  run_short.py  (1d + 7d)   or   run_long.py  (90d + 180d)
  → A_yu_*.py, A_huang_*.py   (Track A NN archs)
  → B_wf_*.py                 (Track B honest NN archs)
  → B_foundation_chronos/timesfm.py  (zero-shot)
  Reads: df_features.pkl, cache/yu_decomp_*.pkl (from Phase 1)
  Writes: predictions_*.csv, eval_*.pkl, checkpoints/*.pkl

       ↓ sync results/ back to local
       
Phase 3 — LOCAL CPU (analysis, ~30 min)
  13_probabilistic_forecast.py   (quantile fan + scenarios)
  C_ensemble.py                  (SLSQP blend)
  C_compare.py                   (leaderboard + benchmark)
  06_analysis_plots.py           (publication figures)
  17_audit_pipeline.py           (artifact audit)
```

### 10.3 Skip-guard (resume after disconnect)

```python
if already_done(name, hk) and not FORCE_RERUN:
    print(f"⏭️  skip {name} {hk} (already done)")
    continue
```

`already_done(name, hk)` checks that both `predictions_{name}_{hk}.csv` and `eval_{name}_{hk}.pkl` exist. Re-running the orchestrator after a GPU disconnect picks up exactly where it stopped, skipping all completed (model × horizon) pairs.

### 10.4 Output folder layout

```
DDP_OUTPUT_DIR/
├── df_features.pkl              — master feature matrix (40 years × 100+ features)
├── wti_raw.pkl, wti_wins.pkl    — raw + winsorized price series
├── feat_cols.pkl                — horizon → feature list mapping
├── regimes.pkl                  — HMM regime assignments + posteriors
├── predictions_{MODEL}_{hk}.csv — per-model forecasts (columns: date, actual, pred, lo, hi)
├── eval_{MODEL}_{hk}.pkl        — per-model evaluation dict (MAE, RMSE, MAPE, metadata)
├── risk_evt_{hk}.pkl            — GPD/EVT tail risk results per horizon
├── garch_evt_{hk}.pkl           — GARCH-EVT dynamic VaR results
├── cache/
│   ├── yu_decomp_{hk}.pkl       — WOA-VMD decomposition (7 min → <1 s on reconnect)
│   └── huang_decomp_{hk}_{K}_{α}.pkl
├── checkpoints/
│   └── {MODEL}_{hk}_fold{NN}.pkl — per-fold NN weights for resumable training
├── figures/all/
│   ├── {name}.html              — interactive Plotly (for thesis defense slides)
│   └── {name}.png               — high-DPI static (for thesis PDF, scale=3 → 3840×2160)
├── tables/all/
│   └── {name}.csv               — numeric tables (also saved as styled go.Table figures)
└── logs/
    └── run_manifest.jsonl       — append-only log: stage, status, MAE, runtime, timestamp
```

---

## 11. Summary of Design Decisions

Every non-obvious choice and the alternative that was rejected:

| Decision | Choice Made | Reasoning | Alternative Rejected |
|---|---|---|---|
| **Price floor for log-transform** | 0.1th percentile (~$10.95/bbl) | Preserves shock magnitude in `raw_*` features; 1st pctile would clip 1998/2002 events | Delete negative prices (loses COVID signal) |
| **Lookahead leakage** | Reproduced deliberately in Track A | Contest the literature; quantify the leakage premium empirically | Ignore or hide the issue |
| **Walk-forward test start** | 2007-01-01 | Includes GFC 2008, OPEC War 2014, COVID 2020 in OOS | 2015 start (misses GFC) |
| **Short-horizon loss** | MSE | Near-Gaussian; fast convergence | Huber (unnecessary robustness adds hyperparameter) |
| **Long-horizon loss** | Student-t NLL (df=5) | Fat-tailed returns; free σ output for PIs | MSE (Gaussian; underestimates tail probability by 10×) |
| **RevIN in Track A** | Removed (Identity) | Band prediction requires level info; RevIN strips it → model → train-mean | Keep (converges to constant prediction) |
| **Decomposition caches** | Disk-backed PKL | GPU reconnect in <1s; 7 min WOA-VMD amortized across 40+ archs | Recompute on reconnect (loses GPU hours) |
| **Regime detection** | 4-state Gaussian HMM | Data-driven; posterior probs richer than binary VIX threshold | VIX > 30 threshold |
| **CFTC lag** | Release date (+3 bdays) | Methodologically defensible; no future information | Snapshot date (Tuesday; information not yet public) |
| **Ensemble method** | SLSQP (MAE objective) | Convex, non-negative weights; interpretable; validated on out-of-sample fold | Equal-weight (ignores relative model skill) |
| **PI coverage level** | 80% (10th/90th percentile) | Standard for hedging decisions; 95% is too wide to drive trading decisions | 95% |
| **Crisis sample weighting** | 3× for GFC/OPEC/COVID | Prevents implicit ignoring of <1% crisis days that dominate practitioner losses | Uniform weights (model ignores crashes) |
| **Feature normalization** | RobustScaler (median, IQR) | Robust to crash-day outliers in every training fold | StandardScaler (mean/std; crash days inflate std and re-scale all features) |
| **COT horizon scope** | MED/LONG features only (90d/180d) | Positioning dynamics are noise at 1d/7d; signal at 90d/180d | Add to BASE (adds noise, degrades short-horizon predictions) |

---

## 12. References

1. **Xu, B. & Niu, H. (2024).** Decomposition lookahead bias in crude oil price forecasting. *Energy Economics*, 128. — Motivates the dual-track leaky/honest design.

2. **Chen, W., Wang, Z., et al. (2007).** Measuring complexity using fuzzy entropy. *Information Sciences*. — Fuzzy entropy metric for WOA fitness function and band clustering.

3. **Dragomiretskiy, K. & Zosso, D. (2014).** Variational Mode Decomposition. *IEEE Transactions on Signal Processing*, 62(3), 531–544. — Foundation of the VMD decomposition used in both tracks.

4. **Yu, P., et al. (2025).** WOA-VMD-FE decomposition protocol for crude oil price prediction. — Direct source for Track A Yu protocol; headline benchmark MAE numbers.

5. **Huang, X., et al. (2024).** VMD-EMD two-stage decomposition for WTI crude oil. — Source of the 3.45 $/bbl @ 180d benchmark; Track A Huang protocol.

6. **Wang, J., et al. (2025).** CEEMDAN-VMD hybrid forecasting for crude oil. *Energy Reports*, S2590123025024612. — Secondary VMD/CEEMDAN combination in 09_vmd_hybrid.py.

7. **Liang, C. & Lv, T. (2020).** Oil volatility risk and expected stock returns. *Journal of Futures Markets*. — OVX (oil-specific VIX) adds 10–20% RMSE reduction; motivates OVX feature inclusion.

8. **Kang, S.H., et al. (2020).** Hedge funds' performance in commodity markets. — CFTC managed-money positioning as medium-horizon WTI predictor.

9. **Yang, G., et al. (2020).** Crude oil price forecasting using Google Trends. arXiv:2002.09656. — Google Trends as a nowcasting signal for crude oil.

10. **Elshendy, M., et al. (2018).** Using Google Trends data to predict oil price. *Energy*, 165. — Validates Google Trends for energy forecasting.

11. **McNeil, A.J. & Frey, R. (2000).** Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*, 7(3–4), 271–300. — GARCH-EVT framework in 07b_garch_evt.py.

12. **Kupiec, P.H. (1995).** Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2). — POF backtest for VaR calibration.

13. **Christoffersen, P.F. (1998).** Evaluating interval forecasts. *International Economic Review*, 39(4). — Independence test for VaR violation clustering.

14. **Diebold, F.X. & Mariano, R.S. (1995).** Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3). — DM test for statistical significance of MAE improvements.

15. **Nie, Y., et al. (2022).** A time series is worth 64 words: Long-term forecasting with transformers. *ICLR 2023*. — PatchTST architecture.

16. **Liu, Y., et al. (2024).** iTransformer: Inverted transformers are effective for time series forecasting. *ICLR 2024*. — iTransformer architecture.

---

*Document generated from DDP_v2 pipeline codebase — `00_setup_data_features.py`, `_paper_helpers.py`, `_run_wf_arch.py`, `_forecast_template.py`, `C_compare.py`, `C_ensemble.py`, `07_risk_evt.py`, `07b_garch_evt.py`, `13_probabilistic_forecast.py`, `_lh_forecasters.py`, `09_vmd_hybrid.py`.*

*Last updated: 2026-06-23*
