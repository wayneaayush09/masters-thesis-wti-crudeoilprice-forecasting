# ════════════════════════════════════════════════════════════════════
#  FILE : _forecast_template.py
#  ROLE : Core PatchTST walk-forward training loop  (Track B backbone)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Trains PatchTST (Nie et al. 2023) for one horizon (HORIZON_KEY)
#         using a walk-forward expanding-window scheme: train on everything
#         up to a cutoff, predict the next freq_months window, step forward
#         and repeat. Each fold trains N_SEEDS=3 seeds and averages. Outputs:
#         predictions_{MODEL_NAME}_{hk}.csv + eval_{MODEL_NAME}_{hk}.pkl.
#
#  WHY  — Walk-forward is the ONLY leakage-free evaluation for time series:
#         each test prediction was made without any future data visible.
#         The 2007 OOS start means GFC (2008), OPEC-War (2014), and COVID
#         (2020) all fall inside the genuine test window. Per-fold
#         checkpointing (see below) survives Colab/Kaggle disconnects.
#
#  HOW  — Per horizon:
#           1. Build WF_SPLITS (12- or 24-month steps from 2007 onward).
#           2. For each split: fit RobustScaler on TRAIN only → no leakage.
#           3. Train PatchTST with the per-horizon loss (MSE at 1/7d,
#              Student-t NLL at 90/180d for heavy-tail robustness).
#           4. Validate → α grid-search (naive-shrinkage) + split-conformal CI.
#           5. Aggregate ensemble across seeds → median point, 10-90% CI.
#           6. Checkpoint each fold to disk → resume after disconnect.
#
#  Architectures:  any key in MODEL_REGISTRY (set MODEL_NAME_OVERRIDE).
#  Loss selection: MSE=short horizons, Huber=90d, Student-t=180d (fat tails).
#  CI calibration: split-conformal quantile regression on validation residuals.
#  Shrinkage α:    grid-searched on val-window price MAE; floors at naive.
#
#  READS    : df_features.pkl, wti_raw.pkl, feat_cols.pkl  (Stage 0)
#             checkpoints/{MODEL}_{hk}_foldNN.pkl  (resume after disconnect)
#             outputs/optuna_best_hparams.pkl  (optional Optuna tuning)
#  WRITES   : predictions_{MODEL}_{hk}.csv, eval_{MODEL}_{hk}.pkl
#             checkpoints/{MODEL}_{hk}_foldNN.pkl  (per-fold resume points)
#  COMPUTE  : GPU-HEAVY  (run on Colab / Kaggle A100 or T4)
#  PIPELINE : exec'd by _run_wf_arch.py for each Track B driver
#  RUNTIME  : ~3-5 min per horizon on A100; ~8-12 min on T4
#             (checkpoint resume cuts this proportionally after a crash)
# ════════════════════════════════════════════════════════════════════
# DO NOT exec this file directly — call via a B_wf_*.py driver or
# _run_wf_arch.py which sets HORIZON_KEY and MODEL_NAME_OVERRIDE.
# ════════════════════════════════════════════════════════════════════

import pickle, warnings, gc
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader
except Exception as _te:
    raise RuntimeError(f"_forecast_template requires torch: {_te}. Run _env.py first.") from _te
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

# ── Per-horizon config (paper-aligned) ────────────────────────
# Patch + lookback chosen so #patches ≈ paper's "64-word" intuition.
HORIZONS = {
    # 1d: ADD-14 — paper-aligned single-step horizon. Tiny lookback, small patch.
    "1d":  {"h": 1,   "lookback": 60,  "patch": 4,  "layers": 2, "d_model": 64,
            "loss": "mse",      "n_quantiles": 1, "dropout": 0.15, "weight_decay": 1e-4},
    # 7d: near-Gaussian short horizon → MSE, light regularization
    "7d":  {"h": 7,   "lookback": 96,  "patch": 8,  "layers": 3, "d_model": 128,
            "loss": "mse",      "n_quantiles": 1, "dropout": 0.20, "weight_decay": 1e-4},
    # 90d/180d: noisy heavy-tailed → Student-t NLL + heavier regularization
    "90d": {"h": 90,  "lookback": 336, "patch": 16, "layers": 3, "d_model": 128,
            "loss": "studentt", "n_quantiles": 1, "dropout": 0.25, "weight_decay": 1e-3},
    "180d":{"h": 180, "lookback": 360, "patch": 16, "layers": 3, "d_model": 128,
            "loss": "studentt", "n_quantiles": 1, "dropout": 0.25, "weight_decay": 1e-3},
}

# ADD-14: Optuna-tuned hyperparameters override defaults if 16_hyperopt.py has been run.
try:
    with open(OUTPUT_DIR/"optuna_best_hparams.pkl", "rb") as _fh:
        _best = pickle.load(_fh)
    for _hk, _params in _best.items():
        if _hk in HORIZONS:
            HORIZONS[_hk].update(_params)
            print(f"  ℹ️  HORIZONS[{_hk}] overridden by Optuna best: {_params}")
except FileNotFoundError:
    pass
except Exception as _e:
    print(f"  ⚠️  Optuna override skipped: {_e}")

# Per-horizon walk-forward: start_test=2007 so GFC (2008-09), OPEC War (2014-16),
# and COVID (2020) all fall in genuine out-of-sample test windows.
# 21 years of initial training (1986→2007) >> max lookback 512 days.
WF_CONFIG = {
    "1d":  {"start_test": "2007-01-01", "freq_months": 12},   # ADD-14: ~18 splits at 1-day
    "7d":  {"start_test": "2007-01-01", "freq_months": 12},   # ~18 splits; all 3 crises OOS
    "90d": {"start_test": "2007-01-01", "freq_months": 24},   # ~9 splits
    "180d":{"start_test": "2007-01-01", "freq_months": 24},   # ~9 splits
}

# Optional speed knob: widen the per-horizon test window to retrain less often.
# e.g. WF_FREQ_MONTHS={"1d":24,"7d":24} halves the short-horizon fold count
# (18→~9). Unlike WF_SPLIT_LIMIT this does NOT drop early folds — the wider
# windows still tile 2007→present continuously, so every OOS date (incl. the
# GFC/OPEC/COVID crisis windows) still gets exactly one prediction; the model is
# just up to `freq_months` stale instead of 12. Default None = unchanged.
_wf_freq = globals().get("WF_FREQ_MONTHS", None)
if _wf_freq:
    for _fhk, _fm in _wf_freq.items():   # _fhk not _hk: avoid clobbering _run_wf_arch's loop var
        if _fhk in WF_CONFIG:
            WF_CONFIG[_fhk] = {**WF_CONFIG[_fhk], "freq_months": int(_fm)}
    print(f"  ⏩ WF_FREQ_MONTHS override applied: {_wf_freq}")

CRISES = {
    "GFC":      ("2008-09-01","2009-06-30"),
    "OPEC_War": ("2014-06-01","2016-03-31"),
    "COVID":    ("2020-02-01","2020-09-30"),
}
NEGATIVE_PRICE_DATE = "2020-04-20"
# Model + seeds are overridable by the driver (08_deep_comparators.py) via globals.
# Defaults reproduce the PatchTST headline run (03/04/05 set neither global).
#
# SCOPING NOTE: globals() inside a nested exec() points to the *kernel* namespace,
# not the outer exec's locals. A stale MODEL_NAME="MultiPatchTST" from a prior
# 03_forecast_multi.py run will bleed in even if the driver file set "PatchTST"
# locally. Guard: validate against MODEL_REGISTRY after it loads; reset to
# "PatchTST" if the name is unknown to the single-horizon template.
_raw_model_name = globals().get("MODEL_NAME", "PatchTST")
N_SEEDS         = globals().get("N_SEEDS_OVERRIDE", 3)   # paper-style; comparators use 2
MC_DROPOUT_T    = globals().get("MC_DROPOUT_T", 0)        # MC-Dropout passes at test inference; 0=disabled (default)

# ── Load shared model code ────────────────────────────────────
exec(open('02_model_shared.py', encoding="utf-8").read())   # defines PatchTST, WTIDataset, train_model, MODEL_REGISTRY, etc.
# Re-extend MODEL_REGISTRY with _models_plus archs (TiDE, SegRNN, DLinear, NLinear,
# FEDformerW, FreEformer). 02_model_shared resets the registry each time it runs,
# so we must reload _models_plus immediately after to restore those entries.
import os as _ft_os
if _ft_os.path.exists("_models_plus.py"):
    exec(open("_models_plus.py", encoding="utf-8").read())
# Optional epoch override (heavy models like Time-LLM run fewer epochs)
MAX_EPOCHS = globals().get("MAX_EPOCHS_OVERRIDE", MAX_EPOCHS)

# Validate model name now that MODEL_REGISTRY is available.
MODEL_NAME = _raw_model_name if _raw_model_name in MODEL_REGISTRY else "PatchTST"
if MODEL_NAME != _raw_model_name:
    print(f"  ℹ️  MODEL_NAME='{_raw_model_name}' not in single-horizon registry "
          f"(stale global from prior run?) → reset to 'PatchTST'")
# Filename suffix: "" for PatchTST (keeps 06/07 paths), "_<Model>" otherwise
MODEL_SUFFIX = "" if MODEL_NAME == "PatchTST" else f"_{MODEL_NAME}"

# ── Load data pickles ─────────────────────────────────────────
with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df      = pickle.load(f)
with open(OUTPUT_DIR/"wti_raw.pkl",    "rb") as f: wti_raw = pickle.load(f)
with open(OUTPUT_DIR/"feat_cols.pkl",  "rb") as f: FEAT    = pickle.load(f)
print(f"✅ Data loaded: df {df.shape}")

TARGET_COLS = {"1d":"target_1d","7d":"target_7d","90d":"target_90d","180d":"target_180d"}

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true.abs() > 1
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    print(f"  {label:25s}  MAE={mae:.2f}  MSE={mse:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")
    return {"label":label,"MAE":mae,"MSE":mse,"RMSE":rmse,"MAPE":mape}

# ── Walk-forward splits (per-horizon) ─────────────────────────
def get_wf_splits(df, horizon_key):
    cfg = WF_CONFIG[horizon_key]
    test_starts = pd.date_range(cfg["start_test"], df.index[-1], freq=f"{cfg['freq_months']}MS")
    return [(df.index[0].strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%d"),
             ts.strftime("%Y-%m-%d"),
             min(ts+pd.DateOffset(months=cfg["freq_months"]), df.index[-1]).strftime("%Y-%m-%d"))
            for ts in test_starts]

WF_SPLITS = get_wf_splits(df, HORIZON_KEY)
# Optional: limit to the last N splits (heavy models like Time-LLM) — keeps the
# most recent regimes (incl. COVID) where the text/news signal matters most.
_wf_limit = globals().get("WF_SPLIT_LIMIT", None)
if _wf_limit:
    WF_SPLITS = WF_SPLITS[-int(_wf_limit):]
print(f"Walk-forward splits ({HORIZON_KEY}): {len(WF_SPLITS)}")

# ── MAIN TRAINING LOOP ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Model: {MODEL_NAME}   |   Training horizon: {HORIZON_KEY}   |   seeds: {N_SEEDS}")
print(f"{'='*60}")

cfg       = HORIZONS[HORIZON_KEY]
feat_cols = FEAT[HORIZON_KEY]
tgt_col   = TARGET_COLS[HORIZON_KEY]
lookback  = cfg["lookback"]
h         = cfg["h"]

ensemble_preds  = defaultdict(list)   # date → list of point log-returns (mu)
ensemble_sigma  = defaultdict(list)   # date → list of predictive scales (Student-t sigma)
insample_preds  = defaultdict(list)   # date → list of FITTED (in-sample) log-returns
insample_sigma  = defaultdict(list)   # date → list of fitted predictive scales
val_pred_acc    = []                  # validation point log-returns (for α grid-search)
val_sigma_acc   = []                  # validation predictive scales σ (for conformal CI)
val_date_acc    = []                  # validation origin dates (for price-MAE grid-search)
mc_epistemic_acc = defaultdict(list)  # date → list of per-pass variance estimates (C4: MC-Dropout epistemic UQ)
loss_type       = cfg["loss"]        # cfg defined just above

def _decode_output(out, loss_type):
    """Return (point_logret, sigma_or_None) as numpy arrays from a model batch output."""
    out = out.float().cpu()
    if loss_type == "studentt":
        mu    = out[:, 0].numpy()
        sigma = (torch.nn.functional.softplus(out[:, 1]) + 1e-6).numpy()
        return np.atleast_1d(mu), np.atleast_1d(sigma)
    if loss_type == "quantile":
        return np.atleast_1d(out[:, 2].numpy()), None     # median quantile
    return np.atleast_1d(out.squeeze(-1).numpy()), None    # mse / huber

# ── Per-fold checkpoint setup ──────────────────────────────────────────────────
# Each completed WF split is pickled to checkpoints/{MODEL}_{hk}_foldNN.pkl so a
# reconnected session can skip already-trained folds and resume from the next one.
# Most valuable at 90d/180d where each fold is 5+ min of GPU time.
# Gate: WF_CHECKPOINT (default True).  Disable with WF_CHECKPOINT=False in driver.
WF_CHECKPOINT = globals().get("WF_CHECKPOINT", True)
_ckpt_dir = globals().get("CHECKPOINTS_DIR", OUTPUT_DIR / "checkpoints")
if WF_CHECKPOINT:
    _ckpt_dir.mkdir(parents=True, exist_ok=True)

_completed_folds = set()
if WF_CHECKPOINT and not globals().get("FORCE_RERUN", False):
    for _fi in range(len(WF_SPLITS)):
        _fp = _ckpt_dir / f"{MODEL_NAME}_{HORIZON_KEY}_fold{_fi:02d}.pkl"
        if _fp.exists():
            try:
                with open(_fp, "rb") as _f:
                    _ck = pickle.load(_f)
                # Merge checkpointed predictions/sigma into the live accumulators
                for _dt, _pl in _ck.get("preds", {}).items():
                    ensemble_preds[_dt].extend(_pl)
                for _dt, _sl in _ck.get("sigma", {}).items():
                    ensemble_sigma[_dt].extend(_sl)
                # Merge val calibration data (needed for α grid-search after the loop)
                val_pred_acc.extend(_ck.get("vpreds", []))
                val_sigma_acc.extend(_ck.get("vsigmas", []))
                val_date_acc.extend(_ck.get("vdates", []))
                _completed_folds.add(_fi)
                print(f"  ✅ [ckpt] fold {_fi:02d} loaded ({len(_ck.get('preds',{}))} pred dates)")
            except Exception as _ce:
                print(f"  [warn] checkpoint load failed for fold {_fi}: {_ce}")

for split_idx, (train_start, train_end, test_start, test_end) in enumerate(WF_SPLITS):
    if split_idx in _completed_folds:
        print(f"  ⏭️  fold {split_idx+1}/{len(WF_SPLITS)} already checkpointed, skipping")
        continue
    print(f"\n  Split {split_idx+1}/{len(WF_SPLITS)}: train→{train_end}  test:{test_start}→{test_end}")
    # Track which dates + val entries are added by THIS fold (used for checkpoint save below)
    _fold_keys_before = set(ensemble_preds.keys())
    _fold_vpred_idx   = len(val_pred_acc)

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    test_df  = df[(df.index >= test_start)  & (df.index <= test_end)].copy()

    if len(train_df) < lookback + h + 50:
        print(f"    ⚠️  Not enough training data ({len(train_df)} rows), skipping"); continue
    if len(test_df) < lookback + 1:
        print(f"    ⚠️  Test window too short ({len(test_df)} < lookback {lookback}), skipping"); continue

    sw    = build_sample_weights(train_df, CRISES, neg_date=NEGATIVE_PRICE_DATE)
    n_val = max(int(len(train_df)*0.15), lookback+10)
    val_df = train_df.iloc[-n_val:]; tr_df = train_df.iloc[:-n_val]

    for seed in range(N_SEEDS):
        set_seed(seed*100)
        try:
            # Fit RobustScaler on TRAIN only, then reuse on val/test (no leakage).
            train_ds = WTIDataset(tr_df,   feat_cols, tgt_col, lookback, sw[:len(tr_df)])
            fit_scaler = train_ds.scaler
            val_ds   = WTIDataset(val_df,  feat_cols, tgt_col, lookback, scaler=fit_scaler)
            test_ds  = WTIDataset(test_df, feat_cols, tgt_col, lookback, scaler=fit_scaler)
        except Exception as e:
            print(f"    ⚠️  Dataset build failed: {e}"); continue

        if len(train_ds) < 10 or len(val_ds) < 5 or len(test_ds) < 1:
            print(f"    ⚠️  Insufficient samples (train={len(train_ds)},val={len(val_ds)},test={len(test_ds)})")
            continue

        model = build_model(HORIZON_KEY, len(feat_cols), HORIZONS, model_name=MODEL_NAME)
        model, _ = train_model(
            model, train_ds, val_ds, HORIZON_KEY, HORIZONS,
            seed=seed, verbose=(seed==0))

        # Predict on TEST window
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        pt_all, sig_all = [], []
        with torch.no_grad():
            for Xb, _ in test_loader:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(Xb.to(DEVICE))
                pt, sg = _decode_output(out, loss_type)
                pt_all.append(pt)
                if sg is not None: sig_all.append(sg)

        if not pt_all:
            del model; _free_memory(); continue

        preds_flat = np.concatenate(pt_all)
        sig_flat   = np.concatenate(sig_all) if sig_all else [None]*len(preds_flat)
        pred_dates = test_df.index[lookback: lookback+len(preds_flat)]
        for dt, lret, sg in zip(pred_dates, preds_flat, sig_flat):
            ensemble_preds[dt].append(float(lret))
            if sg is not None: ensemble_sigma[dt].append(float(sg))

        # ── C4: MC-Dropout epistemic uncertainty (Kendall & Gal 2017) ────────
        # Runs MC_DROPOUT_T stochastic forward passes with dropout ACTIVE (model.train())
        # and torch.no_grad() for efficiency.  The variance across T passes per date
        # is the epistemic (model) uncertainty; combined with the Student-t aleatoric σ
        # it gives total predictive uncertainty.
        # Default MC_DROPOUT_T=0 → block is entirely skipped (zero overhead for comparators).
        if MC_DROPOUT_T > 0 and len(preds_flat) > 0:
            model.train()   # activate dropout layers for stochastic sampling
            _mc_passes = []
            with torch.no_grad():
                for _mc_t in range(int(MC_DROPOUT_T)):
                    _batch_pts = []
                    for Xb, _ in test_loader:
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            _out = model(Xb.to(DEVICE))
                        _pt, _ = _decode_output(_out, loss_type)
                        _batch_pts.append(_pt)
                    if _batch_pts:
                        _mc_passes.append(np.concatenate(_batch_pts))
            model.eval()   # restore eval mode for subsequent (val/insample) passes
            if _mc_passes:
                _mc_stack = np.stack(_mc_passes, axis=0)   # (MC_DROPOUT_T, N_dates)
                _mc_var   = _mc_stack.var(axis=0)          # epistemic variance per test date
                for _i, _dt in enumerate(pred_dates):
                    if _i < len(_mc_var):
                        mc_epistemic_acc[_dt].append(float(_mc_var[_i]))
            del _mc_passes

        # Predict on VALIDATION window → α grid-search on price MAE + conformal CI calibration
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        v_pt_all, v_sg_all = [], []
        with torch.no_grad():
            for Xb, _ in val_loader:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(Xb.to(DEVICE))
                pt, sg = _decode_output(out, loss_type)
                v_pt_all.append(pt)
                if sg is not None: v_sg_all.append(sg)
        if v_pt_all:
            v_preds = np.concatenate(v_pt_all)
            val_pred_acc.append(v_preds)
            # Origin dates: val_df index starting at lookback (NaN targets only at end → consecutive)
            val_date_acc.append(val_df.index[lookback: lookback + len(v_preds)].values)
            if v_sg_all:
                val_sigma_acc.append(np.concatenate(v_sg_all))

        # Predict on the TRAINING window → in-sample (fitted) line for full-history plot.
        # NOTE: in-sample = fitted, optimistic; NOT a genuine out-of-sample forecast.
        # Only needed for the PatchTST headline (Figure 11); skipped for comparators
        # to avoid a costly extra inference pass across ~396 comparator trainings.
        try:
            insample_ds = (WTIDataset(train_df, feat_cols, tgt_col, lookback, scaler=fit_scaler)
                           if MODEL_NAME == "PatchTST" else None)
        except Exception:
            insample_ds = None
        if insample_ds is not None and len(insample_ds) >= 1:
            insample_loader = DataLoader(insample_ds, batch_size=BATCH_SIZE, shuffle=False)
            is_pt_all, is_sig_all = [], []
            with torch.no_grad():
                for Xb, _ in insample_loader:
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        out = model(Xb.to(DEVICE))
                    pt, sg = _decode_output(out, loss_type)
                    is_pt_all.append(pt)
                    if sg is not None: is_sig_all.append(sg)
            if is_pt_all:
                is_flat   = np.concatenate(is_pt_all)
                is_sig    = np.concatenate(is_sig_all) if is_sig_all else [None]*len(is_flat)
                is_dates  = train_df.index[lookback: lookback+len(is_flat)]
                for dt, lret, sg in zip(is_dates, is_flat, is_sig):
                    insample_preds[dt].append(float(lret))
                    if sg is not None: insample_sigma[dt].append(float(sg))
            del insample_loader, is_pt_all, is_sig_all
        if insample_ds is not None: del insample_ds

        # FREE GPU MEMORY — critical for long ensemble runs
        del model, train_ds, val_ds, test_ds, test_loader, val_loader, pt_all, sig_all
        del v_pt_all, v_sg_all
        _free_memory()

    # ── Save fold checkpoint ──────────────────────────────────────────────────
    # Pickle this fold's contribution so a reconnected session can merge it back
    # and skip training.  preds/sigma: only dates newly produced in this fold.
    # vpreds/vsigmas/vdates: the seed-by-seed val arrays this fold added to the
    # accumulators (needed to reproduce the α grid-search on reconnect).
    if WF_CHECKPOINT:
        _fold_new_keys = set(ensemble_preds.keys()) - _fold_keys_before
        _ck_data = {
            "preds":   {dt: list(ensemble_preds[dt]) for dt in _fold_new_keys},
            "sigma":   {dt: list(ensemble_sigma[dt]) for dt in _fold_new_keys
                        if dt in ensemble_sigma},
            "vpreds":  val_pred_acc[_fold_vpred_idx:],
            "vsigmas": (val_sigma_acc[_fold_vpred_idx:]
                        if len(val_sigma_acc) > _fold_vpred_idx else []),
            "vdates":  val_date_acc[_fold_vpred_idx:],
        }
        _fp = _ckpt_dir / f"{MODEL_NAME}_{HORIZON_KEY}_fold{split_idx:02d}.pkl"
        try:
            with open(_fp, "wb") as _f:
                pickle.dump(_ck_data, _f)
            print(f"  💾 [ckpt] fold {split_idx:02d} saved ({len(_fold_new_keys)} pred dates)")
        except Exception as _ce:
            print(f"  [warn] checkpoint save failed for fold {split_idx}: {_ce}")

# ── Naive-shrinkage α: grid-search on validation PRICE MAE ───
# Optimises the EXACT same objective (price MAE) the model is judged on.
# α=0 (pure naive = hold last price) is explicit in the grid, so 180d
# floors at ≈naive instead of amplifying anti-signal.
# Also computes split-conformal CI offsets on validation residuals
# so the 10–90% band achieves ≈80% empirical coverage.
conf_lo   = None   # conformal lower offset (log-return units)
conf_hi   = None   # conformal upper offset
use_sigma = False  # True → CI scaled by Student-t σ head; False → raw residuals
alpha_shrink = 0.0 if (loss_type != "mse") else 1.0   # safe default

if val_pred_acc and val_date_acc:
    vp = np.concatenate(val_pred_acc)
    vd = pd.DatetimeIndex(np.concatenate(val_date_acc))
    # Origin price (winsorized, consistent with test) and actual future price
    op = df["price"].reindex(vd).values
    fut_idx = vd + pd.tseries.offsets.BDay(h)
    af = wti_raw["price"].reindex(fut_idx).values
    ok = np.isfinite(vp) & np.isfinite(op) & np.isfinite(af)
    if ok.sum() > 20:
        vp_ok, op_ok, af_ok = vp[ok], op[ok], af[ok]
        # Grid search: price MAE across α ∈ [0, 1] (21 points)
        best_alpha, best_mae = 0.0, float("inf")
        for a in np.linspace(0.0, 1.0, 21):
            mae_a = float(np.mean(np.abs(op_ok * np.exp(a * vp_ok) - af_ok)))
            if mae_a < best_mae - 1e-9:
                best_mae, best_alpha = mae_a, a   # ties → prefer smaller α (parsimony)
        alpha_shrink = best_alpha

        # ── Split-conformal calibration (on validation, never test) ──────
        # Actual h-day log-return on validation (same direction as pred)
        vt_logret = np.log(af_ok / op_ok)
        vp_shrunk = best_alpha * vp_ok
        if loss_type == "studentt" and val_sigma_acc:
            vs_all = np.concatenate(val_sigma_acc)[ok]
            z = (vt_logret - vp_shrunk) / (vs_all + 1e-6)
            conf_lo   = float(np.nanpercentile(z, 10))
            conf_hi   = float(np.nanpercentile(z, 90))
            use_sigma = True
        else:
            r = vt_logret - vp_shrunk
            conf_lo   = float(np.nanpercentile(r, 10))
            conf_hi   = float(np.nanpercentile(r, 90))
            use_sigma = False
    else:
        print(f"    ⚠️  Too few finite val samples ({ok.sum()}) for α grid-search; defaulting to 0")

print(f"\n  Naive-shrinkage α ({HORIZON_KEY}) = {alpha_shrink:.3f}"
      f"   (1.0 = full model, 0.0 = pure naive)")
if conf_lo is not None:
    print(f"  Conformal CI offsets: [{conf_lo:.4f}, {conf_hi:.4f}]"
          f"   σ-scaled={use_sigma}")

# ── Aggregate ensemble ────────────────────────────────────────
# Uses split-conformal calibrated CI when available; falls back to
# ensemble-percentile spread if no conformal offsets were computed.
horizon_results = {}
for dt, preds_list in ensemble_preds.items():
    if len(preds_list) < 2: continue
    if dt not in df.index: continue
    arr          = np.array(preds_list) * alpha_shrink     # shrink toward naive
    last_price   = df.loc[dt, "price"]
    point_logret = float(np.median(arr))
    target_date  = dt + pd.tseries.offsets.BDay(h)
    actual_price = wti_raw.loc[target_date,"price"] if target_date in wti_raw.index else np.nan
    sig_list     = ensemble_sigma.get(dt, [])

    # Calibrated predictive interval via split-conformal offsets
    if conf_lo is not None:
        sig_pred = float(np.mean(sig_list)) if (sig_list and use_sigma) else 0.0
        if use_sigma and sig_pred > 0:
            lo_logret = point_logret + sig_pred * conf_lo
            hi_logret = point_logret + sig_pred * conf_hi
        else:
            lo_logret = point_logret + conf_lo
            hi_logret = point_logret + conf_hi
        lower_price = last_price * np.exp(lo_logret)
        upper_price = last_price * np.exp(hi_logret)
    else:
        # Fallback: ensemble percentile spread (no conformal calibration available)
        pred_prices = last_price * np.exp(arr)
        lower_price = float(np.percentile(pred_prices, 10))
        upper_price = float(np.percentile(pred_prices, 90))

    # ── C4: Epistemic + total uncertainty (MC-Dropout) ─────────
    # epistemic_sigma: model uncertainty (MC-Dropout variance across T passes)
    # total_sigma: sqrt(aleatoric² + epistemic²) — Kendall & Gal (2017) decomposition
    _mc_list       = mc_epistemic_acc.get(dt, [])
    _aleatoric_sig = float(np.mean(sig_list)) if sig_list else np.nan
    _epistemic_var = float(np.mean(_mc_list)) if _mc_list else np.nan
    _epistemic_sig = float(np.sqrt(_epistemic_var)) if np.isfinite(_epistemic_var) else np.nan
    if np.isfinite(_epistemic_sig) and np.isfinite(_aleatoric_sig):
        _total_sig = float(np.sqrt(_epistemic_sig**2 + _aleatoric_sig**2))
    elif np.isfinite(_epistemic_sig):
        _total_sig = _epistemic_sig
    elif np.isfinite(_aleatoric_sig):
        _total_sig = _aleatoric_sig
    else:
        _total_sig = np.nan

    horizon_results[dt] = {
        "pred":             last_price * np.exp(point_logret),
        "lower":            lower_price,
        "upper":            upper_price,
        "actual":           actual_price,
        "pred_logret":      point_logret,                        # for risk/EVT layer
        "sigma_logret":     _aleatoric_sig,                      # aleatoric σ (Student-t head)
        "epistemic_sigma":  _epistemic_sig,                      # C4: MC-Dropout epistemic σ
        "total_sigma":      _total_sig,                          # C4: total predictive σ
        "n_models":         len(preds_list),
    }

res_df = pd.DataFrame(horizon_results).T
res_df.index = pd.to_datetime(res_df.index)

if "actual" in res_df.columns:
    res_df = res_df.dropna(subset=["actual"])
else:
    print(f"  ⚠️  No predictions collected for {HORIZON_KEY} — check lookback vs window size")
    res_df = pd.DataFrame()

print(f"\n  {HORIZON_KEY} results: {len(res_df)} prediction dates")

# ── Evaluation ────────────────────────────────────────────────
eval_summary = {}
eval_summary[f"alpha_shrink_{HORIZON_KEY}"] = alpha_shrink
eval_summary[f"loss_{HORIZON_KEY}"]         = loss_type
eval_summary[f"conf_lo_{HORIZON_KEY}"]      = conf_lo
eval_summary[f"conf_hi_{HORIZON_KEY}"]      = conf_hi
eval_summary[f"use_sigma_{HORIZON_KEY}"]    = use_sigma
if not res_df.empty:
    y_true = res_df["actual"].astype(float)
    y_pred = res_df["pred"].astype(float)
    eval_summary[f"Overall_{HORIZON_KEY}"] = compute_metrics(y_true, y_pred, f"Overall {HORIZON_KEY}")

    def normal_mask(dates):
        m = np.ones(len(dates), dtype=bool)
        for s,e in CRISES.values():
            m &= ~((dates>=s)&(dates<=e))
        return m
    nm = normal_mask(res_df.index)
    if nm.sum() > 10:
        eval_summary[f"Normal_{HORIZON_KEY}"] = compute_metrics(y_true[nm], y_pred[nm], f"Normal {HORIZON_KEY}")

    for cname,(cs,ce) in CRISES.items():
        cm = (res_df.index>=cs)&(res_df.index<=ce)
        if cm.sum() > 5:
            eval_summary[f"{cname}_{HORIZON_KEY}"] = compute_metrics(y_true[cm], y_pred[cm], f"{cname} {HORIZON_KEY}")
        # Post-crisis recovery (12 months after crisis end)
        post_s = pd.Timestamp(ce) + pd.DateOffset(days=1)
        post_e = post_s + pd.DateOffset(months=12)
        pm = (res_df.index >= post_s) & (res_df.index <= post_e)
        if pm.sum() > 5:
            eval_summary[f"{cname}_Post_{HORIZON_KEY}"] = compute_metrics(
                y_true[pm], y_pred[pm], f"{cname}+12mo {HORIZON_KEY}")

    neg_dt = pd.Timestamp(NEGATIVE_PRICE_DATE)
    if neg_dt in res_df.index:
        row = res_df.loc[neg_dt]
        print(f"  April-20-2020: actual=${float(row['actual']):.2f}  pred=${float(row['pred']):.2f}"
              f"  CI=[{float(row['lower']):.2f},{float(row['upper']):.2f}]")

# ── In-sample (fitted) predictions across full training history ──
# Same shrinkage + log-return→price conversion as the out-of-sample path,
# but evaluated on the training windows so the plotted line spans ~1986→cutoff.
insample_rows = {}
for dt, preds_list in insample_preds.items():
    if len(preds_list) < 1 or dt not in df.index: continue
    arr          = np.array(preds_list) * alpha_shrink
    last_price   = df.loc[dt, "price"]
    pred_prices  = last_price * np.exp(arr)
    target_date  = dt + pd.tseries.offsets.BDay(h)
    actual_price = wti_raw.loc[target_date,"price"] if target_date in wti_raw.index else np.nan
    insample_rows[dt] = {
        "pred":     np.median(pred_prices),
        "lower":    np.percentile(pred_prices, 10),
        "upper":    np.percentile(pred_prices, 90),
        "actual":   actual_price,
        "n_models": len(preds_list),
    }
insample_df = pd.DataFrame(insample_rows).T
if not insample_df.empty:
    insample_df.index = pd.to_datetime(insample_df.index)
    insample_df = insample_df.sort_index()
# In-sample (fitted) line is only consumed by Figure 11 for the PatchTST headline.
if MODEL_NAME == "PatchTST":
    insample_df.to_csv(OUTPUT_DIR/f"predictions_insample_{HORIZON_KEY}.csv")
    print(f"  ✅ in-sample (fitted) predictions: {len(insample_df)} dates "
          f"→ predictions_insample_{HORIZON_KEY}.csv")

# ── Save ──────────────────────────────────────────────────────
res_df.to_csv(OUTPUT_DIR/f"predictions{MODEL_SUFFIX}_{HORIZON_KEY}.csv")
eval_summary["model"] = MODEL_NAME
with open(OUTPUT_DIR/f"eval{MODEL_SUFFIX}_{HORIZON_KEY}.pkl","wb") as f:
    pickle.dump(eval_summary, f)

print(f"\n✅ {MODEL_NAME} {HORIZON_KEY} COMPLETE — "
      f"predictions{MODEL_SUFFIX}_{HORIZON_KEY}.csv + eval{MODEL_SUFFIX}_{HORIZON_KEY}.pkl saved")
