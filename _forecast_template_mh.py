# ============================================================
# MULTI-HORIZON FORECAST TEMPLATE
# Trains ONE MultiHorizonPatchTST model that simultaneously predicts
# 7d, 90d, and 180d log-returns using a shared PatchTST backbone
# and three separate output heads.
#
# KEY GOAL: minimise the performance degradation as horizon increases.
# Sharing the backbone forces it to learn representations that are
# predictive at ALL timescales, rather than overfitting to short-term
# noise. The 7d head benefits from macro features; the 180d head
# benefits from short-term calibration — both improve.
#
# Walk-forward: uses 7d splits (annual, ~18 from 2007) so that GFC,
# OPEC War, and COVID are all genuinely out-of-sample.
#
# Loss: weighted sum across the three horizon heads.
#   total = λ_7d*MSE_7d + λ_90d*StudentT_90d + λ_180d*StudentT_180d
#   Default λ = [0.5, 0.3, 0.2] (upweight 7d to keep short-term sharp)
#
# Outputs (per horizon):
#   predictions_MultiPatchTST_{hk}.csv
#   eval_MultiPatchTST_{hk}.pkl
#   predictions_insample_MultiPatchTST_{hk}.csv
#
# DO NOT run directly. Driver: 03_forecast_multi.py
# ============================================================

import pickle, warnings, gc
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
except Exception as _te:
    raise RuntimeError(f"_forecast_template_mh requires torch: {_te}. Run _env.py first.") from _te
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)

# ── Horizon configs (same as single-horizon template) ─────────
HORIZONS = {
    "7d":  {"h": 7,   "lookback": 96,  "patch": 8,  "layers": 3, "d_model": 128,
            "loss": "mse",      "n_quantiles": 1, "dropout": 0.20, "weight_decay": 1e-4},
    "90d": {"h": 90,  "lookback": 336, "patch": 16, "layers": 3, "d_model": 128,
            "loss": "studentt", "n_quantiles": 1, "dropout": 0.25, "weight_decay": 1e-3},
    "180d":{"h": 180, "lookback": 360, "patch": 16, "layers": 3, "d_model": 128,
            "loss": "studentt", "n_quantiles": 1, "dropout": 0.25, "weight_decay": 1e-3},
}

# The shared backbone uses the 180d config (longest lookback / most capacity)
BACKBONE_HK = "180d"

# Walk-forward: annual splits from 2007 (same as 7d single-horizon template)
WF_CONFIG = {
    "7d":  {"start_test": "2007-01-01", "freq_months": 12},
}

CRISES = {
    "GFC":      ("2008-09-01","2009-06-30"),
    "OPEC_War": ("2014-06-01","2016-03-31"),
    "COVID":    ("2020-02-01","2020-09-30"),
}
NEGATIVE_PRICE_DATE = "2020-04-20"

# Global knobs (overridable by the driver)
N_SEEDS        = globals().get("N_SEEDS_OVERRIDE",  3)
LAMBDA_WEIGHTS = globals().get("LAMBDA_WEIGHTS",    [0.5, 0.3, 0.2])   # 7d / 90d / 180d
MAX_EPOCHS_MH  = globals().get("MAX_EPOCHS_MH",     100)
MODEL_NAME     = "MultiPatchTST"
MODEL_SUFFIX   = "_MultiPatchTST"

# ── Load shared model code ────────────────────────────────────
exec(open('02_model_shared.py', encoding="utf-8").read())

# ── Load data ─────────────────────────────────────────────────
with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df      = pickle.load(f)
with open(OUTPUT_DIR/"wti_raw.pkl",    "rb") as f: wti_raw = pickle.load(f)
with open(OUTPUT_DIR/"feat_cols.pkl",  "rb") as f: FEAT    = pickle.load(f)
print(f"✅ Data loaded: df {df.shape}")

TARGET_COLS = {"7d":"target_7d", "90d":"target_90d", "180d":"target_180d"}
ALL_HKS     = ["7d", "90d", "180d"]

# Use the feature set for 7d as the input features (BASE_FEATURES).
# 7d features are the most numerous / broadest set; the backbone will
# learn which features matter at each horizon via the per-head projection.
FEAT_COLS = FEAT.get("7d", FEAT[list(FEAT.keys())[0]])
N_FEATURES = len(FEAT_COLS)

print(f"\n  MultiHorizonPatchTST — {N_FEATURES} features")
print(f"  Backbone config:  {BACKBONE_HK} ({HORIZONS[BACKBONE_HK]['lookback']}d lookback)")
print(f"  Lambda weights:   {LAMBDA_WEIGHTS}  [7d / 90d / 180d]")
print(f"  Seeds:            {N_SEEDS}")

# Backbone hyperparams from 180d config
LOOKBACK  = HORIZONS[BACKBONE_HK]["lookback"]
PATCH_LEN = HORIZONS[BACKBONE_HK]["patch"]

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.sum() else np.nan
    print(f"  {label:30s}  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")
    return {"label":label,"MAE":mae,"RMSE":rmse,"MAPE":mape}

def get_wf_splits(df):
    cfg = WF_CONFIG["7d"]
    starts = pd.date_range(cfg["start_test"], df.index[-1], freq=f"{cfg['freq_months']}MS")
    return [(df.index[0].strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%d"),
             ts.strftime("%Y-%m-%d"),
             min(ts+pd.DateOffset(months=cfg["freq_months"]), df.index[-1]).strftime("%Y-%m-%d"))
            for ts in starts]

WF_SPLITS = get_wf_splits(df)
print(f"\nWalk-forward splits (MultiHorizon): {len(WF_SPLITS)}")

# ── Loss functions ─────────────────────────────────────────────
_mse_fn     = MSESqueeze()
_studentt_7d = _mse_fn        # not used but kept for clarity
_studentt_fn = StudentTNLL()

LOSS_FNS = {"7d": _mse_fn, "90d": _studentt_fn, "180d": _studentt_fn}
LAMBDAS  = {"7d": LAMBDA_WEIGHTS[0], "90d": LAMBDA_WEIGHTS[1], "180d": LAMBDA_WEIGHTS[2]}

def _decode_mh(out_dict):
    """Decode MultiHorizonPatchTST output → (point_logret, sigma) per horizon."""
    result = {}
    for hk, out in out_dict.items():
        out = out.float().cpu()
        if hk == "7d":
            result[hk] = (np.atleast_1d(out.squeeze(-1).numpy()), None)
        else:  # studentt: 3 outputs
            mu    = out[:, 0].numpy()
            sigma = (torch.nn.functional.softplus(out[:, 1]) + 1e-6).numpy()
            result[hk] = (np.atleast_1d(mu), np.atleast_1d(sigma))
    return result

# ── Accumulators (per horizon) ─────────────────────────────────
ensemble_preds = {hk: defaultdict(list) for hk in ALL_HKS}
ensemble_sigma = {hk: defaultdict(list) for hk in ALL_HKS}
insample_preds = {hk: defaultdict(list) for hk in ALL_HKS}
insample_sigma = {hk: defaultdict(list) for hk in ALL_HKS}
val_pred_acc   = {hk: [] for hk in ALL_HKS}
val_sigma_acc  = {hk: [] for hk in ALL_HKS}
val_date_acc   = {hk: [] for hk in ALL_HKS}

# ── Main walk-forward loop ─────────────────────────────────────
print(f"\n{'='*60}")
print(f"  MULTI-HORIZON JOINT TRAINING  |  seeds: {N_SEEDS}")
print(f"{'='*60}")

for split_idx, (train_start, train_end, test_start, test_end) in enumerate(WF_SPLITS):
    print(f"\n  Split {split_idx+1}/{len(WF_SPLITS)}: train→{train_end}  test:{test_start}→{test_end}")

    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    test_df  = df[(df.index >= test_start)  & (df.index <= test_end)].copy()

    if len(train_df) < LOOKBACK + 180 + 50:
        print(f"    ⚠️  Insufficient training data ({len(train_df)}), skipping"); continue
    if len(test_df)  < LOOKBACK + 1:
        print(f"    ⚠️  Test window too short ({len(test_df)}), skipping"); continue

    sw    = build_sample_weights(train_df, CRISES, neg_date=NEGATIVE_PRICE_DATE)
    n_val = max(int(len(train_df)*0.15), LOOKBACK + 10)
    val_df = train_df.iloc[-n_val:]; tr_df = train_df.iloc[:-n_val]

    for seed in range(N_SEEDS):
        set_seed(seed * 100)
        try:
            train_ds = WTIDatasetMH(tr_df,   FEAT_COLS, TARGET_COLS, LOOKBACK, sw[:len(tr_df)])
            fit_scaler = train_ds.scaler
            val_ds   = WTIDatasetMH(val_df,  FEAT_COLS, TARGET_COLS, LOOKBACK, scaler=fit_scaler)
            test_ds  = WTIDatasetMH(test_df, FEAT_COLS, TARGET_COLS, LOOKBACK, scaler=fit_scaler)
        except Exception as e:
            print(f"    ⚠️  Dataset build failed: {e}"); continue

        if len(train_ds) < 10 or len(val_ds) < 5 or len(test_ds) < 1:
            print(f"    ⚠️  Insufficient samples"); continue

        # Build the joint model
        model = build_multi_model(N_FEATURES, HORIZONS)

        # ── Training loop with multi-horizon weighted loss ──────
        w          = torch.from_numpy(train_ds.weights)
        sampler    = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        tr_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
        optimizer  = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=HORIZONS["180d"]["weight_decay"])
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS_MH, eta_min=LR*0.1)
        amp_scaler = torch.cuda.amp.GradScaler(enabled=IS_CUDA)
        best_val, best_state, patience_cnt = float("inf"), None, 0

        for epoch in range(1, MAX_EPOCHS_MH + 1):
            # ── Train epoch ──────────────────────────────────────
            model.train(); train_loss = 0.0
            for Xb, yb in tr_loader:
                Xb = Xb.to(DEVICE)
                yb = {hk: v.to(DEVICE) for hk, v in yb.items()}
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=IS_CUDA):
                    out = model(Xb)
                    loss = torch.tensor(0.0, device=DEVICE)
                    for hk in ALL_HKS:
                        valid = torch.isfinite(yb[hk])
                        if not valid.any(): continue
                        lf = LOSS_FNS[hk]
                        loss = loss + LAMBDAS[hk] * lf(out[hk][valid], yb[hk][valid])
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer); amp_scaler.update()
                train_loss += loss.item() * len(Xb)
            scheduler.step()

            # ── Validation ───────────────────────────────────────
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(DEVICE)
                    yb = {hk: v.to(DEVICE) for hk, v in yb.items()}
                    with torch.cuda.amp.autocast(enabled=IS_CUDA):
                        out = model(Xb)
                        v_loss = torch.tensor(0.0, device=DEVICE)
                        for hk in ALL_HKS:
                            valid = torch.isfinite(yb[hk])
                            if not valid.any(): continue
                            v_loss = v_loss + LAMBDAS[hk] * LOSS_FNS[hk](
                                out[hk][valid], yb[hk][valid])
                    val_loss += v_loss.item() * len(Xb)
            val_loss /= max(len(val_ds), 1)

            if val_loss < best_val:
                from copy import deepcopy
                best_val = val_loss; best_state = deepcopy(model.state_dict()); patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    if seed == 0:
                        print(f"    Early stop epoch {epoch} (best val={best_val:.6f})")
                    break
            if seed == 0 and epoch % 10 == 0:
                print(f"    seed={seed} epoch {epoch:3d}  val={val_loss:.6f}")

        model.load_state_dict(best_state)
        model.eval()

        # ── TEST pass: collect per-horizon predictions ──────────
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        batch_preds = {hk: [] for hk in ALL_HKS}
        with torch.no_grad():
            for Xb, _ in test_loader:
                with torch.cuda.amp.autocast(enabled=IS_CUDA):
                    out = model(Xb.to(DEVICE))
                decoded = _decode_mh(out)
                for hk in ALL_HKS:
                    batch_preds[hk].append(decoded[hk])

        if not batch_preds["7d"]:
            del model; _free_memory(); continue

        for hk in ALL_HKS:
            pts  = np.concatenate([p[0] for p in batch_preds[hk]])
            sigs = [p[1] for p in batch_preds[hk] if p[1] is not None]
            sigs_flat = np.concatenate(sigs) if sigs else [None]*len(pts)
            pred_dates = test_df.index[LOOKBACK: LOOKBACK+len(pts)]
            for dt, lret, sg in zip(pred_dates, pts, sigs_flat):
                ensemble_preds[hk][dt].append(float(lret))
                if sg is not None: ensemble_sigma[hk][dt].append(float(sg))

        # ── VAL pass: calibration (α + conformal CI) ────────────
        v_batch_preds = {hk: [] for hk in ALL_HKS}
        with torch.no_grad():
            for Xb, _ in val_loader:
                with torch.cuda.amp.autocast(enabled=IS_CUDA):
                    out = model(Xb.to(DEVICE))
                decoded = _decode_mh(out)
                for hk in ALL_HKS:
                    v_batch_preds[hk].append(decoded[hk])

        for hk in ALL_HKS:
            if v_batch_preds[hk]:
                v_pts  = np.concatenate([p[0] for p in v_batch_preds[hk]])
                v_sigs = [p[1] for p in v_batch_preds[hk] if p[1] is not None]
                val_pred_acc[hk].append(v_pts)
                val_date_acc[hk].append(val_df.index[LOOKBACK: LOOKBACK+len(v_pts)].values)
                if v_sigs: val_sigma_acc[hk].append(np.concatenate(v_sigs))

        # ── IN-SAMPLE pass (for full-history plot) ───────────────
        try:
            insample_ds = WTIDatasetMH(train_df, FEAT_COLS, TARGET_COLS, LOOKBACK, scaler=fit_scaler)
        except Exception:
            insample_ds = None
        if insample_ds and len(insample_ds) >= 1:
            is_loader = DataLoader(insample_ds, batch_size=BATCH_SIZE, shuffle=False)
            is_preds  = {hk: [] for hk in ALL_HKS}
            with torch.no_grad():
                for Xb, _ in is_loader:
                    with torch.cuda.amp.autocast(enabled=IS_CUDA):
                        out = model(Xb.to(DEVICE))
                    decoded = _decode_mh(out)
                    for hk in ALL_HKS:
                        is_preds[hk].append(decoded[hk])
            for hk in ALL_HKS:
                if is_preds[hk]:
                    is_pts  = np.concatenate([p[0] for p in is_preds[hk]])
                    is_sigs = [p[1] for p in is_preds[hk] if p[1] is not None]
                    is_sigs_flat = np.concatenate(is_sigs) if is_sigs else [None]*len(is_pts)
                    is_dates = train_df.index[LOOKBACK: LOOKBACK+len(is_pts)]
                    for dt, lret, sg in zip(is_dates, is_pts, is_sigs_flat):
                        insample_preds[hk][dt].append(float(lret))
                        if sg is not None: insample_sigma[hk][dt].append(float(sg))
            del is_loader, is_preds, insample_ds

        del model, train_ds, val_ds, test_ds, tr_loader, val_loader, test_loader
        _free_memory()

# ── Per-horizon α + conformal CI + aggregation + save ─────────
for hk in ALL_HKS:
    h         = HORIZONS[hk]["h"]
    loss_type = HORIZONS[hk]["loss"]
    print(f"\n{'─'*50}")
    print(f"  Post-processing {hk} (h={h})…")

    # ── α grid-search on validation price MAE ────────────────
    alpha_shrink = 0.0 if (loss_type != "mse") else 1.0
    conf_lo = conf_hi = None; use_sigma = False

    if val_pred_acc[hk] and val_date_acc[hk]:
        vp = np.concatenate(val_pred_acc[hk])
        vd = pd.DatetimeIndex(np.concatenate(val_date_acc[hk]))
        op = df["price"].reindex(vd).values
        fut_idx = vd + pd.tseries.offsets.BDay(h)
        af = wti_raw["price"].reindex(fut_idx).values
        ok = np.isfinite(vp) & np.isfinite(op) & np.isfinite(af)
        if ok.sum() > 20:
            vp_ok, op_ok, af_ok = vp[ok], op[ok], af[ok]
            best_alpha, best_mae = 0.0, float("inf")
            for a in np.linspace(0.0, 1.0, 21):
                mae_a = float(np.mean(np.abs(op_ok * np.exp(a*vp_ok) - af_ok)))
                if mae_a < best_mae - 1e-9:
                    best_mae, best_alpha = mae_a, a
            alpha_shrink = best_alpha

            vt_logret = np.log(af_ok / op_ok)
            vp_shrunk = best_alpha * vp_ok
            if loss_type == "studentt" and val_sigma_acc[hk]:
                vs = np.concatenate(val_sigma_acc[hk])[ok]
                z = (vt_logret - vp_shrunk) / (vs + 1e-6)
                conf_lo, conf_hi = float(np.nanpercentile(z, 10)), float(np.nanpercentile(z, 90))
                use_sigma = True
            else:
                r = vt_logret - vp_shrunk
                conf_lo, conf_hi = float(np.nanpercentile(r, 10)), float(np.nanpercentile(r, 90))
        else:
            print(f"    ⚠️  Too few valid val samples ({ok.sum()}) for {hk} α grid-search")

    print(f"  α_shrink ({hk}) = {alpha_shrink:.3f}  "
          f"CI=[{conf_lo:.4f}, {conf_hi:.4f}]  σ-scaled={use_sigma}"
          if conf_lo is not None else
          f"  α_shrink ({hk}) = {alpha_shrink:.3f}  CI=ensemble-fallback")

    # ── Aggregate ensemble → predictions CSV ──────────────────
    horizon_results = {}
    for dt, preds_list in ensemble_preds[hk].items():
        if len(preds_list) < 1 or dt not in df.index: continue
        arr          = np.array(preds_list) * alpha_shrink
        last_price   = df.loc[dt, "price"]
        point_logret = float(np.median(arr))
        target_date  = dt + pd.tseries.offsets.BDay(h)
        actual_price = wti_raw.loc[target_date,"price"] if target_date in wti_raw.index else np.nan
        sig_list     = ensemble_sigma[hk].get(dt, [])

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
            pred_prices = last_price * np.exp(arr)
            lower_price = float(np.percentile(pred_prices, 10))
            upper_price = float(np.percentile(pred_prices, 90))

        horizon_results[dt] = {
            "pred":         last_price * np.exp(point_logret),
            "lower":        lower_price,
            "upper":        upper_price,
            "actual":       actual_price,
            "pred_logret":  point_logret,
            "sigma_logret": float(np.mean(sig_list)) if sig_list else np.nan,
            "n_models":     len(preds_list),
        }

    res_df = pd.DataFrame(horizon_results).T
    if not res_df.empty:
        res_df.index = pd.to_datetime(res_df.index)
        res_df = res_df.dropna(subset=["actual"])

    print(f"  {hk} results: {len(res_df)} prediction dates")

    # ── Evaluation ───────────────────────────────────────────
    eval_summary = {
        f"alpha_shrink_{hk}": alpha_shrink,
        f"loss_{hk}":         loss_type,
        f"conf_lo_{hk}":      conf_lo,
        f"conf_hi_{hk}":      conf_hi,
        f"use_sigma_{hk}":    use_sigma,
        "model":              MODEL_NAME,
    }
    if not res_df.empty:
        y_true = res_df["actual"].astype(float)
        y_pred = res_df["pred"].astype(float)
        eval_summary[f"Overall_{hk}"] = compute_metrics(y_true.values, y_pred.values, f"Overall {hk}")

        def normal_mask(dates):
            m = np.ones(len(dates), dtype=bool)
            for s,e in CRISES.values():
                m &= ~((dates>=s)&(dates<=e))
            return m
        nm = normal_mask(res_df.index)
        if nm.sum() > 10:
            eval_summary[f"Normal_{hk}"] = compute_metrics(
                y_true.values[nm], y_pred.values[nm], f"Normal {hk}")

        for cname, (cs, ce) in CRISES.items():
            cm = (res_df.index>=cs) & (res_df.index<=ce)
            if cm.sum() > 5:
                eval_summary[f"{cname}_{hk}"] = compute_metrics(
                    y_true.values[cm], y_pred.values[cm], f"{cname} {hk}")

    # ── In-sample CSV ────────────────────────────────────────
    insample_rows = {}
    for dt, preds_list in insample_preds[hk].items():
        if len(preds_list) < 1 or dt not in df.index: continue
        arr         = np.array(preds_list) * alpha_shrink
        last_price  = df.loc[dt, "price"]
        pred_prices = last_price * np.exp(arr)
        target_date = dt + pd.tseries.offsets.BDay(h)
        insample_rows[dt] = {
            "pred":   np.median(pred_prices),
            "lower":  np.percentile(pred_prices, 10),
            "upper":  np.percentile(pred_prices, 90),
            "actual": wti_raw.loc[target_date,"price"] if target_date in wti_raw.index else np.nan,
        }
    insample_df = pd.DataFrame(insample_rows).T
    if not insample_df.empty:
        insample_df.index = pd.to_datetime(insample_df.index)
        insample_df.sort_index().to_csv(
            OUTPUT_DIR/f"predictions_insample_MultiPatchTST_{hk}.csv")
        print(f"  ✅ in-sample ({hk}): {len(insample_df)} dates saved")

    # ── Save predictions + eval ──────────────────────────────
    if not res_df.empty:
        res_df.to_csv(OUTPUT_DIR/f"predictions_MultiPatchTST_{hk}.csv")
    with open(OUTPUT_DIR/f"eval_MultiPatchTST_{hk}.pkl","wb") as f:
        pickle.dump(eval_summary, f)
    print(f"  ✅ MultiPatchTST {hk} COMPLETE — saved")

print(f"\n{'='*60}")
print("  MULTI-HORIZON TEMPLATE COMPLETE")
print(f"  Skill scores (1 - MAE_model / MAE_naive) computed in 06_analysis_plots.py")
print(f"{'='*60}")
