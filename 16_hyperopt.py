# ============================================================
# FILE 16 — Optuna Hyperparameter Sweep for PatchTST  (ADDENDUM 14 B1)
#
# Sweeps PatchTST hyperparameters on the FIRST walk-forward split of each
# horizon (proxy for full-WF MAE). Best params per horizon are persisted to
# outputs/optuna_best_hparams.pkl. _forecast_template.py auto-loads them on
# next run and overrides HORIZONS defaults.
#
# Compute: 30 trials × 3 horizons × ~3 min/trial ≈ 4.5 GPU-hours on T4.
# Skip on time-constrained runs — pipeline falls back to hand-picked defaults.
#
# Run AFTER 00, 01a. Run BEFORE 02a/03/04/05 to benefit from tuned params.
# Usage:  exec(open('16_hyperopt.py', encoding="utf-8").read())
# ============================================================

import pickle, subprocess, sys, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch
except Exception as _te:
    raise RuntimeError(f"16_hyperopt requires torch: {_te}. Run _env.py first.") from _te
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=False)
    import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load shared modules (PatchTST, WTIDataset, train_model, build_model, MODEL_REGISTRY)
exec(open('02_model_shared.py', encoding="utf-8").read())

# Load data
with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df      = pickle.load(f)
with open(OUTPUT_DIR/"wti_raw.pkl",    "rb") as f: wti_raw = pickle.load(f)
with open(OUTPUT_DIR/"feat_cols.pkl",  "rb") as f: FEAT    = pickle.load(f)
print(f"✅ Data loaded: df {df.shape}")

# ── Config ─────────────────────────────────────────────────
SWEEP_HORIZONS = ["7d", "90d", "180d"]       # 1d skipped (paper files cover that segment)
N_TRIALS       = int(globals().get("OPTUNA_N_TRIALS", 30))
SWEEP_EPOCHS   = int(globals().get("OPTUNA_EPOCHS",   20))   # short proxy training
TARGET_COLS    = {"7d":"target_7d","90d":"target_90d","180d":"target_180d"}

# ── Walk-forward helper (mirrors template's WF_CONFIG[hk]) ─
WF_CONFIG = {
    "7d":  {"start_test": "2007-01-01", "freq_months": 12},
    "90d": {"start_test": "2007-01-01", "freq_months": 24},
    "180d":{"start_test": "2007-01-01", "freq_months": 24},
}

def get_first_split(df, hk):
    """Use the FIRST walk-forward fold's (train, val) for the Optuna proxy."""
    cfg = WF_CONFIG[hk]
    test_start = pd.Timestamp(cfg["start_test"])
    train_end  = test_start - pd.DateOffset(days=1)
    val_start  = test_start - pd.DateOffset(months=cfg["freq_months"])
    train_df = df.loc[df.index[0]:val_start]
    val_df   = df.loc[val_start:train_end]
    return train_df, val_df

# Mapping horizon → (h, default loss, default lookback range)
LOSS_PER_HK   = {"7d":"mse", "90d":"studentt", "180d":"studentt"}
HORIZON_H     = {"7d":7, "90d":90, "180d":180}


def objective(trial, hk):
    """Optuna objective: validation log-return MAE."""
    h        = HORIZON_H[hk]
    loss     = LOSS_PER_HK[hk]
    feat_cols = FEAT[hk]
    tgt_col   = TARGET_COLS[hk]

    # Hyperparameter search space
    d_model    = trial.suggest_categorical("d_model",  [64, 128, 256])
    valid_heads = [n for n in [4, 8, 16] if d_model % n == 0]
    n_heads    = trial.suggest_categorical("n_heads",  valid_heads)
    n_layers   = trial.suggest_int("n_layers", 2, 4)
    patch_len  = trial.suggest_categorical("patch", [8, 16, 32])
    dropout    = trial.suggest_float("dropout", 0.1, 0.35, step=0.05)
    lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    lookback   = trial.suggest_categorical("lookback", [96, 192, 336, 480])
    if lookback <= patch_len: raise optuna.TrialPruned()

    # Compose horizons dict for build_model — keep all other keys defaulted
    horizons_dict = {
        hk: {"h": h, "lookback": lookback, "patch": patch_len, "layers": n_layers,
             "d_model": d_model, "loss": loss, "n_quantiles": 1,
             "dropout": dropout, "weight_decay": weight_decay,
             "n_heads": n_heads}
    }
    # train_model reads MAX_EPOCHS at module scope — temporarily shrink
    global MAX_EPOCHS
    _orig_epochs = MAX_EPOCHS
    MAX_EPOCHS = SWEEP_EPOCHS

    try:
        train_df, val_df = get_first_split(df, hk)
        train_ds = WTIDataset(train_df, feat_cols, tgt_col, lookback)
        val_ds   = WTIDataset(val_df,   feat_cols, tgt_col, lookback, scaler=train_ds.scaler)
        if len(train_ds) < 50 or len(val_ds) < 20:
            raise optuna.TrialPruned()

        model = build_model(hk, len(feat_cols), horizons_dict, model_name="PatchTST")
        # train_model expects n_heads in horizons_dict; build_model only reads keys it needs.
        # But PatchTST __init__ doesn't take n_heads from dict — patch via reflection:
        # Safer: pass n_heads explicitly by rebuilding here.
        from torch import nn
        # PatchTST signature: (n_features, patch_len, lookback, d_model, n_heads, n_layers, d_ff, n_outputs, dropout)
        n_out = 3 if loss == "studentt" else 1
        model = PatchTST(
            n_features=len(feat_cols), patch_len=patch_len, lookback=lookback,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_model*2,
            n_outputs=n_out, dropout=dropout,
        ).to(DEVICE)
        model, hist = train_model(model, train_ds, val_ds, hk, horizons_dict,
                                  seed=42, verbose=False)

        # Compute validation MAE on log-returns (cheap proxy)
        model.eval()
        from torch.utils.data import DataLoader
        dl = DataLoader(val_ds, batch_size=64, shuffle=False)
        preds_acc, true_acc = [], []
        with torch.no_grad():
            for X, y in dl:
                X = X.to(DEVICE); y = y.to(DEVICE)
                out = model(X)
                if loss == "studentt":
                    mu = out[..., 0]
                else:
                    mu = out.squeeze(-1) if out.ndim > 1 else out
                preds_acc.append(mu.detach().cpu().numpy().ravel())
                true_acc.append(y.detach().cpu().numpy().ravel())
        if not preds_acc: raise optuna.TrialPruned()
        preds = np.concatenate(preds_acc); trues = np.concatenate(true_acc)
        mae = float(mean_absolute_error(trues, preds))

        del model, train_ds, val_ds
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return mae

    finally:
        MAX_EPOCHS = _orig_epochs


# ── Run sweep per horizon ──────────────────────────────────
best_params_per_hk = {}
for hk in SWEEP_HORIZONS:
    print("\n" + "=" * 60)
    print(f"  Optuna sweep — horizon {hk}  ({N_TRIALS} trials × {SWEEP_EPOCHS} epochs)")
    print("=" * 60)
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42),
                                 pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    try:
        study.optimize(lambda t: objective(t, hk), n_trials=N_TRIALS, show_progress_bar=False)
    except Exception as e:
        print(f"  ⚠️  Optuna sweep aborted for {hk}: {e}")
        continue
    best = study.best_params
    print(f"\n  ✅ Best {hk}:  MAE={study.best_value:.5f}")
    print(f"     params: {best}")
    best_params_per_hk[hk] = best

# ── Save ────────────────────────────────────────────────────
out = {}
for hk, p in best_params_per_hk.items():
    # Translate Optuna keys → HORIZONS dict keys consumed by template/02_model_shared
    out[hk] = {
        "lookback":     p["lookback"],
        "patch":        p["patch"],
        "layers":       p["n_layers"],
        "d_model":      p["d_model"],
        "n_heads":      p["n_heads"],
        "dropout":      p["dropout"],
        "weight_decay": p["weight_decay"],
        "lr_override":  p["lr"],
    }

with open(OUTPUT_DIR/"optuna_best_hparams.pkl", "wb") as f:
    pickle.dump(out, f)
print("\n✅ FILE 16 COMPLETE — best hyperparameters saved to outputs/optuna_best_hparams.pkl")
print(f"   Re-run 02a/03/04/05 to use tuned hparams.")
