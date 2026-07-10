# ============================================================
# DDP_v2/T_thesis_leaky_repro.py
# Rigorous reproduction of the IIT-Bombay thesis (Mahajan 2024) protocol:
#   • Whole-series MinMaxScaler [0,1] on Close price (LEAKY — future included)
#   • Window size 10, 1-step-ahead next-day close
#   • 75/25 chronological train/test split
#   • 50 epochs, Adam, MSE loss
#   • Reports NORMALIZED metrics (as in thesis) + $-scale MAE
#
# Purpose: verify we can reproduce N-BEATS MAE≈0.0092 (normalised),
# demonstrating the gap vs our honest walk-forward is protocol/scale, not model.
# Also runs ILSTM, ILSTMCnn, ESN, LSTNet on the same leaky setup.
# Run from inside DDP_v2/.
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as _te:
    raise RuntimeError(f"T_thesis_leaky_repro requires torch: {_te}. Run _env.py first.") from _te
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))

if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())
if "ILSTM" not in globals():
    exec(open("_thesis_models.py", encoding="utf-8").read())

# ── config ────────────────────────────────────────────────────
WINDOW   = 10          # input lookback (thesis default)
EPOCHS   = 50
BATCH    = 32
LR       = 1e-3
SPLIT    = 0.75        # 75/25 train/test
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_TO_RUN = globals().get("THESIS_MODELS", ["NBEATSLite", "ILSTMCnn", "ESN", "LSTNet", "ILSTM"])

print(f"T_thesis_leaky_repro: protocol = whole-series MinMax + {int(SPLIT*100)}/{int((1-SPLIT)*100)} split (LEAKY)")
print(f"  window={WINDOW}  epochs={EPOCHS}  device={DEVICE}")

# ── load data ─────────────────────────────────────────────────
with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
    df = pickle.load(f)
# Thesis used raw Close price (not log-price)
prices = np.exp(df["log_price"].values).astype(np.float32)  # back to $

# WHOLE-SERIES MinMax (leaky — thesis does this)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_norm = scaler.fit_transform(prices.reshape(-1, 1)).ravel().astype(np.float32)

# ── dataset ───────────────────────────────────────────────────
class _ThesisDataset(Dataset):
    def __init__(self, series, window):
        self.X = np.stack([series[i:i+window] for i in range(len(series)-window)]).reshape(-1, window, 1)
        self.y = series[window:].reshape(-1, 1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

n_total = len(prices_norm) - WINDOW
n_train = int(n_total * SPLIT)
train_ds = _ThesisDataset(prices_norm[:n_train + WINDOW], WINDOW)
test_ds  = _ThesisDataset(prices_norm[n_train:], WINDOW)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

# ── train + evaluate one model ────────────────────────────────
def run_thesis_model(arch_name):
    from copy import deepcopy
    Cls = MODEL_REGISTRY.get(arch_name)
    if Cls is None:
        print(f"  ⚠️  {arch_name} not in MODEL_REGISTRY")
        return None
    model = Cls(n_features=1, lookback=WINDOW, d_model=128, n_layers=2,
                d_ff=256, n_outputs=1, dropout=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss, best_state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(X).squeeze(-1)
            loss = loss_fn(out, y.squeeze(-1))
            loss.backward(); opt.step()
        # track train loss for early-stop proxy
        with torch.no_grad():
            model.eval()
            ep_loss = loss_fn(model(train_ds.X[:256].to(DEVICE)),
                              train_ds.y[:256].to(DEVICE).squeeze(-1)).item() if len(train_ds) >= 256 else float("inf")
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()

    # Predict test set
    with torch.no_grad():
        X_test = torch.from_numpy(test_ds.X).to(DEVICE)
        preds_norm = model(X_test).squeeze(-1).cpu().numpy()
    actuals_norm = test_ds.y.ravel()

    # Metrics in NORMALIZED space (as thesis reports)
    mse_norm = float(mean_squared_error(actuals_norm, preds_norm))
    mae_norm = float(mean_absolute_error(actuals_norm, preds_norm))

    # Inverse-transform to $ scale
    preds_dollar   = scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()
    actuals_dollar = scaler.inverse_transform(actuals_norm.reshape(-1, 1)).ravel()
    mae_dollar = float(mean_absolute_error(actuals_dollar, preds_dollar))
    rmse_dollar = float(np.sqrt(mean_squared_error(actuals_dollar, preds_dollar)))
    mask = actuals_dollar > 1
    mape = float(np.mean(np.abs((actuals_dollar[mask] - preds_dollar[mask]) / actuals_dollar[mask])) * 100) if mask.any() else float("nan")

    print(f"  ✅ {arch_name:14s} | NORMALIZED  MSE={mse_norm:.6f}  MAE={mae_norm:.6f}"
          f"  | $-scale  MAE={mae_dollar:.3f}  RMSE={rmse_dollar:.3f}  MAPE={mape:.2f}%")

    # Save predictions CSV
    dates_test = df.index[n_train + WINDOW + WINDOW:]
    if len(dates_test) >= len(preds_dollar):
        dates_test = dates_test[:len(preds_dollar)]
    out_df = pd.DataFrame({
        "pred": preds_dollar[:len(dates_test)],
        "actual": actuals_dollar[:len(dates_test)],
        "pred_norm": preds_norm[:len(dates_test)],
        "actual_norm": actuals_norm[:len(dates_test)],
    }, index=dates_test[:len(preds_dollar)])
    out_df.to_csv(OUTPUT_DIR / f"predictions_ThesisLeaky_{arch_name}_1d.csv")

    # Save eval pkl (C_compare compatible)
    ev = {
        "model": f"ThesisLeaky_{arch_name}", "horizon": "1d",
        "n": len(preds_dollar),
        "Overall_1d": {"label": "Overall 1d", "MAE": mae_dollar, "RMSE": rmse_dollar, "MAPE": mape},
        "alpha_shrink_1d": 1.0, "loss_1d": "mse",
        "conf_lo_1d": -1.65, "conf_hi_1d": 1.65, "use_sigma_1d": False,
        "leaky": True, "protocol": "thesis_leaky_whole_series_minmax",
        "mse_normalized": mse_norm, "mae_normalized": mae_norm,
        "thesis_reported_mse": {"NBEATSLite": 0.0002, "ILSTMCnn": 0.0005,
                                "ESN": 0.0017, "LSTNet": 7.989, "ILSTM": 9.429}.get(arch_name),
        "thesis_reported_mae": {"NBEATSLite": 0.0092, "ILSTMCnn": 0.015,
                                "ESN": 0.0111, "LSTNet": 1.552, "ILSTM": 1.811}.get(arch_name),
        "arch": arch_name,
    }
    with open(OUTPUT_DIR / f"eval_ThesisLeaky_{arch_name}_1d.pkl", "wb") as f:
        pickle.dump(ev, f)
    return ev


print("=" * 72)
print("  Thesis leaky reproduction — whole-series MinMax, 1-step, 75/25 split")
print("=" * 72)
print("  Thesis reported (Table XX):")
print("    NBEATSLite: MSE=0.0002  MAE=0.0092 (normalized)")
print("    ILSTMCnn:   MSE=0.0005  MAE=0.0150 (normalized)")
print("    ESN:        MSE=0.0017  MAE=0.0111 (normalized)")
print("    LSTNet:     MSE=7.989   MAE=1.552  (raw $)")
print("    ILSTM:      MSE=9.429   MAE=1.811  (raw $)")
print()

_results = {}
for _arch in MODELS_TO_RUN:
    _results[_arch] = run_thesis_model(_arch)

print(f"\n{'='*72}")
print(f"  T_thesis_leaky_repro complete — {len(_results)} models")
print(f"  Check: NBEATSLite normalized MAE should be ≈ 0.0092 if we reproduce the thesis")
print(f"  Outputs: predictions_ThesisLeaky_*_1d.csv + eval_ThesisLeaky_*_1d.pkl")
