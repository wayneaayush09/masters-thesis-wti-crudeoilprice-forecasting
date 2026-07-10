# ============================================================
# DDP_v2/E_uni_vs_multi.py  — Univariate vs Multivariate deep-dive
# For each architecture, predict h-ahead LOG-PRICE on a 70/30 chronological split,
# twice on the SAME windows:
#   • UNIVARIATE  : input = log_price window only  (n_features=1)
#   • MULTIVARIATE: input = full feature panel      (n_features=k, incl. _features_plus)
# Reports per-(arch,horizon) MAE/MAPE and the multivariate benefit Δ.
# Self-contained (own compact trainer via build_model); honest (no decomposition leakage).
# Run from inside DDP_v2/.
# ============================================================
import os, pickle
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except Exception as _te:
    raise RuntimeError(f"E_uni_vs_multi requires torch: {_te}. Run _env.py first.") from _te
from sklearn.preprocessing import RobustScaler

if "MODEL_REGISTRY" not in globals():
    exec(open("_models_plus.py", encoding="utf-8").read())             # all archs incl. new ones
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "outputs"

ARCHS    = globals().get("UNI_MULTI_ARCHS",
                         ["PatchTST", "TCN", "TimeMixer", "NHiTS", "iTransformer",
                          "FEDformer", "FEDformerW", "FreEformer", "DLinearNet", "TiDE"])
HORIZONS = globals().get("TRACK_HORIZONS", ["7d", "90d", "180d"])
LOOKBACK = {"1d": 30, "7d": 60, "90d": 180, "180d": 360}
HMAP     = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
EPOCHS   = globals().get("UM_EPOCHS", 40)


def _windows(arr2d, target1d, lookback, h):
    X, y = [], []
    for i in range(lookback, len(arr2d) - h):
        X.append(arr2d[i - lookback:i]); y.append(target1d[i + h])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def _train_eval(model_name, feats, hk, df):
    h = HMAP[hk]; lb = LOOKBACK[hk]
    logp = df["log_price"].values.astype(np.float32)
    n = len(df); cut = int(n * 0.7)
    F = np.asarray(feats, np.float32)                 # (n, k)
    sc = RobustScaler().fit(F[:cut])
    Fs = np.clip(sc.transform(F), -10, 10).astype(np.float32)
    Xtr, ytr = _windows(Fs[:cut], logp[:cut], lb, h)
    Xte, yte = [], []
    for o in range(cut, n - h):
        Xte.append(Fs[o - lb:o]); yte.append((o + h, logp[o + h]))
    if len(Xtr) < 64 or not Xte:
        return None
    Xte_arr = np.asarray([x for x in Xte], np.float32)
    k = F.shape[1]
    hd = {"_b": {"h": h, "lookback": lb, "patch": 8 if lb % 8 == 0 else 4, "layers": 2,
                 "d_model": 64, "loss": "mse", "n_quantiles": 1, "dropout": 0.15}}
    try:
        model = build_model("_b", n_features=k, horizons_dict=hd, model_name=model_name).to(DEVICE)
    except Exception as e:
        return {"error": str(e)}
    if hasattr(model, "revin"):
        model.revin = nn.Identity()                   # predicting level, not return
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    bs = 128
    model.train()
    for ep in range(EPOCHS):
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = idx[i:i + bs]
            xb = Xt[b].to(DEVICE); yb = yt[b].to(DEVICE)
            opt.zero_grad()
            out = model(xb); pred = out.squeeze(-1) if out.dim() > 1 else out
            if pred.dim() > 1: pred = pred[:, 0]
            loss = lossf(pred, yb); loss.backward(); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_arr), 256):
            xb = torch.from_numpy(Xte_arr[i:i + 256]).to(DEVICE)
            out = model(xb).squeeze().cpu().numpy()
            preds.extend(np.atleast_1d(out).ravel()[:len(xb)] if out.ndim else [float(out)])
    preds = np.asarray(preds[:len(Xte)], float)
    price_pred = np.exp(preds); price_true = np.exp(np.asarray([v for _, v in Xte], float))
    mae = float(np.mean(np.abs(price_pred - price_true)))
    mape = float(np.mean(np.abs((price_true - price_pred) / price_true)) * 100)
    return {"MAE": round(mae, 4), "MAPE": round(mape, 3)}


with open(os.path.join(OUT, "df_features.pkl"), "rb") as f:
    df = pickle.load(f)
with open(os.path.join(OUT, "feat_cols.pkl"), "rb") as f:
    feat_cols = pickle.load(f)

rows = []
for hk in HORIZONS:
    multi_cols = [c for c in feat_cols.get(hk, []) if c in df.columns] or ["log_price"]
    uni = df[["log_price"]].values
    multi = df[multi_cols].fillna(0.0).values
    for arch in ARCHS:
        ru = _train_eval(arch, uni, hk, df)
        rm = _train_eval(arch, multi, hk, df)
        if not ru or not rm or "error" in (ru or {}) or "error" in (rm or {}):
            print(f"  ⚠️  {arch}/{hk} skipped ({(ru or rm or {}).get('error','no data')})")
            continue
        rows.append({"arch": arch, "horizon": hk,
                     "uni_MAE": ru["MAE"], "multi_MAE": rm["MAE"],
                     "delta_MAE": round(ru["MAE"] - rm["MAE"], 4),
                     "uni_MAPE": ru["MAPE"], "multi_MAPE": rm["MAPE"]})
        print(f"  {arch:<12} {hk:<5}  uni MAE={ru['MAE']:.3f}  multi MAE={rm['MAE']:.3f}"
              f"  Δ={ru['MAE']-rm['MAE']:+.3f}")

if rows:
    out = pd.DataFrame(rows).sort_values(["horizon", "multi_MAE"])
    out.to_csv(os.path.join(OUT, "uni_vs_multi.csv"), index=False)
    print(f"\n✅ saved outputs/uni_vs_multi.csv ({len(out)} rows). "
          "Positive delta_MAE = multivariate helps.")
else:
    print("⚠️  no uni_vs_multi rows produced")
