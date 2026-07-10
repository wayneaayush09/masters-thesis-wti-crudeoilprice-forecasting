# ════════════════════════════════════════════════════════════════════
#  FILE : _paper_helpers.py
#  ROLE : Shared engine for Track A (leaky paper-faithful reproductions)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Provides the core building blocks exec'd by every A_yu_*.py
#         and A_huang_*.py driver:
#
#         whole_series_woa_vmd_fe(log_price)
#           Whale-Optimisation-Algorithm search over (K, α) → VMD with
#           K* modes → Fuzzy Entropy clustering into LF / MF / HF bands.
#           Reproduces Yu et al. (2025) protocol verbatim.
#
#         whole_series_vmd_emd(log_price, K, α)
#           VMD(K modes) → per-mode EMD → ~15 IMFs.
#           Reproduces Huang et al. (2024) protocol verbatim.
#
#         rolling_predict_band(band, train_len, h, model_fn)
#           Walk-forward NN predictor on a single band: at each test
#           origin, trains model_fn on band[:t], predicts h steps ahead.
#
#         save_paper_predictions(band_preds, log_price, dates, ...)
#           Reconstructs log-price predictions from summed band forecasts,
#           evaluates MAE/RMSE/MAPE, and saves predictions_*.csv +
#           eval_*.pkl. Tags result with leaky=True.
#
#         run_paper_yu(hk, model_name, epochs, ...)
#         run_paper_huang(hk, model_name, epochs, ...)
#           End-to-end runner: decompose → per-band NN → save. These are
#           the functions called by _run_paper_arch.py.
#
#  WHY  — ⚠️  PAPER PROTOCOL: these helpers DELIBERATELY perform whole-
#         series decomposition (VMD/WOA/EMD) BEFORE the train/test split.
#         This is LOOKAHEAD LEAKAGE by design (Xu & Niu 2024, Energy
#         Economics 128): future test-period data contaminates every mode,
#         making past modes artificially smooth and predictable.
#         We reproduce it verbatim to:
#           (a) directly contest the published headline numbers (MAE 3.45),
#           (b) quantify the leakage premium = Track A MAE − Track B MAE
#               at the same horizon, which is a publishable finding.
#         The `leaky=True` tag in all output pickles makes the accounting
#         explicit and allows C_compare to segregate tracks correctly.
#         For the leakage-free counterpart see 01a_vmd_features.py
#         (rolling WOA-VMD) and 09_vmd_hybrid.py (rolling VMD+AR).
#
#  HOW  — In-memory decomposition caches (_YU_DECOMP_CACHE / _HUANG_DECOMP_CACHE)
#         are shared via globals across all exec'd files in the session.
#         Both caches are also persisted to disk (cache/yu_decomp_{hk}.pkl)
#         so a reconnected session loads the ~7 min WOA-VMD result in <1 s.
#         The disk-backed cache pattern is mirrored in _lh_runners.py for
#         the extrapolator track (run_lh_yu / run_lh_huang).
#
#  READS    : df_features.pkl (log_price), cache/yu|huang_decomp_*.pkl
#  WRITES   : predictions_PaperYu_{ARCH}_{hk}.csv (+ Huang variants)
#             eval_PaperYu_{ARCH}_{hk}.pkl
#             cache/yu_decomp_{hk}.pkl (on first run per hk)
#  COMPUTE  : GPU (NN per-band training) + CPU (WOA-VMD decomposition)
#             WOA-VMD: ~7 min/horizon; NN per band: ~3–5 min/horizon
#  PIPELINE : exec'd by _run_paper_arch.py → A_yu_*.py / A_huang_*.py
#  RUNTIME  : ~15 min / horizon (first run); <1 s decomp on reconnect
# ════════════════════════════════════════════════════════════════════
# ============================================================

import os as _os_ph
import pickle, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except Exception as _te:
    # torch is needed ONLY for the NN per-band training paths (run_paper_yu /
    # run_paper_huang with a neural arch). The CPU extrapolator paths used by
    # A_components_linear/classical/hybrid (DLinear/Ridge/AR via _lh_runners →
    # _lh_forecasters) are numpy/sklearn and need no torch. So we degrade
    # gracefully: the module still loads (CPU components run locally even with a
    # broken torch install); only an actual NN path errors, and only when called.
    _TORCH_OK = False
    torch = None
    nn = None
    Dataset = object        # so `class _UnivariateDataset(Dataset)` still parses
    DataLoader = None
    print(f"  ⚠️  _paper_helpers: torch unavailable ({_te}); "
          f"NN per-band training disabled — CPU extrapolators still work.")

warnings.filterwarnings("ignore")
OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs")); OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu") if _TORCH_OK else "cpu"

# ── Dependency check ─────────────────────────────────────────
try:
    from vmdpy import VMD
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vmdpy"], check=False)
    from vmdpy import VMD

try:
    from PyEMD import EMD
    _EMD_OK = True
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "EMD-signal"], check=False)
        from PyEMD import EMD
        _EMD_OK = True
    except Exception:
        _EMD_OK = False
        print("⚠️  PyEMD unavailable — Huang's EMD secondary decomposition will fall back to residual-only")

# ── Search-space defaults (overridable per file) ─────────────
K_GRID_DEFAULT = [3, 4, 5, 6, 7]
A_GRID_DEFAULT = [500, 1000, 2000, 3000, 5000]


# ============================================================
# Fuzzy Entropy — reused logic from 01a_vmd_features.py
# (re-implemented here so paper files are standalone)
# ============================================================
def fuzzy_entropy(x, m=2, n=2, r=None):
    """Fuzzy entropy (Chen et al. 2007). Lower = more regular/smooth."""
    x = np.asarray(x, float)
    L = len(x)
    if L < m + 2: return np.nan
    if r is None: r = 0.2 * float(np.std(x) + 1e-12)
    def _phi(M):
        emb = np.array([x[i:i+M] for i in range(L - M + 1)])
        emb = emb - emb.mean(axis=1, keepdims=True)
        D = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        mu = np.exp(-(D ** n) / (r + 1e-12))
        np.fill_diagonal(mu, 0.0)
        denom = len(emb) * (len(emb) - 1)
        return mu.sum() / denom if denom > 0 else np.nan
    p_m, p_m1 = _phi(m), _phi(m + 1)
    if p_m is None or p_m1 is None or p_m <= 0 or p_m1 <= 0:
        return np.nan
    return float(np.log(p_m) - np.log(p_m1))


# ============================================================
# WOA — bounded discrete search over (K, α)
# Paper protocol: search on the ENTIRE log_price series (leakage by design).
# ============================================================
def whole_series_woa(log_price, K_grid=None, A_grid=None, pop=10, iters=10, seed=0):
    """Returns (K*, α*) minimizing envelope/fuzzy entropy of residual."""
    K_grid = K_grid or K_GRID_DEFAULT
    A_grid = A_grid or A_GRID_DEFAULT
    sig = log_price if len(log_price) % 2 == 0 else log_price[1:]
    cands = [(K, a) for K in K_grid for a in A_grid]
    rng = np.random.default_rng(seed)
    budget = min(pop * iters, len(cands))
    visited = set()
    best, best_score = (5, 2000), float("inf")
    for _ in range(budget):
        K, a = cands[rng.integers(len(cands))]
        if (K, a) in visited: continue
        visited.add((K, a))
        try:
            u, _, _ = VMD(sig, a, 0.0, K, 0, 1, 1e-7)
            residual = sig[:u.shape[1]] - u.sum(axis=0)
            score = fuzzy_entropy(residual)
            if np.isfinite(score) and score < best_score:
                best_score, best = score, (K, a)
        except Exception:
            continue
    return best


# ============================================================
# Yu et al. 2025: WOA-VMD-FE decomposition on whole series
# ============================================================
def whole_series_woa_vmd_fe(log_price):
    """Returns three numpy arrays (LF, MF, HF) each same length as log_price.

    Protocol (Yu et al. 2025):
    1. WOA(K, α) on whole log_price.
    2. VMD(log_price, K*, α*) → K modes.
    3. Fuzzy entropy per mode.
    4. Cluster modes into LF / MF / HF by 33rd/67th percentile of FE.
    5. Return summed sub-series per band.
    """
    sig = log_price if len(log_price) % 2 == 0 else log_price[1:]
    K_star, A_star = whole_series_woa(sig)
    print(f"  [Yu] WOA selected K={K_star}  α={A_star}")
    u, _, _ = VMD(sig, A_star, 0.0, K_star, 0, 1, 1e-7)   # (K, T)

    fes = np.array([fuzzy_entropy(m) for m in u])
    fes = np.nan_to_num(fes, nan=np.nanmean(fes) if np.any(np.isfinite(fes)) else 0.0)
    print(f"  [Yu] mode fuzzy-entropies: {[f'{e:.3f}' for e in fes]}")
    p33, p67 = np.percentile(fes, [33, 67])
    lf_idx = np.where(fes <= p33)[0]
    mf_idx = np.where((fes > p33) & (fes <= p67))[0]
    hf_idx = np.where(fes >  p67)[0]
    if len(lf_idx) == 0: lf_idx = np.array([int(np.argmin(fes))])
    if len(hf_idx) == 0: hf_idx = np.array([int(np.argmax(fes))])
    if len(mf_idx) == 0:
        mf_idx = np.array([i for i in range(len(fes)) if i not in lf_idx and i not in hf_idx])
        if len(mf_idx) == 0:
            mf_idx = lf_idx  # fallback: degenerate clustering

    lf_series = u[lf_idx, :].sum(axis=0)
    mf_series = u[mf_idx, :].sum(axis=0)
    hf_series = u[hf_idx, :].sum(axis=0)

    # Pad to original length if VMD trimmed (odd-length input)
    if len(sig) < len(log_price):
        lf_series = np.concatenate([[lf_series[0]], lf_series])
        mf_series = np.concatenate([[mf_series[0]], mf_series])
        hf_series = np.concatenate([[hf_series[0]], hf_series])
    return [lf_series, mf_series, hf_series], (K_star, A_star)


# ============================================================
# Huang et al. 2024: VMD-EMD decomposition on whole series
# ============================================================
def whole_series_vmd_emd(log_price, K=3, alpha=2000):
    """Returns list of K VMFs + (optionally) EMD-IMFs of the residual.

    Protocol (Huang et al. 2024):
    1. VMD(log_price, K=3) → 3 VMFs.
    2. residual = log_price − Σ VMFs
    3. EMD(residual) → IMFs (if PyEMD available; else use residual itself)
    4. Return list = [VMF_0, VMF_1, VMF_2, IMF_0, IMF_1, ...]
    """
    sig = log_price if len(log_price) % 2 == 0 else log_price[1:]
    u, _, _ = VMD(sig, alpha, 0.0, K, 0, 1, 1e-7)
    residual = sig[:u.shape[1]] - u.sum(axis=0)
    components = list(u)

    if _EMD_OK:
        try:
            emd = EMD()
            imfs = emd.emd(residual.astype(float))
            print(f"  [Huang] EMD on residual → {imfs.shape[0]} IMFs")
            for imf in imfs:
                components.append(imf)
        except Exception as e:
            print(f"  [Huang] EMD failed ({e}) — using raw residual as one component")
            components.append(residual)
    else:
        components.append(residual)

    if len(sig) < len(log_price):
        components = [np.concatenate([[c[0]], c]) for c in components]
    return components


# ============================================================
# 70/30 chronological split
# ============================================================
def paper_split(values, dates, ratio=0.7):
    """Returns (train_vals, test_vals, train_dates, test_dates)."""
    n = len(values)
    cut = int(n * ratio)
    return values[:cut], values[cut:], dates[:cut], dates[cut:]


# ============================================================
# Univariate per-band PatchTST trainer
# (Bands are smooth so a small d_model suffices.)
# ============================================================
class _UnivariateDataset(Dataset):
    """Per-band univariate windowed dataset.

    PAPER-FAITHFUL FIX (Yu 2025 / Huang 2024 protocol):
      - Each band is min-max normalized to [0,1] using TRAIN-only statistics
        before windowing. The fitted scaler is exposed on .scaler so the
        caller can inverse-transform predictions back to band scale at
        inference time.
      - Target index fixed to s[i + h] (was off-by-one s[i + h - 1]).
    """
    def __init__(self, series, h, lookback, scaler=None, scaler_type="minmax", seq_path=False):
        # ── Ablation toggles (Phase 3) ─────────────────────────────────────
        #   scaler_type : "minmax" (default, [0,1]) | "standard" (z-score).
        #                 z-score has no implied [0,1] ceiling, so it extrapolates
        #                 trend-carrying bands better at long horizons (Ablation C).
        #   seq_path    : if True the target is the FULL future path s[i+1:i+1+h]
        #                 (length h) instead of the single point s[i+h]. Gives the
        #                 model h supervision signals per window (Ablation D).
        s = np.asarray(series, dtype=np.float32).reshape(-1, 1)
        if scaler is None:
            if scaler_type == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler().fit(s)                      # fit on TRAIN only
            else:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(s)   # fit on TRAIN only
        s_norm = scaler.transform(s).ravel().astype(np.float32)
        self.scaler = scaler
        self.seq_path = seq_path
        self.X, self.y = [], []
        for i in range(lookback, len(s_norm) - h):
            self.X.append(s_norm[i - lookback:i].reshape(-1, 1))
            if seq_path:
                self.y.append(s_norm[i + 1:i + 1 + h])                # full path, len h
            else:
                self.y.append(s_norm[i + h])                          # fixed off-by-one
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def _ensure_model_registry_loaded():
    """Ensure 02_model_shared.py is exec'd into globals so MODEL_REGISTRY / build_model exist."""
    if globals().get("MODEL_REGISTRY") is None or globals().get("build_model") is None:
        try:
            exec(open("02_model_shared.py").read(), globals())
        except Exception as e:
            raise RuntimeError(f"Failed to load 02_model_shared.py: {e}")
    if globals().get("MODEL_REGISTRY") is None:
        raise RuntimeError("MODEL_REGISTRY missing — check 02_model_shared.py")


def train_band_model(train_series, h, lookback=64, epochs=80, batch_size=64,
                     d_model=64, n_layers=2, patch_len=8, lr=1e-3,
                     model_name="PatchTST", device=None, seed=42, verbose=False,
                     scaler_type="minmax", seq_path=False, max_patience=10):
    """Train ANY registered architecture on a univariate band series.

    ADD-15: Generalises the previous PatchTST-only `train_band_patchtst` to dispatch
    via `build_model()` from 02_model_shared.py — supports every entry of
    MODEL_REGISTRY (PatchTST, TCN, TCNBiLSTM, TimeMixer, NHiTS, Informer,
    Autoformer, iTransformer, TFT, CNNBiLSTMAttn).

    Returns: (trained_model, last_lookback_window_np_array)
    Raises a RuntimeError if architecture build fails (caller should catch + skip).
    """
    if device is None: device = DEVICE
    torch.manual_seed(seed); np.random.seed(seed)

    _ensure_model_registry_loaded()
    build_model_fn = globals().get("build_model")
    registry       = globals().get("MODEL_REGISTRY", {})
    if model_name not in registry:
        raise RuntimeError(f"Unknown architecture '{model_name}'. "
                           f"Available: {list(registry)}")

    # Synthetic horizons_dict — build_model reads cfg[horizon_key] for keys:
    # h, lookback, d_model, layers, dropout, loss, n_quantiles, patch (PatchTST/TimeLLM only)
    horizons_dict = {
        "_band": {
            "h": h, "lookback": lookback, "patch": patch_len, "layers": n_layers,
            "d_model": d_model, "loss": "mse", "n_quantiles": 1,
            "dropout": 0.15, "weight_decay": 1e-4,
            # Ablation D: emit a length-h path instead of a single scalar.
            "n_outputs_override": (h if seq_path else None),
        }
    }

    try:
        model = build_model_fn("_band", n_features=1,
                                horizons_dict=horizons_dict,
                                model_name=model_name).to(device)
    except Exception as e:
        raise RuntimeError(f"build_model failed for {model_name}: {e}")

    # ── CRITICAL FIX: disable RevIN for paper-protocol band models ──────────
    # RevIN (per-instance normalization) subtracts the window mean and divides
    # by window std.  This is correct for the walk-forward pipeline where
    # targets are log-returns (near zero-mean).  But for band-value prediction
    # (Yu 2025 / Huang 2024), targets ARE absolute band values. RevIN strips
    # exactly the level information the model needs, causing it to converge to
    # a constant (training-mean) prediction → MAE ≈ $33 regardless of arch.
    #
    # The papers use MinMaxScaler [0,1] only — no RevIN.  Replacing RevIN
    # with Identity makes the pipeline paper-faithful and lets the model see
    # the actual [0,1]-scaled band dynamics.
    if hasattr(model, "revin"):
        model.revin = nn.Identity()

    # PAPER-FAITHFUL FIX: split TRAIN into 90% fit / 10% val for early stopping.
    # The 10% val window is the most-recent slice of training data (chronological).
    arr = np.asarray(train_series, dtype=np.float32)
    val_cut = int(len(arr) * 0.9)
    fit_arr  = arr[:val_cut]
    val_arr  = arr[val_cut - lookback:]   # include lookback overlap so val has windows

    train_ds = _UnivariateDataset(fit_arr, h, lookback, scaler_type=scaler_type, seq_path=seq_path)
    if len(train_ds) < batch_size:
        if verbose: print(f"    train set too small ({len(train_ds)}) — returning untrained model")
        return model, np.asarray(train_series[-lookback:], dtype=np.float32), train_ds.scaler
    # val uses train's scaler (no leakage)
    val_ds = _UnivariateDataset(val_arr, h, lookback, scaler=train_ds.scaler,
                                scaler_type=scaler_type, seq_path=seq_path) if len(val_arr) > lookback + h else None

    dl    = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_ds is not None and len(val_ds) > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    # ── AMP mixed precision (DDP_SPEED-gated) ────────────────────────────────
    # balanced/aggressive on a GPU → fp16 forward/backward with an fp32 master
    # copy (GradScaler). The biggest win for the heavy transformers
    # (Autoformer/Informer/FEDformer/TFT). Safe here: _AutoCorrelation already
    # forces the FFT to fp32 (02_model_shared.py). 'full' (or CPU) → enabled=False,
    # which is a byte-for-byte no-op so the exact paper numbers are preserved.
    _use_amp = (_os_ph.environ.get("DDP_SPEED", "full").lower() in ("balanced", "aggressive")
                and torch.cuda.is_available() and str(device) != "cpu")
    _amp_scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

    def _forward_loss(X, y):
        out = model(X)
        if seq_path:
            # out: (B, h), y: (B, h) — MSE over the whole predicted path.
            pred = out if out.dim() == 2 else out.reshape(out.size(0), -1)
            return loss_fn(pred, y), pred
        pred = out.squeeze(-1) if out.dim() > 1 else out
        if pred.dim() > 1: pred = pred.squeeze(-1)
        return loss_fn(pred, y), pred

    best_val, patience = float("inf"), 0
    best_state = None
    for ep in range(epochs):
        model.train()
        running = 0.0
        for X, y in dl:
            X = X.to(device); y = y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=_use_amp):
                loss, _ = _forward_loss(X, y)
            _amp_scaler.scale(loss).backward()
            _amp_scaler.step(opt)
            _amp_scaler.update()
            running += float(loss.item())
        train_loss = running / max(len(dl), 1)

        # Validation pass for early stopping
        if val_dl is not None:
            model.eval()
            v_running, v_n = 0.0, 0
            with torch.no_grad():
                for X, y in val_dl:
                    X = X.to(device); y = y.to(device)
                    with torch.cuda.amp.autocast(enabled=_use_amp):
                        loss, _ = _forward_loss(X, y)
                    v_running += float(loss.item()) * len(y)
                    v_n += len(y)
            val_loss = v_running / max(v_n, 1)
            sched.step(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if verbose and (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1:02d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}  patience={patience}/{max_patience}")
            if patience >= max_patience:
                if verbose: print(f"    early stop at epoch {ep+1} (best val={best_val:.5f})")
                break
        else:
            if verbose and (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1:02d}/{epochs}  train={train_loss:.5f}  (no val)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, np.asarray(train_series[-lookback:], dtype=np.float32), train_ds.scaler


def train_band_patchtst(train_series, h, lookback=64, epochs=80, batch_size=64,
                        d_model=64, n_heads=4, n_layers=2, patch_len=8, lr=1e-3,
                        device=None, seed=42, verbose=False):
    """[Back-compat] Thin wrapper around train_band_model with model_name='PatchTST'.
    Kept so 14*/15* files written before ADD-15 still work.
    Returns (model, last_window, scaler) — 3-tuple (post-paper-fix)."""
    return train_band_model(train_series, h=h, lookback=lookback, epochs=epochs,
                            batch_size=batch_size, d_model=d_model, n_layers=n_layers,
                            patch_len=patch_len, lr=lr, model_name="PatchTST",
                            device=device, seed=seed, verbose=verbose)


def rolling_predict_band(model, full_band, train_len, h, lookback, scaler, device=None, seq_path=False):
    """Walk every test origin in the test segment and emit h-ahead prediction
    in BAND SCALE (not normalized scale).

    PAPER-FAITHFUL FIX:
      - Normalize the entire band series with the TRAIN-only scaler before windowing.
      - Predict in normalized [0,1] space.
      - Inverse-transform each prediction back to band scale.
    This is exactly Yu et al. 2025 / Huang et al. 2024 protocol.

    full_band: np.array of length N (full series, RAW band values).
    train_len: index where test starts (= len(train)).
    scaler: sklearn MinMaxScaler already FIT ON TRAIN ONLY.
    Returns dict {target_date_idx: predicted_band_value_at_band_scale}.
    """
    if device is None: device = DEVICE
    model.eval()
    band = np.asarray(full_band, dtype=np.float32).reshape(-1, 1)
    band_norm = scaler.transform(band).ravel().astype(np.float32)
    preds = {}
    with torch.no_grad():
        for origin in range(train_len, len(band_norm) - h):
            window = band_norm[origin - lookback:origin].reshape(1, lookback, 1).astype(np.float32)
            if not np.all(np.isfinite(window)): continue
            x = torch.from_numpy(window).to(device)
            out = model(x).squeeze().cpu().numpy()
            arr = np.asarray(out).flatten()
            if seq_path:
                # Ablation D: output is the length-h path; the h-ahead point is the LAST element.
                val_norm = float(arr[-1]) if arr.size else float("nan")
            elif hasattr(out, "shape") and out.shape != ():
                val_norm = float(arr[0])
            else:
                val_norm = float(out)
            # Inverse-transform back to band scale (paper-faithful)
            val_band = float(scaler.inverse_transform(np.array([[val_norm]]))[0, 0])
            preds[origin + h] = val_band
    return preds


# ============================================================
# I/O helpers — write predictions/eval in 06-compatible format
# ============================================================
def save_paper_predictions(preds_dict, log_price, dates, h, hk, model_name,
                           extra_meta=None):
    """preds_dict: {target_idx: predicted log_price}.
    Converts log-price predictions → price using exp(); aligns to wti_raw.

    Writes:
      outputs/predictions_{model_name}_{hk}.csv  — columns: pred, actual, target_date
      outputs/eval_{model_name}_{hk}.pkl
    """
    with open(OUTPUT_DIR/"wti_raw.pkl", "rb") as f:
        wti_raw = pickle.load(f)

    rows = []
    for tgt_idx, pred_log in preds_dict.items():
        if tgt_idx >= len(dates): continue
        tgt_date = dates[tgt_idx]
        pred_price = float(np.exp(pred_log))
        actual = wti_raw["price"].get(tgt_date, np.nan) if hasattr(wti_raw["price"], "get") \
            else (wti_raw.loc[tgt_date, "price"] if tgt_date in wti_raw.index else np.nan)
        # Use origin (target - h) for the index, matching how 06 reads predictions
        origin_idx = tgt_idx - h
        if origin_idx < 0 or origin_idx >= len(dates): continue
        origin_date = dates[origin_idx]
        rows.append({
            "origin": origin_date, "pred": pred_price,
            "actual": float(actual) if pd.notna(actual) else np.nan,
            "target_date": tgt_date,
        })
    res = pd.DataFrame(rows).dropna(subset=["actual"])
    if res.empty:
        print(f"  ⚠️  No valid predictions for {model_name} {hk}")
        return None
    res = res.set_index("origin").sort_index()
    res.to_csv(OUTPUT_DIR/f"predictions_{model_name}_{hk}.csv")

    # Compute simple eval summary
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = res["actual"].astype(float).values
    y_pred = res["pred"].astype(float).values
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = np.abs(y_true) > 1
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    # PAPER-FAITHFUL FIX: detect "predictions collapsed to a constant" failure mode.
    # If the model never learned dynamics, std(pred) ≪ std(actual) and MAE looks
    # plausible but is really the gap between constant-mean prediction and actual mean.
    std_ratio = float(np.std(y_pred) / (np.std(y_true) + 1e-9))
    if std_ratio < 0.10:
        print(f"  ⚠️  {model_name} {hk}: predictions look CONSTANT  "
              f"(std(pred)/std(actual)={std_ratio:.3f}). "
              "Check band normalization / scaler pipeline.")
    elif std_ratio < 0.30:
        print(f"  ⚠️  {model_name} {hk}: predictions have low variance  "
              f"(std(pred)/std(actual)={std_ratio:.3f}) — model may be undertrained.")

    eval_summary = {
        "model": model_name, "horizon": hk, "n": int(len(res)),
        f"Overall_{hk}": {"label": f"Overall {hk}", "MAE": mae, "RMSE": rmse, "MAPE": mape},
        # 06/13 expect these keys to exist even if no naive shrinkage was done
        f"alpha_shrink_{hk}": 1.0,
        f"loss_{hk}": "mse",
        f"conf_lo_{hk}": -1.65, f"conf_hi_{hk}": 1.65, f"use_sigma_{hk}": False,
        f"std_ratio_{hk}": std_ratio,    # paper-protocol non-constant guard
        "protocol": "paper_whole_series_decomposition",
    }
    if extra_meta:
        eval_summary.update(extra_meta)
    with open(OUTPUT_DIR/f"eval_{model_name}_{hk}.pkl", "wb") as f:
        pickle.dump(eval_summary, f)

    print(f"  ✅ {model_name} {hk}: MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  ({len(res)} dates)")
    return eval_summary


# ============================================================
# Top-level driver entry-points (called by 14*/15* files)
# ============================================================
# ── ADD-15 cache for whole-series decompositions ───────────────
# The decomposition is the same regardless of which architecture trains on the bands,
# so cache per (paper, hk) to avoid recomputing K* and α* for each of the 10 architectures.
_YU_DECOMP_CACHE    = {}    # hk → ((lf, mf, hf), (K*, α*))
_HUANG_DECOMP_CACHE = {}    # (hk, K, α) → list of components


# ============================================================
# Per-band / per-component checkpoint (no-work-lost resume)
# ============================================================
# Track A trains one NN per band (Yu: 3) or per component (Huang: ~15). Without
# a checkpoint, a disconnect mid-architecture discards every finished band and
# restarts that arch from scratch (the decomposition is already disk-cached, but
# the NN work — the expensive part — was lost). These helpers persist the running
# {done, band_preds_by_origin} after every band/component to the CHECKPOINTS_DIR
# on the (Drive/Kaggle) results root, so re-running resumes from the first
# unfinished band. FORCE_RERUN ignores any existing checkpoint. Mirrors Track B's
# per-fold checkpointing in _forecast_template.py.
def _paper_ckpt_path(save_name):
    base = globals().get("CHECKPOINTS_DIR") or (OUTPUT_DIR / "checkpoints")
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    return base / f"{save_name}.partial.pkl"


def _paper_ckpt_load(save_name):
    """Return (done_set, band_preds_by_origin). Empty when absent or FORCE_RERUN."""
    if globals().get("FORCE_RERUN", False):
        return set(), {}
    p = _paper_ckpt_path(save_name)
    if not p.exists():
        return set(), {}
    try:
        with open(p, "rb") as f:
            st = pickle.load(f)
        return set(st.get("done", [])), dict(st.get("band_preds_by_origin", {}))
    except Exception as e:
        print(f"  [warn] paper checkpoint load failed ({e}) — starting this arch fresh")
        return set(), {}


def _paper_ckpt_save(save_name, done, band_preds_by_origin):
    """Atomically persist progress (tmp + replace) so a teardown can't truncate it."""
    p = _paper_ckpt_path(save_name)
    try:
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"done": list(done),
                         "band_preds_by_origin": band_preds_by_origin}, f)
        tmp.replace(p)
    except Exception as e:
        print(f"  [warn] paper checkpoint save failed ({e})")


def _paper_ckpt_clear(save_name):
    try:
        p = _paper_ckpt_path(save_name)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def run_paper_yu(hk, split_ratio=0.7, epochs=30, lookback=None,
                 model_name="PatchTST", verbose_train=False,
                 train_kwargs=None, model_tag="", diagnose=False):
    """Run Yu 2025 WOA-VMD-FE-Transformer reproduction at horizon `hk` with any architecture.

    ADD-15: `model_name` selects which architecture from MODEL_REGISTRY to use for the
    per-band trainer. Outputs are saved as `PaperYu_{model_name}` (e.g. PaperYu_PatchTST,
    PaperYu_Autoformer, …). The whole-series WOA-VMD-FE decomposition is cached per
    horizon so it isn't recomputed across architectures.

    Phase 3 ablation hooks:
      train_kwargs : dict forwarded to train_band_model (d_model, scaler_type,
                     seq_path, max_patience, lr, …). epochs from train_kwargs
                     overrides the `epochs` arg.
      model_tag    : suffix appended to the saved model name (e.g. "_zscore"),
                     so ablations land on distinct comparator rows.
      diagnose     : if True, print per-band band-scale MAE to locate the
                     component(s) driving long-horizon error.

    hk ∈ {"1d","7d","90d","180d"}.
    """
    tk = dict(epochs=epochs)
    if train_kwargs:
        tk.update(train_kwargs)
    _seq_path = bool(tk.get("seq_path", False))
    HORIZONS_H = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
    h = HORIZONS_H[hk]
    if lookback is None:
        lookback = {"1d": 30, "7d": 60, "90d": 180, "180d": 360}[hk]

    print("=" * 60)
    print(f"  Paper Yu 2025 (WOA-VMD-FE + {model_name})  —  horizon {hk}  (h={h} days)")
    print(f"  Protocol: whole-series WOA-VMD-FE + 70/30 chronological split  (PAPER PROTOCOL — has lookahead leakage by design)")
    print("=" * 60)

    # Reuse cached decomposition if available — in-memory first, then the
    # ON-DISK cache (survives a runtime disconnect; Yu's WOA-VMD is ~7 min).
    if hk not in _YU_DECOMP_CACHE and globals().get("cache_load"):
        _disk = cache_load(f"yu_decomp_{hk}")
        if _disk is not None:
            _YU_DECOMP_CACHE[hk] = _disk
            print("  [Yu] loaded decomposition from disk cache (no recompute)")
    if hk in _YU_DECOMP_CACHE:
        (lf, mf, hf), (K_star, A_star) = _YU_DECOMP_CACHE[hk]
        print(f"  [Yu] reusing cached decomposition (K={K_star}, α={A_star})")
        with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df = pickle.load(f)
        log_price = df["log_price"].values.astype(float)
        dates     = df.index
    else:
        with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df = pickle.load(f)
        log_price = df["log_price"].values.astype(float)
        dates     = df.index
        t0 = time.time()
        (lf, mf, hf), (K_star, A_star) = whole_series_woa_vmd_fe(log_price)
        print(f"  [Yu] decomposition done in {time.time()-t0:.1f}s")
        _YU_DECOMP_CACHE[hk] = ((lf, mf, hf), (K_star, A_star))
        if globals().get("cache_save"):           # persist for the next session
            cache_save(f"yu_decomp_{hk}", _YU_DECOMP_CACHE[hk])

    train_len = int(len(log_price) * split_ratio)
    print(f"  [Yu] train_len={train_len}  test_len={len(log_price)-train_len}")
    # Resume from the per-band checkpoint (no-work-lost on a disconnect).
    _save_name = f"PaperYu_{model_name}{model_tag}_{hk}"
    _done, band_preds_by_origin = _paper_ckpt_load(_save_name)
    if _done:
        print(f"  [Yu] resume: bands {sorted(_done)} already trained — skipping them")
    band_names = ["LF", "MF", "HF"]
    for name, band in zip(band_names, [lf, mf, hf]):
        if name in _done:
            continue
        print(f"  [Yu/{model_name}] training band {name} …")
        model, _, scaler = train_band_model(band[:train_len], h=h, lookback=lookback,
                                            model_name=model_name, verbose=verbose_train, **tk)
        preds = rolling_predict_band(model, band, train_len, h, lookback,
                                     scaler=scaler, seq_path=_seq_path)
        if diagnose and preds:
            _idx = [k for k in preds if k < len(band)]
            _mae = float(np.mean([abs(preds[k] - band[k]) for k in _idx])) if _idx else float("nan")
            print(f"    [diag] band {name}: band-scale MAE={_mae:.5f}  (n={len(_idx)})")
        for k, v in preds.items():
            band_preds_by_origin[k] = band_preds_by_origin.get(k, 0.0) + v
        _done.add(name)
        _paper_ckpt_save(_save_name, _done, band_preds_by_origin)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    _res = save_paper_predictions(band_preds_by_origin, log_price, dates,
                                  h=h, hk=hk, model_name=f"PaperYu_{model_name}{model_tag}",
                                  extra_meta={"K_star": K_star, "alpha_star": A_star,
                                              "arch": model_name, "ablation": model_tag})
    _paper_ckpt_clear(_save_name)      # arch×horizon done → drop the partial file
    return _res


def run_paper_huang(hk, split_ratio=0.7, epochs=30, lookback=None, K=3, alpha=2000,
                    model_name="PatchTST", verbose_train=False,
                    train_kwargs=None, model_tag="", diagnose=False, fe_keep_quantile=None):
    """Run Huang 2024 VMD-EMD-Transformer reproduction at horizon `hk` with any architecture.

    ADD-15: `model_name` parameter selects the architecture. Outputs saved as
    `PaperHuang_{model_name}`. Whole-series VMD+EMD decomposition is cached
    per (hk, K, α) so it isn't recomputed across architectures.

    Phase 3 ablation hooks (see run_paper_yu for train_kwargs/model_tag/diagnose):
      fe_keep_quantile : if set (e.g. 0.67), rank components by fuzzy entropy of
                         their TRAIN slice and only TRAIN models on the lowest-FE
                         (most predictable) fraction; the high-FE / noise-like
                         components are predicted as their train-period mean
                         (a constant) instead of by a model. Reduces error
                         accumulation from unpredictable IMFs (Ablation B).
    """
    tk = dict(epochs=epochs)
    if train_kwargs:
        tk.update(train_kwargs)
    _seq_path = bool(tk.get("seq_path", False))
    HORIZONS_H = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
    h = HORIZONS_H[hk]
    if lookback is None:
        lookback = {"1d": 30, "7d": 60, "90d": 180, "180d": 360}[hk]

    print("=" * 60)
    print(f"  Paper Huang 2024 (VMD-EMD + {model_name})  —  horizon {hk}  (h={h} days)")
    print(f"  Protocol: whole-series VMD(K={K})+EMD + 70/30 chronological split  (PAPER PROTOCOL)")
    print("=" * 60)

    cache_key = (hk, K, alpha)
    _disk_key = f"huang_decomp_{hk}_{K}_{alpha}"
    if cache_key not in _HUANG_DECOMP_CACHE and globals().get("cache_load"):
        _disk = cache_load(_disk_key)
        if _disk is not None:
            _HUANG_DECOMP_CACHE[cache_key] = _disk
            print("  [Huang] loaded decomposition from disk cache (no recompute)")
    if cache_key in _HUANG_DECOMP_CACHE:
        components = _HUANG_DECOMP_CACHE[cache_key]
        print(f"  [Huang] reusing cached decomposition ({len(components)} components)")
        with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df = pickle.load(f)
        log_price = df["log_price"].values.astype(float)
        dates     = df.index
    else:
        with open(OUTPUT_DIR/"df_features.pkl","rb") as f: df = pickle.load(f)
        log_price = df["log_price"].values.astype(float)
        dates     = df.index
        t0 = time.time()
        components = whole_series_vmd_emd(log_price, K=K, alpha=alpha)
        print(f"  [Huang] decomposition done in {time.time()-t0:.1f}s ({len(components)} components)")
        _HUANG_DECOMP_CACHE[cache_key] = components
        if globals().get("cache_save"):
            cache_save(_disk_key, components)

    train_len = int(len(log_price) * split_ratio)
    print(f"  [Huang] train_len={train_len}  test_len={len(log_price)-train_len}")

    # Pre-pad all components to len(log_price) once.
    comps = []
    for comp in components:
        comp = np.asarray(comp, dtype=float)
        if len(comp) < len(log_price):
            comp = np.concatenate([[comp[0]], comp])
        comps.append(comp)

    # Ablation B: rank components by fuzzy entropy of their TRAIN slice; only model
    # the lowest-FE (predictable) fraction, predict the rest as their train mean.
    model_idx = set(range(len(comps)))
    if fe_keep_quantile is not None and 0 < fe_keep_quantile < 1:
        fes = np.array([fuzzy_entropy(c[:train_len]) for c in comps])
        fes = np.nan_to_num(fes, nan=np.nanmax(fes) if np.any(np.isfinite(fes)) else 0.0)
        thresh = np.quantile(fes, fe_keep_quantile)
        model_idx = {i for i, fe in enumerate(fes) if fe <= thresh}
        print(f"  [Huang] FE-drop: modeling {len(model_idx)}/{len(comps)} low-FE components "
              f"(FE≤{thresh:.4f}); remaining predicted as train-mean.")

    # Resume from the per-component checkpoint (Huang has ~15 → the most to lose).
    _save_name = f"PaperHuang_{model_name}{model_tag}_{hk}"
    _done, band_preds_by_origin = _paper_ckpt_load(_save_name)
    if _done:
        print(f"  [Huang] resume: {len(_done)}/{len(comps)} components already done — skipping them")
    for i, comp in enumerate(comps):
        if i in _done:
            continue
        if i in model_idx:
            print(f"  [Huang/{model_name}] training component {i+1}/{len(comps)} …")
            model, _, scaler = train_band_model(comp[:train_len], h=h, lookback=lookback,
                                                model_name=model_name, verbose=verbose_train, **tk)
            preds = rolling_predict_band(model, comp, train_len, h, lookback,
                                         scaler=scaler, seq_path=_seq_path)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # Constant train-mean predictor for unpredictable (high-FE) component.
            cmean = float(np.mean(comp[:train_len]))
            preds = {origin + h: cmean for origin in range(train_len, len(comp) - h)}
        if diagnose and preds:
            _idx = [k for k in preds if k < len(comp)]
            _mae = float(np.mean([abs(preds[k] - comp[k]) for k in _idx])) if _idx else float("nan")
            _kind = "model" if i in model_idx else "mean "
            print(f"    [diag] comp {i+1:02d} ({_kind}): band-scale MAE={_mae:.5f}  (n={len(_idx)})")
        for k, v in preds.items():
            band_preds_by_origin[k] = band_preds_by_origin.get(k, 0.0) + v
        _done.add(i)
        _paper_ckpt_save(_save_name, _done, band_preds_by_origin)

    _res = save_paper_predictions(band_preds_by_origin, log_price, dates,
                                  h=h, hk=hk, model_name=f"PaperHuang_{model_name}{model_tag}",
                                  extra_meta={"K": K, "alpha": alpha,
                                              "n_components": len(comps),
                                              "arch": model_name, "ablation": model_tag,
                                              "fe_keep_quantile": fe_keep_quantile})
    _paper_ckpt_clear(_save_name)      # arch×horizon done → drop the partial file
    return _res

def run_standalone(hk, model_name="FEDformer", split_ratio=0.7, epochs=80, lookback=None,
                   verbose_train=False, train_kwargs=None, model_tag=""):
    """Train any MODEL_REGISTRY architecture directly on the RAW log-price series
    (NO VMD/EMD decomposition). Saves as `Standalone_{model_name}{model_tag}_{hk}`.

    This is the 'model decomposes internally (or not at all)' control: it lets us ask
    whether external VMD/EMD still helps once a model like FEDformer does its own
    seasonal-trend + frequency decomposition. Reuses train_band_model /
    rolling_predict_band by treating the whole log-price as a single band.
    """
    HORIZONS_H = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
    h = HORIZONS_H[hk]
    if lookback is None:
        lookback = {"1d": 30, "7d": 60, "90d": 180, "180d": 360}[hk]

    print("=" * 60)
    print(f"  Standalone (raw log-price, NO decomposition) ({model_name})  —  horizon {hk}  (h={h} days)")
    print("=" * 60)

    with open(OUTPUT_DIR/"df_features.pkl", "rb") as f:
        df = pickle.load(f)
    log_price = df["log_price"].values.astype(float)
    dates     = df.index

    tk = dict(epochs=epochs)
    if train_kwargs:
        tk.update(train_kwargs)
    _seq_path = bool(tk.get("seq_path", False))

    train_len = int(len(log_price) * split_ratio)
    print(f"  [Standalone] train_len={train_len}  test_len={len(log_price)-train_len}")
    model, _, scaler = train_band_model(log_price[:train_len], h=h, lookback=lookback,
                                        model_name=model_name, verbose=verbose_train, **tk)
    preds = rolling_predict_band(model, log_price, train_len, h, lookback,
                                 scaler=scaler, seq_path=_seq_path)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return save_paper_predictions(preds, log_price, dates,
                                  h=h, hk=hk, model_name=f"Standalone_{model_name}{model_tag}",
                                  extra_meta={"arch": model_name, "ablation": model_tag,
                                              "protocol": "standalone_no_decomposition"})


print("✅ _paper_helpers.py loaded")
