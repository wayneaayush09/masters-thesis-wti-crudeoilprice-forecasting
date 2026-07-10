# ============================================================
# DDP_v2/_models_plus.py  — register ALL new architectures + improved configs
# Additive: execs the SOTA models (DLinearNet/NLinearNet/TiDE/FITS/SegRNN) and the
# FEDformer family (FEDformerW/FreEformer) into the in-memory MODEL_REGISTRY.
# 02_model_shared.py is never edited.
# Also exposes improved per-horizon training knobs for long-horizon accuracy.
# Run / exec from inside DDP_v2/.
# ============================================================
import os as _os

if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())
if "DLinearNet" not in MODEL_REGISTRY and _os.path.exists("_lh_sota_models.py"):
    exec(open("_lh_sota_models.py", encoding="utf-8").read())          # DLinearNet/NLinearNet/TiDE/FITS/SegRNN
if "FreEformer" not in MODEL_REGISTRY and _os.path.exists("_fedformer_plus.py"):
    exec(open("_fedformer_plus.py", encoding="utf-8").read())          # FEDformerW/FreEformer + FEDFORMER_CONFIG

# ── improved per-horizon training knobs (long-horizon focus) ─────────────────
# Bigger capacity + longer lookback at 90d/180d; longer training + patience.
IMPROVED_CONFIG = {
    "1d":   {"epochs": 80,  "d_model": 64,  "n_layers": 2, "max_patience": 10},
    "7d":   {"epochs": 80,  "d_model": 128, "n_layers": 3, "max_patience": 10},
    "90d":  {"epochs": 120, "d_model": 128, "n_layers": 3, "max_patience": 15},
    "180d": {"epochs": 120, "d_model": 256, "n_layers": 4, "max_patience": 15},
}


def improved_train_kwargs(hk, arch=None):
    """Return train_kwargs (for run_paper_*/train_band_model) tuned for horizon hk.
    FEDformer family also gets its modes/d_model from FEDFORMER_CONFIG."""
    cfg = dict(IMPROVED_CONFIG.get(hk, {}))
    if arch and "FEDformer" in arch and "FEDFORMER_CONFIG" in globals():
        fc = FEDFORMER_CONFIG.get(hk, {})
        cfg["d_model"] = fc.get("d_model", cfg.get("d_model"))
        cfg["n_layers"] = fc.get("n_layers", cfg.get("n_layers"))
    # train_band_model accepts: epochs,d_model,n_layers,max_patience,lr,scaler_type,seq_path
    return {k: v for k, v in cfg.items()
            if k in {"epochs", "d_model", "n_layers", "max_patience", "lr",
                     "scaler_type", "seq_path"}}


print(f"✅ DDP_v2/_models_plus.py loaded — MODEL_REGISTRY now has {len(MODEL_REGISTRY)} archs")
print("   new:", [m for m in ("DLinearNet","NLinearNet","TiDE","FITS","SegRNN","FEDformerW","FreEformer")
                  if m in MODEL_REGISTRY])
