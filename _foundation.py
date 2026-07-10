# ============================================================
# DDP_v2/_foundation.py  — zero-shot Time-Series Foundation Models
# Amazon Chronos-Bolt + Google TimesFM as window forecasters on log-price.
# Produce the same {target_idx: predicted_log_price} dict as rolling_predict_band,
# so save_paper_predictions reconstructs/scores them unchanged. Strided over test
# origins (foundation inference is heavy) + forward-filled. leaky=False (honest:
# each forecast uses only the trailing window ≤ origin).
# Lazy pip install + graceful skip if unavailable / offline.
# Run / exec from inside DDP_v2/.
# ============================================================
import os, subprocess, sys, pickle
import numpy as np

if "save_paper_predictions" not in globals():
    exec(open("_paper_helpers.py", encoding="utf-8").read())

_LB = {"1d": 64, "7d": 96, "90d": 336, "180d": 512}   # context length per horizon
_HMAP = {"1d": 1, "7d": 7, "90d": 90, "180d": 180}
_chronos = None
_timesfm = None


def _pip(pkg):
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    except Exception:
        pass


def _ensure_chronos():
    global _chronos
    if _chronos is not None:
        return _chronos
    try:
        try:
            from chronos import BaseChronosPipeline
        except Exception:
            _pip("chronos-forecasting"); from chronos import BaseChronosPipeline
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _chronos = BaseChronosPipeline.from_pretrained(
            os.environ.get("CHRONOS_MODEL", "amazon/chronos-bolt-base"),
            device_map=dev, torch_dtype="auto")
        print("✅ Chronos-Bolt loaded")
    except Exception as e:
        print(f"⚠️  Chronos unavailable ({e}) — skipping")
        _chronos = False
    return _chronos


def _ensure_timesfm():
    global _timesfm
    if _timesfm is not None:
        return _timesfm
    try:
        try:
            import timesfm
        except Exception:
            _pip("timesfm"); import timesfm
        _timesfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                           horizon_len=256),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        print("✅ TimesFM loaded")
    except Exception as e:
        print(f"⚠️  TimesFM unavailable ({e}) — skipping")
        _timesfm = False
    return _timesfm


def _forecast(series, train_len, h, lookback, kind, stride=10):
    s = np.asarray(series, float)
    origins = list(range(train_len, len(s) - h))
    sampled = origins[::stride]
    raw = {}
    if kind == "chronos":
        pipe = _ensure_chronos()
        if not pipe:
            return {}
        import torch
        for o in sampled:
            ctx = torch.tensor(s[max(0, o - lookback):o], dtype=torch.float32)
            try:
                q, mean = pipe.predict_quantiles(context=ctx, prediction_length=h,
                                                 quantile_levels=[0.5])
                raw[o] = float(np.asarray(mean)[0, -1])
            except Exception:
                raw[o] = float(s[o - 1])
    elif kind == "timesfm":
        tfm = _ensure_timesfm()
        if not tfm:
            return {}
        for o in sampled:
            ctx = s[max(0, o - lookback):o].astype(np.float32)
            try:
                fc, _ = tfm.forecast([ctx], freq=[0])
                raw[o] = float(np.asarray(fc)[0, h - 1])
            except Exception:
                raw[o] = float(s[o - 1])
    # forward-fill to every origin
    preds, last, si = {}, None, 0
    for o in origins:
        if si < len(sampled) and o >= sampled[si]:
            last = raw.get(sampled[si], last); si += 1
        if last is not None:
            preds[o + h] = last
    return preds


def run_foundation_standalone(hk, model="chronos", split_ratio=0.7, stride=10):
    h = _HMAP[hk]; lb = _LB[hk]
    with open(OUTPUT_DIR / "df_features.pkl", "rb") as f:
        df = pickle.load(f)
    logp = df["log_price"].values.astype(float); dates = df.index
    train_len = int(len(logp) * split_ratio)
    print("=" * 60)
    print(f"  Foundation [{model}] standalone (zero-shot)  —  {hk} (h={h}, lb={lb})")
    print("=" * 60)
    preds = _forecast(logp, train_len, h, lb, kind=model, stride=stride)
    if not preds:
        print(f"  (skipped — {model} unavailable)"); return None
    return save_paper_predictions(preds, logp, dates, h=h, hk=hk,
                                  model_name=f"Foundation_{model}",
                                  extra_meta={"leaky": False, "arch": model,
                                              "protocol": "foundation_zero_shot"})


print("✅ DDP_v2/_foundation.py loaded — run_foundation_standalone(hk, model='chronos'|'timesfm')")
