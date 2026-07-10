# ============================================================
# DDP_v2/F_fedformer_sweep.py  — FEDformer deep-dive sweep (extra focus)
# Runs the whole FEDformer family in the leaky decomposition (Yu+Huang) AND
# standalone, with tuned per-horizon configs:
#   FEDformer (Fourier lite) · FEDformerW (wavelet) · FreEformer (2025)
# Tags each so C_compare ranks them; reports which basis/variant wins per horizon.
# Run from inside DDP_v2/.
# ============================================================
exec(open("_models_plus.py", encoding="utf-8").read())                 # registers the family + improved_train_kwargs
if "run_paper_yu" not in globals():
    exec(open("_paper_helpers.py", encoding="utf-8").read())
if "run_standalone" not in globals():
    pass  # run_standalone lives in _paper_helpers (already loaded)

FAMILY   = globals().get("FEDFORMER_FAMILY", ["FEDformer", "FEDformerW", "FreEformer"])
HORIZONS = globals().get("TRACK_HORIZONS", ["7d", "90d", "180d"])

_results, _failures = {}, {}
for _arch in FAMILY:
    for _hk in HORIZONS:
        tk = improved_train_kwargs(_hk, arch=_arch)
        # (1) inside Yu decomposition  (2) inside Huang  (3) standalone on raw log-price
        jobs = [("yu", lambda: run_paper_yu(hk=_hk, model_name=_arch, train_kwargs=tk,
                                            model_tag="_tuned")),
                ("huang", lambda: run_paper_huang(hk=_hk, model_name=_arch, train_kwargs=tk,
                                                  model_tag="_tuned")),
                ("std", lambda: run_standalone(hk=_hk, model_name=_arch, train_kwargs=tk,
                                               model_tag="_tuned"))]
        for _mode, _job in jobs:
            key = f"{_arch}_{_mode}_{_hk}"
            try:
                _results[key] = _job()
            except Exception as _e:
                print(f"  ⚠️  {key} FAILED: {_e}")
                _failures[key] = str(_e)

print(f"\n{'='*60}\n  FEDformer sweep — {len(_results)} ok, {len(_failures)} failed\n{'='*60}")
if _failures: print("  Failures:", list(_failures))
