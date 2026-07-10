# ============================================================
# DDP_v2/T_thesis_honest.py
# Run the 5 thesis models in our HONEST walk-forward harness for apples-to-apples.
# Uses the standard _run_wf_arch.py template (same splits/features/metrics as
# PatchTST / FEDformer / NHiTS etc.) — target = log-return, $-scale MAE output.
# Horizons: 1d (primary), 7d, 90d, 180d.
# Run from inside DDP_v2/.
# ============================================================
import os as _os
if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())
if "ILSTM" not in globals():
    exec(open("_thesis_models.py", encoding="utf-8").read())
if "crisis_sample_weights" not in globals() and _os.path.exists("_crisis_weighting.py"):
    exec(open("_crisis_weighting.py", encoding="utf-8").read())
if "fuse_signals" not in globals() and _os.path.exists("_fusion.py"):
    exec(open("_fusion.py", encoding="utf-8").read())

THESIS_ARCHS = globals().get("THESIS_ARCHS",
    ["NBEATSLite", "ILSTMCnn", "ESN", "LSTNet", "ILSTM"])
HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])

print("\n" + "#"*72)
print("#  T_thesis_honest — thesis models in honest walk-forward harness")
print(f"#  archs={THESIS_ARCHS}  horizons={HORIZONS}")
print("#"*72)

_all_results, _all_failures = {}, {}
for _arch in THESIS_ARCHS:
    for _hk in HORIZONS:
        print(f"\n{'─'*56}\n  {_arch} @ {_hk}  (honest walk-forward)\n{'─'*56}")
        try:
            globals()["MODEL_NAME"] = _arch
            globals()["HORIZONS"]   = [_hk]
            exec(open("_run_wf_arch.py", encoding="utf-8").read())
            _all_results[f"{_arch}_{_hk}"] = True
        except Exception as _e:
            import traceback as _tb
            print(f"  ⚠️  {_arch}/{_hk} FAILED: {_e}")
            _all_failures[f"{_arch}_{_hk}"] = str(_e)
        finally:
            globals().pop("MODEL_NAME", None)
            globals().pop("HORIZONS",   None)

# Restore HORIZONS for callers
globals()["HORIZONS"] = ["1d", "7d", "90d", "180d"]

print(f"\n{'='*72}")
print(f"  T_thesis_honest complete — {len(_all_results)} ok, {len(_all_failures)} failed")
print("  Outputs: predictions_{arch}_{hk}.csv + eval_{arch}_{hk}.pkl")
print("  Run T_thesis_compare.py next to build the head-to-head table.")
if _all_failures: print("  Failures:", list(_all_failures))
