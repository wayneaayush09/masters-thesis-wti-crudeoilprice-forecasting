# ════════════════════════════════════════════════════════════════════
#  FILE : _orchestrate.py
#  ROLE : Shared sequential stage runner (engine behind run_* scripts)
# ════════════════════════════════════════════════════════════════════
#  WHAT — Runs the list in the global `STAGES`, one file after another,
#         in the SAME namespace so loaded data / caches / model classes
#         persist across stages. Logs each stage to the run manifest and
#         continues past failures.
#
#  WHY  — run_short / run_long / run_local_cpu differ ONLY in their
#         STAGES list, track, and horizons. Factor the loop here so the
#         three runners stay tiny and identical in behaviour (the user's
#         earlier concern: avoid duplicated maintenance).
#
#  HOW  — Assumes the caller already exec'd _env.py + _plotting.py (so
#         OUTPUT_DIR, HORIZONS, log_stage, save_fig are in globals) and
#         set `STAGES`. Each stage is exec'd into the caller's globals;
#         each driver self-bootstraps its own heavy deps (02_model_shared,
#         _paper_helpers, …) via the existing `if X not in globals()` guards.
#
#  READS    : globals STAGES, HORIZONS, log_stage      WRITES : logs/run_manifest.jsonl
#  COMPUTE  : inherits each stage's tier                PIPELINE : engine (not a stage)
#  RUNTIME  : sum of its stages
# ════════════════════════════════════════════════════════════════════

import os as _os
import time as _time
import traceback as _tb

_STAGES = globals().get("STAGES", [])
_HORIZONS = globals().get("TRACK_HORIZONS", ["1d", "7d", "90d", "180d"])
_track = _os.environ.get("DDP_TRACK", "all")

print(f"\n{'#'*64}\n#  ORCHESTRATE  |  track='{_track}'  |  horizons={_HORIZONS}\n"
      f"#  {len(_STAGES)} stage(s)\n{'#'*64}")

_passed = _failed = 0
_wall = _time.time()

# ── Stage-level skip: read manifest for stages already completed ─────────────
# On restart the orchestrator skips any stage already logged as "done" so a
# kernel restart resumes from the first incomplete stage rather than from
# stage 1. Disabled when FORCE_RERUN=1.
_manifest_done = set()
_manifest_path = LOGS_DIR / "run_manifest.jsonl"
if _manifest_path.exists() and not (globals().get("FORCE_RESTAGE", False) or FORCE_RERUN):
    import json as _json2
    with open(_manifest_path, encoding="utf-8") as _mf:
        for _line in _mf:
            try:
                _rec = _json2.loads(_line.strip())
                if _rec.get("status") == "done":
                    # Key by (stage, track): a stage finished for one horizon/track
                    # must NOT skip the same stage for a different one when they share
                    # a results root (e.g. per-horizon 1d/7d/90d/180d runs).
                    _manifest_done.add((_rec.get("stage", ""), _rec.get("track", "")))
            except Exception:
                pass
    if _manifest_done:
        print(f"  ℹ️  manifest: {len(_manifest_done)} (stage,track) pair(s) already done — will skip")

for _i, _stage in enumerate(_STAGES, 1):
    print(f"\n{'='*64}\n  [{_i}/{len(_STAGES)}]  {_stage}   (track={_track})\n{'='*64}")
    if (_stage, _track) in _manifest_done:
        print(f"  ⏭️  {_stage} — already done (manifest, track={_track}) — skipping")
        _passed += 1
        continue
    if not _os.path.exists(_stage):
        print(f"  ⚠️  not found — skipping: {_stage}")
        log_stage(_stage, status="missing")
        continue
    _t = _time.time()
    try:
        # exec into THIS namespace so globals (HORIZONS, OUTPUT_DIR, helpers,
        # and any data already loaded) are visible to the stage and persist.
        exec(compile(open(_stage, encoding="utf-8").read(), _stage, "exec"), globals())
        _dt = _time.time() - _t
        print(f"\n  ✅ {_stage} done in {_dt:.1f}s")
        log_stage(_stage, status="done", seconds=round(_dt, 1))
        _passed += 1
    except Exception:
        _dt = _time.time() - _t
        print(f"\n  ❌ {_stage} FAILED after {_dt:.1f}s:")
        _tb.print_exc()
        log_stage(_stage, status="failed", seconds=round(_dt, 1))
        _failed += 1

_mins = (_time.time() - _wall) / 60
print(f"\n{'#'*64}\n#  ORCHESTRATE COMPLETE [{_track}] — {_passed} ok, {_failed} failed"
      f"  ({_mins:.1f} min)\n{'#'*64}")
log_stage("__orchestrate__", status="complete", track=_track,
          passed=_passed, failed=_failed, minutes=round(_mins, 1))
