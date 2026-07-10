# ============================================================
# DDP_v2/F_fedformer_official.py  — official seq2seq FEDformer (standalone)
# Reuses the robust official-FEDformer runner from the base repo's 18c file
# (path discovery + graceful fallback to FEDformerLite if the PatchTST/ repo
# isn't present). Self-contained: looks for 18c in the base dir or DDP_v2/.
# Run from inside DDP_v2/.
# ============================================================
import os as _os

_cands = ["18c_fedformer_official.py", _os.path.join("..", "18c_fedformer_official.py")]
_src = next((c for c in _cands if _os.path.exists(c)), None)
if _src is None:
    print("⚠️  18c_fedformer_official.py not found in DDP_v2/ or parent. "
          "Copy it in to run the official seq2seq FEDformer; "
          "FEDformerLite (compact) is already covered by F_fedformer_sweep.py.")
else:
    exec(open(_src, encoding="utf-8").read())
