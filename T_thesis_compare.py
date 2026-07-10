# ============================================================
# DDP_v2/T_thesis_compare.py
# Master head-to-head table: thesis-reported vs our leaky-repro vs our-honest.
# Surfaces the key academic finding: thesis numbers are scale/leakage-inflated.
# Reads all eval_*.pkl from outputs/, matches Thesis* + our models, prints and
# saves thesis_comparison.csv.
# Run from inside DDP_v2/ after T_thesis_leaky_repro and T_thesis_honest.
# ============================================================
import os, glob, pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = globals().get("OUTPUT_DIR", Path("outputs"))    # Path object — DO NOT change to str (breaks / operator)

# ── Thesis-reported results (Table XX, Mahajan 2024) ──────────
THESIS_REPORTED = {
    "NBEATSLite": {"mse_norm": 0.0002, "mae_norm": 0.0092, "scale": "normalized[0,1]"},
    "ILSTMCnn":   {"mse_norm": 0.0005, "mae_norm": 0.0150, "scale": "normalized[0,1]"},
    "ESN":        {"mse_norm": 0.0017, "mae_norm": 0.0111, "scale": "normalized[0,1]"},
    "LSTNet":     {"mse_norm": 7.989,  "mae_norm": 1.552,  "scale": "raw_dollar"},
    "ILSTM":      {"mse_norm": 9.429,  "mae_norm": 1.811,  "scale": "raw_dollar"},
}
NOTE = ("⚠️  N-BEATS/ESN/ILSTMCnn metrics in thesis are on normalized[0,1] scale."
        " LSTNet/ILSTM are raw-$ scale. Mixed comparison in their Table XX.")

def _load_eval(pattern):
    rows = []
    for fp in sorted(glob.glob(str(OUTPUT_DIR / pattern))):
        try:
            with open(fp, "rb") as f: ev = pickle.load(f)
        except Exception: continue
        if not isinstance(ev, dict): continue
        model = ev.get("model", os.path.basename(fp)[5:-4])
        hk = ev.get("horizon")
        if not hk:
            for h in ("180d","90d","7d","1d"):
                if fp.endswith(f"_{h}.pkl"): hk = h; break
        ov = ev.get(f"Overall_{hk}", {})
        mae, rmse, mape = ov.get("MAE"), ov.get("RMSE"), ov.get("MAPE")
        if mae is None: continue
        rows.append({
            "model": model, "horizon": hk,
            "MAE_dollar": round(float(mae),4),
            "RMSE_dollar": round(float(rmse),4) if rmse else np.nan,
            "MAPE_pct": round(float(mape),2) if mape else np.nan,
            "leaky": ev.get("leaky"),
            "mae_normalized": ev.get("mae_normalized"),
            "protocol": ev.get("protocol",""),
        })
    return rows

# Load our leaky repro
leaky_rows = _load_eval("eval_ThesisLeaky_*.pkl")
# Load our honest walk-forward for thesis archs
honest_rows = [r for r in _load_eval("eval_*.pkl")
               if any(a in r["model"] for a in THESIS_REPORTED) and not r["model"].startswith("ThesisLeaky")]
# Load our own best models for context
our_rows = [r for r in _load_eval("eval_*.pkl")
            if not any(a in r["model"] for a in THESIS_REPORTED) and "Leaky" not in r["model"]]

all_rows = leaky_rows + honest_rows + our_rows

if not all_rows:
    print("⚠️  No eval_*.pkl found. Run T_thesis_leaky_repro.py and T_thesis_honest.py first.")
else:
    df = pd.DataFrame(all_rows)
    # Sort by horizon then MAE
    _ord = {"1d":0,"7d":1,"90d":2,"180d":3}
    df["_h"] = df["horizon"].map(_ord).fillna(9)
    df = df.sort_values(["_h","MAE_dollar"]).drop(columns="_h")
    df.to_csv(OUTPUT_DIR / "thesis_comparison.csv", index=False)

    # Pretty-print with thesis-reported alongside
    print("\n" + "="*90)
    print("  THESIS vs OURS — Head-to-Head Comparison (Mahajan IIT-B 2024 vs our pipeline)")
    print(f"  {NOTE}")
    print("="*90)

    for hk in globals().get("TRACK_HORIZONS", ["1d","7d","90d","180d"]):
        sub = df[df["horizon"]==hk]
        if sub.empty: continue
        print(f"\n  ── Horizon {hk} ──")
        print(f"  {'Model':<36} {'Protocol':<28} {'MAE($)':<10} {'MAPE%':<8} {'mae_norm':<10}")
        print(f"  {'─'*36} {'─'*28} {'─'*10} {'─'*8} {'─'*10}")
        # thesis reported
        for arch, tr in THESIS_REPORTED.items():
            print(f"  {'THESIS_reported_'+arch:<36} {'leaky_whole_MinMax (paper)':<28} "
                  f"{'?':10} {'?':8} {tr['mae_norm']:<10.4f}  [{tr['scale']}]")
        # our models
        for _, row in sub.iterrows():
            mn = f"{row['mae_normalized']:.6f}" if pd.notna(row.get("mae_normalized")) else "—"
            print(f"  {row['model']:<36} {str(row['protocol'])[:28]:<28} "
                  f"{row['MAE_dollar']:<10.4f} {(str(round(row['MAPE_pct'],1))+'%') if pd.notna(row['MAPE_pct']) else '—':8} {mn:<10}")

    print(f"\n  ✅ Saved outputs/thesis_comparison.csv  ({len(df)} rows)")
    print("\n  KEY FINDING:")
    print("  • Thesis best (N-BEATS MAE=0.0092) is on normalized[0,1] scale, 1-step, whole-series")
    print("    MinMax BEFORE split → leakage inflates the number.")
    print("  • Our leaky-repro should confirm ≈ same normalized MAE for N-BEATS.")
    print("  • Our honest walk-forward $-scale MAE is the true comparison baseline.")
