# DDP v2 — WTI crude-oil forecasting thesis

This repository contains the code supporting a master's thesis on crude-oil price forecasting. The project compares classical, neural, decomposition-based, foundation-model, text/RAG, probabilistic, and tail-risk approaches across short and long forecasting horizons.

The research design explicitly contrasts two tracks:

- **Track A:** reproductions of decomposition-based architectures reported in the literature.
- **Track B:** leakage-safe walk-forward implementations used for the primary empirical evaluation.

The full methodological rationale is in [THESIS_METHODOLOGY.md](THESIS_METHODOLOGY.md), while execution instructions are in [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

## Repository layout

The current flat code layout is intentional: the experiment scripts load shared modules by filename and are runnable from the repository root. It will be refactored only after imports are converted to package imports.

| Contents | Purpose |
| --- | --- |
| `00_*.py`–`17_*.py` | Numbered stages of the data, modelling, risk, and audit pipeline |
| `A_*.py`, `B_*.py`, `C_*.py` | Track A, Track B, and comparison/ensemble experiments |
| `_*.py` | Shared helpers and model implementations |
| `run_*.py` | Local, cloud/GPU, and phase orchestrators |
| `*_GUIDE.md`, `THESIS_METHODOLOGY.md` | Thesis methodology and run documentation |
| `outputs/`, `results/`, `cache/` | Generated artefacts; deliberately excluded from Git |

## Quick start

1. Use Python 3.10 or later and create an isolated environment.
2. Install the base dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Supply the local data described in [DATA.md](DATA.md). The WTI spot-price CSV is included; large source files and generated data are intentionally not versioned.
4. Start with the staged workflow in [RUNNING_GUIDE.md](RUNNING_GUIDE.md). For the CPU preparation phase, run:

   ```powershell
   python run_local_cpu.py
   ```

## Reproducibility and data

Generated results, model artefacts, caches, and large/local data are excluded from version control. This keeps the repository lightweight and prevents accidental publication of data that may have licensing, access, or size restrictions. Record exact data provenance and execution settings when producing thesis tables and figures.

## Status

This is an active research repository. It is best kept private until the thesis has been examined and any third-party data licence requirements have been checked.
