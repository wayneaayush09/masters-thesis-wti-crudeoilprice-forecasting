# Data and generated artefacts

## Versioned data

`Cushing_OK_WTI_Spot_Price_FOB.csv` is the small WTI spot-price input distributed with this repository.

## Local-only data

The following files are intentionally excluded from Git because they are large or should be handled according to their data licence:

- `F_Disagg06_16.txt`
- `f_year.txt`
- any files under `data/raw/` and `data/processed/`

Place required source data in the repository root only when running the existing scripts, or update paths centrally before moving data into `data/`. Do not commit downloaded, proprietary, or access-controlled data without confirming redistribution rights.

## Generated artefacts

The pipeline writes intermediate datasets, caches, predictions, metrics, and figures to `outputs/`, `results/`, `cache/`, and `figures/`. They are excluded because they can be recreated from the code and may be too large for GitHub.

For a complete run order, see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).
