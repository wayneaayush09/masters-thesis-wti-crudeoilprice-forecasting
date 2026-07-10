# Data sources and provenance

This repository does not redistribute the complete raw-data archive. Obtain source data directly from the relevant provider and record the download date and any transformation performed before running the pipeline.

| Local filename in the DDP1 workspace | Expected source / subject | Repository policy |
| --- | --- | --- |
| `DCOILWTICO.csv` | FRED WTI spot-price series | Download from FRED; do not rely on an unverified local copy. |
| `FRED dataset.csv` | FRED-derived economic time series | Document the underlying FRED series identifiers and retrieval date before sharing. |
| `PET_PRI_SPT_S1_M.xls` | Petroleum-price source data | Confirm the provider's licence and redistribution terms before committing. |
| `Crude_Oil_Data_24.csv` | Local crude-oil dataset snapshot | Preserve source URL, date range, and licence before sharing. |
| `crude_oil_2025_price.csv` | Local recent-price snapshot | Treat as a derived/local snapshot unless provenance is documented. |

The small `Cushing_OK_WTI_Spot_Price_FOB.csv` file is included in the repository for the existing scripts. Large source data, downloaded archives, and processed data remain excluded by `.gitignore`.
