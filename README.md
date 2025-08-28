# AI Governance Grants Explorer

Public, non-commercial repository aggregating and analyzing Open Philanthropy grants with a focus on AI governance / safety.

> **License summary**  
> - **Data** (including derived CSVs): **CC BY-NC 4.0** (attribution + non-commercial).  
> - **Code** (`/analysis`): **MIT**.  
> - **Docs/Charts**: **CC BY 4.0** unless noted.

## What's here
- `/analysis/OpenPhil_Grants_Toolkit.py` — robust cleaner + analyzer (command-line & notebook-friendly).
- `/derived/` — publishable AI-focused extracts (CSV).
- `/charts/` — a couple of example figures.
- `/data/` — (empty) add `OpenPhilGrants.csv` here if you want the repo to include the raw export.

## Quickstart
```bash
# (A) Run analysis locally
python analysis/OpenPhil_Grants_Toolkit.py data/OpenPhilGrants.csv

# (B) Or run in a notebook
from analysis.OpenPhil_Grants_Toolkit import run_analysis
run_analysis('data/OpenPhilGrants.csv')
```

Outputs write to `data/openphil_grants_report/` by default (Excel + CSVs + PNGs).

## Attribution
Contains information © Open Philanthropy, used under **CC BY-NC 4.0**.  
Edits, analysis, and visualizations © 2025 Your Name.

See **ATTRIBUTION.md** and **LICENSE-DATA** for details.

## How to cite
If you mint a DOI (e.g., via Zenodo), update `CITATION.cff`. Example:
```
@dataset{your-id,
  title={AI Governance Grants Explorer},
  author={Your Name},
  year={2025},
  url={https://github.com/your/repo},
  version={v1.0}
}
```


## Reports
- Full generated outputs are under `/reports/openphil_grants_report/` (Excel, all charts, all CSV extracts, and a README_Analysis.md).
