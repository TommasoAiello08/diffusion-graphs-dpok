# Scripts

Utility scripts for environment checks, figure generation, and HPC data transfer.

| Script | Purpose |
|--------|---------|
| `smoke_test_imports.py` | Verify Python dependencies load (no GPU training). Run after `pip install -r requirements.txt`. |
| `build_report_figures.py` | Regenerate PNG figures in `figures/` from local eval CSVs. |
| `pull_figures_from_hpc.sh` | `scp` image grids and snapshots from Bocconi HPC (requires SSH access). |

## Usage

```bash
# After installing dependencies
python scripts/smoke_test_imports.py

# Rebuild report figures (matplotlib only)
python scripts/build_report_figures.py

# Pull missing PNGs from cluster (edit USER/HOST inside script first)
bash scripts/pull_figures_from_hpc.sh
```
