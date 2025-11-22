# Cocoa Project (Python)

Research scaffold for studying cocoa prices, structural breaks, and
nonparametric vs ML forecasting.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# (optional) create a Jupyter kernel
python -m ipykernel install --user --name cocoa-project --display-name "Cocoa Project"
```

Copy `.env.example` to `.env` and adjust as needed.

Place your raw data file under `data/raw/` (e.g. `cocoa_raw.csv`), making sure
the date and price columns match what `config.py` expects.

Run the baseline experiment:

```bash
python -m cocoa.experiments.run_cocoa_baseline
```
