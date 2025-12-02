# Cocoa Project (Python)

Research scaffold for studying cocoa prices, structural breaks, and
nonparametric vs ML forecasting.

This repo currently focuses on validating the Cai–Gao–Selk (CGS) weighted local linear (WLL) method on cocoa prices, and comparing it to standard machine learning models (Random Forest, XGBoost). A second phase will study how tree ensembles can be interpreted as adaptive local nonparametric smoothers.

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

## Core experiment: WLL vs ML on cocoa

Main notebook (recommended entry point):

- `notebooks/WLL_Cocoa_Experiment.ipynb`

What it does:

- builds / loads the processed cocoa dataset,
- detects a single structural break using Mohr–Selk (2020),
- configures the train / test (OOS) split,
- trains nonparametric benchmarks (Pre-Break LL, Post-Break LL, WLL),
- trains ML competitors (Random Forest, XGBoost) and their weighted combos,
- evaluates out-of-sample MSFE and cumulative squared error,
- runs Modified Diebold–Mariano tests,
- saves results and predictions under `output/experiment_results/`.

To reproduce the main results, activate the environment and run the notebook top to bottom. It assumes the processed cocoa CSV exists at the path in `src/cocoa/models/assets.py` (`PROCESSED_DATA_PATH`).

## Gamma vs break date: NP convex combo

File:

- `src/cocoa/experiments/run_np_combo_cv.py`

Key pieces:

- `gamma_break_grid(start_index, end_index, jump_size=1, save_plots=True, output_dir=...)`  
  sweeps candidate structural break indices, runs the NP convex combination model via `ConvexComboExperimentRunner`, and returns a DataFrame with
  - `break_index` (1-based index),
  - `break_date`,
  - `gamma` (optimal weight on the pre-break model),
  - `in_sample_cv_mse` (MFV score),
  - `oos_mse` (test MSE when available).

- When `save_plots=True` it saves a two-panel figure showing
  - gamma vs break date,
  - CV MSE vs break date
  as `gamma_and_cv_vs_break_date_<start>_<end>_<jump>.png` in `output_dir`.

Typical usage around the detected break (from the Mohr–Selk step, e.g. index 6117):

```python
from cocoa.experiments.run_np_combo_cv import gamma_break_grid

detected_break = 6117
df_gamma = gamma_break_grid(
    start_index=detected_break - 3,
    end_index=detected_break + 3,
    jump_size=1,
    save_plots=True,
    output_dir="w:/Research/NP/Cocoa/output",
)

print(df_gamma)
```

This is mainly for understanding how the optimal gamma moves when you slide the assumed break date.

## Project structure (high level)

- `src/cocoa/models/` – model implementations (NP, ML, combo, CV tools, evaluation).
- `src/cocoa/experiments/` – experiment runners and scripts, including
  - `runner.py` (generic and convex combo runners),
  - `run_np_combo_cv.py` (gamma vs break date analysis).
- `notebooks/` – Jupyter notebooks for end-to-end experiments:
  - `WLL_Cocoa_Experiment.ipynb` – main WLL vs ML experiment.
- `data/` – raw and processed cocoa data (paths configured in `assets.py`).
- `output/` – experiment outputs, figures, and saved prediction tables.
- `reports/` – LaTeX proposal / write-up.

In a later phase, RF/XGB “local smoother” diagnostics and interpretability helpers will live under a small module such as `src/cocoa/interpretability/` and a separate notebook (e.g. `notebooks/ML_Local_Smoother_Interpretation.ipynb`). For now, the core WLL and gamma-vs-break functionality is stable.
```
