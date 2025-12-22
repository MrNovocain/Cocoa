# Cocoa Project (Python)

Research scaffold for studying cocoa prices, structural breaks, and nonparametric vs ML forecasting.

This repo currently focuses on validating the **CGS weighted local linear (WLL)** method on cocoa prices, and comparing it to standard machine learning models (Random Forest, XGBoost).

---

## Quick Start (Reproducibility)

**Goal**: Reproduce the core experimental finding: WLL outperforms ML baselines (Random Forest, XGBoost) during the recent "El Niño" structural break in cocoa prices (2023-2024).

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure the processed data file is present:
- Path: `data/processed/cocoa_ghana_full.csv`
- *Note: This file should be pre-generated from raw data. If missing, place strict raw data in `data/raw/` and run processing scripts.*

### 3. Run Reproduction Demo
Run the one-click reproduction script to generate the "Rank 1" evidence table and plot:

```bash
python reproduce_demo.py
```

**Expected Output**:
1.  **Break Detection**: The script runs Mohr-Selk and identifies the structural break around 2023/2024.
2.  **Performance Table**: A comparison showing WLL (OOS MSFE) < XGBoost/RF.
3.  **Plot**: A file `reproduction_results.png` is saved in the root directory, visualizing the error comparison.

---

## Full Rolling Experiment

To run the comprehensive rolling-window backtest (validating the model over multiple years, not just the single recent break):

```bash
# Entry point for the full backtest
python run_full_experiment.py
```

This will:
- Iterate backwards from the most recent data to 2021.
- For each step, detect breaks, train WLL/ML models, and record OOS errors.
- Save detailed logs and plots to `output/`.

**Main Notebook**: 
Alternatively, use `notebooks/WLL_Cocoa_Experiment.ipynb` for an interactive walkthrough of the full pipeline.

---

## Analysis: Gamma vs Break Date

To study how the optimal smoothing parameter ($\gamma$) changes with different potential break dates:

```python
# Usage Example
from cocoa.experiments.run_np_combo_cv import gamma_break_grid

detected_break = 6117
df_gamma = gamma_break_grid(
    start_index=detected_break - 3,
    end_index=detected_break + 3,
    jump_size=1,
    save_plots=True,
    output_dir="output/"
)
```

## Project Structure

- `src/cocoa/models/` – Implementations (NP, ML, Combo, Evaluation).
- `src/cocoa/experiments/` – Runners for WLL, Rolling, and Sensitivity experiments.
- `notebooks/` – Interactive analysis.
- `data/` – Dataset storage.
- `output/` – Experiment artifacts (Plots, CSVs).
