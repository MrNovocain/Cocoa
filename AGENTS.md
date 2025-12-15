# Repository Guidelines & Research Context

## 1. Research Narrative: "Glass-Box vs. Black-Box"
Our core objective is to benchmark the Cai-Gao-Sun (CGS) Weighted Local Linear (WLL) estimator against Machine Learning (ML) baselines (Random Forest, XGBoost) for cocoa forecasting.
- **The Conflict:** WLL is transparent ("Glass-Box") but linear. ML is flexible but opaque ("Black-Box").
- **The Empirical Reality (Verified):**
    - **Sensitivity:** WLL is "Structurally Honest." It jumps when the break date changes (100% correlation confirmed).
    - **Over-smoothing:** ML is "Stable." It smooths over break dates, potentially masking risk.
    - **Manual Failure:** Manually adding nonlinearity to WLL failed (0.00% improvement). The data rejects hand-crafted features.
- **The Goal:** Use WLL not just to compete, but to **AUDIT** ML. If WLL jumps and ML is flat, trust WLL's warning of structural uncertainty.

## 2. Project Structure & Module Organization
- Source: `src/cocoa` (models, utils, experiments); experiment entry scripts in `src/cocoa/experiments`. Core constants live in `src/cocoa/models/assets.py`.
- Notebooks: `notebooks/` (e.g., `WLL_Cocoa_Experiment.ipynb` for the end-to-end workflow).
- Data & outputs: `data/` (`raw/`, `processed/`); results/plots under `output/` and `reports/`. Automated checks live in `tests/`.

## 3. Key Findings (for Agents to Know)
1.  **Validation 1 (Sensitivity):** WLL behavior is dominated by the break date $\hat{T}_1$. Large MSFE jumps align perfectly with $\hat{T}_1$ shifts.
2.  **Validation 2 (Extension):** `Log_WLL` (Extension) $\approx$ `Std_WLL` (Baseline). Optimal $\beta \to 0$. Do not try to manually engineer nonlinearity again.
3.  **Validation 3 (Gamma):** Optimal $\gamma \to 0$ in modern samples, meaning WLL correctly discards pre-break history.

## 4. First-Run Commands & Reasoning
- Create venv (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1` – isolates dependencies.
- Install deps: `pip install -r requirements.txt` – matches the project’s pinned set.
- Editable install for notebook/CLI work: `pip install -e .` – allows importing `cocoa` modules during experimentation.

## 5. Build, Test, and Development Commands
- Run tests: `pytest` – executes the test suite in `tests/`.
- Core experiments:
  - Rolling WLL vs ML: `python src/cocoa/experiments/rolling_wll.py`
  - Validation Scripts: `python src/cocoa/experiments/verify_rolling_jumps.py`

## 6. Coding Style & Naming Conventions
- Python 3 with PEP8 (4-space indent, snake_case for functions/vars, CapWords for classes).
- Keep modules focused; prefer explicit imports. Add concise comments only where logic is non-obvious (e.g., break handling, MFV/BVD steps).

## 7. Weighted Local Linear Estimation (Theory)
- **Definition**: Local linear regression with a break-aware weight $\gamma$ blending pre- and post-break estimators.
- **Hyperparameter tuning**: MFV (multi-fold validation) for bandwidths and $\gamma$.
- **Breakdate selection**: Mohr structural break test.

## 8. Understanding and alignment
- For complex task, always align with the user before making any progress.

## 9. Advisor Notebook Strategy ("Lite" Version)
To present `WLL_Cocoa_Experiment final.ipynb` to Professor Cai:
1.  **The Process (Current):**
    - Load Data -> Describe Stats -> **Detect Break (Mohr-Selk)** -> Train WLL/ML independently -> Bandwidth/Gamma Tuning -> **Forecast (OOS)**.
2.  **The Simplify Plan:**
    - **Hide Code:** Use `nbconvert` or Jupyter extensions to hide input cells. Cai needs to see the *flow*, not the `import pandas`.
    - **Focus on 4 Key Plots:**
        1.  **The Break:** `gamma_vs_break_date` (shows volatility shift).
        2.  **The Funnel:** `oos_forecast` (shows WLL tracking the trend).
        3.  **The Comparison:** `msfe_results` table (Rank 1).
        4.  **The Speed:** `hypothesis_immediate_postbreak` (Cumulative SE).
    - **Narrative Headers:** Rename headers to be questions: "Does the volatility shift?", "Can WLL adapt?", "Is it robust?".
