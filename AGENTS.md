# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/cocoa` (models, utils, experiments); experiment entry scripts in `src/cocoa/experiments`. Core constants live in `src/cocoa/models/assets.py`.
- Notebooks: `notebooks/` (e.g., `WLL_Cocoa_Experiment.ipynb` for the end-to-end workflow).
- Data & outputs: `data/` (`raw/`, `processed/`); results/plots under `output/` and `reports/`. Automated checks live in `tests/`.

## First-Run Commands & Reasoning
- Create venv (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1` � isolates dependencies.
- Install deps: `pip install -r requirements.txt` � matches the project�s pinned set.
- Editable install for notebook/CLI work: `pip install -e .` � allows importing `cocoa` modules during experimentation.

## Build, Test, and Development Commands
- Run tests: `pytest` � executes the test suite in `tests/`.
- Core experiments:
  - NP/ML convex combos: `python -m cocoa.experiments.run_np_combo_cv` (gamma sweep) or `run_ml_combo_cv` / `run_krr_combo_cv`.
  - Regressor subset search: `python -m cocoa.experiments.testing_regressors`.
  - Bias�variance sweep: `python -m cocoa.models.fitting_bvd`.

## Coding Style & Naming Conventions
- Python 3 with PEP8 (4-space indent, snake_case for functions/vars, CapWords for classes).
- Keep modules focused; prefer explicit imports. Add concise comments only where logic is non-obvious (e.g., break handling, MFV/BVD steps).

## Testing Guidelines
- Framework: `pytest`; name files `test_*.py` and mirror source layout when possible.
- Cover new logic (experiment runners, MFV/BVD helpers). Run `pytest` before opening a PR.

## Commit & Pull Request Guidelines
- Commits: imperative, specific summaries (e.g., "Add gamma sweep plot for NP combo CV"). Group related changes; reference issues when relevant.
- PRs: include what/why, mention key outputs (paths to plots/CSV), and cite test evidence (`pytest`). Add figure paths or screenshots when plots are generated.

## Security & Configuration Tips
- Keep `.env` private; base it on `.env.example` and adjust paths/keys locally.
- Avoid committing large data/outputs; write artifacts to `output/` or the configured experiment directories.

## Weighted Local Linear Estimation
- **Definition**: For each evaluation point `x0`, estimate `(beta0, beta1)` via `min_{beta0, beta1} sum_i K((x_i - x0)/h) * (y_i - beta0 - beta1 * (x_i - x0))^2`; the local prediction is `m_hat(x0) = beta0` with slope `beta1`, which captures first-order trends and reduces boundary bias relative to a constant local mean.
- **Hyperparameter tuning**: Use MFV (multi-fold validation) independently on the pre-break and post-break samples to select kernel, polynomial order, and bandwidth `h` for each regime so that each segment minimizes its validated weighted loss on its own domain.
- **Gamma weight**: After fitting the post-break regime, run MFV on post-break folds to choose the convex weight `gamma` that blends the post-break WLL estimator with its paired component (e.g., NP/ML combo), ensuring the weight is optimized for the post-break risk profile.
- **Breakdate selection**: Apply the Mohr structural break test as an intermediate screen to narrow candidate break dates; only test-approved dates enter the MFV sweep where the pre/post models are re-estimated on their respective domains and the final breakdate is the candidate minimizing the validated objective.

## Rolling WLL Experiment (not implemented)
- Direction: loop OOS origins from recent to past using args `start_idx`, `end_idx` (default last obs), `step`; move backward by `step`, and if the remaining span is shorter than `step`, run one final earliest-admissible origin.
- Per origin: train <= origin-1, test >= origin; run Mohr with trimming that excludes any break indices found in more-recent runs; record detected break (abs and origin-relative).
- Models: tune pre/post local linear via MFV; tune post-break gamma via MFV; baseline is either best ML by MSFE (RF/XGB) or fixed RF—log which rule is used.
- Metrics: MSFE_WLL, MSFE_base, `PI = 1 - MSFE_WLL / MSFE_base`, hit flag (PI > 0); optional CSE velocity/acceleration gaps are an extension.
- Plots/outputs: break vs origin; PI vs break with shading where PI > 0; hit rate scalar/curve; PI distribution (box/violin/hist); persist per-origin table and plots under `output/rolling_experiments/`.
- Guardrails: trimming grid cannot include previously detected break indices; handle `step` > remaining span gracefully; keep feature/target setup consistent with the notebook.
- Optional/TODO: mature-region tagging (stability band and recomputed metrics) and pooled DM test in that mature region.

## Understanding and alignment
- For complex task, always align with the user before making any progress.
