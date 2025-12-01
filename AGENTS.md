# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/cocoa` (models, utils, experiments); experiment entry scripts in `src/cocoa/experiments`. Core constants live in `src/cocoa/models/assets.py`.
- Notebooks: `notebooks/` (e.g., `WLL_Cocoa_Experiment.ipynb` for the end-to-end workflow).
- Data & outputs: `data/` (`raw/`, `processed/`); results/plots under `output/` and `reports/`. Automated checks live in `tests/`.

## First-Run Commands & Reasoning
- Create venv (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1` — isolates dependencies.
- Install deps: `pip install -r requirements.txt` — matches the project’s pinned set.
- Editable install for notebook/CLI work: `pip install -e .` — allows importing `cocoa` modules during experimentation.

## Build, Test, and Development Commands
- Run tests: `pytest` — executes the test suite in `tests/`.
- Core experiments:
  - NP/ML convex combos: `python -m cocoa.experiments.run_np_combo_cv` (gamma sweep) or `run_ml_combo_cv` / `run_krr_combo_cv`.
  - Regressor subset search: `python -m cocoa.experiments.testing_regressors`.
  - Bias–variance sweep: `python -m cocoa.models.fitting_bvd`.

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