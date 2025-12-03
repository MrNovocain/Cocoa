# Rolling WLL Experiment Plan (Not Implemented)

**Rolling Experiment Definition**
- Direction: iterate OOS origins from recent -> past.
- Inputs: `start_idx` (most-recent OOS origin, 0-based), `end_idx` (default = last observation), `step` (advance size; 1 = every day; n = every n-th origin).
- Stepping rule: move backward by `step` until you would cross below `end_idx`; if the remaining span is shorter than `step`, run one final iteration at the earliest admissible origin and stop (no overshoot).
- Per-iteration data: for each origin `t`, define train <= `t-1`, test >= `t` (consistent with the notebook split), and rerun the full WLL vs ML pipeline.

**Mohr Break Detection with Trimming Constraint**
- Maintain a set of break indices detected in more-recent iterations.
- For current origin, configure Mohr's trimmed sample and candidate grid so no candidate break index touches any previously detected break index (i.e., trim boundaries exclude that set).
- Run Mohr; record detected break (absolute index + date, and origin-relative index).

**Models and Validation**
- Pre-break and post-break local linear models tuned via MFV on their respective regimes for that origin.
- WLL gamma tuned via MFV on the post-break regime.
- Baseline `b`: either (a) auto-select best ML (RF/XGB) by MSFE for that origin or (b) fixed RF for determinism; choose one and log which rule was used.

**Metrics per Origin**
- Break path: detected break index and date.
- Performance: MSFE_WLL, MSFE_baseline; percent improvement `PI = 1 - MSFE_WLL / MSFE_b`; hit flag `PI > 0`.
- Optional dynamics: CSE velocity/acceleration gaps (first/second differences of post-break cumulative SE) if we want them now; otherwise note as possible extension.

**Plots/Outputs**
- Break stabilization: break index vs OOS origin (reverse time). Add rolling variance band; no mature-region gating in code yet.
- Improvement vs break: PI vs detected break; shade where PI > 0.
- Hit rate: aggregate fraction of origins with PI > 0 (single scalar) plus optional running hit-rate curve.
- Distribution of PI: box/violin/hist over PI; restrict to all origins for now. (Extension: restrict to the mature region once defined.)
- Artifacts: per-origin table/CSV with origin index/date, detected break (abs/relative), MSFEs, PI, hit flag, baseline choice; plots saved to `output/rolling_experiments/`.

**Guardrails**
- Trimming must exclude any previously detected break indices from the candidate grid.
- Handle step > remaining span gracefully (final iteration at earliest admissible origin).
- Consistent feature/target setup per the original notebook.

**Optional/Not Implemented Yet**
- Mature-region tagging (break stability within a band of size k for at least m consecutive origins) and recomputed hit rate/average improvement only within that tag.
- Global DM test over pooled errors in the mature region.

## Implementation Skeleton (Not Implemented)
- Module placement: `src/cocoa/experiments/run_rolling_wll.py` (runner with CLI) plus optional helpers (e.g., `rolling_wll_utils.py`).
- Config/data prep: parse `start_idx`, `end_idx` (default last obs), `step`, baseline mode (`auto_best_ml` or `fixed_rf`), output dir, plot flags. Load processed data and features via existing notebook helpers; build train/test masks per origin.
- Rolling controller: generate origins from `start_idx` down to `end_idx` stepping backward by `step`, with a final earliest-admissible origin when remaining span < step. Maintain `prior_breaks` (set of detected breaks from more-recent runs).
- Per-origin pipeline:
  1) Trim + Mohr: build candidate grid excluding `prior_breaks`; run Mohr; record break (absolute and origin-relative).
  2) Model fit: tune pre/post local linear via MFV; tune post-break gamma via MFV; fit baseline(s) (RF/XGB) and pick baseline per mode.
  3) Metrics: MSFE_WLL, MSFE_base; `PI = 1 - MSFE_WLL / MSFE_base`; hit flag (`PI > 0`); optional CSE velocity/acceleration deltas.
  4) Persist row: append origin info, break info, baseline choice, metrics to a DataFrame.
- Post-loop aggregation: write per-origin table to `output/rolling_experiments/rolling_results_<timestamp>.csv`; derive overall hit rate and average PI; keep mature-region tagging as a TODO hook.
- Plots: break path vs origin (reverse time, with rolling variance band); PI vs break with shading where PI > 0; hit rate scalar/curve; PI distribution (box/violin/hist) over all origins (mature-region filter is a TODO hook). Save to the same output folder.
- Guardrails: trimming grid must exclude `prior_breaks`; handle `step` > remaining span with a final iteration; keep feature/target setup identical to the notebook utilities.
