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
