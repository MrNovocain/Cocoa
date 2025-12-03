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

## Revised Mohr Handling (Planned, Not Implemented)
- Pilot-only Mohr: run a single Mohr test on the full dataset up front to define a candidate break date `T_*`.
- Rolling stop rule: in the backward loop, stop if `T_*` would be trimmed by the current origin or if `end_idx` is reached.
- Candidate set: keep the prior multi-candidate logic dormant; for now, use `T_*` as the only candidate so we can toggle approaches later without removal.

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

## Pseudocode Sketch
```
function run_rolling_wll(
    start_idx,
    end_idx=None,
    step=1,
    baseline_mode="auto_best_ml",   # or "fixed_rf"
    output_dir
):
    data = CocoaDataset(processed_path, FEATURE_COLS, TARGET_COL)
    dates = data.dates
    if end_idx is None:
        end_idx = len(dates) - 1

    origins = build_descending_origins(start_idx, end_idx, step)
    prior_breaks = set()
    rows = []

    for origin_idx in origins:
        oos_start_date = dates.iloc[origin_idx]

        # 1) Mohr with trimming that excludes prior breaks; halt if no grid remains
        candidate_grid = build_candidate_grid_excluding(prior_breaks, data, origin_idx)
        if candidate_grid is None or candidate_grid.empty:
            break  # avoid trimming away prior detected breaks (potential convergence point)
        detected_break = estimate_break_mohr_ll(df=data.df, candidate_grid=candidate_grid)
        prior_breaks.add(detected_break)
        break_1based = detected_break + 1

        # 2) WLL via ConvexComboExperimentRunner
        wll_runner = ConvexComboExperimentRunner(
            combo_type="NP",
            model_name="WLL",
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            data_path=processed_path,
            oos_start_date=oos_start_date,
            sample_start_index=break_1based,
            poly_order=1,
            save_results=False,
        )
        wll_out = wll_runner.run()
        msfe_wll = wll_out["oos_mse"]

        # 3) Baselines discovered per-origin
        rf_out = ExperimentRunner(
            model_name="RF",
            model_class=RFModel,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            data_path=processed_path,
            oos_start_date=oos_start_date,
            save_results=False,
        ).run()
        xgb_out = ExperimentRunner(
            model_name="XGB",
            model_class=XGBModel,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            data_path=processed_path,
            oos_start_date=oos_start_date,
            save_results=False,
        ).run()
        msfe_rf, msfe_xgb = rf_out["oos_mse"], xgb_out["oos_mse"]

        if baseline_mode == "fixed_rf":
            baseline_name, msfe_base = "RF", msfe_rf
        else:
            baseline_name, msfe_base = min([("RF", msfe_rf), ("XGB", msfe_xgb)], key=lambda kv: kv[1])

        # 4) Metrics
        pi  = 1 - msfe_wll / msfe_base
        hit = pi > 0

        # 5) Log row
        rows.append({
            "origin_idx": origin_idx,
            "origin_date": oos_start_date,
            "detected_break_idx": detected_break,
            "detected_break_date": dates.iloc[detected_break],
            "baseline": baseline_name,
            "msfe_wll": msfe_wll,
            "msfe_base": msfe_base,
            "pi": pi,
            "hit": hit,
        })

    # 6) Persist
    df = DataFrame(rows).sort_values("origin_idx")
    save_csv(df, output_dir / f"rolling_results_{timestamp()}.csv")

    # 7) Plots (shade hit region)
    plot_break_vs_origin(df["origin_idx"], df["detected_break_idx"], outfile=output_dir/"break_path.png", reverse_time=True, variance_band=True)
    plot_pi_vs_break(break_idx=df["detected_break_idx"], pi=df["pi"], hit=df["hit"], outfile=output_dir/"pi_vs_break.png")
    plot_hit_rate(hits=df["hit"], outfile=output_dir/"hit_rate.png")
    plot_pi_distribution(pi=df["pi"], outfile=output_dir/"pi_distribution.png")

    return df
```
