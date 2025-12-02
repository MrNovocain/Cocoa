"""Rolling OOS evaluation helpers inspired by the analysis notebook."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from cocoa.models.evaluation import mean_squared_error


def rolling_oos_window_msfe(
    pred_df: pd.DataFrame,
    model_cols: Iterable[str],
    start_index: int,
    window_length: int,
    baseline_model: Optional[str] = "WLL",
    step: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute rolling (or chunked) MSFE slices over the prediction horizon.

    Parameters
    ----------
    pred_df:
        DataFrame with a ``date`` column, a ``y_true`` column, and one column per model prediction.
    model_cols:
        Iterable of column names in ``pred_df`` to score.
    start_index:
        1-based index indicating where to begin the evaluation window.
    window_length:
        Number of observations per window.
    baseline_model:
        Model name used for relative MSFE (%). Set to ``None`` to skip.
    step:
        How far to shift the window start each iteration. Defaults to ``window_length``
        (non-overlapping windows, matching the notebook); set to ``1`` for a fully rolling view.
    """
    if window_length <= 0:
        raise ValueError("window_length must be positive.")

    step_size = window_length if step is None else step
    if step_size <= 0:
        raise ValueError("step must be positive.")

    required_cols = {"date", "y_true"}
    missing_required = required_cols - set(pred_df.columns)
    if missing_required:
        raise ValueError(f"pred_df is missing required columns: {missing_required}")

    missing_models = [m for m in model_cols if m not in pred_df.columns]
    if missing_models:
        raise ValueError(f"pred_df is missing model prediction columns: {missing_models}")

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_index < 1 or start_index > len(df):
        raise ValueError(f"start_index must be between 1 and {len(df)}, got {start_index}.")

    df = df.iloc[start_index - 1 :].reset_index(drop=True)
    base_offset = start_index - 1

    rows: List[dict] = []
    window_id = 1
    for window_start in range(0, len(df), step_size):
        window_end = min(window_start + window_length, len(df))
        if window_start >= len(df):
            break

        y_slice = df["y_true"].iloc[window_start:window_end].reset_index(drop=True)
        if y_slice.empty:
            continue

        for model in model_cols:
            preds_slice = df[model].iloc[window_start:window_end].reset_index(drop=True)
            msfe_val = mean_squared_error(y_slice, preds_slice)
            rows.append(
                {
                    "window_id": window_id,
                    "start_idx": base_offset + window_start + 1,
                    "end_idx": base_offset + window_end,
                    "start_date": df["date"].iloc[window_start],
                    "end_date": df["date"].iloc[window_end - 1],
                    "n_obs": len(y_slice),
                    "model": model,
                    "msfe": msfe_val,
                }
            )

        window_id += 1
        if window_end == len(df):
            break

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    if baseline_model is not None:
        if baseline_model not in model_cols:
            raise ValueError(f"baseline_model '{baseline_model}' is not in model_cols.")

        baseline = (
            results[results["model"] == baseline_model][["window_id", "msfe"]]
            .rename(columns={"msfe": "baseline_msfe"})
            .copy()
        )
        results = results.merge(baseline, on="window_id", how="left")
        denom = results["baseline_msfe"].replace({0: np.nan})
        results["rel_to_baseline_pct"] = (results["msfe"] / denom - 1) * 100
        results.drop(columns=["baseline_msfe"], inplace=True)
    else:
        results["rel_to_baseline_pct"] = np.nan

    return results.sort_values(["window_id", "model"]).reset_index(drop=True)
