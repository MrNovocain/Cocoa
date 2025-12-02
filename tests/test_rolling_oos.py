import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.experiments.rolling_oos import rolling_oos_window_msfe


def test_rolling_oos_window_msfe_non_overlapping():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    y_true = np.arange(6)
    baseline = y_true + 0.5
    tight = y_true + 0.1

    pred_df = pd.DataFrame(
        {
            "date": dates,
            "y_true": y_true,
            "baseline": baseline,
            "tight": tight,
        }
    )

    result = rolling_oos_window_msfe(
        pred_df,
        model_cols=["baseline", "tight"],
        start_index=2,
        window_length=2,
        baseline_model="baseline",
    )

    # After trimming to start_index=2 there are 5 rows; with window_length=2 the default
    # step creates three windows (2, 2, and 1 observations).
    assert result["window_id"].nunique() == 3
    assert len(result) == 6

    first_baseline = result[(result["window_id"] == 1) & (result["model"] == "baseline")].iloc[0]
    assert first_baseline["start_idx"] == 2
    assert first_baseline["end_idx"] == 3
    assert first_baseline["n_obs"] == 2
    assert np.isclose(first_baseline["msfe"], 0.25)

    last_tight = result[(result["window_id"] == 3) & (result["model"] == "tight")].iloc[0]
    assert last_tight["n_obs"] == 1
    assert np.isclose(last_tight["msfe"], 0.01)
    assert last_tight["rel_to_baseline_pct"] < 0  # tighter model should dominate baseline
