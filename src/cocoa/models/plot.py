import pandas as pd
from matplotlib import pyplot as plt
import os


def plot_forecast(
    df: pd.DataFrame,
    target_col: str,
    y_pred,
    oos_start_date=None,
    model_label: str = "Model",
    output_path: str = "data/processed/point_forecast.png",
):
    """
    Plot full history of the target series and overlay model forecasts.

    Parameters
    ----------
    df : DataFrame
        Full cocoa DataFrame with 'date' and target_col.
    target_col : str
        Name of the target variable column in df.
    y_pred : array-like
        Model predictions over the full sample (must align with dates in df).
    oos_start_date : str or Timestamp
        First date included in the OOS test window (same date you used for splitting).
        If provided, the plot will distinguish between in-sample and out-of-sample forecasts.
    model_label : str
        Label for the prediction line in the legend.
    output_path : str
        Where to save the PNG.
    """
    # Full series
    all_dates = df["date"]
    all_y = df[target_col]

    if len(all_dates) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(all_dates)} dates in df vs {len(y_pred)} predictions."
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(all_dates, all_y, label=f"Actual {target_col} (Full History)", linestyle="-", color="black", linewidth=1.0)
    ax.plot(all_dates, y_pred, label=f"{model_label} Point Forecast (Full Sample)", linestyle="--", color="firebrick")

    ax.set_title(f"Cocoa {target_col}: Full History and Forecast", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)

    # Ensure the output directory exists before saving
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()