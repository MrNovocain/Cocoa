import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
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
    Plot full history of the price series and overlay model-implied price forecasts.

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
    if len(df) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(df)} rows in df vs {len(y_pred)} predictions."
        )

    plot_df = df[["date", "log_price"]].copy()
    plot_df["log_return_hat"] = y_pred

    # Reconstruct the predicted log price level
    # log_price_hat(t) = log_price(t-1) + log_return_hat(t)
    # The feature 'log_price_lagt' is log_price(t-1)
    if "log_price_lagt" not in df.columns:
        raise KeyError("The DataFrame must contain 'log_price_lagt' to reconstruct price forecasts.")
    plot_df["log_price_hat"] = df["log_price_lagt"] + plot_df["log_return_hat"]

    # Convert log prices to actual prices
    plot_df["price"] = np.exp(plot_df["log_price"])
    plot_df["price_hat"] = np.exp(plot_df["log_price_hat"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(plot_df["date"], plot_df["price"], label="Actual Price", linestyle="-", color="black", linewidth=1.0)
    ax.plot(plot_df["date"], plot_df["price_hat"], label=f"{model_label} Forecasted Price", linestyle="--", color="firebrick")

    ax.set_title(f"Cocoa Price: Actual vs. {model_label} Forecast", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)

    # Ensure the output directory exists before saving
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()