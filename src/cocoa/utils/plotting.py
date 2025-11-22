import matplotlib.pyplot as plt
import pandas as pd


def plot_series(y: pd.Series, title: str = "Cocoa price series") -> None:
    """Quick helper to visualize a time series."""
    ax = y.plot()
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y.name or "value")
    plt.tight_layout()
    plt.show()
