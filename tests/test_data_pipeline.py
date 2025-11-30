import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.data import ingest, preprocess, features


def test_load_cocoa_raw_parses_dates_and_sorts(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()
    csv_path = raw_dir / "cocoa_raw.csv"
    df = pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-01", "2020-01-02"],
            "log_price": [2.0, 1.0, 1.5],
        }
    )
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(ingest, "PATHS", SimpleNamespace(data_raw=raw_dir))

    loaded = ingest.load_cocoa_raw("cocoa_raw.csv")
    assert loaded.iloc[0]["date"].date().isoformat() == "2020-01-01"
    assert np.all(np.diff(loaded["date"].values.astype("datetime64[ns]")) >= np.timedelta64(0, "D"))


def test_preprocess_cocoa_creates_log_target_when_missing():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=4, freq="D"),
            "price": [100, 110, 90, 95],
        }
    )
    processed = preprocess.preprocess_cocoa(df)
    assert "log_price" in processed.columns
    assert processed["log_price"].notna().all()


def test_build_features_outputs_expected_columns(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    weather = pd.DataFrame(
        {
            "DATE": np.repeat(dates, 1),
            "STATION": ["A"] * len(dates),
            "TAVG": np.linspace(24, 26, len(dates)),
            "PRCP": [0.0, 1.0, 0.2, 0.0, 0.5, 0.0],
        }
    )
    weather.to_csv(raw_dir / "Ghana_data.csv", index=False)

    prices = pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "ICCO daily price (US$/tonne)": ["2,500", "2,550", "2,530", "2,520", "2,540", "2,560"],
        }
    )
    prices.to_csv(raw_dir / "Daily Prices_ICCO.csv", index=False)

    result = features.build_features(
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        reading_path="Ghana_data.csv",
        file_name="cocoa_test.csv",
    )

    expected_cols = {
        "PRCP_anom_mean",
        "TAVG_anom_mean",
        "log_price",
        "log_return",
        "log_return_forecast_target",
    }
    assert expected_cols.issubset(set(result.columns))
    assert (processed_dir / "cocoa_test.csv").exists()
    assert result["log_return_forecast_target"].notna().all()


