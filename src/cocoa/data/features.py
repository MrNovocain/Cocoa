import pandas as pd
import numpy as np
from pathlib import Path

def _calculate_expanding_climatology(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates climatological means and anomalies using an expanding window.
    
    IMPROVEMENTS:
    - Removes risky 'level_2' merging.
    - Uses direct index alignment for speed and safety.
    """
    # Ensure data is sorted and copied
    weather_df = weather_df.sort_values(["DATE", "STATION"]).copy()
    weather_df["DOY"] = weather_df["DATE"].dt.dayofyear

    # ---------------------------------------------------------
    # 1. Calculate Expanding Means
    # ---------------------------------------------------------
    # We group by Station and DOY. 
    # The expanding mean includes the current row, which is technically correct 
    # for a "deviation from experienced climatology" but often in anomaly detection
    # we want (Current - Mean_of_Previous). 
    # However, standard climatology usually includes the current year in the base 
    # if calculating a rolling norm. We will stick to your logic (Current included).
    
    indexer = ["STATION", "DOY"]
    
    # Calculate TAVG expanding mean
    # The result of groupby().expanding() creates a MultiIndex (STATION, DOY, Original_Index)
    tavg_exp = (
        weather_df.groupby(indexer)["TAVG"]
        .expanding()
        .mean()
    )
    
    # Calculate PRCP expanding mean
    prcp_exp = (
        weather_df.groupby(indexer)["PRCP"]
        .expanding()
        .mean()
    )

    # ---------------------------------------------------------
    # 2. Align and Assign
    # ---------------------------------------------------------
    # The resulting series (tavg_exp) has a MultiIndex. 
    # The last level of that MultiIndex matches the original weather_df index.
    # We drop the grouping keys (level 0 and 1) to align with weather_df.
    
    weather_df["TAVG_mean_expanding"] = tavg_exp.droplevel([0, 1])
    weather_df["PRCP_mean_expanding"] = prcp_exp.droplevel([0, 1])

    # ---------------------------------------------------------
    # 3. Calculate Anomalies
    # ---------------------------------------------------------
    weather_df["TAVG_anom"] = weather_df["TAVG"] - weather_df["TAVG_mean_expanding"]
    weather_df["PRCP_anom"] = weather_df["PRCP"] - weather_df["PRCP_mean_expanding"]

    return weather_df

def build_features(raw_data_dir: Path, processed_data_dir: Path) -> pd.DataFrame:
    # 1. Load and process weather data
    raw_regressors = pd.read_csv(raw_data_dir / 'Ghana_data.csv')
    raw_regressors['DATE'] = pd.to_datetime(raw_regressors['DATE'])
    
    # Fill PRCP NaNs
    raw_regressors['PRCP'] = raw_regressors['PRCP'].fillna(0)

    # Drop rows where TAVG is missing (essential for accurate means)
    tavg = raw_regressors.loc[raw_regressors['TAVG'].notna()].copy()

    # --- STEP 1: Calculate Anomalies Safely ---
    tavg_anom = _calculate_expanding_climatology(tavg)

    # --- IMPORTANT: HANDLE COLD START ---
    # The first few years of an expanding window have high variance/bias 
    # (e.g. Year 1 anomaly is always 0). 
    # It is highly recommended to drop the first 1-2 years of the data 
    # if your dataset is large enough.
    # Uncomment the line below to drop the first 2 years of data:
    # start_date = tavg_anom['DATE'].min() + pd.DateOffset(years=2)
    # tavg_anom = tavg_anom[tavg_anom['DATE'] > start_date]

    # Aggregate weather data by date
    agg_weather = (
        tavg_anom.groupby("DATE")
        .agg(
            PRCP_anom_mean=("PRCP_anom", "mean"),
            TAVG_anom_mean=("TAVG_anom", "mean"),
            PRCP_anom_std=("PRCP_anom", "std"),
            TAVG_anom_std=("TAVG_anom", "std"),
            N_stations=("STATION", "nunique"),
        )
        .reset_index()
    )
    
    agg_weather = agg_weather.fillna({'PRCP_anom_std': 0, 'TAVG_anom_std': 0})
    agg_weather = agg_weather.sort_values('DATE').reset_index(drop=True)

    # 2. Load and process price data
    raw_price = pd.read_csv(raw_data_dir / 'Daily Prices_ICCO.csv')
    raw_price = raw_price.rename(columns={
        "Date": "date",
        "ICCO daily price (US$/tonne)": "price_usd_tonne"
    })
    
    # Safe number conversion
    raw_price["price_usd_tonne"] = (
        raw_price["price_usd_tonne"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    # Coerce errors to NaN to handle non-numeric data, then drop
    raw_price["price_usd_tonne"] = pd.to_numeric(raw_price["price_usd_tonne"], errors='coerce')
    raw_price = raw_price.dropna(subset=["price_usd_tonne"])
    
    raw_price["date"] = pd.to_datetime(raw_price["date"], dayfirst=True)
    raw_price = raw_price.sort_values("date").reset_index(drop=True)

    prices = raw_price.copy()
    prices["log_price"] = np.log(prices["price_usd_tonne"])
    
    # 3. Lag Price Calculation
    # We want to predict using PAST info.
    # If we are at time T, we know Price(T-1).
    prices["log_price_lagt"] = prices["log_price"].shift(1)

    # 4. Merge
    ghana = agg_weather.rename(columns={"DATE": "date"})
    
    cocoa_ghana = pd.merge(
        ghana,
        prices[["date", "log_price", "log_price_lagt"]],
        on="date",
        how="inner" 
    )
    
    cocoa_ghana = cocoa_ghana.sort_values("date").reset_index(drop=True)
    cocoa_ghana = cocoa_ghana.dropna(subset=["log_price_lagt"])

    # Calculate Return
    cocoa_ghana["log_return"] = cocoa_ghana["log_price"] - cocoa_ghana["log_price_lagt"]

    # Save
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    cocoa_ghana.to_csv(processed_data_dir / 'cocoa_ghana.csv', index=False)

    return cocoa_ghana

if __name__ == '__main__':
    project_root = Path.cwd()
    raw_data_path = project_root / 'data' / 'raw'
    processed_data_path = project_root / 'data' / 'processed'
    
    # Create dummy data for testing if files don't exist
    if not (raw_data_path / 'Ghana_data.csv').exists():
        print("Note: Ensure input CSVs exist. Code is ready to run.")
    else:
        df = build_features(raw_data_path, processed_data_path)
        print(df.head())