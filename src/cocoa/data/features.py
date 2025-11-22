import pandas as pd
import numpy as np
from pathlib import Path


raw_regressors = pd.read_csv('data/raw/Ghana_data.csv')
def _calculate_expanding_climatology(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates climatological means and anomalies using an expanding window to prevent look-ahead bias.

raw_price = pd.read_csv('data/raw/Daily Prices_ICCO.csv')
    For each day, the historical mean for that day-of-year (DOY) is calculated using all data *up to that point in time*.
    """
    weather_df = weather_df.sort_values("DATE").copy()
    weather_df["DOY"] = weather_df["DATE"].dt.dayofyear

    raw_regressors['DATE'] = pd.to_datetime(raw_regressors['DATE'])
    # Use an expanding window grouped by station and day-of-year to calculate means
    # This prevents using future data to calculate the mean for a given day.
    tavg_means = weather_df.groupby(["STATION", "DOY"])["TAVG"].expanding().mean().reset_index(name="TAVG_mean_expanding")
    prcp_means = weather_df.groupby(["STATION", "DOY"])["PRCP"].expanding().mean().reset_index(name="PRCP_mean_expanding")

    raw_regressors = raw_regressors.sort_values(['DATE','NAME'])
    # The 'level_2' index from expanding corresponds to the original DataFrame's index
    weather_df = weather_df.merge(
        tavg_means[["level_2", "TAVG_mean_expanding"]],
        left_index=True,
        right_on="level_2",
        how="left"
    ).drop(columns="level_2")
    weather_df = weather_df.merge(
        prcp_means[["level_2", "PRCP_mean_expanding"]],
        left_index=True,
        right_on="level_2",
        how="left"
    ).drop(columns="level_2")

    raw_regressors['PRCP'] = raw_regressors['PRCP'].fillna(0)
    # Calculate anomalies using the time-safe expanding means
    weather_df["TAVG_anom"] = weather_df["TAVG"] - weather_df["TAVG_mean_expanding"]
    weather_df["PRCP_anom"] = weather_df["PRCP"] - weather_df["PRCP_mean_expanding"]
    return weather_df

# print(raw_regressors.head())

    raw_regressors['DOY'] = raw_regressors['DATE'].dt.dayofyear
def build_features(raw_data_dir: Path, processed_data_dir: Path) -> pd.DataFrame:
    """
    Loads raw weather and price data, computes time-safe features, and returns a merged DataFrame.
    """
    # 1. Load and process weather data
    raw_regressors = pd.read_csv(raw_data_dir / 'Ghana_data.csv')
    raw_regressors['DATE'] = pd.to_datetime(raw_regressors['DATE'])
    raw_regressors = raw_regressors.sort_values(['DATE', 'NAME'])
    raw_regressors['PRCP'] = raw_regressors['PRCP'].fillna(0)

    print(raw_regressors.head())
    # Drop rows where TAVG is missing before calculating anomalies
    tavg = raw_regressors.loc[raw_regressors['TAVG'].notna()].copy()

    print(raw_regressors['TAVG'].isnull().sum())
    # Calculate anomalies safely
    tavg_anom = _calculate_expanding_climatology(tavg)

    tavg = raw_regressors.loc[raw_regressors['TAVG'].notna()].copy()
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
    raw_price["date"] = pd.to_datetime(raw_price["date"], dayfirst=True)
    raw_price["price_usd_tonne"] = (
        raw_price["price_usd_tonne"]
          .astype(str)
          .str.replace(",", "", regex=False)
          .astype(float)
    )
    raw_price = raw_price.sort_values("date").reset_index(drop=True)

    climatology = (
    tavg.groupby(["STATION", "DOY"], as_index=False)
    .agg(
        PRCP_mean=("PRCP", "mean"),
        TAVG_mean=("TAVG", "mean"),
        )
    )
    tavg = tavg.merge(
    climatology,
    on=["STATION", "DOY"],
    how="left",
    validate="many_to_one",
    )
    tavg["PRCP_anom"] = tavg["PRCP"] - tavg["PRCP_mean"]
    tavg["TAVG_anom"] = tavg["TAVG"] - tavg["TAVG_mean"]
    prices = raw_price.copy()
    prices["log_price"] = np.log(prices["price_usd_tonne"])
    prices["log_price_lagt"] = prices["log_price"].shift(1)

    print(tavg.head())
    # 3. Merge weather and price data
    ghana = agg_weather.rename(columns={"DATE": "date"})
    cocoa_ghana = pd.merge(
        ghana,
        prices[["date", "log_price", "log_price_lagt"]],
        on="date",
        how="inner"  # Use inner merge to keep only dates with both weather and price
    )
    cocoa_ghana = cocoa_ghana.sort_values("date").reset_index(drop=True)
    cocoa_ghana = cocoa_ghana.dropna(subset=["log_price_lagt"]) # Drop first row with NaN lag

    agg_weather = (
    tavg.groupby("DATE")
    .agg(
        PRCP_anom_mean=("PRCP_anom", "mean"),
        TAVG_anom_mean=("TAVG_anom", "mean"),
        PRCP_anom_std=("PRCP_anom", "std"),
        TAVG_anom_std=("TAVG_anom", "std"),
        N_stations=("STATION", "nunique"),
    )
    .reset_index(   
    )
)
    agg_weather['PRCP_anom_std'] = agg_weather['PRCP_anom_std'].fillna(0)
    agg_weather['TAVG_anom_std'] = agg_weather['TAVG_anom_std'].fillna(0)
    print(agg_weather.head())
    cocoa_ghana["log_return"] = cocoa_ghana["log_price"] - cocoa_ghana["log_price_lagt"]

    agg_weather = agg_weather.sort_values('DATE').reset_index(drop=True)
    # Save processed data
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    cocoa_ghana.to_csv(processed_data_dir / 'cocoa_ghana.csv', index=False)

    agg_weather.to_csv(processed_data_dir / 'agg_weather.csv', index=False)
    return cocoa_ghana

print(raw_regressors.head())
if __name__ == '__main__':
    # Example of how to run this script
    # Assumes the script is run from the project root 'w:\Research\NP\Cocoa\'
    # and the data is in 'data/raw'
    project_root = Path.cwd()
    raw_data_path = project_root / 'data' / 'raw'
    processed_data_path = project_root / 'data' / 'processed'
    
    print("Building features...")
    final_df = build_features(raw_data_path, processed_data_path)
    print("Feature building complete.")
    print("Saved processed data to 'data/processed/cocoa_ghana.csv'")
    print("Final DataFrame head:")
    print(final_df.head())



    raw_price = raw_price.rename(columns={
    "Date": "date",
    "ICCO daily price (US$/tonne)": "price_usd_tonne"
    })

    raw_price["date"] = pd.to_datetime(raw_price["date"], dayfirst=True)

    raw_price["price_usd_tonne"] = (
        raw_price["price_usd_tonne"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

raw_price = raw_price.sort_values("date").reset_index(drop=True)

    print(raw_price["price_usd_tonne"].dtype)

prices = raw_price.copy()

prices["log_price"] = np.log(prices["price_usd_tonne"])

t = 1

prices["log_price_lagt"] = prices["log_price"].shift(t)

prices_cgs = prices.iloc[:-t].copy()
print(prices_cgs.head())
ghana = pd.read_csv('data/processed/agg_weather.csv')

ghana = ghana.rename(columns={"DATE": "date"})

ghana['date'] = pd.to_datetime(ghana['date'])

cocoa_ghana = (
    ghana.merge(
        prices_cgs[["date", "log_price", "log_price_lagt"]],
        on="date",
        how="left"  # or "inner" depending on what you want
    )
)
cocoa_ghana = cocoa_ghana.sort_values("date").reset_index(drop=True)
cocoa_ghana = cocoa_ghana.dropna(subset=["log_price_lagt"])

cocoa_ghana["log_return"] = cocoa_ghana["log_price"] - cocoa_ghana["log_price_lagt"]


cocoa_ghana.to_csv('data/processed/cocoa_ghana.csv', index=False)
