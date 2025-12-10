
from cocoa.models import CocoaDataset
import pandas as pd

ds = CocoaDataset()
print(f"First date: {ds.dates.min()}")
print(f"Last date: {ds.dates.max()}")

# Find end of 2021 index
date_2021 = pd.Timestamp("2021-12-31")
# find closest date <= 2021-12-31
closest_date = ds.dates[ds.dates <= date_2021].max()
idx_2021 = ds.get_1_based_index_from_date(closest_date)
print(f"End 2021 Date: {closest_date}, Index: {idx_2021}")

# Last date index
last_date = ds.dates.max()
idx_last = ds.get_1_based_index_from_date(last_date)
print(f"Last Date: {last_date}, Index: {idx_last}")
