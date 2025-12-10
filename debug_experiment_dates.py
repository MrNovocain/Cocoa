
from cocoa.models import CocoaDataset
import pandas as pd

def debug_dates():
    ds = CocoaDataset()
    
    # Logic from run_full_experiment.py
    last_date = ds.dates.max()
    start_index = ds.get_1_based_index_from_date(last_date)
    
    date_2021 = pd.Timestamp("2021-12-31")
    closest_date_2021 = ds.dates[ds.dates <= date_2021].max()
    end_index = ds.get_1_based_index_from_date(closest_date_2021)
    
    step = 5
    
    print(f"Start Index: {start_index} ({last_date})")
    print(f"End Index: {end_index} ({closest_date_2021})")
    print(f"Step: {step}")
    
    # Using RollingWLLExperiment logic
    # init logic: if start > end, step = -abs(step)
    if start_index > end_index:
        step = -abs(step)
        
    indices = list(range(start_index, end_index - 1, step))
    
    print(f"Total iterations: {len(indices)}")
    print("First 5 dates:")
    for i in indices[:5]:
        d = ds.get_date_from_1_based_index(i)
        print(f"  Idx {i}: {d}")
        
    print("Last 5 dates:")
    for i in indices[-5:]:
        d = ds.get_date_from_1_based_index(i)
        print(f"  Idx {i}: {d}")

if __name__ == "__main__":
    debug_dates()
