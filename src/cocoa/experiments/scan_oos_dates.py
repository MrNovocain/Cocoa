
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from cocoa.models.cocoa_data import CocoaDataset
from cocoa.experiments.break_detection import MohrRunner

def scan_oos_dates():
    # Load dataset to inspect dates
    ds = CocoaDataset()
    last_date = ds.dates.iloc[-1]
    print(f"Dataset ends at: {last_date.date()}")
    
    # Define scan range: Last 2 years, every 14 days
    # We want OOS Start Date to be the "end of training".
    # So we pick dates from (End - 2 years) to (End - 30 days)
    
    end_scan = last_date - pd.Timedelta(days=30)
    start_scan = last_date - pd.Timedelta(days=365*2)
    
    scan_dates = pd.date_range(start=start_scan, end=end_scan, freq='28D') # Every 4 weeks to be faster
    
    results = []
    
    print(f"Scanning {len(scan_dates)} OOS start dates from {start_scan.date()} to {end_scan.date()}...")
    
    for oos_date in tqdm(scan_dates):
        try:
            # MohrRunner takes OOS Start Date, and uses data BEFORE it for training/break detection
            runner = MohrRunner(oos_start_date=oos_date)
            
            # Suppress prints for cleaner output
            # (Optional: redirect stdout if needed, but tqdm might conflict)
            break_idx = runner.run_mohr_break_detection()
            
            break_date = ds.get_date_from_1_based_index(break_idx)
            
            # Calculate training end index (1-based)
            # The split includes everything BEFORE oos_start_date.
            # Train indices: 0 to test_start_idx-1
            # Length = test_start_idx
            # So 1-based index of last training point is test_start_idx.
            test_start_idx_0based = ds.get_1_based_index_from_date(oos_date) - 1
            train_end_idx_1based = test_start_idx_0based # effectively the count
            
            t_post = train_end_idx_1based - break_idx
            
            results.append({
                "oos_date": oos_date,
                "break_date": break_date,
                "t_post": t_post,
                "break_idx": break_idx
            })
            
        except Exception as e:
            print(f"Error for date {oos_date.date()}: {e}")
            
    # Convert to DataFrame and display sorted by T_post
    res_df = pd.DataFrame(results)
    print("\n--- Scan Results ---")
    if not res_df.empty:
        # Filter for "interesting" cases: recent breaks (small T_post)
        # But also show general trend
        res_df = res_df.sort_values("oos_date")
        print(res_df)
        
        print("\n--- Candidates for Log NP (Small T_post) ---")
        candidates = res_df[res_df['t_post'] < 500].sort_values("t_post")
        print(candidates)
        
        # Save to csv for inspection
        output_path = Path(__file__).resolve().parent / "oos_scan_results.csv"
        res_df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    scan_oos_dates()
