
import pandas as pd
import os

csv_path = r"w:\Research\NP\Cocoa\output\rolling_experiments\20251208_190830_2025-02-26_to_2021-12-31\rolling_results.csv"

def check_trimmed():
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for trimmed flags
    trimmed = df[df['trimmed_flag'] == True]
    
    print(f"Total rows: {len(df)}")
    print(f"Trimmed rows: {len(trimmed)}")
    
    if len(trimmed) > 0:
        # Sort by date
        trimmed = trimmed.sort_values('origin_date')
        
        print("\n--- Trimmed Flag Analysis ---")
        print(f"Earliest Date (Chronological): {trimmed.iloc[0]['origin_date']}")
        print(f"Latest Date (Chronological):   {trimmed.iloc[-1]['origin_date']}")
        
        # Scan backward context (Recent -> Past)
        # The loop runs from Recent to Past. 
        # So the "first" flag encountered in the loop is the Latest Chronological date.
        # The "last" flag encountered in the loop (deepest history) is the Earliest Chronological date.
        print("\nScan Backward (Recent -> Past) view:")
        print(f"First encounter (Most Recent): {trimmed.iloc[-1]['origin_date']}")
        print(f"Last encounter (Oldest):       {trimmed.iloc[0]['origin_date']}")
        
        print("\nAll Trimmed Dates:")
        print(trimmed['origin_date'].tolist())
        
        print("\nAssociated Break Dates for these trimmed records:")
        print(trimmed['detected_break_date'].unique())

if __name__ == "__main__":
    check_trimmed()
