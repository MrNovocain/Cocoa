
import pandas as pd
import os

# The path from the user's active document
csv_path = r"w:\Research\NP\Cocoa\output\rolling_experiments\20251208_190830_2025-02-26_to_2021-12-31\rolling_results.csv"

def find_outlier():
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We look for high MSFE in WLL, RF, and XGB
    cols_to_check = ['msfe_wll', 'msfe_RF', 'msfe_XGb']
    
    print(f"Total rows: {len(df)}")
    
    for col in cols_to_check:
        if col not in df.columns:
            continue
            
        print(f"\n--- Checking {col} ---")
        max_val = df[col].max()
        max_row = df.loc[df[col].idxmax()]
        
        print(f"Max {col}: {max_val}")
        print(f"Date: {max_row['origin_date']}")
        print(f"Break Date detected: {max_row['detected_break_date']}")
        
        # Show top 3 just in case
        print(f"Top 3 {col}:")
        print(df.nlargest(3, col)[['origin_date', col]])

if __name__ == "__main__":
    find_outlier()
