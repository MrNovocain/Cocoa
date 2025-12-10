
import pandas as pd
import os

csv_path = r"w:\Research\NP\Cocoa\output\rolling_experiments\20251208_114025_2025-02-26_to_2021-12-31\rolling_results.csv"

if os.path.exists(csv_path):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Rows: {len(df)}")
        print("Columns:", df.columns.tolist())
        print("\nNull Counts:")
        print(df.isnull().sum())
        
        print("\nSample Data (first 5):")
        print(df.head())
        
        if 'pi' in df.columns:
            print("\nPI Stats:")
            print(df['pi'].describe())
            
        if 'msfe_wll' in df.columns:
             print("\nMSFE WLL Stats:")
             print(df['msfe_wll'].describe())
             
        if 'msfe_RF' in df.columns:
             print("\nMSFE RF Stats:")
             print(df['msfe_RF'].describe())
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print("CSV not found.")
