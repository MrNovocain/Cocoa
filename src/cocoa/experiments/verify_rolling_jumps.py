import pandas as pd
import numpy as np

# Path to the rolling results CSV
csv_path = r"w:\Research\NP\Cocoa\output\rolling_experiments\20251208_190830_2025-02-26_to_2021-12-31\rolling_results.csv"

def verify_jumps():
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Sort by origin date (descending usually in file, but let's ensure linear time order for diff)
    df['origin_date'] = pd.to_datetime(df['origin_date'])
    df = df.sort_values('origin_date')
    
    # Calculate changes
    df['break_shifted'] = df['detected_break_date'].shift(1)
    df['break_changed'] = df['detected_break_date'] != df['break_shifted']
    
    df['msfe_wll_shifted'] = df['msfe_wll'].shift(1)
    df['msfe_jump'] = (df['msfe_wll'] - df['msfe_wll_shifted']).abs()
    
    # Define "Large Jump"
    # Let's look at the distribution of jumps
    jump_mean = df['msfe_jump'].mean()
    jump_std = df['msfe_jump'].std()
    threshold = jump_mean + 2 * jump_std
    
    print(f"Jump Threshold (Mean + 2*Std): {threshold:.6f}")
    
    large_jumps = df[df['msfe_jump'] > threshold]
    
    print(f"\nTotal Time Steps: {len(df)}")
    print(f"Total Break Changes: {df['break_changed'].sum()}")
    print(f"Total Large MSFE Jumps: {len(large_jumps)}")
    
    # Coincidence
    jumps_due_to_break = large_jumps[large_jumps['break_changed']]
    
    print(f"Large Jumps coincide with Break Change: {len(jumps_due_to_break)}")
    print(f"Correlation Percentage: {len(jumps_due_to_break) / len(large_jumps) * 100:.2f}%")
    
    if len(jumps_due_to_break) > 0:
        print("\nExample Coincidences:")
        print(jumps_due_to_break[['origin_date', 'detected_break_date', 'break_shifted', 'msfe_wll', 'msfe_jump']].head())
        
    # Reverse check: Do break changes always cause jumps?
    break_changes = df[df['break_changed']]
    break_changes_with_jump = break_changes[break_changes['msfe_jump'] > threshold]
    print(f"\nBreak Changes that caused Large Jump: {len(break_changes_with_jump)} / {len(break_changes)}")

if __name__ == "__main__":
    verify_jumps()
