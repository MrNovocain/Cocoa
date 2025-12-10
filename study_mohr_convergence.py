
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from cocoa.models import CocoaDataset
from cocoa.experiments.break_detection import MohrRunner

def run_study():
    print("Loading Dataset...")
    dataset = CocoaDataset()
    
    # Study Parameters
    start_date_str = "2022-01-01"
    end_date_str = "2024-03-11"
    step = 5  # Weekly-ish
    
    print(f"Initializing Mohr Convergence Study")
    print(f"Range: {start_date_str} to {end_date_str}")
    print(f"Step: {step} indices")
    
    try:
        start_idx = dataset.get_1_based_index_from_date(start_date_str)
    except ValueError:
        # Fallback if 1st Jan is holiday
        # Try finding nearest valid date or just use a known trading day nearby
        # Just searching for the first date >= 2022-01-01
        valid_dates = dataset.dates[dataset.dates >= start_date_str]
        if valid_dates.empty:
            raise ValueError("Start date out of range")
        actual_start_date = valid_dates.iloc[0]
        start_idx = dataset.get_1_based_index_from_date(actual_start_date)
        print(f"Adjusted start date to trading day: {actual_start_date.date()}")

    end_idx = dataset.get_1_based_index_from_date(end_date_str)
    
    indices = range(start_idx, end_idx + 1, step)
    
    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "mohr_study", f"{timestamp}_FullConvergence")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f"Total trials to run: {len(indices)}")
    
    print(f"Total trials to run: {len(indices)}")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_trial(cur_idx):
        cur_date_val = dataset.get_date_from_1_based_index(cur_idx)
        cur_date_str = cur_date_val.strftime("%Y-%m-%d")
        try:
            # Re-instantiate dataset inside thread? CocoaDataset logic seems safe to share 
            # as long as we treat it read-only. But MohrRunner creates new engines.
            # Just passing 'dataset' is fine if it's thread-safe.
            # Safest is to just pass it.
            runner = MohrRunner(oos_start_date=cur_date_str, dataset=dataset)
            break_idx = runner.run_mohr_break_detection()
            break_date = dataset.get_date_from_1_based_index(break_idx)
            trim_start, trim_end = runner.get_trimmed_indexies()
            return {
                "oos_date": cur_date_val,
                "break_date": break_date,
                "break_index": break_idx,
                "trim_start_idx": trim_start,
                "trim_end_idx": trim_end
            }
        except Exception as e:
            print(f"Error at {cur_date_str}: {e}")
            return None

    # Run in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(process_trial, idx): idx for idx in indices}
        
        completed = 0
        total = len(indices)
        
        for future in as_completed(future_to_idx):
            res = future.result()
            if res:
                results.append(res)
            completed += 1
            print(f"Progress: [{completed}/{total}] completed.", end="\r")

    print(f"\nCollected {len(results)} successful trials.")

    print("\nStudy execution complete. Analyzing results...")

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    
    # --- Post-hoc Trimming Flag Logic ---
    valid_breaks = {pd.Timestamp(r.date()) for r in df["break_date"].unique() if not pd.isna(r)}
    
    df['trimmed_flag'] = False
    
    for idx, row in df.iterrows():
        # Get trim dates
        t_start = dataset.dates.iloc[row['trim_start_idx']]
        t_end = dataset.dates.iloc[row['trim_end_idx']]
        
        # Check against unique breaks
        for b_date in valid_breaks:
            if t_start <= b_date <= t_end:
                 df.at[idx, 'trimmed_flag'] = True
                 break
    
    flagged_count = df['trimmed_flag'].sum()
    print(f"Post-process: {flagged_count} trials flagged as 'blind spot'.")
    
    # Calculate simplistic convergence metrics
    # Count how many times the break date changed
    break_changes = df['break_date'].ne(df['break_date'].shift()).sum() - 1 # -1 for first row
    unique_regimes = df['break_date'].nunique()
    
    print(f"Convergence Metrics:")
    print(f"  - Unique Break Dates Found: {unique_regimes}")
    print(f"  - Regime Switches (Flips): {break_changes}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "mohr_convergence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="oos_date", y="break_date", marker="o", drawstyle="steps-post", label="Detected Break Date")
    
    # Shade trimmed regions
    df_sorted = df.sort_values("oos_date")
    is_trimmed = df_sorted['trimmed_flag'].astype(bool).values
    dates = df_sorted['oos_date'].values
    
    import numpy as np
    if is_trimmed.any():
        padded = np.concatenate(([False], is_trimmed, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        labeled = False
        for start_idx, end_idx in zip(starts, ends):
            t1 = dates[start_idx]
            t2 = dates[end_idx - 1]
            
            label = "Trimmed (Blind Spot)" if not labeled else None
            plt.axvspan(t1, t2, color='red', alpha=0.2, label=label)
            labeled = True

    plt.title(f"Mohr Convergence Study ({start_date_str} to {end_date_str})")
    plt.xlabel("OOS Start Date (Rolling Origin)")
    plt.ylabel("Detected Break Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(output_dir, "convergence_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_study()
