import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from cocoa.models import CocoaDataset
from cocoa.experiments.break_detection import MohrRunner


def debug_mohr_range():
    print("Loading Dataset...")
    dataset = CocoaDataset()
    
    # Define Index Range directly
    start_idx = dataset.get_1_based_index_from_date("2023-11-01")
    end_idx = dataset.get_1_based_index_from_date("2024-03-11")
    step = 10
    
    # Get associated dates for logging/folder creation
    start_date = dataset.get_date_from_1_based_index(start_idx)
    end_date = dataset.get_date_from_1_based_index(end_idx)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Running Mohr Test Index Range: {start_idx} ({start_date_str}) to {end_idx} ({end_date_str}) with step {step}")
    
    indices = range(start_idx, end_idx + 1, step)
    
    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "mohr_debug", f"{timestamp}_Idx{start_idx}_to_{end_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for cur_idx in indices:
        cur_date_val = dataset.get_date_from_1_based_index(cur_idx)
        cur_date_str = cur_date_val.strftime("%Y-%m-%d")
        
        try:
            print(f"Processing {cur_date_str}...")
            # We must re-instantiate MohrRunner or reset it because it might cache stuff? 
            # MohrRunner uses `oos_start_date` in init.
            runner = MohrRunner(oos_start_date=cur_date_str, dataset=dataset)
            break_idx = runner.run_mohr_break_detection()
            break_date = dataset.get_date_from_1_based_index(break_idx)
            
            trim_start, trim_end = runner.get_trimmed_indexies()
            
            results.append({
                "oos_date": cur_date_val,
                "break_date": break_date,
                "break_index": break_idx,
                "trim_start_idx": trim_start,
                "trim_end_idx": trim_end
            })
            print(f"  -> Detected Break: {break_date.date()}")
            
        except Exception as e:
            print(f"  -> Error: {e}")

    # Plot
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    
    df = pd.DataFrame(results)
    
    # --- Post-hoc Trimming Flag Logic ---
    valid_breaks = {pd.Timestamp(r.date()) for r in df["break_date"].unique() if not pd.isna(r)}
    
    # FOR VERIFICATION ONLY: Inject a synthetic break known to be in the recent trim windows
    print("DEBUG: Injecting synthetic break '2024-02-01' to verify shading logic.")
    valid_breaks.add(pd.Timestamp("2024-02-01"))
    
    print(f"\nDebug: Collected unique break dates across all {len(df)} trials:")
    for b in sorted(valid_breaks):
        print(f"  - {b.date()}")
    
    # We need to map indices back to dates for comparison
    
    # We need to map indices back to dates for comparison
    # It's faster to do it once if we had the map, but loop is fine here
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
    print(f"Post-process: {flagged_count} trials flagged as having a break in the trim window.")
    
    # ------------------------------------

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="oos_date", y="break_date", marker="o", drawstyle="steps-post", label="Detected Break Date")
    
    # Shade trimmed regions
    # Identifying contiguous blocks of True
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
            # end_idx is index AFTER the block in original array (because of padding shift)
            # block range in `dates` is [start_idx, end_idx - 1]
            t1 = dates[start_idx]
            t2 = dates[end_idx - 1]
            
            # For visual width, maybe extend t2 slightly if it's the same as t1?
            # Or just use the points.
            label = "Trimmed (Blind Spot)" if not labeled else None
            plt.axvspan(t1, t2, color='red', alpha=0.2, label=label)
            labeled = True

    
    plt.title(f"Mohr Detected Break Date vs OOS Start Date ({start_date_str} to {end_date_str})")
    plt.xlabel("OOS Start Date (Rolling Origin)")
    plt.ylabel("Detected Break Date")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "break_stability_curve.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "mohr_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    debug_mohr_range()
