"""
Script to run the full rolling WLL experiment from the end of the dataset
back to the end of 2021.
"""
from cocoa.experiments.rolling_wll import RollingWLLExperiment
from cocoa.models import CocoaDataset
import pandas as pd

def run_experiment():
    print("Initializing Full Rolling Experiment...")
    ds = CocoaDataset()
    
    # Define Date Boundaries
    # Start: Most recent data
    last_date = ds.dates.max()
    start_index = ds.get_1_based_index_from_date(last_date)
    
    # End: End of 2021
    date_2021 = pd.Timestamp("2021-12-31")
    closest_date_2021 = ds.dates[ds.dates <= date_2021].max()
    end_index = ds.get_1_based_index_from_date(closest_date_2021)
    
    step = 5  # Weeklyish step
    
    print(f"Rolling Range: {last_date.date()} (idx={start_index}) -> {closest_date_2021.date()} (idx={end_index})")
    print(f"Step size: {step}")
    
    # Initialize Experiment
    exp = RollingWLLExperiment(
        start_index=start_index, 
        end_index=end_index, 
        step=step
    )
    
    # Run
    # Using "auto_best_ml" for real usage, n_workers > 1 for speed
    print("Starting execution...")
    exp.run(baseline_mode="auto_best_ml", n_workers=4)
    print(f"Experiment completed. Processed {len(exp.records)} trials.")
    
    # Generate Plots
    print("Generating plots...")
    exp.plot_break_vs_origin()
    exp.plot_pi_vs_origin()
    exp.plot_msfe_comparison()
    exp.plot_hit_rate()
    exp.plot_pi_distribution()
    print("Plots saved.")

if __name__ == "__main__":
    run_experiment()
