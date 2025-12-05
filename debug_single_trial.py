
import pandas as pd
from cocoa.models import CocoaDataset
from cocoa.experiments.rolling_wll import RollingWLLExperiment

def run_debug_trial():
    print("Initializing Dataset...")
    dataset = CocoaDataset()
    target_date_str = "2024-03-11"
    target_date = pd.Timestamp(target_date_str)
    
    # Get the 1-based index for the target date
    try:
        # We need the index essentially to pass to the experiment, 
        # but the experiment usually iterates. We will hack it to run just one.
        # Check if date exists
        if target_date not in dataset.dates.values:
            print(f"Date {target_date_str} not found in dataset.")
            # Find closest previous date?
            return

        index = dataset.get_1_based_index_from_date(target_date)
        print(f"Index for {target_date_str} is {index}")

        # Instantiate experiment
        # We set start/end/step such that we can access the internal methods, 
        # bounds don't matter much if we call run_single_trial directly.
        exp = RollingWLLExperiment(
            start_index=index+20,
            end_index=index - 20,  # Rolling window of 40 trials
            step=1,
            dataset=dataset
        )
        
        print("Running Pilot Mohr...")
        # Pilot mohr is called inside run() as well, but no harm calling it early if needed, 
        # actually exp.run() calls it. 
        # exp.run_pilot_mohr() 
        
        print(f"Running Rolling Experiment from index {index+20} to {index-20}...")
        exp.run(baseline_mode="wll_only", n_workers=1)
        
        print(f"Completed {len(exp.records)} trials.")
        
        if exp.records:
            print("Generating Plots...")
            exp.plot_break_vs_origin()
            exp.plot_pi_vs_origin()
            exp.plot_msfe_comparison()
            print("Plots generated in output/rolling_experiments/")
        else:
            print("No records found, skipping plots.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug_trial()
