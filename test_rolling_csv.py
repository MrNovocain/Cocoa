
from cocoa.experiments.rolling_wll import RollingWLLExperiment
from cocoa.models import CocoaDataset
import os
import pandas as pd

def test_csv_creation():
    print("Testing CSV creation...")
    dataset = CocoaDataset()
    date = dataset.get_last_date()
    index = dataset.get_1_based_index_from_date(date)
    
    # Run a tiny experiment: just 2 steps, mock mode
    # index-1 to index-5, step 2
    exp = RollingWLLExperiment(index-1, index-5, 2)
    exp.run(baseline_mode="mock", n_workers=1)
    
    # Check if CSV exists
    csv_path = os.path.join(exp.output_dir, "rolling_results.csv")
    if os.path.exists(csv_path):
        print(f"SUCCESS: CSV found at {csv_path}")
        df = pd.read_csv(csv_path)
        print("Columns found:", df.columns.tolist())
        print(f"Rows found: {len(df)}")
        if len(df) > 0:
            print("Audit PASSED")
        else:
            print("Audit FAILED: DataFrame is empty")
    else:
        print(f"Audit FAILED: CSV not found at {csv_path}")

if __name__ == "__main__":
    test_csv_creation()
