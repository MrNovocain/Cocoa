
import os
import json
import glob

base_dir = r"w:\Research\NP\Cocoa\output\cocoa_forecast"

max_msfe = -1
max_file = ""
max_date = ""

# Iterate over all config files or oos_metrics
for metrics_file in glob.glob(os.path.join(base_dir, "*", "oos_metrics.json")):
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            msfe = data.get("MSFE", 0)
            
            if msfe > max_msfe:
                max_msfe = msfe
                max_file = metrics_file
                
                # Try to read config to get the date
                config_path = os.path.join(os.path.dirname(metrics_file), "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as cf:
                        config = json.load(cf)
                        # The start date of the OOS test set is likely the "origin date"
                        max_date = config.get("test_set_start_date")
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")

print(f"Max MSFE: {max_msfe}")
print(f"Date: {max_date}")
print(f"File: {max_file}")
