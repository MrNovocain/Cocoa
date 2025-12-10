
import json
import os

path = r"w:\Research\NP\Cocoa\output\cocoa_forecast\20251209_115344_WLL_Outlier_Debug\bias_variance_decomposition.json"

if os.path.exists(path):
    with open(path, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=4))
else:
    print(f"File not found: {path}")
