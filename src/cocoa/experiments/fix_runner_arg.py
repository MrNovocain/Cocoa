import re

file_path = r"w:\Research\NP\Cocoa\src\cocoa\experiments\run_np_combo_cv.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace sample_start_index= with break_index= in ConvexComboExperimentRunner calls
# But be careful not to replace the function argument 'sample_start_index' in run_np_combo_cv_for_gamma_analysis
# The call usage is:
# sample_start_index=idx,
# or
# sample_start_index=i,

# We want to change the kwargs passed to ConvexComboExperimentRunner.
# Regex replace `sample_start_index=` with `break_index=` WHERE IT IS A KWARG.
# Pattern: `\s+sample_start_index=` inside a call.

new_content = content.replace("sample_start_index=idx", "break_index=idx")
new_content = new_content.replace("sample_start_index=i", "break_index=i")
# Also line 143: sample_start_index=best_break_index
new_content = new_content.replace("sample_start_index=best_break_index", "break_index=best_break_index")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Patched {file_path}")
