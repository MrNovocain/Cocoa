import json
import re

nb_path = r"w:\Research\NP\Cocoa\notebooks\WLL_Cocoa_Experiment final.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- Helper Function: Get Cell ---
def get_cell(idx):
    return nb["cells"][idx]

def set_cell_source(idx, source_lines):
    # Ensure lines end with \n
    final_source = []
    for line in source_lines:
        if not line.endswith("\n"):
            final_source.append(line + "\n")
        else:
            final_source.append(line)
    nb["cells"][idx]["source"] = final_source

# --- Step 1: Extract modified_dm_test and move to Cell 2 ---
# Cell 2 is the Helper Functions cell (index 2).
# Cell 28 contains the function definition + test logic.
# We will split Cell 28.

cell_28_source = get_cell(28)["source"]
# Find where the function definition ends.
# It ends at "return mdm_stat, p_value" line.
# Let's find the line index.
func_end_idx = -1
for i, line in enumerate(cell_28_source):
    if "return mdm_stat, p_value" in line:
        func_end_idx = i
        break

if func_end_idx != -1:
    dm_func_lines = cell_28_source[:func_end_idx+1]
    dm_test_logic_lines = cell_28_source[func_end_idx+1:]
    
    # Append to Cell 2
    cell_2_source = get_cell(2)["source"]
    # Add a newline separator if needed
    if cell_2_source and not cell_2_source[-1].endswith("\n"):
        cell_2_source[-1] += "\n"
    
    new_cell_2_source = cell_2_source + ["\n", "# --- Modified Diebold-Mariano Test ---\n"] + dm_func_lines
    set_cell_source(2, new_cell_2_source)
    
    # Update Cell 28 to only have the test logic
    set_cell_source(28, dm_test_logic_lines)
    print("Moved modified_dm_test to Cell 2.")

# --- Step 2: Suppress output in Cell 17 (WLL) ---
cell_17_source = get_cell(17)["source"]
new_17_source = []
for line in cell_17_source:
    if "print(f'  Tuning gamma via MFV...')" in line:
        new_17_source.append("# " + line)
    elif "print(f'  Gamma grid:" in line:
        new_17_source.append("# " + line)
    else:
        new_17_source.append(line)
set_cell_source(17, new_17_source)
print("Suppressed output in Cell 17.")

# --- Step 3: Suppress output in Cell 19 (RF) ---
cell_19_source = get_cell(19)["source"]
new_19_source = []
for line in cell_19_source:
    if "print(f'  Grid size:" in line:
        new_19_source.append("# " + line)
    elif "print('  Running MFV hyperparameter tuning...')" in line:
        new_19_source.append("# " + line)
    else:
        new_19_source.append(line)
set_cell_source(19, new_19_source)
print("Suppressed output in Cell 19.")

# --- Step 4: Suppress output in Cell 20 (XGB) ---
cell_20_source = get_cell(20)["source"]
new_20_source = []
for line in cell_20_source:
    if "print(f'  Grid size:" in line:
        new_20_source.append("# " + line)
    elif "print('  Running MFV hyperparameter tuning...')" in line:
        new_20_source.append("# " + line)
    else:
        new_20_source.append(line)
set_cell_source(20, new_20_source)
print("Suppressed output in Cell 20.")

# --- Save Notebook ---
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Refactoring complete.")
