import json

nb_path = r"w:\Research\NP\Cocoa\notebooks\WLL_Cocoa_Experiment final.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"]).strip()
        first_line = source.split("\n")[0] if source else "EMPTY"
        print(f"Cell {cell_idx}: {first_line[:100]}")
    elif cell["cell_type"] == "markdown":
        source = "".join(cell["source"]).strip()
        first_line = source.split("\n")[0] if source else "EMPTY"
        print(f"Cell {cell_idx} (MD): {first_line[:100]}")
