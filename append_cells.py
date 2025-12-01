import nbformat
from nbformat.v4 import new_code_cell
from pathlib import Path

nb_path = Path('notebooks/WLL_Cocoa_Experiment.ipynb')
nb = nbformat.read(nb_path, as_version=4)

cells = []

cells.append(new_code_cell('''from pathlib import Path\nimport pandas as pd\nimport numpy as np\n\n# Load latest predictions file\npred_dir = Path.cwd().parent / "output" / "experiment_results"\npred_file = sorted(pred_dir.glob("predictions_*.csv"))[-1]\npred_use = pd.read_csv(pred_file).sort_values('date').reset_index(drop=True)\n\nmodels_to_check = ["WLL", "Random Forest", "XGBoost"]\ny_true = pred_use["y_true"].to_numpy()\ndates = pd.to_datetime(pred_use["date"])\n\nprint(f"Loaded predictions file: {pred_file.name}")'''))

cells.append(new_code_cell('''# H1: overall post-break MSFE\n\nimport numpy as np\n\ndef msfe(y, yhat):\n    return np.mean((y - yhat) ** 2)\n\nmsfe_rows = []\nfor m in models_to_check:\n    m_msfe = msfe(y_true, pred_use[m].to_numpy())\n    msfe_rows.append({"Model": m, "MSFE": m_msfe})\n\nmsfe_df = pd.DataFrame(msfe_rows)\nwll_msfe = msfe_df.loc[msfe_df["Model"] == "WLL", "MSFE"].values[0]\nmsfe_df["Rel_to_WLL_%"] = (msfe_df["MSFE"] / wll_msfe - 1) * 100\nmsfe_df.sort_values("MSFE")'''))

cells.append(new_code_cell('''# H2: early post-break cumulative squared error\nearly_k = 60  # adjust window length as needed\nearly_df = pred_use.iloc[:early_k].copy()\n\ncse_df = pd.DataFrame({"date": pd.to_datetime(early_df["date"] )})\nfor m in models_to_check:\n    se = (early_df["y_true"] - early_df[m]) ** 2\n    cse_df[f"CSE_{m}"] = se.cumsum()\n\ncse_df.head()'''))

cells.append(new_code_cell('''import matplotlib.pyplot as plt\n\nplt.figure(figsize=(10, 5))\nfor m in models_to_check:\n    plt.plot(cse_df["date"], cse_df[f"CSE_{m}"], label=m)\n\nplt.xlabel("Date")\nplt.ylabel("Cumulative squared error")\nplt.title(f"Early post-break CSE (first {early_k} observations)")\nplt.legend()\nplt.grid(True, linestyle="--", alpha=0.5)\nplt.show()'''))

cells.append(new_code_cell('''# Scalar summary at day K\ncse_end = {m: cse_df[f"CSE_{m}"].iloc[-1] for m in models_to_check}\n\ncse_summary = (\n    pd.DataFrame({\n        "Model": models_to_check,\n        "CSE_first_K": [cse_end[m] for m in models_to_check],\n    })\n    .assign(Rel_to_WLL_%=lambda df: (df["CSE_first_K"] / df.loc[df["Model"] == "WLL", "CSE_first_K"].values[0] - 1) * 100)\n)\ncse_summary.sort_values("CSE_first_K")'''))

cells.append(new_code_cell('''# Optional: windowed MSFE slices (e.g., 1-20, 21-40, 41-60)\nwindows = [(0, 20), (20, 40), (40, 60)]  # adjust as needed\n\nwin_rows = []\nfor start, end in windows:\n    y_slice = y_true[start:end]\n    for m in models_to_check:\n        yhat_slice = pred_use[m].to_numpy()[start:end]\n        win_rows.append({\n            "Window": f"{start+1}-{end}",\n            "Model": m,\n            "MSFE": msfe(y_slice, yhat_slice),\n        })\n\nwin_df = pd.DataFrame(win_rows)\n\nwin_rel = []\nfor window in win_df["Window"].unique():\n    tmp = win_df[win_df["Window"] == window].copy()\n    wll_val = tmp.loc[tmp["Model"] == "WLL", "MSFE"].values[0]\n    tmp["Rel_to_WLL_%"] = (tmp["MSFE"] / wll_val - 1) * 100\n    win_rel.append(tmp)\n\nwin_df_rel = pd.concat(win_rel, ignore_index=True)\nwin_df_rel'''))

cells.append(new_code_cell('''# Optional: gamma vs break date (small band around current break)\nfrom cocoa.experiments.run_np_combo_cv import run_np_combo_cv_for_gamma_analysis\n\nstart_idx = 6117 - 3\nend_idx = 6117 + 3\nrun_np_combo_cv_for_gamma_analysis(start_index=start_idx, end_index=end_idx, jump_size=1)'''))

nb.cells.extend(cells)
nbformat.write(nb, nb_path)
print('Appended hypothesis test cells to notebook.')
