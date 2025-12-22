import json

nb_path = r"w:\Research\NP\Cocoa\notebooks\WLL_Cocoa_Experiment final.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Helper to set cell source
def set_cell_source(idx, source):
    # Splits strings into lines list for JSON
    lines = source.split("\n")
    final_source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    nb["cells"][idx]["source"] = final_source

# --- 1. Fix Plotting Cells (Use CONFIG['colors']) ---
# Indices 25 and 26 in original numbering.
# Since we haven't shifted yet, these are valid.

cell_25_code = """# MSFE comparison chart
model_colors = CONFIG['colors']

if 'results_df' in locals():
    fig, ax = plt.subplots(figsize=(12, 7))
    # Sort for chart
    df_plot = results_df.sort_values('MSFE', ascending=False)
    
    bars = ax.barh(
        df_plot['Model'], df_plot['MSFE'],
        color=[model_colors.get(m, '#95A5A6') for m in df_plot['Model']],
        edgecolor='white', linewidth=1.5
    )

    for bar, msfe in zip(bars, df_plot['MSFE']):
        ax.text(bar.get_width() + 0.00001, bar.get_y() + bar.get_height()/2,
                f'{msfe:.6f}', va='center', fontsize=10)

    ax.set_xlabel('Mean Squared Forecast Error (MSFE)')
    ax.set_title('Out-of-Sample Forecast Performance Comparison')
    # ax.invert_yaxis() # already sorted
    
    # Highlight best if we know it?
    # ax.axvline(...)
    
    plt.tight_layout()
    plt.show()
else:
    print("results_df not found. Run predictions first.")
"""
set_cell_source(25, cell_25_code)

cell_26_code = """# Cumulative squared error over time
if 'predictions' in locals() and 'y_test' in locals():
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate CSE
    for name, preds in predictions.items():
        sq_err = (y_test.values - preds) ** 2
        cum_se = np.cumsum(sq_err)
        color = model_colors.get(name, '#95A5A6')
        
        lw = 3 if name == 'WLL' else 1.5
        alpha = 1.0 if name == 'WLL' else 0.7
        
        ax.plot(test_dates, cum_se, label=name, color=color, linewidth=lw, alpha=alpha)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Squared Error')
    ax.set_title('Cumulative Out-of-Sample Squared Forecast Error')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Predictions not ready.")
"""
set_cell_source(26, cell_26_code)

print("Updated plot cells.")

# --- 2. Insert Narrative Cells ---
# Helper for md cell creation
def create_md_cell(source_list):
    return {"cell_type": "markdown", "metadata": {}, "source": source_list}

# A. Insert "### 5.1 WLL" before Cell 17
md_wll = create_md_cell([
    "### 5.1 Weighted Local Linear Model (WLL)\n",
    "\n",
    "The WLL model combines the Pre-Break and Post-Break local linear estimators. The weight $\\gamma$ is tuned using the MFV criterion, balancing the bias of the pre-break model against the variance of the post-break model."
])
nb["cells"].insert(17, md_wll)
print("Inserted WLL intro.")

# Index Shifts:
# Cells 0-16: same
# New 17: MD WLL
# Old 17 -> New 18
# All subsequent +1

# B. Insert "Model Evaluation" intro after "6. OOS Evaluation" header.
# Old Cell 23 (MD) is now Cell 24.
# Old Cell 24 (Code) is now Cell 25.
# We insert at index 25.
md_oos = create_md_cell([
    "We evaluate forecasting performance using the Mean Squared Forecast Error (MSFE) over the test window. We also examine the cumulative squared error to diagnose *when* models outperform each other (e.g., immediate adaptation vs. long-run stability)."
])
nb["cells"].insert(25, md_oos)
print("Inserted OOS intro.")

# All subsequent +2 (total)

# C. Insert "Summary" text if needed, or rely on existing.
# Old Cell 29 (Summary) is now 29 + 2 = 31.
# It's "## 8. Summary". Good enough.

# --- Save ---
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook final touches applied.")
