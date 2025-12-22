
import json
import os
from pathlib import Path

notebook_path = Path(r"w:\Research\NP\Cocoa\notebooks\WLL_Cocoa_Experiment final.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
             "# Cocoa Price Forecasting: Weighted Local Linear vs Machine Learning Methods\n",
             "\n",
             "**Project:** Empirical Validation of the CGS Weighted Local Linear Framework\n",
             "**Reference:** CGS (2025) - Nonparametric Time Series Forecasting with Structural Breaks\n",
             "\n",
             "---\n",
             "\n",
             "## Abstract & Methodology\n",
             "\n",
             "This notebook implements the experimental framework to validate the CGS weighted local linear (WLL) methodology on cocoa price data. We compare the predictive performance of WLL against standard ML benchmarks (Random Forest, XGBoost) and \"honest\" nonparametric baselines.\n",
             "\n",
             "### The Problem: Structural Uncertainty\n",
             "Standard ML models often assume a stable data generating process or require explicit feature engineering to handle breaks. CGS proposes a \"Weighted Local Linear\" estimator that optimally combines pre-break and post-break information without needing to perfect date the break.\n",
             "\n",
             "### Methodology\n",
             "1.  **Data Preparation**: Load processed cocoa price and weather data.\n",
             "2.  **Structural Break Detection**: Identify the primary regime shift (detected at index 6116).\n",
             "3.  **Model Training & Forecasting**:\n",
             "    *   **NP Benchmarks**: Pre-break only, Post-break only.\n",
             "    *   **WLL**: Convex combination tuned via MFV (Metric-based Fold Validation).\n",
             "    *   **ML Models**: RF and XGBoost trained on full history.\n",
             "4.  **Evaluation**: Compare Out-of-Sample Mean Squared Forecast Error (MSFE)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
             "# ==========================================\n",
             "# 1. Configuration & Setup\n",
             "# ==========================================\n",
             "import warnings\n",
             "import sys\n",
             "import subprocess\n",
             "from pathlib import Path\n",
             "import numpy as np\n",
             "import pandas as pd\n",
             "import matplotlib.pyplot as plt\n",
             "from scipy import stats\n",
             "\n",
             "# Suppress warnings for cleaner output\n",
             "warnings.filterwarnings('ignore')\n",
             "\n",
             "# Graphic settings\n",
             "plt.style.use('seaborn-v0_8-whitegrid')\n",
             "plt.rcParams['figure.figsize'] = (12, 6)\n",
             "plt.rcParams['font.size'] = 11\n",
             "\n",
             "# Experiment Configuration\n",
             "CONFIG = {\n",
             "    \"data_path\": \"../data/processed/cocoa_price_weather.csv\",\n",
             "    \"target_col\": \"log_return_forecast_target\",\n",
             "    \"break_detection\": {\n",
             "        \"method\": \"Mohr-P\",\n",
             "        \"fixed_break_index\": 6116, # Fixed for reproducibility in this run\n",
             "        \"break_date\": \"2022-09-23\"\n",
             "    },\n",
             "    \"test_start_date\": \"2024-01-01\",\n",
             "    \"colors\": {\n",
             "        'Pre-Break LL': '#E74C3C',\n",
             "        'Post-Break LL': '#3498DB',\n",
             "        'WLL': '#2ECC71',\n",
             "        'Random Forest': '#9B59B6',\n",
             "        'XGBoost': '#F39C12',\n",
             "        'RF Combo': '#1ABC9C',\n",
             "        'XGB Combo': '#E67E22',\n",
             "    }\n",
             "}\n",
             "\n",
             "# --- Import Project Modules ---\n",
             "# Ensure local src is in path\n",
             "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n",
             "\n",
             "from cocoa.data.features import build_features\n",
             "from cocoa.models.np_combo import NPConvexCombinationModel\n",
             "from cocoa.models.np_regime import NPRegimeModel\n",
             "from cocoa.models.ml_models import RFModel, XGBModel\n",
             "from cocoa.models.ml_combo import MLConvexCombinationModel\n",
             "from cocoa.models.mfv_CV import MFVValidator\n",
             "from cocoa.models.assets import (\n",
             "    DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, \n",
             "    RF_PARAM_GRID, XGB_PARAM_GRID\n",
             ")\n",
             "\n",
             "print(\"Configuration loaded. Project modules imported successfully.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
             "# ==========================================\n",
             "# 2. Helper Functions\n",
             "# ==========================================\n",
             "\n",
             "def load_and_prep_data(config):\n",
             "    \"\"\"Loads processed data or builds it if missing.\"\"\"\n",
             "    data_path = Path(config[\"data_path\"])\n",
             "    \n",
             "    if not data_path.exists():\n",
             "        print('Processed data not found. Building from raw data...')\n",
             "        project_root = Path.cwd().parent\n",
             "        df = build_features(\n",
             "            project_root / 'data' / 'raw',\n",
             "            project_root / 'data' / 'processed',\n",
             "            reading_path='Ghana_data_full.csv',\n",
             "            file_name='cocoa_ghana_full.csv'\n",
             "        )\n",
             "    else:\n",
             "        df = pd.read_csv(data_path)\n",
             "    \n",
             "    df['date'] = pd.to_datetime(df['date'])\n",
             "    df = df.sort_values('date').reset_index(drop=True)\n",
             "    \n",
             "    print(f\"Data Loaded: {len(df):,} observations\")\n",
             "    print(f\"Range: {df['date'].min().date()} to {df['date'].max().date()}\")\n",
             "    \n",
             "    return df\n",
             "\n",
             "def plot_forecast_comparison(results_df, config):\n",
             "    \"\"\"Visualizes MSFE comparison.\"\"\"\n",
             "    fig, ax = plt.subplots(figsize=(10, 6))\n",
             "    \n",
             "    # Sort for better visualization\n",
             "    results_df = results_df.sort_values('MSFE', ascending=False)\n",
             "    \n",
             "    bars = ax.barh(\n",
             "        results_df['Model'], results_df['MSFE'],\n",
             "        color=[config['colors'].get(m, 'gray') for m in results_df['Model']],\n",
             "        edgecolor='white'\n",
             "    )\n",
             "    \n",
             "    ax.set_xlabel('Mean Squared Forecast Error (MSFE)')\n",
             "    ax.set_title('Out-of-Sample Forecast Performance')\n",
             "    \n",
             "    # Add values\n",
             "    for bar in bars:\n",
             "        width = bar.get_width()\n",
             "        ax.text(width, bar.get_y() + bar.get_height()/2, \n",
             "                f'{width:.6f}', ha='left', va='center', fontsize=10)\n",
             "        \n",
             "    plt.tight_layout()\n",
             "    plt.show()\n",
             "\n",
             "print(\"Helper functions defined.\")"
        ]
    }
]

# Keep cells from the "Data Loading" section onwards
# The original file had cells 0-5 as setup. Cell 6 was "2. Data Loading...".
# We want to replace the first 5 cells with our 3 new cells.
# Actually, let's identify Cell 6 by its content to be safe.
start_index = 0
for i, cell in enumerate(nb['cells']):
    if "2. Data Loading and Exploration" in "".join(cell.get("source", [])):
        start_index = i
        break

print(f"Old setup cells count: {start_index}")
# Keep the rest of the notebook
remaining_cells = nb['cells'][start_index:]

# Combine
nb['cells'] = new_cells + remaining_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook refactored successfully.")
