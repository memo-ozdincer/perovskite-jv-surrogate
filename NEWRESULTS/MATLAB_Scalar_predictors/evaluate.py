# standalone_evaluator.py
#
# Description:
#   This script compares predicted solar cell performance scalars (Isc, Voc, FF, PCE)
#   against ground truths calculated from raw I-V curve data.
#
#   It performs the following steps:
#   1. Calculates the "true" Isc, Voc, FF, and PCE from the `iV_m.txt` file using
#      PCHIP interpolation for accuracy.
#   2. Loads the predicted scalar values from four separate .txt files.
#   3. Aligns the true and predicted data, dropping any invalid curves.
#   4. Computes and prints summary statistics (MAE, RMSE, R²).
#   5. Generates and saves two plots:
#      - A 2x2 grid of parity plots (Predicted vs. True).
#      - A correlation heatmap of all scalars.
#
# Requirements:
#   pip install numpy pandas matplotlib seaborn scikit-learn scipy
#
# --- SETUP ---
# 1. Place this script in a directory.
# 2. Create a subdirectory named 'data' (or change the `DATA_DIR` path).
# 3. Inside 'data', place your five input files:
#    - iV_m.txt
#    - NN_predicted_ISC.txt
#    - NN_predicted_VOC.txt
#    - NN_predicted_FF.txt
#    - NN_predicted_PCE.txt
# 4. Run the script: `python standalone_evaluator.py`

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. CONFIGURATION: Define Paths and Constants ---

# Directory containing all your input .txt files
DATA_DIR = Path("./data")

# Input files
GROUND_TRUTH_IV_PATH = DATA_DIR / "iV_m.txt"
PREDICTED_ISC_PATH   = DATA_DIR / "NN_predicted_ISC.txt"
PREDICTED_VOC_PATH   = DATA_DIR / "NN_predicted_VOC.txt"
PREDICTED_FF_PATH    = DATA_DIR / "NN_predicted_FF.txt"
PREDICTED_PCE_PATH   = DATA_DIR / "NN_predicted_PCE.txt"

# Output files
OUTPUT_PARITY_PLOT_PATH = Path("./parity_plots.png")
OUTPUT_CORR_PLOT_PATH   = Path("./correlation_heatmap.png")

# Constants
P_INPUT = 100.0  # Input power in mW/cm^2 (standard 1-sun condition)
V_COARSE_GRID = np.concatenate([
    np.arange(0, 0.4 + 1e-8, 0.1),
    np.arange(0.425, 1.4 + 1e-8, 0.025)
]).astype(np.float32)

# --- 2. GROUND TRUTH CALCULATION ---

def compute_true_scalars(v_grid: np.ndarray, i_curve: np.ndarray) -> dict | None:
    """
    Calculates performance scalars from a single I-V curve using PCHIP interpolation.
    Returns a dictionary of scalars or None if the curve is invalid.
    """
    if np.count_nonzero(~np.isnan(i_curve)) < 4:
        return None
    
    # Short-circuit current (Isc) is the current at V=0
    isc = float(i_curve[0])
    if isc <= 1e-9:
        return None

    try:
        # Create a high-fidelity interpolator for the curve
        pi = PchipInterpolator(v_grid, i_curve, extrapolate=False)
        v_fine = np.linspace(v_grid[0], v_grid[-1], 2000)
        i_fine = pi(v_fine)
        
        # Find Open-circuit voltage (Voc) where current crosses zero
        zero_cross_indices = np.where(i_fine <= 0)[0]
        if len(zero_cross_indices) == 0:
            return None # Curve doesn't cross zero, invalid Voc
        voc = float(v_fine[zero_cross_indices[0]])

        # Find Maximum Power Point (MPP)
        search_mask = v_fine <= voc
        v_search, i_search = v_fine[search_mask], i_fine[search_mask]
        if len(v_search) == 0:
            return None
            
        power = v_search * i_search
        mpp_index = np.argmax(power)
        vmpp, impp = float(v_search[mpp_index]), float(i_search[mpp_index])

        # Calculate Fill Factor (FF) and Power Conversion Efficiency (PCE)
        if voc * isc < 1e-9:
            return None # Avoid division by zero
            
        ff = (vmpp * impp) / (voc * isc)
        pce = (vmpp * impp) / P_INPUT * 100.0  # Efficiency in percent

        return {
            "Isc_true": isc,
            "Voc_true": voc,
            "FF_true": ff,
            "PCE_true": pce
        }
    except (ValueError, IndexError):
        # Catch any interpolation or indexing errors
        return None

def calculate_all_ground_truths(iv_path: Path) -> pd.DataFrame:
    """
    Iterates through the raw IV data file and computes true scalars for each curve.
    """
    print(f"Calculating ground truths from '{iv_path}'...")
    iv_data = np.loadtxt(iv_path, delimiter=',')
    
    results = []
    for i, curve in enumerate(iv_data):
        scalars = compute_true_scalars(V_COARSE_GRID, curve)
        if scalars:
            results.append({"sample_id": i, **scalars})
            
    df_true = pd.DataFrame(results).set_index("sample_id")
    print(f"Successfully calculated truths for {len(df_true)} / {len(iv_data)} curves.")
    return df_true

# --- 3. DATA LOADING AND ANALYSIS ---

def load_predicted_data() -> pd.DataFrame:
    """
    Loads the four predicted scalar .txt files into a single DataFrame.
    """
    print("Loading predicted scalar files...")
    df_pred = pd.DataFrame({
        "Isc_pred": np.loadtxt(PREDICTED_ISC_PATH),
        "Voc_pred": np.loadtxt(PREDICTED_VOC_PATH),
        "FF_pred":  np.loadtxt(PREDICTED_FF_PATH),
        "PCE_pred": np.loadtxt(PREDICTED_PCE_PATH),
    })
    df_pred.index.name = "sample_id"
    return df_pred

def generate_statistics(df: pd.DataFrame):
    """
    Computes and prints MAE, RMSE, and R² for each scalar.
    """
    print("\n--- Performance Statistics ---")
    metrics = []
    for scalar in ["Isc", "Voc", "FF", "PCE"]:
        y_true = df[f"{scalar}_true"]
        y_pred = df[f"{scalar}_pred"]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics.append({"Scalar": scalar, "MAE": mae, "RMSE": rmse, "R²": r2})
        
    stats_df = pd.DataFrame(metrics).set_index("Scalar")
    print(stats_df.round(4))
    print("----------------------------\n")

def plot_parity_plots(df: pd.DataFrame):
    """
    Generates and saves a 2x2 grid of parity plots (Predicted vs. True).
    """
    print(f"Generating parity plots, saving to '{OUTPUT_PARITY_PLOT_PATH}'...")
    scalars = ["Isc", "Voc", "FF", "PCE"]
    units = ["(mA/cm²)", "(V)", "(unitless)", "(%)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Parity Plots: Predicted vs. True Values", fontsize=16, weight='bold')
    
    for i, (scalar, unit) in enumerate(zip(scalars, units)):
        ax = axes.flatten()[i]
        y_true = df[f"{scalar}_true"]
        y_pred = df[f"{scalar}_pred"]
        
        # Plot data
        ax.scatter(y_true, y_pred, alpha=0.3, edgecolors='k', s=20)
        
        # Plot identity line (y=x)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', lw=2, label='Ideal (y=x)')
        
        # Add R² score to the plot
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Formatting
        ax.set_xlabel(f"True {scalar} {unit}", fontsize=12)
        ax.set_ylabel(f"Predicted {scalar} {unit}", fontsize=12)
        ax.set_title(f"{scalar} Parity", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PARITY_PLOT_PATH, dpi=150)
    plt.close()
    print("Done.")

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Generates and saves a correlation heatmap for all scalars.
    """
    print(f"Generating correlation heatmap, saving to '{OUTPUT_CORR_PLOT_PATH}'...")
    
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5
    )
    plt.title("Correlation Matrix of True and Predicted Scalars", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_CORR_PLOT_PATH, dpi=150)
    plt.close()
    print("Done.")


# --- 4. MAIN EXECUTION (REVISED) ---

if __name__ == "__main__":
    # Calculate ground truths from the raw IV data
    df_true = calculate_all_ground_truths(GROUND_TRUTH_IV_PATH)

    # Load the NN's predictions for the three CORE scalars
    print("Loading predicted scalar files (Isc, Voc, FF)...")
    df_pred_base = pd.DataFrame({
        "Isc_pred": np.loadtxt(PREDICTED_ISC_PATH),
        "Voc_pred": np.loadtxt(PREDICTED_VOC_PATH),
        "FF_pred":  np.loadtxt(PREDICTED_FF_PATH),
    })
    df_pred_base.index.name = "sample_id"

    # --- FIX: Derive PCE_pred from the other predictions ---
    # This avoids the unit errors from the original PCE prediction file.
    # The formula is PCE = Pmax / Pin * 100 = (Isc * Voc * FF) / 100 * 100 = Isc * Voc * FF
    # (Assuming Isc in mA/cm^2, Voc in V, and Pin = 100 mW/cm^2)
    df_pred_base["PCE_pred"] = df_pred_base["Isc_pred"] * df_pred_base["Voc_pred"] * df_pred_base["FF_pred"]
    print("Derived 'PCE_pred' from Isc, Voc, and FF predictions.")

    # Merge true and predicted values, keeping only samples valid in both
    df_merged = pd.merge(df_true, df_pred_base, left_index=True, right_index=True, how="inner")

    print(f"\nFound {len(df_merged)} samples with both valid truths and predictions for analysis.")

    # Perform analysis with the correctly derived PCE
    generate_statistics(df_merged)
    plot_parity_plots(df_merged)
    plot_correlation_heatmap(df_merged)

    print("\nAnalysis complete. Check for 'parity_plots.png' and 'correlation_heatmap.png'.")
