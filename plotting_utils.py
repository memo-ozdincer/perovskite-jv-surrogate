"""
Plotting and visualization utilities for J-V curve reconstruction.
Includes PCHIP-based reconstruction plotting matching the KNOWN WORKING pipeline.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import r2_score
from PIL import Image


def plot_reconstructed_curves(
    v_slices: np.ndarray,
    i_true_slices: np.ndarray,
    i_pred_slices: np.ndarray,
    v_fine: np.ndarray = None,
    i_fine: np.ndarray = None,
    indices: list = None,
    title: str = "Curve Reconstruction",
    save_path: str = None,
    metrics: pd.DataFrame = None
):
    """
    Plot reconstructed J-V curves with PCHIP interpolation.

    Args:
        v_slices: (N, seq_len) voltage slices for coarse grid
        i_true_slices: (N, seq_len) true current slices
        i_pred_slices: (N, seq_len) predicted current slices
        v_fine: (N, n_fine) fine-grid voltages (optional, for ground truth)
        i_fine: (N, n_fine) fine-grid currents (optional, for ground truth)
        indices: List of sample indices to plot
        title: Plot title
        save_path: Path to save figure
        metrics: DataFrame with R² scores (optional)

    Returns:
        fig, axes
    """
    if indices is None:
        indices = range(min(8, len(v_slices)))

    n_samples = len(indices)
    nrows, ncols = (n_samples + 3) // 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), squeeze=False, constrained_layout=True)
    fig.suptitle(title.replace("_", " "), fontsize=20, weight='bold')

    axes_flat = axes.flatten()

    for plot_idx, sample_idx in enumerate(indices):
        ax = axes_flat[plot_idx]

        v_slice = v_slices[sample_idx]
        i_true = i_true_slices[sample_idx]
        i_pred = i_pred_slices[sample_idx]

        # Plot fine-grid ground truth if available
        if v_fine is not None and i_fine is not None:
            v_fine_sample = v_fine[sample_idx]
            i_fine_sample = i_fine[sample_idx]

            # Filter out NaN/inf
            mask = ~np.isnan(v_fine_sample) & ~np.isinf(v_fine_sample)
            v_fine_sample = v_fine_sample[mask]
            i_fine_sample = i_fine_sample[mask]

            if len(v_fine_sample) > 0:
                ax.plot(v_fine_sample, i_fine_sample, 'k-', alpha=0.7, lw=2, label='Actual (Fine Grid)')

                # Reconstruct prediction using PCHIP
                try:
                    pchip = PchipInterpolator(v_slice, i_pred, extrapolate=False)
                    i_pred_fine = pchip(v_fine_sample)
                    ax.plot(v_fine_sample, i_pred_fine, 'r--', lw=2, label='Predicted (Reconstructed)')
                except:
                    pass  # Skip if PCHIP fails

        # Plot coarse-grid points
        ax.plot(v_slice, i_true, 'bo', ms=6, label='Actual Points')
        ax.plot(v_slice, i_pred, 'rx', ms=6, mew=2, label='Predicted Points')

        # Add R² to title if available
        title_str = f"Sample #{sample_idx}"
        if metrics is not None and sample_idx in metrics.index:
            r2 = metrics.loc[sample_idx, 'r2']
            title_str += f" (R² = {r2:.4f})"
        ax.set_title(title_str)

        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (mA/cm²)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Set limits
        if v_fine is not None and i_fine is not None and len(v_fine_sample) > 0:
            ax.set_xlim(left=-0.05, right=max(v_fine_sample.max() * 1.05, 0.1))
            ax.set_ylim(bottom=-max(i_fine_sample.max() * 0.05, 1))

    # Remove empty subplots
    for j in range(n_samples, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig, axes


def generate_comparison_plots(
    results: list[dict],
    output_dir: Path,
    metrics_to_plot: dict = None
):
    """
    Generate summary comparison plots across multiple experiments.

    Args:
        results: List of dicts with metrics from different runs
        output_dir: Directory to save plots
        metrics_to_plot: Dict mapping display names to metric keys

    Example:
        results = [
            {'run_name': 'baseline', 'r2_median': 0.95, 'mae_median': 0.5, ...},
            {'run_name': 'improved', 'r2_median': 0.98, 'mae_median': 0.3, ...},
        ]
    """
    if not results:
        print("No results to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out error results
    df = pd.DataFrame([r for r in results if 'error' not in r])
    if df.empty:
        print("DataFrame is empty, cannot generate plots.")
        return

    # Default metrics
    if metrics_to_plot is None:
        metrics_to_plot = {
            'Median R² Score': 'r2_median',
            'Median Abs. MAE': 'mae_median',
            'Median Voc Error (V)': 'voc_error_abs_median',
            'Median Isc Error (mA/cm²)': 'isc_error_abs_median'
        }

    # Filter to available metrics
    plot_keys = [v for v in metrics_to_plot.values() if v in df.columns]

    if not plot_keys:
        print("No valid metrics found for plotting.")
        return

    nrows = (len(plot_keys) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows), constrained_layout=True)
    axes = axes.flatten() if nrows > 1 else [axes] if nrows == 1 else axes

    for i, key in enumerate(plot_keys):
        ax = axes[i]

        # Box plot
        sns.boxplot(data=df, x=key, y='run_name', ax=ax, orient='h', palette='viridis')

        # Get display name
        display_name = [k for k, v in metrics_to_plot.items() if v == key][0]
        ax.set_title(f'Comparison of {display_name}', fontsize=14, weight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Experiment', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    summary_plot_path = output_dir / "experiment_summary_comparison.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved experiment summary plot to: {summary_plot_path}")


class PlotManager:
    """
    Manager for generating plots during training/evaluation.
    Compatible with PyTorch Lightning callbacks or standalone use.
    """

    def __init__(self, output_dir: Path, n_samples: int = 8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples

    def plot_best_worst_random(
        self,
        v_slices: np.ndarray,
        i_true: np.ndarray,
        i_pred: np.ndarray,
        v_fine: np.ndarray = None,
        i_fine: np.ndarray = None,
        prefix: str = "test"
    ):
        """
        Generate three sets of plots: best, worst, and random samples.

        Args:
            v_slices: (N, seq_len) voltage grid
            i_true: (N, seq_len) true currents
            i_pred: (N, seq_len) predicted currents
            v_fine: (N, n_fine) optional fine-grid voltages
            i_fine: (N, n_fine) optional fine-grid currents
            prefix: Filename prefix
        """
        # Compute R² for ranking
        valid_mask = [i for i in range(len(i_true)) if np.var(i_true[i]) > 1e-6]

        if not valid_mask:
            print("Could not compute R² for any samples. Skipping plotting.")
            return

        metrics_df = pd.DataFrame({
            'r2': [r2_score(i_true[i], i_pred[i]) for i in valid_mask]
        }, index=valid_mask)

        n_samples = min(self.n_samples, len(metrics_df))
        if n_samples == 0:
            return

        # Define sample groups
        plot_groups = {
            f"{prefix}_random": np.random.choice(metrics_df.index, n_samples, replace=False),
            f"{prefix}_best": metrics_df.nlargest(n_samples, 'r2').index.values,
            f"{prefix}_worst": metrics_df.nsmallest(n_samples, 'r2').index.values,
        }

        # Generate plots for each group
        for name, indices in plot_groups.items():
            filename = self.output_dir / f"{name}_plots.png"

            plot_reconstructed_curves(
                v_slices=v_slices,
                i_true_slices=i_true,
                i_pred_slices=i_pred,
                v_fine=v_fine,
                i_fine=i_fine,
                indices=indices,
                title=name.replace("_", " ").title(),
                save_path=str(filename),
                metrics=metrics_df
            )

        print(f"Generated plots for {len(plot_groups)} groups")

    def plot_training_curves(
        self,
        history: dict,
        metrics: list[str] = None
    ):
        """
        Plot training curves (loss, metrics over epochs).

        Args:
            history: Dict with 'train' and 'val' keys containing lists of metric dicts
            metrics: List of metric names to plot (default: ['loss', 'mse'])
        """
        if metrics is None:
            metrics = ['loss', 'mse']

        available_metrics = []
        for m in metrics:
            if any(m in d for d in history.get('train', [])):
                available_metrics.append(m)

        if not available_metrics:
            print("No metrics found in history.")
            return

        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            train_values = [d.get(metric, np.nan) for d in history.get('train', [])]
            val_values = [d.get(metric, np.nan) for d in history.get('val', [])]

            epochs = range(1, len(train_values) + 1)

            ax.plot(epochs, train_values, 'b-', label='Train')
            ax.plot(epochs, val_values, 'r-', label='Val')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Training Curve: {metric.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Saved training curves to {save_path}")
