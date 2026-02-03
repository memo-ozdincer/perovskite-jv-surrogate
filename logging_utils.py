"""
Structured logging utilities for training pipeline.
Provides consistent logging across all models and training stages.
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class ConstraintViolationLog:
    """Track constraint violation attempts before projection."""
    epoch: int
    batch: int
    jsc_negative: int = 0
    voc_negative: int = 0
    vmpp_exceeds_voc: int = 0
    jmpp_exceeds_jsc: int = 0
    total_samples: int = 0

    @property
    def violation_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        total_violations = self.jsc_negative + self.voc_negative + self.vmpp_exceeds_voc + self.jmpp_exceeds_jsc
        return total_violations / (self.total_samples * 4)


@dataclass
class MultiTaskLossLog:
    """Track multi-task loss components and learned sigmas."""
    epoch: int
    loss_anchor: float
    loss_curve: float
    sigma_anchor: float
    sigma_curve: float
    loss_continuity: float = 0.0
    loss_tail: float = 0.0
    loss_total: float = 0.0


@dataclass
class MonotonicityLog:
    """Track monotonicity violations in spline knots."""
    epoch: int
    batch: int
    region1_violations: int = 0
    region2_violations: int = 0
    total_samples: int = 0


@dataclass
class OutlierLog:
    """Track outlier detection results per target variable."""
    target_name: str
    n_samples: int
    n_outliers: int
    outlier_pct: float
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    n_below: int
    n_above: int
    min_value: float
    max_value: float
    mean_value: float


@dataclass
class ModelComparisonMetrics:
    """Standardized metrics for model comparison (Split-Spline vs CVAE)."""
    model_name: str
    mse_full_curve: float = 0.0
    mse_region1: float = 0.0
    mse_region2: float = 0.0
    mae_jsc: float = 0.0
    mae_voc: float = 0.0
    mae_vmpp: float = 0.0
    mae_jmpp: float = 0.0
    mape_ff: float = 0.0
    violations_jsc_negative: int = 0
    violations_voc_negative: int = 0
    violations_vmpp_invalid: int = 0
    violations_jmpp_invalid: int = 0
    violations_j_exceeds_jsc: int = 0
    inference_time_ms: float = 0.0
    total_samples: int = 0
    # New: more interpretable metrics
    r2_full_curve: float = 0.0
    r2_region1: float = 0.0
    r2_region2: float = 0.0
    nrmse_full_pct: float = 0.0  # Normalized RMSE as percentage
    median_curve_r2: float = 0.0  # Median per-sample R²

    @property
    def total_violations(self) -> int:
        return (self.violations_jsc_negative + self.violations_voc_negative +
                self.violations_vmpp_invalid + self.violations_jmpp_invalid +
                self.violations_j_exceeds_jsc)

    @property
    def violations_per_1000(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.total_violations / self.total_samples) * 1000


class TrainingLogger:
    """Centralized logging for the training pipeline."""

    def __init__(self, output_dir: str | Path, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Initialize log storage
        self.constraint_violations: list[ConstraintViolationLog] = []
        self.multitask_losses: list[MultiTaskLossLog] = []
        self.monotonicity_logs: list[MonotonicityLog] = []
        self.model_comparisons: list[ModelComparisonMetrics] = []
        self.outlier_logs: list[OutlierLog] = []
        self.feature_correlations: dict = {}
        self.multicollinearity_report: dict = {}

        # Start timestamp
        self.start_time = datetime.now()

    def log_constraint_violations(
        self,
        epoch: int,
        batch: int,
        raw_outputs,  # torch.Tensor (N, 4)
        total_samples: int
    ):
        """Log how often the network tries to violate constraints before projection."""
        import torch

        jsc_neg = (raw_outputs[:, 0] < 1e-6).sum().item()
        voc_neg = (raw_outputs[:, 1] < 1e-6).sum().item()
        vmpp_exceeds = (raw_outputs[:, 2] >= raw_outputs[:, 1]).sum().item()
        jmpp_exceeds = (raw_outputs[:, 3] >= raw_outputs[:, 0]).sum().item()

        log = ConstraintViolationLog(
            epoch=epoch,
            batch=batch,
            jsc_negative=jsc_neg,
            voc_negative=voc_neg,
            vmpp_exceeds_voc=vmpp_exceeds,
            jmpp_exceeds_jsc=jmpp_exceeds,
            total_samples=total_samples
        )
        self.constraint_violations.append(log)

        if self.verbose and batch == 0:  # Log once per epoch
            rate = log.violation_rate * 100
            print(f"  [Constraints] Violation rate: {rate:.2f}% "
                  f"(Jsc<0: {jsc_neg}, Voc<0: {voc_neg}, Vmpp>Voc: {vmpp_exceeds}, Jmpp>Jsc: {jmpp_exceeds})")

    def log_multitask_loss(
        self,
        epoch: int,
        loss_anchor: float,
        loss_curve: float,
        sigma_anchor: float,
        sigma_curve: float,
        loss_continuity: float = 0.0,
        loss_tail: float = 0.0,
        loss_total: float = 0.0
    ):
        """Log multi-task loss components and learned sigmas."""
        log = MultiTaskLossLog(
            epoch=epoch,
            loss_anchor=loss_anchor,
            loss_curve=loss_curve,
            sigma_anchor=sigma_anchor,
            sigma_curve=sigma_curve,
            loss_continuity=loss_continuity,
            loss_tail=loss_tail,
            loss_total=loss_total
        )
        self.multitask_losses.append(log)

        if self.verbose:
            # Check for task imbalance
            imbalance_warning = ""
            if sigma_anchor < 0.1 and sigma_curve > 0.5:
                imbalance_warning = " [!] Anchor task too easy"
            elif sigma_curve < 0.1 and sigma_anchor > 0.5:
                imbalance_warning = " [!] Curve task too easy"

            print(f"  [MultiTask] sigma_anchor={sigma_anchor:.4f}, sigma_curve={sigma_curve:.4f}, "
                  f"L_anchor={loss_anchor:.6f}, L_curve={loss_curve:.6f}{imbalance_warning}")

    def log_monotonicity(
        self,
        epoch: int,
        batch: int,
        region1_violations: int,
        region2_violations: int,
        total_samples: int
    ):
        """Log monotonicity violations in spline construction."""
        log = MonotonicityLog(
            epoch=epoch,
            batch=batch,
            region1_violations=region1_violations,
            region2_violations=region2_violations,
            total_samples=total_samples
        )
        self.monotonicity_logs.append(log)

        if self.verbose and (region1_violations > 0 or region2_violations > 0):
            print(f"  [Monotonicity] R1 violations: {region1_violations}, R2 violations: {region2_violations}")

    def log_model_comparison(self, metrics: ModelComparisonMetrics):
        """Log model comparison metrics."""
        self.model_comparisons.append(metrics)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Model: {metrics.model_name}")
            print(f"{'='*60}")
            print(f"  MSE (full curve): {metrics.mse_full_curve:.6f}")
            print(f"  FF MAPE: {metrics.mape_ff:.2f}%")
            print(f"  Violations per 1000 samples: {metrics.violations_per_1000:.2f}")
            print(f"  Inference time: {metrics.inference_time_ms:.2f} ms/sample")

    def log_multicollinearity(
        self,
        feature_names: list[str],
        corr_matrix: np.ndarray,
        threshold: float = 0.85
    ):
        """Log multicollinearity analysis results."""
        high_corr_pairs = []
        n_features = len(feature_names)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr_matrix[i, j])
                    })

        self.multicollinearity_report = {
            'threshold': threshold,
            'n_high_corr_pairs': len(high_corr_pairs),
            'pairs': high_corr_pairs
        }

        if self.verbose:
            print(f"\n[Multicollinearity] Found {len(high_corr_pairs)} pairs with |r| > {threshold}")
            for pair in high_corr_pairs[:10]:  # Show top 10
                print(f"  {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}")
            if len(high_corr_pairs) > 10:
                print(f"  ... and {len(high_corr_pairs) - 10} more pairs")

        return high_corr_pairs

    def log_outliers(
        self,
        target_name: str,
        values: np.ndarray,
        iqr_multiplier: float = 1.5
    ) -> OutlierLog:
        """
        Detect and log outliers using IQR method.

        Args:
            target_name: Name of the target variable (e.g., 'Jsc', 'Voc')
            values: Array of target values
            iqr_multiplier: Multiplier for IQR bounds (default 1.5)

        Returns:
            OutlierLog with detection results
        """
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        below_mask = values < lower_bound
        above_mask = values > upper_bound
        n_below = int(below_mask.sum())
        n_above = int(above_mask.sum())
        n_outliers = n_below + n_above

        log = OutlierLog(
            target_name=target_name,
            n_samples=len(values),
            n_outliers=n_outliers,
            outlier_pct=100.0 * n_outliers / len(values),
            q1=float(q1),
            q3=float(q3),
            iqr=float(iqr),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            n_below=n_below,
            n_above=n_above,
            min_value=float(values.min()),
            max_value=float(values.max()),
            mean_value=float(values.mean())
        )
        self.outlier_logs.append(log)

        if self.verbose:
            if n_outliers > 0:
                print(f"  [{target_name}] Outliers: {n_outliers} ({log.outlier_pct:.2f}%) "
                      f"[{n_below} below, {n_above} above]")
                print(f"           Range: [{log.min_value:.4f}, {log.max_value:.4f}], "
                      f"IQR bounds: [{log.lower_bound:.4f}, {log.upper_bound:.4f}]")
            else:
                print(f"  [{target_name}] No outliers detected (IQR method, multiplier={iqr_multiplier})")

        return log

    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table for all models."""
        if not self.model_comparisons:
            return "No model comparisons logged."

        # Primary table with interpretable metrics (R², NRMSE)
        table = "| Model | R² (Full) | R² (R1) | R² (R2) | NRMSE (%) | FF MAPE (%) | Violations/1000 |\n"
        table += "|-------|-----------|---------|---------|-----------|-------------|------------------|\n"

        for m in self.model_comparisons:
            table += f"| {m.model_name} | {m.r2_full_curve:.4f} | {m.r2_region1:.4f} | {m.r2_region2:.4f} | "
            table += f"{m.nrmse_full_pct:.2f} | {m.mape_ff:.2f} | {m.violations_per_1000:.2f} |\n"

        # Secondary table with raw MSE for reference
        table += "\n*Raw MSE values (for reference):*\n"
        table += "| Model | MSE Full | MSE R1 | MSE R2 |\n"
        table += "|-------|----------|--------|--------|\n"
        for m in self.model_comparisons:
            table += f"| {m.model_name} | {m.mse_full_curve:.2f} | {m.mse_region1:.2f} | {m.mse_region2:.2f} |\n"

        return table

    def save_all_logs(self):
        """Save all logs to files."""
        # Constraint violations
        if self.constraint_violations:
            with open(self.output_dir / 'constraint_violations.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.constraint_violations[0]).keys())
                writer.writeheader()
                for log in self.constraint_violations:
                    writer.writerow(asdict(log))

        # Multi-task losses
        if self.multitask_losses:
            with open(self.output_dir / 'multitask_losses.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.multitask_losses[0]).keys())
                writer.writeheader()
                for log in self.multitask_losses:
                    writer.writerow(asdict(log))

        # Monotonicity logs
        if self.monotonicity_logs:
            with open(self.output_dir / 'monotonicity.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.monotonicity_logs[0]).keys())
                writer.writeheader()
                for log in self.monotonicity_logs:
                    writer.writerow(asdict(log))

        # Model comparisons
        if self.model_comparisons:
            with open(self.output_dir / 'model_comparison.json', 'w') as f:
                json.dump([asdict(m) for m in self.model_comparisons], f, indent=2)

            # Also save markdown table
            with open(self.output_dir / 'model_comparison.md', 'w') as f:
                f.write("# Model Comparison Results\n\n")
                f.write(self.generate_comparison_table())

        # Outlier detection logs
        if self.outlier_logs:
            with open(self.output_dir / 'outlier_detection.json', 'w') as f:
                json.dump([asdict(log) for log in self.outlier_logs], f, indent=2)

            # Also save CSV for easy viewing
            with open(self.output_dir / 'outlier_detection.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.outlier_logs[0]).keys())
                writer.writeheader()
                for log in self.outlier_logs:
                    writer.writerow(asdict(log))

        # Multicollinearity report
        if self.multicollinearity_report:
            with open(self.output_dir / 'multicollinearity.json', 'w') as f:
                json.dump(self.multicollinearity_report, f, indent=2)

        # Summary
        self._save_training_summary()

    def _save_training_summary(self):
        """Save comprehensive training summary."""
        duration = datetime.now() - self.start_time

        summary = {
            'start_time': self.start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_human': str(duration),
            'n_constraint_violation_logs': len(self.constraint_violations),
            'n_multitask_loss_logs': len(self.multitask_losses),
            'n_monotonicity_logs': len(self.monotonicity_logs),
            'n_model_comparisons': len(self.model_comparisons),
        }

        # Add final sigma values if available
        if self.multitask_losses:
            final_loss = self.multitask_losses[-1]
            summary['final_sigma_anchor'] = final_loss.sigma_anchor
            summary['final_sigma_curve'] = final_loss.sigma_curve

        # Add constraint violation summary
        if self.constraint_violations:
            total_violations = sum(log.jsc_negative + log.voc_negative +
                                   log.vmpp_exceeds_voc + log.jmpp_exceeds_jsc
                                   for log in self.constraint_violations)
            total_samples = sum(log.total_samples for log in self.constraint_violations)
            summary['total_constraint_violations'] = total_violations
            summary['total_samples_checked'] = total_samples
            summary['overall_violation_rate'] = total_violations / max(1, total_samples * 4)

        # Add outlier detection summary
        if self.outlier_logs:
            summary['outlier_detection'] = {
                log.target_name: {
                    'n_outliers': log.n_outliers,
                    'outlier_pct': log.outlier_pct,
                    'n_below': log.n_below,
                    'n_above': log.n_above,
                    'iqr_bounds': [log.lower_bound, log.upper_bound]
                }
                for log in self.outlier_logs
            }
            total_outliers = sum(log.n_outliers for log in self.outlier_logs)
            summary['total_outliers_detected'] = total_outliers

        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Duration: {summary['duration_human']}")
        if 'final_sigma_anchor' in summary:
            print(f"Final sigma_anchor: {summary['final_sigma_anchor']:.4f}")
            print(f"Final sigma_curve: {summary['final_sigma_curve']:.4f}")
        if 'overall_violation_rate' in summary:
            print(f"Overall constraint violation rate: {summary['overall_violation_rate']*100:.2f}%")
        if self.model_comparisons:
            print("\nModel Comparison:")
            print(self.generate_comparison_table())
        print("=" * 60)


def compute_multicollinearity(features: np.ndarray, threshold: float = 0.85) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Compute feature correlation matrix and identify highly correlated pairs.

    Returns:
        corr_matrix: (n_features, n_features) correlation matrix
        high_corr_pairs: list of (i, j) tuples for features with |r| > threshold
    """
    # Standardize features
    std = features.std(axis=0)
    std[std < 1e-10] = 1.0  # Avoid division by zero
    features_norm = (features - features.mean(axis=0)) / std

    # Compute correlation matrix
    corr_matrix = np.corrcoef(features_norm.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Find high correlation pairs
    high_corr_pairs = []
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((i, j))

    return corr_matrix, high_corr_pairs


def suggest_features_to_drop(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    target_correlations: np.ndarray,
    threshold: float = 0.85
) -> list[int]:
    """
    Given highly correlated feature pairs, suggest which to drop.
    Strategy: Keep the feature with higher target correlation.

    Args:
        corr_matrix: Feature-feature correlation matrix
        feature_names: Names of features
        target_correlations: Max |r| with any target for each feature
        threshold: Correlation threshold for multicollinearity

    Returns:
        List of feature indices to drop
    """
    n = corr_matrix.shape[0]
    to_drop = set()

    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Drop the one with lower target correlation
                if target_correlations[i] >= target_correlations[j]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)

    return sorted(list(to_drop))
