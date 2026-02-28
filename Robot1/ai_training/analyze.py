# analyze.py
# Analysis and Visualization Tools for VRR AI Pipeline
# =====================================================
# Provides tools for:
# - Training curve visualization
# - Iteration comparison
# - Data quality analysis
# - Progress reporting
#
# Usage:
#   python scripts/analyze.py plot --iteration 1     # Plot training curves
#   python scripts/analyze.py compare                # Compare all iterations
#   python scripts/analyze.py data                   # Analyze training data
#   python scripts/analyze.py summary                # Full summary report

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not installed. Install with: pip install matplotlib")


class TrainingAnalyzer:
    """Analyzes training logs and generates visualizations."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)

    def get_iterations(self) -> List[int]:
        """Get list of completed iterations."""
        iterations = []
        for d in sorted(self.experiments_dir.iterdir()):
            if d.is_dir() and d.name.startswith("iteration_"):
                try:
                    num = int(d.name.split("_")[1])
                    if (d / "training_log.csv").exists():
                        iterations.append(num)
                except (ValueError, IndexError):
                    pass
        return iterations

    def load_training_log(self, iteration: int) -> Optional[pd.DataFrame]:
        """Load training log for an iteration."""
        log_path = self.experiments_dir / f"iteration_{iteration:03d}" / "training_log.csv"
        if log_path.exists():
            return pd.read_csv(log_path)
        return None

    def load_iteration_results(self, iteration: int) -> Optional[Dict]:
        """Load iteration results JSON."""
        results_path = self.experiments_dir / f"iteration_{iteration:03d}" / "iteration_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def plot_training_curves(self, iteration: int, save_path: Optional[Path] = None, show: bool = True):
        """
        Plot training and validation loss curves for an iteration.

        Args:
            iteration: Iteration number
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if not HAS_MATPLOTLIB:
            print("[Error] matplotlib required for plotting")
            return

        df = self.load_training_log(iteration)
        if df is None:
            print(f"[Error] No training log found for iteration {iteration}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Analysis - Iteration {iteration}', fontsize=14, fontweight='bold')

        # Plot 1: Total Loss
        ax1 = axes[0, 0]
        ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=1.5)

        # Mark best epoch
        best_idx = df['val_loss'].idxmin()
        best_epoch = df.loc[best_idx, 'epoch']
        best_loss = df.loc[best_idx, 'val_loss']
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
        ax1.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5, marker='*')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: Torque Loss
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['train_torque_loss'], 'b-', label='Train', linewidth=1.5)
        ax2.plot(df['epoch'], df['val_torque_loss'], 'r-', label='Val', linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Torque Loss (Drive)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Steering Loss
        ax3 = axes[1, 0]
        ax3.plot(df['epoch'], df['train_steer_loss'], 'b-', label='Train', linewidth=1.5)
        ax3.plot(df['epoch'], df['val_steer_loss'], 'r-', label='Val', linewidth=1.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Steering Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        ax4 = axes[1, 1]
        ax4.plot(df['epoch'], df['learning_rate'], 'g-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyze] Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def compare_iterations(self, save_path: Optional[Path] = None, show: bool = True):
        """
        Compare training progress across all iterations.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if not HAS_MATPLOTLIB:
            print("[Error] matplotlib required for plotting")
            return

        iterations = self.get_iterations()
        if not iterations:
            print("[Error] No iterations found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Iteration Comparison', fontsize=14, fontweight='bold')

        colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))

        # Collect summary data
        summary_data = []

        for idx, iteration in enumerate(iterations):
            df = self.load_training_log(iteration)
            results = self.load_iteration_results(iteration)

            if df is None:
                continue

            color = colors[idx]
            label = f'Iter {iteration}'

            # Plot 1: Validation Loss curves
            axes[0, 0].plot(df['epoch'], df['val_loss'], color=color, label=label, linewidth=1.5)

            # Collect summary
            if results:
                summary_data.append({
                    'iteration': iteration,
                    'best_val_loss': results.get('training', {}).get('best_val_loss', df['val_loss'].min()),
                    'best_epoch': results.get('training', {}).get('best_epoch', df['val_loss'].idxmin() + 1),
                    'total_epochs': results.get('training', {}).get('total_epochs', len(df)),
                    'samples': results.get('training', {}).get('total_samples', 0),
                })

        # Plot 1: Validation Loss
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].set_title('Validation Loss Over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Plot 2: Best Val Loss per Iteration
            axes[0, 1].bar(summary_df['iteration'], summary_df['best_val_loss'], color='steelblue')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Best Validation Loss')
            axes[0, 1].set_title('Best Loss per Iteration')
            axes[0, 1].grid(True, alpha=0.3, axis='y')

            # Plot 3: Training Duration (epochs)
            axes[1, 0].bar(summary_df['iteration'], summary_df['total_epochs'], color='coral')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Epochs')
            axes[1, 0].set_title('Training Duration')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Plot 4: Dataset Size
            axes[1, 1].bar(summary_df['iteration'], summary_df['samples'], color='mediumseagreen')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Samples')
            axes[1, 1].set_title('Dataset Size')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyze] Comparison plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all iterations."""
        iterations = self.get_iterations()
        rows = []

        for iteration in iterations:
            results = self.load_iteration_results(iteration)
            df = self.load_training_log(iteration)

            if results:
                training = results.get('training', {})
                manifest = results.get('manifest', {})

                rows.append({
                    'Iteration': iteration,
                    'Status': results.get('status', 'unknown'),
                    'Samples': training.get('total_samples', 0),
                    'Runs': manifest.get('total_runs', 0),
                    'Epochs': training.get('total_epochs', 0),
                    'Best Epoch': training.get('best_epoch', 0),
                    'Best Val Loss': training.get('best_val_loss', 0),
                    'Final Val Loss': training.get('final_val_loss', 0),
                })
            elif df is not None:
                rows.append({
                    'Iteration': iteration,
                    'Status': 'partial',
                    'Samples': 0,
                    'Runs': 0,
                    'Epochs': len(df),
                    'Best Epoch': df['val_loss'].idxmin() + 1,
                    'Best Val Loss': df['val_loss'].min(),
                    'Final Val Loss': df['val_loss'].iloc[-1],
                })

        return pd.DataFrame(rows)


class DataAnalyzer:
    """Analyzes training data quality."""

    def __init__(self, training_data_dir: Path):
        self.training_data_dir = Path(training_data_dir)

    def get_runs(self) -> List[Path]:
        """Get all run directories."""
        runs = []
        for d in sorted(self.training_data_dir.iterdir()):
            if d.is_dir() and d.name.startswith("run_"):
                if (d / "metadata.csv").exists():
                    runs.append(d)
        return runs

    def analyze_run(self, run_dir: Path) -> Dict:
        """Analyze a single run."""
        metadata_path = run_dir / "metadata.csv"
        df = pd.read_csv(metadata_path)

        # Basic stats
        total_frames = len(df)
        racing_frames = len(df[df['status'].isin(['Lap1', 'Lap2', 'Finish'])])

        # Final status
        final_status = df.iloc[-1]['status'] if len(df) > 0 else 'Unknown'

        # Time
        race_time = df['race_time_ms'].max() / 1000.0 if 'race_time_ms' in df.columns else 0

        # Control stats
        if 'drive_torque' in df.columns and 'steer_angle' in df.columns:
            racing_df = df[df['status'].isin(['Lap1', 'Lap2', 'Finish'])]
            if len(racing_df) > 0:
                torque_mean = racing_df['drive_torque'].mean()
                torque_std = racing_df['drive_torque'].std()
                steer_mean = racing_df['steer_angle'].mean()
                steer_std = racing_df['steer_angle'].std()
            else:
                torque_mean = torque_std = steer_mean = steer_std = 0
        else:
            torque_mean = torque_std = steer_mean = steer_std = 0

        return {
            'run_name': run_dir.name,
            'final_status': final_status,
            'total_frames': total_frames,
            'racing_frames': racing_frames,
            'race_time_sec': race_time,
            'torque_mean': torque_mean,
            'torque_std': torque_std,
            'steer_mean': steer_mean,
            'steer_std': steer_std,
        }

    def generate_data_report(self) -> pd.DataFrame:
        """Generate report of all training runs."""
        runs = self.get_runs()
        rows = [self.analyze_run(r) for r in runs]
        return pd.DataFrame(rows)

    def plot_control_distribution(self, save_path: Optional[Path] = None, show: bool = True):
        """Plot distribution of control values across all runs."""
        if not HAS_MATPLOTLIB:
            print("[Error] matplotlib required for plotting")
            return

        runs = self.get_runs()
        all_torque = []
        all_steer = []

        for run_dir in runs:
            df = pd.read_csv(run_dir / "metadata.csv")
            racing_df = df[df['status'].isin(['Lap1', 'Lap2', 'Finish'])]

            if 'drive_torque' in racing_df.columns:
                all_torque.extend(racing_df['drive_torque'].tolist())
            if 'steer_angle' in racing_df.columns:
                all_steer.extend(racing_df['steer_angle'].tolist())

        if not all_torque or not all_steer:
            print("[Error] No control data found")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Control Value Distribution (Training Data)', fontsize=12, fontweight='bold')

        # Torque histogram
        axes[0].hist(all_torque, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Drive Torque')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Torque Distribution (n={len(all_torque)})')
        axes[0].axvline(x=np.mean(all_torque), color='red', linestyle='--', label=f'Mean: {np.mean(all_torque):.3f}')
        axes[0].legend()

        # Steering histogram
        axes[1].hist(all_steer, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Steering Angle (rad)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Steering Distribution (n={len(all_steer)})')
        axes[1].axvline(x=np.mean(all_steer), color='red', linestyle='--', label=f'Mean: {np.mean(all_steer):.3f}')
        axes[1].legend()

        # 2D scatter (subsampled)
        sample_size = min(5000, len(all_torque))
        indices = np.random.choice(len(all_torque), sample_size, replace=False)
        sampled_torque = [all_torque[i] for i in indices]
        sampled_steer = [all_steer[i] for i in indices]

        axes[2].scatter(sampled_steer, sampled_torque, alpha=0.3, s=5)
        axes[2].set_xlabel('Steering Angle (rad)')
        axes[2].set_ylabel('Drive Torque')
        axes[2].set_title(f'Control Space (sampled n={sample_size})')
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[2].axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyze] Control distribution plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def print_summary(experiments_dir: Path, training_data_dir: Path):
    """Print comprehensive summary."""
    print("\n" + "=" * 70)
    print("VRR AI Pipeline - Analysis Summary")
    print("=" * 70)

    # Training data summary
    data_analyzer = DataAnalyzer(training_data_dir)
    data_report = data_analyzer.generate_data_report()

    print("\n[Training Data]")
    print("-" * 50)
    if len(data_report) > 0:
        print(f"  Total runs: {len(data_report)}")
        print(f"  Total racing frames: {data_report['racing_frames'].sum()}")
        print(f"  Status breakdown:")
        for status in data_report['final_status'].unique():
            count = (data_report['final_status'] == status).sum()
            print(f"    {status}: {count}")
    else:
        print("  No training data found")

    # Iteration summary
    training_analyzer = TrainingAnalyzer(experiments_dir)
    iter_report = training_analyzer.generate_summary_table()

    print("\n[Iterations]")
    print("-" * 50)
    if len(iter_report) > 0:
        print(iter_report.to_string(index=False))

        # Progress analysis
        if len(iter_report) > 1:
            first_loss = iter_report.iloc[0]['Best Val Loss']
            last_loss = iter_report.iloc[-1]['Best Val Loss']
            improvement = (first_loss - last_loss) / first_loss * 100
            print(f"\n  Overall improvement: {improvement:.1f}%")
    else:
        print("  No iterations completed")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VRR AI Pipeline Analysis Tools")
    parser.add_argument("command", choices=["plot", "compare", "data", "summary"],
                        help="Analysis command")
    parser.add_argument("--iteration", "-i", type=int, default=1,
                        help="Iteration number (for plot command)")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save plots to files")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (useful for batch processing)")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent
    experiments_dir = robot_dir / "experiments"
    training_data_dir = robot_dir / "training_data"

    show = not args.no_show

    if args.command == "plot":
        analyzer = TrainingAnalyzer(experiments_dir)
        save_path = None
        if args.save:
            save_path = experiments_dir / f"iteration_{args.iteration:03d}" / "training_curves.png"
        analyzer.plot_training_curves(args.iteration, save_path=save_path, show=show)

    elif args.command == "compare":
        analyzer = TrainingAnalyzer(experiments_dir)
        save_path = None
        if args.save:
            save_path = experiments_dir / "iteration_comparison.png"
        analyzer.compare_iterations(save_path=save_path, show=show)

    elif args.command == "data":
        analyzer = DataAnalyzer(training_data_dir)
        save_path = None
        if args.save:
            save_path = experiments_dir / "control_distribution.png"
        analyzer.plot_control_distribution(save_path=save_path, show=show)

        print("\n[Training Data Report]")
        print("-" * 50)
        report = analyzer.generate_data_report()
        print(report.to_string(index=False))

    elif args.command == "summary":
        print_summary(experiments_dir, training_data_dir)

        # Also generate and save plots if matplotlib available
        if HAS_MATPLOTLIB and args.save:
            training_analyzer = TrainingAnalyzer(experiments_dir)
            iterations = training_analyzer.get_iterations()

            for iteration in iterations:
                save_path = experiments_dir / f"iteration_{iteration:03d}" / "training_curves.png"
                training_analyzer.plot_training_curves(iteration, save_path=save_path, show=False)

            if len(iterations) > 0:
                save_path = experiments_dir / "iteration_comparison.png"
                training_analyzer.compare_iterations(save_path=save_path, show=False)

            data_analyzer = DataAnalyzer(training_data_dir)
            save_path = experiments_dir / "control_distribution.png"
            data_analyzer.plot_control_distribution(save_path=save_path, show=False)


if __name__ == "__main__":
    main()
