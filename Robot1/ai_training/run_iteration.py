# run_iteration.py
# Single Iteration Runner for VRR AI Pipeline
# ============================================
# Executes one complete iteration of the DAgger-style training loop:
#   1. Create/Update dataset manifest
#   2. Train model
#   3. Evaluate results
#   4. Check stop conditions
#
# The AI rollout (running in Unity simulator) must be done manually
# between iterations. This script handles the ML pipeline parts.
#
# Usage:
#   python scripts/run_iteration.py --iteration 1    # Run iteration 1
#   python scripts/run_iteration.py --iteration 2    # Run iteration 2 (after new data collected)

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Add parent directory and scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_filter import DatasetManager, RunAnalyzer
from train import train, load_config, set_seed


class IterationRunner:
    """
    Manages a single iteration of the DAgger-style training loop.

    Responsibilities:
    - Create dataset manifest for the iteration
    - Run training
    - Evaluate results
    - Check stop conditions
    - Generate iteration report
    """

    def __init__(self, robot_dir: Path, config: Dict):
        """
        Initialize iteration runner.

        Args:
            robot_dir: Path to Robot1 directory
            config: Configuration dictionary
        """
        self.robot_dir = Path(robot_dir)
        self.config = config

        self.training_data_dir = self.robot_dir / config['paths']['training_data']
        self.experiments_dir = self.robot_dir / config['paths']['experiments']

    def get_iteration_dir(self, iteration: int) -> Path:
        """Get directory for a specific iteration."""
        return self.experiments_dir / f"iteration_{iteration:03d}"

    def run(self, iteration: int) -> Dict:
        """
        Run a complete iteration.

        Args:
            iteration: Iteration number (1-indexed)

        Returns:
            Iteration results dictionary
        """
        print("\n" + "=" * 70)
        print(f"VRR AI Pipeline - Iteration {iteration}")
        print("=" * 70)

        iteration_dir = self.get_iteration_dir(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "iteration": iteration,
            "started_at": datetime.now().isoformat(),
            "status": "running"
        }

        try:
            # Step 1: Create/Update dataset manifest
            print("\n[Step 1/4] Creating dataset manifest...")
            manifest_results = self._create_manifest(iteration)
            results["manifest"] = manifest_results

            if manifest_results["total_runs"] == 0:
                results["status"] = "failed"
                results["error"] = "No valid training data found"
                return results

            # Step 2: Train model
            print("\n[Step 2/4] Training model...")
            manifest_path = iteration_dir / "dataset_manifest.json"
            train_results = train(
                self.config,
                iteration,
                self.robot_dir,
                manifest_path
            )
            results["training"] = train_results

            # Step 3: Evaluate results
            print("\n[Step 3/4] Evaluating results...")
            eval_results = self._evaluate(iteration, train_results)
            results["evaluation"] = eval_results

            # Step 4: Check stop conditions
            print("\n[Step 4/4] Checking stop conditions...")
            stop_check = self._check_stop_conditions(iteration, eval_results)
            results["stop_check"] = stop_check

            results["status"] = "completed"
            results["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            import traceback
            results["traceback"] = traceback.format_exc()
            print(f"\n[Error] Iteration failed: {e}")

        # Save iteration results
        results_path = iteration_dir / "iteration_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._print_summary(results)

        return results

    def _create_manifest(self, iteration: int) -> Dict:
        """Create dataset manifest for this iteration."""
        manager = DatasetManager(self.training_data_dir, self.experiments_dir)

        # Get filtering config
        filter_config = self.config.get('data_filtering', {})
        filtering_rules = {
            'min_frames': filter_config.get('min_frames', 100),
            'valid_final_status': filter_config.get('valid_final_status', ['Finish', 'Lap2', 'Lap1']),
            'exclude_fallen': filter_config.get('exclude_conditions', {}).get('fallen', True),
            'exclude_force_end': filter_config.get('exclude_conditions', {}).get('force_end', False),
        }

        manifest = manager.create_manifest(
            iteration=iteration,
            config=filtering_rules,
            description=f"Dataset for iteration {iteration}"
        )

        manager.save_manifest(manifest, iteration)

        return {
            "total_runs": manifest["statistics"]["total_runs"],
            "total_frames": manifest["statistics"]["total_frames"],
            "total_racing_frames": manifest["statistics"]["total_racing_frames"],
        }

    def _evaluate(self, iteration: int, train_results: Dict) -> Dict:
        """Evaluate training results."""
        # Basic evaluation from training results
        eval_results = {
            "best_val_loss": train_results.get("best_val_loss", float('inf')),
            "best_epoch": train_results.get("best_epoch", 0),
            "total_epochs": train_results.get("total_epochs", 0),
            "total_samples": train_results.get("total_samples", 0),
        }

        # Compare with previous iteration if available
        if iteration > 1:
            prev_results = self._load_previous_results(iteration - 1)
            if prev_results:
                prev_loss = prev_results.get("training", {}).get("best_val_loss", float('inf'))
                improvement = (prev_loss - eval_results["best_val_loss"]) / prev_loss if prev_loss > 0 else 0
                eval_results["improvement_from_previous"] = improvement
                eval_results["previous_val_loss"] = prev_loss

        return eval_results

    def _check_stop_conditions(self, iteration: int, eval_results: Dict) -> Dict:
        """Check if training should stop."""
        iter_config = self.config.get('iteration', {})
        stop_config = iter_config.get('stop_conditions', {})

        checks = {
            "should_stop": False,
            "reasons": [],
            "max_iterations_reached": False,
            "performance_target_met": False,
            "no_improvement": False,
        }

        # Check max iterations
        max_iter = iter_config.get('max_iterations', 10)
        if iteration >= max_iter:
            checks["max_iterations_reached"] = True
            checks["reasons"].append(f"Reached maximum iterations ({max_iter})")
            checks["should_stop"] = True

        # Check consecutive no improvement
        if iteration > 1:
            consecutive_threshold = stop_config.get('consecutive_no_improve', 3)
            no_improve_count = self._count_consecutive_no_improvement(iteration)

            if no_improve_count >= consecutive_threshold:
                checks["no_improvement"] = True
                checks["reasons"].append(
                    f"No improvement for {no_improve_count} consecutive iterations"
                )
                checks["should_stop"] = True

        # Note: completion_rate check requires actual rollout data
        # This will be checked when rollout results are available

        return checks

    def _load_previous_results(self, iteration: int) -> Optional[Dict]:
        """Load results from a previous iteration."""
        results_path = self.get_iteration_dir(iteration) / "iteration_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _count_consecutive_no_improvement(self, current_iteration: int) -> int:
        """Count consecutive iterations without improvement."""
        count = 0
        best_loss = float('inf')

        for i in range(current_iteration, 0, -1):
            results = self._load_previous_results(i)
            if not results:
                break

            val_loss = results.get("training", {}).get("best_val_loss", float('inf'))

            if val_loss < best_loss:
                best_loss = val_loss
                break
            else:
                count += 1

        return count

    def _print_summary(self, results: Dict):
        """Print iteration summary."""
        print("\n" + "=" * 70)
        print(f"Iteration {results['iteration']} Summary")
        print("=" * 70)

        print(f"\nStatus: {results['status'].upper()}")

        if results['status'] == 'completed':
            manifest = results.get('manifest', {})
            training = results.get('training', {})
            evaluation = results.get('evaluation', {})
            stop_check = results.get('stop_check', {})

            print(f"\nDataset:")
            print(f"  Runs: {manifest.get('total_runs', 'N/A')}")
            print(f"  Racing frames: {manifest.get('total_racing_frames', 'N/A')}")

            print(f"\nTraining:")
            print(f"  Best epoch: {training.get('best_epoch', 'N/A')}")
            print(f"  Best val loss: {training.get('best_val_loss', 'N/A'):.6f}")

            if 'improvement_from_previous' in evaluation:
                imp = evaluation['improvement_from_previous']
                print(f"  Improvement: {imp:+.2%}")

            print(f"\nStop conditions:")
            if stop_check.get('should_stop'):
                print(f"  STOP RECOMMENDED")
                for reason in stop_check.get('reasons', []):
                    print(f"    - {reason}")
            else:
                print(f"  Continue training")

        elif results['status'] == 'failed':
            print(f"\nError: {results.get('error', 'Unknown error')}")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VRR AI Pipeline - Single Iteration Runner")
    parser.add_argument("--iteration", type=int, required=True,
                        help="Iteration number to run")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = robot_dir / "experiments" / "config.yaml"

    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)

    # Run iteration
    runner = IterationRunner(robot_dir, config)
    results = runner.run(args.iteration)

    # Exit with appropriate code
    if results['status'] == 'failed':
        sys.exit(1)
    elif results.get('stop_check', {}).get('should_stop'):
        print("\n[Info] Stop condition met. Consider ending the training loop.")
        sys.exit(0)
    else:
        print("\n[Info] Iteration complete. Ready for next iteration.")
        sys.exit(0)


if __name__ == "__main__":
    main()
