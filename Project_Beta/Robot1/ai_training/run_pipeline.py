# run_pipeline.py
# Full Pipeline Orchestrator for VRR AI Training
# ===============================================
# Orchestrates the complete DAgger-style training pipeline.
#
# The pipeline consists of:
#   1. Initial training on expert data
#   2. Iterative loop:
#      a. AI rollout in simulator (MANUAL - requires Unity)
#      b. Data aggregation
#      c. Retraining
#      d. Stop condition check
#
# Since Unity rollouts require manual intervention, this script provides:
#   - Pipeline status tracking
#   - Next action guidance
#   - Progress visualization
#
# Usage:
#   python scripts/run_pipeline.py status          # Show current status
#   python scripts/run_pipeline.py next            # Run next available step
#   python scripts/run_pipeline.py train           # Train current iteration
#   python scripts/run_pipeline.py report          # Generate progress report

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_filter import DatasetManager
from run_iteration import IterationRunner


class PipelineOrchestrator:
    """
    Orchestrates the full training pipeline across multiple iterations.

    Tracks state, provides guidance, and manages transitions between
    pipeline stages.
    """

    # Pipeline stages
    STAGE_INIT = "init"
    STAGE_TRAIN = "train"
    STAGE_ROLLOUT = "rollout"
    STAGE_AGGREGATE = "aggregate"
    STAGE_COMPLETE = "complete"

    def __init__(self, robot_dir: Path, config: Dict):
        """
        Initialize orchestrator.

        Args:
            robot_dir: Path to Robot1 directory
            config: Configuration dictionary
        """
        self.robot_dir = Path(robot_dir)
        self.config = config

        self.experiments_dir = self.robot_dir / config['paths']['experiments']
        self.training_data_dir = self.robot_dir / config['paths']['training_data']
        self.state_file = self.experiments_dir / "pipeline_state.json"

    def get_state(self) -> Dict:
        """Load or initialize pipeline state."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "current_iteration": 0,
                "current_stage": self.STAGE_INIT,
                "iterations_completed": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

    def save_state(self, state: Dict):
        """Save pipeline state."""
        state["updated_at"] = datetime.now().isoformat()
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def get_iteration_results(self, iteration: int) -> Optional[Dict]:
        """Load results for a specific iteration."""
        results_path = self.experiments_dir / f"iteration_{iteration:03d}" / "iteration_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def count_new_runs_since(self, iteration: int) -> int:
        """Count new runs added since last iteration."""
        # Get runs from current manifest
        current_manifest_path = self.experiments_dir / f"iteration_{iteration:03d}" / "dataset_manifest.json"
        current_runs = set()

        if current_manifest_path.exists():
            with open(current_manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                current_runs = {r["run_name"] for r in manifest.get("runs", [])}

        # Count all runs in training_data
        all_runs = set()
        for d in self.training_data_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                if (d / "metadata.csv").exists():
                    all_runs.add(d.name)

        # New runs = all runs - current runs
        new_runs = all_runs - current_runs
        return len(new_runs)

    def status(self) -> Dict:
        """Get comprehensive pipeline status."""
        state = self.get_state()

        status = {
            "state": state,
            "iterations": [],
            "recommendations": [],
        }

        # Gather iteration info
        max_iter = self.config.get('iteration', {}).get('max_iterations', 10)

        for i in range(1, state["current_iteration"] + 2):
            iter_dir = self.experiments_dir / f"iteration_{i:03d}"
            iter_info = {"iteration": i, "exists": iter_dir.exists()}

            if iter_dir.exists():
                results = self.get_iteration_results(i)
                if results:
                    iter_info["status"] = results.get("status", "unknown")
                    iter_info["val_loss"] = results.get("training", {}).get("best_val_loss")
                    iter_info["samples"] = results.get("training", {}).get("total_samples")

                # Check for model
                iter_info["has_model"] = (iter_dir / "model.pth").exists()
                iter_info["has_manifest"] = (iter_dir / "dataset_manifest.json").exists()

            status["iterations"].append(iter_info)

        # Generate recommendations
        current_iter = state["current_iteration"]
        current_stage = state["current_stage"]

        if current_stage == self.STAGE_INIT:
            status["recommendations"].append({
                "action": "train",
                "description": "Start initial training with expert data",
                "command": "python scripts/run_pipeline.py train"
            })

        elif current_stage == self.STAGE_TRAIN:
            # Check if we should stop
            if current_iter > 0:
                results = self.get_iteration_results(current_iter)
                if results and results.get("stop_check", {}).get("should_stop"):
                    status["recommendations"].append({
                        "action": "stop",
                        "description": "Stop conditions met. Training complete!",
                        "reasons": results["stop_check"].get("reasons", [])
                    })
                else:
                    status["recommendations"].append({
                        "action": "rollout",
                        "description": "Run AI in simulator to collect new data",
                        "command": "Set MODE_NUM=4 in robot_config.txt, then run main.py"
                    })

        elif current_stage == self.STAGE_ROLLOUT:
            new_runs = self.count_new_runs_since(current_iter)
            if new_runs > 0:
                status["recommendations"].append({
                    "action": "aggregate",
                    "description": f"New data available ({new_runs} runs). Ready to aggregate and train.",
                    "command": "python scripts/run_pipeline.py train"
                })
            else:
                status["recommendations"].append({
                    "action": "collect",
                    "description": "Run more AI rollouts to collect training data",
                    "command": "Set MODE_NUM=4 in robot_config.txt, run main.py"
                })

        # Check max iterations
        if current_iter >= max_iter:
            status["recommendations"].append({
                "action": "complete",
                "description": f"Maximum iterations ({max_iter}) reached",
            })

        return status

    def train_next(self) -> Dict:
        """Train the next iteration."""
        state = self.get_state()
        next_iter = state["current_iteration"] + 1

        print(f"\n[Pipeline] Starting iteration {next_iter}...")

        # Run iteration
        runner = IterationRunner(self.robot_dir, self.config)
        results = runner.run(next_iter)

        # Update state
        if results["status"] == "completed":
            state["current_iteration"] = next_iter
            state["current_stage"] = self.STAGE_ROLLOUT
            state["iterations_completed"].append({
                "iteration": next_iter,
                "completed_at": datetime.now().isoformat(),
                "val_loss": results.get("training", {}).get("best_val_loss")
            })

            # Check stop conditions
            if results.get("stop_check", {}).get("should_stop"):
                state["current_stage"] = self.STAGE_COMPLETE

        self.save_state(state)
        return results

    def print_status(self):
        """Print formatted status report."""
        status = self.status()
        state = status["state"]

        print("\n" + "=" * 70)
        print("VRR AI Training Pipeline - Status Report")
        print("=" * 70)

        print(f"\nCurrent State:")
        print(f"  Iteration: {state['current_iteration']}")
        print(f"  Stage: {state['current_stage']}")
        print(f"  Last updated: {state.get('updated_at', 'N/A')}")

        print(f"\nIteration History:")
        print("-" * 50)

        if not status["iterations"]:
            print("  No iterations yet")
        else:
            for iter_info in status["iterations"]:
                i = iter_info["iteration"]
                if iter_info.get("exists"):
                    st = iter_info.get("status", "?")
                    loss = iter_info.get("val_loss")
                    samples = iter_info.get("samples")
                    loss_str = f"{loss:.6f}" if loss else "N/A"
                    print(f"  Iteration {i}: {st.upper():10} | Val Loss: {loss_str} | Samples: {samples or 'N/A'}")
                else:
                    print(f"  Iteration {i}: NOT STARTED")

        print(f"\nRecommended Actions:")
        print("-" * 50)

        for rec in status["recommendations"]:
            print(f"\n  [{rec['action'].upper()}]")
            print(f"    {rec['description']}")
            if "command" in rec:
                print(f"    Command: {rec['command']}")
            if "reasons" in rec:
                for reason in rec["reasons"]:
                    print(f"    - {reason}")

        print("\n" + "=" * 70)

    def generate_report(self) -> str:
        """Generate a markdown progress report."""
        status = self.status()
        state = status["state"]

        lines = [
            "# VRR AI Training Pipeline - Progress Report",
            f"\n**Generated**: {datetime.now().isoformat()}",
            f"\n**Current Iteration**: {state['current_iteration']}",
            f"\n**Stage**: {state['current_stage']}",
            "\n---",
            "\n## Iteration History\n",
            "| Iteration | Status | Val Loss | Samples |",
            "|-----------|--------|----------|---------|",
        ]

        for iter_info in status["iterations"]:
            if iter_info.get("exists"):
                i = iter_info["iteration"]
                st = iter_info.get("status", "?")
                loss = iter_info.get("val_loss")
                samples = iter_info.get("samples")
                loss_str = f"{loss:.6f}" if loss else "N/A"
                lines.append(f"| {i} | {st} | {loss_str} | {samples or 'N/A'} |")

        lines.extend([
            "\n---",
            "\n## Training Progress\n",
        ])

        # Add loss trend if multiple iterations
        completed = [it for it in status["iterations"] if it.get("val_loss")]
        if len(completed) > 1:
            lines.append("### Validation Loss Trend\n")
            lines.append("```")
            for it in completed:
                loss = it["val_loss"]
                bar_len = int(50 * (1 - min(loss, 1)))
                bar = "#" * bar_len
                lines.append(f"Iter {it['iteration']}: {loss:.4f} |{bar}")
            lines.append("```")

        lines.extend([
            "\n---",
            "\n## Next Steps\n",
        ])

        for rec in status["recommendations"]:
            lines.append(f"### {rec['action'].upper()}")
            lines.append(f"\n{rec['description']}\n")
            if "command" in rec:
                lines.append(f"```bash\n{rec['command']}\n```\n")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VRR AI Pipeline Orchestrator")
    parser.add_argument("command", choices=["status", "train", "next", "report", "reset"],
                        help="Command to execute")
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
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)

    orchestrator = PipelineOrchestrator(robot_dir, config)

    if args.command == "status":
        orchestrator.print_status()

    elif args.command in ["train", "next"]:
        results = orchestrator.train_next()
        if results["status"] == "failed":
            sys.exit(1)

    elif args.command == "report":
        report = orchestrator.generate_report()
        report_path = robot_dir / "experiments" / "progress_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
        print("\n" + report)

    elif args.command == "reset":
        state_file = robot_dir / "experiments" / "pipeline_state.json"
        if state_file.exists():
            state_file.unlink()
            print("[Pipeline] State reset successfully")
        else:
            print("[Pipeline] No state file to reset")


if __name__ == "__main__":
    main()
