# auto_collect.py
# Automated Data Collection Script for VRR AI Training
# =====================================================
# Runs rule-based driving multiple times to collect training data automatically.
#
# Usage:
#   python scripts/auto_collect.py --runs 10
#   python scripts/auto_collect.py --runs 20 --timeout 120
#
# This script:
#   1. Runs main.py with MODE_NUM=3 (rule-based)
#   2. Waits for race completion
#   3. Checks if the run was successful (Finish status)
#   4. Repeats for specified number of runs
#   5. Logs all results with timing information

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


class AutoCollector:
    """
    Automated data collection using rule-based driving.
    """

    def __init__(self, robot_dir: Path, target_runs: int = 10, timeout: int = 120):
        """
        Initialize collector.

        Args:
            robot_dir: Path to Robot1 directory
            target_runs: Number of successful runs to collect
            timeout: Maximum seconds to wait for each run
        """
        self.robot_dir = Path(robot_dir)
        self.project_dir = self.robot_dir.parent
        self.target_runs = target_runs
        self.timeout = timeout

        self.training_data_dir = self.robot_dir / "training_data"
        self.log_file = self.robot_dir / "scripts" / "auto_collect_log.json"

        # Config file paths
        self.global_config = self.project_dir / "config.txt"
        self.robot_config = self.robot_dir / "robot_config.txt"

        # Original config values (for restoration)
        self.original_configs = {}

        # Results tracking
        self.results = {
            "started_at": datetime.now().isoformat(),
            "target_runs": target_runs,
            "timeout_per_run": timeout,
            "runs": [],
            "summary": {}
        }

    def backup_and_set_config(self):
        """
        Backup original config and set required values for auto collection.
        """
        print("[AutoCollect] Configuring for rule-based auto collection...")

        # Read and backup global config
        if self.global_config.exists():
            with open(self.global_config, "r", encoding="utf-8") as f:
                self.original_configs["global"] = f.read()

            # Set DEBUG_MODE=0 for auto Unity launch
            content = self.original_configs["global"]
            if "DEBUG_MODE=1" in content:
                content = content.replace("DEBUG_MODE=1", "DEBUG_MODE=0")
                with open(self.global_config, "w", encoding="utf-8") as f:
                    f.write(content)
                print("  - Set DEBUG_MODE=0 (auto Unity launch)")

        # Read and backup robot config
        if self.robot_config.exists():
            with open(self.robot_config, "r", encoding="utf-8") as f:
                self.original_configs["robot"] = f.read()

            # Set MODE_NUM=3 for rule-based driving
            content = self.original_configs["robot"]
            import re
            if re.search(r"MODE_NUM=\d", content):
                content = re.sub(r"MODE_NUM=\d", "MODE_NUM=3", content)
                with open(self.robot_config, "w", encoding="utf-8") as f:
                    f.write(content)
                print("  - Set MODE_NUM=3 (rule-based driving)")

    def restore_config(self):
        """
        Restore original config values.
        """
        print("[AutoCollect] Restoring original configuration...")

        if "global" in self.original_configs:
            with open(self.global_config, "w", encoding="utf-8") as f:
                f.write(self.original_configs["global"])
            print("  - Restored global config")

        if "robot" in self.original_configs:
            with open(self.robot_config, "w", encoding="utf-8") as f:
                f.write(self.original_configs["robot"])
            print("  - Restored robot config")

    def get_latest_run(self) -> Path:
        """Get the most recent run folder."""
        runs = sorted(self.training_data_dir.glob("run_*"), reverse=True)
        return runs[0] if runs else None

    def get_existing_runs(self) -> set:
        """Get set of existing run folder names."""
        return {d.name for d in self.training_data_dir.glob("run_*") if d.is_dir()}

    def check_run_status(self, run_dir: Path) -> dict:
        """
        Check the status of a run.

        Returns:
            dict with status info
        """
        result = {
            "run_name": run_dir.name,
            "path": str(run_dir),
            "status": "unknown",
            "total_frames": 0,
            "racing_frames": 0,
            "race_time_sec": 0,
        }

        metadata_path = run_dir / "metadata.csv"
        if not metadata_path.exists():
            result["status"] = "no_metadata"
            return result

        try:
            df = pd.read_csv(metadata_path)
            result["total_frames"] = len(df)

            # Get final status
            if len(df) > 0:
                final_status = df.iloc[-1]["status"]
                result["status"] = final_status

                # Count racing frames
                racing = df[df["status"].isin(["Racing", "Lap0", "Lap1", "Lap2"])]
                result["racing_frames"] = len(racing)

                # Get race time
                if "race_time_ms" in df.columns:
                    result["race_time_sec"] = df["race_time_ms"].max() / 1000.0

        except Exception as e:
            result["status"] = f"error: {e}"

        return result

    def run_once(self, run_number: int) -> dict:
        """
        Execute one data collection run.

        Args:
            run_number: Current run number (1-indexed)

        Returns:
            dict with run results
        """
        print(f"\n{'='*60}")
        print(f"[AutoCollect] Run {run_number}/{self.target_runs}")
        print(f"{'='*60}")

        run_result = {
            "run_number": run_number,
            "started_at": datetime.now().isoformat(),
            "success": False,
            "duration_sec": 0,
            "error": None,
        }

        # Get existing runs before starting
        existing_runs = self.get_existing_runs()

        # Start main.py
        main_py = self.project_dir / "main.py"
        start_time = time.time()

        print(f"[AutoCollect] Starting main.py...")

        try:
            # Run main.py with timeout
            # Note: main.py should exit after race completion
            process = subprocess.Popen(
                [sys.executable, str(main_py)],
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Wait for completion with timeout
            try:
                stdout, _ = process.communicate(timeout=self.timeout)
                run_result["duration_sec"] = time.time() - start_time

                # Print last few lines of output
                lines = stdout.strip().split("\n")
                print(f"[AutoCollect] Process output (last 10 lines):")
                for line in lines[-10:]:
                    print(f"  {line}")

            except subprocess.TimeoutExpired:
                print(f"[AutoCollect] Timeout after {self.timeout}s, terminating...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                run_result["error"] = "timeout"
                run_result["duration_sec"] = self.timeout

        except Exception as e:
            run_result["error"] = str(e)
            run_result["duration_sec"] = time.time() - start_time
            print(f"[AutoCollect] Error: {e}")
            return run_result

        # Find the new run folder
        time.sleep(1)  # Wait for file system
        new_runs = self.get_existing_runs() - existing_runs

        if not new_runs:
            print(f"[AutoCollect] WARNING: No new run folder created")
            run_result["error"] = "no_new_run"
            return run_result

        # Should be exactly one new run
        new_run_name = sorted(new_runs)[-1]  # Get latest if multiple
        new_run_dir = self.training_data_dir / new_run_name

        print(f"[AutoCollect] New run folder: {new_run_name}")

        # Check run status
        status_info = self.check_run_status(new_run_dir)
        run_result.update(status_info)

        # Determine success
        if status_info["status"] == "Finish":
            run_result["success"] = True
            print(f"[AutoCollect] SUCCESS! Frames: {status_info['racing_frames']}, "
                  f"Time: {status_info['race_time_sec']:.1f}s")
        else:
            print(f"[AutoCollect] Run ended with status: {status_info['status']}")
            run_result["error"] = f"status_{status_info['status']}"

        run_result["completed_at"] = datetime.now().isoformat()
        return run_result

    def collect(self) -> dict:
        """
        Run the full collection process.

        Returns:
            Summary results
        """
        print(f"\n{'#'*60}")
        print(f"# VRR Auto Data Collection")
        print(f"# Target: {self.target_runs} successful runs")
        print(f"# Timeout: {self.timeout}s per run")
        print(f"{'#'*60}")

        # Setup config for auto collection
        self.backup_and_set_config()

        try:
            successful_runs = 0
            total_attempts = 0
            max_attempts = self.target_runs * 2  # Allow some failures

            while successful_runs < self.target_runs and total_attempts < max_attempts:
                total_attempts += 1

                run_result = self.run_once(successful_runs + 1)
                self.results["runs"].append(run_result)

                if run_result["success"]:
                    successful_runs += 1

                # Save intermediate results
                self.save_log()

                # Brief pause between runs
                if successful_runs < self.target_runs:
                    print(f"\n[AutoCollect] Waiting 3 seconds before next run...")
                    time.sleep(3)

            # Final summary
            self.results["completed_at"] = datetime.now().isoformat()
            self.results["summary"] = {
                "total_attempts": total_attempts,
                "successful_runs": successful_runs,
                "failed_runs": total_attempts - successful_runs,
                "total_duration_sec": sum(r["duration_sec"] for r in self.results["runs"]),
                "total_racing_frames": sum(r.get("racing_frames", 0) for r in self.results["runs"] if r["success"]),
            }

            self.save_log()
            self.print_summary()

        finally:
            # Always restore original config
            self.restore_config()

        return self.results["summary"]

    def save_log(self):
        """Save results to JSON log file."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print collection summary."""
        s = self.results["summary"]

        print(f"\n{'='*60}")
        print(f"AUTO COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total attempts:     {s['total_attempts']}")
        print(f"Successful runs:    {s['successful_runs']}")
        print(f"Failed runs:        {s['failed_runs']}")
        print(f"Total racing frames: {s['total_racing_frames']}")
        print(f"Total duration:     {s['total_duration_sec']:.1f}s ({s['total_duration_sec']/60:.1f} min)")
        print(f"Log saved to:       {self.log_file}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Automated data collection for VRR AI")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of successful runs to collect (default: 10)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds per run (default: 120)")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    print(f"\n[AutoCollect] VRR Auto Data Collection")
    print(f"[AutoCollect] Robot dir: {robot_dir}")
    print(f"[AutoCollect] Target runs: {args.runs}")
    print(f"[AutoCollect] Timeout: {args.timeout}s per run")
    print(f"")
    print(f"[AutoCollect] This script will:")
    print(f"  1. Temporarily set MODE_NUM=3 (rule-based)")
    print(f"  2. Temporarily set DEBUG_MODE=0 (auto Unity launch)")
    print(f"  3. Run {args.runs} successful laps")
    print(f"  4. Restore original config when done")
    print(f"")

    response = input("Start auto collection? (y/n): ")
    if response.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    collector = AutoCollector(robot_dir, args.runs, args.timeout)
    collector.collect()


if __name__ == "__main__":
    main()
