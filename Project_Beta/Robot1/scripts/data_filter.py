# data_filter.py
# Data Filtering Utility for VRR AI Training Pipeline
# ====================================================
# This module provides utilities for:
# - Assessing run quality
# - Filtering valid training data
# - Creating dataset manifests for reproducibility
#
# Usage:
#   python scripts/data_filter.py --scan           # Scan all runs and show summary
#   python scripts/data_filter.py --create-manifest  # Create dataset manifest

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


class RunAnalyzer:
    """
    Analyzes a single training run directory for quality assessment.

    A run directory contains:
    - metadata.csv: Frame-by-frame data
    - images/: JPEG frames
    - terminal_log.txt: Console output (optional)
    - unity_log.txt: Unity runtime log (optional)
    """

    # Valid status values for racing (after start)
    RACING_STATUS = ["Lap1", "Lap2", "Finish"]

    # Status values indicating successful completion
    SUCCESS_STATUS = ["Finish"]

    # Status values indicating partial success
    PARTIAL_STATUS = ["Lap1", "Lap2"]

    # Status values indicating failure
    FAILURE_STATUS = ["Fallen", "ForceEnd"]

    def __init__(self, run_dir: Path):
        """
        Initialize analyzer for a run directory.

        Args:
            run_dir: Path to run_YYYYMMDD_HHMMSS directory
        """
        self.run_dir = Path(run_dir)
        self.metadata_path = self.run_dir / "metadata.csv"
        self.images_dir = self.run_dir / "images"

        self._metadata: Optional[pd.DataFrame] = None
        self._analysis: Optional[Dict] = None

    @property
    def exists(self) -> bool:
        """Check if run directory and metadata exist."""
        return self.run_dir.exists() and self.metadata_path.exists()

    @property
    def metadata(self) -> pd.DataFrame:
        """Load and cache metadata DataFrame."""
        if self._metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"metadata.csv not found in {self.run_dir}")
            self._metadata = pd.read_csv(self.metadata_path)
        return self._metadata

    def analyze(self) -> Dict:
        """
        Perform full analysis of the run.

        Returns:
            Dictionary containing analysis results
        """
        if self._analysis is not None:
            return self._analysis

        df = self.metadata

        # Basic counts
        total_frames = len(df)
        start_frames = len(df[df["status"] == "StartSequence"])
        racing_frames = len(df[df["status"].isin(self.RACING_STATUS)])

        # Final status
        final_status = df.iloc[-1]["status"] if len(df) > 0 else "Unknown"

        # Time calculations
        race_time_ms = df["race_time_ms"].max() if "race_time_ms" in df.columns else 0
        session_time_ms = df["session_time_ms"].max() if "session_time_ms" in df.columns else 0

        # SOC (State of Charge) analysis
        soc_start = df.iloc[0]["soc"] if "soc" in df.columns and len(df) > 0 else 1.0
        soc_end = df.iloc[-1]["soc"] if "soc" in df.columns and len(df) > 0 else 1.0
        soc_consumed = soc_start - soc_end

        # Movement analysis (detect stationary frames)
        if "drive_torque" in df.columns:
            racing_df = df[df["status"].isin(self.RACING_STATUS)]
            if len(racing_df) > 0:
                stationary_frames = len(racing_df[racing_df["drive_torque"].abs() < 0.01])
                stationary_ratio = stationary_frames / len(racing_df)
            else:
                stationary_frames = 0
                stationary_ratio = 0.0
        else:
            stationary_frames = 0
            stationary_ratio = 0.0

        # Lap detection
        lap1_frames = len(df[df["status"] == "Lap1"])
        lap2_frames = len(df[df["status"] == "Lap2"])
        finish_frames = len(df[df["status"] == "Finish"])

        # Check for falls
        has_fallen = "Fallen" in df["status"].values
        fallen_frame = None
        if has_fallen:
            fallen_idx = df[df["status"] == "Fallen"].index[0]
            fallen_frame = int(df.loc[fallen_idx, "id"]) if "id" in df.columns else fallen_idx

        # Check for force end
        has_force_end = "ForceEnd" in df["status"].values

        # Image verification
        image_count = len(list(self.images_dir.glob("*.jpg"))) if self.images_dir.exists() else 0
        images_match = image_count >= racing_frames * 0.9  # Allow 10% missing

        # Completion assessment
        completed_lap1 = lap1_frames > 0 or lap2_frames > 0 or finish_frames > 0
        completed_lap2 = lap2_frames > 0 or finish_frames > 0
        completed_race = finish_frames > 0

        self._analysis = {
            # Identification
            "run_name": self.run_dir.name,
            "run_path": str(self.run_dir),

            # Frame counts
            "total_frames": total_frames,
            "start_frames": start_frames,
            "racing_frames": racing_frames,
            "lap1_frames": lap1_frames,
            "lap2_frames": lap2_frames,
            "finish_frames": finish_frames,

            # Status
            "final_status": final_status,
            "has_fallen": has_fallen,
            "fallen_frame": fallen_frame,
            "has_force_end": has_force_end,

            # Timing
            "race_time_ms": race_time_ms,
            "session_time_ms": session_time_ms,
            "race_time_sec": race_time_ms / 1000.0,

            # Battery
            "soc_start": soc_start,
            "soc_end": soc_end,
            "soc_consumed": soc_consumed,

            # Movement quality
            "stationary_frames": stationary_frames,
            "stationary_ratio": stationary_ratio,

            # Completion flags
            "completed_lap1": completed_lap1,
            "completed_lap2": completed_lap2,
            "completed_race": completed_race,

            # Data integrity
            "image_count": image_count,
            "images_match": images_match,
        }

        return self._analysis

    def get_quality_score(self) -> Tuple[float, List[str]]:
        """
        Calculate a quality score for this run.

        Returns:
            Tuple of (score 0-100, list of issues)
        """
        analysis = self.analyze()
        score = 100.0
        issues = []

        # Deductions for various issues

        # Fallen: Major penalty
        if analysis["has_fallen"]:
            score -= 50
            issues.append("Robot fell during run")

        # Force end: Minor penalty
        if analysis["has_force_end"]:
            score -= 10
            issues.append("Run was force-ended")

        # No racing frames: Critical
        if analysis["racing_frames"] < 10:
            score -= 40
            issues.append(f"Too few racing frames ({analysis['racing_frames']})")

        # High stationary ratio: Moderate penalty
        if analysis["stationary_ratio"] > 0.3:
            score -= 20
            issues.append(f"High stationary ratio ({analysis['stationary_ratio']:.1%})")

        # Missing images: Minor penalty
        if not analysis["images_match"]:
            score -= 10
            issues.append("Some images missing")

        # Low battery at end: Minor penalty
        if analysis["soc_end"] < 0.5:
            score -= 5
            issues.append(f"Low battery at end ({analysis['soc_end']:.1%})")

        # Bonuses for completion
        if analysis["completed_race"]:
            score = min(100, score + 20)
        elif analysis["completed_lap2"]:
            score = min(100, score + 10)
        elif analysis["completed_lap1"]:
            score = min(100, score + 5)

        return max(0, score), issues

    def is_valid_for_training(self, config: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Determine if this run should be included in training.

        Args:
            config: Optional configuration dict with filtering rules

        Returns:
            Tuple of (is_valid, reason)
        """
        analysis = self.analyze()

        # Default config
        if config is None:
            config = {
                "min_frames": 100,
                "valid_final_status": ["Finish", "Lap2", "Lap1"],
                "exclude_fallen": True,
                "exclude_force_end": False,
                "min_soc_at_end": 0.0,
                "max_stationary_ratio": 0.5,
            }

        # Check minimum frames
        if analysis["racing_frames"] < config.get("min_frames", 100):
            return False, f"Insufficient racing frames ({analysis['racing_frames']} < {config['min_frames']})"

        # Check final status
        valid_status = config.get("valid_final_status", ["Finish", "Lap2", "Lap1"])
        if analysis["final_status"] not in valid_status:
            return False, f"Invalid final status: {analysis['final_status']}"

        # Check fallen
        if config.get("exclude_fallen", True) and analysis["has_fallen"]:
            return False, "Robot fell during run"

        # Check force end
        if config.get("exclude_force_end", False) and analysis["has_force_end"]:
            return False, "Run was force-ended"

        # Check battery
        if analysis["soc_end"] < config.get("min_soc_at_end", 0.0):
            return False, f"Battery too low at end ({analysis['soc_end']:.1%})"

        # Check stationary ratio
        if analysis["stationary_ratio"] > config.get("max_stationary_ratio", 0.5):
            return False, f"Too many stationary frames ({analysis['stationary_ratio']:.1%})"

        return True, "Passed all checks"


class DatasetManager:
    """
    Manages the training dataset across multiple runs.

    Responsibilities:
    - Scan and catalog all available runs
    - Filter runs based on quality criteria
    - Create reproducible dataset manifests
    - Support dataset versioning
    """

    def __init__(self, training_data_dir: Path, experiments_dir: Path):
        """
        Initialize dataset manager.

        Args:
            training_data_dir: Path to training_data/ directory
            experiments_dir: Path to experiments/ directory
        """
        self.training_data_dir = Path(training_data_dir)
        self.experiments_dir = Path(experiments_dir)
        self._runs: Optional[List[RunAnalyzer]] = None

    def scan_runs(self) -> List[RunAnalyzer]:
        """
        Scan training_data directory for all run folders.

        Returns:
            List of RunAnalyzer objects for each run
        """
        if self._runs is not None:
            return self._runs

        self._runs = []

        if not self.training_data_dir.exists():
            print(f"[DatasetManager] Warning: {self.training_data_dir} does not exist")
            return self._runs

        for run_dir in sorted(self.training_data_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                analyzer = RunAnalyzer(run_dir)
                if analyzer.exists:
                    self._runs.append(analyzer)

        print(f"[DatasetManager] Found {len(self._runs)} runs in {self.training_data_dir}")
        return self._runs

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all runs.

        Returns:
            DataFrame with one row per run
        """
        runs = self.scan_runs()
        summaries = []

        for run in runs:
            try:
                analysis = run.analyze()
                score, issues = run.get_quality_score()
                is_valid, reason = run.is_valid_for_training()

                summaries.append({
                    "run_name": analysis["run_name"],
                    "final_status": analysis["final_status"],
                    "racing_frames": analysis["racing_frames"],
                    "race_time_sec": analysis["race_time_sec"],
                    "soc_end": analysis["soc_end"],
                    "has_fallen": analysis["has_fallen"],
                    "quality_score": score,
                    "is_valid": is_valid,
                    "reason": reason if not is_valid else "",
                })
            except Exception as e:
                summaries.append({
                    "run_name": run.run_dir.name,
                    "final_status": "ERROR",
                    "racing_frames": 0,
                    "race_time_sec": 0,
                    "soc_end": 0,
                    "has_fallen": False,
                    "quality_score": 0,
                    "is_valid": False,
                    "reason": str(e),
                })

        return pd.DataFrame(summaries)

    def filter_runs(self, config: Optional[Dict] = None) -> List[RunAnalyzer]:
        """
        Filter runs based on quality criteria.

        Args:
            config: Filtering configuration

        Returns:
            List of valid RunAnalyzer objects
        """
        runs = self.scan_runs()
        valid_runs = []

        for run in runs:
            is_valid, reason = run.is_valid_for_training(config)
            if is_valid:
                valid_runs.append(run)
            else:
                print(f"[DatasetManager] Excluding {run.run_dir.name}: {reason}")

        print(f"[DatasetManager] {len(valid_runs)}/{len(runs)} runs passed filtering")
        return valid_runs

    def create_manifest(
        self,
        iteration: int = 1,
        config: Optional[Dict] = None,
        description: str = ""
    ) -> Dict:
        """
        Create a dataset manifest for reproducibility.

        The manifest records:
        - Which runs are included
        - Filtering criteria used
        - Frame counts and statistics
        - Timestamp for versioning

        Args:
            iteration: Iteration number
            config: Filtering configuration
            description: Optional description

        Returns:
            Manifest dictionary
        """
        valid_runs = self.filter_runs(config)

        # Gather statistics
        total_frames = 0
        total_racing_frames = 0
        run_details = []

        for run in valid_runs:
            analysis = run.analyze()
            total_frames += analysis["total_frames"]
            total_racing_frames += analysis["racing_frames"]

            run_details.append({
                "run_name": analysis["run_name"],
                "final_status": analysis["final_status"],
                "racing_frames": analysis["racing_frames"],
                "race_time_sec": analysis["race_time_sec"],
            })

        manifest = {
            "manifest_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "iteration": iteration,
            "description": description,

            "filtering_config": config or {},

            "statistics": {
                "total_runs": len(valid_runs),
                "total_frames": total_frames,
                "total_racing_frames": total_racing_frames,
            },

            "runs": run_details,
        }

        return manifest

    def save_manifest(self, manifest: Dict, iteration: int) -> Path:
        """
        Save manifest to experiments directory.

        Args:
            manifest: Manifest dictionary
            iteration: Iteration number

        Returns:
            Path to saved manifest file
        """
        iteration_dir = self.experiments_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = iteration_dir / "dataset_manifest.json"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[DatasetManager] Manifest saved to {manifest_path}")
        return manifest_path


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="VRR Data Filtering Utility")
    parser.add_argument("--scan", action="store_true", help="Scan all runs and show summary")
    parser.add_argument("--create-manifest", action="store_true", help="Create dataset manifest")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number for manifest")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent  # Robot1/
    training_data_dir = robot_dir / "training_data"
    experiments_dir = robot_dir / "experiments"

    # Load config if specified
    config = None
    if args.config:
        config = load_config(Path(args.config))
        if "data_filtering" in config:
            config = config["data_filtering"]

    # Initialize manager
    manager = DatasetManager(training_data_dir, experiments_dir)

    if args.scan:
        print("\n" + "=" * 70)
        print("VRR Training Data Scan")
        print("=" * 70 + "\n")

        summary = manager.get_summary()

        if len(summary) == 0:
            print("No runs found in training_data/")
            return

        # Display summary table
        print(summary.to_string(index=False))

        print("\n" + "-" * 70)
        print("Summary:")
        print(f"  Total runs: {len(summary)}")
        print(f"  Valid runs: {summary['is_valid'].sum()}")
        print(f"  Total racing frames: {summary['racing_frames'].sum()}")

        # Status breakdown
        print("\nStatus breakdown:")
        for status in summary["final_status"].unique():
            count = (summary["final_status"] == status).sum()
            print(f"  {status}: {count}")

        if args.verbose:
            print("\n" + "-" * 70)
            print("Detailed Analysis:")
            for run in manager.scan_runs():
                print(f"\n{run.run_dir.name}:")
                analysis = run.analyze()
                for key, value in analysis.items():
                    print(f"  {key}: {value}")

    if args.create_manifest:
        print("\n" + "=" * 70)
        print(f"Creating Dataset Manifest (Iteration {args.iteration})")
        print("=" * 70 + "\n")

        manifest = manager.create_manifest(
            iteration=args.iteration,
            config=config,
            description=f"Dataset for iteration {args.iteration}"
        )

        manifest_path = manager.save_manifest(manifest, args.iteration)

        print("\nManifest contents:")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
