#!/usr/bin/env python3
"""
create_iteration.py
===================
Create a new iteration folder, copy data sources, and prepare for training

Usage:
    python scripts/create_iteration.py --data training_data_selected
    python scripts/create_iteration.py --data training_data --runs run_20260104_140000 run_20260104_140300

Features:
    - Create iteration_YYMMDD_HHMMSS folder
    - Copy all data source run_* directories
    - Automatically analyze data statistics
    - Auto-generate training_config.yaml
    - Automatic folder structure setup
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml


class IterationCreator:
    """Class for creating and managing new iteration folders"""

    def __init__(self, robot_dir: Path):
        """
        Initialize iteration creator.

        Args:
            robot_dir: Path to the Robot1 directory
        """
        self.robot_dir = Path(robot_dir)
        self.experiments_dir = self.robot_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

    def create(
        self,
        data_source_dir: Path,
        specific_runs: Optional[List[str]] = None,
        description: str = ""
    ) -> Path:
        """
        Create a new iteration folder

        Args:
            data_source_dir: Source directory for training data
            specific_runs: List of specific run names (None for all runs)
            description: Description of the iteration

        Returns:
            Path to the created iteration directory
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        iteration_dir = self.experiments_dir / f"iteration_{timestamp}"

        print(f"\n{'='*70}")
        print(f"Creating Iteration: iteration_{timestamp}")
        print(f"{'='*70}\n")

        # Create folder structure
        print("[1/5] Creating folder structure...")
        self._create_folder_structure(iteration_dir)

        # Copy data sources
        print("\n[2/5] Copying data sources...")
        run_dirs = self._copy_data_sources(
            data_source_dir,
            iteration_dir / "data_sources",
            specific_runs
        )

        if not run_dirs:
            print("\n[Error] No valid training data found!")
            shutil.rmtree(iteration_dir)
            sys.exit(1)

        # Analyze data statistics
        print("\n[3/5] Analyzing data statistics...")
        data_stats = self._analyze_data_sources(iteration_dir / "data_sources")

        # Create training_config.yaml
        print("\n[4/5] Creating training_config.yaml...")
        self._create_training_config(
            iteration_dir,
            timestamp,
            data_source_dir,
            run_dirs,
            data_stats,
            description
        )

        # Create README
        print("\n[5/5] Creating README.md...")
        self._create_readme(iteration_dir, timestamp, data_stats)

        print(f"\n{'='*70}")
        print(f"[SUCCESS] Iteration created successfully!")
        print(f"{'='*70}")
        print(f"\nIteration directory: {iteration_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review data_sources/ ({data_stats['total_runs']} runs)")
        print(f"  2. Run training:")
        print(f"     python train_model.py \\")
        print(f"       --data {iteration_dir / 'data_sources'} \\")
        print(f"       --output {iteration_dir} \\")
        print(f"       --epochs 50")
        print(f"  3. Evaluate with 3 test runs")
        print()

        return iteration_dir

    def _create_folder_structure(self, iteration_dir: Path):
        """Create folder structure"""
        folders = [
            iteration_dir,
            iteration_dir / "data_sources",
            iteration_dir / "evaluation",
            iteration_dir / "logs",
        ]

        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] {folder.relative_to(self.robot_dir)}")

    def _copy_data_sources(
        self,
        source_dir: Path,
        dest_dir: Path,
        specific_runs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Copy data sources

        Args:
            source_dir: Source directory to copy from
            dest_dir: Destination directory to copy to
            specific_runs: List of specific run names

        Returns:
            List of copied run directories
        """
        source_dir = Path(source_dir)
        run_dirs = []

        if specific_runs:
            # Copy only specific runs
            for run_name in specific_runs:
                run_path = source_dir / run_name
                if run_path.exists() and run_path.is_dir():
                    dest_path = dest_dir / run_name
                    print(f"  Copying {run_name}...", end=" ")
                    shutil.copytree(run_path, dest_path)
                    run_dirs.append(dest_path)
                    print("[OK]")
                else:
                    print(f"  [Warning] Run not found: {run_name}")
        else:
            # Copy all runs
            all_runs = sorted(source_dir.glob("run_*"))
            if not all_runs:
                print(f"  [Warning] No run_* directories found in {source_dir}")
                return []

            for run_path in all_runs:
                if run_path.is_dir():
                    dest_path = dest_dir / run_path.name
                    print(f"  Copying {run_path.name}...", end=" ")
                    shutil.copytree(run_path, dest_path)
                    run_dirs.append(dest_path)
                    print("[OK]")

        print(f"\n  Total runs copied: {len(run_dirs)}")
        return run_dirs

    def _analyze_data_sources(self, data_sources_dir: Path) -> Dict:
        """
        Analyze data source statistics

        Args:
            data_sources_dir: The data_sources directory

        Returns:
            Dictionary of statistical information
        """
        stats = {
            "total_runs": 0,
            "total_frames": 0,
            "total_racing_frames": 0,
            "avg_steer": 0.0,
            "left_steer_ratio": 0.0,
            "right_steer_ratio": 0.0,
            "neutral_ratio": 0.0,
            "runs_detail": []
        }

        all_steers = []
        left_count = 0
        right_count = 0
        neutral_count = 0

        for run_dir in sorted(data_sources_dir.glob("run_*")):
            metadata_path = run_dir / "metadata.csv"
            if not metadata_path.exists():
                print(f"  [Warning] metadata.csv not found in {run_dir.name}")
                continue

            try:
                df = pd.read_csv(metadata_path)

                # Extract racing frames only (exclude StartSequence)
                # Check status column (final_status or status)
                status_col = 'final_status' if 'final_status' in df.columns else 'status'
                racing_df = df[df[status_col] != 'StartSequence']

                if len(racing_df) == 0:
                    continue

                # Check timestamp column
                time_col = 'timestamp' if 'timestamp' in racing_df.columns else 'race_time_ms'
                duration = racing_df[time_col].max() / 1000.0  # ms -> s
                avg_steer = racing_df['steer_angle'].mean()

                # Left/right count (threshold: 0.05 rad)
                left = (racing_df['steer_angle'] < -0.05).sum()
                right = (racing_df['steer_angle'] > 0.05).sum()
                neutral = len(racing_df) - left - right

                run_info = {
                    "name": run_dir.name,
                    "frames": len(df),
                    "racing_frames": len(racing_df),
                    "duration_sec": round(duration, 2),
                    "avg_steer": round(avg_steer, 4),
                    "left_count": int(left),
                    "right_count": int(right),
                    "neutral_count": int(neutral),
                }

                stats["runs_detail"].append(run_info)
                stats["total_runs"] += 1
                stats["total_frames"] += len(df)
                stats["total_racing_frames"] += len(racing_df)

                all_steers.extend(racing_df['steer_angle'].tolist())
                left_count += left
                right_count += right
                neutral_count += neutral

                print(f"  [OK] {run_dir.name}: {len(racing_df)} frames, "
                      f"{duration:.1f}s, avg_steer={avg_steer:.3f}")

            except Exception as e:
                print(f"  [Error] Failed to analyze {run_dir.name}: {e}")

        # Calculate overall statistics
        if all_steers:
            stats["avg_steer"] = round(sum(all_steers) / len(all_steers), 4)

            total_steers = left_count + right_count + neutral_count
            stats["left_steer_ratio"] = round(left_count / total_steers * 100, 2)
            stats["right_steer_ratio"] = round(right_count / total_steers * 100, 2)
            stats["neutral_ratio"] = round(neutral_count / total_steers * 100, 2)

        print(f"\n  Summary:")
        print(f"    Total runs: {stats['total_runs']}")
        print(f"    Total racing frames: {stats['total_racing_frames']}")
        print(f"    Average steer: {stats['avg_steer']:.4f} rad")
        print(f"    Left/Right/Neutral: {stats['left_steer_ratio']:.1f}% / "
              f"{stats['right_steer_ratio']:.1f}% / {stats['neutral_ratio']:.1f}%")

        # Balance check
        if abs(stats['avg_steer']) > 0.05:
            print(f"\n  [WARNING] Steering bias detected! avg_steer = {stats['avg_steer']:.4f}")
        elif abs(stats['left_steer_ratio'] - stats['right_steer_ratio']) > 20:
            print(f"\n  [WARNING] Left/Right imbalance! "
                  f"({stats['left_steer_ratio']:.1f}% vs {stats['right_steer_ratio']:.1f}%)")
        else:
            print(f"\n  [OK] Good balance!")

        return stats

    def _create_training_config(
        self,
        iteration_dir: Path,
        timestamp: str,
        source_dir: Path,
        run_dirs: List[Path],
        data_stats: Dict,
        description: str
    ):
        """Create training_config.yaml"""
        config = {
            "iteration": {
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "description": description or f"Iteration {timestamp}",
            },
            "data_source": {
                "original_dir": str(source_dir),
                "total_runs": len(run_dirs),
                "run_names": [run_dir.name for run_dir in run_dirs],
            },
            "data_statistics": {
                "total_frames": data_stats["total_frames"],
                "total_racing_frames": data_stats["total_racing_frames"],
                "avg_steer": data_stats["avg_steer"],
                "left_steer_ratio": data_stats["left_steer_ratio"],
                "right_steer_ratio": data_stats["right_steer_ratio"],
                "neutral_ratio": data_stats["neutral_ratio"],
            },
            "training": {
                "status": "not_started",
                "model_path": None,
                "best_val_loss": None,
                "best_epoch": None,
            },
            "evaluation": {
                "test_runs": [],
                "completion_rate": None,
            }
        }

        config_path = iteration_dir / "training_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        print(f"  [OK] {config_path.relative_to(self.robot_dir)}")

        # Also save as JSON (easier to load)
        json_path = iteration_dir / "training_config.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _create_readme(self, iteration_dir: Path, timestamp: str, data_stats: Dict):
        """Create README.md"""
        readme_content = f"""# Iteration {timestamp}

**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Data Source

- **Total runs:** {data_stats['total_runs']}
- **Total frames:** {data_stats['total_frames']:,}
- **Racing frames:** {data_stats['total_racing_frames']:,}
- **Average steer:** {data_stats['avg_steer']:.4f} rad
- **Left/Right/Neutral:** {data_stats['left_steer_ratio']:.1f}% / {data_stats['right_steer_ratio']:.1f}% / {data_stats['neutral_ratio']:.1f}%

### Run Details

| Run Name | Frames | Racing | Duration(s) | Avg Steer |
|-------|-----------|----------|---------|-----------|
"""
        for run in data_stats['runs_detail']:
            readme_content += f"| {run['name']} | {run['frames']} | {run['racing_frames']} | {run['duration_sec']} | {run['avg_steer']:.4f} |\n"

        readme_content += f"""
## Folder Structure

```
iteration_{timestamp}/
├── README.md                  # This file
├── training_config.yaml       # Training settings and data source info
├── training_config.json       # Same as above (JSON format)
├── data_sources/              # Training data (full copy)
│   ├── run_XXXXXX_XXXXXX/
│   └── ...
├── evaluation/                # Test run results
│   ├── test_run_001.json
│   ├── test_run_002.json
│   ├── test_run_003.json
│   └── evaluation_summary.md
└── logs/                      # Training log
    ├── training_log.txt
    └── loss_curve.png
```

## Next Steps

### 1. Run Training

```bash
python train_model.py \\
  --data {iteration_dir.name}/data_sources \\
  --output {iteration_dir.name} \\
  --epochs 50 \\
  --device cuda
```

### 2. Test Runs (3 times)

Run 3 times manually or via script, and record the evaluation results

### 3. Results Analysis

- Check completion rate
- Analyze crash locations
- Determine the direction for the next iteration

## Training Results

(Automatically updated after training)

## Evaluation Results

(Recorded after test runs)
"""

        readme_path = iteration_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"  [OK] {readme_path.relative_to(self.robot_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create new iteration folder with data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all runs
  python scripts/create_iteration.py --data training_data_selected

  # Copy only specific runs
  python scripts/create_iteration.py --data training_data \\
    --runs run_20260104_140000 run_20260104_140300 run_20260104_140500

  # Add a description
  python scripts/create_iteration.py --data training_data_selected \\
    --description "Manual keyboard runs only, balanced steering"
        """
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data source directory (e.g., training_data_selected)"
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Specific run names to copy (optional, default: all runs)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of this iteration"
    )

    args = parser.parse_args()

    # Determine Robot1 directory
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    # Validate data source directory
    data_source_dir = robot_dir / args.data
    if not data_source_dir.exists():
        print(f"[Error] Data source directory not found: {data_source_dir}")
        sys.exit(1)

    # Create iteration
    creator = IterationCreator(robot_dir)
    iteration_dir = creator.create(
        data_source_dir=data_source_dir,
        specific_runs=args.runs,
        description=args.description
    )

    return iteration_dir


if __name__ == "__main__":
    main()
