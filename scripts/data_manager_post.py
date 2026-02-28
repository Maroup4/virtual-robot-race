#!/usr/bin/env python3
# data_manager_post.py
# =============================================================================
# Virtual Robot Race Beta 1.2 - Post-Processing Data Manager
# Training Data Filename Correction Tool
# =============================================================================
#
# PURPOSE:
#   Post-processing tool to fix the filename mismatch between Unity's
#   DataLogger (tick-based naming) and Python's websocket_client
#   (sequential counting from 1).
#
#   Companion to data_manager.py:
#     - data_manager.py:      Real-time data collection during race
#     - data_manager_post.py: Post-race data verification and correction
#
# PROBLEM:
#   - Unity: tick=329 -> "frame_000329.jpg" in metadata.csv
#   - Python: 1st image received -> saved as "frame_000001.jpg"
#   - Result: 328-frame offset causing total training failure
#
# SOLUTION:
#   Sequential renaming of images to match metadata.csv filenames.
#   Works because: image count = metadata rows, and order is preserved.
#
# USAGE:
#   # Verify a single run (dry-run)
#   python scripts/data_manager_post.py --robot 1 --run run_20260107_193739
#
#   # Fix a single run
#   python scripts/data_manager_post.py --robot 1 --run run_20260107_193739 --apply
#
#   # Fix all runs for Robot1
#   python scripts/data_manager_post.py --robot 1 --all --apply
#
#   # Fix all runs for all robots
#   python scripts/data_manager_post.py --all-robots --all --apply
#
# =============================================================================

import sys
import argparse
from pathlib import Path
import pandas as pd
import shutil


class TrainingDataFixer:
    """
    Training data filename correction tool for Virtual Robot Race.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.stats = {
            'total_runs': 0,
            'fixed_runs': 0,
            'already_correct': 0,
            'failed_runs': 0
        }

    def verify_and_fix_run(self, run_dir: Path, dry_run: bool = True) -> bool:
        """
        Verify and optionally fix a single run directory.

        Args:
            run_dir: Path to run_YYYYMMDD_HHMMSS directory
            dry_run: If True, only verify without modifying

        Returns:
            True if successful or already correct, False if failed
        """
        csv_path = run_dir / "metadata.csv"
        images_dir = run_dir / "images"

        print(f"\n{'='*70}")
        print(f"Processing: {run_dir.relative_to(self.project_root)}")
        print(f"{'='*70}")

        # Basic validation
        if not csv_path.exists():
            print(f"[ERROR] metadata.csv not found")
            return False

        if not images_dir.exists():
            print(f"[ERROR] images/ directory not found")
            return False

        # Load metadata
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to load metadata.csv: {e}")
            return False

        # Get image files
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.jpg'])

        metadata_count = len(df)
        image_count = len(image_files)

        print(f"  Metadata rows: {metadata_count}")
        print(f"  Image files:   {image_count}")

        # Check count match
        if metadata_count != image_count:
            print(f"[ERROR] Count mismatch (diff: {abs(metadata_count - image_count)})")
            return False

        # Build rename mapping
        rename_map = []
        for idx, row in df.iterrows():
            expected_name = row['filename']
            actual_name = image_files[idx].name

            if actual_name != expected_name:
                rename_map.append((image_files[idx], images_dir / expected_name))

        if len(rename_map) == 0:
            print(f"[OK] All filenames already match! No fix needed.")
            return True  # Already correct

        print(f"  Files needing rename: {len(rename_map)}")

        # Show sample
        print(f"\n  Sample changes:")
        for old_path, new_path in rename_map[:3]:
            print(f"    {old_path.name} -> {new_path.name}")
        if len(rename_map) > 6:
            print(f"    ... ({len(rename_map) - 6} more) ...")
        for old_path, new_path in rename_map[-3:]:
            print(f"    {old_path.name} -> {new_path.name}")

        if dry_run:
            print(f"\n[DRY RUN] No changes made (use --apply to fix)")
            return True
        else:
            return self._apply_rename(images_dir, rename_map, df)

    def _apply_rename(self, images_dir: Path, rename_map: list, df: pd.DataFrame) -> bool:
        """
        Apply the rename operation with backup.

        Args:
            images_dir: Path to images/ directory
            rename_map: List of (old_path, new_path) tuples
            df: Metadata DataFrame for verification

        Returns:
            True if successful, False otherwise
        """
        backup_dir = images_dir.parent / "images_backup"

        # Check for existing backup
        if backup_dir.exists():
            print(f"\n[ERROR] Backup directory already exists: {backup_dir.name}")
            print(f"  Please remove or rename it first")
            return False

        try:
            # Create backup
            print(f"\n[BACKUP] Creating backup...")
            shutil.copytree(images_dir, backup_dir)
            print(f"  Backup saved: {backup_dir.name}/")

            # Phase 1: Rename to temporary names (avoid conflicts)
            print(f"[RENAME] Phase 1: Temporary names...")
            temp_map = []
            for old_path, new_path in rename_map:
                temp_path = old_path.with_suffix('.tmp.jpg')
                old_path.rename(temp_path)
                temp_map.append((temp_path, new_path))

            # Phase 2: Rename to final names
            print(f"[RENAME] Phase 2: Final names...")
            for temp_path, new_path in temp_map:
                temp_path.rename(new_path)

            # Verify
            print(f"[VERIFY] Checking all files...")
            missing = []
            for _, row in df.iterrows():
                expected_path = images_dir / row['filename']
                if not expected_path.exists():
                    missing.append(row['filename'])

            if missing:
                print(f"[ERROR] Verification failed! {len(missing)} files missing")
                for fname in missing[:5]:
                    print(f"  Missing: {fname}")
                return False

            print(f"[SUCCESS] {len(rename_map)} files renamed successfully!")
            return True

        except Exception as e:
            print(f"[ERROR] Rename failed: {e}")
            return False

    def process_robot(self, robot_num: int, fix_all: bool, specific_run: str = None, dry_run: bool = True):
        """
        Process training data for a specific robot.

        Args:
            robot_num: Robot number (1-5)
            fix_all: If True, process all runs
            specific_run: If provided, process only this run (e.g., "run_20260107_193739")
            dry_run: If True, only verify without modifying
        """
        robot_dir = self.project_root / f"Robot{robot_num}"
        training_data_dirs = [
            robot_dir / "training_data",
            robot_dir / "training_data_combined",
            robot_dir / "training_data_augmented"
        ]

        print(f"\n{'='*70}")
        print(f"ROBOT {robot_num} - Training Data Correction")
        print(f"{'='*70}")

        if specific_run:
            # Process specific run
            for training_dir in training_data_dirs:
                run_dir = training_dir / specific_run
                if run_dir.exists():
                    self.stats['total_runs'] += 1
                    result = self.verify_and_fix_run(run_dir, dry_run)
                    if result:
                        if (run_dir / "images_backup").exists():
                            self.stats['fixed_runs'] += 1
                        else:
                            self.stats['already_correct'] += 1
                    else:
                        self.stats['failed_runs'] += 1
                    return

            print(f"[ERROR] Run not found: {specific_run}")
            return

        if fix_all:
            # Process all runs in all training data directories
            for training_dir in training_data_dirs:
                if not training_dir.exists():
                    continue

                print(f"\nScanning: {training_dir.relative_to(self.project_root)}/")

                run_dirs = sorted([d for d in training_dir.iterdir()
                                  if d.is_dir() and d.name.startswith("run_")])

                if not run_dirs:
                    print(f"  No run_* directories found")
                    continue

                print(f"  Found {len(run_dirs)} runs")

                for run_dir in run_dirs:
                    self.stats['total_runs'] += 1
                    result = self.verify_and_fix_run(run_dir, dry_run)

                    if result:
                        if (run_dir / "images_backup").exists():
                            self.stats['fixed_runs'] += 1
                        else:
                            self.stats['already_correct'] += 1
                    else:
                        self.stats['failed_runs'] += 1

    def print_summary(self):
        """Print final statistics."""
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Total runs processed:  {self.stats['total_runs']}")
        print(f"  Fixed:                 {self.stats['fixed_runs']}")
        print(f"  Already correct:       {self.stats['already_correct']}")
        print(f"  Failed:                {self.stats['failed_runs']}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Robot Race Beta 1.2 - Training Data Filename Correction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify Robot1's specific run (dry-run)
  python scripts/fix_training_data.py --robot 1 --run run_20260107_193739

  # Fix Robot1's specific run
  python scripts/fix_training_data.py --robot 1 --run run_20260107_193739 --apply

  # Fix all runs for Robot1
  python scripts/fix_training_data.py --robot 1 --all --apply

  # Fix all runs for all robots (Robot1-5)
  python scripts/fix_training_data.py --all-robots --all --apply

Note: Always run without --apply first to verify changes!
        """
    )

    parser.add_argument("--robot", type=int, choices=[1, 2, 3, 4, 5],
                       help="Robot number (1-5)")
    parser.add_argument("--all-robots", action="store_true",
                       help="Process all robots (Robot1-5)")
    parser.add_argument("--run", type=str,
                       help="Specific run to process (e.g., run_20260107_193739)")
    parser.add_argument("--all", action="store_true",
                       help="Process all runs in training_data directories")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply fixes (default is dry-run)")

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Validate arguments
    if not args.all_robots and not args.robot:
        parser.error("Must specify either --robot or --all-robots")

    if not args.all and not args.run:
        parser.error("Must specify either --all or --run")

    dry_run = not args.apply

    # Print mode
    print(f"\n{'='*70}")
    print(f"Virtual Robot Race Beta 1.2")
    print(f"Training Data Filename Correction Tool")
    print(f"{'='*70}")
    if dry_run:
        print(f"MODE: DRY RUN (verification only)")
        print(f"      Use --apply to actually fix files")
    else:
        print(f"MODE: APPLY (will modify files with backup)")
    print(f"{'='*70}")

    # Create fixer
    fixer = TrainingDataFixer(project_root)

    # Process robots
    if args.all_robots:
        for robot_num in range(1, 6):
            robot_dir = project_root / f"Robot{robot_num}"
            if robot_dir.exists():
                fixer.process_robot(robot_num, args.all, args.run, dry_run)
    else:
        fixer.process_robot(args.robot, args.all, args.run, dry_run)

    # Print summary
    fixer.print_summary()

    # Final message
    if dry_run and fixer.stats['total_runs'] > fixer.stats['already_correct']:
        print(f"\nTo apply fixes, add --apply flag to the command")


if __name__ == "__main__":
    main()
