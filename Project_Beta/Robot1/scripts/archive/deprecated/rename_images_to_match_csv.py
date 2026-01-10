# rename_images_to_match_csv.py
# Rename images sequentially to match metadata.csv filenames
# This fixes the tick mismatch between Unity (DataLogger) and Python (websocket_client)

import sys
from pathlib import Path
import pandas as pd
import shutil

def rename_images(run_dir, dry_run=True):
    """
    Rename images to match metadata.csv filenames

    Args:
        run_dir: Path to run directory
        dry_run: If True, only show what would be renamed without modifying
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "metadata.csv"
    images_dir = run_dir / "images"

    print(f"\n{'='*70}")
    print(f"Processing: {run_dir.name}")
    print(f"{'='*70}")

    if not csv_path.exists():
        print(f"[ERROR] metadata.csv not found: {csv_path}")
        return False

    if not images_dir.exists():
        print(f"[ERROR] images/ directory not found: {images_dir}")
        return False

    # Load metadata.csv
    df = pd.read_csv(csv_path)
    print(f"[INFO] metadata.csv loaded: {len(df)} rows")

    # Get actual image files (sorted)
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.jpg'])
    print(f"[INFO] Found {len(image_files)} image files")

    if len(df) != len(image_files):
        print(f"[ERROR] Count mismatch: CSV has {len(df)} rows, but {len(image_files)} images exist")
        return False

    # Create backup directory
    if not dry_run:
        backup_dir = images_dir.parent / "images_backup"
        if backup_dir.exists():
            print(f"[ERROR] Backup directory already exists: {backup_dir}")
            print(f"  Please remove or rename it first to avoid overwriting")
            return False

    # Prepare rename mapping
    rename_map = []
    for idx, row in df.iterrows():
        old_name = image_files[idx].name
        new_name = row['filename']

        if old_name != new_name:
            rename_map.append((image_files[idx], images_dir / new_name))

    if len(rename_map) == 0:
        print(f"\n[OK] All filenames already match! No renaming needed.")
        return True

    print(f"\n[INFO] {len(rename_map)} files need renaming")

    # Show sample of changes
    print(f"\n[PROPOSED CHANGES] (showing first 5 and last 5)")
    for i, (old_path, new_path) in enumerate(rename_map[:5]):
        print(f"  {old_path.name} -> {new_path.name}")
    if len(rename_map) > 10:
        print(f"  ... ({len(rename_map) - 10} more) ...")
    for old_path, new_path in rename_map[-5:]:
        print(f"  {old_path.name} -> {new_path.name}")

    if dry_run:
        print(f"\n[DRY RUN] No changes made. Run with --apply to actually rename.")
        return False
    else:
        # Create backup
        print(f"\n[BACKUP] Creating backup of images/ directory...")
        backup_dir = images_dir.parent / "images_backup"
        shutil.copytree(images_dir, backup_dir)
        print(f"[BACKUP] Backup created: {backup_dir}")

        # Perform renaming in two phases to avoid conflicts
        # Phase 1: Rename to temporary names
        print(f"\n[PHASE 1] Renaming to temporary names...")
        temp_map = []
        for old_path, new_path in rename_map:
            temp_path = old_path.with_suffix('.tmp.jpg')
            old_path.rename(temp_path)
            temp_map.append((temp_path, new_path))

        # Phase 2: Rename to final names
        print(f"[PHASE 2] Renaming to final names...")
        for temp_path, new_path in temp_map:
            temp_path.rename(new_path)

        print(f"[SUCCESS] {len(rename_map)} files renamed successfully!")

        # Verify
        print(f"\n[VERIFICATION] Checking if all filenames now match...")
        verification_errors = 0
        for idx, row in df.iterrows():
            expected_path = images_dir / row['filename']
            if not expected_path.exists():
                print(f"  [ERROR] Expected file not found: {row['filename']}")
                verification_errors += 1

        if verification_errors == 0:
            print(f"[SUCCESS] Verification passed! All filenames now match metadata.csv")
            return True
        else:
            print(f"[ERROR] Verification failed! {verification_errors} files missing")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rename images to match metadata.csv")
    parser.add_argument("run_dir", nargs='?',
                       default="training_data/run_20260107_193739",
                       help="Path to run directory")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply the rename (default is dry-run)")
    parser.add_argument("--all", action="store_true",
                       help="Process all run_* directories in training_data/")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    dry_run = not args.apply

    if dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No changes will be made")
        print("Use --apply flag to actually rename the files")
        print("="*70)

    if args.all:
        # Process all run directories
        training_data_dir = robot_dir / "training_data"
        if not training_data_dir.exists():
            print(f"[ERROR] training_data directory not found: {training_data_dir}")
            return

        run_dirs = sorted([d for d in training_data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            print(f"[ERROR] No run_* directories found in: {training_data_dir}")
            return

        print(f"\n[INFO] Found {len(run_dirs)} run directories")

        success_count = 0
        for run_dir in run_dirs:
            success = rename_images(run_dir, dry_run=dry_run)
            if success:
                success_count += 1

        print(f"\n{'='*70}")
        print(f"Overall Result: {success_count}/{len(run_dirs)} runs processed successfully")
        print(f"{'='*70}")
    else:
        # Process single run directory
        run_dir = robot_dir / args.run_dir
        if not run_dir.exists():
            print(f"[ERROR] Run directory not found: {run_dir}")
            return

        success = rename_images(run_dir, dry_run=dry_run)

        if success and not dry_run:
            print(f"\n{'='*70}")
            print(f"SUCCESS! Images have been renamed to match metadata.csv")
            print(f"Backup saved to: {run_dir / 'images_backup'}")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()
