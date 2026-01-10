# fix_metadata_filenames.py
# Fix metadata.csv filename column to match actual image files

import sys
from pathlib import Path
import pandas as pd
import shutil

def fix_single_run(run_dir, dry_run=True):
    """
    Fix filename column in metadata.csv to match actual image files

    Args:
        run_dir: Path to run directory
        dry_run: If True, only show what would be changed without modifying
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

    # Get actual image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.jpg'])
    print(f"[INFO] Found {len(image_files)} image files in images/")

    if len(df) != len(image_files):
        print(f"[WARN] Row count mismatch: CSV has {len(df)} rows, but {len(image_files)} images exist")

    # Show current state
    print(f"\n[CURRENT STATE]")
    print(f"  CSV first filename: {df['filename'].iloc[0]}")
    print(f"  CSV last filename:  {df['filename'].iloc[-1]}")
    print(f"  Actual first file:  {image_files[0].name}")
    print(f"  Actual last file:   {image_files[-1].name}")

    # Check if filenames match
    mismatches = 0
    for idx, row in df.iterrows():
        expected_filename = image_files[idx].name if idx < len(image_files) else None
        if row['filename'] != expected_filename:
            mismatches += 1

    if mismatches == 0:
        print(f"\n[OK] All filenames already match! No fix needed.")
        return True

    print(f"\n[ISSUE] {mismatches} filename mismatches detected")

    # Create corrected dataframe
    df_fixed = df.copy()

    # Replace filename column with actual file names
    for idx in range(min(len(df), len(image_files))):
        df_fixed.loc[idx, 'filename'] = image_files[idx].name

    # Show what will change
    print(f"\n[PROPOSED CHANGES]")
    print(f"  First 3 rows:")
    for idx in range(min(3, len(df))):
        old = df['filename'].iloc[idx]
        new = df_fixed['filename'].iloc[idx]
        match = "✓" if old == new else "X"
        print(f"    Row {idx}: {old} -> {new} [{match}]")

    print(f"  Last 3 rows:")
    for idx in range(max(0, len(df)-3), len(df)):
        old = df['filename'].iloc[idx]
        new = df_fixed['filename'].iloc[idx]
        match = "✓" if old == new else "X"
        print(f"    Row {idx}: {old} -> {new} [{match}]")

    if dry_run:
        print(f"\n[DRY RUN] No changes made. Run with --apply to actually fix.")
        return False
    else:
        # Backup original
        backup_path = csv_path.with_suffix('.csv.backup')
        shutil.copy2(csv_path, backup_path)
        print(f"\n[BACKUP] Original saved to: {backup_path}")

        # Save fixed CSV
        df_fixed.to_csv(csv_path, index=False)
        print(f"[SAVED] Fixed metadata.csv written")

        # Verify
        df_verify = pd.read_csv(csv_path)
        verified_mismatches = 0
        for idx in range(min(len(df_verify), len(image_files))):
            if df_verify['filename'].iloc[idx] != image_files[idx].name:
                verified_mismatches += 1

        if verified_mismatches == 0:
            print(f"[SUCCESS] Verification passed! All filenames now match.")
            return True
        else:
            print(f"[ERROR] Verification failed! {verified_mismatches} mismatches remain.")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix metadata.csv filename column")
    parser.add_argument("run_dir", nargs='?',
                       default="training_data/run_20260107_193739",
                       help="Path to run directory")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply the fix (default is dry-run)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent
    run_dir = robot_dir / args.run_dir

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return

    dry_run = not args.apply

    if dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No changes will be made")
        print("Use --apply flag to actually fix the metadata.csv")
        print("="*70)

    success = fix_single_run(run_dir, dry_run=dry_run)

    if success and not dry_run:
        print(f"\n{'='*70}")
        print(f"SUCCESS! metadata.csv has been fixed.")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
