# verify_image_csv_alignment.py
# Verify if image count matches metadata.csv row count
# and if sequential renaming would fix the filename mismatch

import sys
from pathlib import Path
import pandas as pd

def verify_alignment(run_dir):
    """
    Verify if images and metadata.csv are aligned for sequential renaming

    Returns:
        (is_aligned, metadata_count, image_count, first_tick, last_tick)
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "metadata.csv"
    images_dir = run_dir / "images"

    print(f"\n{'='*70}")
    print(f"Verifying: {run_dir.name}")
    print(f"{'='*70}")

    if not csv_path.exists():
        print(f"[ERROR] metadata.csv not found")
        return False, 0, 0, 0, 0

    if not images_dir.exists():
        print(f"[ERROR] images/ directory not found")
        return False, 0, 0, 0, 0

    # Load metadata.csv
    df = pd.read_csv(csv_path)
    metadata_count = len(df)

    # Count images
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.jpg'])
    image_count = len(image_files)

    # Get tick range from metadata.csv
    first_tick = df['tick'].iloc[0] if 'tick' in df.columns else df['id'].iloc[0]
    last_tick = df['tick'].iloc[-1] if 'tick' in df.columns else df['id'].iloc[-1]

    print(f"\n[COUNTS]")
    print(f"  metadata.csv rows: {metadata_count}")
    print(f"  JPG files:         {image_count}")
    print(f"  Match:             {'YES' if metadata_count == image_count else 'NO'}")

    print(f"\n[TICK RANGE in metadata.csv]")
    print(f"  First tick: {first_tick}")
    print(f"  Last tick:  {last_tick}")
    print(f"  Tick span:  {last_tick - first_tick + 1}")

    print(f"\n[EXPECTED FILENAMES in metadata.csv]")
    print(f"  First: {df['filename'].iloc[0]}")
    print(f"  Last:  {df['filename'].iloc[-1]}")

    print(f"\n[ACTUAL FILENAMES in images/]")
    print(f"  First: {image_files[0].name}")
    print(f"  Last:  {image_files[-1].name}")

    # Check if counts match
    is_aligned = (metadata_count == image_count)

    if is_aligned:
        print(f"\n{'='*70}")
        print(f"[SUCCESS] Counts match!")
        print(f"Sequential renaming will work:")
        print(f"  {image_files[0].name} -> {df['filename'].iloc[0]}")
        print(f"  {image_files[1].name} -> {df['filename'].iloc[1]}")
        print(f"  ...")
        print(f"  {image_files[-1].name} -> {df['filename'].iloc[-1]}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"[WARNING] Count mismatch!")
        print(f"  Metadata rows: {metadata_count}")
        print(f"  Image files:   {image_count}")
        print(f"  Difference:    {abs(metadata_count - image_count)}")
        print(f"Sequential renaming may cause issues.")
        print(f"{'='*70}")

    return is_aligned, metadata_count, image_count, first_tick, last_tick


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify image/CSV alignment")
    parser.add_argument("run_dir", nargs='?',
                       default="training_data/run_20260107_193739",
                       help="Path to run directory")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent
    run_dir = robot_dir / args.run_dir

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return

    is_aligned, meta_count, img_count, first_tick, last_tick = verify_alignment(run_dir)

    if is_aligned:
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION:")
        print(f"  Run: python scripts/rename_images_to_match_csv.py")
        print(f"  This will rename all images to match metadata.csv filenames")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
