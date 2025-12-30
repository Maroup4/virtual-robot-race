#!/usr/bin/env python3
# augment_training_data.py
# ==============================================================================
#                 TRAINING DATA AUGMENTATION - HORIZONTAL FLIP
# ==============================================================================
# Creates augmented training data by horizontally flipping images and negating
# steering angles to achieve perfect left/right balance.

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse

def augment_run(run_dir, output_dir, skip_existing=False):
    """
    Augment a single training run by flipping images horizontally.

    Args:
        run_dir: Path to original run directory
        output_dir: Path to output directory for augmented data
        skip_existing: If True, skip runs that already exist in output
    """
    run_name = run_dir.name
    output_run_dir = output_dir / run_name

    # Check if already processed
    if skip_existing and output_run_dir.exists():
        print(f"  Skipping {run_name} (already exists)")
        return 0

    # Create output directories
    output_run_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_run_dir / "images"
    output_images_dir.mkdir(exist_ok=True)

    # Load metadata
    metadata_path = run_dir / "metadata.csv"
    if not metadata_path.exists():
        print(f"  Warning: No metadata.csv found in {run_name}, skipping")
        return 0

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"  Error reading metadata for {run_name}: {e}")
        return 0

    # Process each frame
    augmented_count = 0
    images_dir = run_dir / "images"

    for idx, row in df.iterrows():
        # Original image
        img_filename = row['filename']
        img_path = images_dir / img_filename

        if not img_path.exists():
            continue

        # Load and flip image
        try:
            img = Image.open(img_path)
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Save flipped image
            output_img_path = output_images_dir / img_filename
            flipped_img.save(output_img_path)

            augmented_count += 1

        except Exception as e:
            print(f"  Error processing {img_filename}: {e}")
            continue

    # Augment metadata: negate steering angles
    df_augmented = df.copy()
    df_augmented['steer_angle'] = -df_augmented['steer_angle']

    # Also negate yaw (heading) if present
    if 'yaw' in df_augmented.columns:
        df_augmented['yaw'] = -df_augmented['yaw']

    # Save augmented metadata
    output_metadata_path = output_run_dir / "metadata.csv"
    df_augmented.to_csv(output_metadata_path, index=False)

    # Copy other files (terminal_log, unity_log, etc.)
    for file_path in run_dir.glob("*.txt"):
        shutil.copy2(file_path, output_run_dir / file_path.name)

    # Copy video if exists
    video_path = run_dir / "output_video.mp4"
    if video_path.exists():
        # Note: We don't flip the video (too complex), just note it
        with open(output_run_dir / "AUGMENTED_NOTE.txt", "w") as f:
            f.write("This run was created by horizontal flip augmentation.\n")
            f.write("Original run: {}\n".format(run_name))
            f.write("Images are flipped, steering angles are negated.\n")

    return augmented_count

def main():
    parser = argparse.ArgumentParser(
        description="Augment training data by horizontal flip"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="training_data",
        help="Input directory containing training runs (default: training_data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data_augmented",
        help="Output directory for augmented data (default: training_data_augmented)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already exist in output directory"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent
    input_dir = robot_dir / args.input
    output_dir = robot_dir / args.output

    print("="*80)
    print("           TRAINING DATA AUGMENTATION - HORIZONTAL FLIP")
    print("="*80)
    print()
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Check input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Find all run directories
    all_run_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "metadata.csv").exists()]

    if not all_run_dirs:
        print(f"No training runs found in {input_dir}")
        return

    # Filter out AI crash runs (race time < 10 seconds)
    # Only include manual driving data for training
    run_dirs = []
    excluded_runs = []

    for run_dir in all_run_dirs:
        try:
            metadata_path = run_dir / "metadata.csv"
            df = pd.read_csv(metadata_path)
            racing_data = df[df['status'] != 'StartSequence']

            if len(racing_data) > 0:
                race_time_sec = racing_data['race_time_ms'].max() / 1000.0

                # Include only runs with race time > 10 seconds (manual driving)
                if race_time_sec > 10.0:
                    run_dirs.append(run_dir)
                else:
                    excluded_runs.append((run_dir.name, race_time_sec))
        except Exception as e:
            print(f"Warning: Could not read {run_dir.name}: {e}")
            excluded_runs.append((run_dir.name, 0.0))

    print(f"Found {len(all_run_dirs)} total runs")
    print(f"  -> {len(run_dirs)} manual driving runs (race time > 10s)")
    print(f"  -> {len(excluded_runs)} excluded (AI crashes or short runs)")

    if excluded_runs and len(excluded_runs) <= 10:
        print(f"\nExcluded runs:")
        for name, time in excluded_runs:
            print(f"  - {name} ({time:.1f}s)")
    elif len(excluded_runs) > 10:
        print(f"\nExcluded {len(excluded_runs)} runs (showing last 5):")
        for name, time in excluded_runs[-5:]:
            print(f"  - {name} ({time:.1f}s)")

    if not run_dirs:
        print(f"\nNo valid training runs found (all runs were too short)")
        return

    print(f"\n{len(run_dirs)} runs will be augmented")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each run
    total_augmented = 0
    successful_runs = 0

    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] Processing {run_dir.name}...")
        try:
            count = augment_run(run_dir, output_dir, args.skip_existing)
            if count > 0:
                total_augmented += count
                successful_runs += 1
                print(f"  -> Augmented {count} frames")
        except Exception as e:
            print(f"  -> Error: {e}")

    print()
    print("="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)
    print()
    print(f"Successfully augmented: {successful_runs}/{len(run_dirs)} runs")
    print(f"Total frames augmented: {total_augmented:,}")
    print()
    print(f"Augmented data saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Analyze the augmented data:")
    print(f"     cd Robot1")
    print(f"     ../.venv/Scripts/python scripts/analyze_steering_bias.py --input training_data_augmented")
    print()
    print("  2. Update train_model.py to use augmented data:")
    print(f"     TRAINING_DATA_DIR = 'training_data_augmented'")
    print()
    print("  3. Retrain the model:")
    print(f"     ../.venv/Scripts/python train_model.py --epochs 50")
    print()
    print("Good luck! 🏁")

if __name__ == "__main__":
    main()
