#!/usr/bin/env python3
# combine_training_data.py
# ==============================================================================
#         COMBINE ORIGINAL + AUGMENTED DATA FOR BALANCED TRAINING
# ==============================================================================
# Creates a combined dataset with both original and flipped images for
# perfect left/right balance.

import shutil
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    original_dir = robot_dir / "training_data"
    augmented_dir = robot_dir / "training_data_augmented"
    combined_dir = robot_dir / "training_data_combined"

    print("="*80)
    print("      COMBINING ORIGINAL + AUGMENTED DATA")
    print("="*80)
    print()
    print(f"Original data: {original_dir}")
    print(f"Augmented data: {augmented_dir}")
    print(f"Output: {combined_dir}")
    print()

    # Check input directories exist
    if not original_dir.exists():
        print(f"Error: Original data not found: {original_dir}")
        return

    if not augmented_dir.exists():
        print(f"Error: Augmented data not found: {augmented_dir}")
        return

    # Create output directory
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Find all run directories (with metadata.csv filter)
    original_runs = [d for d in original_dir.iterdir()
                     if d.is_dir() and (d / "metadata.csv").exists()]
    augmented_runs = [d for d in augmented_dir.iterdir()
                      if d.is_dir() and (d / "metadata.csv").exists()]

    print(f"Found {len(original_runs)} original runs")
    print(f"Found {len(augmented_runs)} augmented runs")
    print()

    # Copy original runs as-is
    print("[1/2] Copying original runs...")
    copied_original = 0
    for run_dir in original_runs:
        dest_dir = combined_dir / run_dir.name
        if not dest_dir.exists():
            shutil.copytree(run_dir, dest_dir)
            copied_original += 1
            if copied_original % 10 == 0:
                print(f"  Copied {copied_original}/{len(original_runs)} original runs...")

    print(f"  [OK] Copied {copied_original} original runs")
    print()

    # Copy augmented runs with "_flipped" suffix
    print("[2/2] Copying augmented runs with _flipped suffix...")
    copied_augmented = 0
    for run_dir in augmented_runs:
        dest_dir = combined_dir / (run_dir.name + "_flipped")
        if not dest_dir.exists():
            shutil.copytree(run_dir, dest_dir)
            copied_augmented += 1
            if copied_augmented % 10 == 0:
                print(f"  Copied {copied_augmented}/{len(augmented_runs)} augmented runs...")

    print(f"  [OK] Copied {copied_augmented} augmented runs")
    print()

    # Summary
    total_runs = copied_original + copied_augmented
    print("="*80)
    print("COMBINATION COMPLETE")
    print("="*80)
    print()
    print(f"Total runs in combined dataset: {total_runs}")
    print(f"  - Original runs: {copied_original}")
    print(f"  - Flipped runs: {copied_augmented}")
    print()
    print(f"Combined data saved to: {combined_dir}")
    print()
    print("Next steps:")
    print("  1. Analyze the combined data:")
    print(f"     cd Robot1")
    print(f"     ../.venv/Scripts/python scripts/analyze_steering_bias.py --input training_data_combined")
    print()
    print("  2. Train the model with combined data:")
    print(f"     ../.venv/Scripts/python train_model.py --data training_data_combined --epochs 50")
    print()
    print("Expected result: Perfect 50/50 left/right balance!")
    print()

if __name__ == "__main__":
    main()
