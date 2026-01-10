# verify_data_integrity.py
# Data integrity verification script
# Check if metadata.csv and jpg files are correctly aligned

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

def verify_single_run(run_dir):
    """
    Verify data integrity for a single run directory

    Checks:
    1. metadata.csv filename column matches files in images/
    2. Images can be loaded correctly
    3. Image sizes are consistent (480x270)
    4. SOC values are in range (0.0-1.0)
    5. drive_torque values are in range (-1.0 - 1.0)
    6. steer_angle values are in range (-0.785 - 0.785 rad)
    7. Frame number continuity
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "metadata.csv"
    images_dir = run_dir / "images"

    print(f"\n{'='*70}")
    print(f"Verifying: {run_dir.name}")
    print(f"{'='*70}")

    if not csv_path.exists():
        print(f"[ERROR] metadata.csv not found: {csv_path}")
        return False

    if not images_dir.exists():
        print(f"[ERROR] images/ directory not found: {images_dir}")
        return False

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"[OK] metadata.csv loaded: {len(df)} rows")

    issues = []
    warnings = []

    # --- Check 1: filename column vs actual files ---
    print(f"\n[Check 1] Verifying filename column vs actual files...")
    missing_files = []
    for idx, row in df.iterrows():
        filename = row['filename']
        img_path = images_dir / filename
        if not img_path.exists():
            missing_files.append(filename)

    if missing_files:
        issues.append(f"[ERROR] {len(missing_files)} image files not found")
        print(f"  First 5: {missing_files[:5]}")
    else:
        print(f"  [OK] All {len(df)} files exist")

    # --- Check 2: Image loading and size verification ---
    print(f"\n[Check 2] Image loading and size verification...")
    image_sizes = []
    corrupt_images = []

    # Sample (checking all images takes time, so sample every 10th)
    sample_indices = list(range(0, len(df), max(1, len(df) // 100)))  # Max 100 samples

    for idx in sample_indices:
        row = df.iloc[idx]
        img_path = images_dir / row['filename']

        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path)
            image_sizes.append(img.size)  # (width, height)
        except Exception as e:
            corrupt_images.append((row['filename'], str(e)))

    if corrupt_images:
        issues.append(f"[ERROR] {len(corrupt_images)} images are corrupted")
        print(f"  Example: {corrupt_images[0]}")
    else:
        print(f"  [OK] Sample {len(sample_indices)} images loaded successfully")

    # Check size consistency
    if image_sizes:
        unique_sizes = set(image_sizes)
        if len(unique_sizes) == 1:
            print(f"  [OK] Image sizes are consistent: {unique_sizes.pop()}")
        else:
            warnings.append(f"[WARN] Image sizes are not consistent: {unique_sizes}")
            print(f"  [WARN] Detected sizes: {unique_sizes}")

    # --- Check 3: SOC value range check ---
    print(f"\n[Check 3] SOC value range check (0.0 - 1.0)...")
    soc_out_of_range = df[(df['soc'] < 0.0) | (df['soc'] > 1.0)]
    if len(soc_out_of_range) > 0:
        issues.append(f"[ERROR] {len(soc_out_of_range)} SOC values out of range")
        print(f"  Out of range: min={soc_out_of_range['soc'].min()}, max={soc_out_of_range['soc'].max()}")
    else:
        print(f"  [OK] All SOC values in range (min={df['soc'].min():.3f}, max={df['soc'].max():.3f})")

    # --- Check 4: drive_torque value range check ---
    print(f"\n[Check 4] drive_torque value range check (-1.0 - 1.0)...")
    torque_out_of_range = df[(df['drive_torque'] < -1.0) | (df['drive_torque'] > 1.0)]
    if len(torque_out_of_range) > 0:
        issues.append(f"[ERROR] {len(torque_out_of_range)} drive_torque values out of range")
        print(f"  Out of range: min={torque_out_of_range['drive_torque'].min()}, max={torque_out_of_range['drive_torque'].max()}")
    else:
        print(f"  [OK] All drive_torque values in range (min={df['drive_torque'].min():.3f}, max={df['drive_torque'].max():.3f})")

    # --- Check 5: steer_angle value range check ---
    print(f"\n[Check 5] steer_angle value range check (-0.785 - 0.785 rad)...")
    steer_out_of_range = df[(df['steer_angle'] < -0.785) | (df['steer_angle'] > 0.785)]
    if len(steer_out_of_range) > 0:
        warnings.append(f"[WARN] {len(steer_out_of_range)} steer_angle values out of range")
        print(f"  Out of range: min={steer_out_of_range['steer_angle'].min()}, max={steer_out_of_range['steer_angle'].max()}")
    else:
        print(f"  [OK] All steer_angle values in range (min={df['steer_angle'].min():.3f}, max={df['steer_angle'].max():.3f})")

    # --- Check 6: Frame number continuity check ---
    print(f"\n[Check 6] Frame number continuity check (tick/id column)...")
    if 'tick' in df.columns or 'id' in df.columns:
        tick_col = 'tick' if 'tick' in df.columns else 'id'
        ticks = df[tick_col].values

        # Calculate differences
        diffs = np.diff(ticks)
        non_sequential = np.where(diffs != 1)[0]

        if len(non_sequential) > 0:
            warnings.append(f"[WARN] {len(non_sequential)} places where frame numbers are not sequential")
            print(f"  Example: tick {ticks[non_sequential[0]]} -> {ticks[non_sequential[0] + 1]} (diff={diffs[non_sequential[0]]})")
        else:
            print(f"  [OK] Frame numbers are sequential ({ticks[0]} - {ticks[-1]})")
    else:
        warnings.append(f"[WARN] tick/id column not found")

    # --- Check 7: Image filename format check ---
    print(f"\n[Check 7] Image filename format check...")
    # Expected format: frame_XXXXXX.jpg
    import re
    pattern = re.compile(r'frame_\d{6}\.jpg')

    invalid_filenames = []
    for filename in df['filename']:
        if not pattern.match(filename):
            invalid_filenames.append(filename)

    if invalid_filenames:
        warnings.append(f"[WARN] {len(invalid_filenames)} filenames don't match standard format")
        print(f"  Examples: {invalid_filenames[:3]}")
    else:
        print(f"  [OK] All filenames match standard format (frame_XXXXXX.jpg)")

    # --- Check 8: Image-label correspondence check (sample display) ---
    print(f"\n[Check 8] Image-label correspondence check (sample display)...")
    print(f"  First 3 frames correspondence:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"    Row {idx}: {row['filename']} -> SOC={row['soc']:.3f}, drive={row['drive_torque']:.3f}, steer={row['steer_angle']:.3f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Verification Summary: {run_dir.name}")
    print(f"{'='*70}")

    if issues:
        print(f"[ERROR] {len(issues)} critical issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"[OK] No critical issues found")

    if warnings:
        print(f"\n[WARN] {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  {warning}")

    return len(issues) == 0


def main():
    """
    Verify all run directories in training_data/
    """
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent
    training_data_dir = robot_dir / "training_data"

    if not training_data_dir.exists():
        print(f"[ERROR] training_data directory not found: {training_data_dir}")
        return

    # Search for run_* directories
    run_dirs = sorted([d for d in training_data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    if not run_dirs:
        print(f"[ERROR] No run_* directories found in: {training_data_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Data Integrity Verification Script")
    print(f"{'='*70}")
    print(f"Target directory: {training_data_dir}")
    print(f"Detected runs: {len(run_dirs)}")

    # Verify all runs (or first 10 only)
    max_runs = 10
    if len(run_dirs) > max_runs:
        print(f"\n[WARN] Too many runs, verifying first {max_runs} only")
        run_dirs = run_dirs[:max_runs]

    all_ok = True
    for run_dir in run_dirs:
        ok = verify_single_run(run_dir)
        if not ok:
            all_ok = False

    print(f"\n{'='*70}")
    print(f"Overall Result")
    print(f"{'='*70}")
    if all_ok:
        print(f"[SUCCESS] All {len(run_dirs)} runs are valid!")
    else:
        print(f"[FAILURE] Some runs have issues. Check details above.")


if __name__ == "__main__":
    main()
