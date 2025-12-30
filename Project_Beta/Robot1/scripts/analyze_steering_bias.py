#!/usr/bin/env python3
# analyze_steering_bias.py
# ==============================================================================
#                   TRAINING DATA STEERING BIAS ANALYZER
# ==============================================================================
# Analyzes all training data runs to identify steering bias and imbalance

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add Robot1 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_single_run(metadata_path):
    """Analyze a single run's steering data"""
    try:
        df = pd.read_csv(metadata_path)

        # Filter racing data only (exclude StartSequence)
        racing = df[df['status'] != 'StartSequence'].copy()

        if len(racing) == 0:
            return None

        stats = {
            'run_name': Path(metadata_path).parent.name,
            'total_frames': len(racing),
            'mean_steer': racing['steer_angle'].mean(),
            'std_steer': racing['steer_angle'].std(),
            'min_steer': racing['steer_angle'].min(),
            'max_steer': racing['steer_angle'].max(),
            'left_frames': len(racing[racing['steer_angle'] < -0.01]),  # Significant left
            'right_frames': len(racing[racing['steer_angle'] > 0.01]),  # Significant right
            'neutral_frames': len(racing[abs(racing['steer_angle']) <= 0.01]),
            'left_ratio': len(racing[racing['steer_angle'] < -0.01]) / len(racing),
            'right_ratio': len(racing[racing['steer_angle'] > 0.01]) / len(racing),
        }

        return stats
    except Exception as e:
        print(f"Error analyzing {metadata_path}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze steering bias in training data")
    parser.add_argument(
        "--input",
        type=str,
        default="training_data",
        help="Training data directory (default: training_data)"
    )
    args = parser.parse_args()

    print("="*80)
    print("           TRAINING DATA STEERING BIAS ANALYSIS")
    print("="*80)
    print()

    # Find all training data runs
    training_data_dir = Path(__file__).parent.parent / args.input

    if not training_data_dir.exists():
        print(f"Error: Training data directory not found: {training_data_dir}")
        return

    # Find all metadata.csv files
    metadata_files = list(training_data_dir.glob("*/metadata.csv"))

    if not metadata_files:
        print(f"No training data found in {training_data_dir}")
        return

    print(f"Found {len(metadata_files)} training runs\n")

    # Analyze each run
    all_stats = []
    for metadata_path in sorted(metadata_files):
        stats = analyze_single_run(metadata_path)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("No valid training data to analyze")
        return

    # Create DataFrame for analysis
    df_stats = pd.DataFrame(all_stats)

    # Overall statistics
    print("="*80)
    print("OVERALL STATISTICS (All Runs Combined)")
    print("="*80)
    print()

    total_frames = df_stats['total_frames'].sum()
    total_left = df_stats['left_frames'].sum()
    total_right = df_stats['right_frames'].sum()
    total_neutral = df_stats['neutral_frames'].sum()

    print(f"Total training frames: {total_frames:,}")
    print(f"Left steering frames:  {total_left:,} ({total_left/total_frames*100:.1f}%)")
    print(f"Right steering frames: {total_right:,} ({total_right/total_frames*100:.1f}%)")
    print(f"Neutral frames:        {total_neutral:,} ({total_neutral/total_frames*100:.1f}%)")
    print()

    avg_mean_steer = df_stats['mean_steer'].mean()
    print(f"Average mean steer across runs: {avg_mean_steer:.4f} rad")

    if avg_mean_steer < -0.005:
        print("  [WARNING] SIGNIFICANT LEFT BIAS DETECTED")
    elif avg_mean_steer > 0.005:
        print("  [WARNING] SIGNIFICANT RIGHT BIAS DETECTED")
    else:
        print("  [OK] Steering appears balanced")

    print()
    print(f"Min steer observed: {df_stats['min_steer'].min():.4f} rad")
    print(f"Max steer observed: {df_stats['max_steer'].max():.4f} rad")

    # Calculate left/right asymmetry
    left_magnitude = abs(df_stats['min_steer'].min())
    right_magnitude = abs(df_stats['max_steer'].max())
    asymmetry = left_magnitude / right_magnitude if right_magnitude > 0 else 0

    print()
    print(f"Steering asymmetry: {asymmetry:.2f}x")
    print(f"  Left max magnitude:  {left_magnitude:.4f} rad")
    print(f"  Right max magnitude: {right_magnitude:.4f} rad")

    if asymmetry > 1.5:
        print(f"  [WARNING] Left steering is {asymmetry:.1f}x stronger than right!")
    elif asymmetry < 0.67:
        print(f"  [WARNING] Right steering is {1/asymmetry:.1f}x stronger than left!")
    else:
        print("  [OK] Left/right magnitude is balanced")

    # Per-run breakdown
    print()
    print("="*80)
    print("PER-RUN BREAKDOWN")
    print("="*80)
    print()

    # Sort by left bias (most left first)
    df_stats_sorted = df_stats.sort_values('mean_steer')

    print(f"{'Run Name':<30} {'Frames':>8} {'Mean':>8} {'L/R Ratio':>10} {'Bias':<15}")
    print("-"*80)

    for _, row in df_stats_sorted.iterrows():
        run_name = row['run_name'][-28:]  # Truncate if too long
        frames = row['total_frames']
        mean_steer = row['mean_steer']
        lr_ratio = row['left_ratio'] / row['right_ratio'] if row['right_ratio'] > 0 else float('inf')

        # Determine bias
        if mean_steer < -0.005:
            bias = "LEFT BIAS"
        elif mean_steer > 0.005:
            bias = "RIGHT BIAS"
        else:
            bias = "Balanced"

        print(f"{run_name:<30} {frames:>8,} {mean_steer:>8.4f} {lr_ratio:>10.2f} {bias:<15}")

    # Recommendations
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    if total_left > total_right * 1.3:
        print("[CRITICAL] Left steering significantly dominates training data")
        print()
        print("Recommended actions:")
        print("  1. Collect more runs with right steering (left-turn track)")
        print("  2. Apply data augmentation: flip images horizontally and negate steering")
        print("  3. Filter out runs with extreme left bias")
        print()
        print("Quick fix - Add this to train_model.py:")
        print("```python")
        print("# Data augmentation: flip images to balance left/right")
        print("from torchvision import transforms")
        print("augment = transforms.RandomHorizontalFlip(p=0.5)")
        print("# Apply to images, and multiply steering by -1 when flipped")
        print("```")
    elif total_right > total_left * 1.3:
        print("[CRITICAL] Right steering significantly dominates training data")
        print()
        print("Same recommendations as above, but for balancing right bias")
    else:
        print("[OK] Training data steering balance is acceptable")
        print()
        print("If car still shows bias in practice:")
        print("  - Check if specific problematic runs are included")
        print("  - Consider quality over quantity - remove poor runs")
        print("  - Verify data collection was done properly (keyboard vs controller)")

    # Data augmentation example
    print()
    print("="*80)
    print("DATA AUGMENTATION SCRIPT")
    print("="*80)
    print()
    print("To balance left/right automatically, create augmented_training_data/:")
    print()
    print("```bash")
    print("cd Robot1")
    print("python scripts/augment_training_data.py --flip-horizontal --output augmented_training_data/")
    print("```")
    print()
    print("This will:")
    print("  - Keep original images")
    print("  - Add horizontally flipped versions with negated steering")
    print("  - Double your dataset size")
    print("  - Guarantee perfect left/right balance")

    print()
    print("="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
