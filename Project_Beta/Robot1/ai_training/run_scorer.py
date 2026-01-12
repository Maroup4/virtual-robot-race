# run_scorer.py
# ==============================================================================
# Run Scoring Module for DAgger + Reward-based Data Selection
# ==============================================================================
#
# This module calculates scores for each training run based on:
#   - Completion status (Finish, Lap1, Fallen, etc.)
#   - Race time (faster = higher score)
#   - SOC efficiency (more remaining = higher score)
#   - Driving smoothness (less jitter = higher score)
#
# Usage:
#   python run_scorer.py                    # Score all runs in training_data/
#   python run_scorer.py --min-score 500    # List runs with score >= 500
#
# ==============================================================================

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

# ==============================================================================
# SCORING WEIGHTS - Tune these to change what "good" means
# ==============================================================================

SCORING_WEIGHTS = {
    # Completion bonuses
    'finish_bonus': 1000.0,       # Completed 2 laps
    'lap1_bonus': 400.0,          # Completed 1 lap
    'lap0_bonus': 100.0,          # Started but didn't complete a lap

    # Penalties
    'fallen_penalty': -500.0,     # Robot fell
    'force_end_penalty': -100.0,  # Manually stopped (q key)

    # Time scoring (for finished runs)
    # score = time_bonus_base - (race_time_seconds * time_penalty_per_second)
    'time_bonus_base': 500.0,     # Base time score
    'time_penalty_per_second': 2.0,  # Penalty per second
    'time_bonus_min': 0.0,        # Minimum time score (floor)

    # SOC efficiency
    # score = soc_remaining * soc_weight
    'soc_weight': 200.0,          # Multiplied by remaining SOC (0-1)

    # Smoothness scoring
    # Lower steering jerk = higher score
    'smoothness_weight': 100.0,   # Base smoothness score
    'jerk_penalty_factor': 10.0,  # Penalty multiplier for steering jerk
}

# Reference time for normalization (seconds)
# Runs faster than this get bonus, slower get penalty
REFERENCE_TIME_SECONDS = 120.0  # 2 minutes


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================

def calculate_run_score(metadata_path: Path) -> Dict:
    """
    Calculate score for a single run based on its metadata.csv.

    Args:
        metadata_path: Path to metadata.csv file

    Returns:
        dict with score breakdown and total
    """
    result = {
        'path': str(metadata_path.parent),
        'run_name': metadata_path.parent.name,
        'total_score': 0.0,
        'breakdown': {},
        'stats': {},
        'valid': False,
        'error': None,
    }

    try:
        rows = _read_metadata(metadata_path)
        if not rows:
            result['error'] = "Empty or invalid metadata.csv"
            return result

        # Extract key metrics
        final_status = _get_final_status(rows)
        race_time_ms = _get_race_time(rows)
        final_soc = _get_final_soc(rows)
        steering_jerk = _calculate_steering_jerk(rows)
        frame_count = len(rows)

        result['stats'] = {
            'final_status': final_status,
            'race_time_ms': race_time_ms,
            'race_time_sec': race_time_ms / 1000.0 if race_time_ms else 0,
            'final_soc': final_soc,
            'steering_jerk': steering_jerk,
            'frame_count': frame_count,
        }

        # Calculate score components
        breakdown = {}

        # (1) Completion bonus/penalty
        completion_score = _score_completion(final_status)
        breakdown['completion'] = completion_score

        # (2) Time score (only for finished runs)
        time_score = 0.0
        if final_status == 'Finish' and race_time_ms:
            time_score = _score_time(race_time_ms / 1000.0)
        breakdown['time'] = time_score

        # (3) SOC efficiency
        soc_score = _score_soc(final_soc) if final_soc is not None else 0.0
        breakdown['soc'] = soc_score

        # (4) Smoothness
        smoothness_score = _score_smoothness(steering_jerk)
        breakdown['smoothness'] = smoothness_score

        # Total
        total = sum(breakdown.values())

        result['breakdown'] = breakdown
        result['total_score'] = total
        result['valid'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def _read_metadata(path: Path) -> List[Dict]:
    """Read metadata.csv and return list of row dicts."""
    rows = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def _get_final_status(rows: List[Dict]) -> str:
    """Get the final status from metadata rows."""
    if not rows:
        return 'Unknown'

    # Check last few rows for status (sometimes status changes at end)
    for row in reversed(rows[-10:]):
        status = row.get('status', '')
        if status in ['Finish', 'Fallen', 'Force end']:
            return status

    # Fall back to last row's status
    last_status = rows[-1].get('status', 'Unknown')

    # Normalize status names
    if 'Lap1' in last_status or last_status == 'Lap1':
        return 'Lap1'
    elif 'Lap0' in last_status or last_status == 'Lap0':
        return 'Lap0'

    return last_status


def _get_race_time(rows: List[Dict]) -> Optional[int]:
    """Get final race time in milliseconds."""
    if not rows:
        return None

    # Find max race_time_ms
    max_time = 0
    for row in rows:
        try:
            t = int(row.get('race_time_ms', 0))
            max_time = max(max_time, t)
        except (ValueError, TypeError):
            pass

    return max_time if max_time > 0 else None


def _get_final_soc(rows: List[Dict]) -> Optional[float]:
    """Get final SOC value."""
    if not rows:
        return None

    # Get last valid SOC
    for row in reversed(rows):
        try:
            soc = float(row.get('soc', ''))
            if 0 <= soc <= 1:
                return soc
        except (ValueError, TypeError):
            pass

    return None


def _calculate_steering_jerk(rows: List[Dict]) -> float:
    """
    Calculate average steering jerk (rate of change of steering).
    Lower is smoother.
    """
    if len(rows) < 3:
        return 0.0

    steers = []
    for row in rows:
        try:
            steer = float(row.get('steer_angle', 0))
            steers.append(steer)
        except (ValueError, TypeError):
            pass

    if len(steers) < 3:
        return 0.0

    # Calculate jerk (second derivative of steering)
    jerks = []
    for i in range(2, len(steers)):
        # First derivative (rate)
        rate_prev = steers[i-1] - steers[i-2]
        rate_curr = steers[i] - steers[i-1]
        # Second derivative (jerk)
        jerk = abs(rate_curr - rate_prev)
        jerks.append(jerk)

    return sum(jerks) / len(jerks) if jerks else 0.0


def _score_completion(status: str) -> float:
    """Score based on completion status."""
    w = SCORING_WEIGHTS

    if status == 'Finish':
        return w['finish_bonus']
    elif status == 'Lap1':
        return w['lap1_bonus']
    elif status == 'Lap0':
        return w['lap0_bonus']
    elif status == 'Fallen':
        return w['fallen_penalty']
    elif status == 'Force end':
        return w['force_end_penalty']
    else:
        return 0.0


def _score_time(time_seconds: float) -> float:
    """Score based on race time (faster = better)."""
    w = SCORING_WEIGHTS

    # Score = base - (time * penalty_per_second)
    score = w['time_bonus_base'] - (time_seconds * w['time_penalty_per_second'])

    # Apply floor
    return max(score, w['time_bonus_min'])


def _score_soc(soc: float) -> float:
    """Score based on remaining SOC."""
    return soc * SCORING_WEIGHTS['soc_weight']


def _score_smoothness(jerk: float) -> float:
    """Score based on steering smoothness (lower jerk = higher score)."""
    w = SCORING_WEIGHTS

    # Base score minus jerk penalty
    score = w['smoothness_weight'] - (jerk * w['jerk_penalty_factor'])

    return max(score, 0.0)


# ==============================================================================
# BATCH SCORING
# ==============================================================================

def score_all_runs(training_data_dir: Path) -> List[Dict]:
    """
    Score all runs in a training_data directory.

    Args:
        training_data_dir: Path to training_data/ directory

    Returns:
        List of score results, sorted by total_score descending
    """
    results = []

    if not training_data_dir.exists():
        print(f"[Scorer] Directory not found: {training_data_dir}")
        return results

    # Find all run directories
    run_dirs = sorted(training_data_dir.glob("run_*"))

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.csv"
        if metadata_path.exists():
            result = calculate_run_score(metadata_path)
            results.append(result)

    # Sort by score (highest first)
    results.sort(key=lambda x: x['total_score'], reverse=True)

    return results


def filter_runs_by_score(results: List[Dict], min_score: float = 0.0) -> List[Dict]:
    """Filter runs to only include those with score >= min_score."""
    return [r for r in results if r['valid'] and r['total_score'] >= min_score]


def get_top_runs(results: List[Dict], top_percent: float = 50.0) -> List[Dict]:
    """Get top N% of runs by score."""
    valid_results = [r for r in results if r['valid']]
    if not valid_results:
        return []

    n = max(1, int(len(valid_results) * top_percent / 100.0))
    return valid_results[:n]


def save_scores_json(results: List[Dict], output_path: Path) -> None:
    """Save scoring results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Scorer] Scores saved to {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def print_results(results: List[Dict], verbose: bool = False) -> None:
    """Print scoring results in a readable format."""
    print("\n" + "=" * 70)
    print("RUN SCORING RESULTS")
    print("=" * 70)

    valid_count = sum(1 for r in results if r['valid'])
    print(f"Total runs: {len(results)}, Valid: {valid_count}")
    print("-" * 70)

    for i, r in enumerate(results):
        if not r['valid']:
            print(f"{i+1:3}. {r['run_name']}: INVALID - {r['error']}")
            continue

        stats = r['stats']
        score = r['total_score']

        status_icon = {
            'Finish': '[OK]',
            'Lap1': '[L1]',
            'Lap0': '[L0]',
            'Fallen': '[XX]',
            'Force end': '[--]',
        }.get(stats['final_status'], '[??]')

        print(f"{i+1:3}. {status_icon} {r['run_name']}: {score:7.1f} pts")

        if verbose:
            print(f"      Status: {stats['final_status']}, "
                  f"Time: {stats['race_time_sec']:.1f}s, "
                  f"SOC: {stats['final_soc']:.2f}, "
                  f"Frames: {stats['frame_count']}")
            breakdown = r['breakdown']
            print(f"      Breakdown: completion={breakdown['completion']:.0f}, "
                  f"time={breakdown['time']:.0f}, "
                  f"soc={breakdown['soc']:.0f}, "
                  f"smooth={breakdown['smoothness']:.0f}")

    print("=" * 70)

    # Summary stats
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        scores = [r['total_score'] for r in valid_results]
        print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"Average score: {sum(scores)/len(scores):.1f}")

        finish_count = sum(1 for r in valid_results
                         if r['stats']['final_status'] == 'Finish')
        print(f"Finish rate: {finish_count}/{len(valid_results)} "
              f"({100*finish_count/len(valid_results):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Score training runs for DAgger data selection"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help="Path to training_data directory (default: ../training_data)"
    )
    parser.add_argument(
        '--min-score', '-m',
        type=float,
        default=None,
        help="Only show runs with score >= this value"
    )
    parser.add_argument(
        '--top-percent', '-t',
        type=float,
        default=None,
        help="Only show top N%% of runs"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show detailed breakdown"
    )
    parser.add_argument(
        '--list-paths',
        action='store_true',
        help="Output only run paths (for piping to other tools)"
    )

    args = parser.parse_args()

    # Determine training_data path
    if args.data:
        data_dir = Path(args.data)
    else:
        # Default: Robot1/training_data relative to this script
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "training_data"

    print(f"[Scorer] Scanning: {data_dir}")

    # Score all runs
    results = score_all_runs(data_dir)

    if not results:
        print("[Scorer] No runs found")
        return

    # Apply filters
    if args.min_score is not None:
        results = filter_runs_by_score(results, args.min_score)
        print(f"[Scorer] Filtered to {len(results)} runs with score >= {args.min_score}")

    if args.top_percent is not None:
        results = get_top_runs(results, args.top_percent)
        print(f"[Scorer] Filtered to top {args.top_percent}% ({len(results)} runs)")

    # Output
    if args.list_paths:
        # Simple path output for scripting
        for r in results:
            if r['valid']:
                print(r['path'])
    else:
        print_results(results, verbose=args.verbose)

    # Save to JSON if requested
    if args.output:
        save_scores_json(results, Path(args.output))


if __name__ == "__main__":
    main()
