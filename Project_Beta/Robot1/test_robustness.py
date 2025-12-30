# test_robustness.py
# ==============================================================================
#           Unit Test for AI Control Strategy Robustness Improvements
# ==============================================================================
# Tests the three improvements:
# (A) Conditional start boost
# (B) Steering rate limiter
# (C) Corner-aware drive torque cap

import sys
sys.path.insert(0, 'Robot1')
from ai_control_strategy import adjust_output

def reset_adjust_output_state():
    """Clear all state variables from adjust_output function"""
    for attr in ['_race_frame_count', '_race_started_seen', '_prev_steer', '_prev_steer_smoothed']:
        if hasattr(adjust_output, attr):
            delattr(adjust_output, attr)

def test_conditional_start_boost():
    """Test (A): Start boost should activate only when steering is small"""
    print("="*70)
    print("TEST (A): CONDITIONAL START BOOST")
    print("="*70)
    print("Expected: Boost activates when steer <= 0.10, suppressed when > 0.10\n")

    reset_adjust_output_state()

    test_cases = [
        (0.25, 0.00, "Straight line (steer=0.00)", True),
        (0.25, 0.05, "Gentle turn (steer=0.05)", True),
        (0.25, 0.10, "Threshold (steer=0.10)", True),
        (0.25, 0.15, "Sharp turn (steer=0.15)", False),
        (0.25, 0.25, "Very sharp (steer=0.25)", False),
    ]

    for i, (drive_in, steer_in, desc, should_boost) in enumerate(test_cases):
        adj_drive, adj_steer = adjust_output(
            drive_in, steer_in,
            pil_img=None,
            soc=1.0,
            race_started=True
        )

        # Boost should force drive to 0.32 when activated
        boost_active = adj_drive >= 0.32 and drive_in < 0.32

        status = "[PASS]" if boost_active == should_boost else "[FAIL]"
        print(f"Frame {i+1:2d}: {desc:30s} | Boost: {boost_active} | {status}")
        print(f"          Input:  drive={drive_in:.2f}, steer={steer_in:.2f}")
        print(f"          Output: drive={adj_drive:.2f}, steer={adj_steer:.2f}\n")

def test_steering_rate_limiter():
    """Test (B): Steering changes should be limited to 0.03 rad/frame"""
    print("\n" + "="*70)
    print("TEST (B): STEERING RATE LIMITER")
    print("="*70)
    print("Expected: Max steering change = 0.03 rad/frame\n")

    reset_adjust_output_state()

    # Sequence: try to jump from 0.0 to 0.20 instantly
    test_sequence = [
        (0.5, 0.00, "Initial position"),
        (0.5, 0.20, "Attempt large jump to 0.20"),
        (0.5, 0.20, "Continue requesting 0.20"),
        (0.5, 0.20, "Continue requesting 0.20"),
        (0.5, 0.20, "Continue requesting 0.20"),
        (0.5, 0.00, "Attempt jump back to 0.00"),
        (0.5, 0.00, "Continue requesting 0.00"),
    ]

    prev_steer = 0.0
    for i, (drive_in, steer_in, desc) in enumerate(test_sequence):
        adj_drive, adj_steer = adjust_output(
            drive_in, steer_in,
            pil_img=None,
            soc=1.0,
            race_started=True
        )

        delta = abs(adj_steer - prev_steer)
        status = "[PASS]" if delta <= 0.031 else "[FAIL]"  # 0.031 for rounding tolerance

        print(f"Frame {i+1}: {desc:30s} | Delta: {delta:.4f} | {status}")
        print(f"        Request: {steer_in:.3f} -> Output: {adj_steer:.3f}\n")

        prev_steer = adj_steer

def test_corner_aware_torque_cap():
    """Test (C): Drive torque should reduce when steering is large"""
    print("\n" + "="*70)
    print("TEST (C): CORNER-AWARE DRIVE TORQUE CAP")
    print("="*70)
    print("Expected: Torque reduces linearly from steer=0.20 to 0.50\n")

    reset_adjust_output_state()

    # Build up steering gradually to overcome rate limiter
    # First, ramp up to 0.30 over multiple frames
    for i in range(15):
        adjust_output(0.55, 0.30, None, 1.0, True)

    # Now test at various steering angles (already at 0.30)
    print("After ramping to steer=0.30 (15 frames):\n")

    test_cases = [
        (0.55, 0.30, 0.47, "At 0.30 rad (medium corner)"),
        (0.55, 0.40, None, "Ramp to 0.40 (sharp corner)"),
        (0.55, 0.50, None, "Ramp to 0.50 (very sharp)"),
    ]

    for i, (drive_in, steer_target, expected_max, desc) in enumerate(test_cases):
        # Ramp to target over multiple frames
        for _ in range(5):
            adj_drive, adj_steer = adjust_output(
                drive_in, steer_target,
                pil_img=None,
                soc=1.0,
                race_started=True
            )

        # Check if torque was capped
        if expected_max is not None:
            status = "[PASS]" if abs(adj_drive - expected_max) < 0.03 else "[FAIL]"
            print(f"Test {i+1}: {desc:35s} | Expected: ~{expected_max:.2f} | Got: {adj_drive:.2f} | {status}")
        else:
            # Just show the result
            print(f"Test {i+1}: {desc:35s} | Actual steer: {adj_steer:.3f} | Drive: {adj_drive:.2f}")

    # Test the extremes with direct manipulation (bypass rate limiter for verification)
    print("\nDirect tests (bypassing rate limiter for verification):")
    reset_adjust_output_state()

    # Override prev_steer to test corner cap directly
    adjust_output._prev_steer = 0.20
    adjust_output._prev_steer_smoothed = 0.20
    adj_drive, adj_steer = adjust_output(0.55, 0.20, None, 1.0, True)
    print(f"  steer=0.20 (threshold LOW): drive={adj_drive:.2f} (should be ~0.55)")

    adjust_output._prev_steer = 0.35
    adjust_output._prev_steer_smoothed = 0.35
    adj_drive, adj_steer = adjust_output(0.55, 0.35, None, 1.0, True)
    print(f"  steer=0.35 (mid-range):     drive={adj_drive:.2f} (should be ~0.42)")

    adjust_output._prev_steer = 0.50
    adjust_output._prev_steer_smoothed = 0.50
    adj_drive, adj_steer = adjust_output(0.55, 0.50, None, 1.0, True)
    print(f"  steer=0.50 (threshold HIGH): drive={adj_drive:.2f} (should be ~0.30)\n")

def test_combined_scenario():
    """Test real-world scenario: Start -> Straight -> Corner -> Exit"""
    print("\n" + "="*70)
    print("TEST (D): COMBINED SCENARIO")
    print("="*70)
    print("Simulating: Race start -> Straight -> Enter corner -> Exit\n")

    reset_adjust_output_state()

    # Realistic sequence
    scenario = [
        # Frame, Drive, Steer, Description
        (1, 0.30, 0.00, "Race starts (straight)"),
        (2, 0.35, 0.02, "Accelerating (slight right)"),
        (3, 0.40, 0.05, "Building speed"),
        (4, 0.45, 0.08, "Approaching corner"),
        (5, 0.50, 0.15, "Corner entry (boost should suppress)"),
        (6, 0.50, 0.25, "Mid-corner (torque should cap)"),
        (7, 0.50, 0.30, "Apex (max torque cap)"),
        (8, 0.50, 0.20, "Corner exit (cap easing)"),
        (9, 0.50, 0.10, "Straightening out"),
        (10, 0.55, 0.05, "Back to straight"),
    ]

    print(f"{'Frame':<6} {'Desc':<30} {'Drive In':<10} {'Steer In':<10} {'Drive Out':<11} {'Steer Out':<11}")
    print("-" * 88)

    for frame, drive_in, steer_in, desc in scenario:
        adj_drive, adj_steer = adjust_output(
            drive_in, steer_in,
            pil_img=None,
            soc=1.0,
            race_started=True
        )

        print(f"{frame:<6} {desc:<30} {drive_in:<10.2f} {steer_in:<10.2f} {adj_drive:<11.2f} {adj_steer:<11.3f}")

    print("\n" + "="*70)
    print("Look for:")
    print("  - Drive clamped to 0.32 in early frames (start boost)")
    print("  - Boost suppressed at frame 5 (steer > 0.10)")
    print("  - Drive capped below 0.50 during frames 6-8 (corner cap)")
    print("  - Steering changes limited to 0.03/frame")
    print("="*70)

if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("   VRR Beta 1.x - AI Control Strategy Robustness Test Suite")
    print("="*70)
    print()

    test_conditional_start_boost()
    test_steering_rate_limiter()
    test_corner_aware_torque_cap()
    test_combined_scenario()

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    print("\nTo verify in actual race, check logs for:")
    print("  [Strategy] Start boost SUPPRESSED")
    print("  [Strategy] Corner cap: steer=X.XXX, drive capped")
    print()
