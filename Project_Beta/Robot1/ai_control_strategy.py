# ai_control_strategy.py
# ==============================================================================
#                        AI CONTROL STRATEGY
#                    - Race Engineer's Playbook -
# ==============================================================================
#
# Welcome, Race Engineer.
#
# This is where you define your team's winning strategy.
# The car is ready. The model is trained. Now it's time to decide HOW to race.
#
# You have two paths:
#
#   [HYBRID MODE] - "Safety First, Then Speed"
#       Use rule-based algorithms for critical moments (start signal, pit stop),
#       and let the AI handle the driving. This is the recommended approach
#       for most teams. Reliable. Consistent. Gets you to the finish line.
#
#   [PURE E2E MODE] - "Trust the Machine"
#       Let the neural network handle EVERYTHING - including start detection.
#       This is the path for those who have trained extensively and believe
#       their model can see what humans programmed. High risk, high reward.
#       False starts are possible. But if your model is good enough...
#
# Remember: The goal is to WIN, not to prove a point.
# Use every tool at your disposal. Mix strategies. Adapt. Overcome.
#
# "It's not about the car. It's not about the driver.
#  It's about the engineer who knows when to trust which."
#
# ==============================================================================

import sys
import os
import importlib.util
from pathlib import Path

# Add this file's directory to path for rule_based_algorithms
_this_dir = Path(__file__).parent

# Load perception_Startsignal using importlib to avoid module cache conflicts
_startsignal_path = _this_dir / "rule_based_algorithms" / "perception_Startsignal.py"
_startsignal_spec = importlib.util.spec_from_file_location(
    f"perception_Startsignal_{_this_dir.name}",  # Unique module name per robot
    _startsignal_path
)
_startsignal_module = importlib.util.module_from_spec(_startsignal_spec)
_startsignal_spec.loader.exec_module(_startsignal_module)
detect_start_signal = _startsignal_module.detect_start_signal

# === STRATEGY SELECTION ===
# Choose your approach:
#   "hybrid"   - Rule-based start detection + AI driving (RECOMMENDED)
#   "pure_e2e" - Full neural network control (ADVANCED)

STRATEGY = "hybrid"


# ==============================================================================
#                         HYBRID STRATEGY SETTINGS
# ==============================================================================
# Fine-tune which parts use rule-based logic vs AI
# These only apply when STRATEGY = "hybrid"

# Start Signal Detection
# - True:  Use perception_Startsignal.py (reliable, no false starts)
# - False: Let AI decide when to start (requires extensive training data)
HYBRID_START_DETECTION = True

# Timeout for start signal detection (in frames, ~20 frames = 1 second)
# If no red lamps are detected for this many frames, assume race already started.
# This handles the case where AI mode is started after the race has begun.
START_DETECTION_TIMEOUT_FRAMES = 60  # ~3 seconds

# Future expansion examples (not yet implemented):
# HYBRID_PITSTOP_DETECTION = False  # Pit stop timing
# HYBRID_EMERGENCY_BRAKE = False    # Collision avoidance


# ==============================================================================
#                    ROBUSTNESS TUNING PARAMETERS
# ==============================================================================
# Fine-tuned parameters for stable cornering and crash avoidance
# Modified for Beta 1.x robustness improvements

# Start boost settings (reverted to original robustness values)
START_BOOST_FRAMES = 10          # ~1.1 seconds (Original robustness setting)
MIN_DRIVE_TORQUE = 0.32          # Reduced from 0.4 to allow better cornering at start
START_BOOST_STEER_THRESHOLD = 0.50  # rad - If steer demand exceeds this, disable boost

# Steering rate limiter (hard safety)
MAX_STEER_DELTA_PER_FRAME = 0.50   # rad/frame - Original robustness setting (0.05 made it worse)

# Drive torque limits
MAX_DRIVE_TORQUE = 0.32           # Overall speed limit
MAX_STEER_RAD = 0.30              # Maximum steering angle (from original code)

# Corner-aware drive torque cap
CORNER_STEER_THRESHOLD_LOW = 0.20   # rad - Below this, full torque allowed
CORNER_STEER_THRESHOLD_HIGH = 0.50  # rad - At/above this, apply minimum torque
CORNER_MIN_DRIVE_TORQUE = 0.30      # Minimum torque during sharp corners

# Steering smoothing
STEER_SMOOTHING_ALPHA = 0.7       # Low-pass filter coefficient (0.7 = original)


# ==============================================================================
#                         ADVANCED CUSTOMIZATION
# ==============================================================================
# For engineers who want fine-grained control over the AI behavior.
# Modify these functions to implement custom logic.

def should_wait_for_start(pil_img, race_started):
    """
    Determine if the car should wait (output zero torque).

    Args:
        pil_img: Current camera image (PIL Image)
        race_started: Whether the race has already started

    Returns:
        True if should wait (torque=0), False if should drive

    Strategy Logic:
        - hybrid: Use rule-based detection until green light
        - pure_e2e: Never wait here, let the model decide
    """
    if STRATEGY == "pure_e2e":
        # Trust the model completely
        return False

    # Hybrid mode: use rule-based start detection
    if not race_started and HYBRID_START_DETECTION:
        # Initialize detection state
        if not hasattr(should_wait_for_start, '_start_detected'):
            should_wait_for_start._start_detected = False
            should_wait_for_start._wait_frames = 0

        # If start was already detected, don't wait anymore
        if should_wait_for_start._start_detected:
            return False  # GO! (start already detected in previous frame)

        # detect_start_signal is loaded at module level using importlib
        # Returns True only ONCE when all red lamps turn OFF (F1-style start)
        if detect_start_signal(pil_img):
            should_wait_for_start._start_detected = True  # Remember we detected start
            should_wait_for_start._wait_frames = 0
            print("[Strategy] Red lamps OFF detected! Race started.")
            return False  # GO!

        # Check if red lamps were ever detected (ready_to_go flag in perception module)
        lamps_detected = getattr(detect_start_signal, 'ready_to_go', False)

        # Increment wait counter
        should_wait_for_start._wait_frames += 1

        # Timeout: If no red lamps detected for a while, assume race already started
        if not lamps_detected and should_wait_for_start._wait_frames >= START_DETECTION_TIMEOUT_FRAMES:
            print(f"[Strategy] Timeout: No red lamps detected for {START_DETECTION_TIMEOUT_FRAMES} frames. Assuming race started.")
            should_wait_for_start._start_detected = True  # Mark as started
            should_wait_for_start._wait_frames = 0
            return False  # GO! (assume race already running)

        return True   # Keep waiting

    return False


def adjust_output(drive, steer, pil_img, soc, race_started=False):
    """
    Post-process AI output before sending to the car.

    Args:
        drive: Raw drive torque from AI (-1.0 to 1.0)
        steer: Raw steering angle from AI (radians)
        pil_img: Current camera image
        soc: State of Charge (0.0 to 1.0)
        race_started: Whether the race has started (from should_wait_for_start)

    Returns:
        (adjusted_drive, adjusted_steer)

    ROBUSTNESS IMPROVEMENTS (Beta 1.x):
        (A) Conditional start boost - Only applies when steering is small
        (B) Steering rate limiter - Hard safety limit on steering changes
        (C) Corner-aware drive cap - Reduces speed during sharp turns
    """
    # Initialize state variables
    if not hasattr(adjust_output, '_race_frame_count'):
        adjust_output._race_frame_count = 0
        adjust_output._race_started_seen = False
        adjust_output._prev_steer = 0.0
        adjust_output._prev_steer_smoothed = 0.0

    # Default: no adjustment
    adjusted_drive = drive
    adjusted_steer = steer

    # Start counting only after race starts
    if race_started:
        if not adjust_output._race_started_seen:
            adjust_output._race_started_seen = True
            adjust_output._race_frame_count = 0
            print("[Strategy] Race start detected! Start boost countdown begins.")
        adjust_output._race_frame_count += 1

    # === Steering limiter (apply early to use in other calculations) ===
    # Prevent extreme steering that causes over-rotation and crashes
    if abs(adjusted_steer) > MAX_STEER_RAD:
        adjusted_steer = MAX_STEER_RAD if adjusted_steer > 0 else -MAX_STEER_RAD

    # === Smooth steering (low-pass filter) ===
    # Reduces jitter and prevents sudden steering changes
    adjusted_steer = STEER_SMOOTHING_ALPHA * adjusted_steer + \
                     (1 - STEER_SMOOTHING_ALPHA) * adjust_output._prev_steer_smoothed
    adjust_output._prev_steer_smoothed = adjusted_steer

    # === (B) STEERING RATE LIMITER (hard safety) ===
    # Prevents sudden steering changes that can cause instability
    delta_steer = adjusted_steer - adjust_output._prev_steer
    if abs(delta_steer) > MAX_STEER_DELTA_PER_FRAME:
        # Clamp the change
        delta_steer = max(-MAX_STEER_DELTA_PER_FRAME,
                         min(MAX_STEER_DELTA_PER_FRAME, delta_steer))
        adjusted_steer = adjust_output._prev_steer + delta_steer

    adjust_output._prev_steer = adjusted_steer

    # === (A) CONDITIONAL START BOOST (corner-aware) ===
    # Only apply minimum torque when steering demand is small
    # This prevents forcing high torque when entering a corner right after start
    if adjust_output._race_started_seen and adjust_output._race_frame_count <= START_BOOST_FRAMES:
        steer_abs = abs(adjusted_steer)

        if steer_abs <= START_BOOST_STEER_THRESHOLD:
            # Straight or gentle turn: apply start boost
            if adjusted_drive < MIN_DRIVE_TORQUE:
                adjusted_drive = MIN_DRIVE_TORQUE
                if adjust_output._race_frame_count % 20 == 0:
                    print(f"[Strategy] Start boost ON ({adjust_output._race_frame_count}/{START_BOOST_FRAMES}): "
                          f"steer={steer_abs:.3f} <= {START_BOOST_STEER_THRESHOLD:.2f}, drive={MIN_DRIVE_TORQUE:.2f}")
        else:
            # Sharp turn detected: DO NOT force minimum torque
            # Let the model output (or corner cap below) control speed
            if adjust_output._race_frame_count % 20 == 0:
                print(f"[Strategy] Start boost SUPPRESSED ({adjust_output._race_frame_count}/{START_BOOST_FRAMES}): "
                      f"steer={steer_abs:.3f} > {START_BOOST_STEER_THRESHOLD:.2f}")

    # === Speed limiter (overall cap) ===
    if adjusted_drive > MAX_DRIVE_TORQUE:
        adjusted_drive = MAX_DRIVE_TORQUE

    # === (C) CORNER-AWARE DRIVE TORQUE CAP ===
    # Reduce drive torque when steering angle is large
    # Linear interpolation between CORNER_STEER_THRESHOLD_LOW and HIGH
    steer_abs = abs(adjusted_steer)

    if steer_abs >= CORNER_STEER_THRESHOLD_LOW:
        # Calculate interpolation factor (0.0 at LOW, 1.0 at HIGH)
        if steer_abs >= CORNER_STEER_THRESHOLD_HIGH:
            t = 1.0
        else:
            t = (steer_abs - CORNER_STEER_THRESHOLD_LOW) / \
                (CORNER_STEER_THRESHOLD_HIGH - CORNER_STEER_THRESHOLD_LOW)

        # Clamp t to [0, 1]
        t = max(0.0, min(1.0, t))

        # Linear interpolation: lerp(MAX, MIN, t) = MAX + t * (MIN - MAX)
        drive_cap = MAX_DRIVE_TORQUE + t * (CORNER_MIN_DRIVE_TORQUE - MAX_DRIVE_TORQUE)

        if adjusted_drive > drive_cap:
            adjusted_drive = drive_cap
            # Log every 10 frames to avoid spam
            if adjust_output._race_frame_count % 10 == 0:
                print(f"[Strategy] Corner cap: steer={steer_abs:.3f}, "
                      f"drive capped {drive:.2f} -> {drive_cap:.2f}")

    # === Example: Energy conservation when SOC is low ===
    # if soc < 0.2:
    #     adjusted_drive = drive * 0.8  # Reduce power by 20%

    return adjusted_drive, adjusted_steer


def on_race_start():
    """
    Called once when the race starts (green light detected).
    Use this for any initialization needed at race start.
    """
    print("[Strategy] Race started! Executing race start protocol.")
    # Example: Could log telemetry, reset counters, etc.
    pass


def on_lap_complete(lap_number, lap_time):
    """
    Called when a lap is completed (if lap detection is available).
    Use this to adjust strategy mid-race.

    Args:
        lap_number: Which lap was just completed
        lap_time: Time for that lap in seconds
    """
    # Example: Adjust aggression based on lap times
    # if lap_time > target_time:
    #     increase_aggression()
    pass


# ==============================================================================
#                              NOTES FOR ENGINEERS
# ==============================================================================
#
# Q: When should I use pure_e2e?
# A: Only when your model has been trained on LOTS of data that includes
#    the start sequence. The model must learn that red lights = zero torque.
#    If you're not sure, stick with hybrid.
#
# Q: Can I mix more rule-based logic?
# A: Absolutely! That's what adjust_output() is for. You can check the image
#    for specific conditions and override or modify the AI output.
#
# Q: My car is unstable. What should I do?
# A: Try adding smoothing in adjust_output(). Also check if your training
#    data had smooth inputs or jerky keyboard mashing.
#
# Q: How do I add pit stop detection?
# A: Create a perception function in rule_based_algorithms/, then call it
#    from should_wait_for_start() or a new hook function.
#
# ==============================================================================
# Good luck, Engineer. Make your team proud.
# ==============================================================================
