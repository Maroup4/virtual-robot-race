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

# Future expansion examples (not yet implemented):
# HYBRID_PITSTOP_DETECTION = False  # Pit stop timing
# HYBRID_EMERGENCY_BRAKE = False    # Collision avoidance


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
        # detect_start_signal is loaded at module level using importlib
        # Wait unless start signal is detected
        if detect_start_signal(pil_img):
            return False  # GO!
        else:
            return True   # Keep waiting

    return False


def adjust_output(drive, steer, pil_img, soc):
    """
    Post-process AI output before sending to the car.

    Args:
        drive: Raw drive torque from AI (-1.0 to 1.0)
        steer: Raw steering angle from AI (radians)
        pil_img: Current camera image
        soc: State of Charge (0.0 to 1.0)

    Returns:
        (adjusted_drive, adjusted_steer)

    Examples of what you could do here:
        - Apply smoothing filter to reduce jitter
        - Reduce torque when SOC is low to conserve energy
        - Boost torque on straights detected by image analysis
        - Limit steering rate for stability
    """
    # Default: no adjustment
    adjusted_drive = drive
    adjusted_steer = steer

    # === Example: Energy conservation when SOC is low ===
    # if soc < 0.2:
    #     adjusted_drive = drive * 0.8  # Reduce power by 20%

    # === Example: Smooth steering (low-pass filter) ===
    # if not hasattr(adjust_output, '_prev_steer'):
    #     adjust_output._prev_steer = steer
    # alpha = 0.7  # Smoothing factor
    # adjusted_steer = alpha * steer + (1 - alpha) * adjust_output._prev_steer
    # adjust_output._prev_steer = adjusted_steer

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
