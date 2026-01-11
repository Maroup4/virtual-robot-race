# rule_based_input.py (Robot2 version - New Architecture)
# Rule-based driving using sliding window lane detection and driver model
# Compatible with Unity Server + Python Client architecture

import time
import os
import sys
import importlib.util
from pathlib import Path
from PIL import Image

# Module identification
MODULE_SOURCE = "Robot2"
print(f"[rule_based_input] Loaded from {MODULE_SOURCE}/")

# Import data_manager functions for Robot2
_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir.parent))  # Add Project_Beta to path
sys.path.insert(0, str(_this_dir))  # Add Robot2 to path (for rule_based_algorithms)
import data_manager

# Import rule-based algorithms
from rule_based_algorithms import status_Robot
from rule_based_algorithms.sliding_windows import sliding_windows_white
from rule_based_algorithms.driver_model import DriverModel, DriverConfig

# Load perception_Startsignal using importlib to avoid module cache conflicts
_startsignal_path = _this_dir / "rule_based_algorithms" / "perception_Startsignal.py"
_startsignal_spec = importlib.util.spec_from_file_location(
    f"perception_Startsignal_{MODULE_SOURCE}",  # Unique module name per robot
    _startsignal_path
)
_startsignal_module = importlib.util.module_from_spec(_startsignal_spec)
_startsignal_spec.loader.exec_module(_startsignal_module)
detect_start_signal = _startsignal_module.detect_start_signal

# Debug utilities
from rule_based_algorithms.debug_utils import annotate_and_save_canvas
from rule_based_algorithms.debug_utils import overlay_and_save as overlay_and_save_fallback

# --- Globals (control output) ---
robot_id = "R2"
driveTorque: float = 0.0
steerAngle: float = 0.0

# --- Internal state ---
_started_latch = False  # Once GO is detected, stays True
SAVE_DEBUG_OVERLAYS = True  # Save overlay images to debug/ every frame

# Lane-loss handling
_lost_age = 0
HOLD_SEC = 2.0
PERIOD = 0.050
HOLD_FRAMES = max(1, int(HOLD_SEC / PERIOD))

# Mode labels for logging readability
MODE_LABELS = {
    "normal": "Normal",
    "hold": "Hold",
    "search": "Search",
}

# Driver (final decision maker for steer-type control)
_driver = DriverModel(DriverConfig(
    image_width=224,
    forward_sign=+1,
    v_min=0.15,
    v_max=0.75,
    k_theta=0.90,
    k_lateral=0.60,
    steer_limit=0.785,  # ~45 degrees max
    alpha_smooth=0.30,
    torque_limit=1.00,
    theta_hard_limit_deg=80.0,
    use_soc_scaling=True,
    soc_floor=0.30,
))


def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp value within [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def get_latest_command():
    """Return latest control command for WebSocket sender."""
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }


def update():
    """
    Single update cycle for rule-based control.
    Called repeatedly by main.py at ~20Hz.
    Returns True if successful, False if should stop.
    """
    global driveTorque, steerAngle, _started_latch, _lost_age

    try:
        # === Inputs ===
        soc = data_manager.get_latest_soc(robot_id)  # float 0.0..1.0 (can be None)
        rgb_path = data_manager.get_latest_rgb_path(robot_id)

        if not rgb_path or not rgb_path.exists():
            # No image available yet
            return True

        try:
            pil_img = Image.open(rgb_path).convert("RGB")
        except Exception:
            return True

        # === Start signal (latched) ===
        if not _started_latch:
            raw_go = detect_start_signal(pil_img)
            if raw_go:
                _started_latch = True
                status_Robot.set_state(status_Robot.RUN_STRAIGHT)
                print(f"[{robot_id} RuleBased] START latched")

        start_go = _started_latch  # official GO/NO-GO passed to driver

        # === Lane perception (only after GO) ===
        if start_go:
            try:
                sw = sliding_windows_white(
                    pil_img, save_debug=False, src_path=str(rgb_path), return_canvas=True
                )
            except TypeError:
                # Backward compatibility: older version without return_canvas
                sw = sliding_windows_white(pil_img, save_debug=False, src_path=str(rgb_path))

            lane_ok = bool(getattr(sw, "ok", False))
            lateral = getattr(sw, "lateral_px", None) if lane_ok else None
            theta = getattr(sw, "theta_deg", None) if lane_ok else None
            img_w = getattr(sw, "img_width", None) or pil_img.size[0]
            single_side = bool(getattr(sw, "single_side", False))
        else:
            lane_ok = False
            lateral, theta = None, None
            img_w = pil_img.size[0]
            sw = None
            single_side = False

        # === Determine lane_mode from lane state ===
        if lane_ok:
            _lost_age = 0
            lane_mode = "normal"
        else:
            _lost_age += 1
            lane_mode = "hold" if _lost_age <= HOLD_FRAMES else "search"

        # === Driver update (drive/steer decision) ===
        drive, steer = _driver.update(
            lateral_px=lateral,
            theta_deg=theta,
            soc=soc,
            image_width=img_w,
            start_go=start_go,
            valid_lane=lane_ok,
            lane_mode=lane_mode,
            lost_age=_lost_age,
            single_side=single_side,
        )

        driveTorque = saturate(drive)
        steerAngle = saturate(steer, -0.785, 0.785)  # Limit steer to ~±45 deg

        # === Debug overlay save (per frame, after GO) ===
        if SAVE_DEBUG_OVERLAYS and start_go:
            try:
                frame_name = data_manager.get_latest_frame_name(robot_id)
                debug_dir = Path(__file__).parent / "debug"

                # Determine display mode (include Pulse state)
                debug_use_pulse = _driver.last_debug.get("use_pulse", False)
                debug_pulse_phase = _driver.last_debug.get("pulse_phase", "")
                if debug_use_pulse:
                    display_mode = f"Pulse({debug_pulse_phase})"
                else:
                    display_mode = MODE_LABELS.get((lane_mode or "").lower(), lane_mode)

                if sw is not None and getattr(sw, "canvas_bgr", None) is not None:
                    # 1) If SW canvas exists, annotate and save
                    outp = annotate_and_save_canvas(
                        sw.canvas_bgr,
                        out_dir=str(debug_dir),
                        lateral_px=lateral,
                        theta_deg=theta,
                        drive_torque=driveTorque,
                        steer_angle=steerAngle,
                        mode=display_mode,
                        frame_name=frame_name,
                        src_path=str(rgb_path),
                        jpeg_quality=85,
                    )
                else:
                    # 2) Fallback: annotate raw frame with numbers only
                    outp = overlay_and_save_fallback(
                        pil_img,
                        sw_result=sw,
                        driver_debug=_driver.last_debug,
                        out_dir=str(debug_dir),
                    )

                if outp and (_lost_age % 20 == 0 or _lost_age == 1):  # Log periodically
                    print(f"[{robot_id} RuleBased] Saved debug: {outp}")
            except Exception as e:
                if _lost_age % 20 == 0:  # Log errors periodically
                    print(f"[{robot_id} RuleBased] Debug overlay save failed: {e}")

        # === Logging (every 20 frames = ~1 second) ===
        if _lost_age % 20 == 0 or not start_go:
            mode_label = MODE_LABELS.get((lane_mode or "").lower(), str(lane_mode))
            # Check if pulse mode is active
            use_pulse = _driver.last_debug.get("use_pulse", False)
            pulse_phase = _driver.last_debug.get("pulse_phase", "")
            if use_pulse:
                mode_label = f"Pulse({pulse_phase})"
            lat_str = "None" if (lateral is None) else f"{lateral:+.1f}"
            tht_str = "None" if (theta is None) else f"{theta:+.1f}"
            soc_str = "None" if (soc is None) else f"{float(soc):.2f}"
            import math
            steer_deg = math.degrees(steerAngle)

            print(
                f"[{robot_id} RuleBased] Drive={driveTorque:+.2f} Steer={steerAngle:+.3f}rad({steer_deg:+.1f}°) | "
                f"GO={start_go} LaneOK={lane_ok}, {mode_label} LostAge={_lost_age} "
                f"Lat={lat_str} Theta={tht_str} SOC={soc_str}"
            )

        return True

    except Exception as e:
        print(f"[{robot_id} RuleBased] Error: {e}")
        return True


def reset():
    """Reset rule-based control state."""
    global _started_latch, _lost_age, driveTorque, steerAngle
    _started_latch = False
    _lost_age = 0
    driveTorque = 0.0
    steerAngle = 0.0
    print(f"[{robot_id} RuleBased] State reset")
