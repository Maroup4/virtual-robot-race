# rule_based_input.py
# Entry point for rule-based driving:
#  - Start signal is checked until it becomes True once (latched).
#  - Torques are ALWAYS produced by driver_model.update().
#  - Lane perception is only called after GO for efficiency.

import time
import os
from PIL import Image
import data_manager

from rule_based_algorithms import status_Robot
from rule_based_algorithms import perception_Startsignal
from rule_based_algorithms.sliding_windows import sliding_windows_white
from rule_based_algorithms.driver_model import DriverModel, DriverConfig

# --- Debug utilities ---
# Option 1: overlay on SW canvas (preferred, adds HUD and torques)
from rule_based_algorithms.debug_utils import annotate_and_save_canvas
# Option 2: fallback overlay on raw frame (numbers only)
from rule_based_algorithms.debug_utils import overlay_and_save as overlay_and_save_fallback

try:
    # Optional: get frame name if available (used for debug filename)
    from data_manager import get_latest_frame_name
except Exception:
    get_latest_frame_name = None

# --- Globals (read by websocket_server) ---
leftTorque = 0.0
rightTorque = 0.0

# --- Internal flags ---
_started_latch = False        # Once GO is detected, stays True
SAVE_DEBUG_OVERLAYS = True    # Save overlay images to debug/ every frame

# Lane-loss handling
_lost_age = 0
HOLD_SEC = 2.0
PERIOD = 0.050
HOLD_FRAMES = max(1, int(HOLD_SEC / PERIOD))

# Mode labels for logging readability
MODE_LABELS = {
    "normal": "Normal",
    "hold":   "Hold",
    "search": "Search",
}


def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp value within [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def get_latest_rgb_path():
    """
    Return the latest RGB JPG path using the A/B toggle file.
    Falls back to latest_RGB_a.jpg if the flag file is missing.
    """
    flag_path = os.path.join("data_interactive", "latest_RGB_now.txt")
    try:
        with open(flag_path, "r") as f:
            mark = f.read().strip()
            if mark in ("a", "b"):
                return os.path.join("data_interactive", f"latest_RGB_{mark}.jpg")
    except Exception:
        pass
    return os.path.join("data_interactive", "latest_RGB_a.jpg")  # fallback


# Driver (final torque decision maker)
_driver = DriverModel(DriverConfig(
    image_width=224,
    forward_sign=+1,
    v_min=0.15,
    v_max=0.75,
    k_theta=0.90,
    k_lateral=0.60,
    yaw_mix_gain=1.00,
    alpha_smooth=0.30,
    torque_limit=1.00,
    theta_hard_limit_deg=80.0,
    use_soc_scaling=True,
    soc_floor=0.30,
))


def run_rule_based_loop(stop_event):
    """
    Main rule-based control loop (target 50 ms).
    - Waits for start signal (latched).
    - After GO: runs sliding-window lane detection and driver_model update.
    - Always outputs torques (leftTorque, rightTorque).
    """
    global leftTorque, rightTorque, _started_latch, _lost_age

    print("[RuleBased] Control loop started.")
    next_t = time.monotonic()

    while not stop_event.is_set():
        try:
            # === Inputs ===
            soc = data_manager.get_latest_soc()  # float 0.0..1.0 (can be None)
            img_path = get_latest_rgb_path()
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                time.sleep(0.01)
                continue

            # === Start signal (latched) ===
            if not _started_latch:
                raw_go = perception_Startsignal.detect_start_signal(pil_img)
                if raw_go:
                    _started_latch = True
                    status_Robot.set_state(status_Robot.RUN_STRAIGHT)
                    print("[RuleBased] START latched")

            start_go = _started_latch  # official GO/NO-GO passed to driver

            # === Lane perception (only after GO) ===
            if start_go:
                try:
                    sw = sliding_windows_white(
                        pil_img, save_debug=False, src_path=img_path, return_canvas=True
                    )
                except TypeError:
                    # Backward compatibility: older version without return_canvas
                    sw = sliding_windows_white(pil_img, save_debug=False, src_path=img_path)

                lane_ok = bool(getattr(sw, "ok", False))
                lateral = getattr(sw, "lateral_px", None) if lane_ok else None
                theta = getattr(sw, "theta_deg", None) if lane_ok else None
                img_w = getattr(sw, "img_width", None) or pil_img.size[0]
            else:
                lane_ok = False
                lateral, theta = None, None
                img_w = pil_img.size[0]
                sw = None

            # === Determine lane_mode from lane state ===
            if lane_ok:
                _lost_age = 0
                lane_mode = "normal"
            else:
                _lost_age += 1
                lane_mode = "hold" if _lost_age <= HOLD_FRAMES else "search"

            # === Driver update (torque decision) ===
            lt, rt = _driver.update(
                lateral_px=lateral,
                theta_deg=theta,
                soc=soc,
                image_width=img_w,
                start_go=start_go,
                valid_lane=lane_ok,
                lane_mode=lane_mode,
                lost_age=_lost_age,
            )

            leftTorque = saturate(lt)
            rightTorque = saturate(rt)

            # === Debug overlay save (per frame, after GO) ===
            if SAVE_DEBUG_OVERLAYS and start_go:
                try:
                    frame_name = get_latest_frame_name() if get_latest_frame_name else None

                    if sw is not None and getattr(sw, "canvas_bgr", None) is not None:
                        # 1) If SW canvas exists, annotate and save
                        outp = annotate_and_save_canvas(
                            sw.canvas_bgr,
                            out_dir="debug",
                            lateral_px=lateral,
                            theta_deg=theta,
                            torque_left=leftTorque,
                            torque_right=rightTorque,
                            mode=lane_mode,
                            frame_name=frame_name,
                            src_path=img_path,
                            jpeg_quality=85,
                        )
                    else:
                        # 2) Fallback: annotate raw frame with numbers only
                        outp = overlay_and_save_fallback(
                            pil_img,
                            sw_result=sw,
                            driver_debug=_driver.last_debug,
                            out_dir="debug",
                        )

                    if outp:
                        print(f"[RuleBased] Saved debug: {outp}")
                except Exception as e:
                    print(f"[RuleBased] Debug overlay save failed: {e}")

            # === Logging ===
            mode_label = MODE_LABELS.get((lane_mode or "").lower(), str(lane_mode))
            lat_str = "None" if (lateral is None) else f"{lateral:+.1f}"
            tht_str = "None" if (theta is None) else f"{theta:+.1f}"
            soc_str = "None" if (soc is None) else f"{float(soc):.2f}"

            print(
                f"[RuleBased] Torque L={leftTorque:+.2f} R={rightTorque:+.2f} | "
                f"GO={start_go} LaneOK={lane_ok}, {mode_label} LostAge={_lost_age} "
                f"Lat={lat_str} Theta={tht_str} SOC={soc_str}"
            )

        except Exception as e:
            print(f"[RuleBased] Error: {e}")

        # 50 ms cycle
        next_t += PERIOD
        time.sleep(max(0.0, next_t - time.monotonic()))

    print("[RuleBased] Control loop stopped.")
