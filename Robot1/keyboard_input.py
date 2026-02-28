# keyboard_input.py (Robot1 version)
# Allows manual control of drive torque and steering angle using keyboard keys

import threading
import keyboard
import time
import sys

# Module identification
MODULE_SOURCE = "Robot1"
print(f"[keyboard_input] Loaded from {MODULE_SOURCE}/")

# Windows only: clear keyboard input buffer to avoid stuck input
def clear_input_buffer():
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()

# === Control parameters ===
TORQUE_STEP = 0.25     # Step per key press for drive torque
STEER_STEP  = 0.20     # Step per key press for steering angle [rad] (increased for faster response)
MAX_TORQUE  = 1.0
MAX_STEER   = 0.524    # ~30 deg (matches Unity and smartphone)

# === Key mapping ===
# W/Z = forward/backward (drive torque)
# J/L = steer left/right
# I/M = steer reset
KEY_MAP = {"w","z","j","l","i","m"}

# === Current control values ===
driveTorque: float = 0.0
steerAngle:  float = 0.0
robot_id:    str   = "R1"

# Internal state
_key_states = {k: False for k in KEY_MAP}
_listener_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None

def _update_key_state(event):
    name = event.name
    if name in _key_states:
        _key_states[name] = (event.event_type == "down")

def _loop(stop_event: threading.Event):
    global driveTorque, steerAngle
    keyboard.hook(_update_key_state)

    try:
        while not stop_event.is_set():
            # --- Drive torque ---
            if _key_states["w"]:
                driveTorque += TORQUE_STEP
            elif _key_states["z"]:
                driveTorque -= TORQUE_STEP
            else:
                driveTorque = 0.0  # release

            # --- Steering angle (rad) ---
            # Reset on key release (same behavior as smartphone controls)
            if _key_states["j"]:
                steerAngle -= STEER_STEP
            elif _key_states["l"]:
                steerAngle += STEER_STEP
            else:
                steerAngle = 0.0  # release

            # Clamp
            if driveTorque >  MAX_TORQUE: driveTorque =  MAX_TORQUE
            if driveTorque < -MAX_TORQUE: driveTorque = -MAX_TORQUE
            if steerAngle  >  MAX_STEER:  steerAngle  =  MAX_STEER
            if steerAngle  < -MAX_STEER:  steerAngle  = -MAX_STEER

            time.sleep(0.05)
    finally:
        clear_input_buffer()
        print("[Keyboard] Listener stopped.")

# === Public API ===
def start_listener():
    """Start background keyboard listener (idempotent)."""
    global _listener_thread, _stop_event
    if _listener_thread and _listener_thread.is_alive():
        return
    _stop_event = threading.Event()
    _listener_thread = threading.Thread(target=_loop, args=(_stop_event,), daemon=True)
    _listener_thread.start()
    print("[Keyboard] Listener started.")

def stop_listener():
    """Stop background keyboard listener."""
    global _listener_thread, _stop_event
    if _stop_event:
        _stop_event.set()
    _stop_event = None
    _listener_thread = None

def get_latest_command():
    """Return latest control command for WebSocket sender."""
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),  # rad
    }
def debug_print_state():
    print(f"[DEBUG] drive={driveTorque:.2f}, steer={steerAngle:.2f}")
