# keyboard_input.py
# Allows manual control of left and right wheel torque using keyboard keys

import threading
import keyboard
import time
import sys

# Windows only: clear keyboard input buffer to avoid stuck input
def clear_input_buffer():
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()

# === Torque control parameters ===
TORQUE_STEP = 0.25     # Torque applied per key press
DECAY_FACTOR = 0.2     # Decay rate when key is released (not used here)
MAX_TORQUE = 1.0       # Max absolute torque value

# Key-to-direction mapping (w/z for left, i/m for right)
TORQUE_VALUES = {
    "w": (1, 0),     # Left wheel forward
    "z": (-1, 0),    # Left wheel backward
    "i": (0, 1),     # Right wheel forward
    "m": (0, -1)     # Right wheel backward
}

# Current torque values
leftTorque = 0.0
rightTorque = 0.0

# Track key states (pressed/released)
key_states = {key: False for key in TORQUE_VALUES}
key_states["space"] = False  # Reserved for future use (e.g., brake)

# Update key state on press/release
def update_key_state(event):
    if event.name in key_states:
        key_states[event.name] = event.event_type == "down"

# Main input listener loop
def listen_for_input(stop_event):
    global leftTorque, rightTorque

    keyboard.hook(update_key_state)

    while not stop_event.is_set():
        delta_l, delta_r = 0.0, 0.0

        for key, (lt, rt) in TORQUE_VALUES.items():
            if key_states[key]:
                delta_l += TORQUE_STEP * lt
                delta_r += TORQUE_STEP * rt

        # Apply torque changes
        leftTorque += delta_l
        rightTorque += delta_r

        # Instant decay when key is released
        if delta_l == 0:
            leftTorque = 0.0
        if delta_r == 0:
            rightTorque = 0.0

        # Clamp torque to maximum allowed range
        leftTorque = max(-MAX_TORQUE, min(MAX_TORQUE, leftTorque))
        rightTorque = max(-MAX_TORQUE, min(MAX_TORQUE, rightTorque))

        time.sleep(0.05)

    clear_input_buffer()
    print("[Keyboard] Listener stopped.")
