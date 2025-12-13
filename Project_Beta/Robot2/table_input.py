# table_input.py (Robot2 version - New Architecture)
# Reads drive/steer commands from CSV file and provides them for WebSocket sending
# Compatible with Unity Server + Python Client architecture

import pandas as pd
import os
import sys

# Module identification
MODULE_SOURCE = "Robot2"
print(f"[table_input] Loaded from {MODULE_SOURCE}/")

# CSV file path (relative to this file's location)
INPUT_CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_input.csv")

# DataFrame and state
df = None
csv_loaded = False
current_index = 0
robot_id = "R2"

# Latest command cache
driveTorque: float = 0.0
steerAngle: float = 0.0


def load_csv():
    """Load CSV file into DataFrame"""
    global df, csv_loaded

    if csv_loaded:
        return True

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"[TableInput] ERROR: CSV file not found: {INPUT_CSV_FILE}")
        return False

    try:
        df = pd.read_csv(INPUT_CSV_FILE)

        # Normalize column names to lowercase for flexible matching
        df.columns = df.columns.str.lower()

        # Validate required columns exist
        required_columns = ['drive_torque', 'steer_angle']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[TableInput] ERROR: Missing required columns: {missing_columns}")
            print(f"[TableInput] Available columns: {list(df.columns)}")
            print(f"[TableInput] Expected columns: {required_columns}")
            print(f"[TableInput] Hint: Copy data from metadata.csv including 'drive_torque' and 'steer_angle' columns")
            return False

        csv_loaded = True
        print(f"[TableInput] Loaded {len(df)} command rows from CSV")
        print(f"[TableInput] Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"[TableInput] ERROR loading CSV: {e}")
        return False


def advance_command():
    """
    Advance to next command in CSV.
    Returns True if successful, False if end of data.
    """
    global current_index, driveTorque, steerAngle

    if not csv_loaded:
        if not load_csv():
            return False

    if current_index >= len(df):
        print(f"[TableInput] End of CSV data reached (index={current_index})")
        return False

    try:
        row = df.iloc[current_index]
        # Use lowercase column names (normalized in load_csv)
        driveTorque = float(row.get("drive_torque", 0.0))
        steerAngle = float(row.get("steer_angle", 0.0))
        current_index += 1
        return True
    except KeyError as e:
        print(f"[TableInput] ERROR: Column not found in row {current_index}: {e}")
        print(f"[TableInput] Available columns: {list(df.columns)}")
        return False
    except ValueError as e:
        print(f"[TableInput] ERROR: Invalid value in row {current_index}: {e}")
        print(f"[TableInput] Row data: {row.to_dict()}")
        return False
    except Exception as e:
        print(f"[TableInput] ERROR reading row {current_index}: {type(e).__name__}: {e}")
        return False


def get_latest_command():
    """
    Return latest control command for WebSocket sender.
    Auto-advances to next row in CSV.
    Raises RuntimeError if CSV loading fails.
    """
    # Ensure CSV is loaded (will print error details if it fails)
    if not csv_loaded and not load_csv():
        raise RuntimeError("[TableInput] Failed to load CSV. Check the error messages above.")

    # Advance to next command
    if not advance_command():
        # End of data - return zero command (not an error)
        return {
            "type": "control",
            "robot_id": robot_id,
            "driveTorque": 0.0,
            "steerAngle": 0.0,
        }

    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }


def reset():
    """Reset to beginning of CSV"""
    global current_index
    current_index = 0
    print("[TableInput] Reset to beginning of CSV")


def get_progress():
    """Return current progress (index, total)"""
    total = len(df) if df is not None else 0
    return (current_index, total)
