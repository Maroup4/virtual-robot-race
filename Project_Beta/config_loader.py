# config.py
# Reads and applies configuration values from config.txt and Robot{N}/robot_config.txt

import os
import re
from pathlib import Path

CONFIG_PATH = "config.txt"

# Default values for global config (config.txt)
DEFAULT_CONFIG = {
    "HOST": "localhost",
    "PORT": 12346,
    "ACTIVE_ROBOTS": "1",   # Active robots (comma-separated)
    "DEBUG_MODE": 0,        # 0: Launch Unity automatically, 1: Launch manually (debug)
}

# Default values for robot-specific config (Robot{N}/robot_config.txt)
DEFAULT_ROBOT_CONFIG = {
    "MODE_NUM": 1,          # 1: keyboard, 2: table, 3: rule_based, 4: ai
    "ROBOT_ID": "R1",       # Robot ID (e.g., R1, R2, ...)
    "NAME": "Player0000",   # Player name (up to 10 alphanumeric chars)
    "RACE_FLAG": 1,         # 1: participate (POST), 0: watch only
    "DATA_SAVE": 1,         # 1: Save images, 0: Do not save
}

# Keys that should be parsed as integers
INT_KEYS = {
    "PORT",
    "MODE_NUM",
    "DEBUG_MODE",
    "DATA_SAVE",
    "RACE_FLAG",
}

# Global config storage
CONFIG = DEFAULT_CONFIG.copy()

# Robot configs storage (indexed by robot number: 1, 2, 3, 4, 5)
ROBOT_CONFIGS = {}

def _strip_inline_comment(value: str) -> str:
    """
    Remove inline comments starting with '#'.
    Example: 'Andy00   # comment' -> 'Andy00'
    """
    parts = value.split('#', 1)
    return parts[0].strip()

def _strip_quotes(value: str) -> str:
    """
    Remove single or double quotes around the value, if present.
    Example: '"Andy00"' -> 'Andy00'
    """
    if (len(value) >= 2) and ((value[0] == value[-1]) and value.startswith(("'", '"'))):
        return value[1:-1]
    return value

def _load_config_file(file_path: str, defaults: dict, target_dict: dict) -> None:
    """Generic config file loader with inline-comment support."""
    if not os.path.exists(file_path):
        print(f"[Config] {file_path} not found. Using default settings.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                value = _strip_inline_comment(value)
                value = _strip_quotes(value)

                if key in defaults:
                    if key in INT_KEYS:
                        try:
                            target_dict[key] = int(value)
                        except Exception:
                            print(f"[Config] WARNING: Invalid integer for {key}='{value}'. Using default.")
                            target_dict[key] = defaults[key]
                    else:
                        target_dict[key] = value
                # Unknown keys are ignored silently (forward compatibility)
    except Exception as e:
        print(f"[Config] Failed to read {file_path}: {e}")


def load_config():
    """Load key-value pairs from config.txt (global settings)."""
    _load_config_file(CONFIG_PATH, DEFAULT_CONFIG, CONFIG)


def load_robot_config(robot_num: int) -> dict:
    """
    Load robot-specific config from Robot{N}/robot_config.txt.
    Returns a dict with robot-specific settings.
    """
    robot_config = DEFAULT_ROBOT_CONFIG.copy()
    robot_dir = Path(f"Robot{robot_num}")
    config_file = robot_dir / "robot_config.txt"

    if config_file.exists():
        _load_config_file(str(config_file), DEFAULT_ROBOT_CONFIG, robot_config)
        print(f"[Config] Loaded config for Robot{robot_num} from {config_file}")
    else:
        print(f"[Config] Robot{robot_num} config not found at {config_file}. Using defaults.")

    # Override ROBOT_ID based on robot number if not explicitly set
    if robot_config.get("ROBOT_ID") == "R1":  # Default value
        robot_config["ROBOT_ID"] = f"R{robot_num}"

    # Auto-enable video creation when DATA_SAVE=1
    # Fixed parameters (advanced users can modify these constants in this file)
    if robot_config.get("DATA_SAVE", 1) == 1:
        robot_config["AUTO_MAKE_VIDEO"] = 1
        robot_config["VIDEO_FPS"] = 20
        robot_config["INFER_FPS"] = 1
    else:
        robot_config["AUTO_MAKE_VIDEO"] = 0
        robot_config["VIDEO_FPS"] = 20  # Set defaults even when not used
        robot_config["INFER_FPS"] = 1

    return robot_config

def validate_name(name: str) -> str:
    """
    Validate NAME:
      - Must be alphanumeric (A-Z, a-z, 0-9)
      - Must be 10 characters or fewer
      - If invalid, fallback to default and print a warning
    """
    if re.fullmatch(r"[A-Za-z0-9]{1,10}", name or ""):
        return name
    print(f"[Config] WARNING: Invalid NAME='{name}'. Must be <=10 alphanumeric characters. Using default.")
    return DEFAULT_CONFIG["NAME"]

def apply_config():
    """Apply loaded global config values as module-level variables."""
    global HOST, PORT, ACTIVE_ROBOTS, DEBUG_MODE

    load_config()

    HOST = CONFIG["HOST"]
    PORT = CONFIG["PORT"]
    DEBUG_MODE = CONFIG["DEBUG_MODE"]

    # Parse active robots list
    ACTIVE_ROBOTS = [int(r.strip()) for r in CONFIG.get("ACTIVE_ROBOTS", "1").split(",")]

    # Load all robot configs
    for robot_num in ACTIVE_ROBOTS:
        ROBOT_CONFIGS[robot_num] = load_robot_config(robot_num)


def get_robot_config(robot_num: int) -> dict:
    """
    Get config for a specific robot.
    If not loaded yet, load it on demand.
    """
    if robot_num not in ROBOT_CONFIGS:
        ROBOT_CONFIGS[robot_num] = load_robot_config(robot_num)
    return ROBOT_CONFIGS[robot_num]


def get_mode_string(mode_num: int) -> str:
    """Convert MODE_NUM to mode string."""
    MODE_MAP = {1: "keyboard", 2: "table", 3: "rule_based", 4: "ai"}
    return MODE_MAP.get(mode_num, "keyboard")


# Initialize global settings at import time
apply_config()
