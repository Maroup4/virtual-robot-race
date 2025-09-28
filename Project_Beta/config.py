# config.py
# Reads and applies configuration values from config.txt

import os
import re

CONFIG_PATH = "config.txt"

# Default values if config.txt is missing or invalid
DEFAULT_CONFIG = {
    "HOST": "localhost",
    "PORT": 12346,
    "MODE_NUM": 1,          # 1: keyboard, 2: table, 3: rule_based, 4: ai
    "DEBUG_MODE": 0,        # 0: Launch Unity automatically, 1: Launch manually (debug)
    "JPEG_SAVE": 0,         # 1: Save images, 0: Do not save
    "NAME": "Player0000",   # Player name (up to 10 alphanumeric chars)
    "RACE_FLAG": 1,         # 1: participate (POST), 0: watch only

    # --- post race video options ---
    # Use 0/1 in config.txt; converted to bools in apply_config()
    "AUTO_MAKE_VIDEO": 1,         # 1=auto make mp4 after run
    "OPEN_EXPLORER_ON_VIDEO": 1,  # 1=open Explorer after mp4 made (Windows)
    "VIDEO_FPS": 20,              # default video fps
    "INFER_FPS": 1,               # 1=estimate fps from timestamps
}

# Keys that should be parsed as integers
INT_KEYS = {
    "PORT",
    "MODE_NUM",
    "DEBUG_MODE",
    "JPEG_SAVE",
    "RACE_FLAG",
    "AUTO_MAKE_VIDEO",
    "OPEN_EXPLORER_ON_VIDEO",
    "VIDEO_FPS",
    "INFER_FPS",
}

CONFIG = DEFAULT_CONFIG.copy()

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

def load_config():
    """Load key-value pairs from config.txt with inline-comment support."""
    if not os.path.exists(CONFIG_PATH):
        print("[Config] config.txt not found. Using default settings.")
        return

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
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

                if key in DEFAULT_CONFIG:
                    if key in INT_KEYS:
                        try:
                            CONFIG[key] = int(value)
                        except Exception:
                            print(f"[Config] WARNING: Invalid integer for {key}='{value}'. Using default.")
                            CONFIG[key] = DEFAULT_CONFIG[key]
                    else:
                        CONFIG[key] = value
                # Unknown keys are ignored silently (forward compatibility)
    except Exception as e:
        print(f"[Config] Failed to read config.txt: {e}")

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
    """Apply loaded values as module-level variables."""
    global HOST, PORT, MODE_NUM, MODE, DEBUG_MODE, JPEG_SAVE, NAME, RACE_FLAG
    global AUTO_MAKE_VIDEO, OPEN_EXPLORER_ON_VIDEO, VIDEO_FPS, INFER_FPS

    load_config()

    HOST = CONFIG["HOST"]
    PORT = CONFIG["PORT"]
    MODE_NUM = CONFIG["MODE_NUM"]

    MODE_MAP = {1: "keyboard", 2: "table", 3: "rule_based", 4: "ai"}
    MODE = MODE_MAP.get(MODE_NUM, "keyboard")

    DEBUG_MODE = CONFIG["DEBUG_MODE"]
    JPEG_SAVE = CONFIG["JPEG_SAVE"]

    NAME = validate_name(CONFIG.get("NAME", DEFAULT_CONFIG["NAME"]))
    RACE_FLAG = CONFIG["RACE_FLAG"]

    # --- video options ---
    # Convert 0/1 ints to bools for convenience in code
    AUTO_MAKE_VIDEO = bool(CONFIG.get("AUTO_MAKE_VIDEO", DEFAULT_CONFIG["AUTO_MAKE_VIDEO"]))
    OPEN_EXPLORER_ON_VIDEO = bool(CONFIG.get("OPEN_EXPLORER_ON_VIDEO", DEFAULT_CONFIG["OPEN_EXPLORER_ON_VIDEO"]))
    VIDEO_FPS = int(CONFIG.get("VIDEO_FPS", DEFAULT_CONFIG["VIDEO_FPS"]))
    INFER_FPS = bool(CONFIG.get("INFER_FPS", DEFAULT_CONFIG["INFER_FPS"]))

# Initialize settings at import time
apply_config()
