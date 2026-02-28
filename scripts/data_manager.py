# data_manager.py
# Data manager for AAGP streaming protocol (driveTorque / steerAngle unified)
# - Saves per-tick images (JPG)
# - Appends frames_map.csv (tick, driveTorque, steerAngle, SOC, status)
# - On race end, writes metadata.csv from Unity's DataLogger output
# - No wheel_left/right columns anymore

import os
import csv
import json
import time
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Callable

import config_loader

# -------------------------
# Terminal log getter (set by main.py)
# -------------------------
_terminal_log_getter: Optional[Callable[[], str]] = None


def register_terminal_log_getter(getter_func: Callable[[], str]) -> None:
    """Register a function to get terminal log text."""
    global _terminal_log_getter
    _terminal_log_getter = getter_func
    print("[DataManager] Terminal log getter registered")

# -------------------------
# Base directory resolution
# -------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = Path(os.path.dirname(sys.executable))
else:
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# -------------------------
# Interactive artifacts (Robot-specific paths)
# -------------------------
def get_interactive_dir(robot_id: str) -> Path:
    """Get interactive directory for a specific robot (e.g., Robot1/data_interactive)"""
    robot_num = int(robot_id[1:])  # "R1" -> 1
    interactive_dir = BASE_DIR / f"Robot{robot_num}" / "data_interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    return interactive_dir


def get_soc_file(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "latest_SOC.txt"


def get_rgb_file_a(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "latest_RGB_a.jpg"


def get_rgb_file_b(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "latest_RGB_b.jpg"


def get_rgb_now_file(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "latest_RGB_now.txt"


def get_latest_frame_name_file(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "latest_frame_name.txt"


def get_last_run_dir_file(robot_id: str) -> Path:
    return get_interactive_dir(robot_id) / "last_run_dir.txt"


# Legacy support: Default to Robot1 if no robot_id specified
INTERACTIVE_DIR = BASE_DIR / "data_interactive"
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

SOC_FILE = INTERACTIVE_DIR / "latest_SOC.txt"
RGB_FILE_A = INTERACTIVE_DIR / "latest_RGB_a.jpg"
RGB_FILE_B = INTERACTIVE_DIR / "latest_RGB_b.jpg"
RGB_NOW_FILE = INTERACTIVE_DIR / "latest_RGB_now.txt"
LATEST_FRAME_NAME_FILE = INTERACTIVE_DIR / "latest_frame_name.txt"
LAST_RUN_DIR_FILE = INTERACTIVE_DIR / "last_run_dir.txt"

# Unity runtime log (source path may differ per build)
UNITY_LOG_SRC = BASE_DIR / "Windows" / "runtime_Log.txt"


def _safe_replace(src_tmp: Path, dst: Path, retries: int = 10, delay_sec: float = 0.02) -> None:
    for _ in range(retries):
        try:
            if dst.exists():
                dst.unlink(missing_ok=True)
            src_tmp.replace(dst)
            return
        except PermissionError:
            time.sleep(delay_sec)
        except Exception as e:
            print(f"[DataManager] Replace failed once for {dst}: {e}")
            time.sleep(delay_sec)
    print(f"[DataManager] Failed to replace after retries: {dst}")


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[DataManager] Failed to write {path.name}: {e}")


def read_last_run_dir(robot_id: str = "R1") -> Optional[Path]:
    """Read last run directory for a specific robot."""
    try:
        last_run_file = get_last_run_dir_file(robot_id)
        p = last_run_file.read_text(encoding="utf-8").strip()
        if not p:
            return None
        path = Path(p)
        return path if path.exists() else None
    except Exception:
        return None


def get_latest_soc(robot_id: str = "R1") -> Optional[float]:
    """Read the latest SOC value from the interactive data file for a specific robot."""
    try:
        soc_file = get_soc_file(robot_id)
        soc_str = soc_file.read_text(encoding="utf-8").strip()
        return float(soc_str)
    except Exception:
        return None


def get_latest_frame_name(robot_id: str = "R1") -> Optional[str]:
    """Read the latest frame filename from the interactive data file for a specific robot."""
    try:
        frame_file = get_latest_frame_name_file(robot_id)
        return frame_file.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def get_latest_rgb_path(robot_id: str = "R1") -> Optional[Path]:
    """Get the path to the latest RGB image for a specific robot."""
    try:
        rgb_now_file = get_rgb_now_file(robot_id)
        which = rgb_now_file.read_text(encoding="utf-8").strip()
        if which == "a":
            return get_rgb_file_a(robot_id)
        elif which == "b":
            return get_rgb_file_b(robot_id)
        else:
            # Fallback to _a
            return get_rgb_file_a(robot_id)
    except Exception:
        return get_rgb_file_a(robot_id)


class DataManager:
    def __init__(self, base_dir: Path, robot_id: str = "R1"):
        self.base_dir = Path(base_dir)
        self.robot_id = robot_id

        # Robot-specific paths
        robot_num = int(robot_id[1:])  # "R1" -> 1
        self.robot_dir = self.base_dir / f"Robot{robot_num}"
        self.training_data_root = self.robot_dir / "training_data"

        self.current_run_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None

        self._preview_toggle_a = True

        # Interactive file paths
        self.soc_file = get_soc_file(robot_id)
        self.rgb_file_a = get_rgb_file_a(robot_id)
        self.rgb_file_b = get_rgb_file_b(robot_id)
        self.rgb_now_file = get_rgb_now_file(robot_id)
        self.latest_frame_name_file = get_latest_frame_name_file(robot_id)
        self.last_run_dir_file = get_last_run_dir_file(robot_id)

    # -------------------------
    # Run/session management
    # -------------------------
    def start_new_run(self):
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        self.training_data_root.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = self.training_data_root / ts
        self.images_dir = self.current_run_dir / "images"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DataManager] [{self.robot_id}] New run created → {self.current_run_dir}")

        try:
            self.last_run_dir_file.write_text(str(self.current_run_dir), encoding="utf-8")
        except Exception as e:
            print(f"[DataManager] [{self.robot_id}] Failed to record last run: {e}")

        return self.current_run_dir, self.images_dir

    # -------------------------
    # Per-tick image handling
    # -------------------------
    def save_image_bytes(self, path: Path, data: bytes) -> None:
        if self.images_dir is None:
            raise RuntimeError("start_new_run() must be called before saving images.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"[DataManager] [{self.robot_id}] Failed to save image {path}: {e}")

        _write_text(self.latest_frame_name_file, path.name)
        try:
            target = self.rgb_file_a if self._preview_toggle_a else self.rgb_file_b
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            _safe_replace(tmp, target)
            _write_text(self.rgb_now_file, "a" if self._preview_toggle_a else "b")
            self._preview_toggle_a = not self._preview_toggle_a
        except Exception as e:
            print(f"[DataManager] [{self.robot_id}] Failed to update preview: {e}")


    # -------------------------
    # Final metadata
    # -------------------------
    def save_metadata_csv_from_unity_json(self, unity_json_obj: dict) -> None:
        """Convert Unity’s final DataLogger JSON into metadata.csv (no blanks)."""
        if self.current_run_dir is None:
            raise RuntimeError("start_new_run() must be called first.")

        rows = unity_json_obj
        if isinstance(rows, dict):
            if "payload" in rows:
                rows = rows["payload"]
            if isinstance(rows, dict) and "data" in rows:
                rows = rows["data"]

        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
                if isinstance(rows, dict) and "data" in rows:
                    rows = rows["data"]
            except Exception:
                pass

        meta_csv = self.current_run_dir / "metadata.csv"
        with open(meta_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            # === Header ===
            w.writerow([
                "tick", "session_time_ms", "race_time_ms", "filename", "soc",
                "drive_torque", "steer_angle",
                "drive_valid", "steer_valid",
                "status", "pos_x", "pos_y", "pos_z",
                "yaw_deg", "error_code",
                "collision_type", "collision_penalty"  # Beta 1.5
            ])

            # Helper to safely convert missing values
            def f_or_0(v):
                try:
                    if v is None or v == "" or str(v).lower() == "nan":
                        return 0.0
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0

            def i_or_0(v):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return 0

            # === Write data ===
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue

                    # Safe retrieval of numeric values
                    tick         = i_or_0(r.get("tick"))
                    session_ms   = i_or_0(r.get("session_time_ms"))
                    race_ms      = i_or_0(r.get("race_time_ms"))
                    soc          = f_or_0(r.get("soc"))
                    drive        = f_or_0(r.get("drive_torque") or r.get("driveTorque"))
                    steer        = f_or_0(r.get("steer_angle") or r.get("steerAngle"))
                    d_val        = i_or_0(r.get("drive_valid"))
                    s_val        = i_or_0(r.get("steer_valid"))
                    pos_x        = f_or_0(r.get("pos_x"))
                    pos_y        = f_or_0(r.get("pos_y"))
                    pos_z        = f_or_0(r.get("pos_z"))
                    yaw          = f_or_0(r.get("yaw_deg") or r.get("yaw"))
                    err          = i_or_0(r.get("error_code"))
                    # Beta 1.5: Collision data
                    coll_type    = r.get("collision_type") or ""
                    coll_penalty = f_or_0(r.get("collision_penalty"))

                    # Write row
                    w.writerow([
                        tick,
                        session_ms,
                        race_ms,
                        r.get("filename") or f"frame_{tick:06d}.jpg",
                        soc,
                        drive,
                        steer,
                        d_val,
                        s_val,
                        r.get("status") or "unknown",
                        pos_x,
                        pos_y,
                        pos_z,
                        yaw,
                        err,
                        coll_type,       # Beta 1.5
                        coll_penalty,    # Beta 1.5
                    ])
            else:
                print("[DataManager] WARNING: Metadata payload not list, header only.")

        print(f"[DataManager] metadata.csv written → {meta_csv}")
        self._copy_unity_log()
        self._maybe_delete_images_if_flagged()


    # -------------------------
    # Helpers
    # -------------------------
    def _copy_unity_log(self) -> None:
        if self.current_run_dir is None:
            return
        if UNITY_LOG_SRC.exists():
            dst = self.current_run_dir / "UnityLog.txt"
            try:
                shutil.copy(UNITY_LOG_SRC, dst)
                print(f"[DataManager] Copied Unity log to {dst}")
            except Exception as e:
                print(f"[DataManager] Failed to copy Unity log: {e}")

        # Copy table_input.csv if it exists (for table mode runs)
        table_src = self.base_dir / "table_input.csv"
        if table_src.exists():
            dst = self.current_run_dir / "table_input.csv"
            try:
                shutil.copy(table_src, dst)
            except Exception as e:
                print(f"[DataManager] Failed to copy table_input.csv: {e}")

    def _maybe_delete_images_if_flagged(self) -> None:
        if self.images_dir is None:
            return
        try:
            if getattr(config_loader, "DATA_SAVE", 1) == 0:
                print("[DataManager] DATA_SAVE=0 → deleting saved images...")
                for p in self.images_dir.glob("*.jpg"):
                    p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[DataManager] Cleanup failed: {e}")

    # -------------------------
    # Terminal log saving
    # -------------------------
    def save_terminal_log_from_main(self) -> None:
        """Save terminal output from main.py to terminal_log.txt in the run directory."""
        if self.current_run_dir is None:
            print("[DataManager] Cannot save terminal log: no run directory.")
            return

        if _terminal_log_getter is None:
            print(f"[DataManager] [{self.robot_id}] Terminal log getter not registered")
            return

        try:
            log_text = _terminal_log_getter()
            if not log_text:
                print(f"[DataManager] [{self.robot_id}] No terminal log to save")
                return

            log_path = self.current_run_dir / "terminal_log.txt"
            log_path.write_text(log_text, encoding="utf-8")
            print(f"[DataManager] [{self.robot_id}] Terminal log saved → {log_path}")
        except Exception as e:
            print(f"[DataManager] [{self.robot_id}] Failed to save terminal log: {e}")

    # -------------------------
    # Force end (q key) handling
    # -------------------------
    def save_force_end_metadata(self) -> None:
        """
        Save minimal metadata.csv and terminal log when force-ended with 'q' key.
        This ensures logs are preserved even when Unity doesn't send final metadata.
        """
        if self.current_run_dir is None:
            print(f"[DataManager] [{self.robot_id}] Cannot save force-end metadata: no run directory.")
            return

        print(f"[DataManager] [{self.robot_id}] Saving force-end metadata...")

        # Save minimal metadata.csv with "Force end" status
        meta_csv = self.current_run_dir / "metadata.csv"
        try:
            with open(meta_csv, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                # Write header
                w.writerow([
                    "tick", "session_time_ms", "race_time_ms", "filename", "soc",
                    "drive_torque", "steer_angle",
                    "drive_valid", "steer_valid",
                    "status", "pos_x", "pos_y", "pos_z",
                    "yaw_deg", "error_code",
                    "collision_type", "collision_penalty"  # Beta 1.5
                ])
                # Write single row indicating force end
                w.writerow([
                    0, 0, 0, "force_end.jpg", 0.0,
                    0.0, 0.0,
                    0, 0,
                    "Force end", 0.0, 0.0, 0.0,
                    0.0, 0,
                    "", 0.0  # Beta 1.5: no collision
                ])
            print(f"[DataManager] [{self.robot_id}] Force-end metadata.csv written → {meta_csv}")
        except Exception as e:
            print(f"[DataManager] [{self.robot_id}] Failed to write force-end metadata: {e}")

        # Save terminal log
        self.save_terminal_log_from_main()

        # Copy Unity log if available
        self._copy_unity_log()
