# data_manager.py
# Data manager for the new streaming protocol:
# - Saves per-tick images (JPG) as they arrive
# - Appends a lightweight frames_map.csv per tick (optional but useful)
# - On race end, writes metadata.csv from Unity's final JSON (DataLogger output)
# - Maintains "interactive" artifacts (latest SOC, preview A/B JPG, latest frame name)
#
# Public API expected by websocket_server.py:
#   DataManager(base_dir: Path)
#   start_new_run() -> (run_dir: Path, images_dir: Path)
#   save_image_bytes(path: Path, data: bytes) -> None
#   append_frame_map(tick, utc_ms, filename, soc, status, left_tq, right_tq) -> None
#   flush_frame_map() -> None
#   close_frame_map() -> None
#   save_metadata_csv_from_unity_json(unity_json_obj: dict) -> None
#
# Notes:
# - This module does NOT parse legacy "header+JPEG in one packet" anymore.
# - It still updates interactive files:
#     data_interactive/latest_SOC.txt
#     data_interactive/latest_RGB_a.jpg / latest_RGB_b.jpg + latest_RGB_now.txt
#     data_interactive/latest_frame_name.txt
# - At the end of a run, UnityLog is copied (if present) and, if config.JPEG_SAVE == 0,
#   all saved JPGs are removed for lightweight mode.

import os
import csv
import json
import time
import shutil
import sys
from pathlib import Path
from typing import Optional

import config

from typing import Optional, Tuple
from pathlib import Path

# -------------------------
# Base directory resolution
# -------------------------
if getattr(sys, "frozen", False):
    # PyInstaller build
    BASE_DIR = Path(os.path.dirname(sys.executable))
else:
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Interactive artifacts
# -------------------------
INTERACTIVE_DIR = BASE_DIR / "data_interactive"
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

SOC_FILE = INTERACTIVE_DIR / "latest_SOC.txt"
RGB_FILE_A = INTERACTIVE_DIR / "latest_RGB_a.jpg"
RGB_FILE_B = INTERACTIVE_DIR / "latest_RGB_b.jpg"
RGB_NOW_FILE = INTERACTIVE_DIR / "latest_RGB_now.txt"      # contains "a" or "b"
LATEST_FRAME_NAME_FILE = INTERACTIVE_DIR / "latest_frame_name.txt"
LAST_RUN_DIR_FILE = INTERACTIVE_DIR / "last_run_dir.txt"

# Unity runtime log (source) and per-run copy target
UNITY_LOG_SRC = BASE_DIR / "Windows" / "runtime_Log.txt"

def _safe_replace(src_tmp: Path, dst: Path, retries: int = 10, delay_sec: float = 0.02) -> None:
    """Atomic-ish replace of a file with small retries (Windows-friendly)."""
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


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

# Backward-compat shim (prefer using a DataManager instance)
def get_current_run_dirs() -> Tuple[Optional[Path], Optional[Path]]:
    """Use DataManager.current_run_dir / images_dir from your instance instead.
    This shim returns (None, None) so old callers don't crash but can check."""
    return None, None

def read_last_run_dir() -> Path | None:
    """Return the last run directory recorded by DataManager, or None."""
    try:
        p = LAST_RUN_DIR_FILE.read_text(encoding="utf-8").strip()
        if not p:
            return None
        path = Path(p)
        return path if path.exists() else None
    except Exception:
        return None

def get_latest_soc() -> float | None:
    """
    Read the most recent SOC value from interactive file.
    Returns float (0.0–1.0) or None if not available.
    """
    try:
        if SOC_FILE.exists():
            txt = SOC_FILE.read_text(encoding="utf-8").strip()
            return float(txt)
    except Exception as e:
        print(f"[DataManager] Failed to read latest SOC: {e}")
    return None

class DataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.training_data_root = self.base_dir / "training_data"

        # Set during start_new_run()
        self.current_run_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None

        # frames_map.csv writer/handle
        self._frames_map_path: Optional[Path] = None
        self._frames_map_fp = None
        self._frames_map_writer = None

        # preview toggle (A/B double buffer)
        self._preview_toggle_a = True

    # -------------------------
    # Run/session management
    # -------------------------
    def start_new_run(self):
        """Create a new run directory and initialize frames_map.csv."""
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        self.training_data_root.mkdir(parents=True, exist_ok=True)

        self.current_run_dir = self.training_data_root / ts
        self.images_dir = self.current_run_dir / "images"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            LAST_RUN_DIR_FILE.write_text(str(self.current_run_dir), encoding="utf-8")
        except Exception as e:
            print(f"[DataManager] Failed to write LAST_RUN_DIR_FILE: {e}")

        

        return self.current_run_dir, self.images_dir

    # -------------------------
    # Per-tick artifacts
    # -------------------------
    def save_image_bytes(self, path: Path, data: bytes) -> None:
        """Save the raw JPG bytes to the given path under the current run/images,
        and update interactive preview + latest frame name."""
        if self.images_dir is None or self.current_run_dir is None:
            raise RuntimeError("start_new_run() must be called before saving images.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the training image
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"[DataManager] Failed to write training image {path.name}: {e}")

        # Update interactive artifacts
        # 1) latest frame name
        _write_text(LATEST_FRAME_NAME_FILE, path.name)

        # 2) A/B preview file (so GUI tools can read the most recent frame atomically)
        try:
            target = RGB_FILE_A if self._preview_toggle_a else RGB_FILE_B
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            _safe_replace(tmp, target)
            _write_text(RGB_NOW_FILE, "a" if self._preview_toggle_a else "b")
            self._preview_toggle_a = not self._preview_toggle_a
        except Exception as e:
            print(f"[DataManager] Failed to update interactive preview: {e}")

    def append_frame_map(self, tick, utc_ms, filename, soc, status, left_tq, right_tq) -> None:
        """Append one row to frames_map.csv."""
        if self._frames_map_writer is None:
            return
        self._frames_map_writer.writerow([tick, utc_ms, filename, soc, status, left_tq, right_tq])

        # Also keep latest SOC for interactive tools
        try:
            if soc is not None:
                _write_text(SOC_FILE, f"{float(soc):.4f}")
        except Exception:
            pass

    def flush_frame_map(self) -> None:
        """Flush frames_map.csv to disk."""
        if self._frames_map_fp:
            self._frames_map_fp.flush()

    def close_frame_map(self) -> None:
        """Close frames_map.csv handle."""
        if self._frames_map_fp:
            self._frames_map_fp.close()
            self._frames_map_fp = None
            self._frames_map_writer = None

    # -------------------------
    # Final metadata
    # -------------------------
    def save_metadata_csv_from_unity_json(self, unity_json_obj: dict) -> None:
        """Write metadata.csv from Unity's final DataLogger JSON payload.
        Accepts:
          - list[...] (final rows)
          - dict with 'data': list[...]
          - dict with 'payload': list[...] or {'data': list[...]}
        """
        if self.current_run_dir is None:
            raise RuntimeError("start_new_run() must be called before saving metadata.")

        # --- payload / data 
        rows = unity_json_obj
        if isinstance(rows, dict):
            # If wrapped as {"payload": ...}
            if "payload" in rows:
                rows = rows["payload"]
            # If wrapped as {"data": ...}
            if isinstance(rows, dict) and "data" in rows:
                rows = rows["data"]

        # If still a JSON string, try to parse
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
            w.writerow([
                "id", "time_ms", "frame_id", "filename", "soc",
                "wheel_left", "wheel_right", "status",
                "pos_z", "pos_x", "pos_y", "yaw", "error_code"
            ])

            if isinstance(rows, list):
                for r in rows:
                    
                    w.writerow([
                        (r.get("id") if isinstance(r, dict) else None),
                        (r.get("time_ms") if isinstance(r, dict) else None),
                        (r.get("frame_id") if isinstance(r, dict) else None),
                        (r.get("filename") if isinstance(r, dict) else None),
                        (r.get("soc") if isinstance(r, dict) else None),
                        (r.get("wheel_left") if isinstance(r, dict) else None),
                        (r.get("wheel_right") if isinstance(r, dict) else None),
                        (r.get("status") if isinstance(r, dict) else None),
                        (r.get("pos_z") if isinstance(r, dict) else None),
                        (r.get("pos_x") if isinstance(r, dict) else None),
                        (r.get("pos_y") if isinstance(r, dict) else None),
                        (r.get("yaw") if isinstance(r, dict) else None),
                        (r.get("error_code") if isinstance(r, dict) else None),
                    ])
            else:
                print("[DataManager] WARNING: Final metadata payload wasn't a list → wrote header only.")

        print(f"[DataManager] Metadata saved to {meta_csv}")
        self._copy_unity_log()
       
        # Deletion if AUTO_MAKE_VIDEO is ON (main.py will delete after export)
        if not getattr(config, "AUTO_MAKE_VIDEO", True):
            self._maybe_delete_images_if_flagged()
        self.close_frame_map()
    
    
    # -------------------------
    # Helpers
    # -------------------------
    def _copy_unity_log(self) -> None:
        """Copy Unity runtime log to the current run directory (if present)."""
        if self.current_run_dir is None:
            return
        if UNITY_LOG_SRC.exists():
            dst = self.current_run_dir / "UnityLog.txt"
            try:
                shutil.copy(UNITY_LOG_SRC, dst)
                print(f"[DataManager] Copied Unity log to {dst}")
            except Exception as e:
                print(f"[DataManager] Failed to copy Unity log: {e}")

        # Copy table_input.csv if we're in 'table' mode
        if config.MODE == "table":
            table_src = self.base_dir / "table_input.csv"
            if table_src.exists():
                dst = self.current_run_dir / "table_input.csv"
                try:
                    shutil.copy(table_src, dst)
                    print(f"[DataManager] Copied table_input.csv to {dst}")
                except Exception as e:
                    print(f"[DataManager] Failed to copy table_input.csv: {e}")

    def _maybe_delete_images_if_flagged(self) -> None:
        """Delete all saved JPGs if config.JPEG_SAVE == 0 (lightweight mode)."""
        if self.images_dir is None:
            return
        try:
            if getattr(config, "JPEG_SAVE", 1) == 0:
                print("[DataManager] JPEG_SAVE=0 → Deleting all saved JPEGs for lightweight mode")
                for p in self.images_dir.glob("*.jpg"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[DataManager] Failed to delete {p.name}: {e}")
                print("[DataManager] All JPEG images deleted")
        except Exception as e:
            print(f"[DataManager] Failed during JPEG cleanup: {e}")
    
    
