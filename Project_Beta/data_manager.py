# data_manager.py
# Data manager for AAGP streaming protocol (driveTorque / steerAngle unified)
# - Saves per-tick images (JPG)
# - Appends frames_map.csv (tick, driveTorque, steerAngle, SOC, status)
# - On race end, writes metadata.csv from Unity’s DataLogger output
# - No wheel_left/right columns anymore

import os
import csv
import json
import time
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import config

# -------------------------
# Base directory resolution
# -------------------------
if getattr(sys, "frozen", False):
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


def read_last_run_dir() -> Optional[Path]:
    try:
        p = LAST_RUN_DIR_FILE.read_text(encoding="utf-8").strip()
        if not p:
            return None
        path = Path(p)
        return path if path.exists() else None
    except Exception:
        return None


class DataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.training_data_root = self.base_dir / "training_data"

        self.current_run_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None

        self._frames_map_path: Optional[Path] = None
        self._frames_map_fp = None
        self._frames_map_writer = None

        self._preview_toggle_a = True

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

        # create frames_map.csv
        self._frames_map_path = self.current_run_dir / "frames_map.csv"
        self._frames_map_fp = open(self._frames_map_path, "w", newline="", encoding="utf-8")
        self._frames_map_writer = csv.writer(self._frames_map_fp)
        self._frames_map_writer.writerow(
            ["tick", "utc_ms", "filename", "soc", "status", "drive_torque", "steer_angle"]
        )
        print(f"[DataManager] New run created → {self.current_run_dir}")

        try:
            LAST_RUN_DIR_FILE.write_text(str(self.current_run_dir), encoding="utf-8")
        except Exception as e:
            print(f"[DataManager] Failed to record last run: {e}")

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
            print(f"[DataManager] Failed to save image {path}: {e}")

        _write_text(LATEST_FRAME_NAME_FILE, path.name)
        try:
            target = RGB_FILE_A if self._preview_toggle_a else RGB_FILE_B
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            _safe_replace(tmp, target)
            _write_text(RGB_NOW_FILE, "a" if self._preview_toggle_a else "b")
            self._preview_toggle_a = not self._preview_toggle_a
        except Exception as e:
            print(f"[DataManager] Failed to update preview: {e}")

    def append_frame_map(self, tick, utc_ms, filename, soc, status, drive_tq, steer_ang) -> None:
        """Append one telemetry row to frames_map.csv. Missing values are zero-filled."""
        if self._frames_map_writer is None:
            return

        # 0埋め（空欄がCSVに出ないように）
        def f_or_0(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        drive_tq = f_or_0(drive_tq)
        steer_ang = f_or_0(steer_ang)

        self._frames_map_writer.writerow([tick, utc_ms, filename, soc, status, drive_tq, steer_ang])

        # 最新SOCを記録（GUIなどの参照用）
        if soc is not None:
            try:
                _write_text(SOC_FILE, f"{float(soc):.4f}")
            except Exception:
                pass

    def flush_frame_map(self) -> None:
        if self._frames_map_fp:
            self._frames_map_fp.flush()

    def close_frame_map(self) -> None:
        if self._frames_map_fp:
            self._frames_map_fp.close()
            self._frames_map_fp = None
            self._frames_map_writer = None

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
            # === ヘッダ ===
            w.writerow([
                "tick", "time_ms", "filename", "soc",
                "drive_torque", "steer_angle",
                "drive_valid", "steer_valid",
                "status", "pos_x", "pos_y", "pos_z",
                "yaw_deg", "error_code"
            ])

            # 欠損を安全に変換するヘルパー
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

            # === データ書き込み ===
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue

                    # 数値系の安全取得
                    tick   = i_or_0(r.get("tick"))
                    t_ms   = i_or_0(r.get("time_ms"))
                    soc    = f_or_0(r.get("soc"))
                    drive  = f_or_0(r.get("drive_torque") or r.get("driveTorque"))
                    steer  = f_or_0(r.get("steer_angle") or r.get("steerAngle"))
                    d_val  = i_or_0(r.get("drive_valid"))
                    s_val  = i_or_0(r.get("steer_valid"))
                    pos_x  = f_or_0(r.get("pos_x"))
                    pos_y  = f_or_0(r.get("pos_y"))
                    pos_z  = f_or_0(r.get("pos_z"))
                    yaw    = f_or_0(r.get("yaw_deg") or r.get("yaw"))
                    err    = i_or_0(r.get("error_code"))

                    # 行出力
                    w.writerow([
                        tick,
                        t_ms,
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
                    ])
            else:
                print("[DataManager] WARNING: Metadata payload not list, header only.")

        print(f"[DataManager] metadata.csv written → {meta_csv}")
        self._copy_unity_log()
        self._maybe_delete_images_if_flagged()
        self.close_frame_map()


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

        if config.MODE == "table":
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
            if getattr(config, "JPEG_SAVE", 1) == 0:
                print("[DataManager] JPEG_SAVE=0 → deleting saved images...")
                for p in self.images_dir.glob("*.jpg"):
                    p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[DataManager] Cleanup failed: {e}")
