# rule_based_algorithms/debug_utils.py
import os
import cv2
import numpy as np
from datetime import datetime
import math
from typing import Optional, Tuple

# ---------------- helpers ----------------
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (no-op if it does)."""
    os.makedirs(path, exist_ok=True)

def pil_to_bgr(pil_img):
    """Convert PIL.Image to OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _put_text(img: np.ndarray, text: str, xy: Tuple[int, int], *, scale: float = 0.28, thick: int = 1) -> None:
    """Readable white text with black outline (matching sliding_windows.py style)."""
    font = cv2.FONT_HERSHEY_DUPLEX  # Unified with sliding_windows.py
    x, y = xy
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def _resolve_outpath(out_dir: str, *, frame_name: Optional[str] = None, src_path: Optional[str] = None) -> str:
    """
    Decide the output filename.
    - If frame_name is provided, use "debug_{basename(frame_name)}".
    - Otherwise, use timestamp + (a/b fallback inferred from src_path).
    - Output is placed in out_dir/output/ subdirectory.
    """
    # Use output/ subdirectory
    actual_out_dir = os.path.join(out_dir, "output")
    ensure_dir(actual_out_dir)
    if frame_name:
        base = os.path.basename(frame_name)  # avoid accidental subdirs
        out_name = f"debug_{base}"
    else:
        tag = "a" if (src_path and "latest_RGB_a" in src_path) else "b"
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_name = f"{ts}_latest_RGB_{tag}.jpg"
    return os.path.join(actual_out_dir, out_name)

def _draw_steer_vector(img: np.ndarray, drive_tq: float, steer_ang: float, *, radius: int = 34, pad: int = 16) -> None:
    """
    Draw a "steer control vector" in the top-right.
    - Angle = steer_ang (rad, + is right turn)
    - Length = abs(drive_tq) (0..1)
    - Color: green for forward, red for reverse
    """
    h, w = img.shape[:2]
    cx = w - pad - radius
    cy = pad + radius

    # Guide circle + cross (light gray)
    cv2.circle(img, (cx, cy), radius, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (210, 210, 210), 1, cv2.LINE_AA)

    drive = float(drive_tq)
    steer = float(steer_ang)
    length = min(1.0, abs(drive))  # 0..1

    r_pix = int(radius * length)
    dx = int(r_pix * math.sin(steer))
    dy = int(-r_pix * math.cos(steer))  # up is negative in image coords

    # Color: green for forward, red for reverse
    color = (0, 255, 0) if drive >= 0 else (0, 0, 255)  # BGR

    # Centered line (black outline → colored)
    cv2.line(img, (cx, cy), (cx + dx, cy + dy), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx + dx, cy + dy), color, 2, cv2.LINE_AA)

# ------------- 1) Minimal HUD on SW canvas + steer vector (preferred) -------------
def annotate_and_save_canvas(
    canvas_bgr: Optional[np.ndarray],
    *,
    out_dir: str = "debug",
    lateral_px: Optional[float] = None,    # accepted but not drawn (SW HUD handles it)
    theta_deg: Optional[float] = None,     # accepted but not drawn
    drive_torque: float = 0.0,
    steer_angle: float = 0.0,
    mode: str = "normal",
    frame_name: Optional[str] = None,
    src_path: Optional[str] = None,
    jpeg_quality: int = 85,
    origin: Tuple[int, int] = (6, 12),     # top-left start (reduced for smaller font)
    line: int = 12,                        # line spacing (reduced for smaller font)
) -> Optional[str]:
    """
    Overlay a minimal HUD (mode + drive/steer) on the already-rendered SW canvas and save it.
    Returns the output path, or None if canvas_bgr is None.
    """
    if canvas_bgr is None:
        return None

    img = canvas_bgr.copy()

    # Top-left HUD: Frame name / Mode / drive & steer (no black panel)
    steer_deg = math.degrees(steer_angle)
    x, y = origin

    # Frame name (e.g., "frame_000123.jpg")
    if frame_name:
        display_name = os.path.basename(frame_name)
        _put_text(img, display_name, (x, y)); y += line

    _put_text(img, f"Mode: {mode}",                           (x, y)); y += line
    _put_text(img, f"Drive : {drive_torque:+.3f}",            (x, y)); y += line
    _put_text(img, f"Steer : {steer_deg:+.1f} deg", (x, y))

    # Top-right steer vector (reduced size for smaller HUD)
    _draw_steer_vector(img, drive_torque, steer_angle, radius=24, pad=10)

    out_path = _resolve_outpath(out_dir, frame_name=frame_name, src_path=src_path)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    print(f"[DebugUtils] Saved {out_path}")
    return out_path

# ----------- 2) Fallback: full HUD on raw frame + steer vector -----------
def overlay_and_save(
    pil_img,
    sw_result,
    driver_debug: Optional[dict],
    out_dir: str = "debug",
    frame_name: Optional[str] = None,
    src_path: Optional[str] = None,
    jpeg_quality: int = 85,
    origin: Tuple[int, int] = (6, 12),    # reduced for smaller font
    line: int = 12,                        # reduced for smaller font
) -> str:
    """
    When no SW canvas is available, draw a full HUD on the raw frame and save:
      - Mode, Lateral, Theta, Drive/Steer
      - Steer control vector (if drive/steer available)
    Returns the output path.
    """
    ensure_dir(out_dir)
    bgr = pil_to_bgr(pil_img).copy()

    lateral = getattr(sw_result, "lateral_px", None) if sw_result is not None else None
    theta   = getattr(sw_result, "theta_deg",  None) if sw_result is not None else None

    drive = (driver_debug or {}).get("drive_torque")
    steer = (driver_debug or {}).get("steer_angle")
    mode = (driver_debug or {}).get("lane_mode", "unknown")

    x, y = origin

    # Frame name (e.g., "frame_000123.jpg")
    if frame_name:
        display_name = os.path.basename(frame_name)
        _put_text(bgr, display_name, (x, y)); y += line

    _put_text(bgr, f"Mode: {mode}", (x, y)); y += line
    _put_text(bgr, f"Lateral : {'None' if lateral is None else f'{lateral:+.1f} px'}", (x, y)); y += line
    _put_text(bgr, f"Theta   : {'None' if theta   is None else f'{theta:+.1f} deg'}", (x, y)); y += line
    _put_text(bgr, f"Drive   : {'None' if drive is None else f'{drive:+.3f}'}",  (x, y)); y += line
    if steer is not None:
        steer_deg = math.degrees(steer)
        _put_text(bgr, f"Steer   : {steer_deg:+.1f} deg", (x, y))
    else:
        _put_text(bgr, f"Steer   : None", (x, y))

    if drive is not None and steer is not None:
        _draw_steer_vector(bgr, float(drive), float(steer), radius=24, pad=10)

    out_path = _resolve_outpath(out_dir, frame_name=frame_name, src_path=src_path)
    cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    print(f"[DebugUtils] Saved {out_path}")
    return out_path
