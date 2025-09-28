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

def _put_text(img: np.ndarray, text: str, xy: Tuple[int, int], *, scale: float = 0.5, thick: int = 1) -> None:
    """Readable white text with black outline (slightly small by default)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = xy
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def _resolve_outpath(out_dir: str, *, frame_name: Optional[str] = None, src_path: Optional[str] = None) -> str:
    """
    Decide the output filename.
    - If frame_name is provided, use "debug_{basename(frame_name)}".
    - Otherwise, use timestamp + (a/b fallback inferred from src_path).
    """
    ensure_dir(out_dir)
    if frame_name:
        base = os.path.basename(frame_name)  # avoid accidental subdirs
        out_name = f"debug_{base}"
    else:
        tag = "a" if (src_path and "latest_RGB_a" in src_path) else "b"
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_name = f"{ts}_latest_RGB_{tag}.jpg"
    return os.path.join(out_dir, out_name)

def _draw_torque_vector(img: np.ndarray, left_tq: float, right_tq: float, *, radius: int = 34, pad: int = 16) -> None:
    """
    Draw a “combined torque vector” in the top-right (no arrow head).
    Angle = atan2(L - R, L + R)  (0° = up, positive is right)
    Length = clamp(hypot(L+R, L-R)/2, 0..1)
    """
    h, w = img.shape[:2]
    cx = w - pad - radius
    cy = pad + radius

    # Guide circle + cross (light gray)
    cv2.circle(img, (cx, cy), radius, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (210, 210, 210), 1, cv2.LINE_AA)

    L, R = float(left_tq), float(right_tq)
    sum_lr  = L + R
    diff_lr = L - R
    angle   = math.atan2(diff_lr, sum_lr)                  # 0=up, + is right
    length  = min(1.0, math.hypot(sum_lr, diff_lr) / 2.0)  # 0..1

    r_pix = int(radius * length)
    dx = int(r_pix * math.sin(angle))
    dy = int(-r_pix * math.cos(angle))  # up is negative in image coords

    # Centered line (black outline → white)
    cv2.line(img, (cx, cy), (cx + dx, cy + dy), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx + dx, cy + dy), (255, 255, 255), 2, cv2.LINE_AA)

# ------------- 1) Minimal HUD on SW canvas + torque vector (preferred) -------------
def annotate_and_save_canvas(
    canvas_bgr: Optional[np.ndarray],
    *,
    out_dir: str = "debug",
    lateral_px: Optional[float] = None,    # accepted but not drawn (SW HUD handles it)
    theta_deg: Optional[float] = None,     # accepted but not drawn
    torque_left: float = 0.0,
    torque_right: float = 0.0,
    mode: str = "normal",
    frame_name: Optional[str] = None,
    src_path: Optional[str] = None,
    jpeg_quality: int = 85,
    origin: Tuple[int, int] = (10, 22),    # top-left start
    line: int = 22,                        # line spacing
) -> Optional[str]:
    """
    Overlay a minimal HUD (mode + torques) on the already-rendered SW canvas and save it.
    Returns the output path, or None if canvas_bgr is None.
    """
    if canvas_bgr is None:
        return None

    img = canvas_bgr.copy()

    # Top-left HUD: Mode / left & right torques (no black panel)
    x, y = origin
    _put_text(img, f"Mode: {mode}",                 (x, y)); y += line
    _put_text(img, f"LeftTq : {torque_left:+.3f}",  (x, y)); y += line
    _put_text(img, f"RightTq: {torque_right:+.3f}", (x, y))

    # Top-right combined vector (kept compact and padded not to overlap ROI)
    _draw_torque_vector(img, torque_left, torque_right, radius=34, pad=16)

    out_path = _resolve_outpath(out_dir, frame_name=frame_name, src_path=src_path)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    print(f"[DebugUtils] Saved {out_path}")
    return out_path

# ----------- 2) Fallback: full HUD on raw frame + torque vector -----------
def overlay_and_save(
    pil_img,
    sw_result,
    driver_debug: Optional[dict],
    out_dir: str = "debug",
    frame_name: Optional[str] = None,
    src_path: Optional[str] = None,
    jpeg_quality: int = 85,
    origin: Tuple[int, int] = (10, 22),
    line: int = 22,
) -> str:
    """
    When no SW canvas is available, draw a full HUD on the raw frame and save:
      - Mode, Lateral, Theta, Left/Right torques
      - Combined torque vector (if torques available)
    Returns the output path.
    """
    ensure_dir(out_dir)
    bgr = pil_to_bgr(pil_img).copy()

    lateral = getattr(sw_result, "lateral_px", None) if sw_result is not None else None
    theta   = getattr(sw_result, "theta_deg",  None) if sw_result is not None else None

    lt   = (driver_debug or {}).get("left_torque")
    rt   = (driver_debug or {}).get("right_torque")
    mode = (driver_debug or {}).get("lane_mode", "unknown")

    x, y = origin
    _put_text(bgr, f"Mode: {mode}", (x, y)); y += line
    _put_text(bgr, f"Lateral : {'None' if lateral is None else f'{lateral:+.1f} px'}", (x, y)); y += line
    _put_text(bgr, f"Theta   : {'None' if theta   is None else f'{theta:+.1f} deg'}", (x, y)); y += line
    _put_text(bgr, f"LeftTq  : {'None' if lt is None else f'{lt:+.3f}'}",  (x, y)); y += line
    _put_text(bgr, f"RightTq : {'None' if rt is None else f'{rt:+.3f}'}", (x, y))

    if lt is not None and rt is not None:
        _draw_torque_vector(bgr, float(lt), float(rt), radius=34, pad=16)

    out_path = _resolve_outpath(out_dir, frame_name=frame_name, src_path=src_path)
    cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    print(f"[DebugUtils] Saved {out_path}")
    return out_path
