# sliding_windows.py
# Standalone: Histogram-based Sliding Windows for WHITE lane (education-ready)
# Usage:
#   cd project/rule_based_algorithms
#   python sliding_windows.py --folder ../rulebasesample --save_debug
#
# Outputs full-size debug images to ../debug/ (ROI box, windows, midline, polynomial fits, HUD text)

from dataclasses import dataclass
from PIL import Image
import numpy as np
import cv2, os, glob, csv, argparse
from typing import Optional

# ===== Parameters (adjust if needed) =====
ROI_TOP_FRAC  = 0.40       # ROI top (fraction of image height)
ROI_BOT_FRAC  = 0.70       # ROI bottom
NWINDOWS      = 9          # number of vertical windows
MARGIN        = 60         # half-width of each window [px]
MINPIX        = 50         # re-centering threshold (min points in window)
KERNEL        = 3          # morphology kernel size

# Window colors (BGR)
WIN_COLOR_LEFT  = (255, 255, 0)   # bright cyan
WIN_COLOR_RIGHT = (160, 160, 0)   # dark cyan
WIN_THICKNESS   = 2

# project_root = one folder up from "rule_based_algorithms"
project_root = os.path.dirname(os.path.dirname(__file__))
DEBUG_DIR = os.path.join(project_root, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

@dataclass
class SWResult:
    ok: bool
    left_pts: Optional[np.ndarray]
    right_pts: Optional[np.ndarray]
    left_fit: Optional[np.ndarray]   # [a,b,c] for x = a*y^2 + b*y + c
    right_fit: Optional[np.ndarray]
    lateral_px: Optional[float] = None
    theta_deg: Optional[float] = None
    img_width: Optional[int] = None
    canvas_bgr: Optional[np.ndarray] = None   # drawn canvas (BGR)

# ---------- HUD text ----------
def _put_text_bottom(img: np.ndarray, lines, pad=12, max_scale=1.0, min_scale=0.35, thick=1):
    """Draw multiple lines of text bottom-left with auto scaling."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max_scale
    gap_ratio = 0.6
    max_w_limit = w - 2*pad
    max_h_limit = int(h * 0.28)

    def block_metrics(sc):
        sizes = [cv2.getTextSize(t, font, sc, thick)[0] for t in lines]
        max_w = max([s[0] for s in sizes]) if sizes else 0
        hs    = [s[1] for s in sizes]
        gap   = int(max(1, (hs[0] if hs else 0) * gap_ratio))
        total_h = (sum(hs) + (len(lines)-1)*gap) if hs else 0
        return max_w, hs, gap, total_h

    max_w, hs, gap, total_h = block_metrics(scale)
    while (max_w > max_w_limit or total_h > max_h_limit) and scale > min_scale:
        scale -= 0.05
        max_w, hs, gap, total_h = block_metrics(scale)

    y = h - pad - (total_h - (gap if len(lines) > 1 else 0))
    x = pad
    for t, hline in zip(lines, hs):
        org = (x, y + hline)
        cv2.putText(img, t, org, font, scale, (0,0,0), thick+1, cv2.LINE_AA)    # black outline
        cv2.putText(img, t, org, font, scale, (255,255,255), thick, cv2.LINE_AA) # white text
        y += hline + gap

# ---------- WHITE lane binary mask (HSV) ----------
def white_binary(hsv: np.ndarray) -> np.ndarray:
    """
    White lane is assumed to be low saturation / high value.
    Extract by thresholding low S and high V in HSV.
    """
    S_MAX = 60
    V_MIN = 190
    lo = np.array([0,   0,   V_MIN], dtype=np.uint8)
    hi = np.array([180, S_MAX, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    k = np.ones((KERNEL, KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

# ---------- Core: Sliding Windows ----------
def sliding_windows_white(pil_img: Image.Image, save_debug=True, src_path=None, return_canvas=False) -> SWResult:
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    y_top = int(H * ROI_TOP_FRAC)
    y_bot = int(H * ROI_BOT_FRAC)
    roi_bgr = bgr[y_top:y_bot, :]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Debug canvas
    dbg = bgr.copy()
    cv2.rectangle(dbg, (0, y_top), (W, y_bot), (0, 0, 255), 2)   # ROI box
    cx = W // 2
    cv2.line(dbg, (cx, y_top), (cx, y_bot), (0, 255, 0), 2)      # image center line

    # Binary
    binary = white_binary(hsv)
    if np.count_nonzero(binary) < 300:
        # Save (if requested)
        if save_debug:
            _save_debug_image(dbg, pil_img, src_path)
        return SWResult(False, None, None, None, None, lateral_px=None, theta_deg=None, img_width=W,
                        canvas_bgr=dbg if return_canvas else None)

    # Histogram-based initial positions
    half = binary[binary.shape[0] // 2 :, :]
    hist = np.sum(half, axis=0)
    midx = W // 2
    leftx_base  = np.argmax(hist[:midx]) if np.any(hist[:midx]) else midx // 2
    rightx_base = (np.argmax(hist[midx:]) + midx) if np.any(hist[midx:]) else (midx + (W - midx) // 2)

    win_h = binary.shape[0] // NWINDOWS
    nonzero_y, nonzero_x = binary.nonzero()
    leftx_current, rightx_current = int(leftx_base), int(rightx_base)

    left_inds, right_inds = [], []

    # Sliding windows
    for win in range(NWINDOWS):
        win_y_low  = y_bot - (win + 1) * win_h
        win_y_high = y_bot - win * win_h
        ry_low  = win_y_low  - y_top
        ry_high = win_y_high - y_top

        lx_low, lx_high = leftx_current - MARGIN, leftx_current + MARGIN
        rx_low, rx_high = rightx_current - MARGIN, rightx_current + MARGIN

        cv2.rectangle(dbg, (lx_low, win_y_low), (lx_high, win_y_high), WIN_COLOR_LEFT, WIN_THICKNESS)
        cv2.rectangle(dbg, (rx_low, win_y_low), (rx_high, win_y_high), WIN_COLOR_RIGHT, WIN_THICKNESS)

        good_left = ((nonzero_y >= ry_low) & (nonzero_y < ry_high) &
                     (nonzero_x >= lx_low) & (nonzero_x < lx_high)).nonzero()[0]
        good_right = ((nonzero_y >= ry_low) & (nonzero_y < ry_high) &
                      (nonzero_x >= rx_low) & (nonzero_x < rx_high)).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > MINPIX:
            leftx_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > MINPIX:
            rightx_current = int(np.mean(nonzero_x[good_right]))

    left_inds  = np.concatenate(left_inds)  if len(left_inds)  else np.array([], dtype=int)
    right_inds = np.concatenate(right_inds) if len(right_inds) else np.array([], dtype=int)

    leftx  = nonzero_x[left_inds]
    leftyR = nonzero_y[left_inds]
    rightx = nonzero_x[right_inds]
    rightyR= nonzero_y[right_inds]
    lefty  = leftyR + y_top
    righty = rightyR + y_top

    # Fit polynomials
    ok = False
    left_fit = right_fit = None
    if leftx.size > 200 and rightx.size > 200:
        left_fit  = np.polyfit(lefty.astype(np.float32),  leftx.astype(np.float32),  2)
        right_fit = np.polyfit(righty.astype(np.float32), rightx.astype(np.float32), 2)

        # Sample for drawing
        ploty = np.linspace(y_top, y_bot - 1, 100).astype(np.float32)
        left_fitx  = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]).astype(int)
        right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]).astype(int)

        for x, y in zip(left_fitx, ploty.astype(int)):
            cv2.circle(dbg, (x, y), 2, (0, 200, 255), -1)
        for x, y in zip(right_fitx, ploty.astype(int)):
            cv2.circle(dbg, (x, y), 2, (0, 200, 255), -1)

        # Plot midline (yellow)
        center_fitx = ((left_fitx + right_fitx) / 2).astype(int)
        for x, y in zip(center_fitx, ploty.astype(int)):
            cv2.circle(dbg, (x, y), 2, (0, 255, 255), -1)

        ok = True

    # ----- Lateral / Theta: compute & visualize (0° = up, right positive) -----
    if ok and (left_fit is not None) and (right_fit is not None):
        y_center = (y_top + y_bot) // 2

        def x_centerline(y):
            xl = np.polyval(left_fit,  y)
            xr = np.polyval(right_fit, y)
            return 0.5 * (xl + xr)

        # Lateral (difference from image center)
        x_center_lane = float(x_centerline(y_center))
        lateral = x_center_lane - (W / 2)

        # Heading angle (upward finite diff)
        h = 1.0
        xc_up = float(x_centerline(y_center - h))
        xc_dn = float(x_centerline(y_center + h))
        dxdy_up = (xc_up - xc_dn) / (2.0 * h)   # d x / d (−y)
        theta_rad = np.arctan(dxdy_up)
        theta_deg = float(np.degrees(theta_rad))

        # Visualization
        title = "Sliding Windows (WHITE)"
        _put_text_bottom(dbg, [title, f"Lateral = {lateral:+.1f} px", f"Theta  = {theta_deg:+.1f} deg"])
        pt_center = (W // 2, int(y_center))
        pt_xc     = (int(x_center_lane), int(y_center))
        cv2.line(dbg, pt_center, pt_xc, (0, 0, 255), 2)  # red = lateral

        arrow_len = 20
        dx_vis = int(arrow_len * np.sin(theta_rad))   # right positive
        dy_vis = int(arrow_len * np.cos(theta_rad))   # up positive → subtract on image
        pt_theta_end = (pt_xc[0] + dx_vis, pt_xc[1] - dy_vis)
        cv2.line(dbg, pt_xc, pt_theta_end, (255, 0, 0), 2)  # blue = heading

        # Prepare result
        left_pts  = np.column_stack([leftx,  lefty])  if leftx.size  else None
        right_pts = np.column_stack([rightx, righty]) if rightx.size else None
        result = SWResult(
            ok=True,
            left_pts=left_pts,
            right_pts=right_pts,
            left_fit=left_fit,
            right_fit=right_fit,
            lateral_px=float(lateral),
            theta_deg=theta_deg,
            img_width=W,
            canvas_bgr=dbg if return_canvas else None,
        )

        if save_debug:
            _save_debug_image(dbg, pil_img, src_path)
        return result

    # Even on failure, return canvas (ROI/windows drawn)
    result = SWResult(
        ok=False,
        left_pts=None, right_pts=None,
        left_fit=left_fit, right_fit=right_fit,
        lateral_px=None, theta_deg=None, img_width=W,
        canvas_bgr=dbg if return_canvas else None,
    )
    if save_debug:
        _save_debug_image(dbg, pil_img, src_path)
    return result  # important: always return result


def _save_debug_image(dbg_bgr: np.ndarray, pil_img: Image.Image, src_path: Optional[str]):
    """Save debug image (filename follows input; falls back to PIL image name)."""
    if src_path is not None:
        base_name = os.path.basename(src_path)
    else:
        base_name = os.path.basename(getattr(pil_img, "filename", "unknown.jpg"))
    out_name = f"debug_{base_name}"
    out_path = os.path.join(DEBUG_DIR, out_name)
    cv2.imwrite(out_path, dbg_bgr)
    print(f"[SW] Saved {out_path}")

# ---------- CLI ----------
def _iter_images(folder: str):
    pats = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
    paths=[]
    for p in pats:
        paths.extend(glob.glob(os.path.join(folder, p)))
    return sorted(paths)

def run_batch(folder: str, save_debug=True, csv_out=os.path.join(DEBUG_DIR, "sw_results.csv")):
    paths = _iter_images(folder)
    if not paths:
        print(f"[Batch] No images found in: {folder}")
        return
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","ok","left_pts","right_pts","lateral_px","theta_deg"])
        for p in paths:
            try:
                pil = Image.open(p).convert("RGB")
                res = sliding_windows_white(pil, save_debug=save_debug, src_path=p)
                w.writerow([
                    os.path.basename(p), int(res.ok),
                    0 if res.left_pts is None else len(res.left_pts),
                    0 if res.right_pts is None else len(res.right_pts),
                    "" if res.lateral_px is None else f"{res.lateral_px:.2f}",
                    "" if res.theta_deg is None else f"{res.theta_deg:.2f}",
                ])
            except Exception as e:
                print(f"[Batch] Skip {p}: {e}")
    print(f"[Batch] Results saved to: {csv_out}")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image")
    g.add_argument("--folder", help="Folder with images (batch)")
    ap.add_argument("--save_debug", action="store_true", help="Save debug overlays")
    args = ap.parse_args()

    if args.folder:
        run_batch(args.folder, save_debug=args.save_debug)
    else:
        pil = Image.open(args.image).convert("RGB")
        sliding_windows_white(pil, save_debug=args.save_debug, src_path=args.image)

if __name__ == "__main__":
    main()
