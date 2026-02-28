# perception_trackposition.py
# Detect lane position (lane_name + lane_offset) based on a fixed color order.
# AAGP track order (from outside to inside):
# Black | White(outside) | Red | White(center) | Green | Blue | White(inside) | Black

from dataclasses import dataclass
from PIL import Image
import cv2, numpy as np, os, argparse, csv, glob
from typing import Tuple, Optional

@dataclass
class LaneResult:
    ok: bool
    lane_name: str
    lane_offset: float   # normalized offset within the detected lane band: -1..+1 (left negative)
    confidence: float    # simple coverage ratio: 0..1


# === HSV color thresholds ===
COLOR_RANGES = {
    "Red1":  ((0,   100, 80),  (10,  255, 255)),
    "Red2":  ((170, 100, 80),  (180, 255, 255)),
    "Green": ((35,  80,  80),  (85,  255, 255)),
    "Blue":  ((95,  80,  80),  (140, 255, 255)),
    "White": ((0,   0,  200),  (180, 40,  255)),
    "Black": ((0,   0,  0),    (180, 255, 50)),
}

# === ROI settings (relative vertical range) ===
ROI_TOP_FRAC  = 0.60
ROI_BOT_FRAC  = 0.80

# === Debug directory ===
project_root = os.path.dirname(os.path.dirname(__file__))
DEBUG_DIR = os.path.join(project_root, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


def classify_colors(hsv_roi: np.ndarray) -> np.ndarray:
    """
    Build per-color masks and assign each pixel the strongest class.
    Returns a 2D array of string labels in {"Black","White","Red","Green","Blue"}.
    """
    masks = {}
    for name, (lo, hi) in COLOR_RANGES.items():
        masks[name] = cv2.inRange(hsv_roi, np.array(lo, np.uint8), np.array(hi, np.uint8))

    # Merge Red1 + Red2
    masks["Red"] = cv2.bitwise_or(masks["Red1"], masks["Red2"])
    del masks["Red1"], masks["Red2"]

    # Stack and argmax → index map
    order = ["Black", "White", "Red", "Green", "Blue"]
    stacked = np.stack([masks[c] for c in order], axis=-1)
    idx = np.argmax(stacked, axis=-1)

    # Map indices to class names
    label_map = np.vectorize(lambda i: order[i])(idx)
    return label_map


def _put_label_top(img: np.ndarray, text: str, top_frac: float = 0.06) -> None:
    """Draw text near the top safely (auto scales; white text with black outline)."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = 1.0
    thick = 2

    # Auto scale to fit width (min 0.6, max 1.2)
    (tw, th), _ = cv2.getTextSize(text, font, base_scale, thick)
    scale = min(1.2, max(0.6, (w - 40) / max(1, tw)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    # Top margin
    baseline_y = max(th + 16, int(h * top_frac))

    # Semi-transparent panel
    x0, y0 = 10, baseline_y - th - 10
    x1, y1 = x0 + tw + 20, baseline_y + 10
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w - 1, x1); y1 = min(h - 1, y1)
    roi = img[y0:y1, x0:x1].copy()
    panel = roi.copy()
    cv2.rectangle(panel, (0, 0), (x1 - x0, y1 - y0), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, roi, 0.55, 0, roi)  # 45% opacity
    img[y0:y1, x0:x1] = roi

    # Text (black outline → white)
    org = (x0 + 10, baseline_y)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _nearest_color_run(row_labels: np.ndarray, target_color: str, cx: int) -> Optional[Tuple[int, int]]:
    """
    From a 1D label row, compress consecutive target_color pixels into runs and
    return (start_x, end_x) of the run whose center is closest to cx.
    """
    xs = np.where(row_labels == target_color)[0]
    if xs.size == 0:
        return None

    # Compress to consecutive runs
    runs = []
    start = xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = x
            prev = x
    runs.append((start, prev))

    # Choose run closest to center x
    best = min(runs, key=lambda ab: abs((ab[0] + ab[1]) * 0.5 - cx))
    return best  # (left_x, right_x)


def _draw_plus(img: np.ndarray, x: int, y: int, size: int = 6,
               color: Tuple[int, int, int] = (255, 0, 255), thickness: int = 2) -> None:
    """Draw a tilted cross (+) marker (default: purple in BGR)."""
    cv2.drawMarker(img, (int(x), int(y)), color,
                   markerType=cv2.MARKER_TILTED_CROSS,
                   markerSize=size, thickness=thickness)


def detect_lane(pil_img: Image.Image, save_debug: bool = False) -> LaneResult:
    """
    Detect the lane color under the image center and estimate:
      - lane_name (categorical)
      - lane_offset (-1..+1 across the detected colored band)
      - confidence (coverage ratio in the middle row)
    """
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = bgr.shape
    roi_top = int(h * ROI_TOP_FRAC)
    roi_bot = int(h * ROI_BOT_FRAC)
    roi = bgr[roi_top:roi_bot, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    label_map = classify_colors(hsv)

    mid_y = label_map.shape[0] // 2
    row = label_map[mid_y, :]

    cx = w // 2
    lane_color = row[cx]  # class at the image center within ROI

    # --- lane_name classification ---
    lane_name = "Unknown"
    if lane_color == "Red":
        lane_name = "Red"
    elif lane_color == "Green":
        lane_name = "Green"
    elif lane_color == "Blue":
        lane_name = "Blue"
    elif lane_color == "White":
        # Peek left/right neighbors to disambiguate which white band we're on
        left  = row[max(cx - 30, 0)]
        right = row[min(cx + 30, w - 1)]
        if "Red" in (left, right):
            lane_name = "CenterWhite"
        elif "Black" in (left, right):
            lane_name = "OutsideWhite"
        elif "Blue" in (left, right):
            lane_name = "InsideWhite"
        else:
            lane_name = "White"
    elif lane_color == "Black":
        lane_name = "Black"

    # --- lane_offset computation ---
    mask_lane = (row == lane_color)
    xs = np.where(mask_lane)[0]
    if xs.size > 5:
        left_x, right_x = xs[0], xs[-1]
        lane_cx = 0.5 * (left_x + right_x)
        # normalize: -1 at left edge of band, +1 at right edge of band
        lane_offset = (cx - lane_cx) / ((right_x - left_x) / 2.0 + 1e-6)
        confidence = float(xs.size) / float(w)
    else:
        lane_offset, confidence = 0.0, 0.0

    # === Debug output ===
    if save_debug:
        dbg = bgr.copy()
        # ROI (red box)
        cv2.rectangle(dbg, (0, roi_top), (w, roi_bot), (0, 0, 255), 2)
        # Center vertical line (green)
        cv2.line(dbg, (cx, roi_top), (cx, roi_bot), (0, 255, 0), 2)
        # Label (lane name + offset) at safe top region
        txt = f"{lane_name}, offset={lane_offset:+.2f}"
        _put_label_top(dbg, txt, top_frac=0.06)

        # Vertical scan: mark left/right edges of the current lane color with + markers
        STEP_Y = 5  # every 5 pixels in ROI space
        target_color = lane_color
        for yy in range(0, label_map.shape[0], STEP_Y):
            row_labels = label_map[yy, :]
            run = _nearest_color_run(row_labels, target_color, cx)
            if run is None:
                continue
            lx, rx = run
            y_img = roi_top + yy  # ROI → full image y
            _draw_plus(dbg, int(lx), y_img, size=8)  # left edge
            _draw_plus(dbg, int(rx), y_img, size=8)  # right edge

        # Save with incremental counter
        counter_path = os.path.join(DEBUG_DIR, "counter.txt")
        try:
            with open(counter_path, "r") as f:
                cnt = int(f.read().strip())
        except Exception:
            cnt = 0
        cnt += 1
        with open(counter_path, "w") as f:
            f.write(str(cnt))
        out_path = os.path.join(DEBUG_DIR, f"debug_lane_{cnt:06d}.jpg")
        cv2.imwrite(out_path, dbg)
        print(f"[TrackPosition] Saved {out_path}")

    return LaneResult(True, lane_name, float(lane_offset), float(confidence))


def _iter_images(folder: str):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, p)))
    return sorted(paths)


def run_batch(folder: str, save_debug: bool = True,
              csv_out: str = os.path.join(DEBUG_DIR, "trackposition_results.csv")) -> None:
    paths = _iter_images(folder)
    if not paths:
        print(f"[Batch] No images found in {folder}")
        return
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ok", "lane_name", "lane_offset", "confidence"])
        for p in paths:
            try:
                pil = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[Batch] Skip {p}: load error: {e}")
                continue
            res = detect_lane(pil, save_debug=save_debug)
            w.writerow([
                os.path.basename(p), int(res.ok), res.lane_name,
                f"{res.lane_offset:.3f}", f"{res.confidence:.3f}"
            ])
    print(f"[Batch] Results saved: {csv_out}")


# --- CLI ---
def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image")
    g.add_argument("--folder", help="Folder of images (batch)")
    parser.add_argument("--save_debug", action="store_true", help="Save debug overlays to /debug")
    args = parser.parse_args()

    if args.folder:
        run_batch(args.folder, save_debug=args.save_debug)
    else:
        pil = Image.open(args.image).convert("RGB")
        res = detect_lane(pil, save_debug=args.save_debug)
        print(res)


if __name__ == "__main__":
    main()
