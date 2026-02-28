# perception_Lane.py
# Red-lane perception: compute lateral deviation, heading angle, confidence.
# Debug-friendly: run on a specified JPG once and print the result.

from dataclasses import dataclass
from PIL import Image
import cv2, numpy as np, os, argparse
import csv, glob

@dataclass
class LaneObs:
    ok: bool
    lateral: float      # -1..1 (left negative)
    heading_deg: float  # lane direction in degrees (left negative-ish depending on camera)
    confidence: float   # 0..1 (mask area ratio)

# === HSV ranges for RED (two ranges due to hue wrap) ===
# tweak if your lighting differs
RED1_LO, RED1_HI = (0, 80, 60),   (10, 255, 255)
RED2_LO, RED2_HI = (170, 80, 60), (180, 255, 255)

# === ROI (relative heights) ===
ROI_TOP_FRAC  = 0.45
ROI_BOT_FRAC  = 0.90

# === Debug output ===
# Go up one folder from "rule_based_algorithms" to project root
project_root = os.path.dirname(os.path.dirname(__file__))

# Always save debug images here:
DEBUG_DIR = os.path.join(project_root, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


def _make_red_mask(hsv_roi: np.ndarray) -> np.ndarray:
    m1 = cv2.inRange(hsv_roi, np.array(RED1_LO), np.array(RED1_HI))
    m2 = cv2.inRange(hsv_roi, np.array(RED2_LO), np.array(RED2_HI))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

def detect_from_pil(pil_img: Image.Image, save_debug: bool=False) -> LaneObs:
    """Compute lane observation from a PIL image."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = bgr.shape
    roi_top = int(h * ROI_TOP_FRAC)
    roi_bot = int(h * ROI_BOT_FRAC)

    roi = bgr[roi_top:roi_bot, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = _make_red_mask(hsv)
    M = cv2.moments(mask)
    area = M["m00"]

    # confidence: how much of ROI is red-ish
    conf = 0.0 if area <= 0 else float(min(1.0, area / (mask.size * 0.15)))

    if area < 1000:  # too few pixels → treat as lost
        if save_debug:
            _save_debug(bgr, w, roi_top, roi_bot, None, None, mask)
        return LaneObs(False, 0.0, 0.0, conf)

        # centroid (x only; y not needed for lateral)
    cx = M["m10"] / area
    lateral = (cx - (w/2)) / (w/2)  # normalize to -1..1

    # --- NEW: get a sensible cy for the centroid (use mean Y of mask points in ROI) ---
    pts = cv2.findNonZero(mask)
    cy_roi = float(pts[:, :, 1].mean()) if pts is not None else (roi_bot - roi_top) * 0.5
    cy = roi_top + cy_roi

    # --- NEW: angle relative to straight-forward (vertical up) from robot center ---
    # robot center at bottom-middle of ROI
    x0, y0 = (w / 2.0), float(roi_bot)
    dx = cx - x0
    dy_up = (y0 - cy)  # distance in the "forward" (up) direction

    # Angle convention: left = minus, right = plus (deg)
    # atan2(sin-like, cos-like) -> atan2(dx, dy_up)
    heading_deg = float(np.degrees(np.arctan2(dx, dy_up + 1e-6)))

    if save_debug:
        _save_debug(
            bgr, w, roi_top, roi_bot,
            (cx, cy),            # centroid (in full-image coords)
            None,                # no fit line anymore
            mask,
            heading_deg          # pass the new angle to draw text
        )

    return LaneObs(True, float(lateral), heading_deg, conf)


def _save_debug(bgr, w, roi_top, roi_bot, cpt, fit, mask, angle_deg=None):
    dbg = bgr.copy()
    # ROI box + center line
    cv2.rectangle(dbg, (0, roi_top), (w, roi_bot), (0, 0, 255), 2)
    cv2.line(dbg, (w//2, roi_top), (w//2, roi_bot), (0, 255, 0), 2)  # center reference

    # draw center->centroid ray (replace old blue fit line)
    if cpt is not None:
        cx, cy = int(cpt[0]), int(cpt[1])
        # robot center at bottom-middle of ROI
        x0, y0 = int(w//2), int(roi_bot)
        cv2.drawMarker(dbg, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.arrowedLine(dbg, (x0, y0), (cx, cy), (0, 200, 0), 3, tipLength=0.15)

    # remove the fit-line drawing entirely

    # small mask preview (at top-left as you wanted)
    if mask is not None:
        mcolor = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mh, mw = mask.shape
        scale = 0.35
        mrsz = cv2.resize(mcolor, (int(mw*scale), int(mh*scale)))
        h_img, w_img, _ = dbg.shape
        y0 = 10; x0 = 10
        dbg[y0:y0+mrsz.shape[0], x0:x0+mrsz.shape[1]] = mrsz

    # angle overlay text (only angle)
    if angle_deg is not None:
        txt = f"Angle = {angle_deg:+.1f} deg"
        cv2.putText(dbg, txt, (20, roi_top - 10 if roi_top > 30 else 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(dbg, txt, (20, roi_top - 10 if roi_top > 30 else 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # filename with incremental counter (unchanged)
    counter_path = os.path.join(DEBUG_DIR, "counter.txt")
    try:
        cnt = int(open(counter_path, "r").read().strip())
    except:
        cnt = 0
    cnt += 1
    with open(counter_path, "w") as f:
        f.write(str(cnt))

    out_path = os.path.join(DEBUG_DIR, f"debug_red_{cnt:06d}.jpg")
    cv2.imwrite(out_path, dbg)
    print(f"[Perception] Saved debug: {out_path}")





def _fit_line_heading_deg(pts):
    # cv2.fitLine returns small arrays; convert to pure Python floats to avoid FutureWarnings
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx.ravel()[0]); vy = float(vy.ravel()[0])
    x0 = float(x0.ravel()[0]); y0 = float(y0.ravel()[0])
    heading_deg = float(np.degrees(np.arctan2(vy, vx)))
    return (vx, vy, x0, y0), heading_deg

def detect_from_pil(pil_img: Image.Image, save_debug: bool=False) -> LaneObs:
    # (unchanged until fit line)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = bgr.shape
    roi_top = int(h * ROI_TOP_FRAC)
    roi_bot = int(h * ROI_BOT_FRAC)

    roi = bgr[roi_top:roi_bot, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = _make_red_mask(hsv)
    M = cv2.moments(mask)
    area = M["m00"]

    conf = 0.0 if area <= 0 else float(min(1.0, area / (mask.size * 0.15)))

    if area < 1000:
        if save_debug:
            _save_debug(bgr, w, roi_top, roi_bot, None, None, mask)
        return LaneObs(False, 0.0, 0.0, conf)

    cx = M["m10"] / area
    lateral = (cx - (w/2)) / (w/2)

    pts = cv2.findNonZero(mask)
    fit, heading_deg = _fit_line_heading_deg(pts)

    if save_debug:
        # estimate cy for the marker (mean of ROI y in pts)
        cy_roi = float(pts[:,:,1].mean()) if pts is not None else (roi_bot - roi_top) * 0.5
        _save_debug(bgr, w, roi_top, roi_bot, (cx, roi_top + cy_roi), fit, mask)

    return LaneObs(True, float(lateral), heading_deg, conf)

def _iter_images(folder):
    pats = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for p in pats:
        paths.extend(glob.glob(os.path.join(folder, p)))
    return sorted(paths)

def run_batch(folder, save_debug=True, csv_out=os.path.join(DEBUG_DIR, "perception_batch_results.csv")):
    paths = _iter_images(folder)
    if not paths:
        print(f"[Batch] No images found in: {folder}")
        return
    print(f"[Batch] Found {len(paths)} images in {folder}")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ok", "lateral", "heading_deg", "confidence"])
        for p in paths:
            try:
                pil = Image.open(p).convert("RGB")
                obs = detect_from_pil(pil, save_debug=save_debug)
                w.writerow([os.path.basename(p), int(obs.ok), f"{obs.lateral:.6f}", f"{obs.heading_deg:.3f}", f"{obs.confidence:.3f}"])
            except Exception as e:
                print(f"[Batch] Skip {p}: {e}")
    print(f"[Batch] Results saved to: {csv_out}")

# --- CLI (single image OR folder) ---
def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single test image (JPG/PNG)")
    g.add_argument("--folder", help="Folder containing images to process (batch)")
    parser.add_argument("--save_debug", action="store_true", help="Save debug overlay(s) to project/debug")
    args = parser.parse_args()

    if args.folder:
        run_batch(args.folder, save_debug=args.save_debug)
        return

    # single image
    pil = Image.open(args.image).convert("RGB")
    obs = detect_from_pil(pil, save_debug=args.save_debug)
    print(f"ok={obs.ok}  lateral={obs.lateral:.3f}  heading_deg={obs.heading_deg:.1f}  conf={obs.confidence:.2f}")

if __name__ == "__main__":
    main()
