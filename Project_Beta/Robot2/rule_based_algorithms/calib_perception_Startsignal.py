# calib_perception_Startsignal.py
# =================================
# Calibration / tuning script for start signal detection
#
# Usage:
#   python calib_perception_Startsignal.py                           # Default: input_jpg/ -> output/
#   python calib_perception_Startsignal.py --image debug/input_jpg/sample.jpg
#   python calib_perception_Startsignal.py --sweep_thresh --image debug/input_jpg/sample.jpg
#
# Folder structure:
#   Robot1/debug/
# ├── input_jpg/    # Place input images for validation here
# └── output/       # Output results (overlay images + CSV)
#
# Features:
# 1. Visualization of lamp region ROIs
# 2. Display red pixel ratio for each lamp
# 3. Test threshold parameter changes
# 4. BatchProcess + CSVOutput

import os
import sys
import argparse
import csv
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Debug directory (Robot1/debug/)
DEBUG_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug")
INPUT_DIR = os.path.join(DEBUG_BASE, "input_jpg")
OUTPUT_DIR = os.path.join(DEBUG_BASE, "output")

# DirectoryCreate
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class LampResult:
    """Detection result for each lamp"""
    lamp_id: int
    red_pixels: int
    total_pixels: int
    ratio: float
    is_on: bool
    roi: Tuple[int, int, int, int]  # (left, top, right, bottom)


@dataclass
class SignalResult:
    """Detection result for the entire start signal"""
    lamps: List[LampResult]
    red_count: int
    ready_to_go: bool
    is_go: bool


def is_red(pixel, red_thresh=140, green_thresh=130, blue_thresh=130) -> bool:
    """Determine whether an RGB pixel is red"""
    r, g, b = pixel[:3]  # RGBA compatible
    return r > red_thresh and g < green_thresh and b < blue_thresh


def analyze_startsignal(
    img: Image.Image,
    red_thresh: int = 140,
    green_thresh: int = 130,
    blue_thresh: int = 130,
    ratio_thresh: float = 0.03,
    ready_to_go: bool = False,
) -> SignalResult:
    """
    Analyze the start signal and return detailed results

    Args:
        img: PIL.Image (RGB)
        red_thresh: Red channel threshold
        green_thresh: Green channel threshold
        blue_thresh: Blue channel threshold
        ratio_thresh: Ratio threshold for lamp-on detection
        ready_to_go: Previous ready_to_go state

    Returns:
        SignalResult: Detailed detection result
    """
    width, height = img.size
    top = int(height * 0.05)
    bottom = int(height * 0.25)

    # Three lamp regions (Robot2: left side)
    lamp_positions = [
        (int(width * 0.18), int(width * 0.33)),  # Lamp 1: 18%-33%
        (int(width * 0.33), int(width * 0.48)),  # Lamp 2: 33%-48% (shifted left)
        (int(width * 0.48), int(width * 0.63)),  # Lamp 3: 48%-63% (shifted left)
    ]

    lamps = []
    red_count = 0

    for i, (left, right) in enumerate(lamp_positions):
        red_pixels = 0
        total_pixels = 0

        for y in range(top, bottom):
            for x in range(left, right):
                pixel = img.getpixel((x, y))
                if is_red(pixel, red_thresh, green_thresh, blue_thresh):
                    red_pixels += 1
                total_pixels += 1

        ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        is_on = ratio > ratio_thresh

        if is_on:
            red_count += 1

        lamps.append(LampResult(
            lamp_id=i + 1,
            red_pixels=red_pixels,
            total_pixels=total_pixels,
            ratio=ratio,
            is_on=is_on,
            roi=(left, top, right, bottom),
        ))

    # GO determination logic
    new_ready = ready_to_go
    is_go = False

    if red_count == 3:
        new_ready = True
    elif red_count == 0 and ready_to_go:
        is_go = True
        new_ready = False

    return SignalResult(
        lamps=lamps,
        red_count=red_count,
        ready_to_go=new_ready,
        is_go=is_go,
    )


def draw_overlay(
    img: Image.Image,
    result: SignalResult,
    red_thresh: int = 140,
    green_thresh: int = 130,
    blue_thresh: int = 130,
    ratio_thresh: float = 0.03,
) -> Image.Image:
    """
    Generate an image with detection results overlaid

    Args:
        img: Original image
        result: Detection result
        red_thresh, green_thresh, blue_thresh: Thresholds (for display)
        ratio_thresh: Ratio threshold (for display)

    Returns:
        Overlay image
    """
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # Font (fall back to default if system font is unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_large = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_large = font

    width, height = img.size

    # Draw ROI for each lamp (outline only; text is grouped in the lower half)
    for lamp in result.lamps:
        left, top, right, bottom = lamp.roi

        # ROI rectangle (on: green, off: red)
        color = "lime" if lamp.is_on else "red"
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

    # === Place info panel in the lower half ===
    panel_top = int(height * 0.45)  # Start from 45% position of the image
    panel_left = 10
    line_height = 20

    # Lamp information
    lamp_lines = []
    for lamp in result.lamps:
        status = "ON" if lamp.is_on else "OFF"
        color = "lime" if lamp.is_on else "red"
        lamp_lines.append((f"Lamp{lamp.lamp_id}: {lamp.ratio*100:5.1f}% [{status}]", color))

    # Overall status
    status_lines = [
        (f"Red Count: {result.red_count}/3", "white"),
        (f"Ready: {result.ready_to_go}", "yellow" if result.ready_to_go else "white"),
        (f"GO: {result.is_go}", "lime" if result.is_go else "white"),
        (f"---", "gray"),
        (f"Thresh R>{red_thresh} G<{green_thresh} B<{blue_thresh}", "cyan"),
        (f"Ratio>{ratio_thresh*100:.1f}%", "cyan"),
    ]

    all_lines = lamp_lines + status_lines

    # Background box (semi-transparent white)
    box_height = len(all_lines) * line_height + 15
    box_width = 280

    # Create a separate RGBA layer for semi-transparent drawing
    panel_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel_layer)
    panel_draw.rectangle(
        [panel_left, panel_top, panel_left + box_width, panel_top + box_height],
        fill=(255, 255, 255, 160)  # Semi-transparent white
    )
    overlay = Image.alpha_composite(overlay.convert("RGBA"), panel_layer).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    # Draw text (changed to dark colors for readability)
    y = panel_top + 8
    for text, orig_color in all_lines:
        # Convert to dark colors since the background is white
        if orig_color == "white":
            text_color = "black"
        elif orig_color == "lime":
            text_color = "green"
        elif orig_color == "yellow":
            text_color = "orange"
        elif orig_color == "cyan":
            text_color = "blue"
        elif orig_color == "gray":
            text_color = "gray"
        elif orig_color == "red":
            text_color = "darkred"
        else:
            text_color = orig_color
        draw.text((panel_left + 8, y), text, fill=text_color, font=font)
        y += line_height

    return overlay


def save_overlay(
    img: Image.Image,
    result: SignalResult,
    out_path: str,
    **thresh_kwargs,
) -> str:
    """Save the overlay image"""
    overlay = draw_overlay(img, result, **thresh_kwargs)
    overlay.save(out_path, quality=90)
    return out_path


def process_single(
    image_path: str,
    save_overlay_flag: bool = True,
    out_dir: str = OUTPUT_DIR,
    **thresh_kwargs,
) -> SignalResult:
    """Process a single image"""
    img = Image.open(image_path).convert("RGB")
    result = analyze_startsignal(img, **thresh_kwargs)

    if save_overlay_flag:
        base = os.path.basename(image_path)
        # Save to output with the same filename
        out_path = os.path.join(out_dir, base)
        save_overlay(img, result, out_path, **thresh_kwargs)
        print(f"[Calib] Saved: {out_path}")

    return result


def process_batch(
    folder: str = INPUT_DIR,
    save_overlay_flag: bool = True,
    out_dir: str = OUTPUT_DIR,
    csv_out: Optional[str] = None,
    **thresh_kwargs,
) -> List[Tuple[str, SignalResult]]:
    """Batch process images in a folder"""
    # Use lowercase patterns only (to avoid duplicates since Windows is case-insensitive)
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, p)))
    # Remove duplicates and sort
    paths = sorted(set(paths))

    if not paths:
        print(f"[Calib] No images found in: {folder}")
        print(f"[Calib] Please place images in: {INPUT_DIR}")
        return []

    print(f"[Calib] Input:  {folder}")
    print(f"[Calib] Output: {out_dir}")
    print(f"[Calib] Processing {len(paths)} images...")

    results = []
    ready_state = False  # For sequence tracking

    for path in paths:
        img = Image.open(path).convert("RGB")
        result = analyze_startsignal(img, ready_to_go=ready_state, **thresh_kwargs)
        ready_state = result.ready_to_go  # Carry over the state

        results.append((path, result))

        if save_overlay_flag:
            base = os.path.basename(path)
            # Save to output with the same filename
            out_path = os.path.join(out_dir, base)
            save_overlay(img, result, out_path, **thresh_kwargs)

    # CSVOutput
    if csv_out is None:
        csv_out = os.path.join(out_dir, "results.csv")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "red_count", "ready_to_go", "is_go",
            "lamp1_ratio", "lamp1_on",
            "lamp2_ratio", "lamp2_on",
            "lamp3_ratio", "lamp3_on",
        ])
        for path, result in results:
            row = [
                os.path.basename(path),
                result.red_count,
                result.ready_to_go,
                result.is_go,
            ]
            for lamp in result.lamps:
                row.extend([f"{lamp.ratio:.4f}", lamp.is_on])
            writer.writerow(row)

    print(f"[Calib] CSV saved: {csv_out}")
    print(f"[Calib] Done! {len(paths)} images processed.")
    return results


def sweep_threshold(
    image_path: str,
    red_range: Tuple[int, int, int] = (100, 180, 10),
    green_range: Tuple[int, int, int] = (100, 160, 10),
    out_dir: str = OUTPUT_DIR,
) -> None:
    """
    Sweep thresholds to find optimal values

    Args:
        image_path: Test image
        red_range: (min, max, step) for red_thresh
        green_range: (min, max, step) for green_thresh
    """
    img = Image.open(image_path).convert("RGB")

    csv_out = os.path.join(out_dir, "calib_threshold_sweep.csv")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "red_thresh", "green_thresh",
            "lamp1_ratio", "lamp2_ratio", "lamp3_ratio",
            "red_count",
        ])

        for r_th in range(red_range[0], red_range[1] + 1, red_range[2]):
            for g_th in range(green_range[0], green_range[1] + 1, green_range[2]):
                result = analyze_startsignal(
                    img,
                    red_thresh=r_th,
                    green_thresh=g_th,
                    blue_thresh=g_th,  # Use the same value
                )
                writer.writerow([
                    r_th, g_th,
                    f"{result.lamps[0].ratio:.4f}",
                    f"{result.lamps[1].ratio:.4f}",
                    f"{result.lamps[2].ratio:.4f}",
                    result.red_count,
                ])

    print(f"[Calib] Threshold sweep saved: {csv_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration tool for start signal detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Folder structure:
  Robot1/debug/
  ├── input_jpg/    # Place input images here
  └── output/       # Output results

Usage examples:
  python calib_perception_Startsignal.py                    # input_jpg/ -> output/
  python calib_perception_Startsignal.py --image sample.jpg # Single image
  python calib_perception_Startsignal.py --sweep_thresh --image sample.jpg

Default paths:
  Input:  {INPUT_DIR}
  Output: {OUTPUT_DIR}
"""
    )

    # Input (optional; uses default folder if not specified)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--folder", help=f"Path to image folder (default: {INPUT_DIR})")

    # Options
    parser.add_argument("--no_overlay", action="store_true",
                        help="Do not save overlay images")
    parser.add_argument("--sweep_thresh", action="store_true",
                        help="Threshold sweep mode (--image required)")
    parser.add_argument("--out_dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")

    # Threshold parameters
    parser.add_argument("--red_thresh", type=int, default=140,
                        help="Red channel threshold (default: 140)")
    parser.add_argument("--green_thresh", type=int, default=130,
                        help="Green channel threshold (default: 130)")
    parser.add_argument("--blue_thresh", type=int, default=130,
                        help="Blue channel threshold (default: 130)")
    parser.add_argument("--ratio_thresh", type=float, default=0.03,
                        help="Ratio threshold for lamp-on detection (default: 0.03)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_overlay_flag = not args.no_overlay

    thresh_kwargs = {
        "red_thresh": args.red_thresh,
        "green_thresh": args.green_thresh,
        "blue_thresh": args.blue_thresh,
        "ratio_thresh": args.ratio_thresh,
    }

    if args.sweep_thresh:
        if not args.image:
            print("[Error] --sweep_thresh requires --image")
            sys.exit(1)
        sweep_threshold(args.image, out_dir=args.out_dir)
    elif args.image:
        result = process_single(
            args.image,
            save_overlay_flag=save_overlay_flag,
            out_dir=args.out_dir,
            **thresh_kwargs,
        )
        print(f"\n[Result] {os.path.basename(args.image)}")
        print(f"  Red Count: {result.red_count}/3")
        print(f"  Ready: {result.ready_to_go}, GO: {result.is_go}")
        for lamp in result.lamps:
            status = "ON" if lamp.is_on else "OFF"
            print(f"  Lamp{lamp.lamp_id}: {lamp.ratio*100:5.2f}% ({status})")
    else:
        # Default: input_jpg/ -> output/
        folder = args.folder if args.folder else INPUT_DIR
        process_batch(
            folder,
            save_overlay_flag=save_overlay_flag,
            out_dir=args.out_dir,
            **thresh_kwargs,
        )


if __name__ == "__main__":
    main()
