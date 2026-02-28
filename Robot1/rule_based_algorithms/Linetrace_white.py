# Linetrace_white.py
# Rule-based PID control for white line following based on camera image and SOC value.

from PIL import Image
import cv2
import numpy as np
import os
import time

# PID control parameters
Kp = 0.005
Ki = 0.0
Kd = 0.001

# Motion control parameters
FORWARD = 0.3
TURN_GAIN = 1.0
A_WEIGHT = 0.5
B_WEIGHT = 0.5

# Debug mode toggle
DEBUG = True
if DEBUG:
    debug_folder = os.path.join("data_interative", "debug")
    os.makedirs(debug_folder, exist_ok=True)
    with open(os.path.join(debug_folder, "counter.txt"), "w") as f:
        f.write("0")
    print("[LineTrace] Debug counter initialized.")

prev_error = 0
integral = 0

def detect_gravity_and_angle(binary, roi_top):
    """Extracts the centroid and angle of a white line region from a binary mask."""
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 5:
        return None, None, None

    x = coords[:, 0, 0]
    y = coords[:, 0, 1] + roi_top

    x_c = np.mean(x)
    y_c = np.mean(y)

    poly = np.polyfit(x, y, 1)
    theta_rad = np.arctan(poly[0])

    return (x_c, y_c), theta_rad, poly

def run(soc, pil_img):
    global prev_error, integral

    if soc < 0.2:
        return 0.0, 0.0

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    roi_top = int(height * 0.4)
    roi_bottom = int(height * 0.9)
    roi = gray[roi_top:roi_bottom, :]

    _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    center = width // 2
    gravity_point, target_angle, poly = detect_gravity_and_angle(binary, roi_top)

    if gravity_point is None or target_angle is None:
        print("[LineTrace] No valid line detected for gravity + angle tracking.")
        return 0.5, 0.5

    deviation = (gravity_point[0] - center) / center
    theta_norm = target_angle / np.radians(45.0)
    correction = A_WEIGHT * deviation + B_WEIGHT * theta_norm
    turn = TURN_GAIN * correction

    left = np.clip(FORWARD - turn, -1.0, 1.0)
    right = np.clip(FORWARD + turn, -1.0, 1.0)

    print(f"[LineTrace] deviation={deviation:.3f}, angle={np.degrees(target_angle):.1f}°, correction={correction:.3f}, L={left:.2f}, R={right:.2f}")

    if DEBUG:
        debug_full = img.copy()
        cv2.rectangle(debug_full, (0, roi_top), (width, roi_bottom), (0, 0, 255), 2)
        cv2.line(debug_full, (center, roi_top), (center, roi_bottom), (0, 255, 0), 2)
        cv2.drawMarker(debug_full, (int(gravity_point[0]), int(gravity_point[1])), (0, 0, 255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

        x1 = 0
        y1 = int(poly[0] * x1 + poly[1])
        x2 = width
        y2 = int(poly[0] * x2 + poly[1])
        cv2.line(debug_full, (x1, y1), (x2, y2), (255, 0, 0), 2)

        vec_origin = (width // 2, height - 50)
        vec_scale = 40
        end_point = (int(vec_origin[0] + left * vec_scale), int(vec_origin[1] - right * vec_scale))

        cv2.rectangle(debug_full, (vec_origin[0] - vec_scale, vec_origin[1] - vec_scale),
                                (vec_origin[0] + vec_scale, vec_origin[1] + vec_scale), (200, 200, 200), 1)
        cv2.arrowedLine(debug_full, vec_origin, end_point, (0, 0, 255), 2, tipLength=0.2)

        (text_width, text_height), baseline = cv2.getTextSize("Torque Vector", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(debug_full, "Torque Vector", (width // 2 - text_width // 2, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        counter_path = os.path.join(debug_folder, "counter.txt")
        counter = 0
        if os.path.exists(counter_path):
            try:
                with open(counter_path, "r") as f:
                    counter = int(f.read().strip())
            except:
                pass
        counter += 1
        with open(counter_path, "w") as f:
            f.write(str(counter))

        debug_filename = f"debug_latest_RGB_{counter:06d}.jpg"
        debug_path = os.path.join(debug_folder, debug_filename)
        try:
            cv2.imwrite(debug_path, debug_full)
            print(f"[LineTrace] Saved debug image to {debug_path}")
        except Exception as e:
            print(f"[LineTrace] Failed to save debug image: {e}")

    return left, right

def main_batch(input_folder="rulebasesample", output_folder="debug", soc=1.0):
    os.makedirs(output_folder, exist_ok=True)
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
    print(f"[Batch] Found {len(jpg_files)} jpg files in {input_folder}")

    for fname in jpg_files:
        input_path = os.path.join(input_folder, fname)
        try:
            pil_img = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"[Batch] Skipping {fname} due to load error: {e}")
            continue

        run(soc, pil_img)

def test_mode(image_path, soc):
    try:
        pil_img = Image.open(image_path).convert("RGB")
        run(soc, pil_img)
    except Exception as e:
        print(f"[Test] Failed to load test image: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data_interative/latest_RGB.jpg", help="Path to input image")
    parser.add_argument("--soc", type=float, default=1.0, help="Simulated SOC value")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--input_folder", type=str, default="rulebasesample")
    parser.add_argument("--output_folder", type=str, default="debug")
    args = parser.parse_args()

    if args.batch:
        print("[Batch] Linetrace_white.py batch mode")
        main_batch(args.input_folder, args.output_folder, args.soc)
    else:
        print("[Test] Linetrace_white.py test mode")
        print(f"[Test] Using image: {args.image}")
        print(f"[Test] Simulated SOC: {args.soc}")
        test_mode(args.image, args.soc)
