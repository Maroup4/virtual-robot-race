# perception_startsignal.py
# Detects red lamp pattern from a given RGB image (PIL) to determine race start.

from PIL import Image

def is_red(pixel, red_thresh=140, green_thresh=130, blue_thresh=130):
    """Returns True if the given pixel is considered 'red' based on RGB thresholds."""
    r, g, b = pixel
    return r > red_thresh and g < green_thresh and b < blue_thresh

def detect_start_signal(img):
    """
    Analyze a given PIL image to detect red start lamps.
    Returns True only once right after all lamps turn off (after being lit).
    """
    if not hasattr(detect_start_signal, 'ready_to_go'):
        detect_start_signal.ready_to_go = False

    try:
        width, height = img.size
        top = 0
        bottom = int(height * 0.2)

        # Define 3 rectangular regions (start lamps) on the top part of the image
        lamp_positions = [
            (int(width * 0.35), int(width * 0.5)),
            (int(width * 0.55), int(width * 0.7)),
            (int(width * 0.75), int(width * 0.9))
        ]

        red_count = 0
        for left, right in lamp_positions:
            red_pixels = 0
            total_pixels = 0
            for y in range(top, bottom):
                for x in range(left, right):
                    pixel = img.getpixel((x, y))
                    if is_red(pixel):
                        red_pixels += 1
                    total_pixels += 1
            ratio = red_pixels / total_pixels
            if ratio > 0.03:
                red_count += 1

        # Debug visualization (optional)
        DEBUG_MODE = False
        if DEBUG_MODE:
            from PIL import ImageDraw
            debug_img = img.copy()
            draw = ImageDraw.Draw(debug_img)
            for left, right in lamp_positions:
                draw.rectangle([left, top, right, bottom], outline="red", width=2)
            debug_img.save("debug_lamps.jpg")
            print("[StartSignal] Saved debug_lamps.jpg")

        # All 3 red lights are ON → prepare to go
        if red_count == 3:
            detect_start_signal.ready_to_go = True
            return False

        # All lights OFF & ready flag was set → GO!
        if red_count == 0 and detect_start_signal.ready_to_go:
            print("[StartSignal] GO!!")
            detect_start_signal.ready_to_go = False
            return True

        return False

    except Exception as e:
        print(f"[StartSignal] Error: {e}")
        return False
