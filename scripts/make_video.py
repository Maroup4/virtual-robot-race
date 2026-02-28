# make_video.py
# Convert saved frames (frame_*.jpg) to an MP4 video using imageio-ffmpeg (libx264).

import os
import re
import sys
import glob
import math
import argparse
import statistics
import numpy as np
import imageio  # uses imageio-ffmpeg internally

try:
    import cv2  # optional: for resizing
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def natural_key(p: str):
    base = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', base)]


def find_images(images_dir: str):
    patterns = [
        os.path.join(images_dir, "frame_*.jpg"),
        os.path.join(images_dir, "*.jpg"),
    ]
    files = []
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            break
    files.sort(key=natural_key)
    return files


def infer_fps_from_file_times(frames: list[str]) -> float:
    """Estimate FPS from file modification times (median delta)."""
    if len(frames) < 3:
        return 0.0
    ts = [os.path.getmtime(p) for p in frames]
    deltas = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if not deltas:
        return 0.0
    med_dt = statistics.median(deltas)
    if med_dt <= 0:
        return 0.0
    fps = 1.0 / med_dt
    return fps if math.isfinite(fps) and fps > 0 else 0.0


def read_image(path, target_wh=None):
    im = imageio.imread(path)
    if target_wh is None:
        return im
    w, h = target_wh
    if im.shape[1] == w and im.shape[0] == h:
        return im
    if _HAS_CV2:
        return cv2.resize(im, (w, h))
    out = np.zeros((h, w, im.shape[2]), dtype=im.dtype)
    hh = min(h, im.shape[0]); ww = min(w, im.shape[1])
    out[:hh, :ww] = im[:hh, :ww]
    return out


def images_to_video_ffmpeg(images_dir: str, out_path: str, fps: float, infer_fps: bool = False) -> str:
    frames = find_images(images_dir)
    if not frames:
        print(f"[make_video] No jpg frames found in: {images_dir}")
        return ""

    if infer_fps:
        est = infer_fps_from_file_times(frames)
        if est > 0:
            print(f"[make_video] inferred FPS ≈ {est:.2f}")
            fps = est
        else:
            print("[make_video] FPS inference failed → fallback to provided fps")

    f0 = imageio.imread(frames[0])
    if f0 is None:
        print("[make_video] Failed to read first frame")
        return ""
    h, w = f0.shape[:2]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        macro_block_size=None,  # allow non-multiple-of-16 resolutions
        pixelformat="yuv420p",
        quality=6
    )
    try:
        for i, p in enumerate(frames, 1):
            frame = read_image(p, target_wh=(w, h))
            writer.append_data(frame)
            if i % 50 == 0:
                print(f"[make_video] {i}/{len(frames)} processed...")
    finally:
        writer.close()

    try:
        size = os.path.getsize(out_path)
    except Exception:
        size = 0
    if size == 0:
        print("[make_video] Output file is 0 bytes (unexpected).")
        return ""

    print(f"[make_video] Saved → {out_path} @ {fps:.2f} fps, {w}x{h}, frames={len(frames)}")
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(description="Stitch frames to MP4 via imageio-ffmpeg (libx264).")
    ap.add_argument("--images", type=str, default="", help="Path to images dir (…/run_xxx/images)")
    ap.add_argument("--out", type=str, default="", help="Output video path (…/run_xxx/output_video.mp4)")
    ap.add_argument("--run-dir", type=str, default="", help="Run dir (…/run_xxx). If set, images=<run>/images, out=<run>/output_video.mp4")
    ap.add_argument("--fps", type=float, default=20.0, help="Frames per second (default: 20)")
    ap.add_argument("--infer-fps", action="store_true", help="Infer FPS from file timestamps (median Δ).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_dir:
        images_dir = os.path.join(args.run_dir, "images")
        out_path = os.path.join(args.run_dir, "output_video.mp4")
    else:
        images_dir = args.images
        out_path = args.out

    if not images_dir or not out_path:
        print(
            "Usage:\n"
            "  python -u make_video.py --run-dir <training_data/run_YYYYMMDD_HHMMSS> [--fps 20] [--infer-fps]\n"
            "or\n"
            "  python -u make_video.py --images <dir> --out <path.mp4> [--fps 20] [--infer-fps]"
        )
        sys.exit(1)

    images_to_video_ffmpeg(images_dir, out_path, fps=args.fps, infer_fps=args.infer_fps)
