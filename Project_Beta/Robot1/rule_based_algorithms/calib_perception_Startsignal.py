# calib_perception_Startsignal.py
# =================================
# スタートシグナル検出のキャリブレーション・調整用スクリプト # TODO: Manual translation needed
#
# 使用例: # TODO: Manual translation needed
#   python calib_perception_Startsignal.py                           # デフォルト: input_jpg/ → output/
#   python calib_perception_Startsignal.py --image debug/input_jpg/sample.jpg
#   python calib_perception_Startsignal.py --sweep_thresh --image debug/input_jpg/sample.jpg
#
# Folder構造: # TODO: Manual translation needed
#   Robot1/debug/
# ├── input_jpg/    # Validation用のInputImageを置く # TODO: Manual translation needed
# └── output/       # 結果（オーバーレイImage + CSV）をOutput # TODO: Manual translation needed
#
# 機能: # TODO: Manual translation needed
# 1. ランプ領域ROIの可視化 # TODO: Manual translation needed
# 2. 各ランプの赤ピクセル比率表示 # TODO: Manual translation needed
# 3. 閾値Parameterの変更Test # TODO: Manual translation needed
# 4. BatchProcess + CSVOutput

import os
import sys
import argparse
import csv
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# プロジェクトルートをPathにAdd # TODO: Manual translation needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# デバッグDirectory（Robot1/debug/） # TODO: Manual translation needed
DEBUG_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug")
INPUT_DIR = os.path.join(DEBUG_BASE, "input_jpg")
OUTPUT_DIR = os.path.join(DEBUG_BASE, "output")

# DirectoryCreate
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class LampResult:
    """各ランプの検出結果"""
    lamp_id: int
    red_pixels: int
    total_pixels: int
    ratio: float
    is_on: bool
    roi: Tuple[int, int, int, int]  # (left, top, right, bottom)


@dataclass
class SignalResult:
    """スタートシグナル全体の検出結果"""
    lamps: List[LampResult]
    red_count: int
    ready_to_go: bool
    is_go: bool


def is_red(pixel, red_thresh=140, green_thresh=130, blue_thresh=130) -> bool:
    """RGBピクセルが赤かどうかを判定"""
    r, g, b = pixel[:3]  # RGBA対応 # TODO: Manual translation needed
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
    スタートシグナルを解析し、詳細な結果を返す

    Args:
        img: PIL.Image (RGB)
        red_thresh: 赤チャンネル閾値
        green_thresh: 緑チャンネル閾値
        blue_thresh: 青チャンネル閾値
        ratio_thresh: 点灯判定の比率閾値
        ready_to_go: 前回の ready_to_go 状態

    Returns:
        SignalResult: 詳細な検出結果
    """
    width, height = img.size
    top = 0
    bottom = int(height * 0.2)

    # 3つのランプ領域 # TODO: Manual translation needed
    lamp_positions = [
        (int(width * 0.35), int(width * 0.50)),  # ランプ1 # TODO: Manual translation needed
        (int(width * 0.55), int(width * 0.70)),  # ランプ2 # TODO: Manual translation needed
        (int(width * 0.75), int(width * 0.90)),  # ランプ3 # TODO: Manual translation needed
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

    # GO判定ロジック # TODO: Manual translation needed
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
    検出結果をオーバーレイした画像を生成

    Args:
        img: 元画像
        result: 検出結果
        red_thresh, green_thresh, blue_thresh: 閾値（表示用）
        ratio_thresh: 比率閾値（表示用）

    Returns:
        オーバーレイ画像
    """
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # フォント（システムフォントがない場合はデフォルト） # TODO: Manual translation needed
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_large = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_large = font

    width, height = img.size

    # 各ランプのROIを描画（枠線のみ、テキストは下半分にまとめる） # TODO: Manual translation needed
    for lamp in result.lamps:
        left, top, right, bottom = lamp.roi

        # ROI矩形（点灯:緑、消灯:赤） # TODO: Manual translation needed
        color = "lime" if lamp.is_on else "red"
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

    # === 下半分に情報パネルを配置 === # TODO: Manual translation needed
    panel_top = int(height * 0.45)  # Imageの45%PositionからStart # TODO: Manual translation needed
    panel_left = 10
    line_height = 20

    # ランプ情報 # TODO: Manual translation needed
    lamp_lines = []
    for lamp in result.lamps:
        status = "ON" if lamp.is_on else "OFF"
        color = "lime" if lamp.is_on else "red"
        lamp_lines.append((f"Lamp{lamp.lamp_id}: {lamp.ratio*100:5.1f}% [{status}]", color))

    # 全体Status # TODO: Manual translation needed
    status_lines = [
        (f"Red Count: {result.red_count}/3", "white"),
        (f"Ready: {result.ready_to_go}", "yellow" if result.ready_to_go else "white"),
        (f"GO: {result.is_go}", "lime" if result.is_go else "white"),
        (f"---", "gray"),
        (f"Thresh R>{red_thresh} G<{green_thresh} B<{blue_thresh}", "cyan"),
        (f"Ratio>{ratio_thresh*100:.1f}%", "cyan"),
    ]

    all_lines = lamp_lines + status_lines

    # 背景ボックス（半透明の白） # TODO: Manual translation needed
    box_height = len(all_lines) * line_height + 15
    box_width = 280

    # 半透明描画のためRGBAModeで別レイヤーをCreate # TODO: Manual translation needed
    panel_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel_layer)
    panel_draw.rectangle(
        [panel_left, panel_top, panel_left + box_width, panel_top + box_height],
        fill=(255, 255, 255, 160)  # 半透明の白 # TODO: Manual translation needed
    )
    overlay = Image.alpha_composite(overlay.convert("RGBA"), panel_layer).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    # テキスト描画（黒文字に変更して見やすく） # TODO: Manual translation needed
    y = panel_top + 8
    for text, orig_color in all_lines:
        # 白背景なので暗い色にConvert # TODO: Manual translation needed
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
    """オーバーレイ画像を保存"""
    overlay = draw_overlay(img, result, **thresh_kwargs)
    overlay.save(out_path, quality=90)
    return out_path


def process_single(
    image_path: str,
    save_overlay_flag: bool = True,
    out_dir: str = OUTPUT_DIR,
    **thresh_kwargs,
) -> SignalResult:
    """単一画像を処理"""
    img = Image.open(image_path).convert("RGB")
    result = analyze_startsignal(img, **thresh_kwargs)

    if save_overlay_flag:
        base = os.path.basename(image_path)
        # 同名でoutputにSave # TODO: Manual translation needed
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
    """フォルダ内の画像をバッチ処理"""
    # 小文字パターンのみ使用（Windowsは大文字小文字を区別しないため重複を防ぐ） # TODO: Manual translation needed
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, p)))
    # 重複を除去してソート # TODO: Manual translation needed
    paths = sorted(set(paths))

    if not paths:
        print(f"[Calib] No images found in: {folder}")
        print(f"[Calib] Please place images in: {INPUT_DIR}")
        return []

    print(f"[Calib] Input:  {folder}")
    print(f"[Calib] Output: {out_dir}")
    print(f"[Calib] Processing {len(paths)} images...")

    results = []
    ready_state = False  # シーケンス追跡用 # TODO: Manual translation needed

    for path in paths:
        img = Image.open(path).convert("RGB")
        result = analyze_startsignal(img, ready_to_go=ready_state, **thresh_kwargs)
        ready_state = result.ready_to_go  # Stateを引き継ぐ # TODO: Manual translation needed

        results.append((path, result))

        if save_overlay_flag:
            base = os.path.basename(path)
            # 同名でoutputにSave # TODO: Manual translation needed
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
    閾値をスイープして最適値を探索

    Args:
        image_path: テスト画像
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
                    blue_thresh=g_th,  # 同じ値を使用 # TODO: Manual translation needed
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
        description="スタートシグナル検出のキャリブレーションツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
フォルダ構造:
  Robot1/debug/
  ├── input_jpg/    # InputImageを置く # TODO: Manual translation needed
  └── output/       # 結果をOutput # TODO: Manual translation needed

使用例:
  python calib_perception_Startsignal.py                    # input_jpg/ → output/
  python calib_perception_Startsignal.py --image sample.jpg # 単一Image # TODO: Manual translation needed
  python calib_perception_Startsignal.py --sweep_thresh --image sample.jpg

デフォルトパス:
  入力: {INPUT_DIR}
  出力: {OUTPUT_DIR}
"""
    )

    # Input（オプション、指定なしならデフォルトFolder） # TODO: Manual translation needed
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", help="単一画像のパス")
    group.add_argument("--folder", help=f"画像フォルダのパス（デフォルト: {INPUT_DIR}）")

    # オプション # TODO: Manual translation needed
    parser.add_argument("--no_overlay", action="store_true",
                        help="オーバーレイ画像を保存しない")
    parser.add_argument("--sweep_thresh", action="store_true",
                        help="閾値スイープモード（--image必須）")
    parser.add_argument("--out_dir", default=OUTPUT_DIR,
                        help=f"出力ディレクトリ（デフォルト: {OUTPUT_DIR}）")

    # 閾値Parameter # TODO: Manual translation needed
    parser.add_argument("--red_thresh", type=int, default=140,
                        help="赤チャンネル閾値（デフォルト: 140）")
    parser.add_argument("--green_thresh", type=int, default=130,
                        help="緑チャンネル閾値（デフォルト: 130）")
    parser.add_argument("--blue_thresh", type=int, default=130,
                        help="青チャンネル閾値（デフォルト: 130）")
    parser.add_argument("--ratio_thresh", type=float, default=0.03,
                        help="点灯判定の比率閾値（デフォルト: 0.03）")

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
        # デフォルト: input_jpg/ → output/
        folder = args.folder if args.folder else INPUT_DIR
        process_batch(
            folder,
            save_overlay_flag=save_overlay_flag,
            out_dir=args.out_dir,
            **thresh_kwargs,
        )


if __name__ == "__main__":
    main()
