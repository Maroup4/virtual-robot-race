"""
Comment Translation Script
===========================
Translates Japanese comments to English in Python files.

This script uses a simple dictionary-based approach for common phrases
and preserves code structure.

Usage:
    python translate_comments.py --file path/to/file.py
    python translate_comments.py --dir Robot1/
"""

import argparse
import re
from pathlib import Path


# Translation dictionary for common phrases
TRANSLATIONS = {
    # General
    "初期化": "Initialize",
    "設定": "Settings",
    "読み込み": "Load",
    "保存": "Save",
    "実行": "Execute",
    "開始": "Start",
    "終了": "End",
    "接続": "Connect",
    "切断": "Disconnect",
    "送信": "Send",
    "受信": "Receive",
    "処理": "Process",
    "変換": "Convert",
    "計算": "Calculate",
    "確認": "Check",
    "検証": "Validate",
    "更新": "Update",
    "削除": "Delete",
    "追加": "Add",
    "取得": "Get",
    "作成": "Create",
    "生成": "Generate",

    # AI/ML specific
    "学習": "Training",
    "訓練": "Training",
    "推論": "Inference",
    "モデル": "Model",
    "データセット": "Dataset",
    "バッチ": "Batch",
    "エポック": "Epoch",
    "損失": "Loss",
    "精度": "Accuracy",
    "検証": "Validation",
    "テスト": "Test",
    "予測": "Prediction",
    "特徴": "Feature",
    "ラベル": "Label",
    "重み": "Weight",
    "パラメータ": "Parameter",

    # Robot/Control specific
    "ロボット": "Robot",
    "制御": "Control",
    "駆動": "Drive",
    "操舵": "Steering",
    "トルク": "Torque",
    "角度": "Angle",
    "速度": "Speed",
    "位置": "Position",
    "姿勢": "Pose",
    "センサー": "Sensor",
    "カメラ": "Camera",
    "画像": "Image",
    "フレーム": "Frame",

    # Status/State
    "状態": "State",
    "ステータス": "Status",
    "モード": "Mode",
    "フラグ": "Flag",
    "エラー": "Error",
    "警告": "Warning",
    "成功": "Success",
    "失敗": "Failure",

    # File/Data
    "ファイル": "File",
    "フォルダ": "Folder",
    "ディレクトリ": "Directory",
    "パス": "Path",
    "データ": "Data",
    "ログ": "Log",
    "出力": "Output",
    "入力": "Input",

    # Common phrases
    "デバイス設定": "Device settings",
    "再現性のため": "For reproducibility",
    "データ変換": "Data transformation",
    "データ拡張": "Data augmentation",
    "学習パラメータ": "Training parameters",
    "学習ループ": "Training loop",
    "検証フェーズ": "Validation phase",
    "最良のモデル": "Best model",
    "早期終了": "Early stopping",
    "学習率": "Learning rate",
    "バッチサイズ": "Batch size",
    "損失関数": "Loss function",
    "最適化手法": "Optimizer",
    "活性化関数": "Activation function",

    # Specific project terms
    "キーボード入力": "Keyboard input",
    "推論エンジン": "Inference engine",
    "制御戦略": "Control strategy",
    "ルールベース": "Rule-based",
    "ニューラルネットワーク": "Neural network",
    "エンドツーエンド": "End-to-end",
    "ハイブリッド": "Hybrid",
}


def translate_comment(comment: str) -> str:
    """
    Translate a Japanese comment to English.

    Args:
        comment: Japanese comment string

    Returns:
        English comment string
    """
    # If already English (contains mostly ASCII), return as is
    ascii_chars = sum(1 for c in comment if ord(c) < 128)
    if ascii_chars / max(len(comment), 1) > 0.7:
        return comment

    translated = comment

    # Apply dictionary translations
    for jp, en in TRANSLATIONS.items():
        translated = translated.replace(jp, en)

    # If still contains Japanese, mark for manual review
    if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in translated):
        # Contains hiragana, katakana, or kanji
        translated = f"{translated} # TODO: Manual translation needed"

    return translated


def process_file(file_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single Python file and translate comments.

    Args:
        file_path: Path to Python file
        dry_run: If True, don't write changes

    Returns:
        Tuple of (total_comments, translated_comments)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_comments = 0
    translated_comments = 0
    new_lines = []

    for line in lines:
        # Check for inline comment
        if '#' in line and not line.strip().startswith('#'):
            # Inline comment
            code_part, comment_part = line.split('#', 1)
            total_comments += 1

            original_comment = comment_part.strip()
            translated_comment = translate_comment(original_comment)

            if original_comment != translated_comment:
                translated_comments += 1
                new_lines.append(f"{code_part}# {translated_comment}\n")
            else:
                new_lines.append(line)

        elif line.strip().startswith('#'):
            # Full line comment
            total_comments += 1
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            translated_comment = translate_comment(comment_text)

            if comment_text != translated_comment:
                translated_comments += 1
                new_lines.append(f"{' ' * indent}# {translated_comment}\n")
            else:
                new_lines.append(line)

        else:
            new_lines.append(line)

    if not dry_run and translated_comments > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return total_comments, translated_comments


def main():
    parser = argparse.ArgumentParser(description="Translate Japanese comments to English")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Single file to translate")
    group.add_argument("--dir", type=str, help="Directory to translate recursively")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")

    args = parser.parse_args()

    files_to_process = []

    if args.file:
        files_to_process.append(Path(args.file))
    else:
        dir_path = Path(args.dir)
        files_to_process = list(dir_path.rglob("*.py"))

    print("=" * 80)
    print("Comment Translation")
    print("=" * 80)
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Files to process: {len(files_to_process)}")
    print("=" * 80)
    print()

    total_files = 0
    total_comments_all = 0
    total_translated_all = 0

    for file_path in files_to_process:
        if file_path.name == "translate_comments.py":
            continue  # Skip this script itself

        total_comments, translated_comments = process_file(file_path, args.dry_run)

        if translated_comments > 0:
            status = "[DRY-RUN]" if args.dry_run else "[TRANSLATED]"
            print(f"{status} {file_path}: {translated_comments}/{total_comments} comments")
            total_files += 1
            total_comments_all += total_comments
            total_translated_all += translated_comments

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Files processed: {total_files}")
    print(f"Comments translated: {total_translated_all}/{total_comments_all}")

    if args.dry_run:
        print("\nThis was a DRY-RUN. Run without --dry-run to apply changes.")
    else:
        print("\nTranslation complete!")
        print("\nNote: Some comments may be marked with '# TODO: Manual translation needed'")
        print("Please review these manually.")

    print("=" * 80)


if __name__ == "__main__":
    main()
