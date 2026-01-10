#!/usr/bin/env python3
"""
Google Drive 同期スクリプト

ローカルの training_data/ フォルダをGoogle Driveに同期します。

使い方:
    # Google Drive デスクトップアプリがインストール済みの場合 # TODO: Manual translation needed
    python scripts/sync_to_gdrive.py --check
    python scripts/sync_to_gdrive.py --sync-all
    python scripts/sync_to_gdrive.py --sync-new

前提条件:
    - Google Drive デスクトップアプリがインストールされている
    - マイドライブ/virtual-robot-race/training_data/ フォルダが作成されている
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ===========================
# PathSettings
# ===========================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Project_Beta/
ROBOT1_ROOT = PROJECT_ROOT / "Robot1"
LOCAL_TRAINING_DATA = ROBOT1_ROOT / "training_data"

# Google Drive のPath（環境に応じて変更してください） # TODO: Manual translation needed
# Windows: C:\Users\[ユーザー名]\Google Drive\マイドライブ\
# Mac: ~/Google Drive/マイドライブ/
GDRIVE_PATHS = [
    Path(os.path.expanduser("~")) / "Google Drive" / "マイドライブ" / "virtual-robot-race" / "training_data",
    Path(os.path.expanduser("~")) / "Google Drive" / "My Drive" / "virtual-robot-race" / "training_data",
    Path(os.path.expanduser("~")) / "GoogleDrive" / "マイドライブ" / "virtual-robot-race" / "training_data",
    # カスタムPathがあればAdd # TODO: Manual translation needed
]

def find_gdrive_path():
    """Google Drive の training_data パスを検出"""
    for path in GDRIVE_PATHS:
        if path.exists():
            return path

    return None

def get_run_folders(root_path):
    """run_ フォルダのリストを取得"""
    if not root_path.exists():
        return []

    return sorted([
        f for f in root_path.iterdir()
        if f.is_dir() and f.name.startswith('run_')
    ])

def get_run_info(run_folder):
    """run_ フォルダの情報を取得"""
    csv_file = run_folder / "sensor_data.csv"
    if not csv_file.exists():
        return None

    # File数とサイズ # TODO: Manual translation needed
    num_files = sum(1 for _ in run_folder.iterdir())
    total_size = sum(f.stat().st_size for f in run_folder.rglob('*') if f.is_file())

    return {
        'name': run_folder.name,
        'num_files': num_files,
        'size_mb': total_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(run_folder.stat().st_mtime)
    }

def check_status():
    """ローカルとGoogle Driveの同期状態をチェック"""
    print("=" * 80)
    print("Google Drive 同期状態チェック")
    print("=" * 80)

    # Google Drive パス検出
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        print("\n⚠️ Google Drive が見つかりません")
        print("\n確認事項:")
        print("  1. Google Drive デスクトップアプリがインストールされているか")
        print("  2. マイドライブ/virtual-robot-race/training_data/ フォルダが作成されているか")
        print("\n検索したパス:")
        for path in GDRIVE_PATHS:
            print(f"  - {path}")
        return False

    print(f"\n✓ Google Drive 検出: {gdrive_path}")

    # ローカルのrun_Get # TODO: Manual translation needed
    local_runs = get_run_folders(LOCAL_TRAINING_DATA)
    print(f"\n📂 ローカル: {len(local_runs)} 個の run_ フォルダ")

    # Google Driveのrun_取得
    gdrive_runs = get_run_folders(gdrive_path)
    print(f"☁️  Google Drive: {len(gdrive_runs)} 個の run_ フォルダ")

    # 差分Check # TODO: Manual translation needed
    local_names = set(f.name for f in local_runs)
    gdrive_names = set(f.name for f in gdrive_runs)

    new_in_local = local_names - gdrive_names
    new_in_gdrive = gdrive_names - local_names
    common = local_names & gdrive_names

    print("\n" + "-" * 80)

    if new_in_local:
        print(f"\n📤 ローカルのみに存在（アップロード候補）: {len(new_in_local)} 個")
        for name in sorted(new_in_local):
            run_folder = LOCAL_TRAINING_DATA / name
            info = get_run_info(run_folder)
            if info:
                print(f"  - {name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")
    else:
        print("\n✓ 全てのローカルデータが Google Drive に存在します")

    if new_in_gdrive:
        print(f"\n📥 Google Driveのみに存在: {len(new_in_gdrive)} 個")
        for name in sorted(new_in_gdrive):
            print(f"  - {name}")

    if common:
        print(f"\n🔄 両方に存在: {len(common)} 個")

    print("\n" + "=" * 80)
    return True

def sync_new_runs():
    """新しいrun_フォルダのみをGoogle Driveに同期"""
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        print("⚠️ Google Drive が見つかりません。--check で確認してください。")
        return False

    # ローカルとGoogle Driveのrun_Get # TODO: Manual translation needed
    local_runs = get_run_folders(LOCAL_TRAINING_DATA)
    gdrive_runs = get_run_folders(gdrive_path)

    local_names = set(f.name for f in local_runs)
    gdrive_names = set(f.name for f in gdrive_runs)

    new_runs = local_names - gdrive_names

    if not new_runs:
        print("✓ 同期が必要な新しい run_ フォルダはありません")
        return True

    print(f"\n📤 {len(new_runs)} 個の新しい run_ をアップロード中...")
    print("-" * 80)

    for run_name in sorted(new_runs):
        src = LOCAL_TRAINING_DATA / run_name
        dst = gdrive_path / run_name

        info = get_run_info(src)
        if info is None:
            print(f"⚠️  {run_name}: sensor_data.csv がありません。スキップします。")
            continue

        print(f"📂 {run_name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")

        try:
            shutil.copytree(src, dst)
            print(f"   ✓ アップロード完了")
        except Exception as e:
            print(f"   ⚠️ エラー: {e}")

    print("\n" + "=" * 80)
    print("✓ 同期完了")
    print("=" * 80)
    return True

def sync_all_runs(force=False):
    """全てのrun_フォルダをGoogle Driveに同期"""
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        print("⚠️ Google Drive が見つかりません。--check で確認してください。")
        return False

    local_runs = get_run_folders(LOCAL_TRAINING_DATA)

    if not local_runs:
        print("⚠️ ローカルに run_ フォルダがありません")
        return False

    print(f"\n📤 {len(local_runs)} 個の run_ を同期中...")

    if not force:
        print("\n⚠️ 警告: 既存のフォルダは上書きされます。")
        confirm = input("続行しますか？ (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("キャンセルしました")
            return False

    print("-" * 80)

    for run_folder in local_runs:
        run_name = run_folder.name
        src = run_folder
        dst = gdrive_path / run_name

        info = get_run_info(src)
        if info is None:
            print(f"⚠️  {run_name}: sensor_data.csv がありません。スキップします。")
            continue

        print(f"📂 {run_name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")

        try:
            if dst.exists():
                shutil.rmtree(dst)
                print(f"   🗑️  既存フォルダを削除")

            shutil.copytree(src, dst)
            print(f"   ✓ アップロード完了")
        except Exception as e:
            print(f"   ⚠️ エラー: {e}")

    print("\n" + "=" * 80)
    print("✓ 全ての run_ を同期完了")
    print("=" * 80)
    return True

def setup_gdrive_structure():
    """Google Drive上に必要なフォルダ構造を作成"""
    gdrive_base = find_gdrive_path()

    if gdrive_base is None:
        print("⚠️ Google Drive が見つかりません")
        print("\n手動で以下のフォルダを作成してください:")
        print("  マイドライブ/virtual-robot-race/training_data/")
        return False

    # 親FolderのPath # TODO: Manual translation needed
    gdrive_root = gdrive_base.parent  # virtual-robot-race/

    print(f"\n✓ Google Drive 検出: {gdrive_root}")

    # 必要なFolder # TODO: Manual translation needed
    folders_to_create = [
        gdrive_root / "training_data",
        gdrive_root / "experiments",
        gdrive_root / "experiments" / "iterations",
    ]

    print("\n📁 フォルダ構造を作成中...")
    for folder in folders_to_create:
        if folder.exists():
            print(f"  ✓ {folder.relative_to(gdrive_root.parent)} (already exists)")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {folder.relative_to(gdrive_root.parent)} (created)")

    print("\n✓ Google Drive のフォルダ構造セットアップ完了")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Google Drive 同期スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/sync_to_gdrive.py --check          # 同期StateCheck # TODO: Manual translation needed
    python scripts/sync_to_gdrive.py --sync-new       # 新しいrun_のみ同期 # TODO: Manual translation needed
    python scripts/sync_to_gdrive.py --sync-all       # 全て同期（上書き） # TODO: Manual translation needed
    python scripts/sync_to_gdrive.py --setup          # Folder構造Create # TODO: Manual translation needed
        """
    )

    parser.add_argument('--check', action='store_true',
                        help='ローカルとGoogle Driveの同期状態をチェック')
    parser.add_argument('--sync-new', action='store_true',
                        help='新しいrun_フォルダのみをGoogle Driveに同期')
    parser.add_argument('--sync-all', action='store_true',
                        help='全てのrun_フォルダをGoogle Driveに同期（上書き）')
    parser.add_argument('--setup', action='store_true',
                        help='Google Drive上にフォルダ構造を作成')
    parser.add_argument('--force', action='store_true',
                        help='確認なしで実行（--sync-all用）')

    args = parser.parse_args()

    # 引数が何も指定されていない場合はヘルプ表示 # TODO: Manual translation needed
    if not (args.check or args.sync_new or args.sync_all or args.setup):
        parser.print_help()
        return

    # ローカルのtraining_dataFolderCheck # TODO: Manual translation needed
    if not args.setup and not LOCAL_TRAINING_DATA.exists():
        print(f"⚠️ エラー: ローカルの training_data フォルダが見つかりません")
        print(f"   パス: {LOCAL_TRAINING_DATA}")
        sys.exit(1)

    # コマンドExecute # TODO: Manual translation needed
    if args.setup:
        setup_gdrive_structure()

    if args.check:
        check_status()

    if args.sync_new:
        sync_new_runs()

    if args.sync_all:
        sync_all_runs(force=args.force)

if __name__ == "__main__":
    main()
