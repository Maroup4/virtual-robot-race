#!/usr/bin/env python3
"""
Google Drive Sync Script

Syncs the local training_data/ folder to Google Drive.

Usage:
    # If the Google Drive desktop app is installed
    python scripts/sync_to_gdrive.py --check
    python scripts/sync_to_gdrive.py --sync-all
    python scripts/sync_to_gdrive.py --sync-new

Prerequisites:
    - Google Drive desktop app is installed
    - My Drive/virtual-robot-race/training_data/ folder has been created
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # repo root
ROBOT1_ROOT = PROJECT_ROOT / "Robot1"
LOCAL_TRAINING_DATA = ROBOT1_ROOT / "training_data"

# ===========================
# Google Drive Root detection
# ===========================
# Folder names that Google Drive uses for "My Drive"
GDRIVE_FOLDER_NAMES = ["マイドライブ", "My Drive", "MyDrive"]

# Drive letters to check (new Google Drive desktop app mounts as a virtual drive)
GDRIVE_DRIVE_LETTERS = ["G", "H", "I", "J"]

# Legacy paths (older Google Drive app synced to a folder)
GDRIVE_LEGACY_ROOTS = [
    Path(os.path.expanduser("~")) / "Google Drive",
    Path(os.path.expanduser("~")) / "GoogleDrive",
    Path(os.path.expanduser("~")) / "Google ドライブ",
]

PROJECT_SUBPATH = "virtual-robot-race"
TRAINING_SUBPATH = "training_data"


def find_gdrive_root():
    """
    Find the Google Drive 'My Drive' root folder.
    Supports both virtual drive letters (new app) and legacy folder paths.
    Returns Path to the My Drive root, or None if not found.
    """
    # 1. Check virtual drive letters (new Google Drive desktop app)
    for letter in GDRIVE_DRIVE_LETTERS:
        drive = Path(f"{letter}:\\")
        if drive.exists():
            for folder_name in GDRIVE_FOLDER_NAMES:
                candidate = drive / folder_name
                if candidate.exists():
                    return candidate
            # Drive exists but no named subfolder — might be the root itself
            # Check for .shortcut-targets-by-id which Google Drive creates
            if (drive / ".shortcut-targets-by-id").exists():
                # The drive root IS My Drive
                return drive

    # 2. Check legacy paths
    for root in GDRIVE_LEGACY_ROOTS:
        if root.exists():
            for folder_name in GDRIVE_FOLDER_NAMES:
                candidate = root / folder_name
                if candidate.exists():
                    return candidate
            # Some versions put files directly in "Google Drive/"
            if root.exists():
                return root

    return None


def find_gdrive_path():
    """
    Find the virtual-robot-race/training_data path on Google Drive.
    Returns Path if the folder already exists, None otherwise.
    """
    gdrive_root = find_gdrive_root()
    if gdrive_root is None:
        return None

    candidate = gdrive_root / PROJECT_SUBPATH / TRAINING_SUBPATH
    return candidate if candidate.exists() else None

def get_run_folders(root_path):
    """Get a list of run_ folders"""
    if not root_path.exists():
        return []

    return sorted([
        f for f in root_path.iterdir()
        if f.is_dir() and f.name.startswith('run_')
    ])

def get_run_info(run_folder):
    """Get information about a run_ folder"""
    csv_file = run_folder / "metadata.csv"
    if not csv_file.exists():
        return None

    # File count and size
    num_files = sum(1 for _ in run_folder.iterdir())
    total_size = sum(f.stat().st_size for f in run_folder.rglob('*') if f.is_file())

    return {
        'name': run_folder.name,
        'num_files': num_files,
        'size_mb': total_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(run_folder.stat().st_mtime)
    }

def check_status():
    """Check the sync status between local and Google Drive"""
    print("=" * 80)
    print("Google Drive Sync Status Check")
    print("=" * 80)

    # Detect Google Drive path
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        gdrive_root = find_gdrive_root()
        if gdrive_root is None:
            print("\nWarning: Google Drive not found")
            print("\nPlease verify:")
            print("  1. Google Drive desktop app is installed and running")
            print("  2. You are signed in")
            print("\nIf Google Drive is mounted as a drive letter (e.g. G:), this script")
            print("will detect it automatically on the next run.")
        else:
            print(f"\nGoogle Drive found at: {gdrive_root}")
            print(f"\nBut the project folder does not exist yet:")
            print(f"  {gdrive_root / PROJECT_SUBPATH / TRAINING_SUBPATH}")
            print(f"\nRun --setup to create it automatically:")
            print(f"  python scripts/sync_to_gdrive.py --setup")
        return False

    print(f"\nGoogle Drive detected: {gdrive_path}")

    # Get local run_ folders
    local_runs = get_run_folders(LOCAL_TRAINING_DATA)
    print(f"\nLocal: {len(local_runs)} run_ folder(s)")

    # Get Google Drive run_ folders
    gdrive_runs = get_run_folders(gdrive_path)
    print(f"Google Drive: {len(gdrive_runs)} run_ folder(s)")

    # Check differences
    local_names = set(f.name for f in local_runs)
    gdrive_names = set(f.name for f in gdrive_runs)

    new_in_local = local_names - gdrive_names
    new_in_gdrive = gdrive_names - local_names
    common = local_names & gdrive_names

    print("\n" + "-" * 80)

    if new_in_local:
        print(f"\nExists only locally (upload candidates): {len(new_in_local)} item(s)")
        for name in sorted(new_in_local):
            run_folder = LOCAL_TRAINING_DATA / name
            info = get_run_info(run_folder)
            if info:
                print(f"  - {name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")
    else:
        print("\nAll local data exists on Google Drive")

    if new_in_gdrive:
        print(f"\nExists only on Google Drive: {len(new_in_gdrive)} item(s)")
        for name in sorted(new_in_gdrive):
            print(f"  - {name}")

    if common:
        print(f"\nExists in both: {len(common)} item(s)")

    print("\n" + "=" * 80)
    return True

def sync_new_runs():
    """Sync only new run_ folders to Google Drive"""
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        print("Warning: Google Drive not found. Please verify with --check.")
        return False

    # Get local and Google Drive run_ folders
    local_runs = get_run_folders(LOCAL_TRAINING_DATA)
    gdrive_runs = get_run_folders(gdrive_path)

    local_names = set(f.name for f in local_runs)
    gdrive_names = set(f.name for f in gdrive_runs)

    new_runs = local_names - gdrive_names

    if not new_runs:
        print("No new run_ folders need syncing")
        return True

    print(f"\nUploading {len(new_runs)} new run_ folder(s)...")
    print("-" * 80)

    for run_name in sorted(new_runs):
        src = LOCAL_TRAINING_DATA / run_name
        dst = gdrive_path / run_name

        info = get_run_info(src)
        if info is None:
            print(f"Warning: {run_name}: metadata.csv not found. Skipping.")
            continue

        print(f"{run_name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")

        try:
            shutil.copytree(src, dst)
            print(f"   Upload complete")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("Sync complete")
    print("=" * 80)
    return True

def sync_all_runs(force=False):
    """Sync all run_ folders to Google Drive"""
    gdrive_path = find_gdrive_path()

    if gdrive_path is None:
        print("Warning: Google Drive not found. Please verify with --check.")
        return False

    local_runs = get_run_folders(LOCAL_TRAINING_DATA)

    if not local_runs:
        print("Warning: No run_ folders found locally")
        return False

    print(f"\nSyncing {len(local_runs)} run_ folder(s)...")

    if not force:
        print("\nWarning: Existing folders will be overwritten.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return False

    print("-" * 80)

    for run_folder in local_runs:
        run_name = run_folder.name
        src = run_folder
        dst = gdrive_path / run_name

        info = get_run_info(src)
        if info is None:
            print(f"Warning: {run_name}: metadata.csv not found. Skipping.")
            continue

        print(f"{run_name} ({info['size_mb']:.1f} MB, {info['num_files']} files)")

        try:
            if dst.exists():
                shutil.rmtree(dst)
                print(f"   Deleted existing folder")

            shutil.copytree(src, dst)
            print(f"   Upload complete")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("All run_ folders synced")
    print("=" * 80)
    return True

def setup_gdrive_structure():
    """Create the required folder structure on Google Drive"""
    gdrive_my_drive = find_gdrive_root()

    if gdrive_my_drive is None:
        print("Warning: Google Drive (My Drive) not found")
        print("\nPlease verify:")
        print("  1. Google Drive desktop app is installed and running")
        print("  2. You are signed in to Google Drive")
        print("\nIf Google Drive is mounted as a drive letter (e.g. G:), check:")
        for letter in GDRIVE_DRIVE_LETTERS:
            print(f"  - {letter}:\\マイドライブ  or  {letter}:\\My Drive")
        return False

    project_root = gdrive_my_drive / PROJECT_SUBPATH
    print(f"\nGoogle Drive (My Drive) detected: {gdrive_my_drive}")
    print(f"Will create folder structure under: {project_root}")

    folders_to_create = [
        project_root / "training_data",
        project_root / "experiments",
    ]

    print("\nCreating folder structure...")
    for folder in folders_to_create:
        if folder.exists():
            print(f"  {folder}  (already exists)")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  {folder}  (created)")

    # Copy Colab notebook
    notebook_src = PROJECT_ROOT / "colab" / "train_on_colab.ipynb"
    notebook_dst = project_root / "train_on_colab.ipynb"
    if notebook_src.exists():
        if notebook_dst.exists():
            print(f"  {notebook_dst}  (already exists, skipped)")
        else:
            import shutil as _shutil
            _shutil.copy2(notebook_src, notebook_dst)
            print(f"  {notebook_dst}  (copied)")
    else:
        print(f"  Warning: Colab notebook not found at {notebook_src}")

    print("\nSetup complete. You can now run --check to verify.")
    print("Open train_on_colab.ipynb in Google Drive to start training on Colab.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Google Drive Sync Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/sync_to_gdrive.py --check          # Check sync status
    python scripts/sync_to_gdrive.py --sync-new       # Sync only new run_ folders
    python scripts/sync_to_gdrive.py --sync-all       # Sync all (overwrite)
    python scripts/sync_to_gdrive.py --setup          # Create folder structure
        """
    )

    parser.add_argument('--check', action='store_true',
                        help='Check sync status between local and Google Drive')
    parser.add_argument('--sync-new', action='store_true',
                        help='Sync only new run_ folders to Google Drive')
    parser.add_argument('--sync-all', action='store_true',
                        help='Sync all run_ folders to Google Drive (overwrite)')
    parser.add_argument('--setup', action='store_true',
                        help='Create folder structure on Google Drive')
    parser.add_argument('--force', action='store_true',
                        help='Execute without confirmation (for --sync-all)')

    args = parser.parse_args()

    # Show help if no arguments are provided
    if not (args.check or args.sync_new or args.sync_all or args.setup):
        parser.print_help()
        return

    # Check local training_data folder
    if not args.setup and not LOCAL_TRAINING_DATA.exists():
        print(f"Error: Local training_data folder not found")
        print(f"   Path: {LOCAL_TRAINING_DATA}")
        sys.exit(1)

    # Execute command
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
