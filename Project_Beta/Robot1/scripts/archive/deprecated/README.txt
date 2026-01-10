# Deprecated Scripts Archive (Robot1/scripts)

This directory contains old/deprecated scripts kept for reference only.

## Contents

- `fix_metadata_filenames.py` - Old metadata correction tool
  - Replaced by: Project_Beta/scripts/data_manager_post.py (now with auto-rename in main.py)
  - Date: Pre-Beta 1.2

- `rename_images_to_match_csv.py` - Old image renaming tool
  - Replaced by: Project_Beta/scripts/data_manager_post.py (now with auto-rename in main.py)
  - Date: Pre-Beta 1.2

- `verify_data_integrity.py` - Data integrity verification tool
  - Purpose: Debug/verification (one-time use)
  - Date: 2026-01-10

- `verify_image_csv_alignment.py` - Image-CSV alignment checker
  - Purpose: Debug/verification (one-time use)
  - Date: 2026-01-10

## Note

These files are not used in the current system.
They are kept for historical reference and can be safely deleted if needed.

The current workflow uses:
- **Data collection**: Automatic rename built into main.py
- **Manual post-processing**: Project_Beta/scripts/data_manager_post.py
