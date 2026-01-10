# Training Data Filename Correction Tool

**Version:** Beta 1.2
**Status:** Official Release
**Date:** 2026-01-10

---

## 📋 Overview

This tool fixes a critical issue in Virtual Robot Race's training data collection system where image filenames saved by Python do not match the filenames recorded in Unity's `metadata.csv`.

### The Problem

**Unity Side (DataLogger.cs):**
- Uses `tickIndex` from TickScheduler for filename
- Example: `tick=329` → `filename="frame_000329.jpg"`
- Records this in `metadata.csv`

**Python Side (websocket_client.py):**
- Uses independent counter starting from 1
- Example: 1st image received → saves as `"frame_000001.jpg"`

**Result:**
- ❌ `metadata.csv` says `frame_000329.jpg` but actual file is `frame_000001.jpg`
- ❌ 328-frame offset causing complete training data corruption
- ❌ Model learns from **completely wrong image-label pairs**
- ❌ This is why the robot cannot complete 2 laps

### The Solution

**Sequential Renaming:**
- Verified: `metadata.csv` row count = JPG file count (always matches)
- Verified: Image reception order = metadata recording order
- Solution: Rename images sequentially to match `metadata.csv` filenames
- Safe: Creates backup before any modification

---

## 🚀 Quick Start

### 1. Verify a Single Run (Recommended First Step)

```bash
# Check what would be fixed (dry-run, no changes)
python scripts/data_manager_post.py --robot 1 --run run_20260107_193739
```

### 2. Fix a Single Run

```bash
# Apply fixes with backup
python scripts/data_manager_post.py --robot 1 --run run_20260107_193739 --apply
```

### 3. Fix All Runs for One Robot

```bash
# Fix all runs in training_data/, training_data_combined/, training_data_augmented/
python scripts/data_manager_post.py --robot 1 --all --apply
```

### 4. Fix All Runs for All Robots

```bash
# Process Robot1 through Robot5
python scripts/data_manager_post.py --all-robots --all --apply
```

---

## 📖 Detailed Usage

### Command Structure

```bash
python scripts/data_manager_post.py [OPTIONS]
```

### Required Options (Choose One Set)

**Robot Selection:**
- `--robot N` - Process specific robot (N = 1-5)
- `--all-robots` - Process all robots (Robot1-5)

**Run Selection:**
- `--run RUN_NAME` - Process specific run (e.g., `run_20260107_193739`)
- `--all` - Process all runs in training data directories

### Optional Flags

- `--apply` - Actually apply fixes (default is dry-run for safety)

### Examples

#### Example 1: Check Before Fixing
```bash
# Always verify first (dry-run)
python scripts/data_manager_post.py --robot 1 --run run_20260107_193739

# Output will show:
#   - Metadata row count
#   - Image file count
#   - Sample of proposed changes
#   - Whether counts match
```

#### Example 2: Fix Single Run
```bash
# After verifying, apply the fix
python scripts/data_manager_post.py --robot 1 --run run_20260107_193739 --apply

# Creates backup at: Robot1/training_data/run_YYYYMMDD_HHMMSS/images_backup/
```

#### Example 3: Fix All Training Data for Robot1
```bash
# Fix all runs in training_data/, training_data_combined/, training_data_augmented/
python scripts/data_manager_post.py --robot 1 --all --apply
```

#### Example 4: Fix Everything
```bash
# Fix all runs for all robots (use with caution!)
python scripts/data_manager_post.py --all-robots --all --apply
```

---

## 🔍 What the Tool Does

### Verification Phase (Dry-Run)

For each run directory:

1. **Load `metadata.csv`**
   - Extract expected filenames from `filename` column

2. **Scan `images/` directory**
   - Get list of actual `.jpg` files (sorted)

3. **Verify Counts Match**
   - If `metadata rows ≠ image count` → ERROR (cannot fix safely)
   - If counts match → proceed

4. **Build Rename Mapping**
   - Map `frame_000001.jpg` → `frame_000329.jpg`
   - Map `frame_000002.jpg` → `frame_000330.jpg`
   - ... (sequential mapping)

5. **Show Sample Changes**
   - Display first 3 and last 3 proposed renames

### Fix Phase (--apply)

When you add `--apply` flag:

1. **Create Backup**
   - Copy entire `images/` folder to `images_backup/`
   - If backup already exists → abort (prevent accidental overwrite)

2. **Two-Phase Renaming**
   - Phase 1: Rename all to temporary names (`.tmp.jpg`)
   - Phase 2: Rename to final names
   - (This avoids filename conflicts)

3. **Verification**
   - Check all expected filenames now exist
   - Report success or failure

4. **Statistics**
   - Total runs processed
   - Fixed / Already correct / Failed

---

## ⚠️ Safety Features

### Backup System
- **Automatic backup** created before any modification
- Backup location: `run_YYYYMMDD_HHMMSS/images_backup/`
- If backup already exists, tool refuses to run (prevents data loss)

### Dry-Run by Default
- Default mode is **verification only** (no changes)
- Must explicitly use `--apply` to modify files
- Always shows what would change before applying

### Validation Checks
- ✅ Verifies `metadata.csv` exists
- ✅ Verifies `images/` directory exists
- ✅ Verifies row count = image count
- ✅ Verifies all expected files exist after rename
- ✅ Prevents processing if counts don't match

---

## 📊 Output Interpretation

### Success Case (Already Correct)
```
[OK] All filenames already match! No fix needed.
```
→ This run doesn't need fixing

### Success Case (Needs Fixing)
```
Files needing rename: 508

Sample changes:
  frame_000001.jpg -> frame_000329.jpg
  frame_000002.jpg -> frame_000330.jpg
  ...

[DRY RUN] No changes made (use --apply to fix)
```
→ Run with `--apply` to fix

### Error Case
```
[ERROR] Count mismatch (diff: 10)
  Metadata rows: 508
  Image files:   498
```
→ Cannot fix safely (data corruption or incomplete recording)

---

## 🔧 Troubleshooting

### Issue: "Backup directory already exists"

**Cause:** Previous fix attempt left a backup folder

**Solution:**
```bash
# Check the backup
cd Robot1/training_data/run_YYYYMMDD_HHMMSS/
ls images_backup/

# If backup is valid, remove it
rm -rf images_backup/

# Or rename it
mv images_backup images_backup_old
```

### Issue: "Count mismatch"

**Cause:** Incomplete data recording or corrupted run

**Solution:**
- Cannot fix automatically
- Either:
  - Discard this run (delete the directory)
  - Manually investigate why counts don't match

### Issue: "metadata.csv not found"

**Cause:** Run directory is incomplete or corrupted

**Solution:**
- This run cannot be used for training
- Delete or move to a separate folder

---

## 📁 File Structure After Fix

```
Robot1/
└── training_data/
    └── run_20260107_193739/
        ├── images/                    # ✅ Fixed filenames
        │   ├── frame_000329.jpg       # (was frame_000001.jpg)
        │   ├── frame_000330.jpg       # (was frame_000002.jpg)
        │   └── ...
        ├── images_backup/             # 🆕 Original files (backup)
        │   ├── frame_000001.jpg
        │   ├── frame_000002.jpg
        │   └── ...
        ├── metadata.csv               # Unchanged
        ├── terminal_log.txt           # Unchanged
        └── unity_log.txt              # Unchanged
```

---

## 🎯 When to Use This Tool

### Required Scenarios

1. **After Manual Keyboard Recording**
   - Every time you record new training data in Keyboard mode
   - Before using the data for training

2. **After Discovering Filename Mismatch**
   - If you suspect training data corruption
   - If model training shows no improvement

3. **Before Training a New Model**
   - Always verify and fix training data first
   - Ensures clean, aligned data

### Optional Scenarios

1. **Periodic Maintenance**
   - Run on all existing training data to ensure consistency

2. **Before Sharing Data**
   - Fix data before sharing with collaborators

---

## 📝 Integration with Training Workflow

### Recommended Workflow

```bash
# 1. Record training data (Keyboard mode in Unity)
#    → Creates: Robot1/training_data/run_YYYYMMDD_HHMMSS/

# 2. Verify the new run
python scripts/data_manager_post.py --robot 1 --run run_YYYYMMDD_HHMMSS

# 3. Fix the new run
python scripts/data_manager_post.py --robot 1 --run run_YYYYMMDD_HHMMSS --apply

# 4. Train the model
cd Robot1
python train_model.py --data training_data_combined --epochs 50

# 5. Test in AI mode
python main.py
```

---

## 🔬 Technical Details

### Why This Works

**Assumption 1: Order Preservation**
- Unity sends images in tick order
- Python receives images in the same order
- Sequential renaming preserves this order

**Assumption 2: No Dropped Frames**
- If `metadata rows = image count`, no frames were lost
- Tool validates this before proceeding

**Assumption 3: Unique Mapping**
- Each image has exactly one corresponding metadata row
- Sequential mapping is deterministic

### Limitations

- Cannot fix runs where `metadata count ≠ image count`
- Cannot recover if images were actually dropped during transmission
- Assumes tick recording started before image recording

---

## 🐛 Known Issues

### None Currently Identified

If you discover any issues, please report them with:
- Command used
- Output/error message
- Run directory structure (`ls -la`)

---

## 📜 Version History

### Beta 1.2 (2026-01-10)
- Initial release
- Supports all 5 robots
- Handles multiple training data directories
- Automatic backup system
- Dry-run by default

---

## 👥 Credits

**Problem Discovery:** User analysis of training data integrity
**Root Cause Analysis:** Comparison of Unity DataLogger and Python websocket_client
**Solution Design:** Sequential renaming approach
**Implementation:** Beta 1.2 official tool

---

## 📞 Support

For issues or questions:
1. Check this README first
2. Verify your command syntax
3. Check output messages for specific errors
4. Report issues with full context (command + output)

---

**Remember:** Always run without `--apply` first to verify changes are correct!
