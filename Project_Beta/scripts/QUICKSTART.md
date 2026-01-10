# Quick Start Guide - Training Data Fix

**For Beta 1.2 Users**

---

## 🚀 5-Minute Fix Guide

### Step 1: Verify Your Data (Dry-Run)

```bash
cd Project_Beta

# Check Robot1's latest training data
python scripts/data_manager_post.py --robot 1 --all
```

**Expected Output:**
```
Processing: Robot1\training_data\run_20260107_193739
  Metadata rows: 508
  Image files:   508
  Files needing rename: 508

  Sample changes:
    frame_000001.jpg -> frame_000329.jpg
    ...

[DRY RUN] No changes made (use --apply to fix)
```

### Step 2: Apply the Fix

```bash
# Fix all runs for Robot1
python scripts/data_manager_post.py --robot 1 --all --apply
```

**Expected Output:**
```
[BACKUP] Creating backup...
  Backup saved: images_backup/
[RENAME] Phase 1: Temporary names...
[RENAME] Phase 2: Final names...
[VERIFY] Checking all files...
[SUCCESS] 508 files renamed successfully!
```

### Step 3: Train Your Model

```bash
cd Robot1

# Train with fixed data
python train_model.py --data training_data_combined --epochs 50
```

### Step 4: Test in AI Mode

```bash
# Run AI mode
python main.py
```

---

## 🎯 What This Fixes

**Before Fix:**
- ❌ Model learns from wrong image-label pairs
- ❌ Cannot complete 2 laps
- ❌ Training loss doesn't improve
- ❌ AI drives randomly

**After Fix:**
- ✅ Model learns from correct image-label pairs
- ✅ Training loss decreases properly
- ✅ AI learns actual driving behavior
- ✅ Higher chance of completing 2 laps

---

## 🔍 Common Questions

### Q: Will this delete my data?
**A:** No! The tool creates a backup (`images_backup/`) before any changes.

### Q: Can I undo the fix?
**A:** Yes! Just delete `images/` and rename `images_backup/` back to `images/`.

### Q: What if counts don't match?
**A:** The tool will refuse to process that run. It's corrupted and should be discarded.

### Q: Do I need to fix every time?
**A:** Yes, after every new keyboard recording session. Or use `--all` periodically.

---

## 📞 Need Help?

See full documentation: [README_fix_training_data.md](README_fix_training_data.md)
