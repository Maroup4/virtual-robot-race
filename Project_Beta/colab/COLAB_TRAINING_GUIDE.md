# Google Colab Training Guide

This guide explains how to train Virtual Robot Race AI models on Google Colab.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Drive Structure](#google-drive-structure)
3. [Workflow](#workflow)
4. [Troubleshooting](#troubleshooting)
5. [Differences from Local Training](#differences-from-local-training)

---

## Prerequisites

### 1. Google Drive Setup

#### Method A: Google Drive Desktop App (Recommended)

1. Install [Google Drive Desktop App](https://www.google.com/drive/download/)
2. Sign in with your Google account
3. Access the `My Drive` folder on your local PC

**Create folder structure:**
```
C:\Users\[username]\Google Drive\My Drive\
└── virtual-robot-race\
    └── training_data\
```

**Upload data:**
- Copy `run_` folders from local `Robot1/training_data/` to the above folder
- Google Drive app will automatically sync

#### Method B: Manual Upload via Web Browser

1. Go to [Google Drive](https://drive.google.com/)
2. Right-click → "New folder" → create `virtual-robot-race` folder
3. Create `training_data` folder inside it
4. Drag & drop `run_` folders to upload

**Note:** Method A is faster for uploading many folders.

### 2. Colab Notebook Setup

1. Upload `colab/train_on_colab.ipynb` to Google Drive
   - Upload destination: `/My Drive/virtual-robot-race/train_on_colab.ipynb`

2. Double-click the file in Google Drive → "Open with Google Colaboratory"

3. **Important:** Runtime → Change runtime type → Select **GPU**
   - T4 GPU (Free): ~15GB memory
   - A100 GPU (Colab Pro): ~40GB memory

---

## Google Drive Structure

Final folder structure should look like this:

```
/My Drive/
└── virtual-robot-race/
    ├── train_on_colab.ipynb           # Training notebook
    ├── training_data/                 # Training data pool
    │   ├── run_20260110_082611/
    │   │   ├── images/
    │   │   │   ├── frame_000045.jpg
    │   │   │   ├── frame_000046.jpg
    │   │   │   └── ...
    │   │   ├── metadata.csv
    │   │   ├── output_video.mp4
    │   │   ├── terminal_log.txt
    │   │   └── unity_log.txt
    │   ├── run_20260110_082524/
    │   └── ...
    └── experiments/                   # Training results (auto-created by Colab)
        └── iteration_260110_090000/
            ├── data_sources/           # run_ folders used in this iteration
            │   ├── run_20260110_082611/
            │   └── run_20260110_082524/
            ├── logs/
            ├── evaluation/
            ├── README.md
            ├── training_config.yaml
            ├── model.pth               # ← Download this
            └── training_log.csv
```

**Folder roles:**
- `training_data/`: Store all training data (shared pool)
- `experiments/iterations/`: Save results of each training experiment
- `data_sources/`: Copy of run_ used in that experiment (ensures reproducibility)

---

## Workflow

### Step 1: Mount Google Drive (Cell 1)

```python
from google.colab import drive
drive.mount('/content/drive')
```

When executed, you'll be asked to authenticate:
1. Click the link
2. Sign in with your Google account
3. Copy & paste the authorization code

### Step 2: Install Libraries (Cell 2)

```python
# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
```

**Checkpoint:**
- GPU name like `CUDA available: Tesla T4` should be displayed
- If not shown → Check runtime settings (select GPU)

### Step 3: Create Iteration Folder (Cell 3)

```python
iteration_name = f"iteration_{timestamp}"
```

Automatically creates `iteration_YYMMDD_HHMMSS` folder.

### Step 4: Select and Copy Training Data (Cell 4)

**Available run_ folders will be listed:**
```
Available run_ folders (5):
--------------------------------------------------------------------------------
 1. run_20260105_074008 (342 samples)
 2. run_20260105_074416 (356 samples)
 3. run_20260105_074514 (349 samples)
 4. run_20260105_080000 (298 samples)
 5. run_20260105_081234 (412 samples)

Select:
```

**Selection methods:**
- Individual: `1,2,3`
- Range: `1-3`
- All: `all`

Selected run_ folders will be copied to `data_sources/`.

### Step 5: Generate Dataset Manifest (Cell 5)

```python
manifest = create_dataset_manifest(data_sources_path, manifest_path)
```

**Example output:**
```
Analyzing 3 run folders...

✓ COMPLETED run_20260105_074008: 342 samples, Lap 2, 45.2s
✓ COMPLETED run_20260105_074416: 356 samples, Lap 2, 43.8s
✓ COMPLETED run_20260105_074514: 349 samples, Lap 2, 44.5s

================================================================================
📊 Dataset Summary
================================================================================
Total runs: 3
  Completed: 3
  Incomplete: 0
Total samples: 1,047
```

### Step 6: Model Definition and Dataset Preparation (Cell 6)

```python
model = RobotRaceModel().to(device)
```

**Checkpoints:**
- `Using device: cuda` is displayed
- GPU name and memory capacity shown
- Dataset sample count displayed

### Step 7: Model Training (Cell 7)

Training starts. Progress is shown for each epoch:

```
Epoch  1/50 | Train Loss: 0.0234 MAE: 0.1123 | Val Loss: 0.0198 MAE: 0.0987 | 42.3s ✓ NEW BEST
Epoch  2/50 | Train Loss: 0.0187 MAE: 0.0945 | Val Loss: 0.0156 MAE: 0.0821 | 41.8s ✓ NEW BEST
...
Epoch 23/50 | Train Loss: 0.0089 MAE: 0.0512 | Val Loss: 0.0092 MAE: 0.0523 | 42.1s

⏹️  Early stopping triggered at epoch 23 (patience=10)

================================================================================
✓ Training complete!
================================================================================
Best validation loss: 0.0085
Model saved to: /content/drive/MyDrive/virtual-robot-race/experiments/iterations/iteration_260105_120000/model.pth
```

**Training time estimates:**
- T4 GPU (Free): 40-50 seconds per epoch (for 1,000 samples)
- A100 GPU (Pro): 20-30 seconds per epoch

### Step 8: Visualize Results (Cell 8)

Training curves are plotted:
- Left graph: Loss (MSE)
- Right graph: MAE (Mean Absolute Error)

**Signs of good training:**
- ✓ Both Train/Val are decreasing
- ✓ Small gap between Train/Val (no overfitting)
- ✓ Val continues decreasing trend

**Signs of bad training:**
- ⚠️ Val increases midway (overfitting)
- ⚠️ Large gap between Train/Val (poor generalization)

### Step 9: Download Model

1. Click folder icon in left sidebar
2. Navigate to `drive/MyDrive/virtual-robot-race/experiments/iterations/iteration_XXXXXX/`
3. Right-click `model.pth` → Download
4. Copy to local `Robot1/models/` folder

### Step 10: Test Inference Locally

```python
# Update MODEL_PATH in Robot1/inference_input.py
MODEL_PATH = r"D:\AARACE\GitProjects\virtual-robot-race\Project_Beta\Robot1\models\model.pth"
```

Test AI mode in Unity!

---

## Troubleshooting

### Issue 1: CUDA not available

**Symptom:**
```
⚠️ WARNING: CUDA not available. Runtime → Change runtime type → Select GPU
```

**Solution:**
1. Runtime → Change runtime type
2. Hardware accelerator → Select **GPU**
3. Save
4. Re-run from Cell 1

### Issue 2: training_data folder not found

**Symptom:**
```
⚠️ WARNING: /content/drive/MyDrive/virtual-robot-race/training_data not found
```

**Solution:**
1. Verify upload to Google Drive is correct
2. Check folder name is `virtual-robot-race` (with hyphen)
3. Confirm `training_data` folder exists

### Issue 3: Session timeout

**Symptom:**
Colab session disconnects during training (90 minute limit)

**Solution:**

**Method A (Recommended):** Subscribe to Colab Pro (~$10/month)
- Relaxed session limits
- Access to faster GPU (A100)

**Method B:** Resume from checkpoint
1. `checkpoint_epoch{N}.pth` is saved periodically in Cell 7
2. Add resume code (see next section)

**Resume from checkpoint (add to Cell 7):**
```python
# Check for checkpoints
checkpoint_files = sorted(iteration_path.glob("checkpoint_epoch*.pth"))
if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    print(f"Found checkpoint: {latest_checkpoint}")

    # Load
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0
```

### Issue 4: Out of Memory (OOM) error

**Symptom:**
```
RuntimeError: CUDA out of memory.
```

**Solution:**
1. Reduce `BATCH_SIZE` (Cell 6)
   ```python
   BATCH_SIZE = 16  # Changed from 32 to 16
   ```
2. Reduce image size
   ```python
   transforms.Resize((128, 128))  # Changed from 224 to 128
   ```
3. Set `num_workers=0` (Cell 6)

---

## Differences from Local Training

### Execution Environment

| Item | Local | Google Colab |
|------|-------|-------------|
| GPU | RTX 3080 | T4 / A100 |
| GPU Memory | 10GB | 15GB / 40GB |
| Runtime Limit | None | 90min (Free) / 24hr (Pro) |
| Data Storage | Local disk | Google Drive |
| Cost | Electricity only | Free / ~$10/month |

### Training Script Differences

**Local (train_model.py):**
- Single Python script
- Execute with command-line arguments
- Results saved to `iterations/`

**Colab (Notebook):**
- Split into multiple cells
- Execute step-by-step with verification
- Results saved to Google Drive

### Data Management Differences

**Local:**
```
Robot1/
├── training_data/
│   └── run_XXXXXX/
├── experiments/
│   └── iterations/
│       └── iteration_XXXXXX/
└── models/
    └── model.pth
```

**Colab:**
```
/content/drive/MyDrive/virtual-robot-race/
├── training_data/
│   └── run_XXXXXX/
└── experiments/
    └── iterations/
        └── iteration_XXXXXX/
            └── model.pth  # ← Download to local models/
```

---

## Additional Tips

### Tip 1: TensorBoard Integration

To monitor training in real-time:

```python
# Add to Cell 6
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=iteration_path / "tensorboard")

# Add to training loop in Cell 7
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

In a separate cell:
```python
%load_ext tensorboard
%tensorboard --logdir {iteration_path}/tensorboard
```

### Tip 2: Compare Multiple Models

Run multiple iterations with different hyperparameters and compare:

```python
# iteration_A: BATCH_SIZE=32, LR=0.001
# iteration_B: BATCH_SIZE=64, LR=0.0005
# iteration_C: Enhanced data augmentation
```

Compare `training_history.json` from each iteration to find the best settings.

### Tip 3: Save Google Drive Space

`run_` folders contain many images and consume storage:

**Method A:** Delete `data_sources/` from finished iterations
- Keep only `model.pth` and `training_history.json`

**Method B:** ZIP compress before upload
```bash
# Compress locally
cd Robot1/training_data
zip -r run_20260105_074008.zip run_20260105_074008/

# Unzip in Colab
!unzip /content/drive/MyDrive/virtual-robot-race/training_data/run_20260105_074008.zip
```

---

## Summary

By using Google Colab:
- ✅ Fast training possible without local GPU
- ✅ Save on electricity costs
- ✅ Access from anywhere
- ✅ Run multiple experiments in parallel (multiple tabs)

Next steps:
1. Initial setup (create Google Drive structure)
2. Run `train_on_colab.ipynb`
3. Download good models to local machine
4. Test in Unity!

Happy Training! 🚀
