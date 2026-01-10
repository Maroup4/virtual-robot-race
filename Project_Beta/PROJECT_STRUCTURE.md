# Virtual Robot Race - Project Structure

**Version:** Beta 1.2
**Last Updated:** 2026-01-10

---

## 📂 Directory Structure

```
Project_Beta/
│
├── scripts/                          # Project-wide tools & utilities
│   ├── data_manager_post.py          # Post-processing: Fix training data filenames
│   ├── README_fix_training_data.md   # Documentation for data_manager_post.py
│   └── QUICKSTART.md                 # 5-minute quick start guide
│
├── archive/                          # Deprecated/old files (for reference)
│   └── deprecated/
│       ├── train_model.py.old        # Old training script (use Robot1/train_model.py)
│       └── README.txt                # Explanation
│
├── main.py                           # Entry point (all robots)
├── data_manager.py                   # Real-time data collection (all robots)
├── websocket_client.py               # Unity communication (all robots)
├── config_loader.py                  # Configuration loader (all robots)
│
├── config.txt                        # Global configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview
├── PROJECT_STRUCTURE.md              # This file
│
├── Robot1/
│   ├── robot_config.txt              # Robot1-specific configuration
│   ├── model.py                      # Neural network model definition
│   ├── train_model.py                # Training script (LATEST VERSION)
│   │
│   ├── inference_input.py            # AI control mode
│   ├── keyboard_input.py             # Keyboard control mode
│   ├── rule_based_input.py           # Rule-based control mode
│   ├── table_input.py                # Table (CSV) control mode
│   │
│   ├── ai_control_strategy.py        # AI strategy settings
│   │
│   ├── training_data/                # Training data from keyboard mode
│   ├── training_data_combined/       # Combined + augmented data
│   ├── training_data_augmented/      # Augmented data (flipped)
│   ├── data_interactive/             # Real-time data for inference
│   ├── models/                       # Trained model files (.pth)
│   └── scripts/                      # Robot1-specific scripts (if any)
│
├── Robot2/ ... Robot5/               # Similar structure for other robots
│
└── Windows/                          # Unity executable
    └── VirtualRobotRace_Beta.exe
```

---

## 📝 File Responsibilities

### Top Level (Project_Beta/)

#### Entry & Core Systems

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `main.py` | Entry point | All robots | `python main.py` |
| `data_manager.py` | Real-time data collection during race | All robots | Runtime |
| `websocket_client.py` | Unity communication (images, telemetry) | All robots | Runtime |
| `config_loader.py` | Configuration file parser | All robots | Runtime |

#### Configuration

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `config.txt` | Global settings (DATA_SAVE, active_robots, etc.) | All robots | Edited by user |
| `requirements.txt` | Python dependencies | All robots | `pip install -r requirements.txt` |

---

### scripts/ (Project-Wide Tools)

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `data_manager_post.py` | Post-processing: Fix training data filename mismatches | All robots | After recording data |
| `README_fix_training_data.md` | Full documentation for data_manager_post.py | - | Reference |
| `QUICKSTART.md` | Quick start guide for data_manager_post.py | - | Quick reference |

**Usage Example:**
```bash
# Fix all training data for Robot1
python scripts/data_manager_post.py --robot 1 --all --apply
```

---

### Robot1/ (Robot-Specific)

#### Configuration

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `robot_config.txt` | Robot1 settings (mode, player_name, race_flag) | Robot1 only | Edited by user |

#### Model & Training

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `model.py` | Neural network architecture (DrivingNetwork) | Robot1 only | Imported by train/inference |
| `train_model.py` | Training script (LATEST VERSION) | Robot1 only | `cd Robot1 && python train_model.py` |

#### Control Modes

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `inference_input.py` | AI control logic | Robot1 only | When mode=AI |
| `keyboard_input.py` | Keyboard control logic | Robot1 only | When mode=keyboard |
| `rule_based_input.py` | Rule-based control logic | Robot1 only | When mode=rule_based |
| `table_input.py` | Table (CSV) control logic | Robot1 only | When mode=table |

#### AI Strategy

| File | Responsibility | Scope | Usage |
|------|---------------|-------|-------|
| `ai_control_strategy.py` | AI strategy settings (hybrid/pure_e2e, start boost, corner tuning) | Robot1 only | Runtime (imported by inference_input.py) |

---

## 🔄 Workflow

### 1. Recording Training Data (Keyboard Mode)

```bash
# 1. Configure Robot1
#    Edit Robot1/robot_config.txt:
#      mode = keyboard
#      player_name = YourName
#      race_flag = 1

# 2. Configure global settings
#    Edit config.txt:
#      DATA_SAVE = 1
#      active_robots = 1

# 3. Run the race
python main.py

# 4. Race with keyboard controls
#    → Saves to: Robot1/training_data/run_YYYYMMDD_HHMMSS/
```

### 2. Fixing Training Data (Post-Processing)

```bash
# Fix filename mismatches
python scripts/data_manager_post.py --robot 1 --all --apply
```

### 3. Training the Model

```bash
cd Robot1
python train_model.py --data training_data_combined --epochs 50
```

### 4. Running AI Mode

```bash
# 1. Configure Robot1
#    Edit Robot1/robot_config.txt:
#      mode = AI

# 2. Run the race
python main.py
```

---

## 🎯 Design Principles

### Separation of Concerns

1. **Top Level** = All robots share these files
   - Entry point (main.py)
   - Data collection (data_manager.py)
   - Communication (websocket_client.py)

2. **scripts/** = Project-wide tools (not robot-specific)
   - Post-processing (data_manager_post.py)
   - Utilities (make_video.py, etc.)

3. **RobotN/** = Robot-specific files
   - Model architecture (model.py)
   - Training script (train_model.py)
   - Control modes (*_input.py)
   - Configuration (robot_config.txt)

### Naming Consistency

- **data_manager.py** = Runtime data collection
- **data_manager_post.py** = Post-race data correction
- (Future) **data_manager_augment.py** = Data augmentation
- (Future) **data_manager_analyze.py** = Data analysis

---

## 📚 Documentation

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Project overview, installation, quick start | All users |
| `PROJECT_STRUCTURE.md` | Detailed project structure explanation | Developers |
| `scripts/README_fix_training_data.md` | Complete guide for data_manager_post.py | Users fixing training data |
| `scripts/QUICKSTART.md` | 5-minute quick start for data fixing | Users in a hurry |

---

## 🔧 Maintenance Notes

### Deprecated Files

Old/deprecated files are moved to `archive/deprecated/` with explanation in `archive/deprecated/README.txt`.

**Current deprecated files:**
- `train_model.py.old` - Old training script (replaced by Robot1/train_model.py)

### Adding New Robots

To add Robot6:

1. Copy `Robot1/` structure to `Robot6/`
2. Update `Robot6/robot_config.txt`
3. Update `config.txt` to include robot 6 in `active_robots`
4. Train Robot6's model: `cd Robot6 && python train_model.py`

---

## 🆕 Version History

### Beta 1.2 (2026-01-10)
- Added `scripts/data_manager_post.py` for training data correction
- Moved deprecated `train_model.py` to `archive/deprecated/`
- Created `PROJECT_STRUCTURE.md` (this file)
- Clarified file responsibilities and naming conventions

### Beta 1.1
- Added Input Vector Scope visualization
- Improved AI control stability

### Beta 1.0
- Initial multi-robot release
- Torque steer dynamics
- Global leaderboard integration

---

**For more information, see [README.md](README.md)**
