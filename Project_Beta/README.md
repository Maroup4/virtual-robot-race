# 🌟 Virtual Robot Race - Beta Version

### 🏁 Race Against Each Other with Realistic Car Physics!

Welcome to the **Beta version** of Virtual Robot Race!
This version introduces **2-robot racing** with **torque steer dynamics** — drive like a real car, not a tank!

You can race against a friend or against your own AI algorithms, then share your lap times on our global leaderboard.

---

## 🆕 What's New in Beta

### 🔧 Beta 1.2 - Training Data Integrity Fix (2026-01-10)
**CRITICAL FIX**: Resolves training data corruption that prevented AI models from learning properly.

- **Problem Identified**: Image filenames saved by Python didn't match Unity's metadata.csv
  - Unity: `tick=329` → `frame_000329.jpg` (tick-based naming)
  - Python: 1st image → `frame_000001.jpg` (sequential counting)
  - Result: 328-frame offset causing complete training failure
- **Solution**: Official training data correction tool (`scripts/data_manager_post.py`)
  - Sequential renaming to align images with metadata
  - Automatic backup system for safety
  - Multi-robot support (Robot1-5)
  - Dry-run verification before applying changes
- **Impact**: Enables proper AI training and significantly improves model performance
- **Documentation**: See [scripts/README_fix_training_data.md](scripts/README_fix_training_data.md)

### 🚗 Torque Steer Driving (Like a Real Car!)
- **Alpha**: Differential drive (tank-style steering)
- **Beta**: Torque steer dynamics (realistic car physics with acceleration and steering)

### 🏆 2-Robot Racing
- Race two robots simultaneously
- Compete head-to-head with friends or test multiple AI algorithms
- Each robot can run different control modes

### 🌐 Global Leaderboard
- Share your race results online
- Compare lap times with racers worldwide
- X (Twitter) integration for sharing achievements
- View results at: [https://virtualrobotrace.com](https://virtualrobotrace.com)

### 🎯 Input Vector Scope (v1.1)
- Real-time visualization of drive torque and steering angle
- Circular radar display with motion trail (afterimage effect)
- Cyan dot for acceleration, red for braking
- Educational value: smooth curves = good AI control, jagged = poor control

---

## 🔍 Overview

This guide walks you through:

1. Downloading the app from GitHub
2. Installing Python and required libraries
3. Understanding the new multi-robot file structure
4. Configuring and racing your robots
5. Sharing your results online

---

## 📁 Step 1: Download the App

Clone or download the repository:

* GitHub: [https://github.com/AAgrandprix/virtual-robot-race](https://github.com/AAgrandprix/virtual-robot-race)

```bash
# Clone with Git
git clone https://github.com/AAgrandprix/virtual-robot-race.git
```

Or download ZIP and extract it.

---

## 🔧 Step 2: Install Python & Libraries

### Python Installation
* Download and install **Python 3.12+ (64-bit)**:
  [https://www.python.org/downloads/](https://www.python.org/downloads/)

  ⚠️ **Important**: During installation, check "Add Python to PATH"

  > 💡 Tested with Python 3.12 and 3.13. Python 3.10/3.11 may work but are not officially supported.

### Quick Setup (Recommended)
📌 **Easiest way**: Simply double-click `setup_env.bat` in the Project_Beta folder.

This will automatically:
- Create virtual environment (.venv)
- Activate it
- Install all required packages

### Manual Setup (Alternative)
If you prefer manual control or setup_env.bat doesn't work:

1. Open **Command Prompt** (not PowerShell)
2. Navigate to Project_Beta:
   ```bash
   cd path\to\virtual-robot-race\Project_Beta
   ```
3. Run these commands one by one:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

✅ You'll see `(.venv)` at the start of your command line when virtual environment is active.

### 🚀 GPU Acceleration (Recommended for AI Training)

If you have an **NVIDIA GPU** and want faster AI model training, install the CUDA-enabled PyTorch:

```bash
# First, activate the virtual environment
.venv\Scripts\activate

# Uninstall CPU version and install CUDA version
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Verify GPU is detected:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If you see `CUDA available: True`, GPU acceleration is working!

> 💡 **No GPU?** The default CPU version works fine for inference (running the AI). GPU mainly speeds up training.

> ⚠️ **CUDA version**: This project uses CUDA 12.4. If you have an older GPU or driver, try `cu118` instead of `cu124`.

---

## 🧠 AI Model Download (Optional)

The AI mode requires a trained model file `model.pth`.

> ⚠️ This file is **not included** in the repository due to GitHub's 100MB limit.

👉 [Download model.pth from Google Drive](https://drive.google.com/file/d/1NDL3A2lWDgXdy7OUWctyoR35jtYqthWD/view?usp=sharing)

After downloading, place the file in:

```
Project_Beta/Robot1/models/model.pth
Project_Beta/Robot2/models/model.pth
```

> 💡 You only need to download the model if you want to use AI control mode (MODE_NUM=4)

---

## 📂 Step 3: Project Structure

```
Project_Beta/
├── main.py                  # Main launcher
├── websocket_client.py      # Communication with Unity
├── config_loader.py         # Configuration loader
├── config.txt               # Global settings (which robots to activate)
├── requirements.txt         # Python dependencies
├── setup_env.bat            # Quick setup script
├── data_manager.py          # Training data manager
├── make_video.py            # Video creation utility
│
├── Robot1/                  # First robot configuration
│   ├── robot_config.txt     # Robot1 settings (mode, name, race flag)
│   ├── keyboard_input.py    # Manual control (MODE_NUM=1)
│   ├── table_input.py       # CSV playback (MODE_NUM=2)
│   ├── table_input.csv      # Recorded control data
│   ├── rule_based_input.py  # Rule-based control (MODE_NUM=3)
│   ├── inference_input.py   # AI inference engine (MODE_NUM=4)
│   ├── ai_control_strategy.py  # AI strategy settings (hybrid/pure_e2e)
│   ├── model.py             # Neural network model definition
│   ├── train_model.py       # Model training script
│   ├── rule_based_algorithms/
│   │   ├── driver_model.py
│   │   ├── perception_Lane.py
│   │   ├── perception_Startsignal.py
│   │   └── ...
│   ├── data_interactive/    # Real-time data (auto-generated, gitignored)
│   │   ├── last_run_dir.txt       # Path to most recent run
│   │   ├── latest_RGB_a.jpg       # Current camera frame (buffer A)
│   │   ├── latest_RGB_b.jpg       # Current camera frame (buffer B)
│   │   ├── latest_RGB_now.txt     # Active buffer indicator (a or b)
│   │   ├── latest_frame_name.txt  # Current frame filename
│   │   ├── latest_SOC.txt         # Current battery state of charge
│   │   └── latest_torque.txt      # Current drive torque value
│   ├── models/
│   │   └── model.pth        # AI model (download separately)
│   └── training_data/       # Recorded runs for AI training
│       └── run_YYYYMMDD_HHMMSS/
│           ├── images/
│           ├── metadata.csv
│           ├── unity_log.txt   # Unity debug log (auto-generated)
│           └── output_video.mp4
│
├── Robot2/                  # Second robot configuration
│   └── (same structure as Robot1)
│
└── Windows/                 # Unity executable
    ├── VirtualRobotRace_Beta.exe
    └── ...
```

---

## ⚙️ Step 4: Configure Your Robots

### Global Settings (`config.txt`)

```ini
HOST=localhost
PORT=12346

# Which robots to activate (comma-separated, Beta: max 2 robots)
ACTIVE_ROBOTS=1,2    # Both robots active
# ACTIVE_ROBOTS=1    # Only Robot1

DEBUG_MODE=0         # 0=Auto-launch Unity (recommended), 1=Manual (advanced)
```

### Per-Robot Settings (`Robot1/robot_config.txt`, `Robot2/robot_config.txt`)

```ini
# Control mode
MODE_NUM=1           # 1=keyboard, 2=table, 3=rule_based, 4=ai

# Robot identifier
ROBOT_ID=R1          # R1, R2, etc.

# Player name (shown on leaderboard)
NAME=Player1234      # Up to 10 alphanumeric characters

# Race participation
RACE_FLAG=1          # 1=Post results to leaderboard, 0=Practice only

# Recording settings
DATA_SAVE=1          # 1=Save CSV and images, 0=Don't save
AUTO_MAKE_VIDEO=0    # 1=Auto-create video after race

# Advanced video settings (usually no need to change)
VIDEO_FPS=20
INFER_FPS=1
```

---

## ▶️ Step 5: Run the Simulator

```bash
python main.py
```

* Unity will auto-launch (if DEBUG_MODE=0)
* Both robots will appear on the track
* Press `q` to end the race anytime

---

## 🎮 Control Modes

### 1. Keyboard Control (MODE_NUM=1)
Manually drive your robot with the keyboard:
- **W**: Accelerate
- **Z**: Brake/Reverse
- **J**: Steer left
- **L**: Steer right
- **I** or **M**: Steer center (neutral)

### 2. Table Playback (MODE_NUM=2)
Replay pre-recorded control data from CSV files.

### 3. Rule-Based  (MODE_NUM=3)
Autonomous driving using:
- Start signal detection
- Lane following algorithms
- Track position estimation

### 4. Neural Network AI (MODE_NUM=4)
AI-powered control using trained PyTorch models.

---

## 📊 Data Recording (DATA_SAVE=1)

When `DATA_SAVE=1` is enabled, race data is automatically saved to the `training_data` folder.

### Folder Structure
```
Robot1/training_data/
└── run_YYYYMMDD_HHMMSS/
    ├── images/              # Camera RGB images (JPEG)
    ├── metadata.csv         # Telemetry data
    └── unity_log.txt   　   # Unity debug log (auto-generated)
```

### metadata.csv Columns

| Column | Description |
|--------|-------------|
| `id` | Tick ID for system tracking (1 tick = 50ms) |
| `session_time` | Game system internal timer |
| `race_time` | Time elapsed since start signal turned GO |
| `filename` | Image filename linked to this tick (for training) |
| `soc` | Robot battery State of Charge (%) |
| `drive_torque` | Drive torque command value sent to robot |
| `steer` | Steering angle command value (radians, positive=right, negative=left) |
| `status` | Race status: `StartSequence`, `Lap0`/`Lap1`/`Lap2`/`Lap3`, `Finish`, `Fallen`, `FalseStart`, `BatteryDepleted`, `ForceEnd` |
| `pos_z` | Position in forward direction (meters) |
| `pos_x` | Position in lateral direction (meters) |
| `yaw` | Heading angle: 0° at start, positive=right, negative=left (degrees) |
| `pos_y` | Position in vertical direction (meters) |
| `error_code` | Error code (currently dummy value: 999) |

### Usage
- **AI Training**: Use `images/` and `metadata.csv` to train neural network models
- **Analysis**: Review driving behavior and optimize control algorithms
- **Replay**: Use metadata for table playback mode (MODE_NUM=2)

---

## 🏁 Racing Scenarios

### Solo Practice
```ini
# config.txt
ACTIVE_ROBOTS=1

# Robot1/robot_config.txt
MODE_NUM=1
RACE_FLAG=0    # Practice mode
```

### Head-to-Head Race
```ini
# config.txt
ACTIVE_ROBOTS=1,2

# Robot1/robot_config.txt
MODE_NUM=1    # You control Robot1
NAME=YourName
RACE_FLAG=1   # Post your result

# Robot2/robot_config.txt
MODE_NUM=4    # AI controls Robot2
NAME=AIDriver
RACE_FLAG=0   # Don't post AI result
```

### Algorithm Competition
```ini
# Compare two different AI approaches
ACTIVE_ROBOTS=1,2

# Robot1/robot_config.txt
MODE_NUM=3    # Rule-based 
NAME=RuleBot

# Robot2/robot_config.txt
MODE_NUM=4    # Neural network AI
NAME=NeuralBot
```

---

## 🌐 Sharing Your Results

After completing a race with `RACE_FLAG=1`:

1. Your lap time will be automatically posted to the leaderboard
2. Visit [https://virtualrobotrace.com](https://virtualrobotrace.com) to see your ranking
3. Share your achievement on X (Twitter)
4. Challenge other racers to beat your time!

> 💡 Tip: Set `RACE_FLAG=0` during practice to avoid posting incomplete runs

---

## 📊 Verified Test Environments

| Device           | CPU                           | GPU                            | RAM      | Status          |
| ---------------- | ----------------------------- | ------------------------------ | -------- | --------------- |
| Dev PC           | 12th Gen Intel Core i5-12450H | NVIDIA GeForce RTX 3060 Laptop | 16.00 GB | ✅ Smooth        |
| Surface Laptop 2 | 8th Gen Intel Core i5         | Intel UHD Graphics 620         | 8GB      | ✅ Works (AI OK) |

---

## 📊 Recommended Specs

This Beta version is verified on **Windows 11**.

If you're using a different setup and it works, we'd love to hear your specs!
Please share your test results with us via Discord or GitHub Issues. 😊

> ⚠️ Mac/Linux support is not yet available

---

## 🎯 Tips for Better Racing

- **Practice first**: Use `RACE_FLAG=0` to learn the track
- **Watch replays**: Create videos with `AUTO_MAKE_VIDEO=1` to analyze your driving
- **Tune your AI**: Training data is saved in `Robot*/training_data/`
- **Compare modes**: Race different control methods against each other

---

## 🆚 Beta vs Alpha Comparison

| Feature          | Alpha               | Beta                    |
| ---------------- | ------------------- | ----------------------- |
| Drive System     | Differential (Tank) | Torque Steer (Car)      |
| Robots           | 1 Robot             | 2 Robots                |
| Multiplayer      | ❌                   | ✅ Head-to-head          |
| Leaderboard      | ❌                   | ✅ Global online         |
| Control Modes    | 4 modes             | 4 modes (same)          |
| Configuration    | Single config file  | Per-robot config        |
| Physics          | Basic               | Realistic car dynamics  |

---

## 😊 Community & Support

* YouTube: https://www.youtube.com/@AAgrand_prix
* Official Website: [https://virtualrobotrace.com](https://virtualrobotrace.com)
* GitHub Issues: [https://github.com/AAgrandprix/virtual-robot-race/issues](https://github.com/AAgrandprix/virtual-robot-race/issues)

---

## 🐛 Troubleshooting

### Unity won't launch
- Set `DEBUG_MODE=1` in config.txt and launch Unity manually
- Check that `Windows/Unity_Build.exe` exists

### Robot doesn't move
- Verify `ACTIVE_ROBOTS` includes your robot number
- Check that the robot's `MODE_NUM` is set correctly
- Try keyboard control (MODE_NUM=1) first

### AI model not found
- Download `model.pth` from Google Drive
- Place it in `Robot*/models/model.pth`
- Make sure the filename is exactly `model.pth`

### Results not posting
- Check your internet connection
- Verify `RACE_FLAG=1` in robot_config.txt
- Ensure `NAME` is 1-10 alphanumeric characters

---

## Changelog

### v1.1 (2025-12-13)
- **New**: Input Vector Scope UI - real-time visualization of drive/steer inputs with motion trail
- **New**: Rule-Based autonomous driving achieves 2-lap goal
- **Fix**: AI mode start signal timing delay resolved with async WebSocket
- **Improve**: AI inference optimization with preload_model() for faster startup
- **Refactor**: Renamed config.py to config_loader.py to prevent user confusion

### v1.0 (2025-11-30)
- **Fix**: Rule-based mode (MODE_NUM=3) now works correctly when launched via `main.py`
  - Fixed module import path issue for `rule_based_algorithms` in Robot1/Robot2

---

Race your Algorithm. Challenge the World. ✨
