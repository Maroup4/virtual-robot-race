# 🌟 Virtual Robot Race - Beta Version

### 🏁 Race Against Each Other with Realistic Car Physics!

Welcome to the **Beta version** of Virtual Robot Race!
This version introduces **2-robot racing** with **torque steer dynamics** — drive like a real car, not a tank!

You can race against a friend or against your own AI algorithms, then share your lap times on our global leaderboard.

---

## 🆕 What's New in Beta

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

* Download and install **Python 3.10 (64-bit)**:
  [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

* Open Command Prompt or Terminal:

```bash
# Move to the Project_Beta directory
cd Project_Beta

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🧠 AI Model Download (Optional)

The AI mode requires a trained model file `model.pth`.

> ⚠️ This file is **not included** in the repository due to GitHub's 100MB limit.

👉 [Download model.pth from Google Drive](https://drive.google.com/file/d/19qWtxAC1ABYiK1CGDg9A0PDX67u39I_v/view?usp=sharing)

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
├── config.py                # Configuration loader
├── config.txt               # Global settings (which robots to activate)
├── requirements.txt         # Python dependencies
├── setup_env.bat            # Quick setup script
├── data_manager.py          # Training data manager
├── make_video.py            # Video creation utility
├── train_model.py           # AI model training
│
├── Robot1/                  # First robot configuration
│   ├── robot_config.txt     # Robot1 settings (mode, name, race flag)
│   ├── keyboard_input.py    # Manual control
│   ├── table_input.py       # CSV playback
│   ├── table_input.csv      # Recorded control data
│   ├── rule_based_input.py  # Rule-based AI
│   ├── inference_input.py   # Neural network AI
│   ├── models/
│   │   └── model.pth        # AI model (download separately)
│   ├── rule_based_algorithms/
│   │   ├── driver_model.py
│   │   ├── Linetrace_white.py
│   │   ├── perception_Lane.py
│   │   └── ...
│   └── training_data/       # Recorded runs
│       └── run_YYYYMMDD_HHMMSS/
│           ├── images/
│           ├── metadata.csv
│           └── output_video.mp4
│
├── Robot2/                  # Second robot configuration
│   └── (same structure as Robot1)
│
└── Windows/                 # Unity executable
    ├── Unity_Build.exe
    └── ...
```

---

## ⚙️ Step 4: Configure Your Robots

### Global Settings (`config.txt`)

```ini
HOST=localhost
PORT=12346

# Which robots to activate (comma-separated)
ACTIVE_ROBOTS=1,2    # Both robots active
# ACTIVE_ROBOTS=1    # Only Robot1

DEBUG_MODE=0         # 0=Auto-launch Unity, 1=Manual launch
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
JPEG_SAVE=1          # 1=Save images, 0=Don't save
AUTO_MAKE_VIDEO=0    # 1=Auto-create video after race

# Video settings
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
- **S**: Brake/Reverse
- **A**: Steer left
- **D**: Steer right

### 2. Table Playback (MODE_NUM=2)
Replay pre-recorded control data from CSV files.

### 3. Rule-Based AI (MODE_NUM=3)
Autonomous driving using:
- Start signal detection
- Lane following algorithms
- Track position estimation

### 4. Neural Network AI (MODE_NUM=4)
AI-powered control using trained PyTorch models.

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
MODE_NUM=3    # Rule-based AI
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

* Discord: [https://discord.gg/BCTd2ctq](https://discord.gg/BCTd2ctq)
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

Race your Algorithm. Challenge the World. ✨
