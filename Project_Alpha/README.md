# 🌟 Virtual Robot Race - Alpha Version

### 🧠 Build and Race Your Own AI!

Welcome to the **Alpha version** of Virtual Robot Race (AAGP)!
This guide helps you set up the race simulator on your Windows PC (Windows 11 only) and control your robot using Python.

You can manually drive the robot, replay pre-recorded torque data, or try rule-based and AI-controlled driving.

---

## 🔍 Overview

This guide walks you through:

1. Downloading the app from GitHub
2. Installing Python and required libraries
3. Understanding the file structure
4. Running the simulator and choosing control modes

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
# Move to the project directory
cd project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🧠 AI Model Download

The AI mode requires a trained model file `model.pth`.

> ⚠️ This file is **not included** in the repository due to GitHub’s 100MB limit.

👉 [Download model.pth from Google Drive](https://drive.google.com/file/d/19qWtxAC1ABYiK1CGDg9A0PDX67u39I_v/view?usp=sharing)

After downloading, place the file in this path:

```
Project_Alpha/models/model.pth
```

Make sure the filename is exactly `model.pth`.

---

## 📂 Step 3: Project Structure

```
project/
├── main.py
├── websocket_server.py
├── config.py
├── config.txt
├── keyboard_input.py
├── table_input.py
├── table_input.csv
├── data_interactive/
├── rule_based_input.py
├── rule_based_algorithms/
│   ├── perception_Startsignal.py
│   ├── Linetrace_white.py
│   └── status_Robot.py
├── inference_input.py
├── models/
│   └── model.pth   <download from Google Drive>
├── data_manager.py
├── Windows/
│   ├── AAgp_test30.exe
│   ├── runtime_log.txt
│   ├── UnityCrashHandler64.exe
│   ├── UnityPlayer.dll
│   ├── AAgp_test30_Data/
│   └── MonoBleedingEdge/
└── training_data/
    └── run_YYYYMMDD_HHMMSS/
        ├── images/
        │   ├── frame_00001.jpg
        │   ├── frame_00002.jpg
        │   └── ...
        ├── metadata.csv
        ├── table_input.csv
        └── UnityLog.txt
```

---

## ▶️ Step 4: Run the Simulator

```bash
python main.py
```

* Unity will auto-launch.
* Press `q` to end the race anytime.

---

## 📲 Choose Your Control Mode

Edit `config.txt` to set your control method:

```ini
# 1 = keyboard (manual)
# 2 = table (CSV playback)
# 3 = rule_based (signal + line follow)
# 4 = ai (PyTorch model)
MODE_NUM=1
```

---

## 📊 Verified Test Environments

| Device           | CPU     | GPU               | RAM  | Status          |
| ---------------- | ------- | ----------------- | ---- | --------------- |
| Dev PC           | Core i5 | RTX 3060          | 16GB | ✅ Smooth        |
| Surface Laptop 4 | Core i5 | Intel Iris Xe GPU | 8GB  | ✅ Works (AI OK) |

---

## 📊 Recommended Specs

* OS: Windows 11 (64-bit)
* CPU: Intel Core i5 (10th Gen+)
* GPU: GTX 1650 or higher
* RAM: 8GB+
* Python: 3.10

*Note: Alpha version only supports Windows. Mac/Linux not yet available.*

---

## 😊 Community & Support

* Discord: [https://discord.gg/BCTd2ctq](https://discord.gg/BCTd2ctq)
* Official Website: [https://virtualrobotrace.com](https://virtualrobotrace.com)

---

Race your Algorithm. ✨

# Project Alpha – Virtual Robot Race

This is the **Alpha version** of the Virtual Robot Race project.  
You can manually drive the robot, replay pre-recorded torque data, or try rule-based and AI-controlled driving.

---

## 🚀 How to Use

1. Clone this repository
2. Install Python 3.10+
3. Install required packages:

pip install -r requirements.txt

4. Run the Python main script:

python main.py

6. Set the control mode in `config.txt`:
- `1 = keyboard`
- `2 = table (CSV)`
- `3 = rule_based`
- `4 = ai (requires model.pth)`

---

## 🧠 AI Model Download

The AI mode requires a trained model file `model.pth`.

> ⚠️ This file is **not included** in the repository due to GitHub’s 100MB limit.

👉 [Download model.pth from Google Drive] https://drive.google.com/file/d/19qWtxAC1ABYiK1CGDg9A0PDX67u39I_v/view?usp=sharing


After downloading, place the file in this path:

Project_Alpha/models/model.pth



Make sure the filename is exactly `model.pth`.

---

## 🗂 Folder Structure

```
project/
├── main.py
├── websocket_server.py
├── config.py
├── config.txt
├── keyboard_input.py
├── table_input.py
├── table_input.csv
├── data_interactive/
├── rule_based_input.py
├── rule_based_algorithms/
│   ├── perception_Startsignal.py
│   ├── Linetrace_white.py
│   └── status_Robot.py
├── inference_input.py
├── models/
│   └── model.pth   <dowonload from google drive>
├── data_manager.py
├── Windows/
│   ├── AAgp_test30.exe
│   ├── runtime_log.txt
│   ├── UnityCrashHandler64.exe
│   ├── UnityPlayer.dll
│   ├── AAgp_test30_Data/
│   └── MonoBleedingEdge/
└── training_data/
│   └──run_YYYYMMDD_HHMMSS/
│       └──images/
│           ├── frame_00001.jpg
│           ├── frame_00002.jpg
│           └── ...
│       └──metadata.csv
│       └──table_input.csv
│       └──UnityLog.txt   
```

---

## 💡 Notes

- Training data is saved in `/training_data/` when enabled.
- Logs and debug images are saved per run.
- This is a work-in-progress Alpha version and may contain bugs or changes in the future.

---

Race your algorithm! 🏁

