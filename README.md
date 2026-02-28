# Virtual Robot Race

**Race your Algorithm. Challenge the World.**

Educational robot racing simulator with Unity + Python — for learning AI, control algorithms, and autonomous driving.

---

## What is Virtual Robot Race?

Virtual Robot Race is a **beginner-friendly robot simulation environment** where you can:
- Race robots with realistic car physics (torque steer dynamics)
- Train and test AI models using PyTorch
- Compete on the global leaderboard
- Learn autonomous driving through hands-on experimentation

Perfect for students, hobbyists, and AI enthusiasts!

---

## Current Version

### **Beta 1.6** (Latest) - [`/Project_Beta`](./Project_Beta)

**Released:** 2026-02-28

This version includes:
- Realistic torque steer driving (like a real car, not a tank!)
- 2-robot head-to-head racing with collision penalty system
- Battery management with SOC (State of Charge) simulation
- Global online leaderboard with X (Twitter) integration
- Input Vector Scope visualization
- Offline reinforcement learning framework (DAgger+, AWR)
- 5 control modes: Keyboard, CSV Playback, Rule-Based, Neural Network AI, Smartphone

**[-> Get Started with Beta](./Project_Beta/README.md)**

---

### Alpha Version - [`alpha` branch](https://github.com/AAgrandprix/virtual-robot-race/tree/alpha/Project_Alpha)

The original version with differential drive (tank-style steering):
- Single robot racing
- 4 control modes
- Training data collection
- AI model training

> Archived in the [`alpha` branch](https://github.com/AAgrandprix/virtual-robot-race/tree/alpha/Project_Alpha).

---

## Quick Start (Beta)

### 1. Clone the Repository
```bash
git clone https://github.com/AAgrandprix/virtual-robot-race.git
cd virtual-robot-race/Project_Beta
```

### 2. Install Python Dependencies
```bash
# Recommended: Python 3.12+
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Or simply double-click **`setup_env.bat`** for automatic setup!

### 3. Download AI Model (Optional)
If you want to use AI control mode (MODE_NUM=4):
- [Download model.pth from Google Drive](https://drive.google.com/file/d/1NDL3A2lWDgXdy7OUWctyoR35jtYqthWD/view?usp=sharing)
- Place it in `Project_Beta/Robot1/models/model.pth`

### 4. Run the Simulator
```bash
python main.py
```

Unity will auto-launch and the race begins!

**[-> Full Setup Guide](./Project_Beta/README.md)**

---

## Repository Structure

```
virtual-robot-race/
├── Project_Beta/          # Latest version (Beta 1.6)
│   ├── main.py           # Main launcher
│   ├── Robot1/           # First robot configuration
│   ├── Robot2/           # Second robot configuration
│   ├── Windows/          # Unity executable
│   └── README.md         # Detailed documentation
│
├── README.md             # This file
├── .gitignore
└── LICENSE
```

> **Alpha version** is archived in the [`alpha` branch](https://github.com/AAgrandprix/virtual-robot-race/tree/alpha/Project_Alpha).

---

## Control Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Keyboard** (1) | Manual WASD control | Practice and data collection |
| **CSV Playback** (2) | Replay recorded runs | Reproduce results |
| **Rule-Based** (3) | Lane detection + algorithms | Traditional robotics approach |
| **Neural Network AI** (4) | PyTorch deep learning | End-to-end autonomous driving |
| **Smartphone** (5) | Phone as wireless gamepad | Intuitive dual-joystick control |

---

## Features

### Beta Version Highlights
- **Realistic Physics**: Torque steer dynamics based on real car behavior
- **Multiplayer Racing**: Race 2 robots simultaneously
- **Collision Penalty System** (Beta 1.5): SOC-based energy-proportional penalties on collision — AI learns to avoid crashes
- **Battery Management**: SOC simulation with discharge curves and collision penalties
- **Global Leaderboard**: Share your lap times online at [virtualrobotrace.com](https://virtualrobotrace.com)
- **AI Training Pipeline**: Collect data -> Train on Colab/local -> Deploy model
- **Offline Reinforcement Learning** (Beta 1.4): DAgger+ and AWR frameworks for learning from recorded data
- **Input Visualization**: Real-time scope showing drive/steer commands
- **Training Data Tools**: Automatic correction for metadata alignment
- **Cross-Platform Training**: Local PC or Google Colab with GPU

---

## AI Training Workflow

1. **Collect Data**: Drive manually (MODE_NUM=1) with DATA_SAVE=1
2. **Train Model**: Use local GPU or Google Colab
3. **Deploy**: Copy trained model.pth to Robot*/models/
4. **Race**: Run with MODE_NUM=4 (AI control)
5. **Iterate**: Analyze results and improve

---

## Community & Support

- **Official Website**: [virtualrobotrace.com](https://virtualrobotrace.com)
- **YouTube**: [@RaceYourAlgo](https://www.youtube.com/@RaceYourAlgo)
- **Issues**: [GitHub Issues](https://github.com/AAgrandprix/virtual-robot-race/issues)
- **Leaderboard**: View global rankings at the official website

---

## Tested Environments

| Device | CPU | GPU | RAM | Status |
|--------|-----|-----|-----|--------|
| Dev PC | Intel Core i5-12450H | RTX 3060 Laptop | 16GB | Smooth |
| Surface Laptop 2 | Intel Core i5 (8th Gen) | Intel UHD 620 | 8GB | Works |

> Windows only. Mac/Linux support coming soon.

---

## Alpha vs Beta Comparison

| Feature | Alpha | Beta |
|---------|-------|------|
| Drive System | Differential (Tank) | Torque Steer (Car) |
| Number of Robots | 1 | 2 |
| Multiplayer | No | Yes |
| Online Leaderboard | No | Yes |
| Control Modes | 4 | 5 |
| Collision Penalty | No | Yes (SOC-based) |
| Battery Simulation | No | Yes |
| Offline RL Training | No | Yes |
| Physics Realism | Basic | Realistic |

---

## License

This project is open to learning and contribution.
See [LICENSE](./LICENSE) for details.

---

## Project Vision

The goal of Virtual Robot Race is to:
- Make autonomous driving accessible to everyone
- Provide a hands-on AI learning environment
- Encourage creativity in algorithm design
- Build a global community of robot racers

**Race your Algorithm. Challenge the World.**

---

**Ready to race? Head to [Project_Beta](./Project_Beta) and start your engines!**
