# Virtual Robot Race - AI Agent Context

This document provides context for AI coding assistants (Claude Code, Copilot, Cursor, etc.) to help developers build racing algorithms.

---

## Race Rules

### Objective
- Complete **2 laps** around the track as fast as possible
- Fastest lap time wins

### Start Procedure
1. Signal lights: **1 -> 2 -> 3 -> ALL OFF** (GO!)
2. Robot must remain stationary until all lights turn off
3. **False Start** = Disqualification (moving before GO signal)

### Disqualification Conditions
- **False Start**: Moving before start signal
- **Track Fall**: Robot falls off the track
- **Battery Depleted**: SOC reaches 0 (not a concern in Beta - battery is generous)

### Allowed Actions
- Robot collision with other robots: **No penalty**
- Aggressive driving: **Allowed**

### Lap Counting
- Correct direction: Lap count increases
- **Wrong direction**: Lap count becomes **negative**

---

## Technical Specifications

### Control Loop
- **Frequency**: 20Hz (50ms interval)
- **Protocol**: WebSocket (JSON messages)
- **Delivery**: Best-effort (not guaranteed per tick)

### Control Commands
```python
{
    "type": "control",
    "robot_id": "R1",        # "R1" or "R2"
    "driveTorque": 0.5,      # Range: 0.0 to 1.0
    "steerAngle": 0.3        # Range: -0.524 to 0.524 (radians, ~30 degrees)
}
```

### Control Values
| Parameter | Min | Max | Unit | Notes |
|-----------|-----|-----|------|-------|
| driveTorque | 0.0 | 1.0 | normalized | 0=stop, 1=full throttle |
| steerAngle | -0.524 | 0.524 | radians | negative=left, positive=right |

### Telemetry Data (from Unity)
| Field | Description | Unit |
|-------|-------------|------|
| pos_x | Lateral position | meters |
| pos_z | Forward position | meters |
| pos_y | Vertical position | meters |
| yaw | Heading angle (0 at start, positive=right) | degrees |
| soc | State of Charge (battery) | 0.0-1.0 |
| race_time | Time since GO signal | milliseconds |
| status | Race state | string |

### Race Status Values
- `StartSequence`: Waiting for start signal
- `Lap0`, `Lap1`, `Lap2`: Current lap number
- `Finish`: Race completed
- `Fallen`: Robot fell off track
- `FalseStart`: Moved before GO signal
- `BatteryDepleted`: SOC reached 0

---

## File Structure

```
Project_Beta/
├── main.py                 # Entry point - launches Unity and control loop
├── websocket_client.py     # WebSocket communication with Unity
├── config.txt              # Global settings (HOST, PORT, ACTIVE_ROBOTS)
├── Robot1/                 # Robot 1 configuration and algorithms
│   ├── robot_config.txt    # MODE_NUM, NAME, RACE_FLAG settings
│   ├── keyboard_input.py   # MODE_NUM=1: Manual keyboard control
│   ├── table_input.py      # MODE_NUM=2: CSV playback
│   ├── rule_based_input.py # MODE_NUM=3: Rule-based autonomous
│   ├── inference_input.py  # MODE_NUM=4: Neural network AI
│   ├── rule_based_algorithms/
│   │   ├── driver_model.py         # Driving logic
│   │   ├── perception_Lane.py      # Lane detection
│   │   └── perception_Startsignal.py # Start signal detection
│   ├── training_data/      # Recorded runs (images + metadata.csv)
│   └── models/             # Neural network weights (model.pth)
└── Robot2/                 # Same structure as Robot1
```

---

## Control Modes

### MODE_NUM=1: Keyboard Control
- Manual driving for data collection
- Keys: W (accelerate), Z (brake), J/L (steer), I/M (center)

### MODE_NUM=2: Table Playback
- Replays recorded commands from `table_input.csv`
- Useful for testing recorded runs
- CSV columns: `time_id`, `drive_torque`, `steer_angle`

### MODE_NUM=3: Rule-Based
- Autonomous driving using computer vision
- Start signal detection + lane following
- Edit `rule_based_algorithms/` to customize

### MODE_NUM=4: Neural Network AI
- End-to-end learning from camera images
- Requires trained `model.pth`
- Train with `train_model.py` using recorded data

---

## Development Tips

### Getting Started
1. Run with `MODE_NUM=1` (keyboard) to understand the track
2. Record training data with `DATA_SAVE=1`
3. Analyze `metadata.csv` to understand telemetry
4. Start with `MODE_NUM=3` (rule-based) for quick autonomous driving

### Creating Custom Algorithms
- Implement `get_latest_command()` function returning:
  ```python
  {
      "type": "control",
      "robot_id": robot_id,
      "driveTorque": float,  # 0.0 to 1.0
      "steerAngle": float    # -0.524 to 0.524
  }
  ```
- Read camera images from `data_interactive/latest_RGB_*.jpg`
- Read telemetry from `data_interactive/` text files

### Performance Optimization
- Control loop runs at 20Hz - keep processing under 50ms
- Pre-load models at startup, not during race
- Use efficient image processing (resize, crop)

### Debugging
- Set `DEBUG_MODE=1` in `config.txt` to launch Unity manually
- Check `training_data/run_*/unity_log.txt` for Unity-side logs
- Use `RACE_FLAG=0` for practice (no leaderboard posting)

---

## Common Pitfalls

1. **Robot doesn't move**: Check `ACTIVE_ROBOTS` includes your robot number
2. **False start**: Don't send non-zero torque during `StartSequence` status
3. **Wrong CSV format**: Use lowercase headers (`drive_torque`, `steer_angle`)
4. **Model not found**: Place `model.pth` in `Robot*/models/` directory

---

## Quick Reference

```python
# Typical control loop structure
while racing:
    # 1. Read current state
    image = read_camera_image()
    telemetry = get_telemetry()

    # 2. Decide action
    if telemetry['status'] == 'StartSequence':
        torque, steer = 0.0, 0.0  # Wait for start!
    else:
        torque, steer = your_algorithm(image, telemetry)

    # 3. Send command
    send_control(torque, steer)

    # 4. Wait for next tick
    await asyncio.sleep(0.05)  # 20Hz
```

---

## Resources

- Leaderboard: https://virtualrobotrace.com
- GitHub: https://github.com/AAgrandprix/virtual-robot-race
- YouTube: https://www.youtube.com/@AAgrand_prix
