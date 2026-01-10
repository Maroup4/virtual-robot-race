# Virtual Robot Race – Beta 1.3 Setup Guide

## Overview

Beta 1.3 introduces **smartphone controller** functionality, allowing users to control robots directly from their smartphone browser via QR code scanning.

## Architecture

```
[Smartphone Browser]
        ↕ WebSocket
[Python Smartphone Server] (Port 8080)
        ↕
[Python WebSocket Client]
        ↕ WebSocket
[Unity WebSocket Server] (Port 12346)
```

## Requirements

### Python Dependencies

Install the new dependencies:

```bash
pip install aiohttp qrcode[pil]
```

Or use the updated requirements.txt:

```bash
pip install -r requirements.txt
```

### Network Setup

**Option 1: Home/Office Wi-Fi (Development)**
- Connect both PC and smartphone to the same Wi-Fi network
- No additional setup required

**Option 2: Exhibition/Demo (Standalone)**
- Bring a portable Wi-Fi router
- Configure router without internet connection (optional)
- Connect PC and allow visitors to connect their smartphones
- Alternative: Provide backup smartphones if visitor devices have compatibility issues

## Configuration

### Enable Smartphone Mode

Edit `Robot{N}/robot_config.txt` to set MODE_NUM=5:

```ini
# Robot1/robot_config.txt

ROBOT_ID=R1
NAME=Demo0001
MODE_NUM=5          # 5 = smartphone controller
DATA_SAVE=1
RACE_FLAG=1
```

### Multi-Robot Setup

You can enable smartphone control for up to 5 robots simultaneously:

```ini
# config.txt
ACTIVE_ROBOTS=1,2,3,4,5
```

Each robot with `MODE_NUM=5` will get its own smartphone controller endpoint.

## Running Beta 1.3

### 1. Start the System

```bash
python main.py
```

The system will:
1. Launch Unity server
2. Connect Python WebSocket clients
3. **Start smartphone server** (if MODE_NUM=5 is detected)
4. Display QR codes and URLs

### 2. Expected Console Output

```
[Main] Starting smartphone server for 1 robot(s)...
============================================================
📱 Smartphone Server Running
============================================================
Local URL:  http://localhost:8080
Network URL: http://192.168.1.100:8080
============================================================
QR Codes available at:
  R1:
    Controller: http://192.168.1.100:8080/controller?robot=R1
    QR Code:    http://192.168.1.100:8080/qr/R1
============================================================
  QR code saved: qr_codes\R1_controller.png
All QR codes saved to qr_codes/
[Main] Smartphone server started and ready for connections
```

### 3. Access Controller from Smartphone

**Method 1: Scan QR Code (Recommended)**

1. Open `qr_codes/R1_controller.png` on your PC screen
2. Scan with smartphone camera
3. Browser opens → Controller UI loads

**Method 2: Direct URL**

1. On smartphone, open browser
2. Navigate to: `http://{PC_IP}:8080/controller?robot=R1`
3. Replace `{PC_IP}` with your PC's local IP (shown in console)

## Controller UI

### Features

- **Camera Feed**: Robot's camera view (5 fps, low latency)
- **Left Joystick**: Steering control (-1.0 to +1.0)
- **Right Joystick**: Throttle/Brake (-1.0 to +1.0)
- **Status Bar**: Connection status and robot ID
- **Debug Info**: Real-time control values and FPS

### Controls

- **Touch & Drag**: Move joysticks to control
- **Dead Zone**: Center area ignores small movements
- **Auto-Reconnect**: Reconnects automatically if connection drops

### Visual Indicators

- 🟢 Green pulse: Connected
- 🔴 Red: Disconnected
- "SCANNING...": Waiting for camera
- "LIVE: ~0.5s DELAY": Camera streaming active
- "NO SIGNAL": Camera unavailable

## Staff Controls (Emergency)

Keyboard shortcuts for staff during demo:

- **F9**: Stop camera streaming
- **F10**: Resume camera streaming
- **F11**: Reset connections
- **F12**: Reset demo state
- **q**: Force-end race

## Troubleshooting

### Smartphone Cannot Connect

1. **Check Network**:
   - Verify both devices are on same Wi-Fi
   - Try accessing `http://{PC_IP}:8080` from smartphone browser

2. **Firewall**:
   - Windows may block port 8080
   - Add firewall exception for Python or disable temporarily

3. **IP Address**:
   - Console shows IP as `192.168.x.x`
   - If shows `127.0.0.1`, network detection failed
   - Manually find PC IP: `ipconfig` (Windows) / `ifconfig` (Mac/Linux)

### Camera Not Displaying

1. **Check Unity**:
   - Verify Unity is writing images to `Robot1/data_interactive/latest_RGB_*.jpg`
   - Check `latest_RGB_now.txt` exists and contains "a" or "b"

2. **File Permissions**:
   - Ensure Python can read from `data_interactive/` directory

3. **Low FPS Normal**:
   - 5 fps is expected (design choice for stability)
   - This is telemetry, not video streaming

### Control Not Responding

1. **WebSocket Connection**:
   - Check console for WebSocket errors
   - Verify Unity server is running on port 12346

2. **Mode Configuration**:
   - Confirm `MODE_NUM=5` in robot config
   - Restart `main.py` after config changes

## Testing Checklist

Before demo/exhibition:

- [ ] QR codes generated successfully
- [ ] Smartphone can scan and load controller
- [ ] Camera feed visible on smartphone
- [ ] Joystick controls move robot in Unity
- [ ] Connection stable for 30+ minutes
- [ ] Multiple smartphones can control different robots
- [ ] Staff controls (F9-F12) work correctly
- [ ] Auto-reconnect works after temporary disconnection

## Exhibition Setup Procedure

1. **Pre-Event**:
   - Print QR codes in large format (A4/A5)
   - Test with multiple smartphone models
   - Prepare backup smartphones

2. **On-Site**:
   - Set up Wi-Fi router (no internet needed)
   - Connect PC to router
   - Display QR codes prominently
   - Start `main.py`
   - Verify one test connection

3. **During Event**:
   - Monitor console for errors
   - Use F9/F10 to control camera if needed
   - Use F12 to reset between visitors
   - Keep backup smartphones ready

4. **Post-Event**:
   - Collect data from `Robot*/experiments/`
   - Review logs for issues
   - Update config based on learnings

## Advanced Configuration

### Change Server Port

Edit `main.py` line 652:

```python
smartphone_server = SmartphoneServer(port=8080)  # Change port here
```

### Adjust Camera FPS

Edit `smartphone_server.py` line 127:

```python
frame_interval = 0.2  # 5 fps (change to 0.1 for 10 fps)
```

**Warning**: Higher FPS may impact control responsiveness.

### Customize Joystick Sensitivity

Edit `smartphone_controller.html` line 18:

```javascript
const DEAD_ZONE = 0.1;  // Adjust dead zone (0.0 - 0.5)
```

## API Reference

### HTTP Endpoints

- `GET /`: Robot selection page
- `GET /controller?robot={R1-R5}`: Controller UI
- `GET /qr/{robot_id}`: QR code image (PNG)

### WebSocket Endpoints

- `ws://host:8080/ws/R1`: Robot 1 control channel
- `ws://host:8080/ws/R2`: Robot 2 control channel
- (R3, R4, R5 available)

### WebSocket Message Format

**Client → Server (Control)**:
```json
{
  "type": "control",
  "steerAngle": 0.5,
  "driveTorque": 0.75
}
```

**Server → Client (Camera)**:
```json
{
  "type": "camera",
  "image": "base64_encoded_jpeg",
  "robot_id": "R1"
}
```

## Known Limitations

- Maximum 5 concurrent robots (architectural limit)
- Camera fixed at 5 fps (stability priority)
- No SOC transfer between robots (Beta 2.0 feature)
- No AI driving demonstration mode (Beta 2.0 feature)
- IPv4 only (no IPv6 support)

## Next Steps

See [beta_1_3_concept.md](beta_1_3_concept.md) for design philosophy and future roadmap.

---

**Last Updated**: 2026-01-10
**Version**: Beta 1.3.0
