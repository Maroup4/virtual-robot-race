# Unity Function Overview — Virtual Robot Race

This document describes the internal processing flow of the Unity simulator and provides an overview of the data ultimately passed to the Python side. For details on the communication interface, refer to `Unity_Interface_Spec.md`.

---

## 1. Role of Unity

Unity operates as a WebSocket server and manages all of the following:

| Responsibility | Details |
|---|---|
| Physics Simulation | Robot driving, turning, collisions |
| Race Progression | Start detection, lap counting, end conditions |
| Battery Simulation | Power consumption and collision penalty calculation |
| Camera Image Generation | Onboard camera JPEG encoding and transmission |
| Data Logging | Tick-by-tick time-series data accumulation |
| Python Communication | Receiving control commands, sending state data |

---

## 2. Startup and Connection Sequence

```
Unity Startup
  │
  ├─ GameManager.Start()
  │    ├─ Create TickScheduler (50ms = 20fps)
  │    └─ Register PlayerCameraRGB to Tick
  │
  ├─ WebSocketServer.Start()
  │    ├─ Start WebSocket server on port 12346 (/robot)
  │    └─ Auto-detect and hide all robots (R1-R5) in the scene
  │
Python Connects (receives "connection" message)
  │    ├─ Set robot ID, player name, mode, raceFlag
  │    ├─ Receive active_robots list → Show and enable only participating robots
  │    └─ Send back connection acknowledgment response
  │
Python "ready" message received (waits for all robots)
  │
RaceManager: Start race start sequence
  │    ├─ Create and register DataLogger for each robot to Tick
  │    └─ Light up countdown with RaceStartLights
  │
"GO" signal fired → Race start time confirmed
```

---

## 3. Tick-Driven Processing (20fps / every 50ms)

The `TickScheduler` fires an `OnTick` event every 50ms, triggering the following in sequence:

```
OnTick(tickIndex, utcMs)
  │
  ├─ PlayerCameraRGB.HandleTick()
  │    └─ Read pixels from RenderTexture → Convert to JPEG → Send via WebSocket as binary
  │
  └─ DataLogger.HandleTick()
       └─ Record one line of data (position, rotation, SOC, steering, collision info, etc.)
```

---

## 4. Major Components and Their Processes

### 4.1 WebSocketServer
Receives messages from Python and routes them to the appropriate components in Unity.

**Receive Handling:**
- `"control"` message → Calls `DriveAndSteerController.SetTorque()` / `ApplySteer()`
- `"connection"` message → Registers robot, handles active_robots
- `"force_end"` message → Forces the race to end

**Send Handling:**
- Camera Images (binary) → To each robot's Python client
- SOC Data (JSON) → Sent periodically
- End-of-Race Metadata (JSON) → Sent once

---

### 4.2 DriveAndSteerController
Applies control values received from Python to the physics engine (WheelCollider).

**Input:**
- `driveTorque` (-1.0 to +1.0): Command value for rear-wheel drive torque

**Internal Processing:**
- Smooths commands with a filter (`smoothing=0.2`)
- Forces torque to zero if battery is depleted
- Applies `motorTorque` to two rear wheels (max 120Nm)
- Applies `steerAngle` to two front wheels (max ±30 degrees)

**Output to DataLogger:**
- `GetCurrentDriveTorque()` → Normalized value (-1 to 1)
- `GetCurrentSteerAngleRad()` → Radian value

---

### 4.3 BatteryManager
Simulates power consumption during driving and collision penalties.

**Consumption Model:**
- Every frame: Consumes `totalTorque × deltaTime` (sum of absolute values of left/right wheels)
- Collision penalty: Instantly subtracts `maxCapacity × fraction`

**State:**
- `soc` (0.0 to 1.0): Current state of charge
- `IsDepleted()`: Battery depleted flag → Unity ignores control commands

---

### 4.4 RobotStatus
Manages the race state for each robot.

**Tracked States:**
| State Name | Meaning |
|---|---|
| `StartSequence` | During countdown |
| `Running` | Racing |
| `Lap1` / `Lap2` ... | After passing a lap gate |
| `Finish` | Completed the required number of laps |
| `FalseStart` | False start detected → Disqualified |
| `Fallen` | Fell off the course → Retired |
| `BatteryDepleted` | Battery depleted → Becomes an obstacle |
| `ForceEnd` | Race was forcibly ended |

**False Start Detection:**
- Automatically detected if the robot moves more than 0.05m from its starting position before the "GO" signal.

**Fall Detection:**
- Detected if the Y-coordinate drops below -0.1m.

**Collision Data Aggregation (Beta 1.5):**
- `RecordCollisionForFrame()` accumulates collisions within a frame.
- `ConsumeCollisionData()` is used by DataLogger to read the data (resets after reading).

---

### 4.5 BodyCollisionHandler
Detects robot body collisions and applies physics responses and battery penalties.

**Collision Types and Handling:**

| Collided With | Velocity Response | Penalty Calculation |
|---|---|---|
| Wall | Reverses velocity along collision normal | `k × v² × 1.0` (100% self-responsibility) |
| Other Robot | Changes velocity in rebound direction | `k × |v_rel|² × R` (responsibility ratio based on approach) |

- 1-second cooldown: The same collision pair is not recalculated within 1 second.
- Penalties are immediately reflected in the BatteryManager.

---

### 4.6 GateTrigger
Updates the lap count when a robot passes through a gate on the course.

**Gate Configuration:**
- Gate 0: Start gate (race timing begins on first pass)
- Gate 1, 2: Finish gates (passing Gate 1 → Gate 2 in order = 1 lap)

Driving in reverse (Gate 2 → Gate 1) will decrement the lap count.

---

### 4.7 PlayerCameraRGB
Captures the onboard camera view and sends it to Python.

**Processing Flow (per Tick):**
1. Follow and move to the camera's offset position.
2. Read pixel data from the `RenderTexture`.
3. Encode to JPEG format (224x224px, 85% quality).
4. Send as a binary message via WebSocket to the corresponding robot's client.

Sending is asynchronous (`SendAsync`) and is not affected by the processing speed on the Python side.

---

### 4.8 DataLogger
Accumulates one line of data per tick into an internal list and outputs it as a CSV file at the end of the race.

**Data Recorded per Line:**

| Field | Content |
|---|---|
| `id` | Sequential index of the Tick |
| `session_time_ms` | Elapsed time from session start [ms] |
| `race_time_ms` | Elapsed time from GO signal [ms] |
| `filename` | Corresponding image file name (`frame_XXXXXX.jpg`) |
| `soc` | State of Charge (0.0 to 1.0) |
| `drive_torque` | Drive torque (-1.0 to +1.0) |
| `steer_angle` | Steering angle [rad] |
| `status` | Robot state string |
| `pos_z`, `pos_x`, `pos_y` | 3D coordinates [m] (Z: forward, X: right, Y: up) |
| `yaw` | Yaw angle [degrees] (forward=0, right=+) |
| `error_code` | Error code |
| `collision_type` | `"wall"` / `"robot"` / `"both"` / `""` |
| `collision_penalty` | Collision penalty rate (0.0~) |

---

### 4.9 RaceManager
Manages the entire race lifecycle.

**Race End Conditions (any of the following):**
1. All robots complete the specified number of laps (goalLap).
2. All robots reach a "completed" state for any reason (Finish/Fallen/FalseStart/BatteryDepleted).
3. Timeout (90 seconds).
4. A `force_end` command is received.

**End Processing:**
1. Stop accepting control commands.
2. Finalize the last line of each DataLogger with `status="Finish"`.
3. Generate metadata by combining CSV data and Unity logs.
4. Send `type: "metadata"` to each Python client.

---

## 5. Unity Internal Data Flow (Overall Picture)

```
Python
  │  "control" (driveTorque, steerAngle)
  ▼
WebSocketServer
  │  ApplyControl()
  ▼
DriveAndSteerController ←─────────────── BatteryManager
  │  SetTorque / ApplySteer              (IsDepleted → Torque Zero)
  │  Apply smoothing
  ▼
WheelCollider (Unity Physics Engine)
  │  Robot moves and turns
  ▼
┌─ GateTrigger ────────────────→ RobotStatus (update lapCount)
│
├─ BodyCollisionHandler ────────→ BatteryManager (apply penalty)
│                           └──→ RobotStatus (record collisionData)
│
└─ RobotStatus (Position/State Monitoring)
     │  y < -0.1 → Fallen
     │  False start detected → FalseStart
     │  lapCount >= goalLap → Finish

[Per Tick (50ms)]
  ├─ PlayerCameraRGB → JPEG → WebSocket → Python (binary)
  └─ DataLogger → Record 1 line (position, SOC, steering, collision data)

[At Race End]
  RaceManager → DataLogger (Generate CSV) → WebSocket → Python (metadata JSON)
```

---

## 6. List of Data Finally Sent to Python

| Timing | Data Type | Format | Content |
|---|---|---|---|
| Immediately after connection | Connection Ack | JSON | `type:"connection"`, status, message |
| Per Tick (50ms) | Camera Image | Binary | 224x224 JPEG (onboard camera view) |
| Periodically | State of Charge | JSON | `type:"soc"`, soc (0.0 to 1.0) |
| At Race End (once) | Race Metadata | JSON | `type:"metadata"`, csv_data, unity_log |

### Metadata CSV Column Structure

```
id,session_time_ms,race_time_ms,filename,soc,drive_torque,steer_angle,status,pos_z,pos_x,yaw,pos_y,error_code,collision_type,collision_penalty
```

> **Note**: The order of pos_z/pos_x in the CSV corresponds to the Unity coordinate system (Z=forward/backward, X=left/right).

---

## 7. Script List and Role Summary

| Script | Main Role |
|---|---|
| `GameManager.cs` | System-wide initialization and shutdown flow |
| `WebSocketServer.cs` | WebSocket server and message routing |
| `RaceManager.cs` | Race progression, end condition checking, metadata sending |
| `TickScheduler.cs` | Fires timer events at a 50ms period |
| `DriveAndSteerController.cs` | Applies control commands to the physics engine |
| `BatteryManager.cs` | SOC consumption and penalty calculation |
| `RobotStatus.cs` | Manages robot state (lap, fallen, false start, etc.) |
| `BodyCollisionHandler.cs` | Collision detection, physics response, penalty calculation |
| `GateTrigger.cs` | Gate pass detection and lap count update |
| `PlayerCameraRGB.cs` | Camera image capture and JPEG sending |
| `DataLogger.cs` | Tick-by-tick data logging and CSV generation |
| `TailLampController.cs` | Tail light display (steering→hue, throttle→brightness) |
