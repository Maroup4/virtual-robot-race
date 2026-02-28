# Unity Interface Specification

This document outlines the communication interface between the Python control scripts and the Unity-based Virtual Robot Race simulator.

## 1. Overview

The system consists of two main components:
- **Unity Application (Server)**: A closed-source Windows executable that runs the robot simulation, physics, and environment rendering. It acts as a WebSocket server.
- **Python Scripts (Client)**: Open-source Python scripts that control the robot's behavior. This includes manual control, rule-based algorithms, AI models, etc. The Python script acts as a WebSocket client.

Communication between these two components occurs in real-time over a WebSocket connection.

## 2. Communication Protocol

- **Protocol**: WebSocket
- **URL**: `ws://<HOST>:<PORT>/robot`
- **Default Values** (from `config.txt`):
  - `HOST`: `localhost`
  - `PORT`: `12346`
- **Default URL**: `ws://localhost:12346/robot`

The connection is initiated from the Python client to the Unity server.

## 3. Python (Client) to Unity (Server) Messages

Unless otherwise specified, all messages are JSON objects. Every message must contain a `type` field.

### 3.1 Handshake (`type: "connection"`)
Sent once immediately after the client connects to the server to register the robot.

**Fields:**
- `type` (string): `"connection"`
- `robot_id` (string): A unique identifier for the robot (e.g., `"R1"`, `"R2"`).
- `player_name` (string): The player's name, used for display and posting race results.
- `mode` (string): The control mode the robot is running in (e.g., `"keyboard"`, `"ai"`).
- `race_flag` (integer): `1` if the robot is participating in the race, `0` if only spectating.
- `active_robots` (list[int], optional): A list of all robot numbers that are active in the current session (e.g., `[1, 2]`). Typically sent only by the first robot to connect.

**Example:**
```json
{
  "type": "connection",
  "robot_id": "R1",
  "player_name": "Player0001",
  "mode": "ai",
  "race_flag": 1,
  "active_robots": [1, 2]
}
```

### 3.2 Ready Signal (`type: "ready"`)
Sent after the client has finished its own initialization (e.g., loading an AI model). The Unity server will wait for all declared `active_robots` to send this signal before starting the race countdown.

**Fields:**
- `type` (string): `"ready"`
- `robot_id` (string): The ID of the robot that is ready.
- `message` (string): A descriptive message (e.g., `"AI model loaded and CUDA warmed up"`).

**Example:**
```json
{
  "type": "ready",
  "robot_id": "R1",
  "message": "AI model loaded and CUDA warmed up"
}
```

### 3.3 Control Command (`type: "control"`)
Sent periodically (typically at 20Hz) to control the robot's motors.

**Fields:**
- `type` (string): `"control"`
- `robot_id` (string): The ID of the robot to be controlled.
- `driveTorque` (float): A normalized value from -1.0 to 1.0 for the torque to apply to the drive wheels.
- `steerAngle` (float): A value specifying the steering angle of the front wheels in radians. In many implementations, this is limited to the range of -0.524 to 0.524, which corresponds to approximately ±30 degrees.

**Example:**
```json
{
  "type": "control",
  "robot_id": "R1",
  "driveTorque": 0.5,
  "steerAngle": -0.262
}
```
> A value of `-0.262 rad` corresponds to approximately **-15 degrees (15 degrees left)**. Negative values are for left turns, positive values for right turns.

### 3.4 Force End (`type: "force_end"`)
Sent when the user manually stops the Python client (e.g., by pressing the 'q' key). This instructs the server to end the race gracefully and send back final metadata.

**Fields:**
- `type` (string): `"force_end"`
- `robot_id` (string): The ID of the robot initiating the stop.
- `message` (string): A descriptive message about why the session is ending.

**Example:**
```json
{
  "type": "force_end",
  "robot_id": "R1",
  "message": "Python client force-ended with 'q' key"
}
```

## 4. Unity (Server) to Python (Client) Messages

### 4.1 Connection Acknowledgment (`type: "connection"`)
The server's response to the client's handshake message.

**Fields:**
- `type` (string): `"connection"`
- `status` (string): The result of the connection attempt (e.g., `"success"`).
- `message` (string): A descriptive message from the server.

**Example:**
```json
{
  "type": "connection",
  "status": "success",
  "message": "Robot R1 connected successfully"
}
```

### 4.2 State of Charge (`type: "soc"`)
Sent periodically from the server to provide the robot's battery level.

**Fields:**
- `type` (string): `"soc"`
- `soc` (float): The current state of charge of the battery, typically from 0.0 to 1.0.

**Example:**
```json
{
  "type": "soc",
  "soc": 0.88
}
```

### 4.3 End-of-Race Metadata (`type: "metadata"`)
Sent once at the end of the race (or after a `force_end` request). It contains all the data logged by Unity during the run.

**Fields:**
- `type` (string): `"metadata"`
- `csv_data` (string): A single string containing the tick-by-tick data for the entire race in CSV format. Newlines and quotes within the string are escaped (`\n`, `\"`).
- `unity_log` (string): A single string containing logs generated by Unity during the session. Newlines are escaped.

### 4.4 Image Data (Binary Message)
In addition to JSON messages, the server streams camera images to the client as raw binary WebSocket messages.
- **Format**: Each message is a byte array representing a single JPEG-encoded image.
- **Frequency**: The frequency of the image stream is configured in the Unity simulation.
- **Handling**: The Python client should receive these as `bytes` and process them separately from the JSON text messages.
