# websocket_server.py
# WebSocket server that talks to Unity:
# - Sends control commands at 20 Hz (driveTorque, steerAngle)
# - Receives image frames (binary) and telemetry JSON (type=data)
# - Receives final race summary JSON (type=race_data) and saves metadata.csv
#
# Protocol (Unity -> Python):
#   Per tick (50ms):
#     1) binary: JPG bytes (optional)
#     2) text:   {"type":"data","tick":...,"utc_ms":...,
#                 "filename":"frame_000123.jpg","soc":...,
#                 "status":...,"driveTorque":...,"steerAngle":...}
#
#   On race end:
#     {"type":"race_data","payload":{...DataLogger JSON...}}
#
# Python side:
#   - Buffers only the latest image, saves it when JSON arrives
#   - Creates frames_map.csv and metadata.csv under training_data/run_YYYYMMDD_HHMMSS/
#   - Handles graceful shutdown

import asyncio
import json
import os
from pathlib import Path
from threading import Event
import websockets
import config
from data_manager import DataManager, BASE_DIR

# ---------------------------------------------------------------------------
# Setup data session
# ---------------------------------------------------------------------------
dm = DataManager(BASE_DIR)
run_dir, images_dir = dm.start_new_run()

# Control input module selection
if config.MODE == "keyboard":
    import keyboard_input as control_module
elif config.MODE == "table":
    import table_input as control_module
elif config.MODE == "rule_based":
    import rule_based_input as control_module
elif config.MODE == "ai":
    import inference_input as control_module
else:
    raise ValueError(f"[Server] Unknown control mode: {config.MODE}")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
TORQUE_FILE = os.path.join("data_interactive", "latest_torque.txt")

frame_received_event = Event()
shutdown_event = asyncio.Event()
connected_websocket = None
_bg_tasks: set[asyncio.Task] = set()
_ws_server = None

_dm: DataManager | None = None
_run_dir: Path | None = None
_images_dir: Path | None = None


def _track(task: asyncio.Task) -> asyncio.Task:
    """Register background task to allow cancellation on shutdown."""
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def write_latest_torque(drive: float, steer: float) -> None:
    """Write latest drive/steer to a file for external monitoring."""
    try:
        os.makedirs(os.path.dirname(TORQUE_FILE), exist_ok=True)
        with open(TORQUE_FILE, "w", encoding="utf-8") as f:
            f.write(f"{drive:.4f},{steer:.4f}")
    except Exception as e:
        print(f"[Server] Failed to write torque file: {e}")


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------
async def send_control_loop(websocket: websockets.WebSocketServerProtocol) -> None:
    """Send control commands (20Hz) to Unity."""
    print("[Server] Starting control-data sender...")
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.05)

            msg = None
            if hasattr(control_module, "get_latest_command"):
                try:
                    msg = control_module.get_latest_command()
                    msg.setdefault("type", "control")
                    msg.setdefault("robot_id", "R1")
                except Exception as e:
                    print(f"[Server] get_latest_command() failed: {e}")

            if msg is None:
                # fallback (backward compatibility)
                drive = float(getattr(control_module, "driveTorque", 0.0))
                steer = float(getattr(control_module, "steerAngle", 0.0))
                msg = {
                    "type": "control",
                    "robot_id": "R1",
                    "driveTorque": drive,
                    "steerAngle": steer,
                }

            try:
                await websocket.send(json.dumps(msg))
            except websockets.exceptions.ConnectionClosed:
                print("[Server] WebSocket closed. Stopping sender.")
                break
    except asyncio.CancelledError:
        pass
    finally:
        print("[Server] Control-data sender exited.")


async def receive_stream(websocket: websockets.WebSocketServerProtocol) -> None:
    """Receive binary image and per-tick JSON from Unity."""
    global _dm, _images_dir
    print("[Server] Ready to receive data from Unity...")

    last_image_buf: bytes | None = None
    first_frame_saved = False

    try:
        async for message in websocket:
            # --- Binary image ---
            if isinstance(message, (bytes, bytearray)):
                last_image_buf = bytes(message)
                continue

            # --- Text (JSON) ---
            try:
                data = json.loads(message)
            except Exception as e:
                print(f"[Server] Invalid JSON skipped: {e}")
                continue

            mtype = data.get("type", "")

            # --- Tick data from Unity ---
            if mtype == "data":
                tick = data.get("tick")
                utc_ms = data.get("utc_ms")
                filename = data.get("filename") or f"frame_{int(tick):06d}.jpg"

                # Save the latest image
                if last_image_buf is not None and filename:
                    try:
                        _dm.save_image_bytes(_images_dir / filename, last_image_buf)
                        last_image_buf = None
                        if not first_frame_saved:
                            first_frame_saved = True
                            frame_received_event.set()
                            print("[Server] First frame saved.")
                    except Exception as e:
                        print(f"[Server] Failed to save image {filename}: {e}")

                # Optional: log per-frame info
                drive = data.get("driveTorque")
                steer = data.get("steerAngle")
                soc = data.get("soc")
                status = data.get("status", "")
                _dm.append_frame_map(tick, utc_ms, filename, soc, status, drive, steer)

            # --- Final race summary ---
            elif mtype in ("race_data", "RaceData"):
                try:
                    print("[Server] Received race metadata (final summary).")
                    payload = data.get("payload", data)
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    _dm.save_metadata_csv_from_unity_json(payload)
                except Exception as e:
                    print(f"[Server] Failed to save final race metadata: {e}")
                break

            # --- Explicit race end ---
            elif mtype == "connection" and data.get("message") == "RaceEnd":
                print("[Server] RaceEnd message received.")
                break

            else:
                # Unknown / heartbeat etc.
                pass

    except websockets.exceptions.ConnectionClosed:
        print("[Server] Client disconnected.")
    except asyncio.CancelledError:
        pass
    finally:
        print("[Server] Reception loop stopped.")


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------
async def handler(websocket: websockets.WebSocketServerProtocol, stop_event: Event) -> None:
    """Handle new WebSocket connection from Unity."""
    global connected_websocket, _dm, _run_dir, _images_dir

    connected_websocket = websocket
    frame_received_event.clear()
    print("[Server] Client connected.")

    # start keyboard listener if exists
    try:
        if hasattr(control_module, "start_listener"):
            control_module.start_listener()
    except Exception as e:
        print(f"[Server] Keyboard listener start failed: {e}")

    # start data manager session
    try:
        base_dir = Path(__file__).parent
        _dm = DataManager(base_dir)
        _run_dir, _images_dir = _dm.start_new_run()
    except Exception as e:
        print(f"[Server] DataManager init failed: {e}")

    # send handshake
    try:
        handshake = {
            "type": "connection",
            "status": "success",
            "name": config.NAME,
            "mode": config.MODE,
            "mode_num": config.MODE_NUM,
            "race_flag": config.RACE_FLAG,
        }
        await websocket.send(json.dumps(handshake))
        print(f"[Server] Sent handshake → name={config.NAME}, mode={config.MODE}, race_flag={config.RACE_FLAG}")
    except websockets.exceptions.ConnectionClosed:
        print("[Server] Connection failed during handshake.")
        return

    send_task = _track(asyncio.create_task(send_control_loop(websocket)))
    recv_task = _track(asyncio.create_task(receive_stream(websocket)))

    try:
        await recv_task
    finally:
        print("[Server] Connection closed (cleanup).")
        for t in (send_task, recv_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(send_task, recv_task, return_exceptions=True)

        # stop keyboard listener if exists
        try:
            if hasattr(control_module, "stop_listener"):
                control_module.stop_listener()
        except Exception as e:
            print(f"[Server] Keyboard listener stop failed: {e}")

        shutdown_event.set()
        stop_event.set()
        connected_websocket = None


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------
async def start_server(stop_event: Event) -> None:
    """Launch WebSocket server and wait for stop_event."""
    global _ws_server
    host, port = config.HOST, config.PORT
    _ws_server = await websockets.serve(lambda ws: handler(ws, stop_event), host, port)
    print(f"[Server] WebSocket server running at ws://{host}:{port}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stop_event.wait)


async def shutdown_server() -> None:
    """Gracefully shut down server and background tasks."""
    global _ws_server
    print("[Server] Shutting down...")

    # cancel background tasks
    if _bg_tasks:
        for t in list(_bg_tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(*_bg_tasks, return_exceptions=True)
        _bg_tasks.clear()

    # close active websocket
    try:
        if connected_websocket and not connected_websocket.closed:
            await connected_websocket.close()
    except Exception:
        pass

    # close server
    if _ws_server is not None:
        _ws_server.close()
        try:
            await _ws_server.wait_closed()
        except Exception:
            pass
        _ws_server = None
        print("[Server] Server closed.")


async def send_race_end_signal() -> None:
    """Send RaceEnd command to Unity."""
    if connected_websocket:
        try:
            message = json.dumps({
                "type": "connection",
                "message": "RaceEnd",
                "name": config.NAME,
                "mode": config.MODE,
                "race_flag": config.RACE_FLAG,
            })
            await connected_websocket.send(message)
            print("[Server] Sent RaceEnd signal.")
        except Exception as e:
            print(f"[Server] Failed to send RaceEnd: {e}")
    else:
        print("[Server] No connected client.")


async def send_control_command_async(drive: float, steer: float, robot_id: str = "R1") -> None:
    """Send manual control command (drive + steer)."""
    if connected_websocket:
        try:
            message = json.dumps({
                "type": "control",
                "robot_id": robot_id,
                "driveTorque": float(drive),
                "steerAngle": float(steer),
            })
            await connected_websocket.send(message)
            print(f"[Server] Sent manual control → drive={drive:.3f}, steer={steer:.3f}")
        except Exception as e:
            print(f"[Server] Error sending control: {e}")
    else:
        print("[Server] No connected client.")
