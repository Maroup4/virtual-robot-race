# websocket_server.py
# WebSocket server that talks to Unity:
# - Sends torque commands at 20 Hz (every 50 ms)
# - Receives images (binary) and per-tick metadata (JSON)
# - On race end, receives final race metadata JSON (no images)
#
# Protocol (from Unity):
#   Per tick (50 ms):
#     1) binary: JPG bytes (optional; may be missing on some ticks)
#     2) text:   JSON  {"type":"data","tick":..., "utc_ms":..., "filename":"frame_000123.jpg",
#                       "soc":..., "status":..., "leftTorque":..., "rightTorque":...}
#   On race end:
#     text: final race JSON (DataLogger summary) -> saved as metadata.csv by DataManager
#
# This server:
#   - Buffers the most recent image bytes (no backlog)
#   - Saves the image when the subsequent JSON arrives (so filenames match)
#   - Appends a simple frames_map.csv per tick (optional but useful for audits)
#   - Keeps torque sender and receiver tasks tracked for graceful shutdown

import asyncio
import json
import os
from pathlib import Path
from threading import Event

import websockets
import config

# Data layer
# Expect DataManager with:
#   start_new_run() -> (run_dir, images_dir)
#   save_image_bytes(path: Path, data: bytes)
#   append_frame_map(tick, utc_ms, filename, soc, status, left_tq, right_tq)
#   save_metadata_csv_from_unity_json(obj: dict)
#   flush_frame_map()
from data_manager import DataManager, BASE_DIR
dm = DataManager(BASE_DIR)
run_dir, images_dir = dm.start_new_run()

# Choose control input module
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

# ---- Globals ---------------------------------------------------------------

TORQUE_FILE = os.path.join("data_interactive", "latest_torque.txt")

frame_received_event = Event()              # fire once when the first frame is saved
shutdown_event = asyncio.Event()            # global shutdown signal
connected_websocket = None                  # current client connection
_bg_tasks: set[asyncio.Task] = set()        # track background tasks
_ws_server = None                           # server handle

# Data session objects
_dm: DataManager | None = None
_run_dir: Path | None = None
_images_dir: Path | None = None


def _track(task: asyncio.Task) -> asyncio.Task:
    """Register a background task so we can cancel/await it on shutdown."""
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


# ---- Utility ---------------------------------------------------------------

def write_latest_torque(left: float, right: float) -> None:
    """Write the latest torque to a file so external tools can read it."""
    try:
        os.makedirs(os.path.dirname(TORQUE_FILE), exist_ok=True)
        with open(TORQUE_FILE, "w", encoding="utf-8") as f:
            f.write(f"{left:.4f},{right:.4f}")
    except Exception as e:
        print(f"[Server] Failed to write torque to file: {e}")


# ---- Loops -----------------------------------------------------------------

async def send_torque_data(websocket: websockets.WebSocketServerProtocol) -> None:
    """Send torque command to Unity every 50 ms (20 Hz)."""
    print("[Server] Starting torque data sender...")
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.05)
            msg = {
                "type": "control",
                "leftTorque": control_module.leftTorque,
                "rightTorque": control_module.rightTorque,
            }
            try:
                write_latest_torque(control_module.leftTorque, control_module.rightTorque)
                await websocket.send(json.dumps(msg))
            except websockets.exceptions.ConnectionClosed:
                print("[Server] WebSocket closed. Stopping torque sender.")
                break
    except asyncio.CancelledError:
        # expected on shutdown
        pass
    finally:
        print("[Server] Torque sender loop exited.")


async def receive_stream(websocket: websockets.WebSocketServerProtocol) -> None:
    """Receive binary images and JSON metadata from Unity.

    Contract:
      - Binary messages are JPG bytes (optional per tick).
      - JSON 'type':'data' follows and carries the filename and per-tick meta.
      - Final race summary JSON (DataLogger output) comes at race end (no images).
    """
    global _dm, _images_dir
    print("[Server] Ready to receive data from Unity...")

    last_image_buf: bytes | None = None
    first_frame_saved = False

    try:
        async for message in websocket:
            # Binary = image
            if isinstance(message, (bytes, bytearray)):
                # Keep only the latest image (no backlog).
                last_image_buf = bytes(message)
                continue

            # Text = JSON
            try:
                data = json.loads(message)
            except Exception as e:
                print(f"[Server] Ignoring non-JSON text: {e}")
                continue

            mtype = data.get("type")

            # Per-tick meta
            if mtype == "data":
                tick = data.get("tick")
                utc_ms = data.get("utc_ms")
                filename = data.get("filename") or (f"frame_{int(tick):06d}.jpg" if tick is not None else "no_image")
                soc = data.get("soc", 0.0)
                status = data.get("status", "unknown")
                left_tq = data.get("leftTorque", 0.0)
                right_tq = data.get("rightTorque", 0.0)

                # If we have a pending image, save it now with the provided filename.
                if last_image_buf is not None and filename != "no_image":
                    try:
                        _dm.save_image_bytes(_images_dir / filename, last_image_buf)
                        last_image_buf = None
                        if not first_frame_saved:
                            first_frame_saved = True
                            frame_received_event.set()
                            print("[Server] First frame saved.")
                    except Exception as e:
                        print(f"[Server] Failed to save image {filename}: {e}")

                

            # Final race data (summary from Unity DataLogger)
            elif mtype in ("race_data", "RaceData", None):
                try:
                    print("[Server] Received race metadata.")                    
                    payload = data.get("payload", data)
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except Exception:
                            pass

                    _dm.save_metadata_csv_from_unity_json(payload)

                    
                except Exception as e:
                    print(f"[Server] Failed to save final race metadata: {e}")
                break

            # Connection/control messages
            elif mtype == "connection":
                if data.get("message") == "RaceEnd":
                    
                    pass
                # Handshake or other connection messages are handled in handler()

            else:
                # Unknown message type; ignore or log
                pass

    except websockets.exceptions.ConnectionClosed:
        print("[Server] Client disconnected.")
    except asyncio.CancelledError:
        # expected on shutdown
        pass
    finally:
        print("[Server] Image/SOC reception stopped.")


# ---- Connection handler ----------------------------------------------------

async def handler(websocket: websockets.WebSocketServerProtocol, stop_event: Event) -> None:
    """Handle a new client connection."""
    global connected_websocket, _dm, _run_dir, _images_dir

    connected_websocket = websocket
    frame_received_event.clear()

    print("[Server] Client connected.")

    # Start a new run directory (images/, frames_map.csv, etc.)
    try:
        base_dir = Path(__file__).parent
        _dm = DataManager(base_dir=base_dir)
        _run_dir, _images_dir = _dm.start_new_run()
    except Exception as e:
        print(f"[Server] DataManager init failed: {e}")
        _dm = None
        _run_dir = None
        _images_dir = None

    # Handshake
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

    # Start background loops
    send_task = _track(asyncio.create_task(send_torque_data(websocket)))
    recv_task = _track(asyncio.create_task(receive_stream(websocket)))

    try:
        # Wait until the receive loop finishes (client closed or race end)
        await recv_task
    finally:
        print("[Server] Connection closed (handler cleanup).")

        # Cancel both loops (if still running)
        for t in (send_task, recv_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(send_task, recv_task, return_exceptions=True)

        # Signal shutdown to main
        shutdown_event.set()
        stop_event.set()
        connected_websocket = None


# ---- Public API for main.py ------------------------------------------------

async def start_server(stop_event: Event) -> None:
    """Launch the WebSocket server and wait until stop_event is set."""
    global _ws_server

    host, port = config.HOST, config.PORT
    _ws_server = await websockets.serve(lambda ws: handler(ws, stop_event), host, port)
    print(f"[Server] WebSocket server running at ws://{host}:{port}")

    # Wait until the blocking threading.Event is set (from main.py)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stop_event.wait)


async def shutdown_server() -> None:
    """Gracefully stop server & background tasks (no pending warnings)."""
    global _ws_server

    print("[Server] Shutting down server...")

    # 1) Cancel background tasks
    if _bg_tasks:
        for t in list(_bg_tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(*list(_bg_tasks), return_exceptions=True)
        _bg_tasks.clear()

    # 2) Close active connection (best-effort)
    try:
        if connected_websocket and not connected_websocket.closed:
            await connected_websocket.close()
    except Exception:
        pass

    # 3) Close server
    if _ws_server is not None:
        _ws_server.close()
        try:
            await _ws_server.wait_closed()
        except Exception:
            pass
        _ws_server = None
        print("[Server] WebSocket server closed.")


async def send_race_end_signal() -> None:
    """Send a 'RaceEnd' command to Unity (optional helper)."""
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
            print(f"[Server] Sent RaceEnd signal → name={config.NAME}, mode={config.MODE}")
        except Exception as e:
            print(f"[Server] Failed to send RaceEnd: {e}")
    else:
        print("[Server] No client connected to send RaceEnd.")


async def send_control_command_async(left: float, right: float) -> None:
    """Send a manual torque command (used in 'table' mode tools)."""
    if connected_websocket:
        try:
            message = json.dumps({
                "type": "control",
                "leftTorque": left,
                "rightTorque": right,
            })
            await connected_websocket.send(message)
            print(f"[Server] Sent manual torque: L={left}, R={right}")
        except Exception as e:
            print(f"[Server] Error sending torque: {e}")
    else:
        print("[Server] No connected client.")
