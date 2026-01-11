# main.py
# New architecture version (Unity=Server, Python=Client)
# Entry point that orchestrates:
#  - Unity process launch (WebSocket Server)
#  - RobotWebSocketClient (Python Client)
#  - Input pipeline (keyboard / table / rule_based / AI)
#  - Post-race video build (MP4)
# Multi-robot support: Loads settings from Robot{N}/robot_config.txt and uses Robot{N}/ modules

import asyncio
import threading
import subprocess
import os
import sys
import time
from typing import Optional
from pathlib import Path
import pandas as pd
import shutil

import config_loader
from websocket_client import RobotWebSocketClient
import data_manager as dm


# -------------------------
# Terminal Log Capture
# -------------------------
class LogCapture:
    """Captures stdout/stderr while still printing to terminal."""
    def __init__(self, original_stream):
        self.logs = []
        self.original = original_stream

    def write(self, text):
        if text:
            self.logs.append(text)
            self.original.write(text)

    def flush(self):
        self.original.flush()

    def get_log_text(self) -> str:
        return "".join(self.logs)


# Install log capture at module load time
_stdout_capture = LogCapture(sys.stdout)
_stderr_capture = LogCapture(sys.stderr)
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture


def get_terminal_log() -> str:
    """Get all captured terminal output."""
    # Combine stdout and stderr logs
    stdout_log = _stdout_capture.get_log_text()
    stderr_log = _stderr_capture.get_log_text()
    if stderr_log:
        return stdout_log + "\n=== STDERR ===\n" + stderr_log
    return stdout_log


# Register the terminal log getter with data_manager
dm.register_terminal_log_getter(get_terminal_log)

# Lazy-import only when used
import make_video
from data_manager import read_last_run_dir

# Shared stop signal
stop_event = threading.Event()

# Global client instances (dictionary keyed by robot_id)
robot_clients: dict[str, RobotWebSocketClient] = {}


def launch_unity_exe() -> Optional[subprocess.Popen]:
    """Launch the built Unity executable if it exists."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base_dir, "Windows", "VirtualRobotRace_Beta.exe")

    if os.path.exists(exe_path):
        print(f"[Main] Launching Unity server: {exe_path}")
        proc = subprocess.Popen([exe_path], shell=False)
        print(f"[Main] Unity server launched (PID: {proc.pid})")
        return proc
    else:
        print(f"[Main] Unity .exe not found at: {exe_path}")
        return None


async def wait_for_unity_server(server_url: str, timeout: float = 30.0) -> bool:
    """
    Wait for Unity WebSocket server to be ready.
    Attempts to connect with exponential backoff.
    """
    import websockets

    print(f"[Main] Waiting for Unity server at {server_url}...")
    start_time = time.time()
    delay = 0.5  # Initial delay

    while time.time() - start_time < timeout:
        try:
            async with websockets.connect(server_url) as ws:
                await ws.close()
                print(f"[Main] Unity server is ready!")
                return True
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2.0)  # Exponential backoff, max 2s

    print(f"[Main] Unity server did not start within {timeout}s")
    return False


def auto_rename_images(run_dir: Path) -> bool:
    """
    Automatically rename images to match metadata.csv filenames.
    This fixes the tick-based vs sequential naming mismatch.

    Args:
        run_dir: Path to run_YYYYMMDD_HHMMSS directory

    Returns:
        True if successful or already correct, False if failed
    """
    csv_path = run_dir / "metadata.csv"
    images_dir = run_dir / "images"

    # Basic validation
    if not csv_path.exists() or not images_dir.exists():
        print("[Main] Auto-rename skipped: metadata.csv or images/ not found")
        return False

    # Load metadata
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Main] Auto-rename failed: Cannot load metadata.csv - {e}")
        return False

    # Get image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.jpg'])

    metadata_count = len(df)
    image_count = len(image_files)

    # Check count match
    if metadata_count != image_count:
        print(f"[Main] Auto-rename failed: Count mismatch (metadata={metadata_count}, images={image_count})")
        return False

    # Build rename mapping
    rename_map = []
    for idx, row in df.iterrows():
        expected_name = row['filename']
        actual_name = image_files[idx].name
        if actual_name != expected_name:
            rename_map.append((image_files[idx], images_dir / expected_name))

    # If already correct, skip
    if len(rename_map) == 0:
        print("[Main] Auto-rename: All filenames already correct")
        return True

    print(f"[Main] Auto-rename: Renaming {len(rename_map)} files to match metadata...")

    # Check for existing backup
    backup_dir = run_dir / "images_backup"
    if backup_dir.exists():
        print(f"[Main] Auto-rename skipped: Backup already exists (images_backup/)")
        return False

    try:
        # Create backup
        shutil.copytree(images_dir, backup_dir)
        print(f"[Main] Auto-rename: Backup created → images_backup/")

        # Phase 1: Rename to temporary names (avoid conflicts)
        temp_map = []
        for old_path, new_path in rename_map:
            temp_path = old_path.with_suffix('.tmp.jpg')
            old_path.rename(temp_path)
            temp_map.append((temp_path, new_path))

        # Phase 2: Rename to final names
        for temp_path, new_path in temp_map:
            temp_path.rename(new_path)

        # Verify
        missing = []
        for _, row in df.iterrows():
            expected_path = images_dir / row['filename']
            if not expected_path.exists():
                missing.append(row['filename'])

        if missing:
            print(f"[Main] Auto-rename failed: {len(missing)} files missing after rename")
            return False

        print(f"[Main] Auto-rename: Successfully renamed {len(rename_map)} files")

        # Success! Remove backup to save disk space
        try:
            shutil.rmtree(backup_dir)
            print(f"[Main] Auto-rename: Backup removed (rename successful)")
        except Exception as cleanup_error:
            print(f"[Main] Auto-rename: Warning - Could not remove backup: {cleanup_error}")

        return True

    except Exception as e:
        print(f"[Main] Auto-rename failed: {e}")
        print(f"[Main] Auto-rename: Backup preserved at images_backup/ for recovery")
        return False


async def build_video_and_open_explorer(robot_config: dict) -> None:
    """Post-race pipeline: Auto-rename images, then build MP4 from the latest run's images."""
    if not robot_config.get("AUTO_MAKE_VIDEO", 1):
        print("[Main] AUTO_MAKE_VIDEO=0 → Skip video pipeline.")
        return

    # Check if DATA_SAVE is enabled
    if not robot_config.get("DATA_SAVE", 1):
        print("[Main] WARNING: AUTO_MAKE_VIDEO=1 but DATA_SAVE=0. Cannot create video without saved images. Skipping video pipeline.")
        return

    robot_id = robot_config.get("ROBOT_ID", "R1")
    run_dir = read_last_run_dir(robot_id)
    if not run_dir:
        print("[Main] Post-race video pipeline skipped: last_run_dir not found.")
        return

    images_dir = run_dir / "images"
    if not images_dir.exists():
        print(f"[Main] Post-race video pipeline skipped: images dir not found → {images_dir}")
        return

    # Auto-rename images to match metadata.csv BEFORE creating video
    auto_rename_images(run_dir)

    out_path = run_dir / "output_video.mp4"
    fps = robot_config.get("VIDEO_FPS", 20)
    infer = robot_config.get("INFER_FPS", 1)

    print(f"[Main] Building MP4 → {out_path} (fps={fps}, infer_fps={infer})")

    loop = asyncio.get_running_loop()

    def _encode():
        make_video.images_to_video_ffmpeg(str(images_dir), str(out_path), fps=fps, infer_fps=infer)

    await loop.run_in_executor(None, _encode)

    # Note: OPEN_EXPLORER_ON_VIDEO was removed from robot_config
    # Always try to open for now (Windows-specific behavior)
    try:
        if sys.platform.startswith("win") and out_path.exists():
            subprocess.Popen(["explorer", f"/select,{str(out_path)}"])
            print("[Main] Explorer opened with the MP4 selected.")
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(run_dir)])
            print("[Main] Opened run directory in file manager.")
    except Exception as e:
        print(f"[Main] Failed to open file manager: {e}")

    if robot_config.get("DATA_SAVE", 1) == 0:
        try:
            if images_dir.exists():
                count = 0
                for p in images_dir.glob("*.jpg"):
                    try:
                        p.unlink(missing_ok=True)
                        count += 1
                    except Exception as e:
                        print(f"[Main] Failed to delete {p.name}: {e}")
                print(f"[Main] DATA_SAVE=0 → Deleted {count} JPG files after video export.")
        except Exception as e:
            print(f"[Main] Cleanup after video failed: {e}")


async def run_control_module(client: RobotWebSocketClient, mode: str, robot_num: int, robot_config: dict):
    """
    Run the control module based on mode string.
    This integrates keyboard/ai/rule_based control with the WebSocket client.
    Imports from Robot{N}/ directory.
    """
    print(f"[Main] Starting control module: {mode} (Robot{robot_num})")

    # Check if keyboard is disabled for this robot
    if mode == "keyboard" and robot_config.get('KEYBOARD_DISABLED', False):
        print(f"[Main] Robot{robot_num} keyboard control is DISABLED (another robot has priority)")
        print(f"[Main] Robot{robot_num} will send zero control commands")
        # Send zero commands indefinitely
        while not stop_event.is_set():
            control_msg = {
                "type": "control",
                "robot_id": client.robot_id,
                "driveTorque": 0.0,
                "steerAngle": 0.0
            }
            await client.send_json(control_msg)
            await asyncio.sleep(0.05)  # 20Hz
        return

    # Import from Robot{N}/ directory using importlib for explicit module loading
    import importlib.util
    robot_dir = Path(f"Robot{robot_num}")

    try:
        if mode == "keyboard":
            # Load keyboard_input from Robot{N}/ directory explicitly
            module_file = robot_dir / "keyboard_input.py"
            spec = importlib.util.spec_from_file_location(
                f"Robot{robot_num}.keyboard_input",
                module_file
            )
            keyboard_input = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(keyboard_input)

            keyboard_input.start_listener()

            # Poll keyboard input and send to Unity
            while not stop_event.is_set():
                try:
                    cmd = keyboard_input.get_latest_command()
                    drive = cmd.get("driveTorque", 0.0)
                    steer = cmd.get("steerAngle", 0.0)

                    # Send control command to Unity
                    control_msg = {
                        "type": "control",
                        "robot_id": client.robot_id,
                        "driveTorque": drive,
                        "steerAngle": steer
                    }
                    await client.send_json(control_msg)

                except Exception as e:
                    print(f"[Main] Keyboard control error: {e}")

                await asyncio.sleep(0.05)  # 20Hz

            keyboard_input.stop_listener()

        elif mode == "ai":
            # Use preloaded inference module if available, otherwise load fresh
            inference_input = robot_config.get('_preloaded_inference_module')
            if inference_input is None:
                # Fallback: Load inference_input from Robot{N}/ directory explicitly
                module_file = robot_dir / "inference_input.py"
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"Robot{robot_num}.inference_input",
                        module_file
                    )
                    inference_input = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(inference_input)
                    # Preload model if not already done
                    inference_input.preload_model()
                except Exception as e:
                    print(f"[Main] Failed to load inference_input: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                print(f"[Main] Using preloaded inference module for Robot{robot_num}")

            # AI mode: Poll AI inference and send to Unity
            print("[Main] AI mode: Autonomous driving with neural network")

            # Poll AI inference and send to Unity
            while not stop_event.is_set():
                try:
                    # Update AI model (loads image, runs inference, updates driveTorque/steerAngle)
                    if not inference_input.update():
                        # If update returns False, stop
                        print("[Main] AI inference requested stop")
                        break

                    # Get latest command
                    cmd = inference_input.get_latest_command()
                    drive = cmd.get("driveTorque", 0.0)
                    steer = cmd.get("steerAngle", 0.0)

                    # Send control command to Unity
                    control_msg = {
                        "type": "control",
                        "robot_id": client.robot_id,
                        "driveTorque": drive,
                        "steerAngle": steer
                    }
                    await client.send_json(control_msg)

                except Exception as e:
                    print(f"[Main] AI control error: {e}")

                await asyncio.sleep(0.05)  # 20Hz

        elif mode == "rule_based":
            # Load rule_based_input from Robot{N}/ directory explicitly
            module_file = robot_dir / "rule_based_input.py"
            spec = importlib.util.spec_from_file_location(
                f"Robot{robot_num}.rule_based_input",
                module_file
            )
            rule_based_input = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rule_based_input)

            # Rule-based mode: Poll lane detection and driver model
            print("[Main] Rule-based mode: Autonomous lane following")

            # Poll rule-based control and send to Unity
            while not stop_event.is_set():
                try:
                    # Update rule-based controller (processes image, detects lane, calculates control)
                    if not rule_based_input.update():
                        # If update returns False, stop
                        print("[Main] Rule-based control requested stop")
                        break

                    # Get latest command
                    cmd = rule_based_input.get_latest_command()
                    drive = cmd.get("driveTorque", 0.0)
                    steer = cmd.get("steerAngle", 0.0)

                    # Send control command to Unity
                    control_msg = {
                        "type": "control",
                        "robot_id": client.robot_id,
                        "driveTorque": drive,
                        "steerAngle": steer
                    }
                    await client.send_json(control_msg)

                except Exception as e:
                    print(f"[Main] Rule-based control error: {e}")

                await asyncio.sleep(0.05)  # 20Hz

        elif mode == "table":
            # Load table_input from Robot{N}/ directory explicitly
            module_file = robot_dir / "table_input.py"
            spec = importlib.util.spec_from_file_location(
                f"Robot{robot_num}.table_input",
                module_file
            )
            table_input = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(table_input)

            # Table mode: Poll CSV data and send to Unity
            print("[Main] Table mode: Sending commands from CSV")

            # Poll table input and send to Unity
            while not stop_event.is_set():
                try:
                    cmd = table_input.get_latest_command()
                    drive = cmd.get("driveTorque", 0.0)
                    steer = cmd.get("steerAngle", 0.0)

                    # Send control command to Unity
                    control_msg = {
                        "type": "control",
                        "robot_id": client.robot_id,
                        "driveTorque": drive,
                        "steerAngle": steer
                    }
                    await client.send_json(control_msg)

                    # Show progress every 100 commands
                    progress = table_input.get_progress()
                    if progress[0] % 100 == 0:
                        print(f"[Main] Table mode progress: {progress[0]}/{progress[1]}")

                    # If both drive and steer are 0, we might be at end of CSV
                    if drive == 0.0 and steer == 0.0:
                        # Check if we're at the end
                        if progress[0] >= progress[1]:
                            print(f"[Main] Table mode: End of CSV reached")
                            break

                except Exception as e:
                    print(f"[Main] Table control error: {e}")

                await asyncio.sleep(0.05)  # 20Hz

        elif mode == "smartphone":
            # Smartphone mode: Control is handled by smartphone_server
            # This mode just keeps the connection alive while smartphone_server forwards control
            print("[Main] Smartphone mode: Control via smartphone (waiting for smartphone_server)")

            # Keep alive - control is forwarded by smartphone_server
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

        else:
            print(f"[Main] Unknown MODE: {mode}")

    finally:
        # Cleanup is handled automatically with importlib approach
        pass


async def keyboard_monitor() -> None:
    """Monitor keyboard for 'q' key press in background"""
    import sys

    print("[Main] ========================================")
    print("[Main] Keyboard Monitor Starting")
    print("[Main] Press 'q' at any time to force stop")
    print("[Main] ========================================")

    # Try keyboard library first (requires admin on Windows sometimes)
    keyboard_lib_works = False
    try:
        import keyboard
        print("[Main] [KeyMon] Testing keyboard library...")
        # Test if keyboard library works
        try:
            keyboard.is_pressed("q")
            keyboard_lib_works = True
            print("[Main] [KeyMon] ✓ keyboard library working")
        except Exception as e:
            print(f"[Main] [KeyMon] ✗ keyboard library test failed: {e}")
            print("[Main] [KeyMon] Will use msvcrt instead")
    except ImportError:
        print("[Main] [KeyMon] keyboard library not installed")

    if keyboard_lib_works:
        print("[Main] [KeyMon] Using keyboard library for monitoring")
        check_count = 0
        while not stop_event.is_set():
            try:
                if keyboard.is_pressed("q"):
                    print("[Main] [KeyMon] ✓✓✓ 'q' KEY DETECTED ✓✓✓")
                    print("[Main] [KeyMon] Setting stop_event...")
                    stop_event.set()
                    break
                # Debug: Print heartbeat every 100 checks (~5 seconds)
                check_count += 1
                if check_count % 100 == 0:
                    print(f"[Main] [KeyMon] Heartbeat: {check_count} checks, still monitoring...")
            except Exception as e:
                print(f"[Main] [KeyMon] ERROR during monitoring: {e}")
                break
            await asyncio.sleep(0.05)  # Check every 50ms

    # Fallback: Use msvcrt on Windows
    if not stop_event.is_set() and not keyboard_lib_works:
        try:
            if sys.platform == 'win32':
                import msvcrt
                print("[Main] [KeyMon] Using msvcrt for monitoring (Windows fallback)")
                check_count = 0
                while not stop_event.is_set():
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        print(f"[Main] [KeyMon] Key pressed: {key}")
                        if key == b'q' or key == b'Q':
                            print("[Main] [KeyMon] ✓✓✓ 'q' KEY DETECTED ✓✓✓")
                            print("[Main] [KeyMon] Setting stop_event...")
                            stop_event.set()
                            break
                    # Debug: Print heartbeat every 100 checks (~5 seconds)
                    check_count += 1
                    if check_count % 100 == 0:
                        print(f"[Main] [KeyMon] Heartbeat: {check_count} checks, still monitoring...")
                    await asyncio.sleep(0.05)
            else:
                print("[Main] [KeyMon] Not on Windows - no fallback available")
        except Exception as e:
            print(f"[Main] [KeyMon] ERROR: {e}")

    print("[Main] [KeyMon] Monitor stopped")


async def main() -> None:
    """
    Main orchestration (Multi-robot version):
      1) Load all active robot configs
      2) Launch Unity (WebSocketServer)
      3) Wait for Unity to be ready
      4) Connect all RobotWebSocketClients
      5) Start control modules for each robot concurrently
      6) Wait for stop_event
      7) Graceful shutdown
      8) Build videos (for each robot)
    """
    global robot_clients

    print("[Main] Starting new architecture (Unity=Server, Python=Client)...")
    print(f"[Main] Active robots: {config_loader.ACTIVE_ROBOTS}")

    # 1) Load all active robot configs
    robot_configs = {}
    keyboard_robot = None  # Track which robot gets keyboard control

    for robot_num in config_loader.ACTIVE_ROBOTS:
        robot_configs[robot_num] = config_loader.get_robot_config(robot_num)
        rc = robot_configs[robot_num]
        mode_num = rc.get('MODE_NUM', 1)
        mode_str = config_loader.get_mode_string(mode_num)

        print(f"[Main] Robot{robot_num} config loaded:")
        print(f"  - ROBOT_ID: {rc.get('ROBOT_ID')}")
        print(f"  - MODE: {mode_str} (MODE_NUM={mode_num})")
        print(f"  - NAME: {rc.get('NAME', 'Player0000')}")
        print(f"  - RACE_FLAG: {rc.get('RACE_FLAG', 1)}")

        # Check for keyboard mode conflict
        if mode_str == "keyboard":
            if keyboard_robot is None:
                keyboard_robot = robot_num
                print(f"[Main] ✓ Robot{robot_num} will use keyboard control")
            else:
                print(f"[Main] ⚠ WARNING: Robot{robot_num} is set to keyboard mode, but Robot{keyboard_robot} already has keyboard control!")
                print(f"[Main] ⚠ Robot{robot_num} will be DISABLED (no control input)")
                # Mark this robot as disabled
                rc['KEYBOARD_DISABLED'] = True

    unity_proc = None

    try:
        # 2) Launch Unity
        if config_loader.DEBUG_MODE == 0:
            unity_proc = launch_unity_exe()
            if not unity_proc:
                print("[Main] Failed to launch Unity. Exiting.")
                return
        else:
            print("[Main] DEBUG_MODE = 1 → Please launch Unity manually.")

        # 3) Wait for Unity server to be ready
        server_url = f"ws://{config_loader.HOST}:{config_loader.PORT}/robot"
        if not await wait_for_unity_server(server_url, timeout=30.0):
            print("[Main] Unity server did not start. Exiting.")
            return

        # 4) Create and connect all clients (but don't start control yet)
        all_tasks = []
        robot_modes = {}  # Store mode and config for each robot

        # Phase 1: Connect all robots
        for i, robot_num in enumerate(config_loader.ACTIVE_ROBOTS):
            rc = robot_configs[robot_num]
            robot_id = rc.get("ROBOT_ID", f"R{robot_num}")
            mode_num = rc.get("MODE_NUM", 1)
            mode = config_loader.get_mode_string(mode_num)

            # Create client (only first robot sends active_robots info)
            client = RobotWebSocketClient(
                robot_id=robot_id,
                server_url=server_url,
                robot_config=rc,
                active_robots=config_loader.ACTIVE_ROBOTS if i == 0 else None  # First robot sends the list
            )
            robot_clients[robot_id] = client
            robot_modes[robot_id] = (mode, robot_num, rc)  # Store config too

            # Connect
            await client.connect()
            print(f"[Main] Robot{robot_num} ({robot_id}) connected")

        print(f"[Main] All {len(robot_clients)} robots connected.")

        # Phase 1.5: Preload AI models BEFORE starting control loops
        # This prevents model loading delays during the start signal sequence
        import importlib.util  # Import here for Phase 1.5
        print("[Main] Preloading AI models for all AI-mode robots...")
        for robot_id, (mode, robot_num, rc) in robot_modes.items():
            if mode == "ai":
                robot_dir = Path(f"Robot{robot_num}")
                module_file = robot_dir / "inference_input.py"
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"Robot{robot_num}.inference_input_preload",
                        module_file
                    )
                    inference_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(inference_module)
                    inference_module.preload_model()
                    # Store the preloaded module for later use
                    rc['_preloaded_inference_module'] = inference_module
                except Exception as e:
                    print(f"[Main] WARNING: Failed to preload model for Robot{robot_num}: {e}")
                    import traceback
                    traceback.print_exc()
        print("[Main] AI model preloading complete.")

        # Phase 1.6: CUDA warmup for AI robots
        # This eliminates 10+ second delay on first inference by initializing GPU kernels
        print("[Main] Warming up CUDA for AI-mode robots...")
        for robot_id, (mode, robot_num, rc) in robot_modes.items():
            if mode == "ai":
                inference_module = rc.get('_preloaded_inference_module')
                if inference_module:
                    try:
                        inference_module.warmup_cuda()
                    except Exception as e:
                        print(f"[Main] WARNING: CUDA warmup failed for Robot{robot_num}: {e}")
        print("[Main] CUDA warmup complete.")

        # Start keyboard monitor early (before smartphone wait)
        print("[Main] Starting keyboard monitor...")
        keyboard_task = asyncio.create_task(keyboard_monitor())
        all_tasks.append(keyboard_task)

        # Phase 1.7: Start smartphone server if any robot uses smartphone mode
        # NOTE: This must happen BEFORE sending ready signals to Unity
        smartphone_server = None
        smartphone_modes = {robot_id: mode for robot_id, (mode, _, _) in robot_modes.items() if mode == "smartphone"}

        if smartphone_modes:
            print(f"[Main] Starting smartphone server for {len(smartphone_modes)} robot(s)...")
            from smartphone_server import SmartphoneServer

            smartphone_server = SmartphoneServer(port=8080)

            # Register robots that use smartphone mode
            for robot_id in smartphone_modes.keys():
                client = robot_clients[robot_id]
                smartphone_server.register_robot(robot_id, client)

            # Start server
            await smartphone_server.start()
            print("[Main] Smartphone server started and ready for connections")

            # Wait for all smartphone controllers to pass readiness test
            print("[Main] " + "=" * 50)
            print("[Main] Waiting for smartphone connection confirmation...")
            print("[Main] Instructions:")
            print("[Main]   1. Scan QR code with your smartphone")
            print("[Main]   2. Press BOTH L+R buttons at the same time")
            print("[Main]   3. Race will start automatically when confirmed")
            print("[Main]   (Press 'q' to cancel)")
            print("[Main] " + "=" * 50)

            all_ready = await smartphone_server.wait_for_all_ready(timeout=300.0, stop_event=stop_event)

            if stop_event.is_set():
                print("[Main] Cancelled by user ('q' key pressed)")
                raise KeyboardInterrupt("User cancelled with 'q' key")

            if not all_ready:
                print("[Main] ERROR: Not all robots confirmed connection within timeout")
                print("[Main] Proceeding anyway, but connection may be unstable")
            else:
                print("[Main] ✓ All smartphones confirmed! Starting race sequence...")
                print("[Main] Waiting 3 seconds before starting...")
                await asyncio.sleep(3.0)

        # Phase 1.8: Send ready signals to Unity
        # For smartphone mode, this happens AFTER smartphone connection is confirmed
        # Unity will wait for all robots to be ready before starting the race
        print("[Main] Sending ready signals to Unity...")
        for robot_id, client in robot_clients.items():
            try:
                await client.send_ready_signal()
            except Exception as e:
                print(f"[Main] WARNING: Failed to send ready signal for {robot_id}: {e}")
        print("[Main] All ready signals sent. Unity will start race now...")

        print("[Main] Starting control modules simultaneously...")

        # Phase 2: Start all control modules and receive loops simultaneously
        for robot_id, client in robot_clients.items():
            mode, robot_num, rc = robot_modes[robot_id]

            # 5) Start control module and receive loop for this robot
            control_task = asyncio.create_task(
                run_control_module(client, mode, robot_num, rc)
            )
            receive_task = asyncio.create_task(
                client.receive_loop()
            )

            all_tasks.extend([control_task, receive_task])

        print(f"[Main] All control modules started at the same time")
        print(f"[Main] Keyboard monitor active - Press 'q' to force stop")

        # 6) Wait for stop_event or any client to stop
        force_ended = False  # Track if ended with 'q' key
        while not stop_event.is_set():
            # Check if any client is still running
            any_running = any(client.running for client in robot_clients.values())
            if not any_running:
                print("[Main] All clients stopped (server disconnected or race ended)")
                break

            await asyncio.sleep(0.1)

        # Check if we stopped because of 'q' key
        if stop_event.is_set():
            print("[Main] Stop event detected → Sending force-end signal to Unity...")
            force_ended = True

            # Send force_end message to Unity to trigger metadata send
            for robot_id, client in robot_clients.items():
                try:
                    force_end_msg = {
                        "type": "force_end",
                        "robot_id": robot_id,
                        "message": "Python client force-ended with 'q' key"
                    }
                    await client.send_json(force_end_msg)
                except Exception as e:
                    print(f"[Main] Failed to send force_end to {robot_id}: {e}")

        # 7) Cleanup
        print("[Main] Shutting down...")

        # If force-ended with 'q', wait for Unity to send metadata before stopping
        if force_ended:
            print("[Main] Force-end detected. Waiting for Unity metadata (up to 1.5s)...")

            # Wait for metadata with timeout (keep receive_loop running)
            wait_start = asyncio.get_event_loop().time()
            metadata_timeout = 1.5  # seconds

            while asyncio.get_event_loop().time() - wait_start < metadata_timeout:
                # Check if metadata received for all robots with DATA_SAVE=1
                all_received = True
                for robot_id, client in robot_clients.items():
                    if client.data_manager is not None:
                        meta_csv = client.data_manager.current_run_dir / "metadata.csv"
                        if meta_csv.exists():
                            try:
                                # Check if CSV has data rows (more than just header)
                                with open(meta_csv, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    # Valid metadata should have header + at least one data row
                                    # Skip "Force end" fallback (which only has 1 data row with all zeros)
                                    if len(lines) > 2:  # Header + multiple data rows
                                        print(f"[Main] {robot_id} metadata received ({len(lines)-1} data rows)")
                                    else:
                                        all_received = False
                            except Exception as e:
                                all_received = False
                        else:
                            all_received = False

                if all_received:
                    print("[Main] All metadata received from Unity!")
                    break

                # Short sleep to allow receive_loop to process messages
                await asyncio.sleep(0.1)

            # Now stop event (after waiting for metadata)
            stop_event.set()

            # Check which robots didn't receive metadata and save fallback
            for robot_id, client in robot_clients.items():
                if client.data_manager is not None:
                    meta_csv = client.data_manager.current_run_dir / "metadata.csv"
                    needs_fallback = True

                    if meta_csv.exists():
                        try:
                            with open(meta_csv, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if len(lines) > 2:  # Has real data from Unity
                                    needs_fallback = False
                        except:
                            pass

                    if needs_fallback:
                        print(f"[Main] {robot_id} did not receive metadata from Unity. Saving fallback...")
                        try:
                            client.data_manager.save_force_end_metadata()
                        except Exception as e:
                            print(f"[Main] Failed to save force-end logs for {robot_id}: {e}")
        else:
            # Normal shutdown - stop immediately
            stop_event.set()

        # Cancel all tasks
        for task in all_tasks:
            task.cancel()

        try:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        except Exception:
            pass

        # Close all clients
        for robot_id, client in robot_clients.items():
            await client.close()
            print(f"[Main] {robot_id} closed")

        # 8) Post-race: build videos for each robot
        for robot_num in config_loader.ACTIVE_ROBOTS:
            rc = robot_configs[robot_num]
            try:
                await build_video_and_open_explorer(rc)
            except Exception as e:
                print(f"[Main] Robot{robot_num} video pipeline failed: {e}")

        print("[Main] System fully stopped.")

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received. Exiting...")
        stop_event.set()

    finally:
        # Terminate Unity process if we started it
        if unity_proc and unity_proc.poll() is None:
            print("[Main] Terminating Unity process...")
            unity_proc.terminate()
            try:
                unity_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                unity_proc.kill()


async def _drain_all_tasks() -> None:
    """Cancel & await remaining asyncio tasks."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(_drain_all_tasks())
        loop.run_until_complete(asyncio.sleep(0.05))
        loop.close()
        print("[Main] Program exited.")
