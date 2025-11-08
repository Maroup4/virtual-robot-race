# main.py
# 新アーキテクチャ版（Unity=Server、Python=Client）
# Entry point that orchestrates:
#  - Unity process launch (WebSocket Server)
#  - RobotWebSocketClient (Python Client)
#  - Input pipeline (keyboard / table / rule_based / AI)
#  - Post-race video build (MP4)
# Robot1対応版: Robot1/robot_config.txt から設定を読み込み、Robot1/ 配下のモジュールを使用

import asyncio
import threading
import subprocess
import os
import sys
import time
from typing import Optional
from pathlib import Path

import config
from websocket_client import RobotWebSocketClient

# Lazy-import only when used
import make_video
from data_manager import read_last_run_dir

# Shared stop signal
stop_event = threading.Event()

# Global client instance
robot_client: Optional[RobotWebSocketClient] = None


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


async def build_video_and_open_explorer(robot_config: dict) -> None:
    """Post-race pipeline: Build MP4 from the latest run's images."""
    if not robot_config.get("AUTO_MAKE_VIDEO", 1):
        print("[Main] AUTO_MAKE_VIDEO=0 → Skip video pipeline.")
        return

    run_dir = read_last_run_dir()
    if not run_dir:
        print("[Main] Post-race video pipeline skipped: last_run_dir not found.")
        return

    images_dir = run_dir / "images"
    if not images_dir.exists():
        print(f"[Main] Post-race video pipeline skipped: images dir not found → {images_dir}")
        return

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

    if robot_config.get("JPEG_SAVE", 1) == 0:
        try:
            if images_dir.exists():
                count = 0
                for p in images_dir.glob("*.jpg"):
                    try:
                        p.unlink(missing_ok=True)
                        count += 1
                    except Exception as e:
                        print(f"[Main] Failed to delete {p.name}: {e}")
                print(f"[Main] JPEG_SAVE=0 → Deleted {count} JPG files after video export.")
        except Exception as e:
            print(f"[Main] Cleanup after video failed: {e}")


async def run_control_module(client: RobotWebSocketClient, mode: str, robot_num: int):
    """
    Run the control module based on mode string.
    This integrates keyboard/ai/rule_based control with the WebSocket client.
    Imports from Robot{N}/ directory.
    """
    print(f"[Main] Starting control module: {mode} (Robot{robot_num})")

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
            # Load inference_input from Robot{N}/ directory explicitly
            module_file = robot_dir / "inference_input.py"
            spec = importlib.util.spec_from_file_location(
                f"Robot{robot_num}.inference_input",
                module_file
            )
            inference_input = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_input)

            # TODO: AI制御の実装
            print("[Main] AI mode not yet implemented in new architecture")
            await asyncio.sleep(1)

        elif mode == "rule_based":
            # Load rule_based_input from Robot{N}/ directory explicitly
            module_file = robot_dir / "rule_based_input.py"
            spec = importlib.util.spec_from_file_location(
                f"Robot{robot_num}.rule_based_input",
                module_file
            )
            rule_based_input = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rule_based_input)

            # TODO: rule_based制御の実装
            print("[Main] Rule-based mode not yet implemented in new architecture")
            await asyncio.sleep(1)

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

        else:
            print(f"[Main] Unknown MODE: {mode}")

    finally:
        # Cleanup is handled automatically with importlib approach
        pass


async def main() -> None:
    """
    Main orchestration (Robot1対応版):
      1) Load Robot1 config
      2) Launch Unity (WebSocketServer)
      3) Wait for Unity to be ready
      4) Connect RobotWebSocketClient
      5) Start control module (from Robot1/)
      6) Wait for stop_event
      7) Graceful shutdown
      8) Build video
    """
    global robot_client

    print("[Main] Starting new architecture (Unity=Server, Python=Client)...")
    print("[Main] Robot1対応版 - Robot1/robot_config.txt から設定を読み込みます")

    # 1) Load Robot1 config
    robot_num = 1  # Phase 1: Robot1のみ対応
    robot_config = config.get_robot_config(robot_num)
    robot_id = robot_config.get("ROBOT_ID", "R1")
    mode_num = robot_config.get("MODE_NUM", 1)
    mode = config.get_mode_string(mode_num)

    print(f"[Main] Robot{robot_num} config loaded:")
    print(f"  - ROBOT_ID: {robot_id}")
    print(f"  - MODE: {mode} (MODE_NUM={mode_num})")
    print(f"  - NAME: {robot_config.get('NAME', 'Player0000')}")
    print(f"  - RACE_FLAG: {robot_config.get('RACE_FLAG', 1)}")

    unity_proc = None

    try:
        # 2) Launch Unity
        if config.DEBUG_MODE == 0:
            unity_proc = launch_unity_exe()
            if not unity_proc:
                print("[Main] Failed to launch Unity. Exiting.")
                return
        else:
            print("[Main] DEBUG_MODE = 1 → Please launch Unity manually.")

        # 3) Wait for Unity server to be ready
        server_url = f"ws://{config.HOST}:{config.PORT}/robot"
        if not await wait_for_unity_server(server_url, timeout=30.0):
            print("[Main] Unity server did not start. Exiting.")
            return

        # 4) Create and connect client
        robot_client = RobotWebSocketClient(
            robot_id=robot_id,
            server_url=server_url
        )

        await robot_client.connect()

        # 5) Start control module and receive loop concurrently
        control_task = asyncio.create_task(
            run_control_module(robot_client, mode, robot_num)
        )
        receive_task = asyncio.create_task(
            robot_client.receive_loop()
        )

        # 6) Wait for stop_event or tasks to complete
        while not stop_event.is_set() and robot_client.running:
            await asyncio.sleep(0.1)

            # Optional: hotkey 'q' to force stop
            try:
                import keyboard
                if keyboard.is_pressed("q"):
                    print("[Main] 'q' pressed → Stopping...")
                    stop_event.set()
                    break
            except Exception:
                pass

        # 7) Cleanup
        print("[Main] Shutting down...")
        stop_event.set()

        # Cancel tasks
        control_task.cancel()
        receive_task.cancel()

        try:
            await asyncio.gather(control_task, receive_task, return_exceptions=True)
        except Exception:
            pass

        # Close client
        if robot_client:
            await robot_client.close()

        # 8) Post-race: build video
        try:
            await build_video_and_open_explorer(robot_config)
        except Exception as e:
            print(f"[Main] Post-race video pipeline failed: {e}")

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
