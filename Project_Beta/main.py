# main.py
# Entry point that orchestrates:
#  - WebSocket server (Unity <-> Python)
#  - Input pipeline (keyboard / table / rule_based / AI)
#  - Optional Unity process launch
#  - Post-race video build (MP4) and opening Explorer for easy drag & drop

import asyncio
import threading
import subprocess
import os
import sys  # ← needed for platform checks when opening Explorer
from typing import Optional

import config
import websocket_server

# Lazy-import only when used to avoid unnecessary hard deps at import time
#   - keyboard_input / inference_input / table_input will be imported inside main()

import make_video
from data_manager import read_last_run_dir  # returns Path to last run directory (created by DataManager)

# Shared stop signal for threads / tasks
stop_event = threading.Event()


def launch_unity_exe() -> None:
    """Launch the built Unity executable if it exists (used when DEBUG_MODE=0)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base_dir, "Windows", "VirtualRobotRace_Beta.exe")

    if os.path.exists(exe_path):
        print(f"[Main] Launching Unity app: {exe_path}")
        # Use shell=False to avoid problems with spaces in the path.
        subprocess.Popen([exe_path], shell=False)
    else:
        print(f"[Main] Unity .exe not found at: {exe_path}")


async def _drain_all_tasks() -> None:
    """Cancel & await remaining asyncio tasks to avoid 'destroyed but pending' warnings (Windows)."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def build_video_and_open_explorer() -> None:
    """
    Post-race pipeline:
      - Build MP4 from the latest run's images via make_video.py
      - Open Explorer selecting the MP4 so the user can drag & drop to X easily
    """
    if not getattr(config, "AUTO_MAKE_VIDEO", True):
        print("[Main] AUTO_MAKE_VIDEO=0 → Skip video pipeline.")
        return

    # Ask data_manager where the latest run directory is
    run_dir = read_last_run_dir()
    if not run_dir:
        print("[Main] Post-race video pipeline skipped: last_run_dir not found.")
        return

    images_dir = run_dir / "images"
    if not images_dir.exists():
        print(f"[Main] Post-race video pipeline skipped: images dir not found → {images_dir}")
        return

    out_path = run_dir / "output_video.mp4"
    fps = getattr(config, "VIDEO_FPS", 20)
    infer = getattr(config, "INFER_FPS", False)

    print(f"[Main] Building MP4 → {out_path} (fps={fps}, infer_fps={infer})")

    # Run the CPU-bound encoding in a thread pool to keep the event loop responsive
    loop = asyncio.get_running_loop()

    def _encode():
        make_video.images_to_video_ffmpeg(str(images_dir), str(out_path), fps=fps, infer_fps=infer)

    await loop.run_in_executor(None, _encode)

    # Open Explorer selecting the MP4 (Windows); fallback to open folder on macOS/Linux
    if getattr(config, "OPEN_EXPLORER_ON_VIDEO", True):
        try:
            if sys.platform.startswith("win") and out_path.exists():
                # '/select,' shows the file selected in its folder
                subprocess.Popen(["explorer", f"/select,{str(out_path)}"])
                print("[Main] Explorer opened with the MP4 selected.")
            else:
                # Simple folder open for non-Windows platforms
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, str(run_dir)])
                print("[Main] Opened run directory in file manager.")
        except Exception as e:
            print(f"[Main] Failed to open file manager: {e}")
    
    if getattr(config, "JPEG_SAVE", 1) == 0:
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


async def main() -> None:
    """
    Orchestrates the whole lifecycle:
      1) Start WebSocket server
      2) Optionally launch Unity
      3) Start input pipeline (keyboard / ai / rule_based / table)
      4) Wait for stop_event (set by server or user)
      5) Graceful shutdown
      6) Post-race: build MP4 and open Explorer selecting the file
    """
    print("[Main] Starting system...")
    race_end_sent = False

    # 1) Start WebSocket server (awaits stop_event internally)
    server_task = asyncio.create_task(websocket_server.start_server(stop_event))

    # 2) Launch Unity unless DEBUG
    if getattr(config, "DEBUG_MODE", 1) == 0:
        launch_unity_exe()
    else:
        print("[Main] DEBUG_MODE = 1 → Please launch Unity manually.")

    # 3) Input pipelines (one of them depending on config.MODE)
    input_thread = None        # for keyboard / ai / rule_based (thread-based)
    input_task = None          # for table mode (async task)

    try:
        mode = getattr(config, "MODE", "keyboard")

        if mode == "keyboard":
            import keyboard_input
            input_thread = threading.Thread(
                target=keyboard_input.listen_for_input, args=(stop_event,), daemon=True
            )
            input_thread.start()

        elif mode == "ai":
            import inference_input
            # Wait until first frame is saved before starting the AI loop
            await asyncio.to_thread(websocket_server.frame_received_event.wait)
            input_thread = threading.Thread(
                target=inference_input.run_ai_loop, args=(stop_event,), daemon=True
            )
            input_thread.start()

        elif mode == "rule_based":
            import rule_based_input
            await asyncio.to_thread(websocket_server.frame_received_event.wait)
            input_thread = threading.Thread(
                target=rule_based_input.run_rule_based_loop, args=(stop_event,), daemon=True
            )
            input_thread.start()

        elif mode == "table":
            import table_input
            await asyncio.to_thread(websocket_server.frame_received_event.wait)
            table_input.start_csv_replay()
            input_task = asyncio.create_task(table_input.run_table_input_loop(stop_event))

        else:
            print(f"[Main] Unknown MODE: {mode}")

        # 4) Main loop: wait for stop_event, allow optional hotkey 'q' to force race end
        try:
            import keyboard  # may fail in some environments; handle gracefully
        except Exception:
            keyboard = None

        while not stop_event.is_set():
            await asyncio.sleep(0.1)

            # Optional hotkey: press 'q' to force RaceEnd to Unity
            if keyboard is not None:
                try:
                    if keyboard.is_pressed("q") and not race_end_sent:
                        print("[Main] 'q' pressed → Forcing race end.")
                        await websocket_server.send_race_end_signal()
                        race_end_sent = True
                except Exception:
                    # Ignore keyboard module limitations on some systems
                    pass

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received. Exiting...")
        stop_event.set()

    # 5) Cleanup sequence
    stop_event.set()  # 5-1) signal all producers to stop

    # 5-2) Stop table async task (if any)
    if input_task:
        input_task.cancel()
        try:
            await input_task
        except asyncio.CancelledError:
            pass

    # 5-3) Stop keyboard/ai/rule_based thread (if any)
    if input_thread:
        input_thread.join(timeout=2.0)

    # 5-4) Shutdown WebSocket server gracefully
    await websocket_server.shutdown_server()

    # 5-5) Wait for start_server() task to finish
    if not server_task.done():
        try:
            await server_task
        except asyncio.CancelledError:
            print("[Main] Server task cancelled.")

    # 6) Post-race: build video and open Explorer (centralized here)
    try:
        await build_video_and_open_explorer()
    except Exception as e:
        print(f"[Main] Post-race video pipeline failed: {e}")

    print("[Main] System fully stopped.")


if __name__ == "__main__":
    # Dedicated event loop to avoid interference with other tools
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        # Ensure no pending tasks remain (prevents warnings on Windows)
        loop.run_until_complete(_drain_all_tasks())
        loop.run_until_complete(asyncio.sleep(0.05))
        loop.close()
        print("[Main] Program exited.")
