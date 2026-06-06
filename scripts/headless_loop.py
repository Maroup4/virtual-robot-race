# headless_loop.py
# ============================================================
# Headless training loop for aira (Beta 1.7+)
#
# Launches Unity once, then repeats N episodes:
#   1. Run one race (python main.py)
#   2. Unity reloads scene automatically (headless mode)
#   3. Wait for Unity to be ready
#   4. [Optional] Run retraining script between episodes
#   5. Repeat
#
# Usage:
#   python scripts/headless_loop.py [episodes]
#   python scripts/headless_loop.py 50
#
# Requirements in config.txt:
#   HEADLESS=1
#   DEBUG_MODE=0  (or 1 if launching Unity manually)
# ============================================================

import os
import sys
import subprocess
import time
import asyncio
import websockets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UNITY_EXE = BASE_DIR / "Windows" / "aira_Beta_1.7.exe"
MAIN_PY   = BASE_DIR / "main.py"

UNITY_READY_TIMEOUT = 40.0   # seconds to wait for Unity to be ready
UNITY_RELOAD_WAIT   = 5.0    # seconds to wait after scene reload before reconnecting


def parse_args():
    episodes = 10  # default
    retrain_cmd = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg.isdigit():
            episodes = int(arg)
        elif arg == "--retrain" and i + 2 <= len(sys.argv) - 1:
            retrain_cmd = " ".join(sys.argv[i + 2:])
    return episodes, retrain_cmd


async def wait_for_unity(host="localhost", port=12346, timeout=UNITY_READY_TIMEOUT):
    """Wait until Unity WebSocket server is accepting connections."""
    url = f"ws://{host}:{port}/robot"
    deadline = asyncio.get_event_loop().time() + timeout
    delay = 0.5
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with websockets.connect(url) as ws:
                await ws.close()
            return True
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2.0)
    return False


def run_episode(episode_num, total):
    """Run one race episode via main.py."""
    print(f"\n[Loop] ===== Episode {episode_num}/{total} =====")
    env = {
        **os.environ,
        "AIRA_SKIP_UNITY_LAUNCH": "1",  # Unity is managed by headless_loop.py
    }
    result = subprocess.run(
        [sys.executable, str(MAIN_PY)],
        env=env,
        cwd=str(BASE_DIR),
    )
    return result.returncode


def run_retrain(retrain_cmd):
    """Run optional retraining script between episodes."""
    if not retrain_cmd:
        return
    print(f"[Loop] Running retrain: {retrain_cmd}")
    result = subprocess.run(retrain_cmd, shell=True, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"[Loop] WARNING: Retrain exited with code {result.returncode}")


def main():
    episodes, retrain_cmd = parse_args()
    print(f"[Loop] Starting headless loop: {episodes} episode(s)")
    if retrain_cmd:
        print(f"[Loop] Retrain command: {retrain_cmd}")

    # Launch Unity once
    unity_proc = None
    if UNITY_EXE.exists():
        print(f"[Loop] Launching Unity: {UNITY_EXE}")
        unity_proc = subprocess.Popen([str(UNITY_EXE)])
        print(f"[Loop] Unity launched (PID={unity_proc.pid}). Waiting for server...")
        ready = asyncio.run(wait_for_unity())
        if not ready:
            print("[Loop] ERROR: Unity did not start within timeout. Aborting.")
            unity_proc.terminate()
            sys.exit(1)
        print("[Loop] Unity ready.")
    else:
        print("[Loop] Unity exe not found → assuming manual launch (DEBUG_MODE=1).")
        print("[Loop] Waiting for Unity server...")
        ready = asyncio.run(wait_for_unity())
        if not ready:
            print("[Loop] ERROR: Unity not reachable. Aborting.")
            sys.exit(1)

    try:
        for episode in range(1, episodes + 1):
            # Run one race
            retcode = run_episode(episode, episodes)
            if retcode != 0:
                print(f"[Loop] Episode {episode} exited with code {retcode}. Continuing...")

            if episode < episodes:
                # Retrain between episodes (optional)
                run_retrain(retrain_cmd)

                # Wait for Unity to reload scene
                print(f"[Loop] Waiting {UNITY_RELOAD_WAIT}s for Unity scene reload...")
                time.sleep(UNITY_RELOAD_WAIT)

                print("[Loop] Waiting for Unity server to be ready...")
                ready = asyncio.run(wait_for_unity())
                if not ready:
                    print("[Loop] ERROR: Unity not ready after reload. Aborting.")
                    break
                print("[Loop] Unity ready for next episode.")

    finally:
        if unity_proc and unity_proc.poll() is None:
            print("[Loop] Terminating Unity...")
            unity_proc.terminate()
            try:
                unity_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                unity_proc.kill()

    print(f"\n[Loop] All {episodes} episode(s) complete.")


if __name__ == "__main__":
    main()
