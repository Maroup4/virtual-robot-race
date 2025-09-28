# table_input.py
# Sends torque values read from a CSV file to Unity via WebSocket

import pandas as pd
import asyncio
import websocket_server
import os
from threading import Event

start_event = Event()  # Trigger event to begin sending

# CSV file path
INPUT_CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_input.csv")

csv_loaded = False
df = None  # DataFrame to hold CSV content

def start_csv_replay():
    """Called once to trigger CSV replay"""
    print("[TableInput] Start signal received.")
    start_event.set()

async def run_table_input_loop(stop_event):
    """Main loop to read and send torque values from CSV"""
    global df
    global csv_loaded

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"[TableInput] CSV file not found: {INPUT_CSV_FILE}")
        return

    if not csv_loaded:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"[TableInput] Loaded {len(df)} torque rows from CSV.")
        csv_loaded = True

    print("[TableInput] Waiting for start event...")
    await asyncio.to_thread(start_event.wait)  # Wait non-blocking

    print("[TableInput] Start event detected. Begin sending torque values.")

    for _, row in df.iterrows():
        if stop_event.is_set():
            break

        left = float(row["Left_Torque"])
        right = float(row["Right_Torque"])

        await websocket_server.send_control_command_async(left, right)
        await asyncio.sleep(0.05)  # Send at 20 FPS (50 ms interval)
