# websocket_client.py
# WebSocket client implementation for Python side
# Connects to Unity server, receives control commands, and sends telemetry

import asyncio
import json
import websockets
from pathlib import Path
import data_manager

class RobotWebSocketClient:
    """WebSocket client for connecting to Unity server"""

    def __init__(self, robot_id: str = "R1", server_url: str = "ws://localhost:12346/robot", robot_config: dict = None, active_robots: list = None):
        self.robot_id = robot_id
        self.server_url = server_url
        self.websocket = None
        self.running = False
        self.robot_config = robot_config or {}
        self.active_robots = active_robots  # List of active robot numbers [1, 2, ...]

        # Control commands (latest values)
        self.drive_torque = 0.0
        self.steer_angle = 0.0

        # DataManager for saving images (if DATA_SAVE=1)
        self.data_manager = None
        if self.robot_config.get('DATA_SAVE', 0) == 1:
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            self.data_manager = data_manager.DataManager(base_dir, robot_id=robot_id)
            self.data_manager.start_new_run()
            print(f"[{self.robot_id}] DataManager initialized for image saving")

    async def connect(self):
        """Connect to Unity WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.running = True
            print(f"[{self.robot_id}] Connected to Unity server at {self.server_url}")

            # Send handshake
            await self.send_handshake()

        except Exception as e:
            print(f"[{self.robot_id}] Connection failed: {e}")
            raise

    async def send_handshake(self):
        """Send initial handshake to server"""
        handshake = {
            "type": "connection",
            "robot_id": self.robot_id,
            "message": "Hello from Python client",
            # Robot identity (for race results posting)
            "player_name": self.robot_config.get("NAME", "Player0000"),
            "mode": self._get_mode_string(),
            "race_flag": self.robot_config.get("RACE_FLAG", 0)
        }

        # Include active_robots list (only if provided)
        if self.active_robots is not None:
            handshake["active_robots"] = self.active_robots
            print(f"[{self.robot_id}] Sending active_robots: {self.active_robots}")

        await self.send_json(handshake)
        print(f"[{self.robot_id}] Handshake sent")

    async def send_ready_signal(self):
        """
        Send ready signal to Unity after AI model initialization is complete.
        Unity will wait for all robots to send ready before starting the race.
        This prevents race start during CUDA warmup (10+ second delay).
        """
        ready_msg = {
            "type": "ready",
            "robot_id": self.robot_id,
            "message": "AI model loaded and CUDA warmed up"
        }
        await self.send_json(ready_msg)
        print(f"[{self.robot_id}] Ready signal sent to Unity")

    def _get_mode_string(self) -> str:
        """Convert MODE_NUM to mode string"""
        mode_num = self.robot_config.get("MODE_NUM", 1)
        mode_map = {1: "keyboard", 2: "table", 3: "rule_based", 4: "ai"}
        return mode_map.get(mode_num, "keyboard")

    async def send_json(self, data: dict):
        """Send JSON message to server"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(data))
            except Exception as e:
                print(f"[{self.robot_id}] Send error: {e}")

    async def send_binary(self, data: bytes):
        """Send binary data (e.g., image) to server"""
        if self.websocket:
            try:
                await self.websocket.send(data)
            except Exception as e:
                print(f"[{self.robot_id}] Binary send error: {e}")

    async def send_telemetry(self, tick: int, utc_ms: float, soc: float, status: str = "active"):
        """Send telemetry data to Unity server"""
        telemetry = {
            "type": "data",
            "robot_id": self.robot_id,
            "tick": tick,
            "utc_ms": utc_ms,
            "soc": soc,
            "status": status,
            "filename": f"frame_{tick:06d}.jpg"
        }
        await self.send_json(telemetry)

    async def receive_loop(self):
        """Receive messages from Unity server"""
        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    await self.handle_json_message(message)
                elif isinstance(message, bytes):
                    await self.handle_binary_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[{self.robot_id}] Connection closed by server (code: {e.code}, reason: {e.reason})")
            print(f"[{self.robot_id}] Server shutdown detected - stopping client...")
        except Exception as e:
            print(f"[{self.robot_id}] Receive error: {e}")
        finally:
            self.running = False
            print(f"[{self.robot_id}] Client stopped (running=False)")

    async def handle_json_message(self, message: str):
        """Handle JSON message from server"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "control":
                # Receive control command
                self.drive_torque = data.get("driveTorque", 0.0)
                self.steer_angle = data.get("steerAngle", 0.0)
                print(f"[{self.robot_id}] Control received: drive={self.drive_torque:.3f}, steer={self.steer_angle:.3f}")

            elif msg_type == "connection":
                # Connection confirmation
                status = data.get("status", "")
                message = data.get("message", "")
                print(f"[{self.robot_id}] Server response: {status} - {message}")

            elif msg_type == "metadata":
                # Receive metadata at race end
                print(f"[{self.robot_id}] Metadata received from Unity")
                await self.save_metadata(data)

            elif msg_type == "soc":
                # Receive SOC (battery) data - write to file for AI/rule-based inference
                soc = data.get("soc", 1.0)
                soc_file = data_manager.get_soc_file(self.robot_id)
                soc_file.write_text(str(soc), encoding="utf-8")

            else:
                print(f"[{self.robot_id}] Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            print(f"[{self.robot_id}] JSON decode error: {e}")

    async def handle_binary_message(self, message: bytes):
        """Handle binary message from server (image data)"""
        try:
            # Debug: Log receive timestamp for timing analysis
            import datetime
            receive_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Save image to interactive directory using double-buffering
            # This allows rule-based controller to read the latest image
            rgb_now_file = data_manager.get_rgb_now_file(self.robot_id)
            rgb_file_a = data_manager.get_rgb_file_a(self.robot_id)
            rgb_file_b = data_manager.get_rgb_file_b(self.robot_id)
            frame_name_file = data_manager.get_latest_frame_name_file(self.robot_id)

            # Determine which buffer to use
            try:
                current = rgb_now_file.read_text(encoding="utf-8").strip()
                use_a = (current != "a")  # Toggle buffer
            except Exception:
                use_a = True

            # Write to selected buffer
            target = rgb_file_a if use_a else rgb_file_b
            target.write_bytes(message)

            # Update pointer
            rgb_now_file.write_text("a" if use_a else "b", encoding="utf-8")

            # Update frame name for debug overlay (format: frame_XXXXXX.jpg)
            if not hasattr(self, '_image_count'):
                self._image_count = 0
            self._image_count += 1
            frame_name = f"frame_{self._image_count:06d}.jpg"
            frame_name_file.write_text(frame_name, encoding="utf-8")

            # Save to training_data if DATA_SAVE=1
            if self.data_manager is not None:
                image_path = self.data_manager.images_dir / frame_name
                self.data_manager.save_image_bytes(image_path, message)

            # Debug logging (disabled to reduce overhead during start sequence)
            # if self._image_count <= 50:
            #     print(f"[{self.robot_id}] [{receive_time}] Image #{self._image_count} received: {len(message)} bytes")
            if self._image_count % 100 == 1:
                # Optional: Periodic logging (every 100 images to reduce spam)
                print(f"[{self.robot_id}] Image received: {len(message)} bytes (count={self._image_count})")

        except Exception as e:
            print(f"[{self.robot_id}] Error saving image: {e}")

    async def save_metadata(self, data: dict):
        """Save race metadata to CSV file and Unity log"""
        try:
            if self.data_manager is None or self.data_manager.current_run_dir is None:
                print(f"[{self.robot_id}] No DataManager - metadata not saved")
                return

            # Save metadata.csv (tick-by-tick data from Unity)
            csv_data = data.get('csv_data', '')
            if csv_data:
                # Unescape the CSV data (reverse the JSON escaping)
                csv_data = csv_data.replace('\\n', '\n').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')

                metadata_file = self.data_manager.current_run_dir / "metadata.csv"
                metadata_file.write_text(csv_data, encoding='utf-8')

                # Count lines for logging
                line_count = len(csv_data.split('\n')) - 1  # Subtract header
                print(f"[{self.robot_id}] Metadata CSV saved to {metadata_file} ({line_count} data rows)")
            else:
                print(f"[{self.robot_id}] Warning: No CSV data in metadata")

            # Save unity_log.txt
            unity_log = data.get('unity_log', '')
            if unity_log:
                # Unescape the log data
                unity_log = unity_log.replace('\\n', '\n').replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\')

                log_file = self.data_manager.current_run_dir / "unity_log.txt"
                log_file.write_text(unity_log, encoding='utf-8')
                print(f"[{self.robot_id}] Unity log saved to {log_file}")
            else:
                print(f"[{self.robot_id}] Warning: No Unity log in metadata")

            # Save terminal_log.txt (Python stdout/stderr)
            # This is handled by data_manager directly
            self.data_manager.save_terminal_log_from_main()

        except Exception as e:
            print(f"[{self.robot_id}] Error saving metadata: {e}")
            import traceback
            traceback.print_exc()

    def get_latest_control(self) -> dict:
        """Get latest control command"""
        return {
            "type": "control",
            "robot_id": self.robot_id,
            "driveTorque": self.drive_torque,
            "steerAngle": self.steer_angle
        }

    async def close(self):
        """Close connection"""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
                print(f"[{self.robot_id}] Connection closed")
            except Exception:
                pass


# ===== Test main function =====
async def main():
    """Test client connection"""
    client = RobotWebSocketClient(robot_id="R1", server_url="ws://localhost:12346/robot")

    try:
        await client.connect()

        # Start receive loop
        receive_task = asyncio.create_task(client.receive_loop())

        # Test: Wait 5 seconds and send telemetry
        for i in range(5):
            await asyncio.sleep(1)
            await client.send_telemetry(
                tick=i,
                utc_ms=asyncio.get_event_loop().time() * 1000,
                soc=100.0 - i * 2,
                status="testing"
            )
            print(f"[R1] Telemetry sent: tick={i}")

        # Stop receive loop
        await client.close()
        await receive_task

    except Exception as e:
        print(f"[Main] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
