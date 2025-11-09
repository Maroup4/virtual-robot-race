# websocket_client.py
# Python側をWebSocketクライアントに変更
# Unityサーバーに接続して、制御コマンドを受信、テレメトリーを送信

import asyncio
import json
import websockets
from pathlib import Path
import data_manager

class RobotWebSocketClient:
    """WebSocket client for connecting to Unity server"""

    def __init__(self, robot_id: str = "R1", server_url: str = "ws://localhost:12346/robot", robot_config: dict = None):
        self.robot_id = robot_id
        self.server_url = server_url
        self.websocket = None
        self.running = False
        self.robot_config = robot_config or {}

        # 制御コマンド（最新値）
        self.drive_torque = 0.0
        self.steer_angle = 0.0

        # DataManager for saving images (if JPEG_SAVE=1)
        self.data_manager = None
        if self.robot_config.get('JPEG_SAVE', 0) == 1:
            from pathlib import Path
            base_dir = Path(__file__).parent
            self.data_manager = data_manager.DataManager(base_dir, robot_id=robot_id)
            self.data_manager.start_new_run()
            print(f"[{self.robot_id}] DataManager initialized for image saving")

    async def connect(self):
        """Connect to Unity WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.running = True
            print(f"[{self.robot_id}] Connected to Unity server at {self.server_url}")

            # ハンドシェイク送信
            await self.send_handshake()

        except Exception as e:
            print(f"[{self.robot_id}] Connection failed: {e}")
            raise

    async def send_handshake(self):
        """Send initial handshake to server"""
        handshake = {
            "type": "connection",
            "robot_id": self.robot_id,
            "message": "Hello from Python client"
        }
        await self.send_json(handshake)
        print(f"[{self.robot_id}] Handshake sent")

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
                # 制御コマンド受信
                self.drive_torque = data.get("driveTorque", 0.0)
                self.steer_angle = data.get("steerAngle", 0.0)
                print(f"[{self.robot_id}] Control received: drive={self.drive_torque:.3f}, steer={self.steer_angle:.3f}")

            elif msg_type == "connection":
                # 接続確認
                status = data.get("status", "")
                message = data.get("message", "")
                print(f"[{self.robot_id}] Server response: {status} - {message}")

            elif msg_type == "metadata":
                # レース終了時のメタデータ受信
                print(f"[{self.robot_id}] Metadata received from Unity")
                await self.save_metadata(data)

            else:
                print(f"[{self.robot_id}] Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            print(f"[{self.robot_id}] JSON decode error: {e}")

    async def handle_binary_message(self, message: bytes):
        """Handle binary message from server (image data)"""
        try:
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

            # Save to training_data if JPEG_SAVE=1
            if self.data_manager is not None:
                image_path = self.data_manager.images_dir / frame_name
                self.data_manager.save_image_bytes(image_path, message)

            # Optional: Periodic logging (every 100 images to reduce spam)
            if self._image_count % 100 == 1:
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


# ===== テスト用メイン関数 =====
async def main():
    """Test client connection"""
    client = RobotWebSocketClient(robot_id="R1", server_url="ws://localhost:12346/robot")

    try:
        await client.connect()

        # 受信ループを起動
        receive_task = asyncio.create_task(client.receive_loop())

        # テスト: 5秒間待機してテレメトリーを送信
        for i in range(5):
            await asyncio.sleep(1)
            await client.send_telemetry(
                tick=i,
                utc_ms=asyncio.get_event_loop().time() * 1000,
                soc=100.0 - i * 2,
                status="testing"
            )
            print(f"[R1] Telemetry sent: tick={i}")

        # 受信ループを終了
        await client.close()
        await receive_task

    except Exception as e:
        print(f"[Main] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
