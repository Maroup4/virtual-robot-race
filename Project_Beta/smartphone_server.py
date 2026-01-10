"""
Smartphone Controller Server for Virtual Robot Race Beta 1.3

This server provides:
1. HTTP server for serving controller web page
2. WebSocket endpoints for each robot (R1-R5)
3. Camera frame streaming @ 5fps
4. Joystick control input from smartphone

Architecture:
- Each robot has a dedicated WebSocket endpoint: /ws/R1, /ws/R2, etc.
- Control input from smartphone is forwarded to Unity via existing RobotWebSocketClient
- Camera frames from Robot{N}/data_interactive/ are streamed to smartphone @ 5fps
"""

import asyncio
import base64
import json
import logging
import os
import socket
from pathlib import Path
from typing import Dict, Optional

from aiohttp import web, WSMsgType
import websockets

try:
    import qrcode
    import qrcode.image.svg
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("qrcode library not installed. QR code generation disabled. Install with: pip install qrcode[pil]")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RobotController:
    """
    Manages smartphone connection for a single robot.
    Handles control input forwarding and camera streaming.
    """

    def __init__(self, robot_id: str, robot_websocket_client=None):
        self.robot_id = robot_id  # e.g., "R1"
        self.robot_num = int(robot_id[1])  # Extract number from "R1" -> 1
        self.ws_client = robot_websocket_client  # Unity WebSocket client
        self.smartphone_ws = None  # Smartphone WebSocket connection
        self.camera_task = None
        self.is_streaming = True

        # Paths for camera image access
        self.data_dir = Path(f"Robot{self.robot_num}/data_interactive")
        self.buffer_pointer_file = self.data_dir / "latest_RGB_now.txt"

        logger.info(f"[{self.robot_id}] Controller initialized")

    def set_websocket_client(self, ws_client):
        """Set the Unity WebSocket client (called after client is created)"""
        self.ws_client = ws_client
        logger.info(f"[{self.robot_id}] WebSocket client connected")

    async def handle_smartphone_connection(self, websocket):
        """Handle WebSocket connection from smartphone"""
        self.smartphone_ws = websocket
        logger.info(f"[{self.robot_id}] Smartphone connected")

        # Start camera streaming task
        self.is_streaming = True
        self.camera_task = asyncio.create_task(self._camera_stream_loop())

        try:
            async for msg in websocket:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"[{self.robot_id}] WebSocket error: {websocket.exception()}")

        except Exception as e:
            logger.error(f"[{self.robot_id}] Connection error: {e}")

        finally:
            await self._cleanup()

    async def _handle_message(self, message_text: str):
        """Handle incoming message from smartphone"""
        try:
            data = json.loads(message_text)
            msg_type = data.get('type')

            if msg_type == 'control':
                # Forward control input to Unity
                await self._forward_control(data)

            elif msg_type == 'ping':
                # Respond to ping for connection check
                await self._send_to_smartphone({'type': 'pong'})

            elif msg_type == 'camera_control':
                # Handle camera on/off
                self.is_streaming = data.get('enabled', True)
                logger.info(f"[{self.robot_id}] Camera streaming: {self.is_streaming}")

        except json.JSONDecodeError as e:
            logger.error(f"[{self.robot_id}] Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"[{self.robot_id}] Message handling error: {e}")

    async def _forward_control(self, control_data: dict):
        """Forward control input to Unity via WebSocket client"""
        if not self.ws_client:
            logger.warning(f"[{self.robot_id}] No Unity WebSocket client available")
            return

        steer_angle = control_data.get('steerAngle', 0.0)
        drive_torque = control_data.get('driveTorque', 0.0)

        # Create control message for Unity (matching existing format)
        control_msg = {
            "type": "control",
            "robot_id": self.robot_id,
            "steerAngle": steer_angle,
            "driveTorque": drive_torque
        }

        try:
            # Send to Unity via existing WebSocket client
            await self.ws_client.send(json.dumps(control_msg))
            logger.debug(f"[{self.robot_id}] Control: steer={steer_angle:.2f}, torque={drive_torque:.2f}")

        except Exception as e:
            logger.error(f"[{self.robot_id}] Failed to forward control: {e}")

    async def _camera_stream_loop(self):
        """Stream camera frames @ 5fps to smartphone"""
        frame_interval = 0.2  # 5 fps = 200ms

        while self.is_streaming and self.smartphone_ws:
            try:
                if not self.smartphone_ws.closed:
                    jpeg_data = self._read_latest_camera_frame()

                    if jpeg_data:
                        # Encode to Base64
                        b64_image = base64.b64encode(jpeg_data).decode('utf-8')

                        # Send to smartphone
                        message = {
                            'type': 'camera',
                            'image': b64_image,
                            'robot_id': self.robot_id
                        }

                        await self._send_to_smartphone(message)

                await asyncio.sleep(frame_interval)

            except asyncio.CancelledError:
                logger.info(f"[{self.robot_id}] Camera stream cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.robot_id}] Camera stream error: {e}")
                await asyncio.sleep(frame_interval)

    def _read_latest_camera_frame(self) -> Optional[bytes]:
        """
        Read latest camera frame using double-buffer system.
        Returns JPEG bytes if successful, None otherwise.
        """
        try:
            # Check if data directory exists
            if not self.data_dir.exists():
                return None

            # Read buffer pointer
            if not self.buffer_pointer_file.exists():
                return None

            with open(self.buffer_pointer_file, 'r') as f:
                buffer_name = f.read().strip()  # "a" or "b"

            # Read corresponding JPEG file
            jpeg_path = self.data_dir / f"latest_RGB_{buffer_name}.jpg"

            if not jpeg_path.exists():
                return None

            with open(jpeg_path, 'rb') as f:
                jpeg_data = f.read()

            # Verify JPEG integrity (FFD9 = JPEG end marker)
            if len(jpeg_data) > 2 and jpeg_data[-2:] == b'\xff\xd9':
                return jpeg_data
            else:
                logger.warning(f"[{self.robot_id}] Incomplete JPEG (missing FFD9)")
                return None

        except Exception as e:
            logger.debug(f"[{self.robot_id}] Camera read error: {e}")
            return None

    async def _send_to_smartphone(self, message: dict):
        """Send JSON message to smartphone"""
        if self.smartphone_ws and not self.smartphone_ws.closed:
            try:
                await self.smartphone_ws.send_str(json.dumps(message))
            except Exception as e:
                logger.error(f"[{self.robot_id}] Failed to send to smartphone: {e}")

    async def _cleanup(self):
        """Cleanup resources on disconnect"""
        logger.info(f"[{self.robot_id}] Smartphone disconnected")

        # Cancel camera streaming
        if self.camera_task and not self.camera_task.done():
            self.camera_task.cancel()
            try:
                await self.camera_task
            except asyncio.CancelledError:
                pass

        self.smartphone_ws = None
        self.is_streaming = False


class SmartphoneServer:
    """
    HTTP + WebSocket server for smartphone controllers.
    Supports up to 5 robots (R1-R5).
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.controllers: Dict[str, RobotController] = {}

        # Setup routes
        self._setup_routes()

        logger.info(f"Smartphone server initialized on port {self.port}")

    def _setup_routes(self):
        """Setup HTTP and WebSocket routes"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/controller', self.handle_controller_page)
        self.app.router.add_get('/qr/{robot_id}', self.handle_qr_code)

        # WebSocket endpoints for each robot
        for i in range(1, 6):  # R1 to R5
            robot_id = f"R{i}"
            self.app.router.add_get(f'/ws/{robot_id}', self.handle_websocket)

        # Static files (if needed)
        # self.app.router.add_static('/static', 'static')

    def register_robot(self, robot_id: str, websocket_client=None):
        """Register a robot controller"""
        if robot_id not in self.controllers:
            self.controllers[robot_id] = RobotController(robot_id, websocket_client)
            logger.info(f"Registered robot: {robot_id}")
        else:
            self.controllers[robot_id].set_websocket_client(websocket_client)

    async def handle_index(self, request):
        """Handle root path"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Virtual Robot Race - Beta 1.3</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                h1 { text-align: center; }
                .robot-link {
                    display: block;
                    padding: 15px;
                    margin: 10px 0;
                    background: rgba(255,255,255,0.2);
                    border-radius: 10px;
                    text-decoration: none;
                    color: white;
                    font-size: 18px;
                    text-align: center;
                    transition: all 0.3s;
                }
                .robot-link:hover {
                    background: rgba(255,255,255,0.3);
                    transform: scale(1.05);
                }
            </style>
        </head>
        <body>
            <h1>🏎️ Virtual Robot Race</h1>
            <h2 style="text-align:center;">Beta 1.3 - Smartphone Controller</h2>
            <p style="text-align:center;">Select your robot:</p>
            <a href="/controller?robot=R1" class="robot-link">Robot 1</a>
            <a href="/controller?robot=R2" class="robot-link">Robot 2</a>
            <a href="/controller?robot=R3" class="robot-link">Robot 3</a>
            <a href="/controller?robot=R4" class="robot-link">Robot 4</a>
            <a href="/controller?robot=R5" class="robot-link">Robot 5</a>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_controller_page(self, request):
        """Serve smartphone controller HTML page"""
        robot_id = request.query.get('robot', 'R1')

        # Load HTML template
        html_path = Path(__file__).parent / 'smartphone_controller.html'

        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            # Replace robot_id placeholder
            html = html.replace('{{ROBOT_ID}}', robot_id)

            return web.Response(text=html, content_type='text/html')
        else:
            # Fallback minimal HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Robot {robot_id} Controller</title></head>
            <body>
                <h1>Robot {robot_id} Controller</h1>
                <p>Controller page not found. Create smartphone_controller.html</p>
            </body>
            </html>
            """
            return web.Response(text=html, content_type='text/html')

    async def handle_qr_code(self, request):
        """Generate QR code for robot controller URL"""
        robot_id = request.match_info.get('robot_id', 'R1')

        if not QRCODE_AVAILABLE:
            return web.Response(text='QR code generation not available. Install qrcode library.', status=503)

        # Get local IP
        local_ip = self._get_local_ip()
        controller_url = f"http://{local_ip}:{self.port}/controller?robot={robot_id}"

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(controller_url)
        qr.make(fit=True)

        # Create PNG image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save to bytes
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        return web.Response(body=buffer.read(), content_type='image/png')

    async def handle_websocket(self, request):
        """Handle WebSocket connection from smartphone"""
        # Extract robot_id from path: /ws/R1 -> R1
        path = request.path  # e.g., '/ws/R1'
        robot_id = path.split('/')[-1]  # 'R1'

        if robot_id not in self.controllers:
            logger.warning(f"Unknown robot: {robot_id}")
            return web.Response(text='Robot not found', status=404)

        # Create WebSocket
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Delegate to robot controller
        controller = self.controllers[robot_id]
        await controller.handle_smartphone_connection(ws)

        return ws

    async def start(self):
        """Start the HTTP server"""
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        # Get local IP for QR code
        local_ip = self._get_local_ip()
        logger.info(f"=" * 60)
        logger.info(f"📱 Smartphone Server Running")
        logger.info(f"=" * 60)
        logger.info(f"Local URL:  http://localhost:{self.port}")
        logger.info(f"Network URL: http://{local_ip}:{self.port}")
        logger.info(f"=" * 60)
        logger.info(f"QR Codes available at:")

        for robot_id in self.controllers.keys():
            logger.info(f"  {robot_id}:")
            logger.info(f"    Controller: http://{local_ip}:{self.port}/controller?robot={robot_id}")
            logger.info(f"    QR Code:    http://{local_ip}:{self.port}/qr/{robot_id}")

        logger.info(f"=" * 60)

        # Generate QR code files if qrcode is available
        if QRCODE_AVAILABLE:
            self._generate_qr_code_files(local_ip)
        else:
            logger.warning("QR code generation skipped (qrcode library not installed)")
            logger.warning("Install with: pip install qrcode[pil]")

    def _generate_qr_code_files(self, local_ip: str):
        """Generate QR code PNG files for each robot"""
        qr_dir = Path("qr_codes")
        qr_dir.mkdir(exist_ok=True)

        for robot_id in self.controllers.keys():
            controller_url = f"http://{local_ip}:{self.port}/controller?robot={robot_id}"

            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(controller_url)
            qr.make(fit=True)

            # Create image
            img = qr.make_image(fill_color="black", back_color="white")

            # Save to file
            qr_file = qr_dir / f"{robot_id}_controller.png"
            img.save(qr_file)

            logger.info(f"  QR code saved: {qr_file}")

        logger.info(f"All QR codes saved to {qr_dir}/")

    @staticmethod
    def _get_local_ip() -> str:
        """Get local IP address for QR code generation"""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


async def main():
    """Test server standalone"""
    server = SmartphoneServer(port=8080)

    # Register test robots
    for i in range(1, 6):
        server.register_robot(f"R{i}")

    await server.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == '__main__':
    asyncio.run(main())
