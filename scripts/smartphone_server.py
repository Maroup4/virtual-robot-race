"""
Smartphone Controller Server for Virtual Robot Race Beta 1.3

This server provides:
1. HTTP server for serving controller web page
2. WebSocket endpoints for each robot (R1-R5)
3. Camera frame streaming (configurable FPS)
4. Joystick control input from smartphone

Architecture:
- Each robot has a dedicated WebSocket endpoint: /ws/R1, /ws/R2, etc.
- Control input from smartphone is forwarded to Unity via existing RobotWebSocketClient
- Camera frames from Robot{N}/data_interactive/ are streamed to smartphone
"""

# Streaming settings
STREAM_MODE = "soc"  # "camera" for video, "soc" for SOC data only
SOC_UPDATE_INTERVAL = 0.5  # SOC update interval in seconds (2 updates per second)

# Camera streaming settings (only used if STREAM_MODE = "camera")
CAMERA_FPS = 15  # FPS for streaming (5-20 recommended)
CAMERA_OPTIMIZE = False  # Set True to re-compress images (adds CPU overhead)
CAMERA_QUALITY = 50  # JPEG quality if CAMERA_OPTIMIZE=True (1-100)
CAMERA_MAX_WIDTH = None  # Max width if CAMERA_OPTIMIZE=True (None = no resize)

# QR code colors per robot (for easy identification)
QR_COLORS = {
    'R1': '#CC0000',  # Red for Robot1
    'R2': '#00AA00',  # Green for Robot2
    'R3': '#0000CC',  # Blue for Robot3
    'R4': '#CC6600',  # Orange for Robot4
    'R5': '#6600CC',  # Purple for Robot5
}

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

    def __init__(self, robot_id: str, robot_websocket_client=None, server=None):
        self.robot_id = robot_id  # e.g., "R1"
        self.robot_num = int(robot_id[1])  # Extract number from "R1" -> 1
        self.ws_client = robot_websocket_client  # Unity WebSocket client
        self.smartphone_ws = None  # Smartphone WebSocket connection
        self.stream_task = None  # Camera or SOC streaming task
        self.is_streaming = True
        self.server = server  # Reference to SmartphoneServer

        # Connection readiness state
        self.is_ready = False  # True when dual-button press confirmed
        self.ready_event = asyncio.Event()  # Signals when ready

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
        # Close existing connection if any (e.g., when another phone scans same QR)
        if self.smartphone_ws and not self.smartphone_ws.closed:
            logger.info(f"[{self.robot_id}] Closing previous smartphone connection")
            try:
                await self.smartphone_ws.close()
            except Exception as e:
                logger.warning(f"[{self.robot_id}] Error closing old connection: {e}")

        self.smartphone_ws = websocket
        logger.info(f"[{self.robot_id}] Smartphone connected")

        # Reset ready state for new connection (allows reconnect)
        self.is_ready = False
        self.ready_event.clear()

        # Close this robot's QR popup when smartphone connects
        if self.server:
            self.server.close_qr_popup(self.robot_id)

        # Send reset message to smartphone to show L+R confirmation screen
        try:
            await websocket.send_str(json.dumps({
                'type': 'reset',
                'message': 'Please press L+R buttons to confirm connection.'
            }))
            logger.info(f"[{self.robot_id}] Reset signal sent to smartphone (reconnection)")
        except Exception as e:
            logger.warning(f"[{self.robot_id}] Failed to send reset on connect: {e}")

        # Start streaming task (camera or SOC based on STREAM_MODE)
        self.is_streaming = True
        if STREAM_MODE == "camera":
            self.stream_task = asyncio.create_task(self._camera_stream_loop())
        else:
            self.stream_task = asyncio.create_task(self._soc_stream_loop())

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

            elif msg_type == 'connect_confirm':
                # Dual button press confirmation from smartphone
                if not self.is_ready:
                    self.is_ready = True
                    self.ready_event.set()
                    logger.info(f"[{self.robot_id}] ✓ Connection confirmed (dual button press)")

                    # Notify smartphone
                    await self._send_to_smartphone({
                        'type': 'ready_confirmed',
                        'message': 'Connection confirmed! Race starting soon.'
                    })

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

        # DO NOT forward control to Unity until connection is confirmed via dual button press
        if not self.is_ready:
            # Silently ignore control input before confirmation
            return

        steer_angle = control_data.get('steerAngle', 0.0)
        drive_torque = control_data.get('driveTorque', 0.0)

        # Only forward control AFTER connection is confirmed
        # Create control message for Unity (matching existing format)
        control_msg = {
            "type": "control",
            "robot_id": self.robot_id,
            "steerAngle": steer_angle,
            "driveTorque": drive_torque
        }

        try:
            # Send to Unity via existing WebSocket client
            await self.ws_client.send_json(control_msg)
            logger.debug(f"[{self.robot_id}] Control: steer={steer_angle:.2f}, torque={drive_torque:.2f}")

        except Exception as e:
            logger.error(f"[{self.robot_id}] Failed to forward control: {e}")

    async def _camera_stream_loop(self):
        """Stream camera frames to smartphone"""
        frame_interval = 1.0 / CAMERA_FPS  # Convert FPS to interval

        while self.is_streaming and self.smartphone_ws:
            try:
                if not self.smartphone_ws.closed:
                    jpeg_data = self._read_latest_camera_frame()

                    if jpeg_data:
                        # Optionally compress/resize for streaming
                        if CAMERA_OPTIMIZE:
                            jpeg_data = self._optimize_image(jpeg_data)

                        # Encode to Base64 and send
                        b64_image = base64.b64encode(jpeg_data).decode('utf-8')
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

    async def _soc_stream_loop(self):
        """Stream SOC (battery) data to smartphone"""
        while self.is_streaming and self.smartphone_ws:
            try:
                if not self.smartphone_ws.closed:
                    soc = self._read_latest_soc()

                    # Send SOC data to smartphone
                    message = {
                        'type': 'soc',
                        'soc': soc,
                        'robot_id': self.robot_id
                    }
                    await self._send_to_smartphone(message)

                await asyncio.sleep(SOC_UPDATE_INTERVAL)

            except asyncio.CancelledError:
                logger.info(f"[{self.robot_id}] SOC stream cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.robot_id}] SOC stream error: {e}")
                await asyncio.sleep(SOC_UPDATE_INTERVAL)

    def _read_latest_soc(self) -> Optional[float]:
        """Read latest SOC value from file"""
        try:
            soc_file = self.data_dir / "latest_SOC.txt"
            if soc_file.exists():
                soc_str = soc_file.read_text(encoding="utf-8").strip()
                return float(soc_str)
            return None
        except Exception:
            return None

    def _optimize_image(self, jpeg_data: bytes) -> Optional[bytes]:
        """Optimize image for streaming (resize and compress)"""
        try:
            from PIL import Image
            from io import BytesIO

            # Load image from bytes
            img = Image.open(BytesIO(jpeg_data))

            # Resize if needed
            if CAMERA_MAX_WIDTH and img.width > CAMERA_MAX_WIDTH:
                ratio = CAMERA_MAX_WIDTH / img.width
                new_height = int(img.height * ratio)
                img = img.resize((CAMERA_MAX_WIDTH, new_height), Image.LANCZOS)

            # Re-encode with lower quality
            output = BytesIO()
            img.save(output, format='JPEG', quality=CAMERA_QUALITY, optimize=True)
            return output.getvalue()

        except Exception as e:
            logger.debug(f"[{self.robot_id}] Image optimization error: {e}")
            return jpeg_data  # Return original if optimization fails

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

        # Cancel streaming task (camera or SOC)
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
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

        # QR popup state
        self._qr_window = None
        self._qr_close_requested = False

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
            self.controllers[robot_id] = RobotController(robot_id, websocket_client, server=self)
            logger.info(f"Registered robot: {robot_id}")
        else:
            self.controllers[robot_id].set_websocket_client(websocket_client)

    async def wait_for_all_ready(self, timeout: float = 300.0, stop_event=None):
        """
        Wait for all registered robots to pass the readiness test.
        Returns True if all ready, False if timeout or cancelled.
        """
        # Reset all controllers and notify smartphones to reset UI
        for robot_id, ctrl in self.controllers.items():
            ctrl.is_ready = False
            ctrl.ready_event.clear()
            # Send reset message to smartphone if connected
            if ctrl.smartphone_ws and not ctrl.smartphone_ws.closed:
                try:
                    await ctrl.smartphone_ws.send_str(json.dumps({
                        'type': 'reset',
                        'message': 'New race starting. Press L+R buttons again.'
                    }))
                    logger.info(f"[{robot_id}] Reset signal sent to smartphone")
                except Exception as e:
                    logger.warning(f"[{robot_id}] Failed to send reset: {e}")

        logger.info(f"Waiting for {len(self.controllers)} robot(s) to confirm connection...")
        logger.info("Please scan QR code and press BOTH L+R buttons:")

        start_time = asyncio.get_event_loop().time()
        last_progress = -1

        while True:
            # Check if stop_event is set (user pressed 'q')
            if stop_event and stop_event.is_set():
                logger.info("Cancelled by user")
                return False

            # Check if all are ready
            all_ready = all(ctrl.is_ready for ctrl in self.controllers.values())

            if all_ready:
                logger.info("✓ All robots are ready!")
                return True

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for readiness ({timeout}s)")
                ready_count = sum(1 for ctrl in self.controllers.values() if ctrl.is_ready)
                logger.warning(f"Only {ready_count}/{len(self.controllers)} robots are ready")
                return False

            # Show progress every 5 seconds (only once per interval)
            current_5s = int(elapsed) // 5
            if current_5s > last_progress:
                last_progress = current_5s
                ready_count = sum(1 for ctrl in self.controllers.values() if ctrl.is_ready)
                logger.info(f"Progress: {ready_count}/{len(self.controllers)} ready ({int(elapsed)}s elapsed)")

            await asyncio.sleep(0.5)

    async def shutdown(self):
        """
        Gracefully shutdown the smartphone server.
        Cancels all streaming tasks and closes all connections.
        """
        logger.info("Shutting down smartphone server...")

        # Cancel all streaming tasks and close connections for each controller
        for robot_id, controller in self.controllers.items():
            # Stop streaming flag first
            controller.is_streaming = False

            # Cancel streaming task
            if controller.stream_task and not controller.stream_task.done():
                controller.stream_task.cancel()
                try:
                    await controller.stream_task
                except asyncio.CancelledError:
                    pass
                logger.info(f"[{robot_id}] Streaming task cancelled")

            # Close smartphone WebSocket connection
            if controller.smartphone_ws and not controller.smartphone_ws.closed:
                try:
                    await controller.smartphone_ws.close()
                except Exception as e:
                    logger.warning(f"[{robot_id}] Error closing smartphone connection: {e}")

        logger.info("Smartphone server shutdown complete")

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

        # Generate QR code with robot-specific color
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(controller_url)
        qr.make(fit=True)

        # Create PNG image with robot-specific color
        qr_color = QR_COLORS.get(robot_id, 'black')
        img = qr.make_image(fill_color=qr_color, back_color="white")

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

            # Generate QR code with robot-specific color
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(controller_url)
            qr.make(fit=True)

            # Create image with robot-specific color
            qr_color = QR_COLORS.get(robot_id, 'black')
            img = qr.make_image(fill_color=qr_color, back_color="white")

            # Save to file
            qr_file = qr_dir / f"{robot_id}_controller.png"
            img.save(qr_file)

            logger.info(f"  QR code saved: {qr_file}")

        logger.info(f"All QR codes saved to {qr_dir}/")

        # Display QR code popup windows (one per robot)
        self._show_qr_popups(qr_dir, local_ip)

    def _show_qr_popups(self, qr_dir: Path, local_ip: str):
        """Show separate QR code windows for each robot (closes individually on connection)"""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
            import tkinter as tk
            import threading

            # Store window references per robot
            self._qr_windows = {}  # {robot_id: {'window': Toplevel, 'close_requested': False}}
            self._tk_root = None
            self._tk_images = {}  # Keep references to prevent garbage collection

            def run_tk_loop():
                """Run tkinter main loop in a separate thread"""
                try:
                    # Create hidden root window
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window
                    self._tk_root = root

                    # Create a Toplevel window for each registered robot
                    for idx, robot_id in enumerate(self.controllers.keys()):
                        qr_file = qr_dir / f"{robot_id}_controller.png"
                        if not qr_file.exists():
                            logger.warning(f"QR code file not found for {robot_id}")
                            continue

                        # Load QR code image
                        qr_img = Image.open(qr_file)
                        qr_width, qr_height = qr_img.size

                        # Calculate layout
                        padding = 30
                        text_height = 70
                        total_width = qr_width + padding * 2
                        total_height = qr_height + text_height + padding * 2 + 30

                        # Create canvas
                        canvas = Image.new('RGB', (total_width, total_height), 'white')
                        draw = ImageDraw.Draw(canvas)

                        # Try to load font (fallback to default if not available)
                        try:
                            title_font = ImageFont.truetype("arial.ttf", 28)
                            url_font = ImageFont.truetype("arial.ttf", 12)
                        except:
                            title_font = ImageFont.load_default()
                            url_font = ImageFont.load_default()

                        # Get robot color for title
                        color_map = {'R1': 'red', 'R2': 'green', 'R3': 'blue', 'R4': 'orange', 'R5': 'purple'}
                        title_color = color_map.get(robot_id, 'black')

                        # Draw title
                        title = f"Robot {robot_id}"
                        draw.text((padding, padding), title, fill=title_color, font=title_font)

                        # Draw URL
                        url = f"http://{local_ip}:{self.port}/controller?robot={robot_id}"
                        draw.text((padding, padding + 35), url, fill='blue', font=url_font)

                        # Draw QR code
                        canvas.paste(qr_img, (padding, padding + text_height))

                        # Draw instruction
                        draw.text(
                            (padding, total_height - 25),
                            "Scan QR -> Press L+R buttons",
                            fill='gray',
                            font=url_font
                        )

                        # Create Toplevel window (not Tk)
                        window = tk.Toplevel(root)
                        window.title(f"QR Code - {robot_id}")
                        window.resizable(False, False)

                        # Position windows side by side
                        x_offset = 100 + idx * (total_width + 50)
                        y_offset = 100
                        window.geometry(f"+{x_offset}+{y_offset}")

                        # Convert PIL image to tkinter PhotoImage
                        tk_image = ImageTk.PhotoImage(canvas)
                        self._tk_images[robot_id] = tk_image  # Keep reference

                        # Create label with image
                        label = tk.Label(window, image=tk_image)
                        label.pack()

                        # Store reference to window
                        self._qr_windows[robot_id] = {
                            'window': window,
                            'close_requested': False
                        }

                        # Handle window close button
                        def make_on_close(rid):
                            def on_close():
                                if rid in self._qr_windows:
                                    self._qr_windows[rid]['window'].destroy()
                                    del self._qr_windows[rid]
                                # Quit if all windows closed
                                if not self._qr_windows:
                                    root.quit()
                            return on_close

                        window.protocol("WM_DELETE_WINDOW", make_on_close(robot_id))

                        logger.info(f"QR code popup window opened for {robot_id}")

                    # Periodic check for close requests
                    def check_close_requests():
                        for rid in list(self._qr_windows.keys()):
                            if self._qr_windows[rid]['close_requested']:
                                self._qr_windows[rid]['window'].destroy()
                                del self._qr_windows[rid]
                                logger.info(f"QR code popup closed for {rid}")
                        # Quit if all windows closed
                        if not self._qr_windows:
                            root.quit()
                        else:
                            root.after(100, check_close_requests)

                    root.after(100, check_close_requests)

                    # Start tkinter main loop
                    root.mainloop()

                except Exception as e:
                    logger.error(f"Failed to show QR popups: {e}")

            # Run tkinter in a separate thread
            popup_thread = threading.Thread(target=run_tk_loop, daemon=True)
            popup_thread.start()

        except ImportError as e:
            logger.warning(f"Required library not available for QR popup display: {e}")
        except Exception as e:
            logger.error(f"Error creating QR popups: {e}")

    def close_qr_popup(self, robot_id: str = None):
        """Close QR code popup window(s) (thread-safe)

        Args:
            robot_id: Specific robot ID to close, or None to close all
        """
        if robot_id:
            # Close specific robot's window
            if hasattr(self, '_qr_windows') and robot_id in self._qr_windows:
                try:
                    self._qr_windows[robot_id]['close_requested'] = True
                    logger.info(f"QR code popup close requested for {robot_id}")
                except Exception as e:
                    logger.warning(f"Error requesting QR popup close for {robot_id}: {e}")
        else:
            # Close all windows (legacy behavior)
            if hasattr(self, '_qr_windows'):
                for rid in list(self._qr_windows.keys()):
                    try:
                        self._qr_windows[rid]['close_requested'] = True
                    except:
                        pass
                logger.info("QR code popup close requested for all")

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
