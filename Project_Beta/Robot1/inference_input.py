# inference_input.py (Robot1 version - New Architecture)
# AI inference using a trained PyTorch model to predict drive/steer from RGB image + SOC
# Compatible with Unity Server + Python Client architecture
# Updated for steer-type control (driveTorque, steerAngle)

import time
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Module identification
MODULE_SOURCE = "Robot1"
print(f"[inference_input] Loaded from {MODULE_SOURCE}/")

# Import data_manager from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_manager

# Global control values to be accessed externally
robot_id = "R1"
driveTorque = 0.0
steerAngle = 0.0

# Model state
_model = None
_transform = None
_model_loaded = False

def get_latest_command():
    """Return the latest control command in the expected format."""
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }


def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp the input value within the specified range."""
    return max(min_val, min(max_val, value))

class SteerNet(nn.Module):
    """Simple MLP for drive/steer prediction based on image + SOC."""
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: [drive_torque, steer_angle]
        )

    def forward(self, x):
        return self.fc(x)


def _load_model():
    """Load the trained model (called once on first update)."""
    global _model, _transform, _model_loaded

    if _model_loaded:
        return

    _model_loaded = True

    # Load trained model
    model_path = Path(__file__).parent / "models" / "model.pth"
    input_size = 224 * 224 * 3 + 1  # Image (flattened) + 1 SOC
    model = SteerNet(input_size)

    try:
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()
        print(f"[{robot_id} Inference] Model loaded from {model_path}")
        _model = model
    except FileNotFoundError:
        print(f"[{robot_id} Inference] WARNING: Model not found at {model_path}")
        print(f"[{robot_id} Inference] Using dummy predictions (drive=0.0, steer=0.0)")
        _model = None

    # Image preprocessing
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def update():
    """
    Single update cycle for AI inference control.
    Called repeatedly by main.py at ~20Hz.
    Returns True if successful, False if should stop.
    """
    global driveTorque, steerAngle

    try:
        # Load model on first call
        _load_model()

        # === Inputs ===
        soc = data_manager.get_latest_soc(robot_id)  # float 0.0..1.0 (can be None)
        rgb_path = data_manager.get_latest_rgb_path(robot_id)

        if not rgb_path or not rgb_path.exists():
            # No image available yet
            return True

        try:
            pil_img = Image.open(rgb_path).convert("RGB")
        except Exception:
            return True

        if soc is None:
            soc = 1.0  # Default SOC if unavailable

        # === Inference ===
        if _model is not None:
            # Run inference
            image_tensor = _transform(pil_img).view(-1)
            soc_tensor = torch.tensor([soc], dtype=torch.float32)
            input_tensor = torch.cat([image_tensor, soc_tensor]).unsqueeze(0)

            with torch.no_grad():
                output = _model(input_tensor)
                raw_drive = output[0][0].item()
                raw_steer = output[0][1].item()

                driveTorque = saturate(raw_drive, -1.0, 1.0)
                steerAngle = saturate(raw_steer, -0.785, 0.785)  # ~±45 deg limit
        else:
            # Dummy output if no model
            driveTorque = 0.0
            steerAngle = 0.0

        # === Logging (every 20 frames = ~1 second) ===
        if not hasattr(update, '_frame_count'):
            update._frame_count = 0
        update._frame_count += 1

        if update._frame_count % 20 == 0:
            import math
            steer_deg = math.degrees(steerAngle)
            soc_str = f"{float(soc):.2f}"
            print(
                f"[{robot_id} Inference] Drive={driveTorque:+.3f}, "
                f"Steer={steerAngle:+.3f}rad({steer_deg:+.1f}°), "
                f"SOC={soc_str}"
            )

        return True

    except Exception as e:
        print(f"[{robot_id} Inference] Error: {e}")
        return True


def reset():
    """Reset AI inference state."""
    global driveTorque, steerAngle, _model, _transform, _model_loaded
    driveTorque = 0.0
    steerAngle = 0.0
    _model = None
    _transform = None
    _model_loaded = False
    if hasattr(update, '_frame_count'):
        update._frame_count = 0
    print(f"[{robot_id} Inference] State reset")
