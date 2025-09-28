# inference_input.py
# Inference loop using a trained PyTorch model to predict wheel torques from RGB image + SOC

import time
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import data_manager  # For reading the latest SOC value

# Global torque values to be accessed externally
leftTorque = 0.0
rightTorque = 0.0

def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp the input value within the specified range."""
    return max(min_val, min(max_val, value))

def get_latest_rgb_path():
    """Determine the most recent RGB image path using the A/B flag."""
    flag_path = os.path.join("data_interactive", "latest_RGB_now.txt")
    try:
        with open(flag_path, "r") as f:
            mark = f.read().strip()
            if mark in ("a", "b"):
                return os.path.join("data_interactive", f"latest_RGB_{mark}.jpg")
    except Exception:
        pass
    return os.path.join("data_interactive", "latest_RGB_a.jpg")  # Default fallback

class TorqueNet(nn.Module):
    """Simple MLP for torque prediction based on image + SOC."""
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)

def run_ai_loop(stop_event):
    """Main loop: loads latest image and SOC, runs inference, outputs torques."""
    global leftTorque, rightTorque

    print("[Inference] AI loop started.")

    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
    input_size = 224 * 224 * 3 + 1  # Image (flattened) + 1 SOC
    model = TorqueNet(input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    while not stop_event.is_set():
        try:
            image_path = get_latest_rgb_path()
            if not os.path.exists(image_path):
                time.sleep(0.05)
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                time.sleep(0.05)
                continue

            image_tensor = transform(image).view(-1)
            soc = data_manager.get_latest_soc()
            soc_tensor = torch.tensor([soc], dtype=torch.float32)

            input_tensor = torch.cat([image_tensor, soc_tensor]).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                raw_left = output[0][0].item()
                raw_right = output[0][1].item()

                leftTorque = saturate(raw_left)
                rightTorque = saturate(raw_right)

            print(f"[Inference] Torque: L={leftTorque:.3f}, R={rightTorque:.3f}, SOC={soc:.2f}")

        except Exception as e:
            print(f"[Inference] Error: {e}")

        time.sleep(0.05)

    print("[Inference] AI loop stopped.")
