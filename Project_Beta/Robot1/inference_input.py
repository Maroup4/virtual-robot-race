# inference_input.py (Robot1 version - Updated for CNN Model)
# ==============================================================================
# AI Inference Engine - The Driver's Brain
# ==============================================================================
#
# This is the core inference engine that runs the AI model.
# It reads camera images, runs neural network inference, and outputs controls.
#
# IMPORTANT: This file is the FRAMEWORK - don't modify it directly.
# To customize AI behavior, edit: ai_control_strategy.py
#
# ==============================================================================

import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# Module identification
MODULE_SOURCE = "Robot1"
print(f"[inference_input] Loaded from {MODULE_SOURCE}/")

# Import data_manager from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add Project_Beta to path
import data_manager

# Get the directory of this file (Robot1/)
_this_dir = Path(__file__).parent

# Import model from same directory using importlib to avoid cache issues
import importlib.util
_model_spec = importlib.util.spec_from_file_location(
    f"{MODULE_SOURCE}.model",
    _this_dir / "model.py"
)
_model_module = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_module)
DrivingNetwork = _model_module.DrivingNetwork

# Import AI control strategy (user-customizable) using importlib
try:
    _strategy_spec = importlib.util.spec_from_file_location(
        f"{MODULE_SOURCE}.ai_control_strategy",
        _this_dir / "ai_control_strategy.py"
    )
    _strategy_module = importlib.util.module_from_spec(_strategy_spec)
    _strategy_spec.loader.exec_module(_strategy_module)
    should_wait_for_start = _strategy_module.should_wait_for_start
    adjust_output = _strategy_module.adjust_output
    on_race_start = _strategy_module.on_race_start
    STRATEGY = _strategy_module.STRATEGY
    print(f"[inference_input] Strategy loaded: {STRATEGY}")
except Exception as e:
    print(f"[inference_input] WARNING: ai_control_strategy.py not found, using defaults: {e}")
    # Default fallback: hybrid with rule-based start detection
    STRATEGY = "hybrid"
    def should_wait_for_start(pil_img, race_started):
        if not race_started:
            # Import using importlib to avoid cache issues
            _percept_spec = importlib.util.spec_from_file_location(
                f"{MODULE_SOURCE}.perception_Startsignal",
                _this_dir / "rule_based_algorithms" / "perception_Startsignal.py"
            )
            _percept_module = importlib.util.module_from_spec(_percept_spec)
            _percept_spec.loader.exec_module(_percept_module)
            return not _percept_module.detect_start_signal(pil_img)
        return False
    def adjust_output(drive, steer, pil_img, soc, race_started=False):
        return drive, steer
    def on_race_start():
        pass

# Global control values to be accessed externally
robot_id = "R1"
driveTorque = 0.0
steerAngle = 0.0

# Model state
_model = None
_transform = None
_model_loaded = False
_device = None

# Race state
_race_started = False


def get_latest_command():
    """Return the latest control command in the expected format."""
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }


def preload_model():
    """
    Preload the AI model before the control loop starts.
    Call this BEFORE the race starts to avoid model loading delays
    during the start signal sequence.
    """
    print(f"[{robot_id} Inference] Preloading AI model...")
    _load_model()
    if _model is not None:
        print(f"[{robot_id} Inference] Model preloaded successfully!")
    else:
        print(f"[{robot_id} Inference] WARNING: Model preload failed (will use dummy output)")


def warmup_cuda():
    """
    Warm up CUDA context with dummy inference.
    This initializes GPU memory allocation and kernels BEFORE the race starts,
    eliminating the 10+ second delay on first inference.

    CRITICAL: Must be called AFTER preload_model() and BEFORE race start.
    """
    global _model, _transform, _device

    if _model is None:
        print(f"[{robot_id} Inference] CUDA warmup skipped: Model not loaded")
        return

    if _device is None or _device.type != 'cuda':
        print(f"[{robot_id} Inference] CUDA warmup skipped: Using CPU")
        return

    print(f"[{robot_id} Inference] Warming up CUDA context...")

    try:
        # Create dummy input tensors matching real inference shape
        dummy_img = Image.new('RGB', (640, 480), color='black')
        dummy_tensor = _transform(dummy_img).unsqueeze(0).to(_device)  # [1, 3, 224, 224]
        dummy_soc = torch.tensor([[1.0]], dtype=torch.float32).to(_device)  # [1, 1]

        # Run dummy inference to initialize CUDA kernels
        with torch.no_grad():
            _ = _model(dummy_tensor, dummy_soc)

        print(f"[{robot_id} Inference] CUDA warmup complete! Ready for race.")

    except Exception as e:
        print(f"[{robot_id} Inference] CUDA warmup failed: {e}")
        import traceback
        traceback.print_exc()


def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp the input value within the specified range."""
    return max(min_val, min(max_val, value))


def _load_model():
    """Load the trained model (called once on first update)."""
    global _model, _transform, _model_loaded, _device

    if _model_loaded:
        return

    _model_loaded = True

    # Setup device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{robot_id} Inference] Using device: {_device}")

    # Load trained model
    model_path = Path(__file__).parent / "models" / "model.pth"

    model = DrivingNetwork()

    try:
        model.load_state_dict(torch.load(str(model_path), map_location=_device))
        model.to(_device)
        model.eval()
        print(f"[{robot_id} Inference] Model loaded from {model_path}")
        _model = model
    except FileNotFoundError:
        print(f"[{robot_id} Inference] WARNING: Model not found at {model_path}")
        print(f"[{robot_id} Inference] Using dummy predictions (drive=0.0, steer=0.0)")
        _model = None
    except Exception as e:
        print(f"[{robot_id} Inference] WARNING: Failed to load model: {e}")
        _model = None

    # Image preprocessing (must match training transforms)
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def update():
    """
    Single update cycle for AI inference control.
    Called repeatedly by main.py at ~20Hz.
    Returns True if successful, False if should stop.

    Control flow is determined by ai_control_strategy.py
    """
    global driveTorque, steerAngle, _race_started

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

        # === Strategy: Check if should wait ===
        # Delegates to ai_control_strategy.py
        if should_wait_for_start(pil_img, _race_started):
            # Waiting for start signal - output zero
            driveTorque = 0.0
            steerAngle = 0.0

            # Log waiting status periodically
            if not hasattr(update, '_frame_count'):
                update._frame_count = 0
            update._frame_count += 1
            if update._frame_count % 40 == 0:  # Every 2 seconds
                print(f"[{robot_id} Inference] Waiting for start signal... (Strategy: {STRATEGY})")
            return True

        # === Race Start Detection ===
        if not _race_started:
            _race_started = True
            print(f"[{robot_id} Inference] RACE STARTED! (Strategy: {STRATEGY})")
            on_race_start()

        # === AI Inference ===
        if _model is not None:
            # Prepare input tensors
            image_tensor = _transform(pil_img).unsqueeze(0).to(_device)  # [1, 3, 224, 224]
            soc_tensor = torch.tensor([[soc]], dtype=torch.float32).to(_device)  # [1, 1]

            # Run inference
            with torch.no_grad():
                output = _model(image_tensor, soc_tensor)
                raw_drive = output[0, 0].item()
                raw_steer = output[0, 1].item()

            # Apply saturation
            raw_drive = saturate(raw_drive, -1.0, 1.0)
            raw_steer = saturate(raw_steer, -0.785, 0.785)  # ~±45 deg limit

            # === Strategy: Adjust output ===
            # Delegates to ai_control_strategy.py
            # Pass race_started=True so Start Boost knows when race actually began
            driveTorque, steerAngle = adjust_output(raw_drive, raw_steer, pil_img, soc, race_started=_race_started)
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
                f"Steer={steerAngle:+.3f}rad({steer_deg:+.1f}deg), "
                f"SOC={soc_str}"
            )

        return True

    except Exception as e:
        print(f"[{robot_id} Inference] Error: {e}")
        import traceback
        traceback.print_exc()
        return True


def reset():
    """Reset AI inference state."""
    global driveTorque, steerAngle, _model, _transform, _model_loaded, _device, _race_started
    driveTorque = 0.0
    steerAngle = 0.0
    _model = None
    _transform = None
    _model_loaded = False
    _device = None
    _race_started = False
    if hasattr(update, '_frame_count'):
        update._frame_count = 0
    # Reset start signal detector state (for hybrid mode)
    try:
        from rule_based_algorithms.perception_Startsignal import detect_start_signal
        if hasattr(detect_start_signal, 'ready_to_go'):
            detect_start_signal.ready_to_go = False
    except ImportError:
        pass
    print(f"[{robot_id} Inference] State reset")
