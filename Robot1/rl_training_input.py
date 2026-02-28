# rl_training_input.py (Robot1 version)
# ==============================================================================
# Reinforcement Learning Training Engine
# ==============================================================================
#
# This module handles reinforcement learning training mode (MODE_NUM=6).
# Unlike inference mode, this mode:
#   - Runs the policy network to select actions
#   - Calculates rewards based on environment feedback
#   - Updates the policy network in real-time
#
# Key differences from inference_input.py:
#   - Model is in training mode (not eval mode)
#   - Actions include exploration noise
#   - Rewards are calculated and logged
#   - Model weights are updated during runtime
#
# ==============================================================================

import os
import sys
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# Module identification
MODULE_SOURCE = "Robot1"
print(f"[rl_training_input] Loaded from {MODULE_SOURCE}/")

# Import data_manager from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_manager

# Get the directory of this file (Robot1/)
_this_dir = Path(__file__).parent

# Import model from same directory using importlib
import importlib.util
_model_spec = importlib.util.spec_from_file_location(
    f"{MODULE_SOURCE}.model",
    _this_dir / "model.py"
)
_model_module = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_module)
DrivingNetwork = _model_module.DrivingNetwork

# Import reward calculator
try:
    _reward_spec = importlib.util.spec_from_file_location(
        f"{MODULE_SOURCE}.rl_reward",
        _this_dir / "rl_reward.py"
    )
    _reward_module = importlib.util.module_from_spec(_reward_spec)
    _reward_spec.loader.exec_module(_reward_module)
    calculate_reward = _reward_module.calculate_reward
    get_terminal_reward = _reward_module.get_terminal_reward
    reset_reward_state = _reward_module.reset_state
    print(f"[rl_training_input] Reward module loaded")
except Exception as e:
    print(f"[rl_training_input] WARNING: rl_reward.py not found, using default reward: {e}")
    def calculate_reward(prev_state, curr_state, action):
        return 0.0
    def get_terminal_reward(final_status, final_soc=None, race_time_ms=None):
        return 0.0
    def reset_reward_state():
        pass

# Import start signal detector for hybrid mode
try:
    _startsignal_path = _this_dir / "rule_based_algorithms" / "perception_Startsignal.py"
    _startsignal_spec = importlib.util.spec_from_file_location(
        f"perception_Startsignal_{_this_dir.name}",
        _startsignal_path
    )
    _startsignal_module = importlib.util.module_from_spec(_startsignal_spec)
    _startsignal_spec.loader.exec_module(_startsignal_module)
    detect_start_signal = _startsignal_module.detect_start_signal
except Exception as e:
    print(f"[rl_training_input] WARNING: perception_Startsignal.py not found: {e}")
    def detect_start_signal(pil_img):
        return True  # Always allow start

# Global control values to be accessed externally
robot_id = "R1"
driveTorque = 0.0
steerAngle = 0.0

# Model and training state
_model = None
_optimizer = None
_transform = None
_model_loaded = False
_device = None

# Race state
_race_started = False
_start_detected = False
_wait_frames = 0
START_DETECTION_TIMEOUT_FRAMES = 60  # ~3 seconds

# Experience buffer for training
_experience_buffer = []
_prev_state = None

# Training hyperparameters
LEARNING_RATE = 1e-5  # Small LR for fine-tuning
EXPLORATION_NOISE = 0.1  # Standard deviation of exploration noise
BUFFER_SIZE = 32  # Update after this many experiences
GAMMA = 0.99  # Discount factor

# Episode statistics
_episode_reward = 0.0
_episode_steps = 0


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
    For RL training, we load the pretrained model as initialization.
    """
    print(f"[{robot_id} RL] Preloading model for reinforcement learning...")
    _load_model()
    if _model is not None:
        print(f"[{robot_id} RL] Model preloaded successfully!")
    else:
        print(f"[{robot_id} RL] WARNING: Model preload failed")


def warmup_cuda():
    """Warm up CUDA context with dummy inference."""
    global _model, _transform, _device

    if _model is None:
        print(f"[{robot_id} RL] CUDA warmup skipped: Model not loaded")
        return

    if _device is None or _device.type != 'cuda':
        print(f"[{robot_id} RL] CUDA warmup skipped: Using CPU")
        return

    print(f"[{robot_id} RL] Warming up CUDA context...")

    try:
        dummy_img = Image.new('RGB', (640, 480), color='black')
        dummy_tensor = _transform(dummy_img).unsqueeze(0).to(_device)
        dummy_soc = torch.tensor([[1.0]], dtype=torch.float32).to(_device)

        with torch.no_grad():
            _ = _model(dummy_tensor, dummy_soc)

        print(f"[{robot_id} RL] CUDA warmup complete!")

    except Exception as e:
        print(f"[{robot_id} RL] CUDA warmup failed: {e}")


def saturate(value, min_val=-1.0, max_val=1.0):
    """Clamp the input value within the specified range."""
    return max(min_val, min(max_val, value))


def _load_model():
    """Load the pretrained model for fine-tuning."""
    global _model, _optimizer, _transform, _model_loaded, _device

    if _model_loaded:
        return

    _model_loaded = True

    # Setup device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{robot_id} RL] Using device: {_device}")

    # Load pretrained model
    model_path = Path(__file__).parent / "models" / "model.pth"

    model = DrivingNetwork()

    try:
        model.load_state_dict(torch.load(str(model_path), map_location=_device))
        model.to(_device)
        model.train()  # Training mode (not eval!)
        print(f"[{robot_id} RL] Pretrained model loaded from {model_path}")
        _model = model

        # Setup optimizer for fine-tuning
        _optimizer = optim.Adam(_model.parameters(), lr=LEARNING_RATE)
        print(f"[{robot_id} RL] Optimizer initialized (LR={LEARNING_RATE})")

    except FileNotFoundError:
        print(f"[{robot_id} RL] ERROR: Model not found at {model_path}")
        print(f"[{robot_id} RL] RL training requires a pretrained model!")
        _model = None
    except Exception as e:
        print(f"[{robot_id} RL] ERROR: Failed to load model: {e}")
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


def _add_exploration_noise(action_drive, action_steer):
    """Add Gaussian noise for exploration."""
    noise_drive = torch.randn(1).item() * EXPLORATION_NOISE
    noise_steer = torch.randn(1).item() * EXPLORATION_NOISE * 0.5  # Less noise for steering

    noisy_drive = saturate(action_drive + noise_drive, -1.0, 1.0)
    noisy_steer = saturate(action_steer + noise_steer, -0.524, 0.524)

    return noisy_drive, noisy_steer


def _get_current_state():
    """
    Get current state from data_manager.

    Note: Currently only SOC is available in real-time via data_interactive.
    pos_x, pos_z, status etc. would require Unity to send additional
    real-time telemetry data (future enhancement for Beta 1.5+).

    For now, we use a simplified state that tracks SOC and step count.
    The reward function uses these available values.
    """
    state = {
        'soc': data_manager.get_latest_soc(robot_id) or 1.0,
        'step': _episode_steps,
    }

    return state


def _update_policy(experiences):
    """
    Update policy using collected experiences.
    Simple policy gradient update.
    """
    global _model, _optimizer

    if _model is None or _optimizer is None:
        return

    if len(experiences) == 0:
        return

    # Calculate returns (discounted rewards)
    returns = []
    G = 0
    for exp in reversed(experiences):
        G = exp['reward'] + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(_device)

    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Update policy
    _optimizer.zero_grad()

    total_loss = 0
    for i, exp in enumerate(experiences):
        # Reconstruct the forward pass
        image_tensor = exp['image_tensor']
        soc_tensor = exp['soc_tensor']

        output = _model(image_tensor, soc_tensor)
        predicted_drive = output[0, 0]
        predicted_steer = output[0, 1]

        # Actual actions taken
        action_drive = exp['action_drive']
        action_steer = exp['action_steer']

        # Simple policy gradient loss
        # Encourage actions that led to positive returns
        drive_loss = -returns[i] * (1 - (predicted_drive - action_drive).pow(2))
        steer_loss = -returns[i] * (1 - (predicted_steer - action_steer).pow(2))

        total_loss += drive_loss + steer_loss

    total_loss = total_loss / len(experiences)
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)

    _optimizer.step()

    avg_return = returns.mean().item()
    print(f"[{robot_id} RL] Policy updated: {len(experiences)} experiences, avg_return={avg_return:.3f}")


def update():
    """
    Single update cycle for RL training.
    Called repeatedly by main.py at ~20Hz.
    """
    global driveTorque, steerAngle, _race_started, _start_detected, _wait_frames
    global _prev_state, _experience_buffer, _episode_reward, _episode_steps

    try:
        # Load model on first call
        _load_model()

        if _model is None:
            driveTorque = 0.0
            steerAngle = 0.0
            return True

        # === Get inputs ===
        soc = data_manager.get_latest_soc(robot_id) or 1.0
        rgb_path = data_manager.get_latest_rgb_path(robot_id)

        if not rgb_path or not rgb_path.exists():
            return True

        try:
            pil_img = Image.open(rgb_path).convert("RGB")
        except Exception:
            return True

        # === Start signal detection (hybrid mode) ===
        if not _race_started:
            if not _start_detected:
                if detect_start_signal(pil_img):
                    _start_detected = True
                    _wait_frames = 0
                    print(f"[{robot_id} RL] Start signal detected!")
                else:
                    _wait_frames += 1
                    if _wait_frames >= START_DETECTION_TIMEOUT_FRAMES:
                        _start_detected = True
                        print(f"[{robot_id} RL] Timeout: Assuming race started")
                    else:
                        driveTorque = 0.0
                        steerAngle = 0.0
                        if _wait_frames % 40 == 0:
                            print(f"[{robot_id} RL] Waiting for start signal...")
                        return True

            if _start_detected and not _race_started:
                _race_started = True
                _episode_reward = 0.0
                _episode_steps = 0
                print(f"[{robot_id} RL] RACE STARTED! Beginning RL training episode.")

        # === Get current state ===
        curr_state = _get_current_state()

        # === Calculate reward from previous action ===
        if _prev_state is not None:
            action = {'drive': driveTorque, 'steer': steerAngle}
            reward = calculate_reward(_prev_state, curr_state, action)
            _episode_reward += reward

            # Store experience
            if len(_experience_buffer) > 0:
                _experience_buffer[-1]['reward'] = reward
                _experience_buffer[-1]['next_state'] = curr_state

        # === Run policy to get action ===
        image_tensor = _transform(pil_img).unsqueeze(0).to(_device)
        soc_tensor = torch.tensor([[soc]], dtype=torch.float32).to(_device)

        with torch.no_grad():
            output = _model(image_tensor, soc_tensor)
            raw_drive = output[0, 0].item()
            raw_steer = output[0, 1].item()

        # Apply saturation
        raw_drive = saturate(raw_drive, -1.0, 1.0)
        raw_steer = saturate(raw_steer, -0.524, 0.524)

        # Add exploration noise
        noisy_drive, noisy_steer = _add_exploration_noise(raw_drive, raw_steer)

        driveTorque = noisy_drive
        steerAngle = noisy_steer

        # === Store experience ===
        experience = {
            'image_tensor': image_tensor,
            'soc_tensor': soc_tensor,
            'action_drive': noisy_drive,
            'action_steer': noisy_steer,
            'reward': 0.0,  # Will be filled in next frame
            'state': curr_state,
            'next_state': None,
        }
        _experience_buffer.append(experience)

        # === Update policy when buffer is full ===
        if len(_experience_buffer) >= BUFFER_SIZE:
            _update_policy(_experience_buffer)
            _experience_buffer = []

        # === Update state ===
        _prev_state = curr_state
        _episode_steps += 1

        # Note: Episode end detection (Finish, Fallen) is handled by main.py
        # based on Unity messages, not detected here in real-time.

        # === Logging ===
        if _episode_steps % 20 == 0:
            steer_deg = math.degrees(steerAngle)
            print(
                f"[{robot_id} RL] Step={_episode_steps}, Drive={driveTorque:+.3f}, "
                f"Steer={steerAngle:+.3f}rad({steer_deg:+.1f}deg), "
                f"SOC={soc:.2f}, Reward={_episode_reward:.2f}"
            )

        return True

    except Exception as e:
        print(f"[{robot_id} RL] Error: {e}")
        import traceback
        traceback.print_exc()
        return True


def reset():
    """Reset RL training state."""
    global driveTorque, steerAngle, _model, _optimizer, _transform
    global _model_loaded, _device, _race_started, _start_detected, _wait_frames
    global _prev_state, _experience_buffer, _episode_reward, _episode_steps

    driveTorque = 0.0
    steerAngle = 0.0
    _model = None
    _optimizer = None
    _transform = None
    _model_loaded = False
    _device = None
    _race_started = False
    _start_detected = False
    _wait_frames = 0
    _prev_state = None
    _experience_buffer = []
    _episode_reward = 0.0
    _episode_steps = 0

    # Reset reward module state
    reset_reward_state()

    print(f"[{robot_id} RL] State reset")


def end_episode(final_status, final_soc=None):
    """
    Called by main.py when episode ends (via Unity message or q key).

    Args:
        final_status: str - 'Finish', 'Lap1', 'Fallen', 'Force end', etc.
        final_soc: float - remaining SOC at end (optional)

    Returns:
        dict with episode statistics
    """
    global _episode_reward, _experience_buffer

    # Get terminal reward
    terminal_reward = get_terminal_reward(final_status, final_soc)
    _episode_reward += terminal_reward

    # Final policy update with remaining experiences
    if len(_experience_buffer) > 0:
        # Add terminal reward to last experience
        if _experience_buffer:
            _experience_buffer[-1]['reward'] += terminal_reward
        _update_policy(_experience_buffer)
        _experience_buffer = []

    print(f"[{robot_id} RL] Episode ended: {final_status}")
    print(f"[{robot_id} RL] Total episode reward: {_episode_reward:.2f}, Steps: {_episode_steps}")

    return {
        'status': final_status,
        'total_reward': _episode_reward,
        'steps': _episode_steps,
        'final_soc': final_soc,
    }


def save_model(path=None):
    """Save the trained model."""
    global _model

    if _model is None:
        print(f"[{robot_id} RL] No model to save")
        return

    if path is None:
        path = Path(__file__).parent / "models" / "model_rl.pth"

    torch.save(_model.state_dict(), str(path))
    print(f"[{robot_id} RL] Model saved to {path}")
