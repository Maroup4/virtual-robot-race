# train.py
# Enhanced Training Script for VRR AI Pipeline
# =============================================
# This is the Phase 2 enhanced version of train_model.py with:
# - Early stopping with patience
# - Training log CSV export
# - config.yaml integration
# - Manifest-based data loading
# - Iteration-aware model saving
# - Auto-create iteration from training data
#
# Usage:
#   python scripts/train.py --data training_data                           # Auto-create iteration and train
#   python scripts/train.py --iteration experiments/iteration_260110_090000  # Use existing iteration
#   python scripts/train.py --config experiments/config.yaml               # Use config file

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import DrivingNetwork, ValueNetwork

# Import run scorer for score-based filtering
try:
    from run_scorer import score_all_runs, filter_runs_by_score, get_top_runs
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False
    print("[Train] Warning: run_scorer not available, score-based filtering disabled")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_frame_reward(row: Dict, run_score: float, total_frames: int) -> float:
    """
    Calculate reward for a single frame (simple mode - Phase 1).

    Args:
        row: DataFrame row with frame data
        run_score: Total score for this run (from run_scorer)
        total_frames: Total number of frames in this run

    Returns:
        Reward value for this frame
    """
    # Base reward: distribute run score across frames
    base_reward = run_score / max(total_frames, 1)

    # Frame-level adjustments
    reward = base_reward

    # Bonus for Finish frames (terminal success)
    status = row.get("status", "")
    if status == "Finish":
        reward += 1.0

    # Small bonus for later frames (survived longer = better)
    # Normalized by total frames
    frame_progress = row.get("race_time_ms", 0) / 30000.0  # Normalize by ~30 seconds
    reward += frame_progress * 0.1

    return reward


# =============================================================================
# Phase 2: Detailed Frame-Level Reward Calculation
# =============================================================================

# Reward weights for detailed calculation
REWARD_CONFIG = {
    # Per-frame rewards
    'survival_bonus': 0.1,           # Reward for each frame survived
    'soc_efficiency': 5.0,           # Multiplier for SOC preservation
    'progress_reward': 1.0,          # Reward per unit of track progress
    'smoothness_penalty': 0.5,       # Penalty multiplier for steering changes

    # Status bonuses/penalties
    'lap1_bonus': 50.0,              # Bonus for completing lap 1
    'finish_bonus': 100.0,           # Bonus for finishing (2 laps)
    'fallen_penalty': -100.0,        # Penalty for falling

    # Cumulative reward settings
    'discount_factor': 0.99,         # Gamma for discounted future rewards
}


def calculate_track_progress(pos_x: float, pos_z: float) -> float:
    """
    Calculate approximate progress on the track.

    The VRR track is roughly oval-shaped. This function estimates
    how far along the track the robot is based on position.

    Args:
        pos_x: X position
        pos_z: Z position

    Returns:
        Progress value (higher = further along track)
    """
    # Simple heuristic: use distance from start position
    # Start position is approximately (0, 0)
    # This can be improved with actual track waypoints
    import math
    distance = math.sqrt(pos_x ** 2 + pos_z ** 2)
    return distance


def calculate_detailed_frame_rewards(
    df: pd.DataFrame,
    final_status: str,
    reward_config: Dict = None
) -> List[float]:
    """
    Calculate detailed per-frame rewards for a run (Phase 2).

    Uses position, SOC, steering smoothness, and status information
    to compute rewards that reflect the quality of each frame's action.

    Args:
        df: DataFrame with run data (must include pos_x, pos_z, soc, steer_angle, status)
        final_status: Final status of the run (Finish, Fallen, etc.)
        reward_config: Reward configuration dict (uses defaults if None)

    Returns:
        List of per-frame rewards
    """
    config = reward_config or REWARD_CONFIG
    rewards = []

    prev_soc = None
    prev_steer = None
    prev_progress = None

    for idx, row in df.iterrows():
        reward = 0.0

        # 1. Survival bonus (every frame)
        reward += config['survival_bonus']

        # 2. SOC efficiency (reward for preserving battery)
        soc = row.get('soc', 1.0)
        if prev_soc is not None:
            soc_change = soc - prev_soc
            # Penalize SOC drop, but less if moving (expected consumption)
            if soc_change < 0:
                # Small penalty for normal consumption
                reward += soc_change * config['soc_efficiency']
        prev_soc = soc

        # 3. Progress reward (moving forward on track)
        pos_x = row.get('pos_x', 0.0)
        pos_z = row.get('pos_z', 0.0)
        progress = calculate_track_progress(pos_x, pos_z)
        if prev_progress is not None:
            progress_delta = progress - prev_progress
            if progress_delta > 0:
                reward += progress_delta * config['progress_reward']
        prev_progress = progress

        # 4. Smoothness penalty (penalize jerky steering)
        steer = row.get('steer_angle', 0.0)
        if prev_steer is not None:
            steer_change = abs(steer - prev_steer)
            reward -= steer_change * config['smoothness_penalty']
        prev_steer = steer

        # 5. Status-based rewards
        status = row.get('status', '')
        if status == 'Lap1' and idx > 0:
            prev_status = df.iloc[idx - 1].get('status', '')
            if prev_status == 'Lap0':
                reward += config['lap1_bonus']
        elif status == 'Finish':
            reward += config['finish_bonus']

        rewards.append(reward)

    # 6. Terminal penalty for falling
    if final_status == 'Fallen' or final_status == 'Force end':
        if rewards:
            rewards[-1] += config['fallen_penalty']

    return rewards


def compute_cumulative_rewards(
    frame_rewards: List[float],
    discount_factor: float = 0.99
) -> List[float]:
    """
    Compute discounted cumulative rewards (returns) for each frame.

    R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    This gives each frame credit for future outcomes, not just immediate reward.

    Args:
        frame_rewards: List of per-frame rewards
        discount_factor: Gamma for discounting future rewards

    Returns:
        List of cumulative rewards for each frame
    """
    n = len(frame_rewards)
    cumulative = [0.0] * n

    # Compute backwards (more efficient)
    running_sum = 0.0
    for i in range(n - 1, -1, -1):
        running_sum = frame_rewards[i] + discount_factor * running_sum
        cumulative[i] = running_sum

    return cumulative


# =============================================================================
# Phase 3: AWR (Advantage-Weighted Regression)
# =============================================================================

def train_value_function(
    value_net: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-4
) -> float:
    """
    Train the value function to predict cumulative rewards.

    Args:
        value_net: ValueNetwork model
        dataloader: DataLoader with samples containing 'reward' (cumulative reward)
        epochs: Number of training epochs
        device: torch device
        lr: Learning rate

    Returns:
        Final validation loss
    """
    value_net.train()
    optimizer = torch.optim.AdamW(value_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            images = batch["image"].to(device)
            soc = batch["soc"].to(device)
            target_values = batch["reward"].to(device)  # Cumulative rewards as targets

            optimizer.zero_grad()
            predicted_values = value_net(images, soc)
            loss = criterion(predicted_values, target_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            count += images.size(0)

        avg_loss = total_loss / count
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Value] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    return best_loss


def compute_advantages(
    value_net: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[int, float]:
    """
    Compute advantages for all samples in the dataset.

    Advantage = Actual cumulative reward - Predicted value V(s)

    Args:
        value_net: Trained ValueNetwork
        dataloader: DataLoader with samples
        device: torch device

    Returns:
        Dict mapping sample index to advantage value
    """
    value_net.eval()
    advantages = {}
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            soc = batch["soc"].to(device)
            actual_rewards = batch["reward"].to(device)

            predicted_values = value_net(images, soc)

            # Advantage = Actual - Predicted
            batch_advantages = (actual_rewards - predicted_values).squeeze().cpu().numpy()

            # Handle single sample case
            if batch_advantages.ndim == 0:
                batch_advantages = [batch_advantages.item()]

            for adv in batch_advantages:
                advantages[sample_idx] = float(adv)
                sample_idx += 1

    return advantages


def train_policy_with_advantages(
    model: nn.Module,
    dataloader: DataLoader,
    advantages: Dict[int, float],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Train policy network using advantage-weighted regression.

    Loss = sum(weight * MSE) where weight = exp(advantage / temperature)

    Args:
        model: Policy network (DrivingNetwork)
        dataloader: DataLoader
        advantages: Dict mapping sample index to advantage
        criterion: Loss function (not used directly, weights computed manually)
        optimizer: Optimizer
        device: torch device
        temperature: Temperature for advantage weighting

    Returns:
        Training metrics dict
    """
    model.train()
    total_loss = 0.0
    total_torque_loss = 0.0
    total_steer_loss = 0.0
    count = 0
    sample_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        soc = batch["soc"].to(device)
        targets = batch["targets"].to(device)

        # Get advantages for this batch
        batch_size = images.size(0)
        batch_advantages = []
        for i in range(batch_size):
            adv = advantages.get(sample_idx + i, 0.0)
            batch_advantages.append(adv)
        sample_idx += batch_size

        # Compute weights from advantages
        adv_tensor = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
        weights = torch.exp(adv_tensor / temperature)

        # Normalize weights to prevent explosion
        weights = weights / (weights.sum() + 1e-8) * batch_size

        optimizer.zero_grad()
        outputs = model(images, soc)

        # Weighted MSE loss
        per_sample_loss = ((outputs - targets) ** 2).mean(dim=1)
        loss = (weights * per_sample_loss).mean()

        loss.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            torque_loss = ((outputs[:, 0] - targets[:, 0]) ** 2).mean().item()
            steer_loss = ((outputs[:, 1] - targets[:, 1]) ** 2).mean().item()

        total_loss += loss.item() * batch_size
        total_torque_loss += torque_loss * batch_size
        total_steer_loss += steer_loss * batch_size
        count += batch_size

    return {
        "loss": total_loss / count,
        "torque_loss": total_torque_loss / count,
        "steer_loss": total_steer_loss / count
    }


class DrivingDataset(Dataset):
    """
    Dataset for driving imitation learning.

    Loads image + SOC as input, drive_torque + steer_angle as targets.
    Supports loading from manifest or directory list.

    For Reward-Weighted BC mode, also loads per-frame rewards.
    """

    VALID_RACING_STATUS = ["Lap0", "Lap1", "Lap2", "Finish"]

    def __init__(
        self,
        data_dirs: List[Path],
        transform=None,
        exclude_start_sequence: bool = True,
        valid_status: Optional[List[str]] = None,
        run_scores: Optional[Dict[str, float]] = None,
        compute_rewards: bool = False,
        reward_type: str = "simple"
    ):
        """
        Initialize dataset.

        Args:
            data_dirs: List of run directories
            transform: Image transforms
            exclude_start_sequence: Skip StartSequence frames
            valid_status: List of valid status values (default: Lap1, Lap2, Finish)
            run_scores: Dict mapping run_name to score (for RW-BC mode)
            compute_rewards: Whether to compute per-frame rewards
            reward_type: "simple" (Phase 1) or "detailed" (Phase 2)
        """
        self.samples = []
        self.transform = transform
        self.valid_status = valid_status or self.VALID_RACING_STATUS
        self.compute_rewards = compute_rewards
        self.reward_type = reward_type

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            csv_path = data_dir / "metadata.csv"

            if not csv_path.exists():
                print(f"[Dataset] Warning: metadata.csv not found in {data_dir}")
                continue

            df = pd.read_csv(csv_path)
            images_dir = data_dir / "images"

            # Get run score if available
            run_name = data_dir.name
            run_score = run_scores.get(run_name, 1000.0) if run_scores else 1000.0
            total_frames = len(df)

            # For detailed rewards, compute all frame rewards upfront
            detailed_rewards = None
            cumulative_rewards = None
            if compute_rewards and reward_type == "detailed":
                final_status = df.iloc[-1].get("status", "") if len(df) > 0 else ""
                frame_rewards = calculate_detailed_frame_rewards(df, final_status)
                cumulative_rewards = compute_cumulative_rewards(frame_rewards)

            loaded = 0
            skipped = 0
            frame_idx = 0

            for _, row in df.iterrows():
                status = row.get("status", "")

                # Filter by status
                if exclude_start_sequence and status == "StartSequence":
                    skipped += 1
                    frame_idx += 1
                    continue

                if status not in self.valid_status:
                    skipped += 1
                    frame_idx += 1
                    continue

                img_path = images_dir / row["filename"]
                if not img_path.exists():
                    frame_idx += 1
                    continue

                sample = {
                    "image_path": str(img_path),
                    "soc": float(row["soc"]),
                    "drive_torque": float(row["drive_torque"]),
                    "steer_angle": float(row["steer_angle"]),
                    "run_name": run_name,
                }

                # Compute reward if requested
                if compute_rewards:
                    if reward_type == "detailed" and cumulative_rewards is not None:
                        # Use cumulative (discounted future) reward
                        sample["reward"] = cumulative_rewards[frame_idx]
                    else:
                        # Simple reward based on run score
                        sample["reward"] = calculate_frame_reward(
                            row.to_dict(), run_score, total_frames
                        )

                self.samples.append(sample)
                loaded += 1
                frame_idx += 1

            reward_info = f", reward_type={reward_type}" if compute_rewards else ""
            print(f"[Dataset] {run_name}: {loaded} samples, score={run_score:.0f}{reward_info}")

        print(f"[Dataset] Total: {len(self.samples)} samples")

        # Normalize rewards if computing them
        if compute_rewards and self.samples:
            rewards = [s["reward"] for s in self.samples]
            min_r, max_r = min(rewards), max(rewards)
            if max_r > min_r:
                for s in self.samples:
                    s["reward"] = (s["reward"] - min_r) / (max_r - min_r)
            print(f"[Dataset] Rewards normalized: min={min_r:.3f}, max={max_r:.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        soc = torch.tensor([sample["soc"]], dtype=torch.float32)
        targets = torch.tensor([
            sample["drive_torque"],
            sample["steer_angle"]
        ], dtype=torch.float32)

        result = {"image": image, "soc": soc, "targets": targets}

        # Include reward if available
        if self.compute_rewards and "reward" in sample:
            result["reward"] = torch.tensor([sample["reward"]], dtype=torch.float32)

        return result


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0001, verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping! Best epoch was {self.best_epoch + 1}")
                return True

        return False


class TrainingLogger:
    """
    Logs training metrics to CSV file.

    Enables post-training analysis and visualization.
    """

    def __init__(self, log_path: Path):
        """
        Initialize logger.

        Args:
            log_path: Path to CSV log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

        # Write header
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_torque_loss', 'train_steer_loss',
                'val_loss', 'val_torque_loss', 'val_steer_loss',
                'learning_rate', 'is_best', 'timestamp'
            ])

    def log(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
            learning_rate: float, is_best: bool):
        """
        Log metrics for one epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            learning_rate: Current learning rate
            is_best: Whether this epoch achieved best validation loss
        """
        row = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_torque_loss': train_metrics['torque_loss'],
            'train_steer_loss': train_metrics['steer_loss'],
            'val_loss': val_metrics['loss'],
            'val_torque_loss': val_metrics['torque_loss'],
            'val_steer_loss': val_metrics['steer_loss'],
            'learning_rate': learning_rate,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }

        self.metrics_history.append(row)

        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(list(row.values()))

    def get_summary(self) -> Dict:
        """Get summary statistics of training."""
        if not self.metrics_history:
            return {}

        best_epoch = min(self.metrics_history, key=lambda x: x['val_loss'])

        return {
            'total_epochs': len(self.metrics_history),
            'best_epoch': best_epoch['epoch'],
            'best_val_loss': best_epoch['val_loss'],
            'final_train_loss': self.metrics_history[-1]['train_loss'],
            'final_val_loss': self.metrics_history[-1]['val_loss'],
        }


def get_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transforms from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_transform, val_transform)
    """
    aug_config = config.get('augmentation', {})
    cj = aug_config.get('color_jitter', {})

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=cj.get('brightness', 0.2),
            contrast=cj.get('contrast', 0.2),
            saturation=cj.get('saturation', 0.1),
            hue=cj.get('hue', 0.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_reward_weighting: bool = False,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function (MSELoss)
        optimizer: Optimizer
        device: Compute device
        use_reward_weighting: If True, use reward-weighted loss (RW-BC mode)
        temperature: Temperature for reward weighting (higher = more uniform)

    Returns:
        Dict with loss metrics
    """
    model.train()
    total_loss = 0.0
    total_torque_loss = 0.0
    total_steer_loss = 0.0
    total_weight = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        soc = batch["soc"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(images, soc)

        if use_reward_weighting and "reward" in batch:
            # Reward-Weighted BC: weight loss by exp(reward / temperature)
            rewards = batch["reward"].to(device)  # Shape: [batch, 1]
            weights = torch.exp(rewards / temperature).squeeze()  # Shape: [batch]

            # Compute per-sample loss
            per_sample_loss = ((outputs - targets) ** 2).mean(dim=1)  # Shape: [batch]

            # Weighted loss
            loss = (weights * per_sample_loss).mean()
            total_weight += weights.sum().item()
        else:
            # Standard BC
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            torque_loss = nn.MSELoss()(outputs[:, 0], targets[:, 0])
            steer_loss = nn.MSELoss()(outputs[:, 1], targets[:, 1])
            total_torque_loss += torque_loss.item()
            total_steer_loss += steer_loss.item()

    n = len(dataloader)
    result = {
        "loss": total_loss / n,
        "torque_loss": total_torque_loss / n,
        "steer_loss": total_steer_loss / n
    }

    if use_reward_weighting:
        result["avg_weight"] = total_weight / (n * dataloader.batch_size)

    return result


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_torque_loss = 0.0
    total_steer_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            soc = batch["soc"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(images, soc)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            torque_loss = nn.MSELoss()(outputs[:, 0], targets[:, 0])
            steer_loss = nn.MSELoss()(outputs[:, 1], targets[:, 1])
            total_torque_loss += torque_loss.item()
            total_steer_loss += steer_loss.item()

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "torque_loss": total_torque_loss / n,
        "steer_loss": total_steer_loss / n
    }


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data_dirs_from_manifest(manifest_path: Path, training_data_dir: Path) -> List[Path]:
    """
    Load data directories from a manifest file.

    Args:
        manifest_path: Path to dataset_manifest.json
        training_data_dir: Base training_data directory

    Returns:
        List of run directory paths
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    data_dirs = []
    for run_info in manifest.get('runs', []):
        run_name = run_info['run_name']
        run_path = training_data_dir / run_name
        if run_path.exists():
            data_dirs.append(run_path)
        else:
            print(f"[Warning] Run directory not found: {run_path}")

    return data_dirs


def find_data_directories(
    base_path: Path,
    min_score: Optional[float] = None,
    top_percent: Optional[float] = None
) -> List[Path]:
    """
    Find all run_* directories containing training data.

    Args:
        base_path: Directory containing run_* folders
        min_score: Only include runs with score >= this value
        top_percent: Only include top N% of runs by score

    Returns:
        List of run directory paths
    """
    data_dirs = []

    # First, find all valid run directories
    all_runs = []
    for d in sorted(base_path.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            if (d / "metadata.csv").exists():
                all_runs.append(d)

    # Apply score-based filtering if requested
    if (min_score is not None or top_percent is not None) and SCORER_AVAILABLE:
        print(f"[Train] Scoring {len(all_runs)} runs for data selection...")
        results = score_all_runs(base_path)

        if min_score is not None:
            results = filter_runs_by_score(results, min_score)
            print(f"[Train] Filtered to {len(results)} runs with score >= {min_score}")

        if top_percent is not None:
            results = get_top_runs(results, top_percent)
            print(f"[Train] Filtered to top {top_percent}% ({len(results)} runs)")

        # Extract paths from filtered results
        for r in results:
            if r['valid']:
                data_dirs.append(Path(r['path']))

        # Print score summary
        if results:
            scores = [r['total_score'] for r in results if r['valid']]
            if scores:
                print(f"[Train] Score range: {min(scores):.1f} - {max(scores):.1f}")
    else:
        data_dirs = all_runs

    return data_dirs


def train(
    config: Dict,
    iteration_dir: Path,
    robot_dir: Path,
    data_source_dir: Optional[Path] = None,
    min_score: Optional[float] = None,
    top_percent: Optional[float] = None,
    mode: str = "bc",
    finetune_path: Optional[Path] = None,
    temperature: float = 1.0,
    reward_type: str = "simple"
) -> Dict:
    """
    Main training function.

    Args:
        config: Configuration dictionary
        iteration_dir: Path to iteration directory
        robot_dir: Path to Robot1 directory
        data_source_dir: Optional training data source directory
        min_score: Only use runs with score >= this value
        top_percent: Only use top N% of runs by score
        mode: Training mode - "bc", "rw" (Reward-Weighted BC), or "awr" (Advantage-Weighted Regression)
        finetune_path: Path to existing model for fine-tuning (optional)
        temperature: Temperature for reward weighting in RW/AWR mode (default: 1.0)
        reward_type: "simple" (Phase 1) or "detailed" (Phase 2) reward calculation

    Returns:
        Training results dictionary
    """
    use_reward_weighting = (mode in ["rw", "awr"])
    use_awr = (mode == "awr")

    mode_desc = {
        "bc": "Behavioral Cloning",
        "rw": "Reward-Weighted BC",
        "awr": "Advantage-Weighted Regression (AWR)"
    }
    print(f"[Train] Mode: {mode.upper()} ({mode_desc.get(mode, mode)})")

    if finetune_path:
        print(f"[Train] Fine-tuning from: {finetune_path}")

    if use_reward_weighting:
        print(f"[Train] Temperature: {temperature}")
        print(f"[Train] Reward type: {reward_type}")

    if use_awr:
        print(f"[Train] AWR: Will train value function first, then policy with advantages")

    # Setup paths
    training_data_dir = robot_dir / config['paths']['training_data']

    # Use data_sources if available, otherwise use training_data
    if (iteration_dir / "data_sources").exists():
        data_load_dir = iteration_dir / "data_sources"
        print(f"[Train] Using data from iteration data_sources: {data_load_dir}")
    elif data_source_dir:
        data_load_dir = data_source_dir
        print(f"[Train] Using data from: {data_source_dir}")
    else:
        data_load_dir = training_data_dir
        print(f"[Train] Using data from: {data_load_dir}")

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"[Train] Random seed: {seed}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Load data directories from data_load_dir (with optional score filtering)
    data_dirs = find_data_directories(
        data_load_dir,
        min_score=min_score,
        top_percent=top_percent
    )

    if not data_dirs:
        raise ValueError(f"No training data found in {data_load_dir}")

    print(f"[Train] Found {len(data_dirs)} data directories")

    # Create transforms
    train_transform, val_transform = get_transforms(config)

    # Get run scores for RW-BC mode
    run_scores = None
    if use_reward_weighting and SCORER_AVAILABLE:
        print(f"[Train] Computing run scores for reward weighting...")
        score_results = score_all_runs(data_load_dir)
        run_scores = {r['run_name']: r['total_score'] for r in score_results if r['valid']}
        print(f"[Train] Loaded scores for {len(run_scores)} runs")

    # Create dataset
    train_config = config.get('training', {})
    dataset = DrivingDataset(
        data_dirs,
        transform=train_transform,
        exclude_start_sequence=config.get('data_filtering', {}).get('exclude_start_sequence', True),
        run_scores=run_scores,
        compute_rewards=use_reward_weighting,
        reward_type=reward_type
    )

    if len(dataset) == 0:
        raise ValueError("No samples loaded from dataset")

    # Split dataset
    val_split = train_config.get('val_split', 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"[Train] Training samples: {train_size}")
    print(f"[Train] Validation samples: {val_size}")

    # Create dataloaders
    batch_size = train_config.get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Create model
    model = DrivingNetwork().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {total_params:,}")

    # Load pretrained weights for fine-tuning
    if finetune_path:
        if finetune_path.exists():
            model.load_state_dict(torch.load(str(finetune_path), map_location=device))
            print(f"[Train] Loaded pretrained weights from {finetune_path}")
        else:
            print(f"[Train] WARNING: Finetune path not found: {finetune_path}")
            print(f"[Train] Starting from scratch...")

    # Loss and optimizer
    criterion = nn.MSELoss()
    lr = train_config.get('learning_rate', 1e-4)

    # Use smaller learning rate for fine-tuning
    if finetune_path:
        lr = train_config.get('finetune_learning_rate', lr * 0.1)
        print(f"[Train] Using fine-tune learning rate: {lr}")

    weight_decay = train_config.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    sched_config = train_config.get('scheduler', {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=sched_config.get('factor', 0.5),
        patience=sched_config.get('patience', 10),
        min_lr=sched_config.get('min_lr', 1e-6)
    )

    # Early stopping
    es_config = train_config.get('early_stopping', {})
    early_stopping = None
    if es_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=es_config.get('patience', 15),
            min_delta=es_config.get('min_delta', 0.0001)
        )

    # Logger
    log_path = iteration_dir / "training_log.csv"
    logger = TrainingLogger(log_path)

    # Model save path
    model_path = iteration_dir / "model.pth"
    best_model_path = robot_dir / config['paths']['models'] / "model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    epochs = train_config.get('epochs', 100)
    best_val_loss = float('inf')

    # AWR: Train value function first, then compute advantages
    advantages = None
    if use_awr:
        print(f"\n[Train] AWR Stage 1: Training value function...")
        print("=" * 70)

        # Create value network
        value_net = ValueNetwork().to(device)
        value_params = sum(p.numel() for p in value_net.parameters())
        print(f"[Train] Value network parameters: {value_params:,}")

        # Need a dataloader without shuffle to maintain index correspondence
        full_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

        # Train value function for fewer epochs
        value_epochs = min(30, epochs // 3)
        value_loss = train_value_function(
            value_net, full_loader, value_epochs, device, lr=lr
        )
        print(f"[Train] Value function training complete. Final loss: {value_loss:.6f}")

        # Save value network
        value_path = iteration_dir / "value_net.pth"
        torch.save(value_net.state_dict(), value_path)
        print(f"[Train] Value network saved to: {value_path}")

        # Compute advantages for all samples
        print(f"\n[Train] AWR Stage 2: Computing advantages...")
        advantages = compute_advantages(value_net, full_loader, device)
        adv_values = list(advantages.values())
        print(f"[Train] Advantages computed: min={min(adv_values):.3f}, max={max(adv_values):.3f}, mean={np.mean(adv_values):.3f}")

        print(f"\n[Train] AWR Stage 3: Training policy with advantages...")
        print("=" * 70)

    print(f"\n[Train] Starting training for up to {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        # Train
        if use_awr and advantages is not None:
            # AWR: Use advantage-weighted training
            train_metrics = train_policy_with_advantages(
                model, train_loader, advantages, criterion, optimizer, device,
                temperature=temperature
            )
        else:
            # BC or RW-BC: Use standard training
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_reward_weighting=use_reward_weighting,
                temperature=temperature
            )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Check for best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), best_model_path)

        # Log metrics
        logger.log(epoch, train_metrics, val_metrics, current_lr, is_best)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Print progress
        marker = " [BEST]" if is_best else ""
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {train_metrics['loss']:.6f} (T:{train_metrics['torque_loss']:.4f} S:{train_metrics['steer_loss']:.4f}) | "
            f"Val: {val_metrics['loss']:.6f} (T:{val_metrics['torque_loss']:.4f} S:{val_metrics['steer_loss']:.4f}) | "
            f"LR: {current_lr:.2e}{marker}"
        )

        # Early stopping check
        if early_stopping and early_stopping(val_metrics['loss'], epoch):
            print(f"[Train] Early stopping triggered at epoch {epoch + 1}")
            break

    print("=" * 70)

    # Get training summary
    summary = logger.get_summary()
    summary.update({
        'iteration_dir': str(iteration_dir),
        'total_samples': len(dataset),
        'train_samples': train_size,
        'val_samples': val_size,
        'model_path': str(model_path),
        'seed': seed,
    })

    # Save metrics summary
    metrics_path = iteration_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Train] Training complete!")
    print(f"[Train] Best validation loss: {best_val_loss:.6f} (epoch {summary.get('best_epoch', 'N/A')})")
    print(f"[Train] Model saved to: {model_path}")
    print(f"[Train] Also copied to: {best_model_path}")
    print(f"[Train] Training log: {log_path}")

    return summary


def create_iteration_folder(robot_dir: Path, data_source: str) -> Path:
    """
    Create iteration folder by calling create_iteration.py script.

    Args:
        robot_dir: Path to Robot1 directory
        data_source: Data source directory name (e.g., 'training_data')

    Returns:
        Path to created iteration directory
    """
    script_path = robot_dir / "ai_training" / "create_iteration.py"

    print(f"\n[Train] Auto-creating iteration folder from {data_source}...")
    result = subprocess.run(
        [sys.executable, str(script_path), "--data", data_source],
        cwd=str(robot_dir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Failed to create iteration folder: {result.stderr}")

    print(result.stdout)

    # Parse output to find iteration directory
    for line in result.stdout.split('\n'):
        if 'Iteration directory:' in line:
            iteration_path = line.split('Iteration directory:')[1].strip()
            return Path(iteration_path)

    # Fallback: find most recent iteration folder
    experiments_dir = robot_dir / "experiments"
    iteration_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('iteration_')])
    if iteration_dirs:
        return iteration_dirs[-1]

    raise RuntimeError("Could not determine iteration directory path")


def main():
    parser = argparse.ArgumentParser(description="VRR AI Training Script (Phase 2)")

    # Main operation mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--data", type=str,
                           help="Training data directory (auto-creates iteration folder)")
    mode_group.add_argument("--iteration", type=str,
                           help="Existing iteration directory path (e.g., experiments/iteration_260110_090000)")

    # Optional overrides
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")

    # Score-based data filtering (DAgger+)
    parser.add_argument("--min-score", type=float, default=None,
                        help="Only use runs with score >= this value")
    parser.add_argument("--top-percent", type=float, default=None,
                        help="Only use top N%% of runs by score (e.g., 50 for top 50%%)")

    # Reward-Weighted BC / AWR options
    parser.add_argument("--mode", type=str, default="bc", choices=["bc", "rw", "awr"],
                        help="Training mode: bc (Behavioral Cloning), rw (Reward-Weighted BC), awr (Advantage-Weighted Regression)")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to existing model for fine-tuning (loads weights, uses smaller LR)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for reward weighting (higher = more uniform, default: 1.0)")
    parser.add_argument("--reward-type", type=str, default="simple", choices=["simple", "detailed"],
                        help="Reward calculation: simple (Phase 1) or detailed (Phase 2 with cumulative)")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = robot_dir / "experiments" / "config.yaml"

    if config_path.exists():
        print(f"[Train] Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print(f"[Train] Config not found, using defaults")
        config = {
            'seed': 42,
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'epochs': 100,
                'val_split': 0.2,
                'early_stopping': {'enabled': True, 'patience': 15},
                'scheduler': {'patience': 10, 'factor': 0.5}
            },
            'augmentation': {
                'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1}
            },
            'data_filtering': {'exclude_start_sequence': True},
            'paths': {
                'training_data': 'training_data',
                'experiments': 'experiments',
                'models': 'models'
            }
        }

    # Apply command-line overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.seed:
        config['seed'] = args.seed

    # Determine iteration directory
    if args.data:
        # Auto-create iteration folder
        data_source_dir = robot_dir / args.data
        if not data_source_dir.exists():
            print(f"[Train] ERROR: Data directory not found: {data_source_dir}")
            sys.exit(1)

        iteration_dir = create_iteration_folder(robot_dir, args.data)
        print(f"[Train] Created iteration: {iteration_dir}")
    else:
        # Use existing iteration folder
        iteration_dir = robot_dir / args.iteration if not Path(args.iteration).is_absolute() else Path(args.iteration)
        if not iteration_dir.exists():
            print(f"[Train] ERROR: Iteration directory not found: {iteration_dir}")
            sys.exit(1)
        print(f"[Train] Using existing iteration: {iteration_dir}")

    # Prepare finetune path if specified
    finetune_path = None
    if args.finetune:
        finetune_path = Path(args.finetune)
        if not finetune_path.exists():
            print(f"[Train] ERROR: Finetune model not found: {finetune_path}")
            sys.exit(1)
        print(f"[Train] Fine-tuning from: {finetune_path}")

    # Print training mode info
    if args.mode == "awr":
        print(f"[Train] Mode: AWR (temperature={args.temperature}, reward_type={args.reward_type})")
    elif args.mode == "rw":
        print(f"[Train] Mode: Reward-Weighted BC (temperature={args.temperature}, reward_type={args.reward_type})")
    else:
        print(f"[Train] Mode: Standard Behavioral Cloning")

    # Run training
    try:
        results = train(
            config,
            iteration_dir,
            robot_dir,
            data_source_dir=None,
            min_score=args.min_score,
            top_percent=args.top_percent,
            mode=args.mode,
            finetune_path=finetune_path,
            temperature=args.temperature,
            reward_type=args.reward_type
        )
        print(f"\n[Train] Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        print(f"\n[Train] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
