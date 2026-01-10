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
from model import DrivingNetwork


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


class DrivingDataset(Dataset):
    """
    Dataset for driving imitation learning.

    Loads image + SOC as input, drive_torque + steer_angle as targets.
    Supports loading from manifest or directory list.
    """

    VALID_RACING_STATUS = ["Lap1", "Lap2", "Finish"]

    def __init__(
        self,
        data_dirs: List[Path],
        transform=None,
        exclude_start_sequence: bool = True,
        valid_status: Optional[List[str]] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dirs: List of run directories
            transform: Image transforms
            exclude_start_sequence: Skip StartSequence frames
            valid_status: List of valid status values (default: Lap1, Lap2, Finish)
        """
        self.samples = []
        self.transform = transform
        self.valid_status = valid_status or self.VALID_RACING_STATUS

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            csv_path = data_dir / "metadata.csv"

            if not csv_path.exists():
                print(f"[Dataset] Warning: metadata.csv not found in {data_dir}")
                continue

            df = pd.read_csv(csv_path)
            images_dir = data_dir / "images"

            loaded = 0
            skipped = 0

            for _, row in df.iterrows():
                status = row.get("status", "")

                # Filter by status
                if exclude_start_sequence and status == "StartSequence":
                    skipped += 1
                    continue

                if status not in self.valid_status:
                    skipped += 1
                    continue

                img_path = images_dir / row["filename"]
                if not img_path.exists():
                    continue

                self.samples.append({
                    "image_path": str(img_path),
                    "soc": float(row["soc"]),
                    "drive_torque": float(row["drive_torque"]),
                    "steer_angle": float(row["steer_angle"]),
                    "run_name": data_dir.name,
                })
                loaded += 1

            print(f"[Dataset] {data_dir.name}: {loaded} samples loaded, {skipped} skipped")

        print(f"[Dataset] Total: {len(self.samples)} samples")

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

        return {"image": image, "soc": soc, "targets": targets}


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
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_torque_loss = 0.0
    total_steer_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        soc = batch["soc"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(images, soc)
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
    return {
        "loss": total_loss / n,
        "torque_loss": total_torque_loss / n,
        "steer_loss": total_steer_loss / n
    }


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


def find_data_directories(base_path: Path) -> List[Path]:
    """Find all run_* directories containing training data."""
    data_dirs = []
    for d in sorted(base_path.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            if (d / "metadata.csv").exists():
                data_dirs.append(d)
    return data_dirs


def train(
    config: Dict,
    iteration_dir: Path,
    robot_dir: Path,
    data_source_dir: Optional[Path] = None
) -> Dict:
    """
    Main training function.

    Args:
        config: Configuration dictionary
        iteration_dir: Path to iteration directory
        robot_dir: Path to Robot1 directory
        data_source_dir: Optional training data source directory

    Returns:
        Training results dictionary
    """
    # Setup paths
    training_data_dir = robot_dir / config['paths']['training_data']

    # Use data_sources if available, otherwise use training_data
    if (iteration_dir / "data_sources").exists():
        data_load_dir = iteration_dir / "data_sources"
        print(f"[Train] Using data from iteration data_sources: {data_load_dir}")
    elif data_source_dir:
        data_load_dir = data_source_dir
        print(f"[Train] Using data from: {data_load_dir}")
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

    # Load data directories from data_load_dir
    data_dirs = find_data_directories(data_load_dir)

    if not data_dirs:
        raise ValueError(f"No training data found in {data_load_dir}")

    print(f"[Train] Found {len(data_dirs)} data directories")

    # Create transforms
    train_transform, val_transform = get_transforms(config)

    # Create dataset
    train_config = config.get('training', {})
    dataset = DrivingDataset(
        data_dirs,
        transform=train_transform,
        exclude_start_sequence=config.get('data_filtering', {}).get('exclude_start_sequence', True)
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

    # Loss and optimizer
    criterion = nn.MSELoss()
    lr = train_config.get('learning_rate', 1e-4)
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

    print(f"\n[Train] Starting training for up to {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

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
    script_path = robot_dir / "scripts" / "create_iteration.py"

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

    # Run training
    try:
        results = train(config, iteration_dir, robot_dir, data_source_dir=None)
        print(f"\n[Train] Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        print(f"\n[Train] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
