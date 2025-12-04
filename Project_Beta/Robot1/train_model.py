# train_model.py
# Training script for End-to-End Driving Model (Imitation Learning)
# Learns to mimic human driving from keyboard mode recordings

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd


# Import model from same directory
from model import DrivingNetwork


class DrivingDataset(Dataset):
    """
    Dataset for driving imitation learning.

    Loads image + SOC as input, drive_torque + steer_angle as targets.
    Only uses data AFTER the start signal (excludes StartSequence).
    """

    def __init__(self, data_dirs, transform=None, exclude_start_sequence=True):
        """
        Args:
            data_dirs: List of paths to data directories (each containing metadata.csv and images/)
            transform: Image transforms to apply
            exclude_start_sequence: If True, skip StartSequence data (recommended)
        """
        self.samples = []
        self.transform = transform

        # Handle single directory or list
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            csv_path = data_dir / "metadata.csv"

            if not csv_path.exists():
                print(f"[Warning] metadata.csv not found in {data_dir}, skipping...")
                continue

            df = pd.read_csv(csv_path)
            images_dir = data_dir / "images"

            loaded_count = 0
            skipped_start = 0
            for _, row in df.iterrows():
                # Skip StartSequence data (before race start)
                if exclude_start_sequence and row.get("status") == "StartSequence":
                    skipped_start += 1
                    continue

                img_path = images_dir / row["filename"]

                if not img_path.exists():
                    continue

                self.samples.append({
                    "image_path": str(img_path),
                    "soc": float(row["soc"]),
                    "drive_torque": float(row["drive_torque"]),
                    "steer_angle": float(row["steer_angle"]),
                })
                loaded_count += 1

            if skipped_start > 0:
                print(f"[Dataset] Loaded {loaded_count} samples from {data_dir.name} (skipped {skipped_start} StartSequence)")
            else:
                print(f"[Dataset] Loaded {loaded_count} samples from {data_dir.name}")

        print(f"[Dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # SOC as tensor
        soc = torch.tensor([sample["soc"]], dtype=torch.float32)

        # Targets: [drive_torque, steer_angle]
        targets = torch.tensor([
            sample["drive_torque"],
            sample["steer_angle"]
        ], dtype=torch.float32)

        return {
            "image": image,
            "soc": soc,
            "targets": targets
        }


def get_data_transforms():
    """Get image transforms for training and validation."""
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

        # Forward pass
        outputs = model(images, soc)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Individual losses for logging
        with torch.no_grad():
            torque_loss = nn.MSELoss()(outputs[:, 0], targets[:, 0])
            steer_loss = nn.MSELoss()(outputs[:, 1], targets[:, 1])
            total_torque_loss += torque_loss.item()
            total_steer_loss += steer_loss.item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "torque_loss": total_torque_loss / n_batches,
        "steer_loss": total_steer_loss / n_batches
    }


def validate(model, dataloader, criterion, device):
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

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "torque_loss": total_torque_loss / n_batches,
        "steer_loss": total_steer_loss / n_batches
    }


def find_data_directories(base_path):
    """Find all run_* directories containing training data."""
    base_path = Path(base_path)
    data_dirs = []

    for d in sorted(base_path.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            csv_path = d / "metadata.csv"
            if csv_path.exists():
                data_dirs.append(d)

    return data_dirs


def main():
    parser = argparse.ArgumentParser(description="Train End-to-End Driving Model")
    parser.add_argument("--data", type=str, default="training_data",
                        help="Path to training data directory (contains run_* folders)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--output", type=str, default="models/model.pth",
                        help="Output model path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Train] Using device: {device}")

    # Find data directories
    script_dir = Path(__file__).parent
    data_base = script_dir / args.data

    data_dirs = find_data_directories(data_base)
    if not data_dirs:
        print(f"[Error] No training data found in {data_base}")
        print("Expected structure: training_data/run_YYYYMMDD_HHMMSS/metadata.csv")
        return

    print(f"[Train] Found {len(data_dirs)} data directories:")
    for d in data_dirs:
        print(f"  - {d.name}")

    # Create dataset
    train_transform, val_transform = get_data_transforms()

    # Use train transform for full dataset (will split later)
    full_dataset = DrivingDataset(data_dirs, transform=train_transform)

    if len(full_dataset) == 0:
        print("[Error] No samples loaded. Check your data.")
        return

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[Train] Training samples: {train_size}")
    print(f"[Train] Validation samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # Create model
    model = DrivingNetwork().to(device)
    print(f"[Train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    best_val_loss = float('inf')
    output_path = script_dir / args.output

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics["loss"])

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_path)
            save_marker = " [SAVED]"
        else:
            save_marker = ""

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.6f} "
              f"(T:{train_metrics['torque_loss']:.4f}, S:{train_metrics['steer_loss']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.6f} "
              f"(T:{val_metrics['torque_loss']:.4f}, S:{val_metrics['steer_loss']:.4f})"
              f"{save_marker}")

    print("=" * 60)
    print(f"[Train] Training complete!")
    print(f"[Train] Best validation loss: {best_val_loss:.6f}")
    print(f"[Train] Model saved to: {output_path}")


if __name__ == "__main__":
    main()
