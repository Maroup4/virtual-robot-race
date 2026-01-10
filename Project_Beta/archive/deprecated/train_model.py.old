# train_model.py
# Training script for steer-type control AI model
# Reads metadata.csv and images from training_data/run_* folders
# Trains a PyTorch model to predict (drive_torque, steer_angle) from RGB + SOC

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Import the model architecture from inference_input
from inference_input import SteerNet


class SteerDataset(Dataset):
    """Dataset for steer-type control training."""

    def __init__(self, csv_paths, transform=None):
        """
        Args:
            csv_paths: List of paths to metadata.csv files
            transform: Image transformations
        """
        self.data = []
        self.transform = transform

        for csv_path in csv_paths:
            csv_path = Path(csv_path)
            run_dir = csv_path.parent
            images_dir = run_dir / "images"

            if not csv_path.exists():
                print(f"[Dataset] WARNING: CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            print(f"[Dataset] Loading {len(df)} samples from {csv_path.name}")

            for _, row in df.iterrows():
                filename = row.get("filename", "")
                if not filename:
                    continue

                img_path = images_dir / filename
                if not img_path.exists():
                    continue

                # Read control targets
                drive = float(row.get("drive_torque", 0.0))
                steer = float(row.get("steer_angle", 0.0))
                soc = float(row.get("soc", 1.0))

                # Filter out invalid data (e.g., status errors)
                status = str(row.get("status", "")).lower()
                if status in ["error", "emergency", "stop"]:
                    continue

                self.data.append({
                    "image_path": str(img_path),
                    "drive_torque": drive,
                    "steer_angle": steer,
                    "soc": soc,
                })

        print(f"[Dataset] Total valid samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load and transform image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Flatten image and concatenate with SOC
        image_flat = image.view(-1)  # (224*224*3,)
        soc_tensor = torch.tensor([sample["soc"]], dtype=torch.float32)
        input_tensor = torch.cat([image_flat, soc_tensor])  # (224*224*3 + 1,)

        # Target outputs
        target = torch.tensor([
            sample["drive_torque"],
            sample["steer_angle"]
        ], dtype=torch.float32)

        return input_tensor, target


def train_model(
    train_csv_paths,
    val_csv_paths=None,
    model_save_path="models/model.pth",
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu"
):
    """Train the SteerNet model."""

    print("[Train] Starting training...")
    print(f"[Train] Device: {device}")
    print(f"[Train] Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = SteerDataset(train_csv_paths, transform=transform)
    if len(train_dataset) == 0:
        print("[Train] ERROR: No training data found!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = None
    if val_csv_paths:
        val_dataset = SteerDataset(val_csv_paths, transform=transform)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

    # Create model
    input_size = 224 * 224 * 3 + 1
    model = SteerNet(input_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"[Train] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"[Train] Saved best model to {model_save_path}")
        else:
            print(f"[Train] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

    # Save final model if no validation set
    if not val_loader:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"[Train] Saved final model to {model_save_path}")

    print("[Train] Training completed!")


def find_metadata_csv_files(training_data_root="training_data"):
    """Find all metadata.csv files in training_data folders."""
    root = Path(training_data_root)
    if not root.exists():
        return []

    csv_files = list(root.glob("run_*/metadata.csv"))
    return sorted(csv_files)


if __name__ == "__main__":
    # Find all training data
    csv_files = find_metadata_csv_files()

    if not csv_files:
        print("[Train] ERROR: No metadata.csv files found in training_data/")
        print("[Train] Please run some races in rule_based or keyboard mode to collect data.")
        sys.exit(1)

    print(f"[Train] Found {len(csv_files)} metadata.csv files:")
    for f in csv_files:
        print(f"  - {f}")

    # Split into train/val (80/20)
    split_idx = int(len(csv_files) * 0.8)
    train_csvs = csv_files[:split_idx] if split_idx > 0 else csv_files
    val_csvs = csv_files[split_idx:] if split_idx < len(csv_files) else None

    print(f"[Train] Training on {len(train_csvs)} runs, Validation on {len(val_csvs) if val_csvs else 0} runs")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train model
    train_model(
        train_csv_paths=train_csvs,
        val_csv_paths=val_csvs,
        model_save_path="models/model.pth",
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        device=device
    )
