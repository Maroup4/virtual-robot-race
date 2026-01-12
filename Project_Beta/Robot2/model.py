# model.py
# CNN + MLP Network for End-to-End Driving (Imitation Learning)
# Input: RGB Image (224x224) + SOC
# Output: drive_torque, steer_angle

import torch
import torch.nn as nn


class DrivingNetwork(nn.Module):
    """
    End-to-End Driving Network for Imitation Learning.

    Architecture:
        - CNN feature extractor (Image -> Feature Vector)
        - Concatenate with SOC
        - MLP head (Feature + SOC -> drive_torque, steer_angle)
    """

    def __init__(self, image_size=224):
        super().__init__()

        # CNN Feature Extractor
        # Input: [B, 3, 224, 224]
        self.cnn = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Global Average Pooling: 14x14 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Output: [B, 256, 1, 1] -> flatten -> [B, 256]

        # MLP Head
        # Input: CNN features (256) + SOC (1) = 257
        self.mlp = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 2)  # Output: [drive_torque, steer_angle]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, image, soc):
        """
        Forward pass.

        Args:
            image: [B, 3, 224, 224] - RGB image tensor (normalized)
            soc: [B, 1] - State of Charge tensor

        Returns:
            output: [B, 2] - [drive_torque, steer_angle]
        """
        # CNN feature extraction
        features = self.cnn(image)           # [B, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 256]

        # Concatenate with SOC
        combined = torch.cat([features, soc], dim=1)  # [B, 257]

        # MLP head
        output = self.mlp(combined)  # [B, 2]

        return output

    def predict(self, image, soc):
        """
        Inference method with output clamping.

        Returns:
            drive_torque: clamped to [-1, 1]
            steer_angle: clamped to [-0.524, 0.524] (~30 degrees, matches Unity)
        """
        with torch.no_grad():
            output = self.forward(image, soc)
            drive_torque = torch.clamp(output[:, 0], -1.0, 1.0)
            steer_angle = torch.clamp(output[:, 1], -0.524, 0.524)

        return drive_torque, steer_angle


# For backward compatibility with existing code
SteerNet = DrivingNetwork


if __name__ == "__main__":
    # Quick test
    model = DrivingNetwork()

    # Dummy input
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_soc = torch.randn(batch_size, 1)

    # Forward pass
    output = model(dummy_image, dummy_soc)
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input SOC shape: {dummy_soc.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
