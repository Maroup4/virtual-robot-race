#!/usr/bin/env python3
# verify_model_input.py
# ==============================================================================
#                     MODEL INPUT/OUTPUT VERIFICATION
# ==============================================================================
# Verifies that:
# 1. Images are correctly loaded from training_data/
# 2. Preprocessing is applied correctly
# 3. Model receives correct input shapes
# 4. Model produces reasonable outputs
# 5. ai_control_strategy.py post-processing works

import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random

# Add Robot1 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model
from model import DrivingNetwork

# Import strategy
from ai_control_strategy import adjust_output

def verify_single_image(image_path, soc=1.0):
    """Verify inference pipeline for a single image"""

    print("="*80)
    print(f"Testing image: {image_path.name}")
    print("="*80)

    # 1. Load image
    try:
        pil_img = Image.open(image_path).convert("RGB")
        print(f"[1] Image loaded: size={pil_img.size}, mode={pil_img.mode}")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return

    # 2. Preprocess (same as inference_input.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        image_tensor = transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]
        print(f"[2] Preprocessed: shape={list(image_tensor.shape)}, "
              f"dtype={image_tensor.dtype}, "
              f"range=[{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return

    # 3. Prepare SOC tensor
    soc_tensor = torch.tensor([[soc]], dtype=torch.float32)  # [1, 1]
    print(f"[3] SOC tensor: shape={list(soc_tensor.shape)}, value={soc}")

    # 4. Load model
    model_path = Path(__file__).parent.parent / "models" / "model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4] Device: {device}")

    try:
        model = DrivingNetwork()
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
        print(f"[4] Model loaded from {model_path.name}")
    except FileNotFoundError:
        print(f"[ERROR] Model not found at {model_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 5. Run inference
    try:
        image_tensor = image_tensor.to(device)
        soc_tensor = soc_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor, soc_tensor)
            raw_drive = output[0, 0].item()
            raw_steer = output[0, 1].item()

        print(f"[5] Model output:")
        print(f"    Raw drive: {raw_drive:+.4f}")
        print(f"    Raw steer: {raw_steer:+.4f} rad ({raw_steer*57.3:+.1f} deg)")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Apply saturation (same as inference_input.py)
    raw_drive = max(-1.0, min(1.0, raw_drive))
    raw_steer = max(-0.785, min(0.785, raw_steer))
    print(f"[6] After saturation:")
    print(f"    Drive: {raw_drive:+.4f}")
    print(f"    Steer: {raw_steer:+.4f} rad ({raw_steer*57.3:+.1f} deg)")

    # 7. Apply ai_control_strategy.py post-processing
    try:
        adj_drive, adj_steer = adjust_output(
            raw_drive, raw_steer,
            pil_img, soc,
            race_started=True  # Assume race started
        )
        print(f"[7] After ai_control_strategy.py:")
        print(f"    Final drive: {adj_drive:+.4f} (delta: {adj_drive-raw_drive:+.4f})")
        print(f"    Final steer: {adj_steer:+.4f} rad ({adj_steer*57.3:+.1f} deg) "
              f"(delta: {adj_steer-raw_steer:+.4f} rad)")
    except Exception as e:
        print(f"[ERROR] Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print()

def main():
    print()
    print("="*80)
    print("           MODEL INPUT/OUTPUT VERIFICATION")
    print("="*80)
    print()

    # Find training data directory
    robot_dir = Path(__file__).parent.parent
    training_data_dir = robot_dir / "training_data"

    if not training_data_dir.exists():
        print(f"Error: Training data directory not found: {training_data_dir}")
        return

    # Find all image directories
    image_dirs = []
    for run_dir in training_data_dir.iterdir():
        if run_dir.is_dir():
            img_dir = run_dir / "images"
            if img_dir.exists():
                image_dirs.append(img_dir)

    if not image_dirs:
        print(f"No image directories found in {training_data_dir}")
        return

    print(f"Found {len(image_dirs)} runs with images")
    print()

    # Select random images from different runs
    num_tests = min(5, len(image_dirs))
    print(f"Testing {num_tests} random images from different runs...")
    print()

    for i in range(num_tests):
        # Pick a random run
        img_dir = random.choice(image_dirs)

        # Pick a random image from that run
        images = list(img_dir.glob("*.jpg"))
        if not images:
            continue

        random_image = random.choice(images)

        # Test with different SOC values
        soc = 1.0 if i < 2 else 0.5  # First 2 with full SOC, rest with half

        verify_single_image(random_image, soc=soc)

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("If you see reasonable drive/steer values above, the pipeline is working.")
    print()
    print("Expected behavior:")
    print("  - Drive: should be between -1.0 and 1.0 (typically 0.2 to 0.6)")
    print("  - Steer: should be between -0.785 and 0.785 rad (-45 to +45 deg)")
    print("  - Post-processing should apply rate limiting and torque capping")
    print()
    print("If you see:")
    print("  - All zeros: Model is not loaded or producing dummy output")
    print("  - Extreme values: Model may not be trained properly")
    print("  - Consistent left bias (negative steer): Model has left steering bias")
    print()

if __name__ == "__main__":
    main()
