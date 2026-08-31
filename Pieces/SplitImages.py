import os
import shutil
import random
from pathlib import Path

def split_images(source_dir, output_dir, train_ratio=0.8):
    """Split images into train/val folders."""
    
    source = Path(source_dir)
    images = list(source.glob("*.png"))
    
    print(f"Total images: {len(images)}")
    
    if len(images) == 0:
        print("✗ No PNG files found!")
        return
    
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create directories
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    
    # Copy files
    for img in train_images:
        shutil.copy(img, f"{output_dir}/images/train/{img.name}")
    
    for img in val_images:
        shutil.copy(img, f"{output_dir}/images/val/{img.name}")
    
    print(f"✓ Train images: {len(train_images)}")
    print(f"✓ Val images: {len(val_images)}")

if __name__ == "__main__":
    split_images(
        source_dir="datasets/board",
        output_dir="datasets/board",
        train_ratio=0.8
    )