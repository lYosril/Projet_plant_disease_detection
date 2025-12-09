# src/data/load_data.py
import os
import glob
import pickle
from PIL import Image
import yaml
import numpy as np

# Resolve project root relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "data_config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = os.path.join(ROOT_DIR, cfg["paths"]["raw_data"])
INTERMEDIATE_FILE = os.path.join(ROOT_DIR, cfg["paths"]["intermediate"])
IMAGE_SIZE = tuple(cfg["image"]["size"])


def load_dataset():
    """Load images from RAW_DIR into numpy arrays and save intermediate .pkl."""
    class_names = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    images = []
    labels = []

    print(f"Found classes: {class_names}")

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(RAW_DIR, class_name)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))

        print(f"Loading {len(image_paths)} images from class {class_name}")

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                images.append(np.array(img, dtype=np.uint8))
                labels.append(idx)
            except Exception as e:
                # don't raise — skip problematic files
                print(f"Warning: error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    # Save intermediate dataset
    os.makedirs(os.path.dirname(INTERMEDIATE_FILE), exist_ok=True)
    with open(INTERMEDIATE_FILE, "wb") as f:
        pickle.dump((images, labels, class_names), f)

    print(f"Dataset saved to {INTERMEDIATE_FILE} -> {len(images)} images, {len(class_names)} classes")
    return images, labels, class_names


if __name__ == "__main__":
    load_dataset()
