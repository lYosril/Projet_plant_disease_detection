import os
import glob
import pickle
import numpy as np
from PIL import Image
import yaml

# Get project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "data_config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = cfg["paths"]["raw_data"]
INTERMEDIATE_FILE = cfg["paths"]["intermediate"]
IMG_SIZE = tuple(cfg["image"]["size"])

def load_dataset():
    """Load images from raw directory and return X, y, classes."""
    class_names = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    images, labels = [], []

    print(f"Found classes: {class_names}")

    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(RAW_DIR, class_name)
        img_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
        print(f"Loading {len(img_paths)} images from class: {class_name}")

        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(label_index)
            except:
                print(f"Error loading image: {img_path}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\nDataset loaded: {len(images)} images, {len(class_names)} classes.")

    return images, labels, class_names

if __name__ == "__main__":
    X, y, classes = load_dataset()
    os.makedirs(os.path.dirname(INTERMEDIATE_FILE), exist_ok=True)
    with open(INTERMEDIATE_FILE, "wb") as f:
        pickle.dump((X, y, classes), f)
    print(f"Dataset saved to {INTERMEDIATE_FILE}")
