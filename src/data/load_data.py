import os
import glob
import pickle
import numpy as np
from PIL import Image
import yaml

def load_config():
    with open("config/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

def load_dataset():
    raw_data_path = cfg["paths"]["raw_data"]
    image_size = tuple(cfg["image"]["size"])

    class_names = sorted([
        d for d in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, d))
    ])

    images = []
    labels = []

    print(f"Found classes: {class_names}")

    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(raw_data_path, class_name)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))

        print(f"Loading {len(image_paths)} images from class: {class_name}")

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)
                images.append(np.array(img))
                labels.append(label_index)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    print(f"\nDataset loaded: {len(images)} images, {len(class_names)} classes")

    return images, labels, class_names

if __name__ == "__main__":
    X, y, classes = load_dataset()
    os.makedirs("data/intermediate", exist_ok=True)
    with open("data/intermediate/dataset.pkl", "wb") as f:
        pickle.dump((X, y, classes), f)
    print("Dataset saved to data/intermediate/dataset.pkl")