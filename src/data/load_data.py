import os
import glob
import numpy as np
from PIL import Image
import pickle


def load_dataset(raw_data_path="data/raw/plantvillage", image_size=(224, 224)):
    """
    Load images and labels from: data/raw/plantvillage/<class_name>/*.jpg
    
    Returns:
        images (np.ndarray): Array of resized images
        labels (np.ndarray): Integer labels
        class_names (list): List of folder names = class names
    """

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ Raw dataset folder not found: {raw_data_path}")

    class_names = sorted([
        d for d in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, d))
    ])

    if len(class_names) == 0:
        raise RuntimeError(f"❌ No class folders found inside {raw_data_path}")

    images = []
    labels = []

    print(f"Found classes ({len(class_names)}): {class_names}")

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
                print(f"⚠️ Error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print("\nDataset loaded:")
    print(f"- Total images: {len(images)}")
    print(f"- Number of classes: {len(class_names)}")

    return images, labels, class_names


def save_dataset(X, y, classes, output_path="data/intermediate/dataset.pkl"):
    """Save dataset to a pickle file for DVC pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": classes}, f)

    print(f"\n✅ Dataset saved to: {output_path}")


if __name__ == "__main__":
    X, y, classes = load_dataset()
    print("Classes:", classes)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    save_dataset(X, y, classes)