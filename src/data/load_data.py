import os
import glob
import numpy as np
from PIL import Image

def load_dataset(raw_data_path="data/raw/plantvillage", image_size=(224, 224)):
    """
    Load images and labels from: data/raw/plantvillage/<class_name>/*.jpg
    
    Returns:
        images (np.ndarray): Array of resized images
        labels (np.ndarray): Integer labels
        class_names (list): List of folder names = class names
    """

    class_names = sorted([d for d in os.listdir(raw_data_path)
                          if os.path.isdir(os.path.join(raw_data_path, d))])

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
                print(f"Error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\nDataset loaded:")
    print(f"- Total images: {len(images)}")
    print(f"- Number of classes: {len(class_names)}")

    return images, labels, class_names


if __name__ == "__main__":
    X, y, classes = load_dataset()
    print("Classes:", classes)
    print("X shape:", X.shape)
    print("y shape:", y.shape)