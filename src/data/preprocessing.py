import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Load config
with open("config/data_config.yaml") as f:
    cfg = yaml.safe_load(f)

processed_dir = cfg["paths"]["processed"]
img_size = tuple(cfg["image"]["size"])
split_cfg = cfg["split"]
aug_cfg = cfg["augmentation"]

def normalize_images(images):
    return images.astype("float32") / 255.0

def preprocess_pipeline():
    # Load raw dataset
    with open("data/intermediate/dataset.pkl", "rb") as f:
        X, y, classes = pickle.load(f)

    print("Normalizing images...")
    X = normalize_images(X)

    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y
    )
    val_ratio = split_cfg["val_size"] / (1 - split_cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=split_cfg["random_state"],
        stratify=y_temp
    )

    # Save datasets as .pkl
    os.makedirs(processed_dir, exist_ok=True)
    datasets = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
    for name, data in datasets.items():
        with open(os.path.join(processed_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f)
    print("Preprocessing complete. Datasets saved to", processed_dir)

if __name__ == "__main__":
    preprocess_pipeline()