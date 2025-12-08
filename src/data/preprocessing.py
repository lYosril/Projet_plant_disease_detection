import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "data_config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

INTERMEDIATE_FILE = cfg["paths"]["intermediate"]
PROCESSED_DIR = cfg["paths"]["processed"]
IMG_SIZE = tuple(cfg["image"]["size"])
BATCH_SIZE = 32
AUG = cfg["augmentation"]["enabled"]

def create_generators():
    """Create lazy ImageDataGenerators for train/val/test from processed folder."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=cfg["augmentation"]["rotation_range"] if AUG else 0,
        width_shift_range=cfg["augmentation"]["width_shift"] if AUG else 0,
        height_shift_range=cfg["augmentation"]["height_shift"] if AUG else 0,
        zoom_range=cfg["augmentation"]["zoom_range"] if AUG else 0,
        horizontal_flip=cfg["augmentation"]["horizontal_flip"] if AUG else False,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        PROCESSED_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training"
    )

    val_gen = test_datagen.flow_from_directory(
        PROCESSED_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation"
    )

    test_gen = test_datagen.flow_from_directory(
        PROCESSED_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def split_dataset_on_disk():
    """Split raw images into train/val/test folders (DVC-friendly)."""
    with open(INTERMEDIATE_FILE, "rb") as f:
        X, y, classes = pickle.load(f)

    from sklearn.model_selection import train_test_split
    import shutil

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for subset in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(PROCESSED_DIR, subset, cls), exist_ok=True)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"], stratify=y)
    val_ratio = cfg["split"]["val_size"] / (1 - cfg["split"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=cfg["split"]["random_state"], stratify=y_temp)

    def save_images(subset, Xs, ys):
        for img, label in zip(Xs, ys):
            cls_name = classes[label]
            idx = np.random.randint(1e6)
            out_path = os.path.join(PROCESSED_DIR, subset, cls_name, f"{idx}.png")
            from PIL import Image
            Image.fromarray(img).save(out_path)

    save_images("train", X_train, y_train)
    save_images("val", X_val, y_val)
    save_images("test", X_test, y_test)
    print(f"Dataset successfully split under '{PROCESSED_DIR}'")

if __name__ == "__main__":
    split_dataset_on_disk()
    print("Generators ready:")
    create_generators()