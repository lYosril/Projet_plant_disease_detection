import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data.load_data import load_dataset, INTERMEDIATE_FILE
from src.data.load_data import cfg as data_cfg

PROCESSED_DIR = data_cfg["paths"]["processed"]

def split_and_save_dataset():
    # Load intermediate dataset (uint8, NOT normalized)
    if os.path.exists(INTERMEDIATE_FILE):
        with open(INTERMEDIATE_FILE, "rb") as f:
            X, y, class_names = pickle.load(f)
            print(f"Loaded intermediate dataset: {len(X)} images, {len(class_names)} classes")
    else:
        X, y, class_names = load_dataset()

    # DO NOT convert to float32 — keeps RAM usage small
    # Normalization is handled later by ImageDataGenerator

    # Splits
    test_size = data_cfg["split"]["test_size"]
    val_size  = data_cfg["split"]["val_size"]
    random_state = data_cfg["split"]["random_state"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    # Save splits
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    splits = {
        "train.pkl": (X_train, y_train),
        "val.pkl":   (X_val, y_val),
        "test.pkl":  (X_test, y_test),
    }

    for file, data in splits.items():
        with open(os.path.join(PROCESSED_DIR, file), "wb") as f:
            pickle.dump(data, f)

    print("Saved dataset splits to:", PROCESSED_DIR)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_generators(batch_size=32):
    def load_split(file):
        with open(os.path.join(PROCESSED_DIR, file), "rb") as f:
            return pickle.load(f)

    X_train, y_train = load_split("train.pkl")
    X_val, y_val     = load_split("val.pkl")
    X_test, y_test   = load_split("test.pkl")

    # Augmentation settings from config
    aug = data_cfg.get("augmentation", {})

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range      = aug.get("rotation_range", 20),
        width_shift_range   = aug.get("width_shift", 0.15),
        height_shift_range  = aug.get("height_shift", 0.15),
        zoom_range          = aug.get("zoom_range", 0.15),
        shear_range         = aug.get("shear_range", 0.10),
        brightness_range    = aug.get("brightness_range", [0.8, 1.2]),
        horizontal_flip     = aug.get("horizontal_flip", True),
        fill_mode="reflect"
    )

    # Validation / test never use augmentation
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = test_val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = test_val_datagen.flow(
        X_test, y_test,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Generators ready. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    split_and_save_dataset()