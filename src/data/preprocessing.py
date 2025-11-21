import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize_images(images):
    """Normalize images to [0, 1]."""
    return images.astype("float32") / 255.0


def augment_data(X_train, y_train, augment=False):
    """
    Apply image augmentation using Keras ImageDataGenerator.
    Useful for model generalization.
    """

    if not augment:
        return X_train, y_train, None  # No generator used

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    datagen.fit(X_train)
    return X_train, y_train, datagen


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    Train: (1 - test_size - val_size)
    """

    # First split train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Now split (train+val) into train and val
    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed(
    X_train, X_val, X_test, y_train, y_val, y_test,
    save_dir="data/processed"
):
    """Save processed datasets as numpy arrays."""
    
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)

    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    print(f"Processed files saved to: {save_dir}")


def preprocess_pipeline(
    X, y,
    augment=False
):
    """
    Full preprocessing pipeline:
    - Normalize
    - Split
    - Augment (optional)
    - Save processed files
    """

    print("Normalizing images...")
    X = normalize_images(X)

    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("Applying augmentation..." if augment else "No augmentation.")
    X_train, y_train, datagen = augment_data(X_train, y_train, augment)

    print("Saving processed data...")
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, datagen


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_dataset

    X, y, classes = load_dataset()

    preprocess_pipeline(X, y, augment=True)
    print("Processing complete.")