import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# Paths and split ratios
# ============================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TEST_RATIO = 0.1   # 10% test
VAL_RATIO = 0.2    # 20% validation from remaining
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


# ============================================================
# Split raw dataset into train/val/test on disk
# ============================================================
def split_dataset_on_disk(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR,
                          test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_state=42):
    """
    Splits images into train/val/test folders on disk for lazy loading.
    Only copies files; skips directories.
    Automatically handles extra top-level folder (e.g., PlantVillage).
    """
    np.random.seed(random_state)

    # Detect if there is an extra top-level folder
    top_level_subdirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if len(top_level_subdirs) == 1:
        raw_dir = os.path.join(raw_dir, top_level_subdirs[0])

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        np.random.shuffle(images)

        n_test = int(len(images) * test_ratio)
        n_val = int(len(images) * val_ratio)

        val_imgs = images[:n_val]
        test_imgs = images[n_val:n_val + n_test]
        train_imgs = images[n_val + n_test:]

        for subset_name, subset_imgs in zip(
            ["train", "val", "test"], [train_imgs, val_imgs, test_imgs]
        ):
            subset_dir = os.path.join(processed_dir, subset_name, cls)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset_imgs:
                src_path = os.path.join(cls_path, img)
                dst_path = os.path.join(subset_dir, img)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

    print(f"Dataset successfully split into train/val/test under '{processed_dir}'.")


# ============================================================
# Create Keras ImageDataGenerators
# ============================================================
def create_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Return train, val, test generators with augmentation for training only."""
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validation/test generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Splitting dataset on disk...")
    split_dataset_on_disk()

    print("Creating generators...")
    train_gen, val_gen, test_gen = create_generators()

    print("Generators ready. You can now feed them directly into model.fit():")
    print(f"Train samples: {train_gen.n}, Validation samples: {val_gen.n}, Test samples: {test_gen.n}")