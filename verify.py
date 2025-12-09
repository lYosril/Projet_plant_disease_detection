import json
import os

RAW_DATA_PATH = "data/raw/plantvillage"
JSON_PATH = "configs/class_names.json"

def get_dataset_classes(raw_data_path):
    """Returns alphabetical folder list from dataset."""
    return sorted([
        d for d in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, d))
    ])

def load_json_classes(json_path):
    """Load class names stored in JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

def save_json_classes(json_path, classes):
    """Overwrite JSON with correct class list."""
    with open(json_path, "w") as f:
        json.dump(classes, f, indent=4)
    print(f"\n✅ Updated {json_path} with corrected class order.\n")

if __name__ == "__main__":
    print("\n🔍 Checking class order...")

    dataset_classes = get_dataset_classes(RAW_DATA_PATH)
    json_classes = load_json_classes(JSON_PATH)

    print("\n=== Dataset Folders (Alphabetical) ===")
    for i, c in enumerate(dataset_classes):
        print(f"{i}: {c}")

    print("\n=== JSON Class Names (Before Fix) ===")
    for i, c in enumerate(json_classes):
        print(f"{i}: {c}")

    if dataset_classes == json_classes:
        print("\n✅ JSON already matches dataset order. No changes needed.\n")
    else:
        print("\n❌ MISMATCH — Fixing JSON now...\n")
        save_json_classes(JSON_PATH, dataset_classes)

        print("=== JSON Class Names (After Fix) ===")
        for i, c in enumerate(dataset_classes):
            print(f"{i}: {c}")

        print("\n🎉 Class order successfully corrected!")