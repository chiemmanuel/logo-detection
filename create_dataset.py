import os
import shutil
import random
from pathlib import Path

def create_yolo_dataset(
    input_dataset_path,
    output_dataset_path,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    input_dataset = Path(input_dataset_path)
    output_dataset = Path(output_dataset_path)

    # Paths inside input dataset
    images_path = input_dataset / "images"
    labels_path = input_dataset / "labels"

    if not images_path.exists() or not labels_path.exists():
        raise ValueError("Input dataset must contain 'images/' and 'labels/' folders.")

    # Prepare output structure
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dataset / sub).mkdir(parents=True, exist_ok=True)

    # Collect image-label pairs
    valid_pairs = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    for image_file in images_path.iterdir():
        if image_file.suffix.lower() not in image_extensions:
            continue

        label_file = labels_path / (image_file.stem + ".txt")

        if label_file.exists():
            valid_pairs.append((image_file, label_file))
        else:
            print(f"[WARNING] No label found for image: {image_file.name}")

    if not valid_pairs:
        raise ValueError("No valid image-label pairs found.")

    # Shuffle dataset
    random.shuffle(valid_pairs)

    # Split
    train_size = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:]

    print(f"Total pairs: {len(valid_pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")

    # Copy files
    def copy_pair(img, lbl, split):
        shutil.copy(img, output_dataset / f"images/{split}/{img.name}")
        shutil.copy(lbl, output_dataset / f"labels/{split}/{lbl.name}")

    for img, lbl in train_pairs:
        copy_pair(img, lbl, "train")

    for img, lbl in val_pairs:
        copy_pair(img, lbl, "val")

    print("Dataset split completed successfully!")
    print(f"Structured dataset created at: {output_dataset}")


create_yolo_dataset(
    input_dataset_path="datasets/test_create_dataset",
    output_dataset_path="datasets/auto_split_dataset",
    train_ratio=0.8
)
