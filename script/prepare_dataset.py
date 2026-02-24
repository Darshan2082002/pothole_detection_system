import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ======================================================
# RESOLVE PROJECT ROOT SAFELY (PRODUCTION SAFE)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_IMAGES_DIR = BASE_DIR / "datasets" / "raw" / "images"
RAW_ANN_DIR = BASE_DIR / "datasets" / "raw" / "annotations"
OUTPUT_DIR = BASE_DIR / "datasets"



CLASS_NAME = "pothole"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.3
TEST_RATIO = 0.1
# ======================================================


def create_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_xml_to_yolo(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    with open(output_txt, "w") as f:
        for obj in root.findall("object"):
            class_id = 0  # single class: pothole
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")


def main():

    print("Project root:", BASE_DIR)
    print("Raw images dir:", RAW_IMAGES_DIR)
    print("Raw annotations dir:", RAW_ANN_DIR)

    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {RAW_IMAGES_DIR}")

    if not RAW_ANN_DIR.exists():
        raise FileNotFoundError(f"Annotations folder not found: {RAW_ANN_DIR}")

    create_dirs()

    # Collect images
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(RAW_IMAGES_DIR.glob(ext))

    print("Total images found:", len(images))

    if len(images) == 0:
        raise ValueError("No images found. Check RAW_IMAGES_DIR path.")

    # Split dataset
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        random_state=42
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    # Process splits
    for split_name, split_images in splits.items():
        print(f"\nProcessing {split_name} set ({len(split_images)} images)...")

        for img_path in tqdm(split_images):
            img_name = img_path.name
            xml_path = RAW_ANN_DIR / f"{img_path.stem}.xml"

            if not xml_path.exists():
                continue

            # Copy image
            shutil.copy(
                img_path,
                OUTPUT_DIR / split_name / "images" / img_name
            )

            # Convert annotation
            output_txt = OUTPUT_DIR / split_name / "labels" / f"{img_path.stem}.txt"
            convert_xml_to_yolo(xml_path, output_txt)

    # Create data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(
            f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
test: test/images

names:
  0: {CLASS_NAME}
"""
        )

    print("\nDataset preparation completed successfully!")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()