from pathlib import Path
from collections import Counter
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "face_mask_dataset"
)

CLASS_NAMES = {
    0: "bad mask",
    1: "mask",
    2: "no mask"
}

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    print("\n" + "=" * 60)
    print(f"VALIDATING: {split.upper()}")
    print("=" * 60)

    if not image_dir.exists():
        print(f"❌ Image directory missing: {image_dir}")
        return

    if not label_dir.exists():
        print(f"❌ Label directory missing: {label_dir}")
        return

    images = [
        file
        for file in image_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    # --------------------------------------------------------
    # Check image-label matching
    # --------------------------------------------------------

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        print(f"⚠️ Images without labels: {len(missing_labels)}")
    else:
        print("✅ Every image has a matching label")

    if orphan_labels:
        print(f"⚠️ Labels without images: {len(orphan_labels)}")
    else:
        print("✅ No orphan labels found")

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    corrupted_images = []

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted_images.append(image_path.name)

    if corrupted_images:
        print(f"❌ Corrupted images: {len(corrupted_images)}")

        for image in corrupted_images[:10]:
            print(f"   - {image}")
    else:
        print("✅ No corrupted images found")

    # --------------------------------------------------------
    # Validate YOLO labels
    # --------------------------------------------------------

    invalid_labels = []
    class_counter = Counter()
    total_boxes = 0

    for label_path in labels:

        try:
            with open(label_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            for line_number, line in enumerate(lines, start=1):

                line = line.strip()

                # Empty lines are ignored
                if not line:
                    continue

                parts = line.split()

                # YOLO detection format:
                # class x_center y_center width height

                if len(parts) != 5:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} "
                        f"must contain 5 values"
                    )
                    continue

                try:
                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                except ValueError:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} "
                        f"contains non-numeric values"
                    )
                    continue

                # Check class ID

                if class_id not in CLASS_NAMES:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} "
                        f"invalid class ID {class_id}"
                    )
                    continue

                # Check normalized coordinates

                values = [
                    x_center,
                    y_center,
                    width,
                    height
                ]

                if not all(0 <= value <= 1 for value in values):
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} "
                        f"coordinates must be between 0 and 1"
                    )
                    continue

                # Width and height must be positive

                if width <= 0 or height <= 0:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} "
                        f"width/height must be greater than 0"
                    )
                    continue

                class_counter[class_id] += 1
                total_boxes += 1

        except Exception as error:

            invalid_labels.append(
                f"{label_path.name}: {error}"
            )

    # --------------------------------------------------------
    # Print label results
    # --------------------------------------------------------

    if invalid_labels:
        print(f"❌ Invalid annotations: {len(invalid_labels)}")

        for error in invalid_labels[:10]:
            print(f"   - {error}")

    else:
        print("✅ All YOLO annotations are valid")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]

        print(
            f"   {class_id} - {class_name:<12}: {count} objects"
        )

    print(f"\nTotal bounding boxes: {total_boxes}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       FACE MASK DATASET VALIDATION")
    print("=" * 60)

    print(f"\nDataset location:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        print("\n❌ Dataset directory does not exist!")
        return

    for split in SPLITS:
        validate_split(split)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()