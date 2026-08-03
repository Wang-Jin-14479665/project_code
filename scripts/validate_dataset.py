import argparse
import csv
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


DATASET_ROOT = Path(
    r"D:\UNSW26T2\COMP9517\Project"
    r"\project_code\data\subset_500"
)

METADATA_DIR = DATASET_ROOT.parent / "metadata"

EXPECTED_COUNTS = {
    "train": 40,
    "val": 10,
    "test": 10,
}

EXPECTED_CLASSES = 500

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    return rows


def list_image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def validate_split_counts(split_dir: Path, split: str) -> tuple[list[Path], int]:
    class_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir()
    )

    if len(class_dirs) != EXPECTED_CLASSES:
        raise ValueError(
            f"{split} has {len(class_dirs)} classes, expected {EXPECTED_CLASSES}."
        )

    total_images = 0

    for class_dir in class_dirs:
        image_paths = list_image_files(class_dir)

        expected_count = EXPECTED_COUNTS[split]
        if len(image_paths) != expected_count:
            raise ValueError(
                f"{class_dir} has {len(image_paths)} images, expected {expected_count}."
            )

        total_images += len(image_paths)

    return class_dirs, total_images


def build_manifest_index(manifest_path: Path, split: str) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_manifest(manifest_path)
    index: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        class_dir_name = f"class_{int(row['class_index']):03d}"
        source_suffix = PurePosixPath(row["source_path"]).suffix.lower() or ".jpg"
        output_name = f"{row['image_id']}{source_suffix}"
        key = (class_dir_name, output_name)

        if key in index:
            raise ValueError(
                f"Duplicate expected output path in {split} manifest: {key}"
            )

        index[key] = row

    return index


def validate_manifest_alignment(split: str, split_dir: Path, manifest_path: Path) -> None:
    expected_rows = build_manifest_index(manifest_path, split)
    actual_files = []

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for image_path in list_image_files(class_dir):
            actual_files.append((class_dir.name, image_path.name))

    if len(actual_files) != len(expected_rows):
        raise ValueError(
            f"{split} has {len(actual_files)} output images but manifest expects {len(expected_rows)}."
        )

    for class_dir_name, output_name in actual_files:
        if (class_dir_name, output_name) not in expected_rows:
            raise ValueError(f"Unexpected output image in {split}: {class_dir_name}/{output_name}")

    for key in expected_rows:
        if key not in {
            (class_dir_name, output_name)
            for class_dir_name, output_name in actual_files
        }:
            raise ValueError(f"Missing output image for {split}: {key}")


def validate_no_overlap(manifests: dict[str, Path]) -> None:
    seen: dict[tuple[str, str], str] = {}

    for split, manifest_path in manifests.items():
        for row in read_manifest(manifest_path):
            key = (row["image_id"], row["source_path"])
            if key in seen:
                raise ValueError(f"Overlap detected between splits: {key}")
            seen[key] = split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the extracted subset dataset")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Number of images to verify with Pillow per split (default: 20)",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Verify every image file with Pillow",
    )
    args = parser.parse_args()

    manifests = {
        "train": METADATA_DIR / "train_manifest.csv",
        "val": METADATA_DIR / "validation_manifest.csv",
        "test": METADATA_DIR / "test_manifest.csv",
    }

    validate_no_overlap(manifests)

    total_images = 0
    invalid_images: list[Path] = []

    for split, manifest_path in manifests.items():
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory missing: {split_dir}")

        class_dirs, split_image_count = validate_split_counts(split_dir, split)
        total_images += split_image_count
        validate_manifest_alignment(split, split_dir, manifest_path)

        print(f"{split}: {len(class_dirs)} classes, {split_image_count} images")

        image_paths = [
            image_path
            for class_dir in class_dirs
            for image_path in list_image_files(class_dir)
        ]

        if args.check_all:
            verify_count = len(image_paths)
        else:
            verify_count = min(args.sample_limit, len(image_paths))

        for image_path in image_paths[:verify_count]:
            try:
                verify_image(image_path)
            except Exception as error:
                invalid_images.append(image_path)
                print(f"Invalid image: {image_path}")
                print(error)

    expected_total = EXPECTED_CLASSES * sum(EXPECTED_COUNTS.values())

    if total_images != expected_total:
        raise ValueError(f"Found {total_images} images, expected {expected_total}.")

    if invalid_images:
        raise ValueError("Some image files could not be opened.")

    print(f"\nTotal images: {total_images}")
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()