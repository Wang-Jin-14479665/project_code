import csv
import json
import random
from collections import defaultdict
from pathlib import Path


RAW_DIR = Path(
    r"D:\UNSW26T2\COMP9517\Project\dataset_raw"
)

PROJECT_DIR = Path(
    r"D:\UNSW26T2\COMP9517\Project\project_code"
)

TRAIN_JSON = RAW_DIR / "train_mini.json"
TEST_JSON = RAW_DIR / "val.json"

METADATA_DIR = PROJECT_DIR / "data" / "metadata"
SELECTED_CLASSES_PATH = METADATA_DIR / "selected_classes.json"

RANDOM_SEED = 9517
TRAIN_IMAGES_PER_CLASS = 40
VALIDATION_IMAGES_PER_CLASS = 10
TEST_IMAGES_PER_CLASS = 10


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_image_records(annotation_data: dict) -> list[dict]:
    """
    Convert COCO-style annotation data into records containing:
    image_id, file_name, category_id.
    """
    image_id_to_file_name = {
        image["id"]: image["file_name"]
        for image in annotation_data["images"]
    }

    records = []

    for annotation in annotation_data["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in image_id_to_file_name:
            raise KeyError(
                f"Image ID {image_id} is missing from the images section."
            )

        records.append({
            "image_id": image_id,
            "file_name": image_id_to_file_name[image_id],
            "category_id": annotation["category_id"],
        })

    return records


def group_by_category(
    records: list[dict],
    selected_class_ids: set[int],
) -> dict[int, list[dict]]:
    grouped = defaultdict(list)

    for record in records:
        category_id = record["category_id"]

        if category_id in selected_class_ids:
            grouped[category_id].append(record)

    return grouped


def write_manifest(path: Path, rows: list[dict]) -> None:
    field_names = [
        "source_archive",
        "source_path",
        "split",
        "original_category_id",
        "class_index",
        "image_id",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    train_data = load_json(TRAIN_JSON)
    test_data = load_json(TEST_JSON)
    selected_data = load_json(SELECTED_CLASSES_PATH)

    selected_class_ids = selected_data["class_ids"]
    selected_class_id_set = set(selected_class_ids)

    # Convert original category IDs into continuous labels 0–499.
    class_to_index = {
        category_id: class_index
        for class_index, category_id in enumerate(selected_class_ids)
    }

    train_records = build_image_records(train_data)
    test_records = build_image_records(test_data)

    train_grouped = group_by_category(
        train_records,
        selected_class_id_set,
    )

    test_grouped = group_by_category(
        test_records,
        selected_class_id_set,
    )

    rng = random.Random(RANDOM_SEED)

    train_rows = []
    validation_rows = []
    test_rows = []

    for category_id in selected_class_ids:
        mini_images = train_grouped.get(category_id, [])
        official_val_images = test_grouped.get(category_id, [])

        expected_mini_count = (
            TRAIN_IMAGES_PER_CLASS
            + VALIDATION_IMAGES_PER_CLASS
        )

        if len(mini_images) < expected_mini_count:
            raise ValueError(
                f"Category {category_id} has only "
                f"{len(mini_images)} train_mini images."
            )

        if len(official_val_images) < TEST_IMAGES_PER_CLASS:
            raise ValueError(
                f"Category {category_id} has only "
                f"{len(official_val_images)} official validation images."
            )

        # Deterministically shuffle within each class.
        mini_images = mini_images.copy()
        official_val_images = official_val_images.copy()

        rng.shuffle(mini_images)
        rng.shuffle(official_val_images)

        selected_train = mini_images[:TRAIN_IMAGES_PER_CLASS]

        selected_validation = mini_images[
            TRAIN_IMAGES_PER_CLASS:
            TRAIN_IMAGES_PER_CLASS + VALIDATION_IMAGES_PER_CLASS
        ]

        selected_test = official_val_images[:TEST_IMAGES_PER_CLASS]

        class_index = class_to_index[category_id]

        for record in selected_train:
            train_rows.append({
                "source_archive": "train_mini.tar.gz",
                "source_path": record["file_name"],
                "split": "train",
                "original_category_id": category_id,
                "class_index": class_index,
                "image_id": record["image_id"],
            })

        for record in selected_validation:
            validation_rows.append({
                "source_archive": "train_mini.tar.gz",
                "source_path": record["file_name"],
                "split": "val",
                "original_category_id": category_id,
                "class_index": class_index,
                "image_id": record["image_id"],
            })

        for record in selected_test:
            test_rows.append({
                "source_archive": "val.tar.gz",
                "source_path": record["file_name"],
                "split": "test",
                "original_category_id": category_id,
                "class_index": class_index,
                "image_id": record["image_id"],
            })

    write_manifest(
        METADATA_DIR / "train_manifest.csv",
        train_rows,
    )

    write_manifest(
        METADATA_DIR / "validation_manifest.csv",
        validation_rows,
    )

    write_manifest(
        METADATA_DIR / "test_manifest.csv",
        test_rows,
    )

    class_mapping_path = METADATA_DIR / "class_mapping.json"

    with class_mapping_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                str(category_id): class_index
                for category_id, class_index
                in class_to_index.items()
            },
            file,
            indent=2,
        )

    print(f"Training images:   {len(train_rows)}")
    print(f"Validation images: {len(validation_rows)}")
    print(f"Test images:       {len(test_rows)}")
    print(f"Classes:           {len(selected_class_ids)}")


if __name__ == "__main__":
    main()