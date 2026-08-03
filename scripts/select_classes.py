import json
import random
from pathlib import Path


TRAIN_JSON = Path(
    r"D:\UNSW26T2\COMP9517\Project\dataset_raw\train_mini.json"
)

OUTPUT_DIR = Path(
    r"D:\UNSW26T2\COMP9517\Project\project_code\data\metadata"
)

RANDOM_SEED = 9517
NUM_CLASSES = 500


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with TRAIN_JSON.open("r", encoding="utf-8") as file:
        train_data = json.load(file)

    category_ids = sorted({
        annotation["category_id"]
        for annotation in train_data["annotations"]
    })

    print(f"Total available categories: {len(category_ids)}")

    if len(category_ids) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(category_ids)} categories are available, "
            f"but {NUM_CLASSES} were requested."
        )

    random_generator = random.Random(RANDOM_SEED)

    selected_class_ids = sorted(
        random_generator.sample(category_ids, NUM_CLASSES)
    )

    output = {
        "random_seed": RANDOM_SEED,
        "num_classes": NUM_CLASSES,
        "class_ids": selected_class_ids,
    }

    output_path = OUTPUT_DIR / "selected_classes.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved selected classes to: {output_path}")
    print("First 10 selected category IDs:")
    print(selected_class_ids[:10])


if __name__ == "__main__":
    main()