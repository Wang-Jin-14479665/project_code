import json
from pathlib import Path


TRAIN_JSON = Path(
    r"D:\UNSW26T2\COMP9517\Project\dataset_raw\train_mini.json"
)

VAL_JSON = Path(
    r"D:\UNSW26T2\COMP9517\Project\dataset_raw\val.json"
)


def inspect_json(json_path: Path) -> None:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\nFile: {json_path.name}")
    print("Top-level keys:", data.keys())

    for key in data:
        value = data[key]

        if isinstance(value, list):
            print(f"{key}: {len(value)} items")

            if value:
                print(f"First {key} item:")
                print(value[0])
        else:
            print(f"{key}: {type(value)}")


inspect_json(TRAIN_JSON)
inspect_json(VAL_JSON)