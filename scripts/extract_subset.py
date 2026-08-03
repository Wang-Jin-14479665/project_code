import csv
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


RAW_DIR = Path(
    r"D:\UNSW26T2\COMP9517\Project\dataset_raw"
)

PROJECT_DIR = Path(
    r"D:\UNSW26T2\COMP9517\Project\project_code"
)

METADATA_DIR = PROJECT_DIR / "data" / "metadata"
OUTPUT_DIR = PROJECT_DIR / "data" / "subset_500"


MANIFESTS = [
    METADATA_DIR / "train_manifest.csv",
    METADATA_DIR / "validation_manifest.csv",
    METADATA_DIR / "test_manifest.csv",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    return rows


def normalize_tar_path(path: str | Path) -> str:
    """Normalize tar member paths for cross-platform matching."""
    raw_path = str(path).replace("\\", "/")

    while raw_path.startswith("./"):
        raw_path = raw_path[2:]

    raw_path = raw_path.lstrip("/")

    parts = [
        part for part in PurePosixPath(raw_path).parts
        if part not in {"", "."}
    ]

    return "/".join(parts)


def build_path_aliases(normalized_path: str) -> list[str]:
    """Create a small alias set that tolerates extra leading directories."""
    aliases = [normalized_path]
    parts = normalized_path.split("/")

    for index in range(1, len(parts)):
        suffix = "/".join(parts[index:])
        if suffix and suffix not in aliases:
            aliases.append(suffix)

    return aliases


def build_requested_lookup(
    records: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for record in records:
        normalized_source = normalize_tar_path(record["source_path"])
        aliases = build_path_aliases(normalized_source)

        for alias in aliases:
            if alias in lookup:
                existing = lookup[alias]
                if existing is not record:
                    raise ValueError(
                        "Ambiguous archive path alias detected: "
                        f"{alias} for {record['source_path']}"
                    )

        for alias in aliases:
            lookup[alias] = record

    return lookup


def extract_records(
    archive_path: Path,
    records: list[dict[str, str]],
    output_dir: Path,
) -> None:
    print(f"\nOpening archive: {archive_path}")
    print(f"Requested images: {len(records)}")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    pending = set()
    lookup = build_requested_lookup(records)

    for record in records:
        normalized_source = normalize_tar_path(record["source_path"])
        pending.add(normalized_source)

    completed_count = 0
    written_count = 0
    members_scanned = 0
    start_time = time.monotonic()

    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isreg():
                continue

            members_scanned += 1
            normalized_member_name = normalize_tar_path(member.name)
            record = lookup.get(normalized_member_name)

            if record is None:
                if members_scanned % 10000 == 0:
                    print(
                        f"[{archive_path.name}] scanned {members_scanned} members; "
                        f"{completed_count}/{len(records)} requested images accounted for"
                    )
                continue

            normalized_source = normalize_tar_path(record["source_path"])
            if normalized_source not in pending:
                continue

            split = record["split"]
            class_index = int(record["class_index"])
            image_id = record["image_id"]
            source_suffix = PurePosixPath(record["source_path"]).suffix or ".jpg"

            destination_dir = output_dir / split / f"class_{class_index:03d}"
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / f"{image_id}{source_suffix}"

            if destination_path.exists() and destination_path.stat().st_size > 0:
                pending.remove(normalized_source)
                completed_count += 1
            else:
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise RuntimeError(f"Unable to read {member.name}")

                with destination_path.open("wb") as output_file:
                    shutil.copyfileobj(source_file, output_file)

                pending.remove(normalized_source)
                completed_count += 1
                written_count += 1

            if (
                members_scanned % 10000 == 0
                or completed_count % 500 == 0
            ):
                elapsed = time.monotonic() - start_time
                rate = completed_count / elapsed if elapsed > 0 else float("inf")
                remaining = max(len(records) - completed_count, 0)
                remaining_time = remaining / rate if rate > 0 else None

                print(
                    f"[{archive_path.name}] scanned {members_scanned} members; "
                    f"completed {completed_count}/{len(records)}; "
                    f"wrote {written_count} new files; "
                    f"elapsed={elapsed:.1f}s; rate={rate:.2f} files/s"
                    + (
                        f"; eta={remaining_time:.1f}s"
                        if remaining_time is not None
                        else ""
                    )
                )

            if not pending:
                break

    if pending:
        missing_paths = sorted(pending)
        raise RuntimeError(
            f"Missing {len(missing_paths)} requested images from {archive_path.name}: "
            f"{missing_paths[:10]}"
        )

    elapsed = time.monotonic() - start_time
    print(
        f"[{archive_path.name}] finished: scanned {members_scanned} members; "
        f"completed {completed_count}/{len(records)} requested images; "
        f"elapsed={elapsed:.1f}s"
    )


def main() -> None:
    records_by_archive: dict[str, list[dict[str, str]]] = {}

    for manifest_path in MANIFESTS:
        records = read_manifest(manifest_path)

        for record in records:
            archive_name = record["source_archive"]
            records_by_archive.setdefault(archive_name, []).append(record)

    for archive_name, records in records_by_archive.items():
        archive_path = RAW_DIR / archive_name
        extract_records(archive_path, records, OUTPUT_DIR)

    print("\nDataset extraction completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()