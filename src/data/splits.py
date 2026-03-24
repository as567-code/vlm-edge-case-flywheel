import json
import random
from pathlib import Path
from collections import defaultdict


def assign_splits(
    entries: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """Assign stratified train/val/test splits to manifest entries.

    Stratifies by category to ensure balanced representation across splits.
    The test split gets exactly the remaining frames after train+val.
    """
    rng = random.Random(seed)

    by_category = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)

    result = []
    for cat, cat_entries in by_category.items():
        rng.shuffle(cat_entries)
        n = len(cat_entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, entry in enumerate(cat_entries):
            if i < n_train:
                entry["split"] = "train"
            elif i < n_train + n_val:
                entry["split"] = "val"
            else:
                entry["split"] = "test"
            result.append(entry)

    return result


def save_manifest(entries: list[dict], path: str | Path) -> None:
    """Save manifest entries to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
