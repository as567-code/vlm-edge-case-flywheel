import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


CATEGORY_TO_IDX = {
    "construction_zone": 0,
    "emergency_vehicle": 1,
    "lane_blockage": 2,
    "normal": 3,
}

IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}


class DrivingFrameDataset(Dataset):
    """PyTorch Dataset for driving frames loaded from a manifest.json."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        transform=None,
        root_dir: str | Path | None = None,
    ):
        with open(manifest_path) as f:
            all_entries = json.load(f)

        if split == "all":
            self.entries = all_entries
        else:
            self.entries = [e for e in all_entries if e["split"] == split]
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else Path(manifest_path).parent.parent

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        img_path = self.root_dir / entry["path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": CATEGORY_TO_IDX[entry["category"]],
            "category": entry["category"],
            "is_ood": entry["is_ood"],
            "frame_id": entry["frame_id"],
        }


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load the full manifest as a list of dicts."""
    with open(manifest_path) as f:
        return json.load(f)
