"""Tests for Stage 1: data pipeline, manifest, splits, and dataset loading."""

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from src.data.dataset import DrivingFrameDataset, CATEGORY_TO_IDX, IDX_TO_CATEGORY
from src.data.augmentations import get_clip_preprocess, get_train_augmentations
from src.data.splits import assign_splits
from src.utils.device import get_device


MANIFEST_PATH = Path("data/manifest.json")
OOD_CATEGORIES = ["construction_zone", "emergency_vehicle", "lane_blockage"]


@pytest.fixture
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("Manifest not yet generated. Run scripts/download_data.py first.")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


class TestManifest:
    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists(), "data/manifest.json does not exist"

    def test_manifest_total_count(self, manifest):
        assert len(manifest) == 4200, f"Expected 4200 entries, got {len(manifest)}"

    def test_manifest_categories(self, manifest):
        cats = set(m["category"] for m in manifest)
        expected = {"construction_zone", "emergency_vehicle", "lane_blockage", "normal"}
        assert cats == expected, f"Categories: {cats}"

    def test_manifest_ood_count(self, manifest):
        ood = [m for m in manifest if m["is_ood"]]
        assert len(ood) == 1200, f"Expected 1200 OOD, got {len(ood)}"

    def test_manifest_normal_count(self, manifest):
        normal = [m for m in manifest if not m["is_ood"]]
        assert len(normal) == 3000, f"Expected 3000 normal, got {len(normal)}"

    def test_manifest_ood_balance(self, manifest):
        ood_cats = Counter(m["category"] for m in manifest if m["is_ood"])
        for cat in OOD_CATEGORIES:
            assert ood_cats[cat] == 400, f"{cat}: expected 400, got {ood_cats.get(cat, 0)}"

    def test_manifest_splits(self, manifest):
        splits = set(m["split"] for m in manifest)
        assert splits == {"train", "val", "test"}, f"Splits: {splits}"

    def test_manifest_split_distribution(self, manifest):
        split_counts = Counter(m["split"] for m in manifest)
        # All OOD in test, normal 70/15/15
        assert split_counts["train"] == 2100
        assert split_counts["val"] == 450
        assert split_counts["test"] == 1650  # 1200 OOD + 450 normal

    def test_ood_test_frames_exact(self, manifest):
        ood_test = [m for m in manifest if m["is_ood"] and m["split"] == "test"]
        assert len(ood_test) == 1200, f"Expected 1200 OOD test frames, got {len(ood_test)}"
        test_cats = Counter(m["category"] for m in ood_test)
        for cat in OOD_CATEGORIES:
            assert test_cats[cat] == 400, f"{cat}: expected 400 in test, got {test_cats.get(cat, 0)}"

    def test_manifest_entry_schema(self, manifest):
        required_keys = {"frame_id", "path", "category", "split", "is_ood", "source"}
        for entry in manifest[:10]:
            assert required_keys.issubset(entry.keys()), f"Missing keys: {required_keys - entry.keys()}"

    def test_frame_files_exist(self, manifest):
        # Check a sample of frame files exist
        for entry in manifest[:20]:
            path = Path(entry["path"])
            assert path.exists(), f"Frame file missing: {path}"


class TestDevice:
    def test_get_device(self):
        device = get_device()
        assert device.type in ("mps", "cuda", "cpu")

    def test_mps_available(self):
        """On M4 Mac, MPS should be available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")
        device = get_device()
        assert device.type == "mps"


class TestDataset:
    def test_dataset_loads(self, manifest):
        transform = get_clip_preprocess()
        ds = DrivingFrameDataset(MANIFEST_PATH, split="train", transform=transform)
        assert len(ds) > 0

    def test_dataset_item_shape(self, manifest):
        transform = get_clip_preprocess()
        ds = DrivingFrameDataset(MANIFEST_PATH, split="train", transform=transform)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)
        assert item["label"] in range(4)
        assert isinstance(item["category"], str)
        assert isinstance(item["is_ood"], bool)

    def test_dataloader_batch(self, manifest):
        transform = get_clip_preprocess()
        ds = DrivingFrameDataset(MANIFEST_PATH, split="train", transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (16, 3, 224, 224)
        assert batch["label"].shape == (16,)

    def test_augmentations_shape(self, manifest):
        transform = get_train_augmentations()
        ds = DrivingFrameDataset(MANIFEST_PATH, split="train", transform=transform)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)


class TestSplits:
    def test_assign_splits_balance(self):
        entries = [
            {"category": "normal", "frame_id": str(i)}
            for i in range(100)
        ]
        result = assign_splits(entries, train_ratio=0.7, val_ratio=0.15)
        split_counts = Counter(e["split"] for e in result)
        assert split_counts["train"] == 70
        assert split_counts["val"] == 15
        assert split_counts["test"] == 15

    def test_assign_splits_deterministic(self):
        entries = [{"category": "normal", "frame_id": str(i)} for i in range(50)]
        r1 = assign_splits(entries.copy(), seed=42)
        r2 = assign_splits(entries.copy(), seed=42)
        splits1 = [e["split"] for e in r1]
        splits2 = [e["split"] for e in r2]
        assert splits1 == splits2


class TestCategoryMapping:
    def test_category_to_idx(self):
        assert len(CATEGORY_TO_IDX) == 4
        assert set(CATEGORY_TO_IDX.keys()) == {
            "construction_zone", "emergency_vehicle", "lane_blockage", "normal"
        }

    def test_idx_to_category_inverse(self):
        for cat, idx in CATEGORY_TO_IDX.items():
            assert IDX_TO_CATEGORY[idx] == cat
