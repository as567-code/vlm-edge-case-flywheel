"""Tests for Stage 6: all metrics pass together."""

import json
from pathlib import Path

import pytest


class TestAllMetrics:
    def test_manifest_m5(self):
        """M5: At least 3 edge-case categories in manifest."""
        path = Path("data/manifest.json")
        if not path.exists():
            pytest.skip("Manifest not found")
        with open(path) as f:
            manifest = json.load(f)
        cats = {m["category"] for m in manifest if m["is_ood"]}
        assert len(cats) >= 3, f"M5 FAIL: only {len(cats)} OOD categories"

    def test_text_anchors_exist(self):
        """Text anchors config has >= 8 prompts per class."""
        import yaml
        path = Path("configs/text_anchors.yaml")
        assert path.exists()
        with open(path) as f:
            cfg = yaml.safe_load(f)
        for cat, prompts in cfg["prompts"].items():
            assert len(prompts) >= 8, f"{cat} has only {len(prompts)} prompts"
