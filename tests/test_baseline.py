"""Tests for Stage 2: vanilla CLIP baseline metrics."""

import json
from pathlib import Path

import pytest


BASELINE_PATH = Path("results/baseline_metrics.json")


@pytest.fixture
def baseline_metrics():
    if not BASELINE_PATH.exists():
        pytest.skip("Baseline not yet run. Execute scripts/run_baseline.py first.")
    with open(BASELINE_PATH) as f:
        return json.load(f)


class TestBaseline:
    def test_baseline_exists(self):
        assert BASELINE_PATH.exists(), "results/baseline_metrics.json does not exist"

    def test_baseline_accuracy_range(self, baseline_metrics):
        acc = baseline_metrics["accuracy"]
        assert 0.30 <= acc <= 0.90, f"Baseline accuracy {acc:.1%} outside expected range"

    def test_per_class_f1(self, baseline_metrics):
        ood_cats = ["construction_zone", "emergency_vehicle", "lane_blockage"]
        for cat in ood_cats:
            metrics = baseline_metrics["per_class"][cat]
            assert metrics["f1"] >= 0.40, f"{cat} F1={metrics['f1']:.3f} < 0.40"

    def test_confusion_matrix_exists(self, baseline_metrics):
        assert "confusion_matrix" in baseline_metrics
        cm = baseline_metrics["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_confusion_matrix_png_exists(self):
        assert Path("results/baseline_confusion_matrix.png").exists()
