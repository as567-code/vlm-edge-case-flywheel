"""Tests for Stage 4-5: flywheel, annotation reduction, and benchmark."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def annotation_metrics():
    path = Path("results/annotation_metrics.json")
    if not path.exists():
        pytest.skip("Annotation reduction not yet measured.")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def flywheel_benchmark():
    path = Path("results/flywheel_benchmark.json")
    if not path.exists():
        pytest.skip("Flywheel benchmark not yet run.")
    with open(path) as f:
        return json.load(f)


class TestAnnotationReduction:
    def test_m1_reduction(self, annotation_metrics):
        pct = annotation_metrics["auto_labeled_fraction"]
        assert pct >= 0.62, f"M1 FAIL: reduction {pct:.1%} < 62%"

    def test_auto_label_accuracy(self, annotation_metrics):
        acc = annotation_metrics["auto_label_accuracy"]
        assert acc >= 0.93, f"Auto-label accuracy {acc:.1%} < 93%"


class TestFlywheelBenchmark:
    def test_m3_routing(self, flywheel_benchmark):
        al_pct = flywheel_benchmark["active_learning_fraction"]
        assert al_pct >= 0.40, f"M3 FAIL: AL routing {al_pct:.1%} < 40%"

    def test_m4_speedup(self, flywheel_benchmark):
        speedup = flywheel_benchmark["curation_speedup"]
        assert speedup >= 3.0, f"M4 FAIL: speedup {speedup:.1f}x < 3.0x"

    def test_all_frames_processed(self, flywheel_benchmark):
        total = flywheel_benchmark["total_frames"]
        assert total >= 4200, f"Only {total} frames processed"
