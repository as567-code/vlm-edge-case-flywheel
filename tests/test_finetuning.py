"""Tests for Stage 3: fine-tuning results."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def finetuned_metrics():
    path = Path("results/finetuned_metrics.json")
    if not path.exists():
        pytest.skip("Fine-tuned model not yet assessed. Run scripts/train.py + scripts/evaluate.py")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def baseline_metrics():
    path = Path("results/baseline_metrics.json")
    if not path.exists():
        pytest.skip("Baseline not yet run.")
    with open(path) as f:
        return json.load(f)


class TestFinetuning:
    def test_checkpoint_exists(self):
        assert Path("checkpoints/best_model.pt").exists(), "No checkpoint found"

    def test_training_log_exists(self):
        assert Path("results/training_log.csv").exists(), "No training log found"

    def test_m2_accuracy(self, finetuned_metrics):
        acc = finetuned_metrics["accuracy"]
        assert acc >= 0.83, f"M2 FAIL: accuracy {acc:.1%} < 83%"

    def test_m6_gain(self, finetuned_metrics, baseline_metrics):
        gain = finetuned_metrics["accuracy"] - baseline_metrics["accuracy"]
        assert gain >= 0.10, f"M6 FAIL: gain {gain:.1%} < 10pp"

    def test_per_class_f1(self, finetuned_metrics):
        for cat, metrics in finetuned_metrics["per_class"].items():
            if cat != "normal":
                assert metrics["f1"] >= 0.75, f"{cat} F1={metrics['f1']:.3f} < 0.75"
