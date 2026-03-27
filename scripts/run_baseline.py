#!/usr/bin/env python3
"""Stage 2: Run vanilla CLIP zero-shot baseline on the OOD test set."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import get_clip_preprocess
from src.data.dataset import DrivingFrameDataset
from src.model.clip_wrapper import CLIPWrapper
from src.model.evaluator import run_zero_shot_classification, print_metrics
from src.model.text_anchors import get_scene_prompts
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger

logger = setup_logger("baseline")


def plot_confusion_matrix(metrics: dict, output_path: str) -> None:
    """Save confusion matrix as PNG."""
    cm = np.array(metrics["confusion_matrix"])
    cats = [c[:12] for c in metrics["categories"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(cats)),
        yticks=range(len(cats)),
        xticklabels=cats,
        yticklabels=cats,
        ylabel="True label",
        xlabel="Predicted label",
        title="Baseline Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run vanilla CLIP zero-shot baseline")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--output", type=str, default="results/baseline_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ood-only", action="store_true", help="Only score OOD frames")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading vanilla CLIP ViT-B/32...")
    clip = CLIPWrapper(device=device)

    # Load dataset
    transform = get_clip_preprocess()
    dataset = DrivingFrameDataset(args.manifest, split=args.split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Loaded {len(dataset)} frames from {args.split} split")

    # Load prompts
    prompts = get_scene_prompts("configs/text_anchors.yaml")

    # Run zero-shot classification
    logger.info("Running zero-shot classification...")
    metrics = run_zero_shot_classification(clip, dataloader, prompts=prompts, ood_only=args.ood_only)

    # Print and save
    print_metrics(metrics)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output)
    logger.info(f"Metrics saved to {args.output}")

    # Save confusion matrix
    cm_path = str(Path(args.output).parent / "baseline_confusion_matrix.png")
    plot_confusion_matrix(metrics, cm_path)

    # Verify baseline range
    acc = metrics["accuracy"]
    logger.info(f"\nBaseline accuracy: {acc:.1%}")
    if 0.55 <= acc <= 0.75:
        logger.info("Baseline accuracy in expected range [55%-75%]")
    elif acc > 0.75:
        logger.warning(f"Baseline accuracy {acc:.1%} is high. Fine-tuning delta may be small.")
    else:
        logger.warning(f"Baseline accuracy {acc:.1%} is below 55%. Check data quality.")


if __name__ == "__main__":
    main()
