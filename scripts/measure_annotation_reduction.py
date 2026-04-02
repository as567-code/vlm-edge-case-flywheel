#!/usr/bin/env python3
"""Stage 4: Measure annotation effort reduction (M1 verification)."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import get_clip_preprocess
from src.data.dataset import DrivingFrameDataset
from src.model.clip_wrapper import CLIPWrapper
from src.model.text_anchors import compute_text_anchors, get_scene_prompts
from src.flywheel.auto_labeler import AutoLabeler
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger

logger = setup_logger("annotation_reduction")


def main():
    parser = argparse.ArgumentParser(description="Measure annotation effort reduction")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--config", type=str, default="configs/flywheel.yaml")
    parser.add_argument("--output", type=str, default="results/annotation_metrics.json")
    parser.add_argument("--split", type=str, default="val", help="Split to measure on")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    flywheel_config = load_config(args.config)

    # Load model
    clip = CLIPWrapper(device=device)
    if Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        clip.model.load_state_dict(state_dict)
    clip.model.requires_grad_(False)

    # Compute text anchors
    prompts = get_scene_prompts("configs/text_anchors.yaml")
    anchors, categories = compute_text_anchors(clip, prompts)

    # Setup auto-labeler
    threshold = flywheel_config["auto_labeling"]["confidence_threshold"]
    labeler = AutoLabeler(clip, anchors, categories, confidence_threshold=threshold)

    # Load dataset
    transform = get_clip_preprocess()
    dataset = DrivingFrameDataset(args.manifest, split=args.split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f"Measuring on {len(dataset)} frames from {args.split} split")

    # Run auto-labeling
    predictions = []
    for batch in tqdm(dataloader, desc="Auto-labeling"):
        image = batch["image"].squeeze(0)
        true_label = batch["category"][0]

        pred_class, status, confidence = labeler.auto_label(image)
        predictions.append({
            "true_class": true_label,
            "predicted_class": pred_class,
            "status": status,
            "confidence": confidence,
            "correct": pred_class == true_label,
        })

    # Compute reduction metrics
    reduction_metrics = labeler.measure_reduction(predictions)

    # Sweep thresholds to find optimal
    logger.info("\nThreshold sweep:")
    best_threshold = threshold
    best_score = 0
    for t in [0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32]:
        auto = [p for p in predictions if p["confidence"] >= t]
        [p for p in predictions if p["confidence"] < t]
        auto_acc = sum(1 for p in auto if p["correct"]) / max(len(auto), 1)
        coverage = len(auto) / max(len(predictions), 1)
        logger.info(f"  threshold={t:.2f}: coverage={coverage:.1%}, auto_acc={auto_acc:.1%}")

        # Score: maximize coverage while keeping accuracy >= 93%
        if auto_acc >= 0.93 and coverage > best_score:
            best_score = coverage
            best_threshold = t

    reduction_metrics["optimal_threshold"] = best_threshold
    reduction_metrics["threshold_used"] = threshold

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(reduction_metrics, args.output)

    logger.info("\n=== Annotation Reduction Results ===")
    logger.info(f"Auto-labeled: {reduction_metrics['auto_labeled_fraction']:.1%}")
    logger.info(f"Auto-label accuracy: {reduction_metrics['auto_label_accuracy']:.1%}")
    logger.info(f"Annotation reduction: {reduction_metrics['annotation_reduction']:.1%}")
    logger.info(f"Optimal threshold: {best_threshold}")


if __name__ == "__main__":
    main()

