#!/usr/bin/env python3
"""Model assessment on test set. Used for both baseline and fine-tuned models."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import get_clip_preprocess
from src.data.dataset import DrivingFrameDataset
from src.model.clip_wrapper import CLIPWrapper
from src.model.evaluator import run_zero_shot_classification, print_metrics
from src.model.text_anchors import get_scene_prompts
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger

logger = setup_logger("assess")


def main():
    parser = argparse.ArgumentParser(description="Run model assessment on test set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--output", type=str, default="results/finetuned_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ood-only", action="store_true")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    clip = CLIPWrapper(device=device)
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        clip.model.load_state_dict(state_dict)
    clip.model.requires_grad_(False)

    # Load dataset
    transform = get_clip_preprocess()
    dataset = DrivingFrameDataset(args.manifest, split=args.split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Loaded {len(dataset)} frames from {args.split} split")

    # Load prompts
    prompts = get_scene_prompts("configs/text_anchors.yaml")

    # Run classification
    logger.info("Running classification...")
    metrics = run_zero_shot_classification(clip, dataloader, prompts=prompts, ood_only=args.ood_only)

    print_metrics(metrics)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output)
    logger.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
