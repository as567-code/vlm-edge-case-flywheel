#!/usr/bin/env python3
"""Stage 5: Run the full flywheel scoring + routing pipeline."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import get_clip_preprocess
from src.data.dataset import DrivingFrameDataset
from src.model.clip_wrapper import CLIPWrapper
from src.model.text_anchors import compute_text_anchors, get_scene_prompts
from src.flywheel.scorer import FrameScorer
from src.flywheel.benchmark import run_flywheel_pipeline
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("flywheel")


def main():
    parser = argparse.ArgumentParser(description="Run flywheel pipeline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data", type=str, default="data/manifest.json")
    parser.add_argument("--config", type=str, default="configs/flywheel.yaml")
    parser.add_argument("--output", type=str, default="output")
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

    # Setup scorer
    scoring_cfg = flywheel_config["scoring"]
    scorer = FrameScorer(
        clip, anchors, categories,
        high_threshold=scoring_cfg["high_confidence_threshold"],
        low_threshold=scoring_cfg["low_confidence_threshold"],
    )

    # Load ALL data (not just one split)
    transform = get_clip_preprocess()
    for split in ["train", "val", "test"]:
        dataset = DrivingFrameDataset(args.data, split=split, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        logger.info(f"Processing {split} split: {len(dataset)} frames")

        stats = run_flywheel_pipeline(scorer, dataloader, args.output)
        logger.info(
            f"  {split}: auto_label={stats['auto_labeled']}, "
            f"active_learning={stats['active_learning']}, "
            f"inference_time={stats.get('total_inference_time_s', 0):.1f}s"
        )

    logger.info(f"\nRouting log: {args.output}/routing_log.jsonl")
    logger.info("Flywheel pipeline complete.")


if __name__ == "__main__":
    main()
