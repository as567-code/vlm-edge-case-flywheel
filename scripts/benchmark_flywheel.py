#!/usr/bin/env python3
"""Stage 5: Benchmark flywheel throughput and compute curation speedup."""

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
from src.flywheel.benchmark import run_flywheel_pipeline, compute_speedup
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger

logger = setup_logger("benchmark")


def main():
    parser = argparse.ArgumentParser(description="Benchmark flywheel throughput")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--config", type=str, default="configs/flywheel.yaml")
    parser.add_argument("--output", type=str, default="results/flywheel_benchmark.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    flywheel_config = load_config(args.config)

    # Load model
    clip = CLIPWrapper(device=device)
    if Path(args.checkpoint).exists():
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        clip.model.load_state_dict(state_dict)
    clip.model.requires_grad_(False)

    # Compute text anchors
    prompts = get_scene_prompts("configs/text_anchors.yaml")
    anchors, categories = compute_text_anchors(clip, prompts)

    scoring_cfg = flywheel_config["scoring"]
    scorer = FrameScorer(
        clip, anchors, categories,
        high_threshold=scoring_cfg["high_confidence_threshold"],
        low_threshold=scoring_cfg["low_confidence_threshold"],
    )

    # Process all data
    transform = get_clip_preprocess()
    all_stats = {
        "auto_label": 0, "active_learning": 0, "low_confidence": 0,
        "total": 0, "total_inference_time_s": 0,
    }

    for split in ["train", "val", "test"]:
        dataset = DrivingFrameDataset(args.manifest, split=split, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        stats = run_flywheel_pipeline(scorer, dataloader, "output")
        all_stats["auto_label"] += stats["auto_labeled"]
        all_stats["active_learning"] += stats["active_learning"]
        all_stats["low_confidence"] += stats.get("low_confidence", 0)
        all_stats["total"] += stats["total_frames"]
        all_stats["total_inference_time_s"] += stats.get("total_inference_time_s", 0)

    # Normalize fractions
    total = max(all_stats["total"], 1)
    all_stats["auto_label_fraction"] = all_stats["auto_label"] / total
    all_stats["active_learning_fraction"] = all_stats["active_learning"] / total
    all_stats["low_confidence_fraction"] = all_stats["low_confidence"] / total
    all_stats["auto_labeled"] = all_stats["auto_label"]
    all_stats["total_frames"] = all_stats["total"]

    # Compute speedup
    benchmark = compute_speedup(all_stats)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(benchmark, args.output)

    logger.info("\n=== Flywheel Benchmark Results ===")
    logger.info(f"Total frames: {benchmark['total_frames']}")
    logger.info(f"Auto-labeled:      {benchmark['auto_labeled']:>5} ({benchmark['auto_label_fraction']:.1%})")
    logger.info(f"Active learning:   {benchmark['active_learning']:>5} ({benchmark['active_learning_fraction']:.1%})")
    logger.info(f"Low confidence:    {benchmark['low_confidence']:>5} ({benchmark['low_confidence_fraction']:.1%})")
    logger.info(f"Manual time: {benchmark['manual_time_s']:.0f}s")
    logger.info(f"Flywheel time: {benchmark['flywheel_time_s']:.0f}s")
    logger.info(f"Curation speedup: {benchmark['curation_speedup']:.1f}x")


if __name__ == "__main__":
    main()
