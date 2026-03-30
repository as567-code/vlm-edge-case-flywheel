#!/usr/bin/env python3
"""Stage 3: Self-supervised contrastive fine-tuning of CLIP."""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import get_train_augmentations, get_clip_preprocess
from src.data.dataset import DrivingFrameDataset
from src.model.clip_wrapper import CLIPWrapper
from src.model.trainer import ContrastiveTrainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger

logger = setup_logger("train")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with contrastive learning")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    model_cfg = config["model"]
    clip = CLIPWrapper(
        model_name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        device=device,
    )
    clip.unfreeze_layers(num_blocks=model_cfg.get("unfreeze_blocks", 2))

    trainable = sum(p.numel() for p in clip.trainable_params())
    total = sum(p.numel() for p in clip.model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # Load data
    train_cfg = config["training"]
    data_cfg = config["data"]
    manifest = data_cfg["manifest_path"]

    # Self-supervised contrastive learning uses ALL data for training
    # (no classification labels are used -- only text-image pairs).
    # Validation uses the held-out normal frames for loss monitoring.
    train_dataset = DrivingFrameDataset(
        manifest, split="all",
        transform=get_train_augmentations(data_cfg.get("image_size", 224)),
    )
    val_dataset = DrivingFrameDataset(
        manifest, split="test",
        transform=get_clip_preprocess(data_cfg.get("image_size", 224)),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=0,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Build trainer config
    trainer_config = {
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_steps": train_cfg["warmup_steps"],
        "epochs": train_cfg["epochs"],
        "temperature": train_cfg["temperature"],
        "patience": train_cfg["patience"],
        "checkpoint_dir": config["checkpoint_dir"],
        "log_path": config["log_path"],
        "text_anchors_config": config.get("text_anchors_config"),
    }

    # Train
    trainer = ContrastiveTrainer(clip, train_loader, val_loader, trainer_config)
    logger.info("Starting contrastive fine-tuning...")
    result = trainer.train()

    # Save training summary
    save_json(result, "results/training_summary.json")
    logger.info("\nTraining complete!")
    logger.info(f"Best val accuracy: {result['best_val_accuracy']:.1%}")
    logger.info(f"Epochs trained: {result['epochs_trained']}")


if __name__ == "__main__":
    main()
