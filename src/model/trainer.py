import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.clip_wrapper import CLIPWrapper
from src.model.text_anchors import get_scene_prompts
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def info_nce_loss(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute InfoNCE contrastive loss with in-batch negatives."""
    # image_features: (B, D), text_features: (B, D) -- already normalized
    logits = (image_features @ text_features.T) / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


class ContrastiveTrainer:
    """Self-supervised contrastive fine-tuning loop for CLIP."""

    def __init__(
        self,
        clip_model: CLIPWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ):
        self.clip = clip_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = clip_model.device

        # Load prompts for text augmentation
        prompts_config = config.get("text_anchors_config")
        self.prompts = get_scene_prompts(prompts_config)
        self.categories = sorted(self.prompts.keys())

        # Pre-encode all text prompts
        self._precompute_text_embeddings()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            clip_model.trainable_params(),
            lr=config.get("lr", 1e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        total_steps = len(train_loader) * config.get("epochs", 20)
        warmup = config.get("warmup_steps", 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup
        )
        self.warmup_steps = warmup
        self.global_step = 0
        self.temperature = config.get("temperature", 0.07)

    def _precompute_text_embeddings(self):
        """Pre-encode all text prompts per category."""
        self.text_embeds_by_category = {}
        for cat in self.categories:
            embeds = self.clip.encode_text(self.prompts[cat])
            self.text_embeds_by_category[cat] = embeds

    def _sample_text_embedding(self, labels: torch.Tensor) -> torch.Tensor:
        """For each image in batch, sample one text embedding from its category."""
        from src.data.dataset import IDX_TO_CATEGORY
        import random

        batch_text = []
        for label in labels:
            cat = IDX_TO_CATEGORY[label.item()]
            embeds = self.text_embeds_by_category[cat]
            idx = random.randint(0, embeds.size(0) - 1)
            batch_text.append(embeds[idx])
        return torch.stack(batch_text)

    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.clip.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]

            # Encode images
            image_features = self.clip.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Sample text embeddings (one per image, from matching category)
            text_features = self._sample_text_embedding(labels)

            loss = info_nce_loss(image_features, text_features, self.temperature)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.clip.trainable_params(), max_norm=1.0)
            self.optimizer.step()

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def run_validation(self) -> float:
        """Run validation and return accuracy."""
        from src.model.text_anchors import compute_text_anchors

        self.clip.model.eval()
        anchors, categories = compute_text_anchors(self.clip, self.prompts)

        correct = 0
        total = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            image_features = self.clip.encode_image(images)
            similarities = image_features @ anchors.T
            preds = similarities.argmax(dim=-1).cpu()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return correct / max(total, 1)

    def train(self) -> dict:
        """Full training loop with early stopping."""
        epochs = self.config.get("epochs", 20)
        patience = self.config.get("patience", 5)
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        log_path = Path(self.config.get("log_path", "results/training_log.csv"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0
        history = []

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_accuracy", "lr", "time_s"])

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self.train_epoch()
            val_acc = self.run_validation()
            elapsed = time.time() - start

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{epochs} -- loss: {train_loss:.4f}, val_acc: {val_acc:.1%}, lr: {lr:.2e}, time: {elapsed:.1f}s"
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{val_acc:.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])

            history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.clip.model.state_dict(), checkpoint_dir / "best_model.pt")
                logger.info(f"  New best model saved (val_acc={val_acc:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                    break

        return {
            "best_val_accuracy": best_val_acc,
            "epochs_trained": len(history),
            "history": history,
        }

