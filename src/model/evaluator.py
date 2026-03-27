import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.clip_wrapper import CLIPWrapper
from src.model.text_anchors import compute_text_anchors, get_scene_prompts
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_zero_shot_classification(
    clip_model: CLIPWrapper,
    dataloader: DataLoader,
    prompts: dict[str, list[str]] | None = None,
    ood_only: bool = False,
) -> dict:
    """Run zero-shot classification and compute metrics.

    Args:
        clip_model: CLIPWrapper instance
        dataloader: DataLoader yielding batches with 'image', 'label', 'is_ood' keys
        prompts: Optional custom prompts dict; defaults to built-in SCENE_PROMPTS
        ood_only: If True, only score OOD frames

    Returns:
        Dict with accuracy, per-class metrics, confusion matrix, similarity distributions
    """
    if prompts is None:
        prompts = get_scene_prompts()

    anchors, categories = compute_text_anchors(clip_model, prompts)
    num_classes = len(categories)

    all_preds = []
    all_labels = []
    all_sims = []

    clip_model.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying", leave=False):
            images = batch["image"].to(clip_model.device)
            labels = batch["label"]
            is_ood = batch["is_ood"]

            image_features = clip_model.encode_image(images)
            similarities = (image_features @ anchors.T).cpu()

            for i in range(labels.size(0)):
                if ood_only and not is_ood[i]:
                    continue
                all_preds.append(similarities[i].argmax().item())
                all_labels.append(labels[i].item())
                all_sims.append(similarities[i].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sims = np.array(all_sims)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    per_class = {}
    for idx in range(num_classes):
        cat = categories[idx]
        tp = ((all_preds == idx) & (all_labels == idx)).sum()
        fp = ((all_preds == idx) & (all_labels != idx)).sum()
        fn = ((all_preds != idx) & (all_labels == idx)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[cat] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((all_labels == idx).sum()),
        }

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    # Similarity distributions per true class
    sim_distributions = {}
    for idx in range(num_classes):
        cat = categories[idx]
        mask = all_labels == idx
        if mask.sum() > 0:
            sim_distributions[cat] = {
                "mean": float(all_sims[mask].mean()),
                "std": float(all_sims[mask].std()),
                "max_correct": float(all_sims[mask, idx].mean()),
            }

    return {
        "accuracy": float(accuracy),
        "num_frames": len(all_labels),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "categories": categories,
        "similarity_distributions": sim_distributions,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print classification metrics."""
    print(f"\nOverall Accuracy: {metrics['accuracy']:.1%} ({metrics['num_frames']} frames)")
    print("\nPer-class metrics:")
    print(f"  {'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*65}")
    for cat, m in metrics["per_class"].items():
        print(f"  {cat:<25} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10d}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    cats = metrics["categories"]
    header = "  " + " " * 20 + "  ".join(f"{c[:8]:>8}" for c in cats)
    print(header)
    for i, cat in enumerate(cats):
        row = metrics["confusion_matrix"][i]
        print(f"  {cat:<20}" + "  ".join(f"{v:>8d}" for v in row))
