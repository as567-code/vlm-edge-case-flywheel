import torch
from PIL import Image

from src.model.clip_wrapper import CLIPWrapper


class AutoLabeler:
    """Confidence-based auto-labeling for the data flywheel."""

    def __init__(
        self,
        clip_model: CLIPWrapper,
        text_anchors: torch.Tensor,
        categories: list[str],
        confidence_threshold: float = 0.85,
    ):
        self.clip = clip_model
        self.text_anchors = text_anchors
        self.categories = categories
        self.confidence_threshold = confidence_threshold

    def auto_label(self, image: Image.Image | torch.Tensor) -> tuple[str, str, float]:
        """Attempt to auto-label an image.

        Returns:
            (predicted_class, status, confidence)
            status is 'auto_labeled' or 'needs_review'
        """
        embedding = self.clip.encode_image(image)
        similarities = (embedding @ self.text_anchors.T).squeeze(0)
        max_sim = float(similarities.max())
        predicted_idx = int(similarities.argmax())
        predicted_class = self.categories[predicted_idx]

        if max_sim >= self.confidence_threshold:
            return predicted_class, "auto_labeled", max_sim
        else:
            return predicted_class, "needs_review", max_sim

    def measure_reduction(
        self, predictions: list[dict]
    ) -> dict:
        """Measure annotation effort reduction from a list of prediction results.

        Each entry should have 'status' ('auto_labeled' or 'needs_review')
        and optionally 'correct' (bool) for accuracy measurement.
        """
        total = len(predictions)
        auto_labeled = [p for p in predictions if p["status"] == "auto_labeled"]
        needs_review = [p for p in predictions if p["status"] == "needs_review"]

        auto_correct = sum(1 for p in auto_labeled if p.get("correct", False))
        auto_accuracy = auto_correct / max(len(auto_labeled), 1)

        return {
            "total_frames": total,
            "auto_labeled_count": len(auto_labeled),
            "needs_review_count": len(needs_review),
            "auto_labeled_fraction": len(auto_labeled) / max(total, 1),
            "auto_label_accuracy": auto_accuracy,
            "annotation_reduction": len(auto_labeled) / max(total, 1),
        }

