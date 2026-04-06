import torch
from PIL import Image

from src.model.clip_wrapper import CLIPWrapper


class FrameScorer:
    """Scores driving frames by cosine similarity to text anchors."""

    def __init__(
        self,
        clip_model: CLIPWrapper,
        text_anchors: torch.Tensor,
        categories: list[str],
        high_threshold: float = 0.85,
        low_threshold: float = 0.60,
    ):
        self.clip = clip_model
        self.text_anchors = text_anchors
        self.categories = categories
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def score(self, image: Image.Image | torch.Tensor) -> dict:
        """Score a single image and determine routing.

        Returns dict with predicted_class, confidence, route, similarities.
        """
        embedding = self.clip.encode_image(image)
        similarities = (embedding @ self.text_anchors.T).squeeze(0)
        max_sim = float(similarities.max())
        predicted_idx = int(similarities.argmax())

        route = self._route(max_sim)

        return {
            "predicted_class": self.categories[predicted_idx],
            "predicted_idx": predicted_idx,
            "confidence": max_sim,
            "route": route,
            "similarities": {
                cat: float(similarities[i]) for i, cat in enumerate(self.categories)
            },
        }

    def score_batch(self, images: torch.Tensor) -> list[dict]:
        """Score a batch of images."""
        embeddings = self.clip.encode_image(images)
        similarities = embeddings @ self.text_anchors.T  # (B, num_classes)

        results = []
        for i in range(similarities.size(0)):
            sims = similarities[i]
            max_sim = float(sims.max())
            predicted_idx = int(sims.argmax())

            results.append({
                "predicted_class": self.categories[predicted_idx],
                "predicted_idx": predicted_idx,
                "confidence": max_sim,
                "route": self._route(max_sim),
                "similarities": {
                    cat: float(sims[j]) for j, cat in enumerate(self.categories)
                },
            })

        return results

    def _route(self, max_sim: float) -> str:
        """Three-zone routing based on confidence thresholds.

        >= high_threshold  → auto_label       (confident, no human review)
        >= low_threshold   → active_learning  (uncertain, human reviews with model assist)
        <  low_threshold   → low_confidence   (routine/normal, no annotation value)
        """
        if max_sim >= self.high_threshold:
            return "auto_label"
        if max_sim >= self.low_threshold:
            return "active_learning"
        return "low_confidence"
