import json
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class FrameRouter:
    """Routes scored frames to auto-label pool or active learning queue."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.auto_label_dir = self.output_dir / "auto_labeled"
        self.active_learning_dir = self.output_dir / "active_learning"
        self.routing_log_path = self.output_dir / "routing_log.jsonl"

        self.auto_label_dir.mkdir(parents=True, exist_ok=True)
        self.active_learning_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {"auto_label": 0, "active_learning": 0, "low_confidence": 0, "total": 0}

    def route(self, frame_id: str, score_result: dict) -> str:
        """Route a single frame based on its score result.

        Returns the route taken ('auto_label' or 'active_learning').
        """
        route = score_result["route"]
        self.stats[route] += 1
        self.stats["total"] += 1

        # Log routing decision
        log_entry = {
            "frame_id": frame_id,
            "predicted_class": score_result["predicted_class"],
            "confidence": score_result["confidence"],
            "route": route,
        }
        with open(self.routing_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return route

    def get_stats(self) -> dict:
        """Return routing statistics."""
        total = max(self.stats["total"], 1)
        return {
            "total_frames": self.stats["total"],
            "auto_labeled": self.stats["auto_label"],
            "active_learning": self.stats["active_learning"],
            "low_confidence": self.stats["low_confidence"],
            "auto_label_fraction": self.stats["auto_label"] / total,
            "active_learning_fraction": self.stats["active_learning"] / total,
            "low_confidence_fraction": self.stats["low_confidence"] / total,
        }
