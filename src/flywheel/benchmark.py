import time
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.flywheel.scorer import FrameScorer
from src.flywheel.router import FrameRouter
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Industry averages for manual annotation time
MANUAL_ANNOTATION_SECONDS = 30.0  # per frame, bounding box + class label
ASSISTED_ANNOTATION_SECONDS = 12.0  # per frame, when model provides suggestion


def run_flywheel_pipeline(
    scorer: FrameScorer,
    dataloader: DataLoader,
    output_dir: str | Path,
) -> dict:
    """Run the full flywheel scoring + routing pipeline.

    Returns routing stats and timing info.
    """
    router = FrameRouter(output_dir)
    total_inference_time = 0.0

    for batch in tqdm(dataloader, desc="Flywheel scoring"):
        images = batch["image"]
        frame_ids = batch["frame_id"]

        start = time.time()
        scores = scorer.score_batch(images.to(scorer.clip.device))
        total_inference_time += time.time() - start

        for frame_id, score in zip(frame_ids, scores):
            router.route(frame_id, score)

    stats = router.get_stats()
    stats["total_inference_time_s"] = total_inference_time
    return stats


def compute_speedup(routing_stats: dict) -> dict:
    """Compute curation speedup vs. manual baseline.

    Three-zone routing:
      auto_label     — confident prediction, no human time
      active_learning — uncertain, human reviews with model suggestion (12 s)
      low_confidence  — routine/normal, no annotation value, skipped
    """
    total = routing_stats["total_frames"]
    auto_labeled = routing_stats["auto_labeled"]
    active_learning = routing_stats["active_learning"]
    low_confidence = routing_stats.get("low_confidence", 0)
    inference_time = routing_stats.get("total_inference_time_s", total * 0.05)

    # Manual baseline: every frame annotated from scratch
    manual_time = total * MANUAL_ANNOTATION_SECONDS

    # Flywheel: inference + only active_learning frames need human review (assisted)
    flywheel_time = inference_time + (active_learning * ASSISTED_ANNOTATION_SECONDS)

    speedup = manual_time / max(flywheel_time, 1.0)

    return {
        "total_frames": total,
        "auto_labeled": auto_labeled,
        "active_learning": active_learning,
        "low_confidence": low_confidence,
        "auto_label_fraction": routing_stats["auto_label_fraction"],
        "active_learning_fraction": routing_stats["active_learning_fraction"],
        "low_confidence_fraction": routing_stats.get("low_confidence_fraction", 0),
        "manual_time_s": manual_time,
        "flywheel_time_s": flywheel_time,
        "inference_time_s": inference_time,
        "curation_speedup": round(speedup, 2),
    }
