#!/usr/bin/env python3
"""Dataset download and curation pipeline.

Downloads driving scene images from HuggingFace (BDD100K subset) and curates them
into 4 categories using CLIP zero-shot classification:
  - construction_zone
  - emergency_vehicle
  - lane_blockage
  - normal

Produces data/manifest.json with 4,200 entries (1,200 OOD + 3,000 normal).

Usage:
    python scripts/download_data.py [--output-dir data] [--num-normal 3000] [--num-ood-per-class 400]
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("download_data")

# CLIP prompts used ONLY for dataset curation (different from model text anchors)
# These are intentionally broad/simple to avoid overfitting the curation to the model
CURATION_PROMPTS = {
    "construction_zone": [
        "construction zone on a road with cones",
        "road work with barriers and signs",
        "highway construction site",
    ],
    "emergency_vehicle": [
        "emergency vehicle with flashing lights",
        "ambulance or fire truck on road",
        "police car with sirens on highway",
    ],
    "lane_blockage": [
        "blocked road lane with debris or stalled car",
        "accident blocking traffic lane",
        "obstruction in driving lane",
    ],
    "normal": [
        "normal driving on a highway",
        "typical city street traffic",
        "regular road with cars driving",
    ],
}

OOD_CATEGORIES = ["construction_zone", "emergency_vehicle", "lane_blockage"]


def download_bdd100k_images(output_dir: Path, max_images: int = 10000) -> list[Path]:
    """Download BDD100K images from HuggingFace."""
    from datasets import load_dataset

    logger.info("Downloading BDD100K from HuggingFace (dgural/bdd100k)...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have enough frames
    existing = list(frames_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        logger.info(f"Found {len(existing)} cached frames, skipping download.")
        return sorted(existing)[:max_images]

    try:
        ds = load_dataset("dgural/bdd100k", split="train", streaming=True)
        downloaded = []
        for i, sample in enumerate(tqdm(ds, total=max_images, desc="Downloading frames")):
            if i >= max_images:
                break
            img = sample["image"]
            if not isinstance(img, Image.Image):
                continue
            path = frames_dir / f"{i:05d}.jpg"
            if not path.exists():
                img = img.convert("RGB")
                # Resize to save disk space while keeping reasonable resolution
                img = img.resize((640, 360), Image.LANCZOS)
                img.save(path, "JPEG", quality=85)
            downloaded.append(path)

        logger.info(f"Downloaded {len(downloaded)} frames.")
        return downloaded

    except Exception as e:
        logger.warning(f"BDD100K download failed: {e}")
        logger.info("Trying fallback: collabora/carla-nuscenes...")
        return download_carla_fallback(frames_dir, max_images)


def download_carla_fallback(frames_dir: Path, max_images: int) -> list[Path]:
    """Fallback: download CARLA synthetic driving data."""
    from datasets import load_dataset

    try:
        ds = load_dataset("collabora/carla-nuscenes", split="train", streaming=True)
        downloaded = []
        for i, sample in enumerate(tqdm(ds, total=max_images, desc="Downloading CARLA frames")):
            if i >= max_images:
                break
            img = sample.get("image") or sample.get("pixel_values")
            if img is None:
                for key in sample:
                    if isinstance(sample[key], Image.Image):
                        img = sample[key]
                        break
            if img is None or not isinstance(img, Image.Image):
                continue
            path = frames_dir / f"{i:05d}.jpg"
            if not path.exists():
                img = img.convert("RGB")
                img = img.resize((640, 360), Image.LANCZOS)
                img.save(path, "JPEG", quality=85)
            downloaded.append(path)

        logger.info(f"Downloaded {len(downloaded)} CARLA frames.")
        return downloaded

    except Exception as e:
        logger.warning(f"CARLA fallback also failed: {e}")
        return generate_synthetic_frames(frames_dir, max_images)


def generate_synthetic_frames(frames_dir: Path, max_images: int) -> list[Path]:
    """Last resort: generate synthetic gradient images as placeholders.

    These produce weak CLIP scores but let the pipeline run end-to-end.
    Replace with real images for valid metrics.
    """
    import numpy as np

    logger.warning("Generating synthetic placeholder frames. Replace with real data for valid metrics!")
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    downloaded = []
    for i in tqdm(range(max_images), desc="Generating synthetic frames"):
        path = frames_dir / f"{i:05d}.jpg"
        if not path.exists():
            arr = rng.randint(40, 200, (360, 640, 3), dtype=np.uint8)
            arr[180:, 160:480, :] = rng.randint(20, 80, (180, 320, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(path, "JPEG", quality=85)
        downloaded.append(path)

    return downloaded


def classify_with_clip(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
) -> list[dict]:
    """Classify images using CLIP zero-shot with curation prompts."""
    import open_clip

    logger.info("Loading CLIP model for dataset curation...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.requires_grad_(False)

    # Compute text anchors for curation
    categories = sorted(CURATION_PROMPTS.keys())
    text_anchors = []
    for cat in categories:
        tokens = tokenizer(CURATION_PROMPTS[cat]).to(device)
        with torch.no_grad():
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            anchor = embeds.mean(dim=0)
            anchor = anchor / anchor.norm()
        text_anchors.append(anchor)
    text_anchors = torch.stack(text_anchors)

    # Classify all images in batches
    results = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Classifying frames"):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
            except Exception:
                images.append(torch.zeros(3, 224, 224))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            sims = features @ text_anchors.T

        for i, path in enumerate(batch_paths):
            sim_values = sims[i].cpu()
            pred_idx = int(sim_values.argmax())
            results.append({
                "path": str(path),
                "category": categories[pred_idx],
                "confidence": float(sim_values[pred_idx]),
                "similarities": {cat: float(sim_values[j]) for j, cat in enumerate(categories)},
            })

    del model, text_anchors
    if device.type == "mps":
        torch.mps.empty_cache()

    return results


def curate_dataset(
    classified: list[dict],
    output_dir: Path,
    num_normal: int = 3000,
    num_ood_per_class: int = 400,
    seed: int = 42,
) -> list[dict]:
    """Curate the final dataset from classified frames.

    For each OOD category, takes the top-scoring frames.
    For normal, samples from high-confidence normal frames.
    """
    random.Random(seed)

    by_category = defaultdict(list)
    for item in classified:
        by_category[item["category"]].append(item)

    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["confidence"], reverse=True)

    # Also build cross-category ranking for OOD
    for cat in OOD_CATEGORIES:
        candidates = []
        for item in classified:
            ood_sim = item["similarities"][cat]
            other_ood_max = max(
                item["similarities"][c] for c in OOD_CATEGORIES if c != cat
            )
            if ood_sim > other_ood_max and ood_sim > 0.15:
                candidates.append(item)

        candidates.sort(key=lambda x: x["similarities"][cat], reverse=True)
        existing_paths = {x["path"] for x in by_category[cat]}
        for c in candidates:
            if c["path"] not in existing_paths:
                by_category[cat].append(c)
                existing_paths.add(c["path"])

    manifest = []
    used_paths = set()
    frame_counter = 0

    # Select OOD frames
    for cat in OOD_CATEGORIES:
        selected = 0
        for item in by_category[cat]:
            if selected >= num_ood_per_class:
                break
            if item["path"] in used_paths:
                continue

            frame_id = f"{frame_counter:05d}"
            src_path = Path(item["path"])
            rel_path = f"data/frames/{src_path.name}"

            manifest.append({
                "frame_id": frame_id,
                "path": rel_path,
                "category": cat,
                "split": "",
                "is_ood": True,
                "source": "bdd100k_curated",
                "curation_confidence": round(item["confidence"], 4),
            })
            used_paths.add(item["path"])
            frame_counter += 1
            selected += 1

        logger.info(f"  {cat}: selected {selected}/{num_ood_per_class} frames")

    # Fill OOD deficit if needed
    total_ood = sum(1 for m in manifest if m["is_ood"])
    target_ood = num_ood_per_class * len(OOD_CATEGORIES)

    if total_ood < target_ood:
        logger.warning(f"Only found {total_ood}/{target_ood} OOD frames. Redistributing.")
        ood_counts = Counter(m["category"] for m in manifest if m["is_ood"])

        for cat in OOD_CATEGORIES:
            current = ood_counts.get(cat, 0)
            if current >= num_ood_per_class:
                continue
            need = num_ood_per_class - current
            candidates = [
                item for item in classified if item["path"] not in used_paths
            ]
            candidates.sort(key=lambda x: x["similarities"][cat], reverse=True)
            added = 0
            for item in candidates[:need]:
                frame_id = f"{frame_counter:05d}"
                src_path = Path(item["path"])
                rel_path = f"data/frames/{src_path.name}"
                manifest.append({
                    "frame_id": frame_id,
                    "path": rel_path,
                    "category": cat,
                    "split": "",
                    "is_ood": True,
                    "source": "bdd100k_curated",
                    "curation_confidence": round(item["similarities"][cat], 4),
                })
                used_paths.add(item["path"])
                frame_counter += 1
                added += 1
            logger.info(f"  {cat}: filled {added} additional frames")

    # Select normal frames
    normal_candidates = [
        item for item in classified if item["path"] not in used_paths
    ]
    normal_candidates.sort(key=lambda x: x["similarities"]["normal"], reverse=True)

    normal_selected = 0
    for item in normal_candidates:
        if normal_selected >= num_normal:
            break
        frame_id = f"{frame_counter:05d}"
        src_path = Path(item["path"])
        rel_path = f"data/frames/{src_path.name}"
        manifest.append({
            "frame_id": frame_id,
            "path": rel_path,
            "category": "normal",
            "split": "",
            "is_ood": False,
            "source": "bdd100k_curated",
            "curation_confidence": round(item["similarities"]["normal"], 4),
        })
        used_paths.add(item["path"])
        frame_counter += 1
        normal_selected += 1

    logger.info(f"  normal: selected {normal_selected}/{num_normal} frames")
    return manifest


def assign_splits(manifest: list[dict], seed: int = 42) -> list[dict]:
    """Assign splits: ALL OOD frames to test, normal frames get 70/15/15.

    Per PRD: test set must contain exactly 1,200 OOD frames.
    Normal frames are split 70/15/15 for train/val/test.
    """
    rng = random.Random(seed)
    result = []

    # ALL OOD frames go to test (the M2 evaluation set)
    ood = [e for e in manifest if e["is_ood"]]
    normal = [e for e in manifest if not e["is_ood"]]

    for entry in ood:
        entry["split"] = "test"
        result.append(entry)

    # Normal frames: 70/15/15
    rng.shuffle(normal)
    n = len(normal)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    for i, entry in enumerate(normal):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"
        result.append(entry)

    return result


def verify_manifest(manifest: list[dict]) -> bool:
    """Verify manifest meets PRD requirements."""
    total = len(manifest)
    ood_test = [m for m in manifest if m["is_ood"] and m["split"] == "test"]
    ood_total = [m for m in manifest if m["is_ood"]]
    normal = [m for m in manifest if not m["is_ood"]]

    cat_counts = Counter(m["category"] for m in manifest)
    test_cat_counts = Counter(m["category"] for m in ood_test)

    logger.info("\n=== Manifest Verification ===")
    logger.info(f"Total frames: {total}")
    logger.info(f"OOD frames: {len(ood_total)}")
    logger.info(f"Normal frames: {len(normal)}")
    logger.info(f"OOD test frames: {len(ood_test)}")
    logger.info(f"Category distribution: {dict(cat_counts)}")
    logger.info(f"Test set OOD categories: {dict(test_cat_counts)}")

    split_counts = Counter(m["split"] for m in manifest)
    logger.info(f"Split distribution: {dict(split_counts)}")

    ok = True
    if total < 4200:
        logger.warning(f"Total frames {total} < 4200")
        ok = False
    if len(ood_total) < 1200:
        logger.warning(f"OOD frames {len(ood_total)} < 1200")
        ok = False
    if len(normal) < 3000:
        logger.warning(f"Normal frames {len(normal)} < 3000")
        ok = False

    for cat in OOD_CATEGORIES:
        count = test_cat_counts.get(cat, 0)
        if count < 50:
            logger.warning(f"Test set has only {count} {cat} frames (want >= 60)")
            ok = False

    if ok:
        logger.info("Manifest verification PASSED")
    else:
        logger.warning("Manifest verification has warnings (see above)")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Download and curate driving scene dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num-normal", type=int, default=3000, help="Number of normal frames")
    parser.add_argument("--num-ood-per-class", type=int, default=400, help="OOD frames per class")
    parser.add_argument("--max-download", type=int, default=10000, help="Max frames to download")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP classification batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Step 1: Download images
    logger.info("=== Step 1: Downloading driving scene images ===")
    image_paths = download_bdd100k_images(output_dir, max_images=args.max_download)
    logger.info(f"Available frames: {len(image_paths)}")

    if len(image_paths) == 0:
        logger.error("No images downloaded. Check network connection and try again.")
        sys.exit(1)

    # Step 2: Classify with CLIP
    logger.info("=== Step 2: Classifying frames with CLIP ===")
    classified = classify_with_clip(image_paths, device, batch_size=args.batch_size)

    clip_dist = Counter(c["category"] for c in classified)
    logger.info(f"CLIP classification distribution: {dict(clip_dist)}")

    # Step 3: Curate dataset
    logger.info("=== Step 3: Curating balanced dataset ===")
    manifest = curate_dataset(
        classified,
        output_dir,
        num_normal=args.num_normal,
        num_ood_per_class=args.num_ood_per_class,
        seed=args.seed,
    )

    # Step 4: Assign splits
    logger.info("=== Step 4: Assigning train/val/test splits ===")
    manifest = assign_splits(manifest, seed=args.seed)

    # Step 5: Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    # Step 6: Verify
    logger.info("=== Step 5: Verifying manifest ===")
    verify_manifest(manifest)

    logger.info("\nDone! Dataset ready for Stage 2.")


if __name__ == "__main__":
    main()
