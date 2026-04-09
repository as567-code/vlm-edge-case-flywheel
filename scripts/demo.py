#!/usr/bin/env python3
"""Interactive demo for VLM Edge-Case Flywheel.

Mode A: Full inference demo (requires data/frames/ and checkpoints/best_model.pt)
Mode B: Results-only display (works with zero dependencies beyond stdlib)
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FRAMES_DIR = ROOT / "data" / "frames"
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"

# ──────────────────────────────────────────────────────────────────
# Shared utilities (stdlib only)
# ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def hbar(width: int = 64):
    print("─" * width)


def section(title: str, width: int = 64):
    print()
    hbar(width)
    print(f"  {title}")
    hbar(width)


# ──────────────────────────────────────────────────────────────────
# Mode B: Results-only (no torch/PIL/open_clip needed)
# ──────────────────────────────────────────────────────────────────

def show_accuracy_comparison(baseline: dict, finetuned: dict):
    section("Baseline vs Fine-tuned Accuracy")
    b_acc = baseline["accuracy"]
    f_acc = finetuned["accuracy"]
    gain = f_acc - b_acc
    print(f"  Vanilla CLIP (zero-shot):   {b_acc:>6.1%}")
    print(f"  Fine-tuned CLIP:            {f_acc:>6.1%}")
    print(f"  Gain:                       +{gain:.1%} ({gain * 100:.1f} percentage points)")


def show_per_class_table(baseline: dict, finetuned: dict):
    section("Per-Class Metrics (Fine-tuned)")
    header = f"  {'Category':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Base F1':>8}"
    print(header)
    print("  " + "-" * 56)
    for cat in ["construction_zone", "emergency_vehicle", "lane_blockage"]:
        f = finetuned["per_class"][cat]
        b = baseline["per_class"][cat]
        print(
            f"  {cat:<25} {f['precision']:>7.3f} {f['recall']:>7.3f} "
            f"{f['f1']:>7.3f} {b['f1']:>8.3f}"
        )


def show_training_curve(training: dict):
    section("Training Curve Summary")
    history = training["history"]
    first = history[0]
    best_epoch = max(history, key=lambda e: e["val_accuracy"])
    last = history[-1]
    print(f"  Epoch  1: loss={first['train_loss']:.3f}  val_acc={first['val_accuracy']:.1%}")
    print(f"  Epoch {best_epoch['epoch']:>2}: loss={best_epoch['train_loss']:.3f}  "
          f"val_acc={best_epoch['val_accuracy']:.1%}  <-- best")
    print(f"  Epoch {last['epoch']:>2}: loss={last['train_loss']:.3f}  val_acc={last['val_accuracy']:.1%}")
    print()
    # ASCII sparkline of val accuracy
    accs = [e["val_accuracy"] for e in history]
    lo, hi = min(accs), max(accs)
    span = hi - lo if hi > lo else 1
    blocks = " ▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(8, int((a - lo) / span * 8))] for a in accs)
    print(f"  Val accuracy: [{spark}]  {lo:.1%} -> {hi:.1%}")


def show_flywheel_routing(flywheel: dict):
    section("Flywheel Routing Breakdown")
    total = flywheel["total_frames"]
    auto = flywheel["auto_labeled"]
    active = flywheel["active_learning"]
    low = flywheel["low_confidence"]
    print(f"  Total frames scored:     {total:>5}")
    print(f"  Auto-labeled (high):     {auto:>5}  ({auto/total:.1%})")
    print(f"  Active learning (mid):   {active:>5}  ({active/total:.1%})")
    print(f"  Low confidence (skip):   {low:>5}  ({low/total:.1%})")

    # Visual bar
    aw = int(auto / total * 40)
    alw = int(active / total * 40)
    lw = 40 - aw - alw
    print()
    bar = "[" + "█" * aw + "▓" * alw + "░" * lw + "]"
    print(f"  {bar}")
    print(f"   {'auto':^{max(aw, 1)}}  {'active-learn':^{max(alw, 1)}}  {'skip':^{max(lw, 1)}}")


def show_speedup_math(flywheel: dict):
    section("Curation Speedup Calculation")
    total = flywheel["total_frames"]
    auto = flywheel["auto_labeled"]
    active = flywheel["active_learning"]
    low = flywheel["low_confidence"]
    manual_s = flywheel["manual_time_s"]
    fw_s = flywheel["flywheel_time_s"]
    infer_s = flywheel["inference_time_s"]

    print(f"  Manual annotation:  {total} frames x 30s = {manual_s:,.0f}s ({manual_s/3600:.1f}h)")
    print("  Flywheel pipeline:")
    print(f"    Inference time:   {infer_s:.1f}s (model scoring all frames)")
    print(f"    Auto-labeled:     {auto} frames x 0s   = 0s (no human review)")
    print(f"    Active learning:  {active} frames x 12s = {active * 12:,}s (model-assisted)")
    print(f"    Low confidence:   {low} frames x 0s   = 0s (skipped)")
    print(f"  Total flywheel:     {fw_s:,.0f}s ({fw_s/3600:.1f}h)")
    print(f"  Speedup:            {manual_s:,.0f}s / {fw_s:,.0f}s = {flywheel['curation_speedup']}x")


def show_metric_dashboard(baseline: dict, finetuned: dict, annotation: dict, flywheel: dict):
    section("Metric Dashboard (M1-M6)")
    b_acc = baseline["accuracy"]
    f_acc = finetuned["accuracy"]
    gain = f_acc - b_acc
    al_frac = flywheel["active_learning_fraction"]
    speedup = flywheel["curation_speedup"]
    ann_red = annotation["auto_labeled_fraction"]

    metrics = [
        ("M1", "Annotation Reduction", ">= 62%", f"{ann_red:.1%}", ann_red >= 0.62),
        ("M2", "OOD Accuracy (1,200 fr)", ">= 83%", f"{f_acc:.1%}", f_acc >= 0.83),
        ("M3", "Active Learning Routing", ">= 40%", f"{al_frac:.1%}", al_frac >= 0.40),
        ("M4", "Curation Speedup", ">= 3x", f"{speedup:.1f}x", speedup >= 3.0),
        ("M5", "Edge-Case Categories", "3", "3", True),
        ("M6", "Fine-tuning Gain", ">= 10pp", f"+{gain:.1%}", gain >= 0.10),
    ]

    print(f"  {'':>4} {'Metric':<26} {'Target':>10} {'Achieved':>10} {'':>4}")
    print("  " + "-" * 58)
    for mid, name, target, achieved, passed in metrics:
        mark = "PASS" if passed else "FAIL"
        print(f"  {mid:>4} {name:<26} {target:>10} {achieved:>10} {mark:>4}")
    print("  " + "-" * 58)

    if all(m[4] for m in metrics):
        print("  ALL 6 METRICS PASSED")
    else:
        failed = [m[0] for m in metrics if not m[4]]
        print(f"  FAILED: {', '.join(failed)}")


def results_only_mode():
    print()
    print("  Running in RESULTS-ONLY mode (no model/data found).")
    print("  To run full demo with live inference:")
    print("    python scripts/download_data.py && python scripts/train.py")
    print()

    baseline = load_json(RESULTS / "baseline_metrics.json")
    finetuned = load_json(RESULTS / "finetuned_metrics.json")
    training = load_json(RESULTS / "training_summary.json")
    annotation = load_json(RESULTS / "annotation_metrics.json")
    flywheel = load_json(RESULTS / "flywheel_benchmark.json")

    missing = []
    for name, data in [("baseline_metrics", baseline), ("finetuned_metrics", finetuned),
                        ("training_summary", training), ("annotation_metrics", annotation),
                        ("flywheel_benchmark", flywheel)]:
        if data is None:
            missing.append(name)

    if missing:
        print(f"  Warning: missing results files: {', '.join(missing)}")
        print("  Some sections will be skipped.\n")

    if baseline and finetuned:
        show_accuracy_comparison(baseline, finetuned)
        show_per_class_table(baseline, finetuned)

    if training:
        show_training_curve(training)

    if flywheel:
        show_flywheel_routing(flywheel)
        show_speedup_math(flywheel)

    if baseline and finetuned and annotation and flywheel:
        show_metric_dashboard(baseline, finetuned, annotation, flywheel)

    print()


# ──────────────────────────────────────────────────────────────────
# Mode A: Full inference demo (requires torch, PIL, open_clip)
# ──────────────────────────────────────────────────────────────────

def full_demo():
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from src.data.augmentations import get_clip_preprocess
    from src.data.dataset import DrivingFrameDataset
    from src.flywheel.scorer import FrameScorer
    from src.model.clip_wrapper import CLIPWrapper
    from src.model.text_anchors import compute_text_anchors, get_scene_prompts
    from src.utils.config import load_config
    from src.utils.device import get_device

    device = get_device()
    print(f"\n  Device: {device}")

    # Load model + checkpoint
    t0 = time.time()
    clip = CLIPWrapper(device=device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    clip.model.load_state_dict(state)
    clip.model.eval()
    clip.model.requires_grad_(False)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Text anchors + scorer
    prompts = get_scene_prompts(str(ROOT / "configs" / "text_anchors.yaml"))
    anchors, cat_names = compute_text_anchors(clip, prompts)
    flywheel_cfg = load_config(str(ROOT / "configs" / "flywheel.yaml"))
    scorer = FrameScorer(
        clip, anchors, cat_names,
        high_threshold=flywheel_cfg["scoring"]["high_confidence_threshold"],
        low_threshold=flywheel_cfg["scoring"]["low_confidence_threshold"],
    )
    transform = get_clip_preprocess()

    # Pick 5 OOD test frames: 1 construction, 2 emergency, 2 lane_blockage
    dataset = DrivingFrameDataset(str(ROOT / "data" / "manifest.json"), split="test", transform=None)
    by_cat = {}
    for entry in dataset.entries:
        by_cat.setdefault(entry["category"], []).append(entry)

    picks = []
    random.seed(42)
    for cat, count in [("construction_zone", 1), ("emergency_vehicle", 2), ("lane_blockage", 2)]:
        pool = by_cat.get(cat, [])
        picks.extend(random.sample(pool, min(count, len(pool))))

    section("Live Inference on 5 OOD Test Frames")

    for i, entry in enumerate(picks, 1):
        img_path = dataset.root_dir / entry["path"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        result = scorer.score(img_tensor)

        route_label = {
            "auto_label": "AUTO-LABEL",
            "active_learning": "ACTIVE-LEARNING",
            "low_confidence": "SKIP",
        }
        print(f"\n  [{i}] {entry['path']}")
        print(f"      True class:      {entry['category']}")
        print(f"      Predicted:       {result['predicted_class']}")
        print(f"      Confidence:      {result['confidence']:.4f}")
        print(f"      Route:           {route_label.get(result['route'], result['route'])}")
        print("      Similarities:")
        for cat, sim in result["similarities"].items():
            bar_len = max(0, int(sim * 150))
            marker = " <--" if cat == result["predicted_class"] else ""
            print(f"        {cat:<25} {sim:.4f} {'█' * bar_len}{marker}")

    # Print results summary
    baseline = load_json(RESULTS / "baseline_metrics.json")
    finetuned = load_json(RESULTS / "finetuned_metrics.json")
    training = load_json(RESULTS / "training_summary.json")
    annotation = load_json(RESULTS / "annotation_metrics.json")
    flywheel = load_json(RESULTS / "flywheel_benchmark.json")

    if baseline and finetuned:
        show_accuracy_comparison(baseline, finetuned)
        show_per_class_table(baseline, finetuned)
    if training:
        show_training_curve(training)
    if flywheel:
        show_flywheel_routing(flywheel)
        show_speedup_math(flywheel)
    if baseline and finetuned and annotation and flywheel:
        show_metric_dashboard(baseline, finetuned, annotation, flywheel)

    print(f"\n  Total demo time: {time.time() - t0:.1f}s")
    print()


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    print()
    print("  VLM Edge-Case Flywheel Demo")
    print("  CLIP ViT-B/32 for Autonomous Driving Edge-Case Detection")

    has_data = FRAMES_DIR.exists() and any(FRAMES_DIR.iterdir()) if FRAMES_DIR.exists() else False
    has_model = CHECKPOINT.exists()

    if has_data and has_model:
        print("  Mode: FULL INFERENCE (model + data detected)")
        full_demo()
    else:
        reasons = []
        if not has_data:
            reasons.append("data/frames/ not found")
        if not has_model:
            reasons.append("checkpoints/best_model.pt not found")
        print(f"  Mode: RESULTS ONLY ({'; '.join(reasons)})")
        results_only_mode()


if __name__ == "__main__":
    main()
