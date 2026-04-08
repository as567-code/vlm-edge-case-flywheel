#!/usr/bin/env python3
"""Stage 6: Unified metric verification — checks all 6 metrics (M1-M6)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    results_dir = Path("results")
    baseline = load(results_dir / "baseline_metrics.json")
    finetuned = load(results_dir / "finetuned_metrics.json")
    annotation = load(results_dir / "annotation_metrics.json")
    flywheel = load(results_dir / "flywheel_benchmark.json")
    manifest = load("data/manifest.json")

    print()
    print("+" + "-" * 52 + "+")
    print("| METRIC VERIFICATION SUMMARY" + " " * 24 + "|")
    print("+" + "-" * 7 + "+" + "-" * 34 + "+" + "-" * 10 + "+")

    all_pass = True

    # M1: Annotation Reduction >= 62%
    if annotation:
        val = annotation.get("auto_labeled_fraction", 0)
        ok = val >= 0.62
        status = f"  {val:.1%}" if ok else f"X {val:.1%}"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M1    | Annotation Reduction >= 62%     | {status:>8} |")

    # M2: OOD Accuracy >= 83%
    if finetuned:
        val = finetuned.get("accuracy", 0)
        ok = val >= 0.83
        status = f"  {val:.1%}" if ok else f"X {val:.1%}"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M2    | OOD Accuracy >= 83%             | {status:>8} |")

    # M3: Active Learning Routing >= 40%
    if flywheel:
        val = flywheel.get("active_learning_fraction", 0)
        ok = val >= 0.40
        status = f"  {val:.1%}" if ok else f"X {val:.1%}"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M3    | Active Learning Routing >= 40%  | {status:>8} |")

    # M4: Curation Speedup >= 3x
    if flywheel:
        val = flywheel.get("curation_speedup", 0)
        ok = val >= 3.0
        status = f"  {val:.1f}x" if ok else f"X {val:.1f}x"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M4    | Curation Speedup >= 3x          | {status:>8} |")

    # M5: 3 Edge-Case Categories
    if manifest:
        cats = set()
        for entry in manifest:
            if entry.get("is_ood"):
                cats.add(entry["category"])
        val = len(cats)
        ok = val >= 3
        status = f"  {val}/3" if ok else f"X {val}/3"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M5    | 3 Edge-Case Categories          | {status:>8} |")

    # M6: Fine-tuning Gain >= 10pp
    if finetuned and baseline:
        b_acc = baseline.get("accuracy", 0)
        f_acc = finetuned.get("accuracy", 0)
        gain = f_acc - b_acc
        ok = gain >= 0.10
        status = f"  +{gain:.1%}" if ok else f"X +{gain:.1%}"
    else:
        ok = False
        status = "X N/A"
    all_pass &= ok
    print(f"| M6    | Fine-tuning Gain >= 10pp        | {status:>8} |")

    print("+" + "-" * 7 + "+" + "-" * 34 + "+" + "-" * 10 + "+")

    if all_pass:
        print("\nALL METRICS PASSED")
        sys.exit(0)
    else:
        print("\nSOME METRICS NOT YET VERIFIED (run remaining stages)")
        sys.exit(1)


if __name__ == "__main__":
    main()
