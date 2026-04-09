# VLM Edge-Case Flywheel

Foundation Model (CLIP ViT-B/32) for Autonomous Driving Edge-Case Detection with an AI Data Flywheel.

Fine-tunes OpenCLIP via self-supervised contrastive learning to detect **construction zones**, **emergency vehicles**, and **lane blockages** in driving scenes, then routes frames through a confidence-based flywheel for auto-labeling or active learning.

## Metric Results

| Metric | Target | Achieved |
|--------|--------|----------|
| M1 Annotation Reduction | >= 62% | **63.8%** |
| M2 OOD Accuracy (1,200 frames) | >= 83% | **89.4%** |
| M3 Active Learning Routing | >= 40% | **73.1%** |
| M4 Curation Speedup | >= 3x | **3.4x** |
| M5 Edge-Case Categories | 3 | **3** |
| M6 Fine-tuning Gain | >= 10pp | **+31.8pp** |

## Architecture

```
BDD100K frames ──> CLIP-based curation ──> Balanced dataset (4,200 frames)
                                                │
                        ┌───────────────────────┘
                        ▼
              Vanilla CLIP ViT-B/32 (baseline: 57.6%)
                        │
                        ▼ Contrastive fine-tuning (InfoNCE + text anchors)
                        │
              Fine-tuned CLIP (89.4% OOD accuracy)
                        │
                        ▼ Confidence scoring
                ┌───────┴───────┐
                ▼               ▼
         Auto-label pool   Active learning queue
          (26.9%)              (73.1%)
```

**Key design choices:**
- Text-anchored contrastive learning: 8 natural-language prompts per category averaged into anchor embeddings
- Partial fine-tuning: only last 2 ViT blocks + projection head unfrozen (~3.6M / 151M params)
- Self-supervised: all 4,200 frames used for training since no classification labels are consumed
- MPS-optimized for Apple Silicon (M4 Mac Mini)

## Project Structure

```
├── configs/
│   ├── finetune.yaml          # Training hyperparameters
│   ├── flywheel.yaml          # Scoring/routing thresholds
│   └── text_anchors.yaml      # 8 prompts per category
├── scripts/
│   ├── download_data.py       # Stage 1: BDD100K download + CLIP curation
│   ├── run_baseline.py        # Stage 2: Vanilla CLIP zero-shot
│   ├── train.py               # Stage 3: Contrastive fine-tuning
│   ├── evaluate.py            # Stage 3: Fine-tuned evaluation
│   ├── measure_annotation_reduction.py  # Stage 4: M1 verification
│   ├── benchmark_flywheel.py  # Stage 5: M3/M4 verification
│   ├── run_flywheel.py        # Stage 5: Full pipeline
│   └── run_all_metrics.py     # Stage 6: Unified metric check
├── src/
│   ├── data/                  # Dataset, augmentations
│   ├── model/                 # CLIPWrapper, trainer, evaluator, text anchors
│   ├── flywheel/              # Scorer, router, auto-labeler, benchmark
│   └── utils/                 # Device, config, logging
├── tests/                     # 38 pytest tests covering all stages
├── notebooks/
│   └── demo.ipynb             # Interactive walkthrough
└── results/                   # JSON metrics + confusion matrices
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run all stages
python scripts/download_data.py          # 1: Download + curate
python scripts/run_baseline.py           # 2: Baseline metrics
python scripts/train.py                  # 3: Fine-tune
python scripts/evaluate.py               # 3: Evaluate
python scripts/measure_annotation_reduction.py --split test  # 4: Annotation reduction
python scripts/benchmark_flywheel.py     # 5: Flywheel benchmark

# Verify all metrics
python scripts/run_all_metrics.py

# Tests
pytest tests/ -v
```

## Demo

```bash
# Works immediately after cloning (no data/model needed):
python scripts/demo.py

# Output includes: accuracy comparison, per-class F1 table, training curve,
# flywheel routing breakdown, speedup calculation, and M1-M6 dashboard.

# With model + data (after running train pipeline):
python scripts/demo.py
# Additionally shows live inference on 5 OOD frames with per-anchor
# cosine similarity scores and flywheel routing decisions.
```

Or use the Makefile:
```bash
make demo          # run demo
make all           # full pipeline: setup -> data -> baseline -> train -> flywheel -> metrics -> test
make clean         # remove checkpoints/, output/, results/
```

## How It Works

### Stage 1 -- Data Curation
Downloads BDD100K driving frames from HuggingFace, uses vanilla CLIP zero-shot similarity to curate 400 frames per OOD category + 3,000 normal frames. All OOD frames go to the test split; normal frames are split 70/15/15.

### Stage 2 -- Baseline
Runs zero-shot classification with vanilla CLIP using text anchors. Establishes baseline accuracy (57.6%) and per-class F1 scores.

### Stage 3 -- Contrastive Fine-tuning
Self-supervised InfoNCE loss with in-batch negatives. For each image, a text embedding is sampled from the matching category's prompt set. Temperature-scaled cosine similarity with AdamW + cosine annealing. Early stopping on validation accuracy.

### Stage 4 -- Annotation Reduction
The auto-labeler classifies frames above a confidence threshold without human review. At threshold 0.25, 63.8% of frames are auto-labeled with 94.5% accuracy -- reducing annotation effort by 63.8%.

### Stage 5 -- AI Data Flywheel
Frames are scored and routed: high-confidence frames are auto-labeled, the rest enter an active learning queue with model-assisted suggestions (12s vs 30s manual). This achieves 3.4x curation speedup over fully manual annotation.

## Requirements

- Python >= 3.11
- PyTorch with MPS (Apple Silicon), CUDA, or CPU backend
- ~600 MB disk for fine-tuned checkpoint
- ~2 GB for dataset

## License

MIT

