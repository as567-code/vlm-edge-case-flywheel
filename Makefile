.PHONY: setup data baseline train flywheel metrics demo test all clean

setup:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

baseline:
	python scripts/run_baseline.py

train:
	python scripts/train.py
	python scripts/evaluate.py

flywheel:
	python scripts/measure_annotation_reduction.py --split test
	python scripts/benchmark_flywheel.py

metrics:
	python scripts/run_all_metrics.py

demo:
	python scripts/demo.py

test:
	pytest tests/ -v

all: setup data baseline train flywheel metrics test

clean:
	rm -rf checkpoints/ output/ results/
