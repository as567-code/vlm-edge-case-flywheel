import json
import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    """Load JSON from a file."""
    with open(path) as f:
        return json.load(f)
