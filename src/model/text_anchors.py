import yaml
from pathlib import Path

import torch

from src.model.clip_wrapper import CLIPWrapper


SCENE_PROMPTS = {
    "construction_zone": [
        "a photo of a construction zone on a road",
        "road construction with orange cones and barriers",
        "highway construction area with workers and machinery",
        "a driving scene with road work ahead signs",
        "construction equipment blocking part of a highway",
        "a road closure due to ongoing construction work",
        "orange barrels and cones marking a construction area",
        "a dashcam view of a highway work zone",
    ],
    "emergency_vehicle": [
        "a photo of an emergency vehicle on the road",
        "an ambulance with flashing lights on a highway",
        "a police car responding to an incident",
        "a fire truck blocking a road lane",
        "emergency vehicles parked on the side of a road",
        "a dashcam view of an ambulance approaching from behind",
        "a police vehicle with flashing sirens on a city street",
        "first responder vehicles at a roadside scene",
    ],
    "lane_blockage": [
        "a photo of a blocked lane on a road",
        "a stalled vehicle blocking a driving lane",
        "road debris causing a lane blockage",
        "an accident scene blocking traffic lanes",
        "a broken down car stopped in the middle of a highway lane",
        "a dashcam view of an obstruction blocking the road ahead",
        "traffic cones diverting cars around a blocked lane",
        "a delivery truck double parked blocking a lane",
    ],
    "normal": [
        "a photo of normal highway driving",
        "a typical urban driving scene",
        "a car driving on a clear road",
        "normal traffic on a city street",
        "a dashcam view of a regular commute",
        "vehicles moving smoothly on a multi-lane highway",
        "a peaceful drive through a suburban neighborhood",
        "a clear road with light traffic ahead",
    ],
}


def get_scene_prompts(config_path: str | Path | None = None) -> dict[str, list[str]]:
    """Load scene prompts from config YAML, falling back to built-in defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)["prompts"]
    return SCENE_PROMPTS


def compute_text_anchors(
    clip_model: CLIPWrapper,
    prompts: dict[str, list[str]] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Compute averaged text anchor embeddings for each scene category.

    Returns:
        anchors: (num_classes, embed_dim) tensor of normalized anchor embeddings
        categories: list of category names in order
    """
    if prompts is None:
        prompts = SCENE_PROMPTS

    categories = sorted(prompts.keys())
    anchors = []
    for cat in categories:
        embeddings = clip_model.encode_text(prompts[cat])  # (num_prompts, embed_dim)
        anchor = embeddings.mean(dim=0)
        anchor = anchor / anchor.norm()
        anchors.append(anchor)

    return torch.stack(anchors), categories


def save_prompts_yaml(prompts: dict[str, list[str]], path: str | Path) -> None:
    """Save prompts to a YAML config file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"prompts": prompts}, f, default_flow_style=False, sort_keys=True)
