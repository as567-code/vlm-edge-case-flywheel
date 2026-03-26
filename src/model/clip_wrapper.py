import torch
import open_clip
from PIL import Image

from src.utils.device import get_device


class CLIPWrapper:
    """Wrapper around OpenCLIP for image/text embedding extraction."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode a single image or batch to a normalized embedding."""
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        features = self.model.encode_image(image)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings to normalized embeddings."""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def get_similarity(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings."""
        return (image_features @ text_features.T).squeeze(0)

    def unfreeze_layers(self, num_blocks: int = 2) -> None:
        """Unfreeze the last N ViT transformer blocks + visual projection head for fine-tuning."""
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N visual transformer blocks
        visual = self.model.visual
        if hasattr(visual, "transformer"):
            blocks = visual.transformer.resblocks
        else:
            blocks = visual.trunk.blocks if hasattr(visual, "trunk") else []

        for block in list(blocks)[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze visual projection
        if hasattr(visual, "proj") and visual.proj is not None:
            visual.proj.requires_grad = True

        # Unfreeze the ln_post (layer norm after transformer)
        if hasattr(visual, "ln_post"):
            for param in visual.ln_post.parameters():
                param.requires_grad = True

        self.model.train()

    def trainable_params(self) -> list[torch.nn.Parameter]:
        """Return list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

