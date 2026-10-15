"""Load the WILDS iWildCam ERM baseline (torchvision ResNet-50, 182 classes).

No fine-tuning: the group uses the released WILDS checkpoint as-is.
"""

import torch
from torch import nn
from torchvision import models

from common import resolve


def get_device(cfg: dict) -> torch.device:
    """`inference.device` from the config; `auto` picks CUDA when available."""
    name = cfg["inference"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_model(cfg: dict, device: torch.device | None = None) -> nn.Module:
    """Build ResNet-50 and load the WILDS checkpoint, in eval mode.

    WILDS saves `{"algorithm": state_dict}` with keys prefixed `model.`
    (e.g. `model.fc.weight`); the prefix is stripped before loading.
    """
    device = device or get_device(cfg)
    model = getattr(models, cfg["model"]["arch"])(weights=None, num_classes=cfg["model"]["num_classes"])

    ckpt = torch.load(resolve(cfg["paths"]["checkpoint"]), map_location="cpu", weights_only=True)
    state = ckpt.get("algorithm", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def split_model(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Return (feature extractor -> 2048-d penultimate features, classifier head)."""
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return backbone, model.fc
