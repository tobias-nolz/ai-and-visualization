"""Load the WILDS iWildCam ERM baseline (torchvision ResNet-50, 182 classes).

The model was fine-tuned by WILDS; the group uses the released checkpoint as-is.
The plain ImageNet ResNet-50 is also available for the linear-probe baseline.
"""

import torch
from torch import nn
from torchvision import models

from common import resolve
from common.evaluate import load_probe


def get_device(cfg: dict) -> torch.device:
    """Pick the device to run the model on.

    Args:
        cfg: Shared config; uses `inference.device` ("auto", "cuda" or "cpu").

    Returns:
        The configured device; "auto" picks CUDA when available, else CPU.
    """
    name = cfg["inference"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_model(cfg: dict, device: torch.device | None = None) -> nn.Module:
    """Build ResNet-50 and load the WILDS fine-tuned checkpoint.

    WILDS saves `{"algorithm": state_dict}` with keys prefixed `model.`
    (e.g. `model.fc.weight`); the prefix is stripped before loading.

    Args:
        cfg: Shared config; uses `model.arch`, `model.num_classes` and `paths.checkpoint`.
        device: Target device. Defaults to `get_device(cfg)`.

    Returns:
        The fine-tuned model in eval mode, outputting logits [N, num_classes].
    """
    device = device or get_device(cfg)
    model = getattr(models, cfg["model"]["arch"])(weights=None, num_classes=cfg["model"]["num_classes"])

    ckpt = torch.load(resolve(cfg["paths"]["checkpoint"]), map_location="cpu", weights_only=True)
    state = ckpt.get("algorithm", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def load_imagenet_backbone(cfg: dict, device: torch.device | None = None) -> nn.Module:
    """Load the ImageNet-pretrained ResNet-50 without its classifier.

    `IMAGENET1K_V1` are the weights WILDS fine-tuned from (`pretrained=True`).
    Used for the linear-probe baseline (see common/evaluate.py).

    Args:
        cfg: Shared config; uses `model.arch` (and `inference.device` if `device` is None).
        device: Target device. Defaults to `get_device(cfg)`.

    Returns:
        Feature extractor in eval mode, outputting features [N, 2048].
    """
    device = device or get_device(cfg)
    model = getattr(models, cfg["model"]["arch"])(weights="IMAGENET1K_V1")
    backbone, _ = split_model(model)
    return backbone.to(device).eval()


def load_probe_model(cfg: dict, device: torch.device | None = None) -> nn.Module:
    """Load the full linear-probe model: ImageNet backbone + the probe from `make baselines`.

    Counterpart of `load_model` for the ImageNet-pretrained model, e.g. for
    saliency methods. Takes normalized images (see `common.data.normalize`).

    Args:
        cfg: Shared config; uses `model.arch`, `model.num_classes` and `paths.probe_checkpoint`.
        device: Target device. Defaults to `get_device(cfg)`.

    Returns:
        Model in eval mode, outputting logits [N, num_classes]; classes unseen
        when training the probe get -inf.
    """
    device = device or get_device(cfg)
    return nn.Sequential(load_imagenet_backbone(cfg, device), load_probe(cfg, device)).eval()


def split_model(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Split a ResNet into feature extractor and classifier head.

    Args:
        model: torchvision ResNet, e.g. from `load_model`.

    Returns:
        `(backbone, head)`: the backbone maps images to penultimate features
        [N, 2048]; the head (`model.fc`) maps those features to logits.
    """
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return backbone, model.fc
