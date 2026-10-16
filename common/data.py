"""WILDS iWildCam v2.0: seeds, metadata, splits and preprocessing.

Expected layout after `make data`:
    data/iwildcam_v2.0/metadata.csv     one row per image (split, y, location, datetime, ...)
    data/iwildcam_v2.0/categories.csv   y -> species name
    data/iwildcam_v2.0/train/*.jpg      all images (every split lives in train/)
"""

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import resolve

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic.

    Args:
        seed: Seed for all random number generators (config key `seed`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_class_names(cfg: dict) -> pd.Series:
    """Load the species name of every class index from categories.csv.

    categories.csv also lists species without images in WILDS v2.0 (all with
    placeholder y = 99999); only the `model.num_classes` real classes are kept.

    Args:
        cfg: Shared config; uses `paths.data_dir` and `model.num_classes`.

    Returns:
        Species names indexed by class index `y` (0 to num_classes - 1).
    """
    cats = pd.read_csv(resolve(cfg["paths"]["data_dir"]) / "categories.csv")
    return cats[cats["y"] < cfg["model"]["num_classes"]].set_index("y")["name"]


def load_metadata(cfg: dict) -> pd.DataFrame:
    """Load metadata.csv and add the species name of each image.

    Args:
        cfg: Shared config; uses `paths.data_dir` and `model.num_classes`.

    Returns:
        One row per image (all splits) with the metadata.csv columns plus `label_name`.
    """
    meta = pd.read_csv(resolve(cfg["paths"]["data_dir"]) / "metadata.csv", index_col=0)
    meta["label_name"] = meta["y"].map(load_class_names(cfg))
    return meta


def select_split(meta: pd.DataFrame, split: str, cfg: dict) -> pd.DataFrame:
    """Select the rows of one WILDS split, optionally as a seeded random subset.

    Args:
        meta: Full metadata from `load_metadata`.
        split: WILDS split name: "train", "val", "test", "id_val" or "id_test".
        cfg: Shared config; uses `dataset.max_samples_per_split` and `seed`.

    Returns:
        The split's rows with a fresh 0..n-1 index. If `max_samples_per_split`
        is set, a random subset of at most that size (the same on every run).
    """
    df = meta[meta["split"] == split]
    n = cfg["dataset"]["max_samples_per_split"]
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=cfg["seed"])
    return df.reset_index(drop=True)


def get_transform(cfg: dict) -> transforms.Compose:
    """Build the WILDS `image_base` eval transform up to (not including) normalization.

    Resizes to `model.image_size` without cropping, as in WILDS training and
    evaluation, and keeps the pixels as uint8. Normalization is done separately
    by `normalize` (on the GPU), so data-loader workers pass 4x less data.

    Args:
        cfg: Shared config; uses `model.image_size`.

    Returns:
        Transform from a PIL image to a uint8 tensor [3, H, W] (values 0-255).
    """
    return transforms.Compose([
        transforms.Resize(tuple(cfg["model"]["image_size"])),
        transforms.PILToTensor(),
    ])


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Scale uint8 images to [0, 1] and apply ImageNet normalization.

    Same operations in the same order as torchvision's `ToTensor` + `Normalize`,
    so the result is bit-identical to the original WILDS preprocessing.

    Args:
        images: uint8 images [N, 3, H, W] or [3, H, W] from `get_transform`, on any device.

    Returns:
        float32 model input of the same shape, on the same device.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(-1, 1, 1)
    # 255 as a tensor, not a Python number: on CUDA, dividing by a Python number
    # multiplies by its reciprocal, which rounds differently and changes predictions
    max_value = torch.tensor(255.0, device=images.device)
    return images.float().div(max_value).sub(mean).div(std)


class IWildCamImages(Dataset):
    """Images for the rows of a metadata table, preprocessed with `get_transform`.

    Each item is `(image, i)`: the resized uint8 image tensor [3, H, W] and its
    row position `i` in the table passed to the constructor. Apply `normalize`
    before feeding images to the model.

    Args:
        df: Metadata rows to load (needs a `filename` column), e.g. from `select_split`.
        cfg: Shared config; uses `paths.data_dir` and `model.image_size`.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.filenames = df["filename"].tolist()
        self.image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
        self.transform = get_transform(cfg)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_dir / self.filenames[i]).convert("RGB")
        return self.transform(image), i
