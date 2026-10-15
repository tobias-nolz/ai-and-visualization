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
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_class_names(cfg: dict) -> pd.Series:
    """Species name per class index `y` (from categories.csv)."""
    return pd.read_csv(resolve(cfg["paths"]["data_dir"]) / "categories.csv").set_index("y")["name"]


def load_metadata(cfg: dict) -> pd.DataFrame:
    """Read metadata.csv and add the species name as `label_name`."""
    meta = pd.read_csv(resolve(cfg["paths"]["data_dir"]) / "metadata.csv", index_col=0)
    meta["label_name"] = meta["y"].map(load_class_names(cfg))
    return meta


def select_split(meta: pd.DataFrame, split: str, cfg: dict) -> pd.DataFrame:
    """Rows of one split, optionally a seeded random subset (`max_samples_per_split`)."""
    df = meta[meta["split"] == split]
    n = cfg["dataset"]["max_samples_per_split"]
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=cfg["seed"])
    return df.reset_index(drop=True)


def get_transform(cfg: dict) -> transforms.Compose:
    """WILDS `image_base` eval transform: resize (no crop), ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize(tuple(cfg["model"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class IWildCamImages(Dataset):
    """Images for the rows of `df`; returns (image tensor, row index)."""

    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.filenames = df["filename"].tolist()
        self.image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
        self.transform = get_transform(cfg)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        image = Image.open(self.image_dir / self.filenames[i]).convert("RGB")
        return self.transform(image), i
