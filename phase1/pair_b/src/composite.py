"""Paste animal cut-outs onto a 3x3 grid; shared paths and camera habitats."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SIZE = 448  # model input size
CELL = SIZE / 3

PAIR_B = Path(__file__).resolve().parents[1]
SPECIES_GROUPS = PAIR_B.parents[1] / "data" / "species_groups.csv"
FOREST_T, OPEN_T = 1 / 3, 2 / 3  # SegFormer openness thresholds
CELL_NAMES = {
    (0, 0): "top left", (0, 1): "top", (0, 2): "top right",
    (1, 0): "left", (1, 1): "centre", (1, 2): "right",
    (2, 0): "bottom left", (2, 1): "bottom", (2, 2): "bottom right",
}


def camera_habitats():
    """SegFormer openness (results/camera_openness.csv) and habitat of every camera scored on daytime frames.

    Cameras scored from the fallback frames (sample_policy "available_frames", night frames allowed) are left
    out: their scores are less reliable, and the committed backgrounds were sampled without them.
    """
    cams = pd.read_csv(PAIR_B / "results" / "camera_openness.csv")
    cams = cams.loc[cams["sample_policy"] == "daytime", ["location", "openness"]].dropna()
    cams["habitat"] = np.select(
        [cams["openness"] > OPEN_T, cams["openness"] < FOREST_T],
        ["open", "forest"],
        "mixed"
    )
    return cams


def load_cutouts():
    """{(group, variant): (rgb, alpha)}, squashed like the source frame and scaled to the group's typical size."""
    table = pd.read_csv(PAIR_B / "cutouts" / "cutouts.csv")
    sizes = pd.read_csv(PAIR_B / "cutouts" / "sizes.csv").set_index("group")["size_px"]
    cutouts = {}
    for _, r in table.iterrows():
        img = Image.open(PAIR_B / "cutouts" / r["file"])
        w = img.width * SIZE / r["src_w"]
        h = img.height * SIZE / r["src_h"]
        scale = sizes[r["group"]] / np.sqrt(w * h)
        new_size = (min(round(w * scale), SIZE), min(round(h * scale), SIZE))
        img = img.resize(new_size, Image.LANCZOS)
        rgba = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()
        cutouts[r["group"], r["variant"]] = (rgba[:3], rgba[3:] / 255)
    return cutouts


def is_infrared(images):
    """Infrared frames: the colour channels are (almost) equal. images: uint8 [N, 3, H, W]."""
    img = images.float()
    red_green = (img[:, 0] - img[:, 1]).abs().mean((1, 2))
    green_blue = (img[:, 1] - img[:, 2]).abs().mean((1, 2))
    return red_green + green_blue < 6


def paste(backgrounds, cutout, row, col, gray=None):
    """Alpha-blend the cut-out centred on cell (row, col), shifted inside the frame; greyscale where `gray`."""
    rgb, alpha = (t.to(backgrounds.device) for t in cutout)
    h, w = rgb.shape[1:]
    y0 = min(max(round((row + 0.5) * CELL - h / 2), 0), SIZE - h)
    x0 = min(max(round((col + 0.5) * CELL - w / 2), 0), SIZE - w)
    rgb = rgb.expand(len(backgrounds), -1, -1, -1)
    if gray is not None:
        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        lum = lum.unsqueeze(1).expand(-1, 3, -1, -1)
        rgb = torch.where(gray.to(backgrounds.device).view(-1, 1, 1, 1), lum, rgb)
    out = backgrounds.clone()
    region = out[..., y0:y0 + h, x0:x0 + w].float()
    blended = alpha * rgb + (1 - alpha) * region
    out[..., y0:y0 + h, x0:x0 + w] = blended.round().to(torch.uint8)
    return out
