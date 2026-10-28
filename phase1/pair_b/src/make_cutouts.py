"""Animal cut-outs (3 per species group): MegaDetector box + SAM mask on the full-resolution LILA originals.

    python phase1/pair_b/src/make_cutouts.py candidates   # candidates + contact sheets
    python phase1/pair_b/src/make_cutouts.py              # cut-outs from cutouts/cutout_choices.csv
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from common import load_config, resolve
from common.data import load_metadata
from common.model import get_device

LILA_ZIP = "https://storage.googleapis.com/public-datasets-lila/iwildcam-2022/iwildcam-2022.zip"
DETECTIONS = "iwildcam-2022/metadata/iwildcam2022_mdv4_detections.json"
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"

PAIR_B = Path(__file__).resolve().parents[1]
SPECIES_GROUPS = PAIR_B.parents[1] / "data" / "species_groups.csv"
GROUPS = [
    "elephant", "giraffe", "zebra/horse", "antelope", "cattle/buffalo", "deer", "pig/peccary", "tapir",
    "big cat", "small cat", "dog/fox", "monkey/ape", "agouti/paca", "porcupine", "bird",
]
# evaluation splits first, train only if a group has too few
SPLITS = ["id_test", "test", "val", "id_val", "train"]
N_CANDIDATES = 40  # per group
N_SHOWN = 12  # per contact sheet


class HttpFile(io.RawIOBase):
    """Read-only file over HTTP range requests, so zipfile can open a remote zip."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.size = int(requests.head(url).headers["content-length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        data = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_lila():
    print("reading the zip index (~10 s) ...")
    return zipfile.ZipFile(io.BufferedReader(HttpFile(LILA_ZIP), buffer_size=1 << 20))


def slug(group):
    return group.replace("/", "_").replace(" ", "_")


def data_path(kind, image_id, cfg):
    """Local cache: kind = "instance_masks" (DeepMAC), "fullres" (original jpg) or "sam_masks"."""
    ext = ".jpg" if kind == "fullres" else ".png"
    return resolve(cfg["paths"]["data_dir"]) / kind / f"{image_id}{ext}"


def fetch(lila, kind, image_id, cfg):
    """Download a DeepMAC mask or a full-resolution original from the LILA zip (once). False if missing."""
    path = data_path(kind, image_id, cfg)
    if path.exists():
        return True
    if kind == "instance_masks":
        names = [f"iwildcam-2022/instance-masks/{image_id}.png"]
    else:  # the original is in one of the two folders
        names = [f"iwildcam-2022/train/{image_id}.jpg", f"iwildcam-2022/test/{image_id}.jpg"]
    name = next((n for n in names if n in lila.NameToInfo), None)
    if name is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lila.read(name))
    return True


def load_detections(cfg, lila=None):
    """MegaDetector boxes per image id (downloaded once, 82 MB)."""
    path = resolve(cfg["paths"]["data_dir"]) / "megadetector" / "iwildcam2022_mdv4_detections.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((lila or open_lila()).read(DETECTIONS))
    images = json.loads(path.read_text())["images"]
    return {Path(im["file"]).stem: im["detections"] for im in images}


def is_color(img):
    """Infrared/night frames are (almost) gray: all three channels equal."""
    a = np.asarray(img.resize((64, 64)), dtype=float)
    return np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() > 6


def find_candidates(cfg, lila):
    """Up to N_CANDIDATES images per group with one clear, complete animal in a colour frame."""
    groups = pd.read_csv(SPECIES_GROUPS).set_index("y")
    meta = load_metadata(cfg)
    meta = meta[meta["split"].isin(SPLITS)].copy()
    meta["is_train"] = meta["split"] == "train"
    meta["group"] = meta["y"].map(groups["group"])
    meta["common_name"] = meta["y"].map(groups["common_name"])
    detections = load_detections(cfg, lila)
    image_dir = resolve(cfg["paths"]["data_dir"]) / "train"

    rows = []
    for group in GROUPS:
        d = meta[meta["group"] == group].sample(frac=1, random_state=cfg["seed"])
        d = d.sort_values("is_train", kind="stable")
        found, seqs = 0, set()
        for _, r in d.iterrows():
            dets = [x for x in detections.get(r["image_id"], []) if x["conf"] > 0.5]
            # exactly one clear animal, not too small, not cut off at the frame border
            if len(dets) != 1 or dets[0]["category"] != "1" or dets[0]["conf"] < 0.95:
                continue
            x, y, w, h = dets[0]["bbox"]
            good_size = 0.02 < w * h < 0.3
            at_border = x < 0.01 or y < 0.01 or x + w > 0.99 or y + h > 0.99
            if not good_size or at_border:
                continue
            if r["seq_id"] in seqs:  # one image per sequence
                continue
            if not is_color(Image.open(image_dir / r["filename"]).convert("RGB")):
                continue
            has_files = (fetch(lila, "instance_masks", r["image_id"], cfg)
                         and fetch(lila, "fullres", r["image_id"], cfg))
            if not has_files:
                continue
            seqs.add(r["seq_id"])
            # the mask index is the position in the full detection list (1-based)
            det_index = detections[r["image_id"]].index(dets[0]) + 1
            rows.append(dict(
                group=group,
                idx=found,
                image_id=r["image_id"],
                split=r["split"],
                species=r["label_name"],
                common_name=r["common_name"],
                location=r["location"],
                det_index=det_index,
                conf=dets[0]["conf"],
                x=x,
                y=y,
                w=w,
                h=h
            ))
            found += 1
            if found == N_CANDIDATES:
                break
        print(f"{group}: {found} candidates")
    return pd.DataFrame(rows)


def sam_masks(cands, cfg):
    """SAM mask for every candidate (cached as png) + a few numbers to rank the candidates by."""
    device = get_device(cfg)
    sam = sam_model_registry["vit_h"](checkpoint=str(resolve(SAM_CHECKPOINT))).to(device)
    predictor = SamPredictor(sam)
    stats = []
    for _, r in tqdm(cands.iterrows(), total=len(cands), desc="SAM"):
        img = Image.open(data_path("fullres", r["image_id"], cfg)).convert("RGB")
        W, H = img.size
        deepmac = Image.open(data_path("instance_masks", r["image_id"], cfg)).resize(img.size, Image.NEAREST)
        deepmac = np.asarray(deepmac) == r["det_index"]
        # MegaDetector boxes are tight, give SAM a few pixels of room for tails and hooves
        pad = 0.03 * max(r["w"] * W, r["h"] * H)
        box = np.array([
            r["x"] * W - pad,
            r["y"] * H - pad,
            (r["x"] + r["w"]) * W + pad,
            (r["y"] + r["h"]) * H + pad,
        ])
        predictor.set_image(np.asarray(img))
        with torch.no_grad():
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        ious = [(m & deepmac).sum() / max((m | deepmac).sum(), 1) for m in masks]
        best = int(np.argmax(ious))
        mask = clean_mask(masks[best])
        path = data_path("sam_masks", r["image_id"], cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        # sharpness = variance of the Laplacian inside the animal (blurry = low)
        gray = np.asarray(img.convert("L"), dtype=float)
        inside = ndimage.binary_erosion(mask, iterations=3)
        stats.append(dict(
            iou_deepmac=ious[best],
            sam_score=float(scores[best]),
            area=int(mask.sum()),
            sharpness=ndimage.laplace(gray)[inside].var()
        ))
    cands = pd.concat([cands, pd.DataFrame(stats, index=cands.index)], axis=1)
    # rank within the group: DeepMAC agreement counts double
    ranks = cands.groupby("group")[["iou_deepmac", "area", "sharpness"]].rank(pct=True)
    cands["score"] = (2 * ranks["iou_deepmac"] + ranks["area"] + ranks["sharpness"]) / 4
    return cands


def clean_mask(mask):
    """Largest blob, holes filled."""
    labels, n = ndimage.label(mask)
    if n > 1:
        mask = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(mask)


def cut_out(row, cfg):
    """RGBA crop of one animal from the full-resolution original, with a soft edge."""
    img = Image.open(data_path("fullres", row["image_id"], cfg)).convert("RGB")
    mask = np.asarray(Image.open(data_path("sam_masks", row["image_id"], cfg))) > 0
    # shrink by 2 px before blurring the edge, otherwise a thin ring of the old background stays
    alpha = ndimage.binary_erosion(mask, iterations=2)
    alpha = Image.fromarray(alpha.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5))
    rgba = img.copy()
    rgba.putalpha(alpha)
    ys, xs = np.nonzero(mask)
    return rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def contact_sheets(cands, cfg):
    """The N_SHOWN best candidates per group on a blue background, so mask errors are visible."""
    out = PAIR_B / "figures" / "cutout_candidates"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for group, d in cands.groupby("group"):
        d = d.sort_values("score", ascending=False).head(N_SHOWN)
        fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
            im = cut_out(r, cfg)
            ax.imshow(np.full((im.height, im.width, 3), (95, 145, 205), np.uint8))
            ax.imshow(im)
            ax.set_title(
                f"{r['idx']}: {r['common_name']} ({r['split']}, loc {r['location']})\n"
                f"IoU {r['iou_deepmac']:.2f}, {im.width}x{im.height}px",
                fontsize=9
            )
        fig.suptitle(f"{group}: best {len(d)} of {(cands['group'] == group).sum()} candidates")
        fig.savefig(out / f"{slug(group)}.png", dpi=90)
        plt.close(fig)
    print(f"contact sheets in {out}")


def typical_sizes(cfg):
    """Median sqrt(box area) per group in the 448x448 input, from single-animal MegaDetector boxes."""
    groups = pd.read_csv(SPECIES_GROUPS).set_index("y")
    meta = load_metadata(cfg)
    meta["group"] = meta["y"].map(groups["group"])
    detections = load_detections(cfg)
    side = cfg["model"]["image_size"][0]
    rows = []
    for image_id, group in zip(meta["image_id"], meta["group"]):
        dets = [x for x in detections.get(image_id, []) if x["conf"] > 0.8]
        if group in GROUPS and len(dets) == 1:
            x, y, w, h = dets[0]["bbox"]
            rows.append((group, side * np.sqrt(w * h)))
    boxes = pd.DataFrame(rows, columns=["group", "px"])
    sizes = boxes.groupby("group")["px"].agg(size_px="median", n_boxes="count").reset_index()
    sizes["size_px"] = sizes["size_px"].round(1)
    return sizes


def build_cutouts(cfg):
    cands = pd.read_csv(PAIR_B / "cutouts" / "candidates.csv")
    choices = pd.read_csv(PAIR_B / "cutouts" / "cutout_choices.csv")
    chosen = choices.merge(cands, on=["group", "idx"], validate="1:1")
    chosen["variant"] = chosen.groupby("group").cumcount()
    chosen["file"] = chosen["group"].map(slug) + "_" + chosen["variant"].astype(str) + ".png"
    for old in (PAIR_B / "cutouts").glob("*.png"):
        old.unlink()
    sizes = []
    for _, r in chosen.iterrows():
        cut_out(r, cfg).save(PAIR_B / "cutouts" / r["file"])
        sizes.append(Image.open(data_path("fullres", r["image_id"], cfg)).size)
    chosen["src_w"], chosen["src_h"] = zip(*sizes)
    columns = [
        "group", "variant", "file", "common_name", "species", "image_id", "split", "location", "src_w", "src_h",
    ]
    chosen[columns].to_csv(PAIR_B / "cutouts" / "cutouts.csv", index=False)
    typical_sizes(cfg).to_csv(PAIR_B / "cutouts" / "sizes.csv", index=False)
    print(f"{len(chosen)} cut-outs written")


if __name__ == "__main__":
    cfg = load_config()
    (PAIR_B / "cutouts").mkdir(exist_ok=True)
    if sys.argv[1:] == ["candidates"]:
        cands = sam_masks(find_candidates(cfg, open_lila()), cfg)
        cands.to_csv(PAIR_B / "cutouts" / "candidates.csv", index=False)
        contact_sheets(cands, cfg)
    else:
        build_cutouts(cfg)
