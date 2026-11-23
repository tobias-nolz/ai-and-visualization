"""Network dissection (Bau et al. 2017) of the WILDS ResNet-50: which layer3 / layer4 units detect scenery
(tree, grass, sky, ...), the animal or the timestamp strips, in the fine-tuned model and its ImageNet start.

    python phase2/pair_b/src/dissect.py probe     # ~4000 daytime test/id_test frames, cache their 448x448 inputs
    python phase2/pair_b/src/dissect.py segment   # SegFormer ADE20K labels on the same 448x448 inputs
    python phase2/pair_b/src/dissect.py units     # threshold, IoU, label and top-12 images per unit
    python phase2/pair_b/src/dissect.py sheets    # contact sheets of the best units per concept
    python phase2/pair_b/src/dissect.py           # all four steps

Method as in the paper: T_k is the top-0.5 % quantile of unit k over all probe images and positions. Each
activation map is upsampled (bilinear) to the 448x448 input and thresholded at T_k; IoU(k, c) = sum |M_k & L_c|
/ sum |M_k | L_c| over the whole probe set. A unit gets its best concept as label if that IoU is above 0.04.
Note: IoU <= |M_k| / |L_c|, and M_k covers ~0.4 % of the pixels, so concepts covering more than ~10 % of the
probe pixels (tree, ground) can never pass 0.04; see max_iou_for_median_unit_mask in dissection_concepts.csv.

All masks live on the 448x448 model input (full frame squashed to 448x448, as in WILDS):
- scene concepts: SegFormer-B2 ADE20K on the frame without the timestamp strips (same crop as Phase 1),
  merged into tree, plant, grass, ground, rock, sky, water
- animal: MegaDetector animal boxes with conf >= 0.8
- info strip: top 3 % and bottom 5 % of the frame (where the timestamp bars are)

Small tables go to phase2/pair_b/results/dissection_*, contact sheets to phase2/pair_b/figures/dissection/,
caches (probe images, ADE label maps, per-image unit maxima) to outputs/dissection/.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from common import ROOT, load_config, resolve
from common.data import get_transform, load_metadata, normalize
from common.model import get_device, load_imagenet_backbone, load_model

sys.path.insert(0, str(ROOT / "phase1" / "pair_b" / "src"))
from composite import camera_habitats, is_infrared  # noqa: E402
from make_cutouts import load_detections  # noqa: E402
from measure_background import load_model as load_segformer, read_image  # noqa: E402

PAIR_B = Path(__file__).resolve().parents[1]
RESULTS = PAIR_B / "results"
FIGURES = PAIR_B / "figures" / "dissection"
CACHE = ROOT / "outputs" / "dissection"

SIZE = 448
STRIP_TOP, STRIP_BOTTOM = round(0.03 * SIZE), round(0.95 * SIZE)  # rows kept by Phase 1 read_image()

# probe set
N_PROBE = 4000
HABITATS = ["open", "mixed", "forest"]
MD_CONF = 0.8          # MegaDetector animal boxes used as the "animal" concept
EMPTY_MAX_CONF = 0.5   # an "empty" probe frame is labelled empty and has no MegaDetector box above this
PER_CAMERA = 60        # at most this many frames per camera and stratum (empty / animal); ~4000 frames in total

# concepts: merged ADE20K classes (SegFormer label names), then animal and info strip
SCENE = {
    "tree": ["tree", "palm"],
    "plant": ["plant", "flower"],
    "grass": ["grass", "field"],
    "ground": ["earth", "sand", "path", "dirt track", "land", "road", "floor"],
    "rock": ["rock", "mountain", "hill"],
    "sky": ["sky"],
    "water": ["water", "sea", "river", "lake", "waterfall"],
}
CONCEPTS = [*SCENE, "animal", "info strip"]
ANIMAL, STRIP = CONCEPTS.index("animal"), CONCEPTS.index("info strip")

# dissection
MODELS = ["finetuned", "imagenet"]
LAYERS = ["layer3", "layer4"]
QUANTILE = 0.005   # top 0.5 % of activations
IOU_MIN = 0.04
TOP = 12
BATCH = 16
UNIT_CHUNK = 64    # units upsampled at once (BATCH x UNIT_CHUNK x 448 x 448 floats, ~0.8 GB)

# contact sheets
SHEET_UNITS = 3    # best units per concept
SHEET_IMAGES = 4   # top images per unit
THUMB = 168


# --------------------------------------------------------------------------------------------- probe set


def candidates(cfg):
    """test / id_test frames at cameras with a daytime habitat, in an "animal" or an "empty" stratum.

    animal: labelled with a species and at least one MegaDetector animal box with conf >= MD_CONF.
    empty: labelled empty and no MegaDetector box (any category) with conf >= EMPTY_MAX_CONF.
    Everything else (e.g. labelled animal without a confident box) is left out.
    """
    meta = load_metadata(cfg)
    meta = meta[meta["split"].isin(["test", "id_test"])]
    habitats = camera_habitats().set_index("location")
    meta = meta[meta["location"].isin(habitats.index)].copy()
    meta["habitat"] = meta["location"].map(habitats["habitat"])
    meta["openness"] = meta["location"].map(habitats["openness"])

    detections = load_detections(cfg)
    meta = meta[meta["image_id"].isin(detections.keys())].copy()
    dets = [detections[i] for i in meta["image_id"]]
    meta["boxes"] = [[d["bbox"] for d in ds if d["category"] == "1" and d["conf"] >= MD_CONF] for ds in dets]
    max_conf = np.array([max((d["conf"] for d in ds), default=0.0) for ds in dets])
    labelled_empty = (meta["y"] == 0).to_numpy()
    has_box = meta["boxes"].map(len).to_numpy() > 0
    meta["stratum"] = np.select(
        [~labelled_empty & has_box, labelled_empty & (max_conf < EMPTY_MAX_CONF)], ["animal", "empty"], ""
    )
    return meta[meta["stratum"] != ""]


def load_input(path, transform):
    """The model input: uint8 [3, 448, 448], the full frame resized without cropping (WILDS eval transform)."""
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def usable(image):
    """Daytime colour frame with enough light and contrast (Phase 1 thresholds, measured without the strips)."""
    band = image[:, STRIP_TOP:STRIP_BOTTOM].float() / 255
    grey = band.mean(0)
    return not is_infrared(image[None]).item() and grey.mean() >= 0.15 and grey.std() >= 0.04


def round_robin(pool, n, load, rng):
    """Up to n usable frames from `pool`, one frame per camera in turn, at most PER_CAMERA per camera.

    First pass: one frame per sequence (neighbouring frames are near-duplicates); second pass: the rest.
    Returns {row label: image tensor}.
    """
    taken, per_camera = {}, dict.fromkeys(pool["location"].unique(), 0)
    for _, tier in pool.groupby("tier"):
        cameras = [list(g.index) for _, g in tier.groupby("location")]
        cameras = [cameras[i] for i in rng.permutation(len(cameras))]
        while cameras and len(taken) < n:
            heads = [queue.pop(0) for queue in cameras]
            for index, image in zip(heads, load(pool.loc[heads, "filename"])):
                location = pool.at[index, "location"]
                if len(taken) < n and per_camera[location] < PER_CAMERA and usable(image):
                    taken[index] = image
                    per_camera[location] += 1
            cameras = [q for q in cameras if q and per_camera[pool.at[q[0], "location"]] < PER_CAMERA]
    return taken


def build_probe(cfg):
    """Pick the probe frames, spread over habitats, animal / empty and cameras; cache the model inputs.

    Each habitat gets an equal share of what is left of N_PROBE (so a habitat that runs short, like open,
    passes the rest on), half of it empty frames; animal frames fill what the empty ones leave.
    """
    rng = np.random.default_rng(cfg["seed"])
    pool = candidates(cfg).sample(frac=1, random_state=cfg["seed"])
    pool["tier"] = pool.duplicated("seq_id").astype(int)
    image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
    transform = get_transform(cfg)
    workers = ThreadPoolExecutor(8)

    def load(filenames):
        return list(workers.map(lambda f: load_input(image_dir / f, transform), filenames))

    chosen = {}
    for i, habitat in enumerate(HABITATS):
        quota = round((N_PROBE - len(chosen)) / (len(HABITATS) - i))
        here = pool[pool["habitat"] == habitat]
        empty = round_robin(here[here["stratum"] == "empty"], quota // 2, load, rng)
        animal = round_robin(here[here["stratum"] == "animal"], quota - len(empty), load, rng)
        chosen |= empty | animal
        print(f"{habitat}: {len(empty)} empty + {len(animal)} animal frames")

    probe = pool.loc[list(chosen)].sort_values(["habitat", "stratum", "location", "datetime"])
    images = np.lib.format.open_memmap(CACHE / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(probe), 3, SIZE, SIZE))
    for row, index in enumerate(probe.index):
        images[row] = chosen[index].numpy()
    images.flush()

    probe = probe.reset_index(drop=True)
    probe["boxes"] = probe["boxes"].map(json.dumps)
    columns = ["image_id", "split", "location", "habitat", "openness", "stratum", "y", "label_name",
               "seq_id", "datetime", "boxes"]
    probe[columns].to_parquet(RESULTS / "dissection_probe.parquet", index=False)
    summary = probe.groupby(["habitat", "stratum"]).agg(
        frames=("image_id", "size"), cameras=("location", "nunique"), sequences=("seq_id", "nunique"))
    print(summary.to_string())
    print(f"{len(probe)} frames from {probe['location'].nunique()} cameras;",
          probe["split"].value_counts().to_dict())


def load_probe():
    probe = pd.read_parquet(RESULTS / "dissection_probe.parquet")
    probe["boxes"] = probe["boxes"].map(json.loads)
    return probe, np.load(CACHE / "images.npy", mmap_mode="r")


# --------------------------------------------------------------------------------------------- concepts


@torch.inference_mode()
def segment():
    """SegFormer ADE20K labels for every probe input; strip rows get 255. Saved as uint8 [N, 448, 448]."""
    _, images = load_probe()
    processor, model, device = load_segformer()
    labels = np.lib.format.open_memmap(CACHE / "ade.npy", mode="w+", dtype=np.uint8, shape=images.shape[:1] + (SIZE, SIZE))
    labels[:] = 255
    for first in tqdm(range(0, len(images), BATCH), desc="SegFormer"):
        batch = images[first:first + BATCH]
        # Phase 1 crop (read_image) of the 448x448 input = rows STRIP_TOP:STRIP_BOTTOM
        bands = [read_image(Image.fromarray(np.ascontiguousarray(image.transpose(1, 2, 0)))) for image in batch]
        assert bands[0].size == (SIZE, STRIP_BOTTOM - STRIP_TOP)
        inputs = processor(images=bands, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
        logits = F.interpolate(logits.float(), size=(STRIP_BOTTOM - STRIP_TOP, SIZE), mode="bilinear",
                               align_corners=False)
        labels[first:first + len(batch), STRIP_TOP:STRIP_BOTTOM] = logits.argmax(1).byte().cpu().numpy()
    labels.flush()
    names = [model.config.id2label[i].strip() for i in range(len(model.config.id2label))]
    (CACHE / "ade_names.json").write_text(json.dumps(names))

    counts = np.zeros(256, np.int64)
    for first in range(0, len(labels), 500):
        counts += np.bincount(labels[first:first + 500].ravel(), minlength=256)
    share = pd.Series(counts[:len(names)] / counts[:len(names)].sum(), index=names).sort_values(ascending=False)
    merged = {name for classes in SCENE.values() for name in classes}
    print("ADE20K classes by share of the scene pixels (* = not merged into a concept):")
    print("\n".join(f"  {'' if n in merged else '*'}{n}: {s:.2%}" for n, s in share.head(25).items()))
    print(f"not merged: {share[~share.index.isin(merged)].sum():.1%} of the scene pixels")


def concept_lut(device):
    """ADE20K class id -> scene concept index (len(SCENE) = no concept)."""
    names = json.loads((CACHE / "ade_names.json").read_text())
    lut = torch.full((256,), len(SCENE), dtype=torch.long, device=device)
    for i, classes in enumerate(SCENE.values()):
        for name in classes:
            lut[names.index(name)] = i
    return lut


def concept_masks(ade, boxes, lut):
    """Binary masks of all CONCEPTS on the 448x448 input: float [B, 448 * 448, len(CONCEPTS)]."""
    masks = torch.zeros(len(ade), SIZE, SIZE, len(CONCEPTS), device=ade.device)
    masks[..., :len(SCENE)] = F.one_hot(lut[ade.long()], len(SCENE) + 1)[..., :len(SCENE)]
    for b, image_boxes in enumerate(boxes):
        for x, y, w, h in image_boxes:  # MegaDetector: relative [x, y, width, height]
            masks[b, round(y * SIZE):round((y + h) * SIZE), round(x * SIZE):round((x + w) * SIZE), ANIMAL] = 1
    masks[:, :STRIP_TOP, :, STRIP] = 1
    masks[:, STRIP_BOTTOM:, :, STRIP] = 1
    return masks.flatten(1, 2)


# --------------------------------------------------------------------------------------------- dissection


def load_trunk(name, cfg, device):
    """conv1 ... layer4 of the fine-tuned model or of the ImageNet ResNet-50 it was fine-tuned from."""
    model = load_model(cfg, device) if name == "finetuned" else load_imagenet_backbone(cfg, device)
    trunk = torch.nn.Sequential(*list(model.children())[:8]).eval()
    assert len(trunk[6]) == 6 and len(trunk[7]) == 3  # layer3, layer4 of a ResNet-50
    return trunk


def activations(trunk, images):
    """{"layer3": [B, 1024, 28, 28], "layer4": [B, 2048, 14, 14]} for uint8 images [B, 3, 448, 448]."""
    x = normalize(images)
    for i, block in enumerate(trunk):
        x = block(x)
        if i == 6:
            layer3 = x
    return {"layer3": layer3, "layer4": x}


def batches(images, device, desc):
    """(first row, uint8 batch on the device), one batch at a time, with a progress bar."""
    for first in tqdm(range(0, len(images), BATCH), desc=desc, mininterval=10):
        yield first, torch.from_numpy(np.array(images[first:first + BATCH])).to(device)


@torch.no_grad()
def thresholds(trunk, images, device):
    """T_k per unit and layer: the k-th largest activation, k = QUANTILE x all images and positions."""
    top = {}
    for _, batch in batches(images, device, "thresholds"):
        for layer, act in activations(trunk, batch).items():
            k = round(QUANTILE * len(images) * act.shape[2] * act.shape[3])
            values = act.transpose(0, 1).flatten(1)
            if layer in top:
                values = torch.cat([top[layer], values], 1)
            top[layer] = values.topk(min(k, values.shape[1]), dim=1).values
    return {layer: values[:, -1] for layer, values in top.items()}


@torch.no_grad()
def overlaps(trunk, thresh, images, ade, boxes, lut, device):
    """Summed intersections |M_k & L_c| and mask sizes |M_k|, |L_c| over the probe set, plus per-image maxima."""
    stats = {layer: dict(inter=0, unit_px=0, image_max=[], image_argmax=[]) for layer in LAYERS}
    concept_px = 0
    for first, batch in batches(images, device, "IoU"):
        labels = torch.from_numpy(np.array(ade[first:first + len(batch)])).to(device)
        masks = concept_masks(labels, boxes[first:first + len(batch)], lut)
        concept_px = concept_px + masks.sum((0, 1)).double()
        for layer, act in activations(trunk, batch).items():
            s = stats[layer]
            peak, where = act.flatten(2).max(2)
            s["image_max"].append(peak.cpu())
            s["image_argmax"].append(where.short().cpu())
            inter, unit_px = [], []
            for c in range(0, act.shape[1], UNIT_CHUNK):
                up = F.interpolate(act[:, c:c + UNIT_CHUNK], size=(SIZE, SIZE), mode="bilinear", align_corners=False)
                up = up.gt_(thresh[layer][c:c + UNIT_CHUNK, None, None]).flatten(2)  # 0/1, in place
                inter.append(torch.bmm(up, masks).sum(0).double())
                unit_px.append(up.sum((0, 2)).double())
            s["inter"] = s["inter"] + torch.cat(inter)
            s["unit_px"] = s["unit_px"] + torch.cat(unit_px)
    for s in stats.values():
        s["image_max"] = torch.cat(s["image_max"])
        s["image_argmax"] = torch.cat(s["image_argmax"])
    return stats, concept_px


def unit_table(model_name, layer, thresh, s, concept_px, n_pixels):
    iou = (s["inter"] / (s["unit_px"][:, None] + concept_px[None] - s["inter"])).cpu().numpy()
    best = iou.argmax(1)
    t = pd.DataFrame({
        "model": model_name,
        "layer": layer,
        "unit": np.arange(len(iou)),
        "threshold": thresh.cpu().numpy(),
        "mask_share": (s["unit_px"] / n_pixels).cpu().numpy(),
        "concept": np.array(CONCEPTS)[best],
        "iou": iou[np.arange(len(iou)), best],
    })
    # dead units (top-0.5 % threshold 0) are never labelled
    t["label"] = t["concept"].where((t["iou"] > IOU_MIN) & (t["threshold"] > 0), "")
    for i, concept in enumerate(CONCEPTS):
        t[f"iou_{concept.replace(' ', '_')}"] = iou[:, i]
    return t


def top_images(model_name, layer, s, grid, probe):
    """The TOP images with the highest maximum activation per unit and where in the image that maximum is."""
    values, rows = s["image_max"].topk(TOP, dim=0)  # [TOP, units]
    cells = s["image_argmax"].gather(0, rows).long()
    units = np.tile(np.arange(values.shape[1]), TOP)
    gy, gx = (cells // grid).numpy().ravel(), (cells % grid).numpy().ravel()
    return pd.DataFrame({
        "model": model_name,
        "layer": layer,
        "unit": units,
        "rank": np.repeat(np.arange(1, TOP + 1), values.shape[1]),
        "image_id": probe["image_id"].to_numpy()[rows.numpy().ravel()],
        "activation": values.numpy().ravel(),
        "row": gy, "col": gx,  # cell of the maximum in the layer's grid
        "y": ((gy + 0.5) * SIZE / grid).round().astype(int), "x": ((gx + 0.5) * SIZE / grid).round().astype(int),
    }).sort_values(["unit", "rank"])


def dissect_units(cfg):
    device = get_device(cfg)
    probe, images = load_probe()
    ade = np.load(CACHE / "ade.npy", mmap_mode="r")
    lut = concept_lut(device)
    n_pixels = len(images) * SIZE * SIZE
    units, tops = [], []
    for model_name in MODELS:
        trunk = load_trunk(model_name, cfg, device)
        thresh = thresholds(trunk, images, device)
        stats, concept_px = overlaps(trunk, thresh, images, ade, probe["boxes"].tolist(), lut, device)
        for layer, s in stats.items():
            units.append(unit_table(model_name, layer, thresh[layer], s, concept_px, n_pixels))
            tops.append(top_images(model_name, layer, s, {"layer3": 28, "layer4": 14}[layer], probe))
            np.savez(CACHE / f"{model_name}_{layer}_image_max.npz",
                     max=s["image_max"].numpy(), argmax=s["image_argmax"].numpy())
        del trunk
        torch.cuda.empty_cache()

    units, tops = pd.concat(units, ignore_index=True), pd.concat(tops, ignore_index=True)
    # camera-specific units: how many cameras the 12 top images come from
    cameras = tops.assign(location=tops["image_id"].map(probe.set_index("image_id")["location"]))
    cameras = cameras.groupby(["model", "layer", "unit"])["location"].nunique().rename("top12_cameras")
    units = units.join(cameras, on=["model", "layer", "unit"])
    units.to_parquet(RESULTS / "dissection_units.parquet", index=False)
    tops.to_parquet(RESULTS / "dissection_top12.parquet", index=False)

    # concept sizes: IoU can never exceed (unit mask) / (concept mask) when the concept is the larger one
    share = (concept_px / n_pixels).cpu().numpy()
    mask_share = units.loc[units["threshold"] > 0, "mask_share"].median()
    concepts = pd.DataFrame({
        "concept": CONCEPTS,
        "ade_classes": [" | ".join(SCENE[c]) for c in SCENE] + ["MegaDetector box, conf >= 0.8", "top 3 % + bottom 5 %"],
        "pixel_share": share,
        "max_iou_for_median_unit_mask": np.minimum(share, mask_share) / np.maximum(share, mask_share),
    })
    concepts.round(4).to_csv(RESULTS / "dissection_concepts.csv", index=False)

    # per model, layer and concept: labelled units (paper rule), units whose best concept it is (any IoU), best IoU
    live = units[units["threshold"] > 0]
    summary = pd.DataFrame({
        "labelled": units[units["label"] != ""].groupby(["model", "layer", "label"]).size(),
        "best_concept": live.groupby(["model", "layer", "concept"]).size(),
        "max_iou": live.groupby(["model", "layer", "concept"])["iou"].max(),
    }).reindex(pd.MultiIndex.from_product([MODELS, LAYERS, CONCEPTS], names=["model", "layer", "concept"]))
    summary = summary.fillna({"labelled": 0, "best_concept": 0}).astype({"labelled": int, "best_concept": int})
    summary.round(4).to_csv(RESULTS / "dissection_summary.csv")

    print(concepts.round(4).to_string(index=False))
    print(f"median unit mask: {mask_share:.4%} of the pixels; dead units: {(units['threshold'] <= 0).sum()}")
    for column in ["labelled", "best_concept"]:
        print(f"\n{column}:\n{summary[column].unstack(['model', 'layer']).reindex(CONCEPTS).to_string()}")


# --------------------------------------------------------------------------------------------- contact sheets


def font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def thumbnail(image, mask):
    """Input image with the unit's above-threshold region bright and outlined, the rest dimmed (as in the paper)."""
    img = Image.fromarray(np.ascontiguousarray(image.transpose(1, 2, 0))).resize((THUMB, THUMB), Image.LANCZOS)
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((THUMB, THUMB), Image.BILINEAR)) > 127
    out = np.asarray(img).astype(float) * np.where(small, 1.0, 0.3)[..., None]
    inner = small.copy()
    inner[1:-1, 1:-1] = small[1:-1, 1:-1] & small[:-2, 1:-1] & small[2:, 1:-1] & small[1:-1, :-2] & small[1:-1, 2:]
    inner[[0, -1]] = False
    inner[:, [0, -1]] = False
    out[small & ~inner] = (255, 214, 0)
    return Image.fromarray(out.astype(np.uint8))


@torch.no_grad()
def contact_sheets(cfg):
    device = get_device(cfg)
    probe, images = load_probe()
    row_of = pd.Series(probe.index, index=probe["image_id"])
    units = pd.read_parquet(RESULTS / "dissection_units.parquet")
    tops = pd.read_parquet(RESULTS / "dissection_top12.parquet")
    FIGURES.mkdir(parents=True, exist_ok=True)
    gap, group_gap, left, head, title_h = 4, 18, 150, 24, 44
    width = left + SHEET_UNITS * (SHEET_IMAGES * (THUMB + gap) - gap) + (SHEET_UNITS - 1) * group_gap + 10
    height = title_h + len(CONCEPTS) * (head + THUMB + 14)

    for model_name in MODELS:
        trunk = load_trunk(model_name, cfg, device)
        for layer in LAYERS:
            u = units[(units["model"] == model_name) & (units["layer"] == layer) & (units["threshold"] > 0)]
            # per concept: units whose best concept it is, labelled ones first, by IoU
            picks = {c: u[u["concept"] == c].assign(labelled=lambda d: d["label"] != "")
                     .sort_values(["labelled", "iou"], ascending=False).head(SHEET_UNITS) for c in CONCEPTS}
            t = tops[(tops["model"] == model_name) & (tops["layer"] == layer) & (tops["rank"] <= SHEET_IMAGES)]
            needed = sorted({row_of[i] for p in picks.values() for i in t[t["unit"].isin(p["unit"])]["image_id"]})
            acts = {}
            for first in range(0, len(needed), BATCH):
                rows = needed[first:first + BATCH]
                batch = torch.from_numpy(np.stack([images[r] for r in rows])).to(device)
                act = activations(trunk, batch)[layer]
                acts.update({r: act[i] for i, r in enumerate(rows)})

            sheet = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((10, 10), f"{model_name} {layer}: best units per concept, top-{SHEET_IMAGES} probe images each "
                      f"(bright + yellow outline = above the unit's top-0.5 % threshold)", fill="black", font=font(18))
            n_labelled = u[u["label"] != ""].groupby("label").size()
            for ci, concept in enumerate(CONCEPTS):
                y0 = title_h + ci * (head + THUMB + 14)
                draw.text((10, y0 + head + THUMB // 2 - 20), concept, fill="black", font=font(18))
                draw.text((10, y0 + head + THUMB // 2 + 4), f"{n_labelled.get(concept, 0)} units labelled",
                          fill="#555555", font=font(13))
                for ui, (_, unit) in enumerate(picks[concept].iterrows()):
                    x0 = left + ui * (SHEET_IMAGES * (THUMB + gap) - gap + group_gap)
                    note = "" if unit["label"] else "  (not labelled)"
                    draw.text((x0, y0 + 4), f"unit {unit['unit']}   IoU {unit['iou']:.3f}{note}",
                              fill="black" if unit["label"] else "#b03030", font=font(14))
                    ut = t[t["unit"] == unit["unit"]].sort_values("rank")
                    threshold = unit["threshold"]
                    for ii, image_id in enumerate(ut["image_id"]):
                        r = row_of[image_id]
                        up = F.interpolate(acts[r][None, unit["unit"]:unit["unit"] + 1], size=(SIZE, SIZE),
                                           mode="bilinear", align_corners=False)[0, 0]
                        sheet.paste(thumbnail(images[r], (up > threshold).cpu().numpy()),
                                    (x0 + ii * (THUMB + gap), y0 + head))
            path = FIGURES / f"{model_name}_{layer}.jpg"
            sheet.save(path, quality=85)
            print(f"wrote {path}")
        del trunk
        torch.cuda.empty_cache()


if __name__ == "__main__":
    steps = sys.argv[1:] or ["probe", "segment", "units", "sheets"]
    config = load_config()
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for step in steps:
        {"probe": build_probe, "segment": lambda _: segment(), "units": dissect_units,
         "sheets": contact_sheets}[step](config)
