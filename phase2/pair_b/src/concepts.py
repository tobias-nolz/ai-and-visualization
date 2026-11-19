"""Which scenery the votes come from: concept profiles of the layer4 units, concept split of the class votes, and
switching off the scenery units. Follow-up of dissect.py (whose IoU rule cannot label large concepts like tree).

    python phase2/pair_b/src/concepts.py cameras      # habitats of the val cameras (Phase 1 openness method)
    python phase2/pair_b/src/concepts.py probe        # dissection probe set + val / id_val frames, SegFormer labels
    python phase2/pair_b/src/concepts.py backgrounds  # concept maps of the 444 composite backgrounds
    python phase2/pair_b/src/concepts.py profiles     # unit profiles + vote split on the real probe images
    python phase2/pair_b/src/concepts.py votes        # vote split of the 179,820 composites
    python phase2/pair_b/src/concepts.py ablation     # switch off tree/plant and ground units
    python phase2/pair_b/src/concepts.py              # all steps (~8 min on an RTX 5080, ~4 GB GPU memory)

Concepts. Every pixel of the 448 x 448 model input gets exactly one of PARTS: the timestamp strips (top 3 %, bottom
5 %), else the animal (MegaDetector box with conf >= 0.8 on real frames; the cut-out's alpha on composites), else the
SegFormer-B2 ADE20K class merged as in dissect.py (tree, plant, grass, ground, rock, sky, water), else "other"
(ADE classes not merged, e.g. wall, fence, animal pixels outside a box). Averaged over the 32 x 32 pixels of a layer4
cell this gives weights w_c(i, j) >= 0 with sum_c w_c(i, j) = 1 on the 14 x 14 grid.

Unit profiles (layer4, both models, real probe images). share_kc = sum_n,ij A_k(n, i, j) w_c(n, i, j) /
sum_n,ij A_k(n, i, j): the part of unit k's total activation that lies on concept c (cell-level, so a cell's
activation is credited to the concepts in proportion to their area in that cell). lift_kc = share_kc / area_c with
area_c the concept's share of all probe cells. lift_within_kc divides instead by the area the unit would hit if its
activation were spread evenly inside each image (sum_n a_nk area_nc / sum_n a_nk, a_nk = total activation of k in
image n), i.e. it ignores which images the unit fires on. The label is described at LABEL_MIN_EXCESS.

Vote split. A class logit is the mean of its class activation map over the cells plus the bias, so with the weights
above it splits exactly into one part per concept: z_c = sum_concepts 1/196 sum_ij w(i, j) CAM_c(i, j) + b_c. Done for
the true and the predicted class of the fine-tuned model and the ImageNet linear probe (CAMs and effective weights as
in evidence_split.py), on the real probe images and on the composites. Composite backgrounds: SegFormer on the frame
itself by day; for greyscale ("night") frames the per-pixel majority label of the N_REF nearest daytime frames of the
same camera (one per sequence first; well-lit greyscale frames for cameras that never take colour frames). The pasted
animal gets weight alpha, the background concepts (1 - alpha).
As in evidence_split.py, a layer4 cell sees far beyond its own 32 x 32 patch: a concept part is the evidence
located on that concept's cells, not evidence computed from its pixels only.

Ablation. Logits are linear in the pooled layer4 features, logits = W f + b. A unit is switched off by setting its
pooled feature to 0 (the unit is silent everywhere). Sets: units labelled tree or plant, units labelled ground, and
(dose-response) the TOP_K units with the largest excess share on tree + plant or on ground; each compared with the
same number of random live units (uniform, and matched on mean activation), RANDOM_DRAWS draws. For the fixed sets
also the composite outcome before -> after. For the linear probe, 0 is far below the training mean of most features
(the probe standardises them), so its random controls already cost much accuracy.

Results (small, float32) go to phase2/pair_b/results/concepts_*, caches (probe images, label maps, per-composite
splits) to outputs/concepts/.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import dissect as D
import evidence_split as E
from common import ROOT, load_config, resolve
from common.data import get_transform, load_metadata, normalize
from common.evaluate import load_probe, wilds_metrics
from common.model import get_device, load_model

sys.path.insert(0, str(ROOT / "phase1" / "pair_b" / "src"))
import measure_background as MB  # noqa: E402
from composite import FOREST_T, OPEN_T, SPECIES_GROUPS, camera_habitats, load_cutouts  # noqa: E402

PAIR_B = Path(__file__).resolve().parents[1]
RESULTS = PAIR_B / "results"
PHASE1 = ROOT / "phase1" / "pair_b" / "results"
CACHE = ROOT / "outputs" / "concepts"
SIZE, GRID = D.SIZE, 14

# probe set: the dissection probe (test / id_test) plus val / id_val frames, same rules and per-camera cap
PER_HABITAT = 2000             # frames per habitat (dissection + added), half of them empty where possible
ADD_SPLITS = ["val", "id_val"]

# concept partition of every pixel
PARTS = [*D.SCENE, "animal", "info strip", "other"]
ANIMAL, STRIP, OTHER = PARTS.index("animal"), PARTS.index("info strip"), PARTS.index("other")

# composite backgrounds
N_REF = 5                      # daytime frames of the same camera whose majority label is used for a night frame
MAX_REF_LOADS = 300            # stop looking for usable daytime frames after this many candidates

# unit labels: see label_units()
LABEL_MIN_EXCESS = 0.10        # strongest concept must hold >= 10 percentage points more activation than its area
LABEL_MIN_LIFT = 1.25          # ... and at least 1.25 x its area share

# ablation
SETS = {"tree/plant": ["tree", "plant"], "ground": ["ground"]}
TOP_K = [64, 256]              # dose-response: the k units with the largest excess share on the set's concepts
RANDOM_DRAWS = 50
OUTCOMES = ["exact species", "same group", "other group", "empty"]

MODELS = {"finetuned": "", "imagenet": "probe_"}  # model name -> column prefix in evidence_split.py
BATCH = 32
SEGFORMER_BATCH = 8            # 16 frames need ~9 GB (upsampled 150-class logits)


# --------------------------------------------------------------------------------------------- val cameras


def val_cameras(cfg):
    """Openness and habitat of the val cameras with the Phase 1 method (measure_background.py, daytime frames).

    camera_openness.csv covers the train / test cameras only. As in camera_habitats(), cameras without a daytime
    score (fewer than 3 sequences with an openness) are left without habitat.
    """
    meta = pd.read_csv(resolve(cfg["paths"]["data_dir"]) / "metadata.csv")
    locations = sorted(meta.loc[meta["split"] == "val", "location"].unique())
    assert not set(locations) & set(camera_habitats()["location"])
    frames = MB.sample_frames(meta[meta["location"].isin(locations)])
    scores = MB.aggregate(MB.infer(frames), locations)
    scores["sample_policy"] = "daytime"
    scores["habitat"] = np.select([scores["openness"] > OPEN_T, scores["openness"] < FOREST_T], ["open", "forest"],
                                  "mixed")
    scores.loc[scores["openness"].isna(), "habitat"] = ""
    scores.to_csv(RESULTS / "concepts_val_cameras.csv", index=False)
    print(scores[["location", "openness", "sequences", "habitat"]].round(3).to_string(index=False))
    print(scores["habitat"].value_counts().to_dict())


def habitats():
    """camera_habitats() of Phase 1 plus the val cameras: columns location, openness, habitat."""
    val = pd.read_csv(RESULTS / "concepts_val_cameras.csv")
    val = val.loc[val["habitat"].fillna("") != "", ["location", "openness", "habitat"]]
    return pd.concat([camera_habitats(), val], ignore_index=True)


# --------------------------------------------------------------------------------------------- probe set


def candidates(cfg, splits):
    """dissect.candidates() for other splits, with the val cameras' habitats."""
    meta = load_metadata(cfg)
    meta = meta[meta["split"].isin(splits)]
    cams = habitats().set_index("location")
    meta = meta[meta["location"].isin(cams.index)].copy()
    meta["habitat"] = meta["location"].map(cams["habitat"])
    meta["openness"] = meta["location"].map(cams["openness"])

    detections = D.load_detections(cfg)
    meta = meta[meta["image_id"].isin(detections.keys())].copy()
    dets = [detections[i] for i in meta["image_id"]]
    meta["boxes"] = [[d["bbox"] for d in ds if d["category"] == "1" and d["conf"] >= D.MD_CONF] for ds in dets]
    max_conf = np.array([max((d["conf"] for d in ds), default=0.0) for ds in dets])
    labelled_empty = (meta["y"] == 0).to_numpy()
    has_box = meta["boxes"].map(len).to_numpy() > 0
    meta["stratum"] = np.select(
        [~labelled_empty & has_box, labelled_empty & (max_conf < D.EMPTY_MAX_CONF)], ["animal", "empty"], ""
    )
    return meta[meta["stratum"] != ""]


def top_up(pool, n, load, rng, counts):
    """dissect.round_robin(), but each camera starts with `counts[location]` frames it already has."""
    taken = {}
    if n <= 0 or pool.empty:
        return taken
    per_camera = {loc: counts.get(loc, 0) for loc in pool["location"].unique()}
    for _, tier in pool.groupby("tier"):
        cameras = [list(g.index) for _, g in tier.groupby("location")]
        cameras = [cameras[i] for i in rng.permutation(len(cameras))]
        cameras = [q for q in cameras if per_camera[pool.at[q[0], "location"]] < D.PER_CAMERA]
        while cameras and len(taken) < n:
            heads = [queue.pop(0) for queue in cameras]
            for index, image in zip(heads, load(pool.loc[heads, "filename"])):
                location = pool.at[index, "location"]
                if len(taken) < n and per_camera[location] < D.PER_CAMERA and D.usable(image):
                    taken[index] = image
                    per_camera[location] += 1
            cameras = [q for q in cameras if q and per_camera[pool.at[q[0], "location"]] < D.PER_CAMERA]
    return taken


@torch.inference_mode()
def segformer_labels(images, segformer):
    """ADE20K label maps uint8 [B, 448, 448] of uint8 inputs [B, 3, 448, 448] (numpy), as in dissect.segment()."""
    processor, model, device = segformer
    labels = np.full((len(images), SIZE, SIZE), 255, np.uint8)
    bands = [MB.read_image(Image.fromarray(np.ascontiguousarray(image.transpose(1, 2, 0)))) for image in images]
    inputs = processor(images=bands, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(**inputs).logits
    logits = F.interpolate(logits.float(), size=(D.STRIP_BOTTOM - D.STRIP_TOP, SIZE), mode="bilinear",
                           align_corners=False)
    labels[:, D.STRIP_TOP:D.STRIP_BOTTOM] = logits.argmax(1).byte().cpu().numpy()
    return labels


def load_segformer():
    segformer = D.load_segformer()
    names = [segformer[1].config.id2label[i].strip() for i in range(len(segformer[1].config.id2label))]
    assert names == json.loads((D.CACHE / "ade_names.json").read_text())
    return segformer


def build_probe(cfg):
    """The dissection probe set plus val / id_val frames picked with the same rules; cache inputs and ADE labels.

    The dissection frames stay (rows 0 ... 3979, same order, their cached SegFormer labels). New frames are added per
    habitat up to PER_HABITAT frames, half of them empty (animal frames fill what the empty ones leave, as in
    dissect.build_probe()), one frame per sequence first, round robin over cameras; the cap of PER_CAMERA frames per
    camera and stratum counts the frames a camera already has. Unlike dissect.py, a habitat that runs short (open)
    does not pass the rest on, so that the other habitats do not grow at its expense.
    """
    old = pd.read_parquet(RESULTS / "dissection_probe.parquet")
    old_images = np.load(D.CACHE / "images.npy", mmap_mode="r")
    old_ade = np.load(D.CACHE / "ade.npy", mmap_mode="r")
    assert len(old) == len(old_images) == len(old_ade)

    rng = np.random.default_rng(cfg["seed"])
    pool = candidates(cfg, ADD_SPLITS).sample(frac=1, random_state=cfg["seed"])
    pool["tier"] = pool.duplicated("seq_id").astype(int)
    image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
    transform = get_transform(cfg)
    workers = ThreadPoolExecutor(8)

    def load(filenames):
        return list(workers.map(lambda f: D.load_input(image_dir / f, transform), filenames))

    chosen = {}
    for habitat in D.HABITATS:
        quota = PER_HABITAT
        have = old[old["habitat"] == habitat]
        here = pool[pool["habitat"] == habitat]
        added = {}
        for stratum in ["empty", "animal"]:
            mine = have[have["stratum"] == stratum]
            target = quota // 2 if stratum == "empty" else quota - (have["stratum"] == "empty").sum() - len(added)
            counts = mine.groupby("location").size().to_dict()
            new = top_up(here[here["stratum"] == stratum], target - len(mine), load, rng, counts)
            added |= new
            print(f"{habitat} {stratum}: {len(mine)} dissection + {len(new)} added frames")
        chosen |= added

    new = pool.loc[list(chosen)].sort_values(["habitat", "stratum", "location", "datetime"])
    new["boxes"] = new["boxes"].map(json.dumps)
    columns = list(old.columns)
    probe = pd.concat([old.assign(source="dissection"), new[columns].assign(source="added")], ignore_index=True)

    images = np.lib.format.open_memmap(CACHE / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(probe), 3, SIZE, SIZE))
    ade = np.lib.format.open_memmap(CACHE / "ade.npy", mode="w+", dtype=np.uint8, shape=(len(probe), SIZE, SIZE))
    for first in range(0, len(old), 500):
        rows = slice(first, min(first + 500, len(old)))
        images[rows] = old_images[rows]
        ade[rows] = old_ade[rows]
    for row, index in enumerate(new.index, len(old)):
        images[row] = chosen[index].numpy()
    segformer = load_segformer()
    for first in tqdm(range(len(old), len(probe), SEGFORMER_BATCH), desc="SegFormer (added frames)"):
        ade[first:first + SEGFORMER_BATCH] = segformer_labels(images[first:first + SEGFORMER_BATCH], segformer)
    images.flush()
    ade.flush()

    probe.to_parquet(RESULTS / "concepts_probe.parquet", index=False, compression="zstd")
    summary = probe.groupby(["habitat", "stratum", "source"]).agg(
        frames=("image_id", "size"), cameras=("location", "nunique")).unstack("source", fill_value=0)
    print(summary.to_string())
    print(f"{len(probe)} frames from {probe['location'].nunique()} cameras;", probe["split"].value_counts().to_dict())


def load_probe_set():
    probe = pd.read_parquet(RESULTS / "concepts_probe.parquet")
    probe["boxes"] = probe["boxes"].map(json.loads)
    return probe, np.load(CACHE / "images.npy", mmap_mode="r"), np.load(CACHE / "ade.npy", mmap_mode="r")


# --------------------------------------------------------------------------------------------- concept maps


def concept_lut(device):
    """ADE20K class id -> index in PARTS (OTHER for classes not merged, and for 255)."""
    lut = D.concept_lut(device)
    return torch.where(lut == len(D.SCENE), OTHER, lut)


def partition(ade, lut, boxes=None):
    """Index in PARTS of every pixel [B, 448, 448]: strips, then animal boxes, then the merged ADE class or other."""
    index = lut[ade.long()]
    for b, image_boxes in enumerate(boxes or []):
        for x, y, w, h in image_boxes:  # MegaDetector: relative [x, y, width, height]
            index[b, round(y * SIZE):round((y + h) * SIZE), round(x * SIZE):round((x + w) * SIZE)] = ANIMAL
    index[:, :D.STRIP_TOP] = STRIP
    index[:, D.STRIP_BOTTOM:] = STRIP
    return index


def cell_weights(index, alpha=None):
    """Weights w [B, len(PARTS), 14, 14] (sum 1 per cell) from a partition [B, 448, 448]; alpha [448, 448] = animal."""
    onehot = F.one_hot(index.long(), len(PARTS)).permute(0, 3, 1, 2).float()
    if alpha is not None:
        onehot = onehot * (1 - alpha)
        onehot[:, ANIMAL] += alpha
    return F.avg_pool2d(onehot, SIZE // GRID)


# --------------------------------------------------------------------------------------------- backgrounds


def reference_candidates(meta, row):
    """Frames of the background's camera, daytime hours and one per sequence first, then nearest in time."""
    same = meta[(meta["location"] == row["location"]) & (meta["seq_id"] != row["seq_id"])].copy()
    same["gap"] = (same["time"] - row["time"]).abs()
    same["night_hour"] = ~same["time"].dt.hour.between(7, 18)
    same["repeat"] = same.sort_values("gap").duplicated("seq_id")
    return same.sort_values(["night_hour", "repeat", "gap"])


def background_maps(cfg):
    """Concept partitions of the 444 composite backgrounds: own SegFormer labels by day, daytime reference at night.

    For every background (day and night) the reference is built from the N_REF nearest usable daytime frames of the
    same camera (other sequences, same image size): per pixel the most frequent concept, ties to the nearer frame. On
    day backgrounds, the agreement between own labels and the reference shows how well a reference stands in for the
    frame; on night backgrounds, the agreement shows how far SegFormer on the infrared frame is off.

    Phase 1's "night" means greyscale (is_infrared). Some cameras take greyscale frames by day as well and have no
    colour frame at all; for them the reference uses well-lit greyscale frames taken between 7:00 and 18:59
    (source "greyscale daytime reference").
    """
    device = get_device(cfg)
    bg = pd.read_parquet(PHASE1 / "backgrounds.parquet")
    meta = load_metadata(cfg)
    meta["time"] = pd.to_datetime(meta["datetime"])
    bg = bg.merge(meta[["image_id", "seq_id", "time"]], left_on="id", right_on="image_id", how="left")
    assert bg["seq_id"].notna().all()
    image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
    transform = get_transform(cfg)
    workers = ThreadPoolExecutor(8)

    def size(filename):
        with Image.open(image_dir / filename) as image:
            return image.size

    def lit(image):
        """Enough light and contrast (dissect.usable() without the colour test)."""
        grey = image[:, D.STRIP_TOP:D.STRIP_BOTTOM].float().mean(0) / 255
        return grey.mean() >= 0.15 and grey.std() >= 0.04

    meta = meta[meta["location"].isin(bg["location"])]
    segformer = load_segformer()
    lut = concept_lut(device)

    def concepts_of(images):
        ade = torch.from_numpy(segformer_labels(images, segformer)).to(device)
        return partition(ade, lut)

    own_all = np.zeros((len(bg), SIZE, SIZE), np.uint8)
    ref_all = np.zeros((len(bg), SIZE, SIZE), np.uint8)
    rows = []
    for i, row in tqdm(bg.iterrows(), total=len(bg), desc="backgrounds"):
        frame = D.load_input(image_dir / row["filename"], transform)[None]
        own = concepts_of(frame.numpy())[0]
        pool = reference_candidates(meta, row)
        colour, grey = ([], []), ([], [])      # (images, metadata rows)
        for first in range(0, min(len(pool), MAX_REF_LOADS), 16):
            chunk = pool.iloc[first:first + 16]
            chunk = chunk[[size(f) == size(row["filename"]) for f in chunk["filename"]]]
            images = list(workers.map(lambda f: D.load_input(image_dir / f, transform), chunk["filename"]))
            for (_, ref), image in zip(chunk.iterrows(), images):
                kind = colour if D.usable(image) else grey if lit(image) and not ref["night_hour"] else None
                if kind is not None and len(kind[0]) < N_REF:
                    kind[0].append(image)
                    kind[1].append(ref)
            if len(colour[0]) == N_REF:
                break
        refs, used = colour if colour[0] or row["light"] == "day" else grey
        votes = torch.zeros(len(PARTS), SIZE, SIZE, device=device)
        if refs:
            maps = concepts_of(torch.stack(refs).numpy())
            for rank, m in enumerate(maps):
                votes += F.one_hot(m.long(), len(PARTS)).permute(2, 0, 1) * (1 + 1e-3 * (N_REF - rank))
        ref = votes.argmax(0) if refs else own
        scene = slice(D.STRIP_TOP, D.STRIP_BOTTOM)
        used = pd.DataFrame(used)
        rows.append({
            "bg_id": row["bg_id"],
            "source": "own frame" if row["light"] == "day" or not refs
            else "daytime reference" if colour[0] else "greyscale daytime reference",
            "n_ref": len(refs),
            "ref_ids": "|".join(used["image_id"]) if refs else "",
            "ref_gap_days": used["gap"].dt.total_seconds().median() / 86400 if refs else np.nan,
            "own_ref_agreement": (own[scene] == ref[scene]).float().mean().item() if refs else np.nan,
        })
        own_all[i] = own.cpu().numpy()
        ref_all[i] = ref.cpu().numpy()

    table = bg[["bg_id", "location", "light", "habitat", "infrared"]].merge(pd.DataFrame(rows), on="bg_id")
    used = np.where((table["source"] == "own frame").to_numpy()[:, None, None], own_all, ref_all)
    np.save(CACHE / "bg_own.npy", own_all)
    np.save(CACHE / "bg_ref.npy", ref_all)
    np.save(CACHE / "bg_concepts.npy", used)
    area = np.stack([(used == c).mean((1, 2)) for c in range(len(PARTS))], 1)
    for c, name in enumerate(PARTS):
        table[f"area_{name.replace(' ', '_')}"] = area[:, c].astype(np.float32)
    table["ref_gap_days"] = table["ref_gap_days"].astype(np.float32)
    table["own_ref_agreement"] = table["own_ref_agreement"].astype(np.float32)
    table.to_parquet(RESULTS / "concepts_backgrounds.parquet", index=False, compression="zstd")

    print(table.groupby(["light", "source"]).agg(
        n=("bg_id", "size"), n_ref=("n_ref", "mean"), gap_days=("ref_gap_days", "median"),
        agreement=("own_ref_agreement", "median")).round(3).to_string())
    areas = [c for c in table.columns if c.startswith("area_")]
    print(table.groupby(["habitat", "light"])[areas].mean().round(3).to_string())


# --------------------------------------------------------------------------------------------- unit profiles


@torch.no_grad()
def profiles(cfg):
    """Unit profiles of layer4 (both models) and the concept split of the true / predicted class on the probe set."""
    device = get_device(cfg)
    probe, images, ade = load_probe_set()
    lut = concept_lut(device)
    heads = E.load_heads(cfg, device, probe=True)
    y_all = torch.tensor(probe["y"].to_numpy(), device=device)

    on = {m: torch.zeros(2048, len(PARTS), dtype=torch.float64, device=device) for m in MODELS}
    per_image = {m: [] for m in MODELS}       # total activation per image and unit [N, 2048]
    votes = {m: [] for m in MODELS}
    areas = []
    for first in tqdm(range(0, len(images), BATCH), desc="profiles", mininterval=10):
        rows = slice(first, first + BATCH)
        batch = torch.from_numpy(np.array(images[rows])).to(device)
        index = partition(torch.from_numpy(np.array(ade[rows])).to(device), lut, probe["boxes"].iloc[rows].tolist())
        w = cell_weights(index).flatten(2)                      # [B, P, 196]
        areas.append(w.mean(2).cpu())
        x = normalize(batch)
        for name, prefix in MODELS.items():
            head = heads[prefix]
            logits, features = head(x)
            a = features.flatten(2)                              # [B, 2048, 196]
            on[name] += torch.bmm(a, w.transpose(1, 2)).sum(0).double()
            per_image[name].append(a.sum(2).double().cpu())
            pred = logits.argmax(1)
            out = {"pred_y": pred, "p_pred": logits.softmax(1).max(1).values}
            for cls, classes in [("true", y_all[rows]), ("pred", pred)]:
                cam = torch.einsum("bkp,bk->bp", a, head.weight[classes])
                parts = torch.einsum("bp,bcp->bc", cam, w) / GRID**2
                out[f"{cls}_logit"] = logits.gather(1, classes[:, None])[:, 0]
                out[f"{cls}_bias"] = head.bias[classes]
                out[f"{cls}_parts"] = parts
            votes[name].append({k: v.float().cpu() for k, v in out.items()})

    areas = torch.cat(areas).double()                            # [N, P]
    area = areas.mean(0)
    units, tables = [], []
    for name in MODELS:
        img = torch.cat(per_image[name])                         # [N, 2048]
        total = img.sum(0)
        share = (on[name].cpu() / total[:, None]).numpy()
        expected = ((img.T @ areas) / total[:, None]).numpy()    # spread evenly inside each image
        t = pd.DataFrame({"model": name, "unit": np.arange(2048),
                          "mean_act": (total / (len(img) * GRID**2)).float().numpy()})
        for c, concept in enumerate(PARTS):
            key = concept.replace(" ", "_")
            t[f"share_{key}"] = share[:, c]
            t[f"lift_{key}"] = share[:, c] / area[c].item()
            t[f"lift_within_{key}"] = share[:, c] / expected[:, c]
        units.append(label_units(t, area.numpy()))

        v = {k: torch.cat([b[k] for b in votes[name]]) for k in votes[name][0]}
        table = pd.DataFrame({"image_id": probe["image_id"], "model": name, "pred_y": v["pred_y"].long().numpy(),
                              "p_pred": v["p_pred"].numpy()})
        for cls in ["true", "pred"]:
            gap = (v[f"{cls}_parts"].sum(1) + v[f"{cls}_bias"] - v[f"{cls}_logit"]).abs().max().item()
            print(f"{name} {cls}: max |sum of concept parts + bias - logit| = {gap:.1e}")
            table[f"{cls}_logit"] = v[f"{cls}_logit"].numpy()
            table[f"{cls}_bias"] = v[f"{cls}_bias"].numpy()
            for c, concept in enumerate(PARTS):
                table[f"{cls}_{concept.replace(' ', '_')}"] = v[f"{cls}_parts"][:, c].numpy()
        tables.append(table)

    units = pd.concat(units, ignore_index=True)
    floats = units.select_dtypes("float64").columns
    units[floats] = units[floats].astype(np.float32)
    units.to_parquet(RESULTS / "concepts_units.parquet", index=False, compression="zstd")
    pd.concat(tables, ignore_index=True).to_parquet(RESULTS / "concepts_probe_votes.parquet", index=False,
                                                    compression="zstd")

    # concept areas per probe image (for the vote tables)
    probe = pd.read_parquet(RESULTS / "concepts_probe.parquet")
    for c, concept in enumerate(PARTS):
        probe[f"area_{concept.replace(' ', '_')}"] = areas[:, c].float().numpy()
    probe.to_parquet(RESULTS / "concepts_probe.parquet", index=False, compression="zstd")

    print("concept area share of the probe cells:", dict(zip(PARTS, area.numpy().round(4))))
    print(units.groupby(["model", "label"]).size().unstack("model").reindex([*PARTS, "mixed", "dead"]).to_string())
    print_votes(probe_vote_summary(), "probe images")


def label_units(t, area):
    """Label = concept with the largest excess share (share - area), if it stands out, else "mixed".

    "Stands out": excess >= LABEL_MIN_EXCESS and lift >= LABEL_MIN_LIFT. The largest lift alone would favour small
    concepts (lift is capped at 1 / area: ~2.1 for tree, ~100 for water), the excess share asks which concept pulls
    the most extra activation towards itself, which is what matters for the votes.
    """
    share = t[[f"share_{c.replace(' ', '_')}" for c in PARTS]].to_numpy()
    excess = share - area[None]
    best = excess.argmax(1)
    rows = np.arange(len(t))
    t["label"] = np.array(PARTS)[best]
    t["label_excess"] = excess[rows, best]
    t["label_lift"] = share[rows, best] / area[best]
    stands_out = (t["label_excess"] >= LABEL_MIN_EXCESS) & (t["label_lift"] >= LABEL_MIN_LIFT)
    t["label"] = t["label"].where(stands_out, "mixed")
    t.loc[t["mean_act"] <= 0, "label"] = "dead"
    return t


# --------------------------------------------------------------------------------------------- vote split


def group_of_class(cfg, device):
    """Species-group index of every class [num_classes] (-1 for classes without group)."""
    groups = pd.read_csv(SPECIES_GROUPS).set_index("y")["group"]
    names = sorted(groups.unique())
    ids = torch.full((cfg["model"]["num_classes"],), -1, dtype=torch.long)
    ids[torch.from_numpy(groups.index.to_numpy())] = torch.tensor([names.index(g) for g in groups])
    return ids.to(device)


def outcome_codes(pred, y, group_of):
    """Index in OUTCOMES per composite, as in run_composites.finish() (exact, same group, empty, other group)."""
    code = torch.full_like(pred, OUTCOMES.index("other group"))
    code[pred == 0] = OUTCOMES.index("empty")
    code[(group_of[pred] == group_of[y]) & (group_of[pred] >= 0)] = OUTCOMES.index("same group")
    code[pred == y] = OUTCOMES.index("exact species")
    return code


def alpha_canvas(cutout, row, col):
    """The cut-out's alpha on the 448 x 448 frame, placed as paste() does (cf. evidence_split.pooled_mask)."""
    _, alpha = cutout
    h, w = alpha.shape[1:]
    y0 = min(max(round((row + 0.5) * E.CELL - h / 2), 0), SIZE - h)
    x0 = min(max(round((col + 0.5) * E.CELL - w / 2), 0), SIZE - w)
    canvas = torch.zeros(SIZE, SIZE)
    canvas[y0:y0 + h, x0:x0 + w] = alpha[0]
    return canvas


def part_columns(prefix):
    return [f"{prefix}_{c.replace(' ', '_')}" for c in PARTS]


def summarize_votes(frame, keys):
    """Means per group of `keys` in long format: one row per concept (plus bias and logit).

    frame: one row per image with area_<c>, pred_<c>, true_<c> (c in PARTS), pred_bias, true_bias, pred_logit,
    true_logit. Columns of the result: keys, concept, n, area, pred, true, margin (= pred - true).
    """
    concepts = [c.replace(" ", "_") for c in PARTS] + ["bias", "logit"]
    cols = [f"{p}_{c}" for p in ["pred", "true"] for c in concepts] + part_columns("area")
    means = frame.groupby(keys)[cols].mean()
    n = frame.groupby(keys).size()
    out = []
    for c, name in zip(concepts, [*PARTS, "bias", "logit"]):
        out.append(pd.DataFrame({
            "concept": name, "n": n,
            "area": means[f"area_{c}"] if f"area_{c}" in means else np.nan,
            "pred": means[f"pred_{c}"], "true": means[f"true_{c}"],
            "margin": means[f"pred_{c}"] - means[f"true_{c}"],
        }))
    out = pd.concat(out).reset_index()
    out[["area", "pred", "true", "margin"]] = out[["area", "pred", "true", "margin"]].astype(np.float32)
    return out.sort_values(keys, kind="stable")


@torch.no_grad()
def composite_votes(cfg):
    """Concept split of the true / predicted class logits of all composites (both models).

    Weights per composite: the cut-out's alpha is "animal", the background concepts get (1 - alpha), all on the pixel
    level, then averaged over each layer4 cell. Writes outputs/concepts/composite_votes.npz (parts [N, model, class,
    concept], area [N, concept], row-aligned with evidence_split.parquet) and the summary
    results/concepts_composite_votes.csv (per model, habitat, light and outcome; light "all", outcome "wrong" =
    everything but the exact species).
    """
    device = get_device(cfg)
    res = pd.read_parquet(RESULTS / "evidence_split.parquet")
    bg = pd.read_parquet(RESULTS / "concepts_backgrounds.parquet")
    n_bg = len(bg)
    assert (bg["bg_id"].to_numpy() == np.arange(n_bg)).all()
    masks = np.load(E.OUT_DIR / "masks.npz")
    assert len(res) == len(masks["mask"]) * n_bg
    cutouts = load_cutouts()
    bg_index = torch.from_numpy(np.load(CACHE / "bg_concepts.npy")).to(device)
    cams = {(m, cls): np.load(E.OUT_DIR / f"{prefix}cam_{cls}.npy", mmap_mode="r")
            for m, prefix in MODELS.items() for cls in ["true", "pred"]}

    parts = np.zeros((len(res), len(MODELS), 2, len(PARTS)), np.float32)
    area = np.zeros((len(res), len(PARTS)), np.float32)
    for p in tqdm(range(len(masks["mask"])), desc="composites", mininterval=10):
        key, row, col = (str(masks["group"][p]), int(masks["variant"][p])), int(masks["row"][p]), int(masks["col"][p])
        rows = slice(p * n_bg, (p + 1) * n_bg)
        r = res.iloc[rows]
        assert (r["group"] == key[0]).all() and (r["variant"] == key[1]).all()
        assert (r["row"] == row).all() and (r["col"] == col).all() and (r["bg_id"] == np.arange(n_bg)).all()
        alpha = alpha_canvas(cutouts[key], row, col).to(device)
        w = torch.cat([cell_weights(bg_index[c:c + 111], alpha) for c in range(0, n_bg, 111)])  # [444, P, 14, 14]
        assert (w[:, ANIMAL] - torch.from_numpy(masks["mask"][p]).to(device)).abs().max() < 1e-5
        area[rows] = w.mean((2, 3)).cpu().numpy()
        for (m, cls), cam in cams.items():
            cam = torch.from_numpy(np.array(cam[rows])).to(device)
            parts[rows, list(MODELS).index(m), ["true", "pred"].index(cls)] = \
                ((cam[:, None] * w).sum((2, 3)) / GRID**2).cpu().numpy()
    np.savez(CACHE / "composite_votes.npz", parts=parts, area=area, models=list(MODELS), classes=["true", "pred"],
             concepts=PARTS)

    group_of = group_of_class(cfg, "cpu")
    frames = []
    for mi, (m, prefix) in enumerate(MODELS.items()):
        f = pd.DataFrame({"model": m, "habitat": res["bg_id"].map(bg["habitat"]),
                          "light": res["bg_id"].map(bg["light"])})
        pred = torch.from_numpy(res[prefix + "pred_y"].to_numpy())
        f["outcome"] = np.array(OUTCOMES)[outcome_codes(pred, torch.from_numpy(res["y"].to_numpy()), group_of)]
        if m == "finetuned":
            assert (f["outcome"] == pd.read_parquet(PHASE1 / "composites.parquet")["outcome"]).all()
        for ci, cls in enumerate(["true", "pred"]):
            f[part_columns(cls)] = parts[:, mi, ci]
            f[f"{cls}_bias"] = res[f"{prefix}{cls}_bias"]
            f[f"{cls}_logit"] = res[f"{prefix}{cls}_logit"]
            gap = (parts[:, mi, ci].sum(1) + f[f"{cls}_bias"] - f[f"{cls}_logit"]).abs().max()
            print(f"{m} {cls}: max |sum of concept parts + bias - logit| = {gap:.1e}")
        f[part_columns("area")] = area
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    f = pd.concat([f, f.assign(light="all")], ignore_index=True)
    f = pd.concat([f, f[f["outcome"] != "exact species"].assign(outcome="wrong")], ignore_index=True)
    summary = summarize_votes(f, ["model", "habitat", "light", "outcome"])
    summary.to_csv(RESULTS / "concepts_composite_votes.csv", index=False, float_format="%.4g")
    print_votes(summary, "composites")


def probe_vote_summary():
    """Summary of concepts_probe_votes.parquet per model, habitat (+ "all"), stratum and correct / wrong."""
    probe = pd.read_parquet(RESULTS / "concepts_probe.parquet")
    votes = pd.read_parquet(RESULTS / "concepts_probe_votes.parquet")
    f = votes.merge(probe[["image_id", "habitat", "stratum", "y", *part_columns("area")]], on="image_id")
    f["outcome"] = np.where(f["pred_y"] == f["y"], "correct", "wrong")
    f = pd.concat([f, f.assign(habitat="all")], ignore_index=True)
    summary = summarize_votes(f, ["model", "habitat", "stratum", "outcome"])
    summary.to_csv(RESULTS / "concepts_probe_votes_summary.csv", index=False, float_format="%.4g")
    return summary


def print_votes(summary, what):
    """Wrong predictions per habitat: concept parts of the winning class and of its margin over the true class."""
    s = summary[(summary["light"] == "all") & (summary["outcome"] == "wrong")] if what == "composites" else \
        summary[(summary["stratum"] == "animal") & (summary["outcome"] == "wrong")]
    for m in MODELS:
        t = s[s["model"] == m]
        for value in ["pred", "margin"]:
            table = t.pivot(index="habitat", columns="concept", values=value)[[*PARTS, "bias", "logit"]]
            table.insert(0, "n", t.groupby("habitat")["n"].first())
            title = "winning-class logit" if value == "pred" else "winning minus true class"
            print(f"\n{what}, {m}, wrong predictions: {title}, mean part per concept (logits)")
            print(table.round(2).to_string())
        print(f"{what}, {m}, wrong: area share per concept")
        print(t.pivot(index="habitat", columns="concept", values="area")[PARTS].round(3).to_string())


# --------------------------------------------------------------------------------------------- ablation


def linear_heads(cfg, device):
    """{model: (W [182, 2048], b [182])} on the raw pooled layer4 features (linear probe: effective weights)."""
    fc = load_model(cfg, device).fc
    probe = load_probe(cfg, device)
    assert torch.equal(probe.classes.cpu(), torch.arange(cfg["model"]["num_classes"]))
    w = probe.linear.weight.detach().double() / probe.std.double()
    b = probe.linear.bias.detach().double() - w @ probe.mean.double()
    return {"finetuned": (fc.weight.detach().float(), fc.bias.detach().float()), "imagenet": (w.float(), b.float())}


def unit_sets(units, area):
    """{model: {name: unit ids}}: the labelled sets, and the top-k units by excess share (dose-response).

    area: {concept: share of the probe cells}, concept names with "_" for " ".
    """
    sets = {}
    for m, u in units.groupby("model"):
        sets[m] = {}
        for name, concepts in SETS.items():
            sets[m][name] = u.loc[u["label"].isin(concepts), "unit"].to_numpy()
            excess = sum(u[f"share_{c}"] - area[c] for c in concepts)
            for k in TOP_K:
                sets[m][f"{name} top {k}"] = u.loc[excess.nlargest(k).index, "unit"].to_numpy()
    return sets


def random_sets(units, chosen, rng, matched):
    """RANDOM_DRAWS random sets of len(chosen) live units outside `chosen`.

    matched: as many units from each decile of mean activation as `chosen` has there; when a decile has too few
    units left, the rest comes from the nearest deciles.
    """
    live = units[units["label"] != "dead"]
    others = live[~live["unit"].isin(chosen)]
    if not matched:
        return [rng.choice(others["unit"].to_numpy(), len(chosen), replace=False) for _ in range(RANDOM_DRAWS)]
    decile = pd.Series(pd.qcut(live["mean_act"], 10, labels=False).to_numpy(), index=live["unit"].to_numpy())
    need = decile.loc[chosen].value_counts().sort_index()
    pools = {d: others.loc[decile.loc[others["unit"]].to_numpy() == d, "unit"].to_numpy() for d in range(10)}
    draws = []
    for _ in range(RANDOM_DRAWS):
        picked = np.array([], dtype=np.int64)
        for d, k in need.items():
            for e in sorted(pools, key=lambda e: (abs(e - d), e)):
                left = np.setdiff1d(pools[e], picked)
                take = rng.choice(left, min(k, len(left)), replace=False)
                picked, k = np.concatenate([picked, take]), k - len(take)
                if k == 0:
                    break
        draws.append(picked)
    return draws


def macro_f1(pred, y, num_classes):
    """WILDS macro F1 (classes present in y, zero_division=0), on the GPU; matches common.evaluate.wilds_metrics."""
    tp = torch.bincount(y[pred == y], minlength=num_classes).double()
    n_pred = torch.bincount(pred, minlength=num_classes).double()
    n_true = torch.bincount(y, minlength=num_classes).double()
    f1 = torch.where(n_pred + n_true > 0, 2 * tp / (n_pred + n_true), torch.zeros_like(tp))
    return f1[n_true > 0].mean().item()


@torch.no_grad()
def ablation(cfg):
    """Switch off unit sets in the pooled features; composite outcomes per habitat and real accuracy vs. random sets.

    Columns units_off (the set, "none" = no unit switched off) and control ("none" = the set itself, "random",
    "random, matched").

    Composites: outcome shares per habitat x light (light "all" too). Real images: accuracy and WILDS macro F1 on
    test and id_test, accuracy per habitat. Random controls: RANDOM_DRAWS sets of the same size (uniform over the
    other live units, and matched on mean activation decile); mean and standard deviation over the draws.
    """
    device = get_device(cfg)
    rng = np.random.default_rng(cfg["seed"])
    n_classes = cfg["model"]["num_classes"]
    units = pd.read_parquet(RESULTS / "concepts_units.parquet")
    probe = pd.read_parquet(RESULTS / "concepts_probe.parquet")
    area = {c: probe[f"area_{c}"].mean() for c in [c.replace(" ", "_") for c in PARTS]}
    sets = unit_sets(units, area)
    heads = linear_heads(cfg, device)
    group_of = group_of_class(cfg, device)

    # composites: habitat x light cell of every row
    res = pd.read_parquet(RESULTS / "evidence_split.parquet", columns=["bg_id", "y", "pred_y", "probe_pred_y"])
    bg = pd.read_parquet(RESULTS / "concepts_backgrounds.parquet")
    cells = [(h, l) for h in D.HABITATS for l in ["day", "night"]]
    cell = torch.tensor([cells.index(k) for k in zip(bg["habitat"], bg["light"])], device=device)
    comp_cell = cell[torch.from_numpy(res["bg_id"].to_numpy()).to(device)]
    comp_y = torch.from_numpy(res["y"].to_numpy()).to(device)

    # real images: test and id_test rows of the predictions / embeddings
    preds = pd.read_parquet(resolve(cfg["paths"]["predictions"]), columns=["split", "label", "prediction", "location"])
    preds_probe = pd.read_parquet(resolve(cfg["paths"]["predictions_probe"]), columns=["split", "prediction"])
    real_rows = np.flatnonzero(preds["split"].isin(["test", "id_test"]).to_numpy())
    real = preds.iloc[real_rows].reset_index(drop=True)
    real["habitat"] = real["location"].map(habitats().set_index("location")["habitat"]).fillna("unknown")
    real_y = torch.from_numpy(real["label"].to_numpy()).to(device)
    real_groups = {(s, h): torch.from_numpy(((real["split"] == s) & ((real["habitat"] == h) | (h == "all")))
                                            .to_numpy()).to(device)
                   for s in ["test", "id_test"] for h in ["all", *D.HABITATS]}

    feature_files = {"finetuned": (E.OUT_DIR / "features.npy", resolve(cfg["paths"]["embeddings"])),
                     "imagenet": (E.OUT_DIR / "probe_features.npy", resolve(cfg["paths"]["embeddings_imagenet"]))}
    stored = {"finetuned": (res["pred_y"], preds["prediction"]),
              "imagenet": (res["probe_pred_y"], preds_probe["prediction"])}
    comp_rows, real_out, transitions = [], [], []
    for m, (W, b) in heads.items():
        comp = torch.from_numpy(np.load(feature_files[m][0])).to(device).float()
        real_f = torch.from_numpy(np.load(feature_files[m][1], mmap_mode="r")[real_rows]).to(device)
        base_comp, base_real = comp @ W.T + b, real_f @ W.T + b
        same = (base_comp.argmax(1).cpu().numpy() == stored[m][0].to_numpy()).mean()
        same_real = (base_real.argmax(1).cpu().numpy() == stored[m][1].to_numpy()[real_rows]).mean()
        print(f"{m}: predictions from the pooled features as stored: composites {same:.4%}, real {same_real:.4%}")

        def composite_codes(off):
            off = torch.as_tensor(np.asarray(off, dtype=np.int64), device=device)
            return outcome_codes((base_comp - comp[:, off] @ W[:, off].T).argmax(1), comp_y, group_of)

        def evaluate(off):
            """Composite outcome counts [cells, outcomes] and real metrics with units `off` switched off."""
            codes = composite_codes(off)
            off = torch.as_tensor(np.asarray(off, dtype=np.int64), device=device)
            pr = (base_real - real_f[:, off] @ W[:, off].T).argmax(1)
            counts = torch.bincount(comp_cell * len(OUTCOMES) + codes, minlength=len(cells) * len(OUTCOMES))
            metrics = {}
            for (s, h), mask in real_groups.items():
                metrics[s, h, "acc"] = (pr[mask] == real_y[mask]).float().mean().item()
                if h == "all":
                    metrics[s, h, "f1_macro"] = macro_f1(pr[mask], real_y[mask], n_classes)
            return counts.view(len(cells), len(OUTCOMES)).cpu().numpy(), metrics

        base_codes = composite_codes([])
        base_counts, base_metrics = evaluate([])
        check = wilds_metrics(real["label"][real["split"] == "test"],
                              base_real.argmax(1).cpu().numpy()[(real["split"] == "test").to_numpy()])
        assert abs(check["f1_macro"] - base_metrics["test", "all", "f1_macro"]) < 1e-9
        runs = [("none", "none", 0, [(base_counts, base_metrics)])]
        for name, off in sets[m].items():
            runs.append((name, "none", len(off), [evaluate(off)]))
            # composite outcome before -> after, per habitat (habitat index = cell // 2)
            k = len(OUTCOMES)
            moved = torch.bincount(((comp_cell // 2) * k + base_codes) * k + composite_codes(off),
                                   minlength=len(D.HABITATS) * k * k).view(len(D.HABITATS), k, k).cpu().numpy()
            for hi, h in enumerate(D.HABITATS):
                for a, before in enumerate(OUTCOMES):
                    for b_, after in enumerate(OUTCOMES):
                        transitions.append({"model": m, "units_off": name, "n_units": len(off), "habitat": h,
                                            "before": before, "after": after, "n": int(moved[hi, a, b_])})
            for control, matched in [("random", False), ("random, matched", True)]:
                runs.append((name, control, len(off), [evaluate(r) for r in random_sets(units[units["model"] == m],
                                                                                       off, rng, matched)]))
        for name, control, n_off, draws in runs:
            counts = np.stack([c for c, _ in draws]).astype(float)                       # [draws, cells, outcomes]
            counts = np.concatenate([counts, counts.reshape(len(draws), 3, 2, -1).sum(2)], 1)  # + light "all"
            shares = counts / counts.sum(2, keepdims=True)
            keys = cells + [(h, "all") for h in D.HABITATS]
            for ci, (h, l) in enumerate(keys):
                for oi, outcome in enumerate(OUTCOMES):
                    comp_rows.append({"model": m, "units_off": name, "control": control, "n_units": n_off,
                                      "habitat": h, "light": l, "outcome": outcome, "n": counts[0, ci].sum(),
                                      "share": shares[:, ci, oi].mean(), "share_sd": shares[:, ci, oi].std()})
            for key in draws[0][1]:
                values = np.array([mt[key] for _, mt in draws])
                real_out.append({"model": m, "units_off": name, "control": control, "n_units": n_off,
                                 "split": key[0], "habitat": key[1], "metric": key[2],
                                 "n": int(real_groups[key[:2]].sum()), "value": values.mean(), "sd": values.std()})
        del comp, real_f, base_comp, base_real
        torch.cuda.empty_cache()

    comp_table, real_table = pd.DataFrame(comp_rows), pd.DataFrame(real_out)
    for t in [comp_table, real_table]:
        floats = t.select_dtypes("float64").columns
        t[floats] = t[floats].astype(np.float32)
    comp_table.to_csv(RESULTS / "concepts_ablation_composites.csv", index=False, float_format="%.5g")
    real_table.to_csv(RESULTS / "concepts_ablation_real.csv", index=False, float_format="%.5g")
    pd.DataFrame(transitions).to_csv(RESULTS / "concepts_ablation_transitions.csv", index=False)
    print_ablation(comp_table, real_table)


def print_ablation(comp, real):
    for m in MODELS:
        c = comp[(comp["model"] == m) & (comp["light"] == "all")]
        c = c.assign(run=c["units_off"] + np.where(c["control"] == "none", "", " / " + c["control"]))
        table = c.pivot_table(index=["run", "n_units"], columns=["habitat", "outcome"], values="share", sort=False)
        print(f"\n{m}: composite outcome shares in % (light all) with units switched off")
        print((100 * table.astype(float)).round(1).to_string())
        r = real[(real["model"] == m) & (real["habitat"].isin(["all", *D.HABITATS]))]
        r = r.assign(run=r["units_off"] + np.where(r["control"] == "none", "", " / " + r["control"]))
        table = r.pivot_table(index=["run", "n_units"], columns=["split", "metric", "habitat"], values="value",
                              sort=False)
        print(f"\n{m}: real accuracy / macro F1 in % with units switched off")
        print((100 * table.astype(float)).round(2).to_string())


def main():
    steps = sys.argv[1:] or ["cameras", "probe", "backgrounds", "profiles", "votes", "ablation"]
    cfg = load_config()
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for step in steps:
        {"cameras": val_cameras, "probe": build_probe, "backgrounds": background_maps, "profiles": profiles,
         "votes": composite_votes, "ablation": ablation}[step](cfg)
        if torch.cuda.is_available():
            print(f"[{step}] peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
