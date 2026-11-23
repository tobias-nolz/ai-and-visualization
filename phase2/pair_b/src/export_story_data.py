"""Data and images for the Phase 2 story page (phase2/pair_b/story/).

Writes small JSON files to story/data/ and WebP images to story/img/. Every number on the page comes from these files.
Chapter 1 and the hero use the Phase 1 parquet files (phase1/pair_b/results/backgrounds.parquet, composites.parquet,
controls.parquet). The hero pictures are re-rendered with paste() from phase1/pair_b/src/composite.py (same frame,
cut-out and cell as their row in composites.parquet), so the answer under each picture is the model's stored answer
for exactly that image. Chapters 2 to 5 use the Phase 2 results in phase2/pair_b/results/ (evidence_split.py,
dissect.py, concepts.py) and the story extracts there (see below); no classification is rerun.

    python phase2/pair_b/src/export_story_data.py           # from the repo root, ai-and-vis env
    python phase2/pair_b/src/export_story_data.py extract   # only if the Phase 2 caches in outputs/ changed

The page export reads only files in git (phase1/pair_b/results/, phase2/pair_b/results/, the cut-outs) plus
data/iwildcam_v2.0 (the frames for the pictures); nothing in outputs/, no model.

`extract` writes the small results/story_* files that the page needs from the large Phase 2 caches in outputs/
(not in git; rebuilt by evidence_split.py and concepts.py):
- results/story_extracts.npz          ch. 2: class maps (14 x 14, true and winning class) and pooled cut-out mask of
                                      one composite per animal group x habitat (outputs/phase2_pair_b/cam_*.npy,
                                      masks.npz); ch. 3: the example units' frames and their layer4 maps
                                      (outputs/concepts/images.npy, one pass of the fine-tuned trunk over it)
- results/story_votes_by_winner.csv   ch. 4: the vote split per habitat and winning class, wrong composites
                                      (outputs/concepts/composite_votes.npz)
- results/story_ablation_dose.csv     ch. 5: forest composite outcomes and real test accuracy with the k most tree- or
                                      ground-leaning units switched off, k in K_GRID, with random controls
                                      (outputs/phase2_pair_b/features.npy, outputs/embeddings.npy, fc weights; CPU)

Files of the page:
- data/hero.json      the two hero composites (open and forest, by day) with the model's answer and confidence
- data/recap.json     Phase 1 recap: territory map (one dot per background frame, colour per view), grey-frame and
                      own-camera controls per species group, share "empty" per paste cell
- data/map/*.json     ch. 1 map picker: the map colours of every animal group at each of the nine paste positions
- data/meta.json      counts for the text and the methods section
- data/evidence.json  ch. 2: lead of the winning class over the true class in wrong composites, split into the part on
                      the animal's cells and on the background (per habitat), and the two class maps of the forest
                      hero composite and of one wrong composite per animal group x habitat
- data/units.json     ch. 3: label of every fine-tuned layer4 unit by the paper's IoU rule and by total activation
                      (with what the threshold sliders need), and three example units
- data/votes.json     ch. 4: the lead split by concept (vote share) against each concept's area, per habitat, for all
                      wrong composites and per frequent winning class
- data/ablation.json  ch. 5: forest composite outcomes with the 256 most tree-leaning units switched off, random
                      controls, where the "empty" answers go, real test accuracy; the same for other k (slider)
- img/hero_*.webp     the hero composites at the source frame's aspect ratio
- img/ev_*.webp       ch. 2: the composites of the animal x habitat picker
- img/unit_*.webp     the example units' top frames (by total activation), brightness = the unit's activation
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1" / "pair_b" / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, resolve  # noqa: E402
from common.data import IWildCamImages, get_transform, load_metadata, normalize  # noqa: E402
from common.model import get_device, load_model  # noqa: E402
from composite import (  # noqa: E402
    CELL_NAMES, FOREST_T, OPEN_T, PAIR_B as PHASE1, SPECIES_GROUPS, load_cutouts, paste,
)
import concepts as C  # noqa: E402
import dissect as D  # noqa: E402
import evidence_split as E  # noqa: E402

PAIR_B = Path(__file__).resolve().parents[1]
STORY = PAIR_B / "story"
DATA, IMG = STORY / "data", STORY / "img"
RESULTS = PHASE1 / "results"
P2_RESULTS = PAIR_B / "results"

OUTCOMES = ["exact species", "same group", "other group", "empty"]  # order = colour order, best first
HABITATS = ["open", "mixed", "forest"]
# Phase 1 called the infrared group "night"; it is greyscale (infrared or black-and-white cameras, often by day)
LIGHTS = ["day", "greyscale"]
LIGHT_NAMES = {"day": "day", "night": "greyscale"}

# hero: one elephant cut-out at the bottom centre (the cell of the Phase 1 snapshot) on two daytime frames
HERO_GROUP = "elephant"
HERO_CELL = (2, 1)
HERO_WRONG = "great argus"
WELL_LIT = (0.35, 0.65)  # frame brightness range that looks good on screen (and is neither washed out nor dark)
WEBP_QUALITY = 80
EV_WIDTH = 560  # ch. 2 picker pictures (shown at most ~560 px wide)
EV_QUALITY = 70

# territory map views: (key, group, cell), as in the notebook's explorer
MAP_VIEWS = [("elephant", "elephant", (2, 1)), ("porcupine", "porcupine", (1, 1))]
MAP_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def r4(x):
    return round(float(x), 4)


def write_json(name, obj, compact=False):
    text = json.dumps(obj, separators=(",", ":")) if compact else json.dumps(obj, indent=1)
    (DATA / name).write_text(text + "\n")
    print(f"wrote {(DATA / name).relative_to(ROOT)} ({len(text) / 1024:.1f} kB)")


def load():
    bg = pd.read_parquet(RESULTS / "backgrounds.parquet")
    bg["light"] = bg["light"].map(LIGHT_NAMES)
    assert bg["light"].notna().all()
    res = pd.read_parquet(RESULTS / "composites.parquet")
    res = res.merge(bg[["bg_id", "light", "habitat", "brightness", "location", "openness", "split"]], on="bg_id")
    controls = pd.read_parquet(RESULTS / "controls.parquet")
    cut_table = pd.read_csv(PHASE1 / "cutouts" / "cutouts.csv")
    return bg, res, controls, cut_table


# ---------------------------------------------------------------- hero

def pick_hero(res):
    """The hero composites: same cut-out and cell, a well-lit open and forest frame by day.

    Rule (fixed, so the pick is reproducible):
    - open: the most confident correct answer; forest: the most confident "great argus";
      the cut-out (variant) is the one where the weaker of these two is strongest (as in the Phase 1 slides).
    """
    d = res[
        (res.group == HERO_GROUP) & (res.row == HERO_CELL[0]) & (res.col == HERO_CELL[1]) & (res.light == "day")
        & res.brightness.between(*WELL_LIT)
    ]
    best = []
    for v, dv in d.groupby("variant"):
        o = dv[(dv.habitat == "open") & (dv.outcome == "exact species")].sort_values("p_pred", ascending=False)
        f = dv[(dv.habitat == "forest") & (dv.pred_species == HERO_WRONG)].sort_values("p_pred", ascending=False)
        if len(o) and len(f):
            best.append((min(o.p_pred.iloc[0], f.p_pred.iloc[0]), v, o.iloc[0], f.iloc[0]))
    _, v, open_row, forest_row = max(best, key=lambda t: t[0])
    return int(v), {"open": open_row, "forest": forest_row}


def render(cfg, bg_row, cutout, cell, name, width=None, quality=WEBP_QUALITY):
    """Paste the cut-out exactly like run_composites.py and save it at the source frame's aspect ratio.

    width: scale to this width (default: the source frame's size). The cell grid of the page's overlays assumes
    the whole frame is shown.
    """
    image, _ = IWildCamImages(pd.DataFrame({"filename": [bg_row.filename]}), cfg)[0]
    gray = torch.tensor([bool(bg_row.infrared)])
    out = paste(image[None], cutout, *cell, gray=gray)[0].permute(1, 2, 0).numpy()
    with Image.open(resolve(cfg["paths"]["data_dir"]) / "train" / bg_row.filename) as src:
        size = src.size  # the model sees the frame squashed to 448 x 448; show it un-squashed
    if width:
        size = (width, round(width * size[1] / size[0]))
    Image.fromarray(out).resize(size, Image.LANCZOS).save(IMG / name, quality=quality, method=6)
    return name, size


def export_hero(cfg, bg, res, cut_table, stats):
    variant, rows = pick_hero(res)
    cutout = load_cutouts()[HERO_GROUP, variant]
    src = cut_table[(cut_table.group == HERO_GROUP) & (cut_table.variant == variant)].iloc[0]
    bgi = bg.set_index("bg_id")
    frames = []
    for hab, r in rows.items():
        b = bgi.loc[r.bg_id]
        name, size = render(cfg, b, cutout, HERO_CELL, f"hero_{hab}.webp")
        frames.append({
            "habitat": hab,
            "img": name,
            "width": size[0],
            "height": size[1],
            "bg_id": int(r.bg_id),
            "camera": int(b.location),
            "split": b.split,
            "camera_openness": r4(b.openness),
            "brightness": r4(b.brightness),
            "answer": r.pred_species,
            "confidence": r4(r.p_pred),
            "outcome": r.outcome,
            "p_true": r4(r.p_species),
            "p_empty": r4(r.p_empty),
        })
    view = res[(res.group == HERO_GROUP) & (res.row == HERO_CELL[0]) & (res.col == HERO_CELL[1])]
    day_forest = view[(view.light == "day") & (view.habitat == "forest")]
    hero = {
        "group": HERO_GROUP,
        "species": src.common_name,
        "variant": variant,
        "cell": CELL_NAMES[HERO_CELL],
        "cutout_camera": int(src.location),
        "cutout_split": src.split,
        "rule": inspect.cleandoc(pick_hero.__doc__).split("\n\n", 1)[1],
        "frames": frames,
        # how typical the example is: this cut-out position, all three cut-outs, all daytime forest frames
        "context": {
            "day_forest_composites": len(day_forest),
            "day_forest_correct": r4((day_forest.outcome == "exact species").mean()),
            "day_forest_wrong_answer": r4((day_forest.pred_species == HERO_WRONG).mean()),
            "day_open_correct": stats["day|open"]["correct"],
        },
    }
    write_json("hero.json", hero)
    return hero


# ---------------------------------------------------------------- recap: territory map

def majority(d, bg_ids):
    """Most common outcome over the cut-outs per background (ties: the better outcome), as in the notebook."""
    counts = pd.crosstab(d.bg_id, d.outcome).reindex(index=bg_ids, columns=OUTCOMES, fill_value=0)
    return counts.to_numpy().argmax(1).tolist()


def habitat_stats(d):
    """Per light and habitat: composites, share of each outcome and the most common wrong (non-empty) answer."""
    out = {}
    for light in LIGHTS:
        for hab in HABITATS:
            g = d[(d.light == light) & (d.habitat == hab)]
            share = g.outcome.value_counts(normalize=True).reindex(OUTCOMES, fill_value=0)
            wrong = g[~g.outcome.isin(["exact species", "empty"])].pred_species.value_counts()
            out[f"{light}|{hab}"] = {
                "n": len(g),
                "correct": r4(share["exact species"]),
                "same": r4(share["same group"]),
                "other": r4(share["other group"]),
                "empty": r4(share["empty"]),
                "wrong": wrong.index[0] if len(wrong) else "",
                "wrong_share": r4(wrong.iloc[0] / len(g)) if len(wrong) else 0,
            }
    return out


def export_map(bg, res):
    bg = bg.sort_values("bg_id").reset_index(drop=True)
    # frames of one camera share their x (camera openness): the notebook's jitter, same seed and order
    jitter = np.random.default_rng(0).uniform(-0.007, 0.007, len(bg))
    frames = [{
        "id": int(r.bg_id),
        "x": r4(r.openness + j),
        "y": round(float(np.clip(r.brightness, 0, 1)), 3),
        "light": r.light,
        "hab": r.habitat,
        "cam": int(r.location),
    } for r, j in zip(bg.itertuples(), jitter)]
    species = sorted(res.pred_species.unique())
    code = {s: i for i, s in enumerate(species)}

    def view(group, cell):
        d = res[(res.group == group) & (res.row == cell[0]) & (res.col == cell[1])]
        answers = {}  # bg_id -> [[species code, outcome index] of each cut-out], for the tooltip
        for r in d.sort_values("variant").itertuples():
            answers.setdefault(r.bg_id, []).append([code[r.pred_species], OUTCOMES.index(r.outcome)])
        return {
            "group": group,
            "species": d.species.iloc[0] if d.species.nunique() == 1 else group,
            "cell": CELL_NAMES[cell],
            "row": cell[0], "col": cell[1],
            "n": len(d),
            "outcome": majority(d, bg.bg_id),
            "answers": [answers[i] for i in bg.bg_id],
            "overall": {o: r4((d.outcome == o).mean()) for o in OUTCOMES},
            "habitats": habitat_stats(d),
        }

    views = {key: view(group, cell) for key, group, cell in MAP_VIEWS}

    # the map picker (any group, any cell): one file per group, loaded when picked. Compact: the answers of the
    # three cut-outs per frame as one character each (index into "codes" = [species code, outcome index]); the page
    # takes the dot colour (majority) from them with the rule of majority()
    (DATA / "map").mkdir(exist_ok=True)
    for old in (DATA / "map").glob("*.json"):
        old.unlink()
    groups = []
    for group in sorted(res.group.unique()):
        file = group.replace("/", "-").replace(" ", "-")
        cells = {}
        for r in range(3):
            for c in range(3):
                v = view(group, (r, c))
                pairs = sorted({tuple(a) for frame in v["answers"] for a in frame})
                assert len(pairs) <= len(MAP_ALPHABET) and all(len(a) == 3 for a in v["answers"])
                char = {p: MAP_ALPHABET[i] for i, p in enumerate(pairs)}
                v["codes"] = [list(p) for p in pairs]
                v["answers"] = "".join(char[tuple(a)] for frame in v["answers"] for a in frame)
                del v["outcome"]
                cells[f"{r}{c}"] = v
        write_json(f"map/{file}.json", cells, compact=True)
        groups.append({"group": group, "species": cells["21"]["species"], "file": f"map/{file}.json"})
    return {"frames": frames, "species": species, "views": views, "groups": groups}


# ---------------------------------------------------------------- recap: controls and the top row

def export_controls(res, controls):
    """Share correct per species group: grey frame (no background), own camera, other cameras (the composites)."""
    ok = lambda d: d.outcome == "exact species"  # noqa: E731
    grey = controls[controls.control == "grey frame"]
    grey_bw = controls[controls.control == "grey frame, greyscale cut-out"]
    own = controls[controls.control == "own camera"]
    rows = []
    for group in res.group.unique():
        rows.append({
            "group": group,
            "grey": r4(ok(grey[grey.group == group]).mean()),
            "grey_n": int((grey.group == group).sum()),
            "grey_bw": r4(ok(grey_bw[grey_bw.group == group]).mean()),
            "own": r4(ok(own[own.group == group]).mean()),
            "own_n": int((own.group == group).sum()),
            "swap": r4(ok(res[res.group == group]).mean()),
            "swap_n": int((res.group == group).sum()),
        })
    rows.sort(key=lambda r: (-r["grey"], -r["swap"]))
    by_name = {r["group"]: r for r in rows}
    overall = {
        "grey": r4(ok(grey).mean()),
        "grey_bw": r4(ok(grey_bw).mean()),
        "own": r4(ok(own).mean()),
        "swap": r4(ok(res).mean()),
        "grey_zero_groups": sum(r["grey"] == 0 for r in rows),
        "grey_n_per_group": rows[0]["grey_n"],
        "groups": len(rows),
    }
    return {"groups": rows, "by_group": by_name, "overall": overall}


def export_cells(res, controls):
    """Share predicted empty per paste cell (3 x 3) and per row: composites, grey frame, own camera."""
    def grid(d):
        e = d.assign(e=d.outcome == "empty")
        cells = e.groupby(["row", "col"]).e.mean().unstack().reindex(index=range(3), columns=range(3))
        return {
            "cells": [[r4(x) for x in row] for row in cells.to_numpy()],
            "rows": [r4(x) for x in e.groupby("row").e.mean().reindex(range(3))],
            "n": len(d),
        }
    out = {
        "swap": grid(res),
        "grey": grid(controls[controls.control == "grey frame"]),
        "own": grid(controls[controls.control == "own camera"]),
    }
    out["swap_by_habitat"] = {
        hab: [r4(x) for x in res[res.habitat == hab].assign(e=lambda d: d.outcome == "empty")
              .groupby("row").e.mean().reindex(range(3))]
        for hab in HABITATS
    }
    out["swap_correct_rows"] = [
        r4(x) for x in res.assign(c=res.outcome == "exact species").groupby("row").c.mean().reindex(range(3))
    ]
    return out


# ---------------------------------------------------------------- meta

def greyscale_facts(cfg, bg):
    """Why Phase 1's "night" frames are called greyscale: no colour camera, and daytime timestamps."""
    grey = bg[bg.light == "greyscale"]
    refs = pd.read_parquet(P2_RESULTS / "concepts_backgrounds.parquet").set_index("bg_id").loc[grey.bg_id]
    no_colour = refs[refs.source == "greyscale daytime reference"]  # cameras without a usable colour frame
    stamps = grey.merge(load_metadata(cfg)[["filename", "datetime"]], on="filename", how="left")
    hour = pd.to_datetime(stamps.datetime).dt.hour
    assert hour.notna().all()
    return {
        "frames": len(grey),
        "infrared_flag": r4(grey.infrared.mean()),
        "no_colour_camera_frames": len(no_colour),
        "no_colour_cameras": int(no_colour.location.nunique()),
        "daytime_stamp": r4(hour.between(7, 18).mean()),  # timestamp between 7:00 and 18:59
    }


def export_meta(cfg, bg, res, controls, cut_table):
    per_habitat = {}
    for light in LIGHTS:
        for hab in HABITATS:
            g = bg[(bg.light == light) & (bg.habitat == hab)]
            per_habitat[f"{light}|{hab}"] = {"frames": len(g), "cameras": int(g.location.nunique())}
            top2 = g.location.value_counts().head(2)
            per_habitat[f"{light}|{hab}"]["top2_cameras_frames"] = int(top2.sum())
    smallest = min(per_habitat, key=lambda k: per_habitat[k]["frames"])
    meta = {
        "backgrounds": len(bg),
        "cameras": int(bg.location.nunique()),
        "splits": {s: {"frames": int((bg.split == s).sum()), "cameras": int(bg[bg.split == s].location.nunique())}
                   for s in ["id_test", "test"]},
        "per_habitat": per_habitat,
        "smallest_habitat": {"key": smallest, **per_habitat[smallest]},
        "groups": int(cut_table.group.nunique()),
        "cutouts": len(cut_table),
        "cells": 9,
        "composites": len(res),
        "controls": {k: int(v) for k, v in controls.control.value_counts().items()},
        "own_camera_frames_per_cutout": r4(
            controls[controls.control == "own camera"].groupby(["group", "variant"]).size().div(9).median()
        ),
        "thresholds": {"forest_below": r4(FOREST_T), "open_above": r4(OPEN_T)},
        "outcomes": OUTCOMES,
        "outcome_shares": {o: r4((res.outcome == o).mean()) for o in OUTCOMES},
        "greyscale": greyscale_facts(cfg, bg),
    }
    write_json("meta.json", meta)
    return meta


# ---------------------------------------------------------------- ch. 2: where is the evidence?

def common_names():
    """Class index -> common name, as pred_species in Phase 1 (data/species_groups.csv)."""
    return pd.read_csv(SPECIES_GROUPS).set_index("y")["common_name"]


def r3(x):
    return round(float(x), 3)


def lead_parts(d):
    """Mean lead of the predicted over the true class, split into animal cells, background cells and bias."""
    return {
        "n": len(d),
        "animal": r3((d.pred_inside - d.true_inside).mean()),
        "background": r3((d.pred_outside - d.true_outside).mean()),
        "bias": r3((d.pred_bias - d.true_bias).mean()),
        "lead": r3((d.pred_logit - d.true_logit).mean()),
        "mask_area": r4(d.mask_area.mean()),
    }


def export_evidence(cfg, bg, hero):
    """Lead of the winner in wrong composites (pred != true species), on the animal vs. on the background."""
    res = pd.read_parquet(P2_RESULTS / "evidence_split.parquet",
                          columns=["bg_id", "group", "variant", "row", "col", "y", "pred_y", "mask_area", "p_pred",
                                   "p_true", *[f"{c}_{p}" for c in ["true", "pred"]
                                               for p in ["logit", "inside", "outside", "bias"]]])
    res = res.merge(bg[["bg_id", "habitat", "light"]], on="bg_id")
    wrong = res[res.pred_y != res.y]
    out = {
        "composites": len(res),
        "wrong": lead_parts(wrong),
        "wrong_share": r4(len(wrong) / len(res)),
        "habitats": {h: {**lead_parts(wrong[wrong.habitat == h]),
                         "lights": {l: lead_parts(wrong[(wrong.habitat == h) & (wrong.light == l)]) for l in LIGHTS}}
                     for h in HABITATS},
        "correct": {"mask_area": r4(res[res.pred_y == res.y].mask_area.mean())},
    }

    # one composite per animal group x habitat with its two class maps (CAM: mean over the 14 x 14 cells + bias =
    # logit); results/story_extracts.npz, picked by pick_evidence(); elephant x forest is the forest hero composite
    x = np.load(EXTRACTS)
    names = common_names()
    f = next(fr for fr in hero["frames"] if fr["habitat"] == "forest")
    cutouts = load_cutouts()
    bgi = bg.set_index("bg_id")
    for old in IMG.glob("ev_*.webp"):
        old.unlink()
    picks = []
    for j, i in enumerate(x["ev_index"]):
        r = res.iloc[int(i)]
        cams = {c: x[f"ev_cam_{c}"][j] for c in ["true", "pred"]}
        for c in cams:
            assert abs(cams[c].mean() + r[f"{c}_bias"] - r[f"{c}_logit"]) < 0.01
        mask = x["ev_mask"][j]
        assert abs(mask.mean() - r.mask_area) < 1e-3 and r.habitat == x["ev_habitat"][j] and r.pred_y != r.y
        rule = str(x["ev_rule"][j])
        if rule == "hero composite":
            assert r.bg_id == f["bg_id"] and r.group == hero["group"] and r.variant == hero["variant"]
            assert names[int(r.pred_y)] == f["answer"] and abs(r.p_pred - f["confidence"]) < 1e-3
            img = {"img": f["img"], "width": f["width"], "height": f["height"]}
        else:
            name = f"ev_{r.group.replace('/', '-').replace(' ', '-')}_{r.habitat}.webp"
            img = dict(zip(["img", "size"], render(cfg, bgi.loc[r.bg_id], cutouts[r.group, int(r.variant)],
                                                   (int(r.row), int(r.col)), name, width=EV_WIDTH, quality=EV_QUALITY)))
            img = {"img": img["img"], "width": img["size"][0], "height": img["size"][1]}
        picks.append({
            **img, "group": r.group, "habitat": r.habitat, "rule": rule, "camera": int(bgi.loc[r.bg_id].location),
            "cell": CELL_NAMES[(int(r.row), int(r.col))],
            "true": names[int(r.y)], "pred": names[int(r.pred_y)], "p_pred": r4(r.p_pred), "p_true": r4(r.p_true),
            "grid": int(cams["true"].shape[0]),
            "cam_true": [[round(float(v), 2) for v in row] for row in cams["true"]],
            "cam_pred": [[round(float(v), 2) for v in row] for row in cams["pred"]],
            "mask": [[round(float(v), 2) for v in row] for row in mask],
            **{k: v for k, v in lead_parts(res.iloc[[int(i)]]).items() if k != "n"},
            "true_logit": r3(r.true_logit), "pred_logit": r3(r.pred_logit),
        })
    hero_pick = next(p for p in picks if p["rule"] == "hero composite")
    out["hero"] = {k: v for k, v in hero_pick.items() if k not in ("group", "habitat", "rule", "cell")}
    out["picks"] = [p if p is not hero_pick else {k: p[k] for k in ("group", "habitat", "rule")} for p in picks]
    out["pick_rule"] = inspect.cleandoc(pick_evidence.__doc__).split("\n\n", 1)[1]
    write_json("evidence.json", out, compact=True)
    return out


# ---------------------------------------------------------------- ch. 3: what does the model see in the trees?

EXAMPLE_LABELS = ["animal", "tree", "sky"]      # one example unit each: the one with the largest excess share
EXAMPLE_IMAGES = 2                              # its frames with the highest total activation, distinct cameras,
EXAMPLE_AREA = (0.2, 0.6)                       # among frames where the concept covers 20-60 % (so the map can
                                                # show where on the frame the activation sits)
THUMB_H = 240
SCENERY = ["tree", "plant", "grass", "ground", "rock", "sky", "water"]


@torch.no_grad()
def unit_totals(cfg):
    """Total layer4 activation of every fine-tuned unit on every concepts probe frame [N, 2048] (cached)."""
    cache = C.CACHE / "story_unit_totals.npy"
    if cache.exists():
        return np.load(cache)
    device = get_device(cfg)
    trunk = D.load_trunk("finetuned", cfg, device)
    images = np.load(C.CACHE / "images.npy", mmap_mode="r")
    out = np.zeros((len(images), 2048), np.float32)
    for first in range(0, len(images), 16):
        x = torch.from_numpy(np.array(images[first:first + 16])).to(device)
        out[first:first + 16] = trunk(normalize(x)).sum((2, 3)).float().cpu().numpy()
    np.save(cache, out)
    del trunk
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def unit_maps(cfg, rows):
    """layer4 maps of the fine-tuned model [len(rows), 2048, 14, 14] for some concepts probe frames (CPU is enough)."""
    trunk = D.load_trunk("finetuned", cfg, torch.device("cpu"))
    images = np.load(C.CACHE / "images.npy", mmap_mode="r")
    return trunk(normalize(torch.from_numpy(np.stack([images[r] for r in rows])))).numpy(), images


def unit_thumbnail(image, amap, scale, size):
    """The frame un-squashed to its own aspect ratio; brightness follows the unit's activation (as in the paper)."""
    img = Image.fromarray(np.ascontiguousarray(image.transpose(1, 2, 0))).resize(size, Image.LANCZOS)
    a = torch.from_numpy(amap)[None, None]
    a = torch.nn.functional.interpolate(a, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0]
    a = (a / scale).clamp(0, 1).numpy()
    return Image.fromarray((np.asarray(img).astype(float) * (0.12 + 0.88 * a[..., None])).astype(np.uint8))


def export_units(cfg):
    units = pd.read_parquet(P2_RESULTS / "concepts_units.parquet")
    ft = units[units.model == "finetuned"].sort_values("unit").reset_index(drop=True)
    diss = pd.read_parquet(P2_RESULTS / "dissection_units.parquet")
    dl4 = diss[(diss.model == "finetuned") & (diss.layer == "layer4")].sort_values("unit").reset_index(drop=True)
    assert (dl4.unit.to_numpy() == ft.unit.to_numpy()).all()
    labels = [*C.PARTS, "mixed", "dead"]
    iou_labels = ["", *D.CONCEPTS]
    probe = pd.read_parquet(P2_RESULTS / "concepts_probe.parquet")
    area = {c: r4(probe[f"area_{c.replace(' ', '_')}"].mean()) for c in C.PARTS}

    # share of each unit's activation on its label, or on the concept with the largest excess share if "mixed"
    share = ft[[f"share_{c.replace(' ', '_')}" for c in C.PARTS]].to_numpy()
    true_area = np.array([probe[f"area_{c.replace(' ', '_')}"].mean() for c in C.PARTS])
    best = (share - true_area).argmax(1)  # as concepts.label_units(): label_excess, label_lift are of this concept
    on = [C.PARTS.index(lab) if lab in C.PARTS else b for lab, b in zip(ft.label, best)]
    assert (np.array(on) == best).all()
    share_on = share[np.arange(len(ft)), on]
    counts = {m: {k: int(v) for k, v in u.label.value_counts().items()} for m, u in units.groupby("model")}
    paper = {f"{m}|{l}": {k or "none": int(v) for k, v in d.label.value_counts().items()}
             for (m, l), d in diss.groupby(["model", "layer"])}
    concepts = pd.read_csv(P2_RESULTS / "dissection_concepts.csv").set_index("concept")
    rows = [[labels.index(lab), iou_labels.index(il), round(float(ex), 5), r3(sh), iou_labels.index(ic),
             round(float(iou), 4), int(b), round(float(lift), 4)]
            for lab, il, ex, sh, ic, iou, b, lift in zip(ft.label, dl4.label, ft.label_excess, share_on, dl4.concept,
                                                         dl4.iou, best, ft.label_lift)]
    # the page relabels units live (threshold sliders): with the default thresholds, the rounded numbers must give
    # exactly the stored labels
    assert not (ft.label == "dead").any()
    for r in rows:
        by_act = labels[r[6]] if r[2] >= C.LABEL_MIN_EXCESS and r[7] >= C.LABEL_MIN_LIFT else "mixed"
        by_iou = iou_labels[r[4]] if r[5] >= D.IOU_MIN else ""
        assert by_act == labels[r[0]] and by_iou == iou_labels[r[1]], r
    out = {
        "n_units": len(ft),
        "labels": labels,
        "iou_labels": iou_labels,
        # per unit: [label by total activation, label by the paper's rule, excess share of the concept with the
        # largest excess share, share of its activation on that concept, best IoU concept, its IoU, the concept with
        # the largest excess share, its lift]
        "units": rows,
        "counts": counts,
        "scenery": {m: sum(c.get(s, 0) for s in SCENERY) for m, c in counts.items()},
        "paper": paper,
        "paper_finetuned_layer4_labelled": int((dl4.label != "").sum()),
        "rules": {"quantile": D.QUANTILE, "iou_min": D.IOU_MIN, "min_excess": C.LABEL_MIN_EXCESS,
                  "min_lift": C.LABEL_MIN_LIFT},
        "area": area,
        "concept_pixels": {c: r4(concepts.loc[c, "pixel_share"]) for c in concepts.index},
        "max_iou": {c: r4(concepts.loc[c, "max_iou_for_median_unit_mask"]) for c in concepts.index},
        "probe_frames": len(probe),
    }

    # example units (results/story_extracts.npz, example_unit_picks()): the strongest unit per label; its frames with
    # the highest total activation, one per camera, and their layer4 maps. The frames are reloaded from data/.
    x = np.load(EXTRACTS)
    meta = load_metadata(cfg).set_index("image_id")
    transform = get_transform(cfg)
    image_dir = resolve(cfg["paths"]["data_dir"]) / "train"
    for old in IMG.glob("unit_*.webp"):
        old.unlink()
    examples = []
    for unit, label, rows_, amaps in zip(x["ex_unit"], x["ex_label"], x["ex_rows"], x["ex_maps"]):
        unit, label = int(unit), str(label)
        u = ft.loc[unit]
        assert u.unit == unit and u.label == label
        scale = amaps.max()
        frames = []
        for j, (row, amap) in enumerate(zip(rows_, amaps)):
            p = probe.iloc[int(row)]
            path = image_dir / meta.loc[p.image_id, "filename"]
            with Image.open(path) as src:
                w, h = src.size
            size = (round(THUMB_H * w / h), THUMB_H)
            name = f"unit_{unit}_{j}.webp"
            image = D.load_input(path, transform).numpy()
            unit_thumbnail(image, amap, scale, size).save(IMG / name, quality=WEBP_QUALITY, method=6)
            frames.append({"img": name, "width": size[0], "height": size[1], "camera": int(p.location),
                           "habitat": p.habitat, "concept_area": r4(p[f"area_{label.replace(' ', '_')}"])})
        examples.append({
            "unit": unit, "label": label,
            "share": r3(u[f"share_{label.replace(' ', '_')}"]), "area": area[label],
            "lift": r3(u.label_lift),
            "iou_concept": iou_labels[iou_labels.index(dl4.loc[unit, "concept"])],
            "iou": round(float(dl4.loc[unit, "iou"]), 4),
            "frames": frames,
        })
    out["examples"] = examples
    write_json("units.json", out, compact=True)
    return out


# ---------------------------------------------------------------- ch. 4: who votes for the great argus?

SMALL_CONCEPTS = ["plant", "grass", "rock", "water", "other"]   # folded into "other scenery" in the figure
GREAT_ARGUS = "great argus"
WINNER_OPTIONS = 4                                              # the most frequent wrong winners in the picker


def export_votes():
    v = pd.read_csv(P2_RESULTS / "concepts_composite_votes.csv")
    v = v[(v.model == "finetuned") & (v.light == "all") & (v.outcome == "wrong")]
    out = {"habitats": {}}
    for h in HABITATS:
        d = v[v.habitat == h].set_index("concept")
        lead = d.loc["logit", "margin"]
        parts = d.drop(["logit", "bias"])
        assert abs(parts.margin.sum() + d.loc["bias", "margin"] - lead) < 0.01 and abs(parts.area.sum() - 1) < 0.01
        rows = [{"concept": c, "area": r4(parts.loc[c, "area"]), "votes": r3(parts.loc[c, "margin"]),
                 "share": r4(parts.loc[c, "margin"] / lead)} for c in parts.index]
        small = parts.loc[SMALL_CONCEPTS]
        folded = [r for r in rows if r["concept"] not in SMALL_CONCEPTS] + [{
            "concept": "other scenery", "area": r4(small.area.sum()), "votes": r3(small.margin.sum()),
            "share": r4(small.margin.sum() / lead), "parts": SMALL_CONCEPTS}]
        out["habitats"][h] = {"n": int(d.n.iloc[0]), "lead": r3(lead), "bias": r3(d.loc["bias", "margin"]),
                              "concepts": {r["concept"]: {k: v for k, v in r.items() if k != "concept"} for r in rows},
                              "folded": folded}

    # per winning class (results/story_votes_by_winner.csv, from the per-composite split): the great argus in the
    # text, the frequent winners in the picker
    w = pd.read_csv(VOTES_BY_WINNER)
    ga = w[(w.habitat == "forest") & (w.winner == GREAT_ARGUS)].set_index("concept")
    out["great_argus_forest"] = {
        "n": int(ga.n.iloc[0]), "lead": r3(ga.lead.iloc[0]),
        "concepts": {c: {"area": r4(ga.loc[c, "area"]), "share": r4(ga.loc[c, "votes"] / ga.lead.iloc[0])}
                     for c in ga.index},
    }
    totals = w[(w.winner != "all") & (w.concept == "tree")].groupby("winner").n.sum().sort_values(ascending=False)
    winners = list(totals.index[:WINNER_OPTIONS])
    if GREAT_ARGUS not in winners:
        winners[-1] = GREAT_ARGUS
    out["winners"] = [{"winner": v, "n": int(totals[v])} for v in winners]
    out["min_n"] = WINNER_MIN_N
    out["by_winner"] = {}
    for v in winners:
        out["by_winner"][v] = {}
        for h in HABITATS:
            d = w[(w.habitat == h) & (w.winner == v)].set_index("concept")
            if not len(d):
                continue
            lead = d.lead.iloc[0]
            assert abs(d.votes.sum() + d.bias.iloc[0] - lead) < 0.01 and abs(d.area.sum() - 1) < 0.01
            small = d.loc[SMALL_CONCEPTS]
            rows = [{"concept": c, "area": r4(d.loc[c, "area"]), "votes": r3(d.loc[c, "votes"]),
                     "share": r4(d.loc[c, "votes"] / lead)} for c in d.index if c not in SMALL_CONCEPTS]
            rows.append({"concept": "other scenery", "area": r4(small.area.sum()), "votes": r3(small.votes.sum()),
                         "share": r4(small.votes.sum() / lead), "parts": SMALL_CONCEPTS})
            out["by_winner"][v][h] = {"n": int(d.n.iloc[0]), "lead": r3(lead), "folded": rows}
    write_json("votes.json", out)
    return out


# ---------------------------------------------------------------- ch. 5: switch off the forest

ABLATION_SET = "tree/plant top 256"
ABLATION_HABITAT = "forest"


def export_ablation():
    comp = pd.read_csv(P2_RESULTS / "concepts_ablation_composites.csv")
    comp = comp[(comp.model == "finetuned") & (comp.light == "all")]
    real = pd.read_csv(P2_RESULTS / "concepts_ablation_real.csv")
    real = real[(real.model == "finetuned") & (real.metric == "acc")]
    trans = pd.read_csv(P2_RESULTS / "concepts_ablation_transitions.csv")
    trans = trans[(trans.model == "finetuned") & (trans.units_off == ABLATION_SET)
                  & (trans.habitat == ABLATION_HABITAT)]

    def shares(units_off, control, habitat=ABLATION_HABITAT):
        d = comp[(comp.units_off == units_off) & (comp.control == control) & (comp.habitat == habitat)]
        d = d.set_index("outcome").loc[OUTCOMES]
        return {"n": int(d.n.iloc[0]), "n_units": int(d.n_units.iloc[0]),
                "share": {o: r4(d.loc[o, "share"]) for o in OUTCOMES},
                "sd": {o: r4(d.loc[o, "share_sd"]) for o in OUTCOMES}}

    def acc(units_off, control):
        d = real[(real.units_off == units_off) & (real.control == control)]
        return {f"{s}|{h}": {"acc": r4(g.value.iloc[0]), "sd": r4(g.sd.iloc[0]), "n": int(g.n.iloc[0])}
                for (s, h), g in d.groupby(["split", "habitat"])}

    runs = [("as is", "none", "none"), ("units off", ABLATION_SET, "none"),
            ("random", ABLATION_SET, "random"), ("random, matched", ABLATION_SET, "random, matched")]
    empty = trans[trans.before == "empty"].set_index("after").n.reindex(OUTCOMES)
    before = trans.groupby("before").n.sum().reindex(OUTCOMES)
    other_sets = {}
    for name in ["tree/plant", "tree/plant top 64", "ground top 256"]:
        other_sets[name] = {"units_off": shares(name, "none"), "random": shares(name, "random")}
    out = {
        "set": ABLATION_SET, "habitat": ABLATION_HABITAT,
        "runs": {k: {**shares(u, c), "acc": acc(u, c)} for k, u, c in runs},
        "draws": int(C.RANDOM_DRAWS),
        "empty_before": int(before["empty"]),
        "empty_after": {o: int(empty[o]) for o in OUTCOMES},
        "empty_after_share": {o: r4(empty[o] / before["empty"]) for o in OUTCOMES},
        # the former "empty" answers that now name an animal: right species vs. wrong one
        "reappear": {"n": int(empty.drop("empty").sum()),
                     "correct": r4(empty["exact species"] / empty.drop("empty").sum()),
                     "wrong": r4(empty[["same group", "other group"]].sum() / empty.drop("empty").sum())},
        "other_sets": other_sets,
    }
    assert out["runs"]["as is"]["n"] == int(before.sum())

    # the slider: the k most tree- or ground-leaning units off, k in K_GRID (results/story_ablation_dose.csv; for
    # k = 64 and 256 these are the numbers of concepts_ablation_*.csv above)
    dose = pd.read_csv(ABLATION_DOSE)

    def run(d):
        v = d.set_index("metric")
        return {"share": {o: r4(v.value[f"share|{o}"]) for o in OUTCOMES},
                "sd": {o: r4(v.sd[f"share|{o}"]) for o in OUTCOMES},
                "acc": {key: {"acc": r4(v.value[f"acc|{key}"]), "sd": r4(v.sd[f"acc|{key}"])}
                        for key in ["test|all", "test|forest"]},
                **({"empty_after": {o: int(v.value[f"empty_to|{o}"]) for o in OUTCOMES}}
                   if "empty_to|empty" in v.index else {})}
    out["dose"] = {"k": K_GRID, "sets": {}}
    for key in DOSE_SETS:
        out["dose"]["sets"][key] = [{c: run(dose[(dose.set == key) & (dose.k == k) & (dose.control == c)])
                                     for c in ["none", "random", "random, matched"]} for k in K_GRID]
    k = out["runs"]["units off"]["n_units"]
    d = out["dose"]["sets"]["tree"][K_GRID.index(k)]
    for name, c in [("units off", "none"), ("random", "random"), ("random, matched", "random, matched")]:
        assert all(abs(d[c]["share"][o] - out["runs"][name]["share"][o]) < 5e-4 for o in OUTCOMES)
    assert d["none"]["empty_after"] == out["empty_after"]
    write_json("ablation.json", out)
    return out


# ---------------------------------------------------------------- extract: outputs/ caches -> results/story_*

EXTRACTS = P2_RESULTS / "story_extracts.npz"
VOTES_BY_WINNER = P2_RESULTS / "story_votes_by_winner.csv"
ABLATION_DOSE = P2_RESULTS / "story_ablation_dose.csv"
WINNER_MIN_N = 30                               # ch. 4: winning classes with at least this many wrong composites
K_GRID = [8, 16, 32, 64, 128, 192, 256, 384, 512]  # ch. 5 slider: units switched off
DOSE_SETS = {"tree": "tree/plant", "ground": "ground"}  # page key -> concepts.SETS


def pick_evidence(res, hero_forest):
    """One wrong composite per animal group and habitat for the chapter 2 picker.

    Rule (fixed, so the pick is reproducible): elephant x forest is the forest hero composite. Otherwise the most
    confident wrong answer ("empty" included) on a well-lit daytime frame with the cut-out at the bottom centre, as in
    the hero; if there is none, the most confident one at any position on a well-lit daytime frame, then on any
    daytime frame. A group that is never wrong there gets no pick.
    """
    wrong = res[(res.outcome != "exact species") & (res.light == "day")]
    picks = []
    for group in sorted(res.group.unique()):
        for hab in HABITATS:
            if (group, hab) == (HERO_GROUP, "forest"):
                picks.append(("hero composite", hero_forest))
                continue
            d = wrong[(wrong.group == group) & (wrong.habitat == hab)]
            lit = d[d.brightness.between(*WELL_LIT)]
            for rule, c in [("bottom centre", lit[(lit.row == HERO_CELL[0]) & (lit.col == HERO_CELL[1])]),
                            ("any position", lit), ("any position, any brightness", d)]:
                if len(c):
                    picks.append((rule, c.sort_values("p_pred", ascending=False, kind="stable").iloc[0]))
                    break
    return picks


def extract_evidence(res):
    """Class maps and pooled mask of the chapter 2 picks (outputs/phase2_pair_b/cam_*.npy, masks.npz)."""
    _, rows = pick_hero(res)
    picks = pick_evidence(res, rows["forest"])
    ev = pd.read_parquet(P2_RESULTS / "evidence_split.parquet", columns=["bg_id", "group", "variant", "row", "col"])
    keys = ["bg_id", "group", "variant", "row", "col"]
    index = pd.MultiIndex.from_frame(ev[keys]).get_indexer(
        pd.MultiIndex.from_tuples([tuple(r[k] for k in keys) for _, r in picks], names=keys))
    assert (index >= 0).all()
    cams = {c: np.load(E.OUT_DIR / f"cam_{c}.npy", mmap_mode="r")[index] for c in ["true", "pred"]}
    masks = np.load(E.OUT_DIR / "masks.npz")
    mkey = pd.MultiIndex.from_arrays([masks["group"], masks["variant"], masks["row"], masks["col"]])
    m = mkey.get_indexer(pd.MultiIndex.from_tuples([(r.group, r.variant, r.row, r.col) for _, r in picks]))
    assert (m >= 0).all()
    return {
        "ev_index": index, "ev_rule": np.array([rule for rule, _ in picks]),
        "ev_habitat": np.array([r.habitat for _, r in picks]),
        "ev_cam_true": cams["true"], "ev_cam_pred": cams["pred"], "ev_mask": masks["mask"][m],
    }


def example_unit_picks(cfg):
    """The example unit per EXAMPLE_LABELS (largest excess share) and its frames (probe rows) with their maps."""
    units = pd.read_parquet(P2_RESULTS / "concepts_units.parquet")
    ft = units[units.model == "finetuned"].sort_values("unit").reset_index(drop=True)
    probe = pd.read_parquet(P2_RESULTS / "concepts_probe.parquet")
    totals = unit_totals(cfg)
    assert len(totals) == len(probe)
    picks = []
    for label in EXAMPLE_LABELS:
        u = ft[ft.label == label].nlargest(1, "label_excess").iloc[0]
        order = np.argsort(-totals[:, int(u.unit)])
        concept_area = probe[f"area_{label.replace(' ', '_')}"].to_numpy()[order]
        order = order[(concept_area >= EXAMPLE_AREA[0]) & (concept_area <= EXAMPLE_AREA[1])]
        rows = list(pd.Series(order, index=probe.location.to_numpy()[order]).groupby(level=0, sort=False).first()
                    .iloc[:EXAMPLE_IMAGES])
        picks.append((int(u.unit), rows))
    all_rows = [r for _, rows in picks for r in rows]
    maps, images = unit_maps(cfg, all_rows)
    # the page export reloads these frames from data/: check that this gives the cached model input
    meta = load_metadata(cfg).set_index("image_id")
    transform = get_transform(cfg)
    for r in all_rows:
        path = resolve(cfg["paths"]["data_dir"]) / "train" / meta.loc[probe.image_id.iloc[r], "filename"]
        assert np.array_equal(D.load_input(path, transform).numpy(), images[r])
    k = EXAMPLE_IMAGES
    return {
        "ex_unit": np.array([u for u, _ in picks]), "ex_label": np.array(EXAMPLE_LABELS),
        "ex_rows": np.array([rows for _, rows in picks]),
        "ex_maps": np.stack([maps[i * k:(i + 1) * k, u] for i, (u, _) in enumerate(picks)]).astype(np.float32),
    }


def extract_votes_by_winner():
    """Vote split per habitat and winning class over the wrong composites (outputs/concepts/composite_votes.npz)."""
    z = np.load(C.CACHE / "composite_votes.npz")
    assert list(z["models"])[0] == "finetuned" and list(z["classes"]) == ["true", "pred"]
    res = pd.read_parquet(P2_RESULTS / "evidence_split.parquet",
                          columns=["bg_id", "y", "pred_y", "pred_logit", "true_logit", "pred_bias", "true_bias"])
    assert len(res) == len(z["area"])
    bgc = pd.read_parquet(P2_RESULTS / "concepts_backgrounds.parquet").set_index("bg_id")
    habitat = res.bg_id.map(bgc.habitat).to_numpy()
    winner = res.pred_y.map(common_names()).to_numpy()
    wrong = (res.pred_y != res.y).to_numpy()
    margin = z["parts"][:, 0, 1] - z["parts"][:, 0, 0]  # fine-tuned model, winner - true class, per concept
    lead = (res.pred_logit - res.true_logit).to_numpy()
    bias = (res.pred_bias - res.true_bias).to_numpy()
    concepts = [str(c) for c in z["concepts"]]
    rows = []
    for hab in HABITATS:
        h = wrong & (habitat == hab)
        counts = pd.Series(winner[h]).value_counts()
        for w in ["all", *counts.index[counts >= WINNER_MIN_N]]:
            sel = h if w == "all" else h & (winner == w)
            for k, c in enumerate(concepts):
                rows.append({"habitat": hab, "winner": w, "n": int(sel.sum()), "lead": lead[sel].mean(),
                             "bias": bias[sel].mean(), "concept": c, "area": z["area"][sel, k].mean(),
                             "votes": margin[sel, k].mean()})
    t = pd.DataFrame(rows)
    # same numbers as concepts.py for all wrong composites
    ref = pd.read_csv(P2_RESULTS / "concepts_composite_votes.csv")
    ref = ref[(ref.model == "finetuned") & (ref.light == "all") & (ref.outcome == "wrong")].set_index(
        ["habitat", "concept"])
    a = t[t.winner == "all"].set_index(["habitat", "concept"])
    assert (a.area - ref.area.reindex(a.index)).abs().max() < 1e-3
    assert (a.votes - ref.margin.reindex(a.index)).abs().max() < 1e-3
    t.to_csv(VOTES_BY_WINNER, index=False, float_format="%.5g")
    print(f"wrote {VOTES_BY_WINNER.relative_to(ROOT)}: {t.groupby('habitat').winner.nunique().to_dict()} winners")


@torch.no_grad()
def extract_ablation_dose(cfg):
    """Forest composite outcomes and real test accuracy with the k most tree- / ground-leaning units off (CPU).

    As concepts.ablation() (units, random controls, outcome codes), for every k in K_GRID. For k in concepts.TOP_K the
    sets and random draws are those of concepts.ablation() (its seed and order, replayed), so these rows reproduce
    results/concepts_ablation_*.csv; other k draw with their own seed.
    """
    units = pd.read_parquet(P2_RESULTS / "concepts_units.parquet")
    probe = pd.read_parquet(P2_RESULTS / "concepts_probe.parquet")
    area = {c: probe[f"area_{c}"].mean() for c in [c.replace(" ", "_") for c in C.PARTS]}
    u = units[units.model == "finetuned"]
    rng = np.random.default_rng(cfg["seed"])
    stored = {}
    for name, off in C.unit_sets(units, area)["finetuned"].items():
        stored[name] = (off, {ctl: C.random_sets(u, off, rng, matched)
                              for ctl, matched in [("random", False), ("random, matched", True)]})

    def top(name, k):
        excess = sum(u[f"share_{c}"] - area[c] for c in C.SETS[name])
        return u.loc[excess.nlargest(k).index, "unit"].to_numpy()

    cpu = torch.device("cpu")
    fc = load_model(cfg, cpu).fc
    W, b = fc.weight.detach().float(), fc.bias.detach().float()
    res = pd.read_parquet(P2_RESULTS / "evidence_split.parquet", columns=["bg_id", "y", "pred_y"])
    bgc = pd.read_parquet(P2_RESULTS / "concepts_backgrounds.parquet").set_index("bg_id")
    forest = np.flatnonzero((res.bg_id.map(bgc.habitat) == ABLATION_HABITAT).to_numpy())
    comp = torch.from_numpy(np.load(E.OUT_DIR / "features.npy", mmap_mode="r")[forest]).float()
    preds = pd.read_parquet(resolve(cfg["paths"]["predictions"]), columns=["split", "label", "location"])
    test = np.flatnonzero((preds.split == "test").to_numpy())
    real_forest = torch.tensor((preds.location.iloc[test].map(C.habitats().set_index("location").habitat)
                                == "forest").to_numpy())
    real = torch.from_numpy(np.load(resolve(cfg["paths"]["embeddings"]), mmap_mode="r")[test]).float()
    X = torch.cat([comp, real])
    del comp, real
    nc = len(forest)
    base = X @ W.T + b
    y_comp = torch.from_numpy(res.y.to_numpy()[forest])
    y_real = torch.from_numpy(preds.label.to_numpy()[test])
    group_of = C.group_of_class(cfg, cpu)
    same = (base[:nc].argmax(1).numpy() == res.pred_y.to_numpy()[forest]).mean()
    print(f"forest composites: predictions from the pooled features as stored: {same:.4%}")
    base_codes = C.outcome_codes(base[:nc].argmax(1), y_comp, group_of)
    was_empty = base_codes == OUTCOMES.index("empty")

    def evaluate(off):
        off = torch.as_tensor(np.asarray(off, dtype=np.int64))
        pred = (base - X[:, off] @ W[:, off].T).argmax(1)
        codes = C.outcome_codes(pred[:nc], y_comp, group_of)
        ok = (pred[nc:] == y_real).float()
        return {**{f"share|{o}": v for o, v in zip(OUTCOMES, (torch.bincount(codes, minlength=4) / nc).tolist())},
                "acc|test|all": ok.mean().item(), "acc|test|forest": ok[real_forest].mean().item(),
                **{f"empty_to|{o}": v for o, v in zip(OUTCOMES, torch.bincount(codes[was_empty], minlength=4).tolist())}}

    rows = [{"set": "none", "k": 0, "control": "none", "metric": key, "value": v, "sd": 0.0}
            for key, v in evaluate([]).items()]
    for si, (key, name) in enumerate(DOSE_SETS.items()):
        for k in K_GRID:
            off = top(name, k)
            if k in C.TOP_K:
                ref_off, draws = stored[f"{name} top {k}"]
                assert np.array_equal(off, ref_off)
            else:
                r = np.random.default_rng([cfg["seed"], si, k])
                draws = {ctl: C.random_sets(u, off, r, matched)
                         for ctl, matched in [("random", False), ("random, matched", True)]}
            runs = {"none": [evaluate(off)], **{ctl: [evaluate(d) for d in ds] for ctl, ds in draws.items()}}
            for ctl, results in runs.items():
                for metric in results[0]:
                    if ctl != "none" and metric.startswith("empty_to"):
                        continue
                    v = np.array([r_[metric] for r_ in results])
                    rows.append({"set": key, "k": k, "control": ctl, "metric": metric, "value": v.mean(),
                                 "sd": v.std()})
            print(f"{key} k={k}: forest empty {runs['none'][0]['share|empty']:.1%}, "
                  f"random {np.mean([r_['share|empty'] for r_ in runs['random']]):.1%}")
    t = pd.DataFrame(rows)

    # k in TOP_K: as in results/concepts_ablation_*.csv (CPU here, GPU there: a near tie may rarely flip)
    comp_ref = pd.read_csv(P2_RESULTS / "concepts_ablation_composites.csv")
    comp_ref = comp_ref[(comp_ref.model == "finetuned") & (comp_ref.light == "all")
                        & (comp_ref.habitat == ABLATION_HABITAT)]
    real_ref = pd.read_csv(P2_RESULTS / "concepts_ablation_real.csv")
    real_ref = real_ref[(real_ref.model == "finetuned") & (real_ref.metric == "acc") & (real_ref.split == "test")]
    worst = 0.0
    for key, name in DOSE_SETS.items():
        for k in C.TOP_K:
            for ctl in ["none", "random", "random, matched"]:
                mine = t[(t.set == key) & (t.k == k) & (t.control == ctl)].set_index("metric").value
                c = comp_ref[(comp_ref.units_off == f"{name} top {k}") & (comp_ref.control == ctl)]
                rr = real_ref[(real_ref.units_off == f"{name} top {k}") & (real_ref.control == ctl)]
                for o, s in zip(c.outcome, c.share):
                    worst = max(worst, abs(mine[f"share|{o}"] - s))
                for h, v in zip(rr.habitat, rr.value):
                    if h in ("all", "forest"):
                        worst = max(worst, abs(mine[f"acc|test|{h}"] - v))
    print(f"k in {C.TOP_K}: largest difference to results/concepts_ablation_*.csv {worst:.2e}")
    assert worst < 5e-4
    t.to_csv(ABLATION_DOSE, index=False, float_format="%.6g")
    print(f"wrote {ABLATION_DOSE.relative_to(ROOT)} ({len(t)} rows)")


def extract(cfg):
    """Write results/story_* from the Phase 2 caches in outputs/ (see the module docstring)."""
    _, res, _, _ = load()
    arrays = {**extract_evidence(res), **example_unit_picks(cfg)}
    np.savez_compressed(EXTRACTS, **arrays)
    print(f"wrote {EXTRACTS.relative_to(ROOT)} ({EXTRACTS.stat().st_size / 1024:.0f} kB)")
    extract_votes_by_winner()
    extract_ablation_dose(cfg)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    if sys.argv[1:] == ["extract"]:
        extract(cfg)
        return
    assert all(f.exists() for f in [EXTRACTS, VOTES_BY_WINNER, ABLATION_DOSE]), \
        "results/story_* missing: run `python phase2/pair_b/src/export_story_data.py extract` once"
    bg, res, controls, cut_table = load()

    recap = {
        "map": export_map(bg, res),
        "controls": export_controls(res, controls),
        "cells": export_cells(res, controls),
    }
    write_json("recap.json", recap, compact=True)
    meta = export_meta(cfg, bg, res, controls, cut_table)
    hero = export_hero(cfg, bg, res, cut_table, recap["map"]["views"]["elephant"]["habitats"])
    evidence = export_evidence(cfg, bg, hero)
    units = export_units(cfg)
    votes = export_votes()
    ablation = export_ablation()

    for f in hero["frames"]:
        print(f"hero {f['habitat']:6s} bg {f['bg_id']:3d} cam {f['camera']:3d}: {f['answer']} {f['confidence']:.0%}")
    print(f"{meta['composites']:,} composites, {meta['backgrounds']} backgrounds")
    w = evidence["wrong"]
    print(f"ch2: {w['n']:,} wrong composites, lead on the animal {w['animal']:+.2f}, background {w['background']:+.2f}")
    print(f"ch3: labelled by total activation {units['counts']['finetuned']}, by the paper's rule "
          f"{units['paper_finetuned_layer4_labelled']}; examples {[e['unit'] for e in units['examples']]}")
    f = votes["habitats"]["forest"]
    print(f"ch4: forest lead {f['lead']}, " + ", ".join(f"{c['concept']} {c['area']:.0%} area / {c['share']:.0%} votes"
                                                        for c in f["folded"]))
    r = ablation["runs"]
    print(f"ch5: forest empty {r['as is']['share']['empty']:.1%} -> {r['units off']['share']['empty']:.1%} "
          f"(random {r['random']['share']['empty']:.1%}); test acc {r['as is']['acc']['test|all']['acc']:.2%} -> "
          f"{r['units off']['acc']['test|all']['acc']:.2%}")
    for f in sorted(IMG.glob("*.webp")):
        print(f"{f.relative_to(ROOT)}: {f.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
