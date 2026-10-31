"""Background-swap experiment: classify every cut-out on every 3x3 cell of empty backgrounds, plus controls.

Backgrounds: empty id_test/test frames, sampled evenly over the six habitats (SegFormer habitat x day/night).

    python phase1/pair_b/src/run_composites.py   # needs results/camera_openness.csv (measure_background.py)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import load_config, resolve
from common.data import IWildCamImages, normalize, set_seed
from common.model import get_device, load_model
from composite import PAIR_B, SPECIES_GROUPS, camera_habitats, is_infrared, load_cutouts, paste

PER_HABITAT = 80
PER_CAMERA = 20
OWN_CAMERA = 20             # frames per cut-out for the "own camera" control
BATCH = 64


def load_images(df, cfg):
    loader = DataLoader(
        IWildCamImages(df, cfg),
        batch_size=BATCH,
        num_workers=cfg["inference"]["num_workers"]
    )
    return torch.cat([x for x, _ in loader])


def empty_frames(cfg, splits):
    """Frames labelled and predicted empty, one per sequence."""
    cols = ["id", "split", "label", "prediction", "location", "seq_id", "filename"]
    preds = pd.read_parquet(resolve(cfg["paths"]["predictions"]), columns=cols)
    keep = preds["split"].isin(splits) & (preds["label"] == 0) & (preds["prediction"] == 0)
    frames = preds[keep]
    return frames.drop_duplicates("seq_id").rename_axis("emb_row").reset_index()   # row in predictions.parquet


def pick_backgrounds(cfg):
    pool = empty_frames(cfg, ["id_test", "test"]).merge(camera_habitats(), on="location")
    loader = DataLoader(
        IWildCamImages(pool, cfg),
        batch_size=128,
        num_workers=cfg["inference"]["num_workers"]
    )
    pool["infrared"] = torch.cat([is_infrared(x) for x, _ in loader]).numpy()
    cell = [pool["habitat"], pool["infrared"]]
    pool = pool.sample(frac=1, random_state=cfg["seed"])
    pool["turn"] = pool.groupby(cell + [pool["location"]]).cumcount()
    pool = pool[pool["turn"] < PER_CAMERA].sort_values("turn", kind="stable")
    bg = pool[pool.groupby(cell).cumcount() < PER_HABITAT]
    return bg.sort_values("emb_row").drop(columns="turn").reset_index(drop=True)


def classify(model, images, cutout, row, col, gray, y, in_group):
    """Paste the cut-out into a batch of backgrounds and classify; one dict of columns per batch."""
    probs = model(normalize(paste(images, cutout, row, col, gray=gray))).softmax(1)
    return {
        "row": row,
        "col": col,
        "y": y,
        "pred_y": probs.argmax(1).cpu().numpy(),
        "p_pred": probs.max(1).values.cpu().numpy(),
        "p_species": probs[:, y].cpu().numpy(),
        "p_group": probs[:, in_group].sum(1).cpu().numpy(),
        "p_empty": probs[:, 0].cpu().numpy(),
    }


def finish(results, groups):
    results = pd.concat(results, ignore_index=True)
    results["species"] = results["y"].map(groups["common_name"])
    results["pred_species"] = results["pred_y"].map(groups["common_name"])
    results["pred_group"] = results["pred_y"].map(groups["group"])
    results["outcome"] = np.select(
        [
            results["pred_y"] == results["y"],
            results["pred_group"] == results["group"],
            results["pred_y"] == 0,
        ],
        ["exact species", "same group", "empty"],
        "other group"
    )
    for c in ["p_pred", "p_species", "p_group", "p_empty"]:
        results[c] = results[c].astype("float32").round(4)
    return results


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)

    bg = pick_backgrounds(cfg)
    images = load_images(bg, cfg).to(device)
    bg["brightness"] = (images.float().mean((1, 2, 3)) / 255).cpu().numpy().round(3)
    bg["light"] = np.where(bg["infrared"], "night", "day")
    columns = [
        "bg_id", "emb_row", "id", "split", "location", "filename", "infrared", "brightness", "light", "openness",
        "habitat",
    ]
    bg = bg.rename_axis("bg_id").reset_index()[columns]
    bg.to_parquet(PAIR_B / "results" / "backgrounds.parquet", index=False)
    counts = bg.groupby(["light", "habitat", "split"]).agg(frames=("id", "size"), cameras=("location", "nunique"))
    print(counts.unstack().to_string())

    groups = pd.read_csv(SPECIES_GROUPS).set_index("y")
    in_group = {
        g: torch.tensor(groups.index[groups["group"] == g].to_numpy(), device=device)
        for g in groups["group"].unique()
    }
    table = pd.read_csv(PAIR_B / "cutouts" / "cutouts.csv")
    class_of = groups.reset_index().set_index("name")["y"]
    species_y = dict(zip(zip(table["group"], table["variant"]), table["species"].map(class_of)))

    infrared = torch.from_numpy(bg["infrared"].to_numpy().copy()).to(device)
    bg_ids = bg["bg_id"].to_numpy()
    model = load_model(cfg, device)
    cutouts = load_cutouts()
    results = []
    with torch.no_grad():
        # sanity check: the backgrounds alone should be predicted empty
        plain = torch.cat([
            model(normalize(images[i:i + BATCH])).argmax(1)
            for i in range(0, len(images), BATCH)
        ])
        print(f"backgrounds without animal predicted empty: {(plain == 0).float().mean():.1%}")
        for (group, variant), cutout in tqdm(cutouts.items(), desc="cut-outs"):
            y = species_y[group, variant]
            for row in range(3):
                for col in range(3):
                    for start in range(0, len(images), BATCH):
                        batch = slice(start, start + BATCH)
                        cols = classify(
                            model,
                            images[batch],
                            cutout,
                            row,
                            col,
                            gray=infrared[batch],
                            y=y,
                            in_group=in_group[group]
                        )
                        results.append(pd.DataFrame({
                            "bg_id": bg_ids[batch],
                            "group": group,
                            "variant": variant,
                            **cols,
                        }))
    results = finish(results, groups)
    results.to_parquet(PAIR_B / "results" / "composites.parquet", index=False)
    print(f"{len(results):,} composites")
    print(results["outcome"].value_counts(normalize=True).round(3).to_string())
    del images

    # controls: grey frame (colour and greyscale cut-out) and empty frames of the cut-out's own camera
    grey = torch.full((2, 3, 448, 448), 128, dtype=torch.uint8, device=device)
    own = empty_frames(cfg, ["train", "id_val", "val", "id_test", "test"])
    own = own[own["location"].isin(table["location"])]
    own = own.sample(frac=1, random_state=cfg["seed"])
    predictions = pd.read_parquet(resolve(cfg["paths"]["predictions"]), columns=["id", "seq_id"])
    seq_of = predictions.set_index("id")["seq_id"]
    controls = []
    with torch.no_grad():
        grey_class = model(normalize(grey[:1])).argmax(1).item()
        print(f"grey frame without animal: class {grey_class} (0 = empty)")
        for _, r in tqdm(table.iterrows(), total=len(table), desc="controls"):
            key = (r["group"], r["variant"])
            y = species_y[key]
            same_camera = own["location"] == r["location"]
            other_sequence = own["seq_id"] != seq_of.get(r["image_id"])
            frames = own[same_camera & other_sequence].head(OWN_CAMERA)
            if len(frames):
                imgs = load_images(frames, cfg).to(device)
                ir = is_infrared(imgs)
            else:
                imgs, ir = None, None

            for row in range(3):
                for col in range(3):
                    cols = classify(
                        model,
                        grey,
                        cutouts[key],
                        row,
                        col,
                        gray=torch.tensor([False, True]),
                        y=y,
                        in_group=in_group[r["group"]]
                    )
                    controls.append(pd.DataFrame({
                        "control": ["grey frame", "grey frame, greyscale cut-out"],
                        "filename": None,
                        "infrared": [False, True],
                        "group": r["group"],
                        "variant": r["variant"],
                        **cols,
                    }))
                    if imgs is not None:
                        cols = classify(
                            model,
                            imgs,
                            cutouts[key],
                            row,
                            col,
                            gray=ir,
                            y=y,
                            in_group=in_group[r["group"]]
                        )
                        controls.append(pd.DataFrame({
                            "control": "own camera",
                            "filename": frames["filename"].to_numpy(),
                            "infrared": ir.cpu().numpy(),
                            "group": r["group"],
                            "variant": r["variant"],
                            **cols,
                        }))
    controls = finish(controls, groups)
    controls.to_parquet(PAIR_B / "results" / "controls.parquet", index=False)

    shares = controls.groupby("control")["outcome"].value_counts(normalize=True).unstack().round(3)
    print(shares.to_string())
    own_camera = controls[controls["control"] == "own camera"]
    frames_per_cutout = own_camera.groupby(["group", "variant"]).size().div(9)  # 9 positions per frame
    print("own-camera frames per cut-out:", frames_per_cutout.describe().round(1).to_dict())


if __name__ == "__main__":
    (PAIR_B / "results").mkdir(exist_ok=True)
    main()
