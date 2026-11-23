"""Evidence split of the Phase 1 background-swap composites: how much of a logit sits on the pasted animal.

A ResNet-50 logit is the mean of the class activation map (CAM) over the 14 x 14 layer4 grid plus the bias:

    z_c = 1/196 sum_ij CAM_c(i, j) + b_c,    CAM_c(i, j) = sum_k W[c, k] A_k(i, j)

with A the layer4 features [2048, 14, 14] (448 x 448 input) and W, b the fc layer. With m the cut-out's alpha mask
average-pooled to the same grid (share of each 32 x 32 cell covered by the animal, soft at the edges):

    inside_c  = 1/196 sum_ij m(i, j) CAM_c(i, j)          evidence on the animal's cells
    outside_c = 1/196 sum_ij (1 - m(i, j)) CAM_c(i, j)    evidence on the background cells
    z_c = inside_c + outside_c + b_c

The split is exact, but "inside" is evidence located on the animal, not evidence from animal pixels only: a layer4
cell sees far beyond its own 32 x 32 patch (context around the animal; animal edges reach neighbouring cells).

It is done for the true class y and the predicted class of two models:
- the WILDS fine-tuned ResNet-50 (`common.model.load_model`): columns without prefix;
- the linear probe on frozen ImageNet features (`common.model.load_probe_model`): columns `probe_*`. The probe
  standardises the pooled features f, logits = W (f - mu) / sigma + b, i.e. a linear layer on f with weights
  W / sigma and bias b - W mu / sigma. CAM and bias use these effective weights, so in both models a cell with zero
  features carries no evidence. (Split in the standardised space instead: inside - mask_area * (W mu / sigma)_c,
  bias + (W mu / sigma)_c.)

The composites are rebuilt exactly as phase1/pair_b/src/run_composites.py does (same backgrounds, cut-outs, cells,
greyscale cut-out on infrared frames, batches of 64), and the predictions of the fine-tuned model are checked against
phase1/pair_b/results/composites.parquet.

Outputs, rows in the order of composites.parquet (cut-out as in cutouts.csv, then cell row * 3 + col, then bg_id):
- phase2/pair_b/results/evidence_split.parquet: one row per composite. Keys bg_id, group, variant, row, col; y;
  mask_area (share of the frame covered by the animal = mean of m); per model pred_y, p_pred, p_true and, for the
  true class (true_*) and the predicted class (pred_*), logit, inside, outside and bias
  (logit = inside + outside + bias).
- outputs/phase2_pair_b/ (row-aligned with the parquet):
  features.npy, probe_features.npy: pooled layer4 features [N, 2048] (the penultimate features), float16
  cam_true.npy, cam_pred.npy, probe_cam_true.npy, probe_cam_pred.npy: CAM of the true / predicted class [N, 14, 14],
  float32 (mean over the grid + bias = logit)
  masks.npz: the pooled masks m [405, 14, 14] of every cut-out and cell, with keys group, variant, row, col

    python phase2/pair_b/src/evidence_split.py              # ~25 min on a (shared) RTX 5080, 3.3 GB GPU memory
    python phase2/pair_b/src/evidence_split.py --no-probe   # fine-tuned model only, about half the time
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ROOT, load_config, resolve
from common.data import IWildCamImages, normalize, set_seed
from common.model import get_device, load_model, load_probe_model

sys.path.insert(0, str(ROOT / "phase1" / "pair_b" / "src"))
from composite import CELL, PAIR_B, SIZE, SPECIES_GROUPS, is_infrared, load_cutouts, paste  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = resolve("outputs") / "phase2_pair_b"
BATCH = 64      # as in run_composites.py, so cuDNN sees the same batches
GRID = 14       # layer4 grid for 448 x 448 input
PARTS = ["logit", "inside", "outside", "bias"]


class Head:
    """A ResNet whose layer4 output is kept on every forward pass, plus its linear head on the pooled features.

    Args:
        model: Full model, normalized images -> logits [N, C].
        layer4: The model's layer4 module (its output is kept).
        weight: Linear head on the pooled layer4 features [C, 2048].
        bias: Its bias [C].
    """

    def __init__(self, model, layer4, weight, bias):
        self.model = model
        self.weight = weight
        self.bias = bias
        self.features = None
        layer4.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.features = output

    def __call__(self, x):
        """Logits [N, C] and layer4 features [N, 2048, 14, 14] of the unmodified model."""
        logits = self.model(x)
        return logits, self.features


def load_heads(cfg, device, probe=True):
    """{"": fine-tuned model, "probe_": linear probe}; the key is the column prefix."""
    model = load_model(cfg, device)
    heads = {"": Head(model, model.layer4, model.fc.weight, model.fc.bias)}
    if probe:
        probe_model = load_probe_model(cfg, device)
        backbone, lin = probe_model                     # backbone: ResNet children up to avgpool, + Flatten
        assert isinstance(backbone[8], torch.nn.AdaptiveAvgPool2d)         # so backbone[7] is layer4
        assert torch.equal(lin.classes.cpu(), torch.arange(cfg["model"]["num_classes"]))  # no -inf classes
        w = lin.linear.weight.double() / lin.std.double()
        b = lin.linear.bias.double() - w @ lin.mean.double()
        heads["probe_"] = Head(probe_model, backbone[7], w.float(), b.float())
    return heads


def species_classes():
    """{(group, variant): class index y of the cut-out's species}, as in run_composites.py."""
    groups = pd.read_csv(SPECIES_GROUPS).set_index("y")
    table = pd.read_csv(PAIR_B / "cutouts" / "cutouts.csv")
    class_of = groups.reset_index().set_index("name")["y"]
    return dict(zip(zip(table["group"], table["variant"]), table["species"].map(class_of)))


def load_backgrounds(cfg, device):
    """The Phase 1 backgrounds (backgrounds.parquet) and their images, uint8 [N, 3, 448, 448] on `device`."""
    bg = pd.read_parquet(PAIR_B / "results" / "backgrounds.parquet")
    assert (bg["bg_id"].to_numpy() == np.arange(len(bg))).all()
    loader = DataLoader(IWildCamImages(bg, cfg), batch_size=BATCH, num_workers=cfg["inference"]["num_workers"])
    images = torch.cat([x for x, _ in loader]).to(device)
    # same frames as in Phase 1: stored brightness and infrared flag
    brightness = (images.float().mean((1, 2, 3)) / 255).cpu().numpy()
    assert np.abs(brightness - bg["brightness"].to_numpy()).max() < 1e-3
    assert (is_infrared(images).cpu().numpy() == bg["infrared"].to_numpy()).all()
    return bg, images


def pooled_mask(cutout, row, col):
    """Alpha mask of the cut-out where `paste` puts it, average-pooled to the layer4 grid: [14, 14] in [0, 1]."""
    rgb, alpha = cutout
    h, w = alpha.shape[1:]
    y0 = min(max(round((row + 0.5) * CELL - h / 2), 0), SIZE - h)   # placement as in paste()
    x0 = min(max(round((col + 0.5) * CELL - w / 2), 0), SIZE - w)
    canvas = torch.zeros(1, 1, SIZE, SIZE)
    canvas[..., y0:y0 + h, x0:x0 + w] = alpha
    # same placement as paste(): a white cut-out on a black frame gives round(255 * alpha)
    white = paste(torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8), (torch.full_like(rgb, 255), alpha), row, col)
    assert torch.equal(white[0, 0], (255 * canvas[0, 0]).round().to(torch.uint8))
    return F.avg_pool2d(canvas, SIZE // GRID)[0, 0]


def split(features, head, classes, mask):
    """CAM [B, 14, 14] and inside part, outside part and bias [B] of `classes` [B]; mask [14, 14]."""
    cam = torch.einsum("bkij,bk->bij", features, head.weight[classes])
    inside = (cam * mask).sum((1, 2)) / GRID**2
    outside = (cam * (1 - mask)).sum((1, 2)) / GRID**2
    return cam, inside, outside, head.bias[classes]


def run(images, infrared, cutouts, species_y, heads, out_dir):
    """Classify every composite; returns the per-composite table and the masks, writes the arrays to `out_dir`."""
    n_bg = len(images)
    n = len(cutouts) * 9 * n_bg
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix in heads:
        arrays[prefix + "features"] = np.lib.format.open_memmap(
            out_dir / f"{prefix}features.npy", mode="w+", dtype=np.float16, shape=(n, 2048))
        for cls in ["true", "pred"]:
            arrays[f"{prefix}cam_{cls}"] = np.lib.format.open_memmap(
                out_dir / f"{prefix}cam_{cls}.npy", mode="w+", dtype=np.float32, shape=(n, GRID, GRID))

    cols = {
        "bg_id": np.tile(np.arange(n_bg), len(cutouts) * 9),
        "group": np.repeat([g for g, _ in cutouts], 9 * n_bg),
        "variant": np.repeat([v for _, v in cutouts], 9 * n_bg),
        "row": np.tile(np.repeat([0, 0, 0, 1, 1, 1, 2, 2, 2], n_bg), len(cutouts)),
        "col": np.tile(np.repeat([0, 1, 2] * 3, n_bg), len(cutouts)),
        "y": np.repeat([species_y[key] for key in cutouts], 9 * n_bg),
        "mask_area": np.zeros(n, np.float32),
    }
    for prefix in heads:
        cols[prefix + "pred_y"] = np.zeros(n, np.int64)
        for c in ["p_pred", "p_true"] + [f"{cls}_{part}" for cls in ["true", "pred"] for part in PARTS]:
            cols[prefix + c] = np.zeros(n, np.float32)

    masks = []
    i = 0
    with torch.no_grad():
        for (group, variant), cutout in tqdm(cutouts.items(), desc="cut-outs"):
            y = species_y[group, variant]
            for row in range(3):
                for col in range(3):
                    mask = pooled_mask(cutout, row, col)
                    masks.append({"group": group, "variant": variant, "row": row, "col": col, "mask": mask.numpy()})
                    mask = mask.to(images.device)
                    for start in range(0, n_bg, BATCH):
                        batch = slice(start, start + BATCH)
                        x = normalize(paste(images[batch], cutout, row, col, gray=infrared[batch]))
                        rows = slice(i, i + len(x))
                        i += len(x)
                        cols["mask_area"][rows] = mask.mean().item()
                        for prefix, head in heads.items():
                            logits, features = head(x)
                            probs = logits.softmax(1)
                            pred = probs.argmax(1)      # as in run_composites.py
                            true = torch.full_like(pred, y)
                            out = {
                                "pred_y": pred,
                                "p_pred": probs.max(1).values,
                                "p_true": probs[:, y],
                            }
                            for cls, classes in [("true", true), ("pred", pred)]:
                                cam, inside, outside, bias = split(features, head, classes, mask)
                                out[f"{cls}_logit"] = logits.gather(1, classes[:, None])[:, 0]
                                out[f"{cls}_inside"] = inside
                                out[f"{cls}_outside"] = outside
                                out[f"{cls}_bias"] = bias
                                arrays[f"{prefix}cam_{cls}"][rows] = cam.cpu().numpy()
                            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
                            arrays[prefix + "features"][rows] = pooled.half().cpu().numpy()
                            for c, v in out.items():
                                cols[prefix + c][rows] = v.cpu().numpy()
    assert i == n
    for a in arrays.values():
        a.flush()
    masks = pd.DataFrame(masks)
    np.savez(
        out_dir / "masks.npz",
        group=masks["group"].to_numpy(str), variant=masks["variant"].to_numpy(), row=masks["row"].to_numpy(),
        col=masks["col"].to_numpy(), mask=np.stack(masks["mask"]),
    )
    return pd.DataFrame(cols), masks


def check_predictions(res):
    """Fine-tuned predictions against phase1/pair_b/results/composites.parquet."""
    ref = pd.read_parquet(PAIR_B / "results" / "composites.parquet")
    assert len(ref) == len(res)
    for k in ["bg_id", "group", "variant", "row", "col", "y"]:
        assert (res[k].to_numpy() == ref[k].to_numpy()).all(), k
    same = res["pred_y"].to_numpy() == ref["pred_y"].to_numpy()
    print(f"predicted class as in composites.parquet: {same.sum():,} / {len(same):,}")
    for mine, theirs in [("p_pred", "p_pred"), ("p_true", "p_species")]:
        diff = np.abs(res[mine].round(4) - ref[theirs]).max()
        print(f"  {mine} vs {theirs}: max abs difference {diff:.1e} (Phase 1 rounded to 4 decimals)")
    return same.all()


def check_split(res, heads, images, infrared, cutouts, out_dir, n=6, seed=0):
    """logit = inside + outside + bias: on all rows (float32), and on a few rebuilt composites in float64.

    A composite is rebuilt in its original batch of 64: with TF32 convolutions (the cuDNN default, also in
    Phase 1) a batch of one gives slightly different features (logits differ by ~0.005 fine-tuned, ~0.07 probe).
    """
    for prefix in heads:
        for cls in ["true", "pred"]:
            c = prefix + cls
            gap = (res[c + "_inside"] + res[c + "_outside"] + res[c + "_bias"] - res[c + "_logit"]).abs()
            print(f"{c + '_':>12}: max |inside + outside + bias - logit| over all rows = {gap.max():.1e}")

    names = [prefix + f"cam_{cls}" for prefix in heads for cls in ["true", "pred"]]
    maps = {k: np.load(out_dir / f"{k}.npy", mmap_mode="r") for k in names}
    picks = np.random.default_rng(seed).choice(len(res), n, replace=False)
    lines = []
    with torch.no_grad():
        for idx in picks:
            r = res.iloc[idx]
            cutout = cutouts[r.group, r.variant]
            start = r.bg_id // BATCH * BATCH
            batch = slice(start, start + BATCH)
            x = normalize(paste(images[batch], cutout, r.row, r.col, gray=infrared[batch]))
            mask = pooled_mask(cutout, r.row, r.col).double().to(images.device)
            for prefix, head in heads.items():
                logits, features = head(x)       # unmodified forward pass
                j = r.bg_id - start
                for cls, k in [("true", r.y), ("pred", r[prefix + "pred_y"])]:
                    cam = torch.einsum("kij,k->ij", features[j].double(), head.weight[k].double())
                    inside = (cam * mask).sum().item() / GRID**2
                    outside = (cam * (1 - mask)).sum().item() / GRID**2
                    bias = head.bias[k].double().item()
                    logit = logits[j, k].item()
                    lines.append({
                        "row": idx, "model": prefix or "fine-tuned", "class": f"{cls} ({k})",
                        "logit (model)": logit, "inside": inside, "outside": outside, "bias": bias,
                        "sum - logit": inside + outside + bias - logit,
                        "parts: stored - recomputed": max(
                            abs(r[f"{prefix}{cls}_inside"] - inside), abs(r[f"{prefix}{cls}_outside"] - outside)
                        ),
                        "map: stored - recomputed": np.abs(maps[f"{prefix}cam_{cls}"][idx] - cam.cpu().numpy()).max(),
                    })
    lines = pd.DataFrame(lines)
    print(lines.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return lines


def summarize(res, bg, prefix=""):
    """Mean parts of the predicted-class logit, correct vs. wrong, per habitat."""
    df = res.merge(bg[["bg_id", "habitat", "light"]], on="bg_id")
    df["correct"] = df[prefix + "pred_y"] == df["y"]
    p = prefix + "pred_"
    table = df.groupby(["habitat", "light", "correct"]).agg(
        n=("y", "size"),
        mask_area=("mask_area", "mean"),
        inside=(p + "inside", "mean"),
        outside=(p + "outside", "mean"),
        bias=(p + "bias", "mean"),
        logit=(p + "logit", "mean"),
    )
    print(f"\n{prefix or 'fine-tuned '}predicted-class logit, means per habitat:")
    print(table.round(3).to_string())
    return table


def compact(res):
    """Copy for the parquet file in git: logits and parts to 0.001, probabilities and mask area to 0.0001.

    Both are far below the run-to-run noise of ~0.005 (TF32, see check_split); Phase 1 rounds probabilities
    to 4 decimals too. Shrinks the file from ~18 MB to ~7 MB. The checks run on the unrounded values.
    """
    out = res.copy()
    for c in out.columns:
        if out[c].dtype == np.float32:
            fine = c == "mask_area" or c.startswith(("p_", "probe_p_"))
            out[c] = out[c].round(4 if fine else 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-probe", action="store_true", help="skip the linear-probe model")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)
    bg, images = load_backgrounds(cfg, device)
    infrared = torch.from_numpy(bg["infrared"].to_numpy().copy()).to(device)
    heads = load_heads(cfg, device, probe=not args.no_probe)
    cutouts = load_cutouts()

    res, _ = run(images, infrared, cutouts, species_classes(), heads, OUT_DIR)
    RESULTS.mkdir(exist_ok=True)
    compact(res).to_parquet(RESULTS / "evidence_split.parquet", index=False, compression="zstd", compression_level=9)
    print(f"{len(res):,} composites -> {RESULTS / 'evidence_split.parquet'}, arrays in {OUT_DIR}")

    check_predictions(res)
    check_split(res, heads, images, infrared, cutouts, OUT_DIR)
    for prefix in heads:
        summarize(res, bg, prefix)


if __name__ == "__main__":
    main()
