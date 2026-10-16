"""Run the model on the configured splits and save one row per image.

Columns: id, split, label, label_name, prediction, prediction_name, confidence,
prob_0 ... prob_181, plus `dataset.metadata_columns` from the config.
Optional embeddings are saved as a separate .npy, row-aligned with the parquet file.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import resolve
from common.data import IWildCamImages, load_class_names, load_metadata, normalize, select_split, set_seed
from common.model import get_device, load_model, split_model


@torch.no_grad()
def extract_features(
    cfg: dict, df: pd.DataFrame, backbone: nn.Module, device: torch.device, desc: str | None = None
) -> torch.Tensor:
    """Compute backbone features for a set of images, with a progress bar.

    Args:
        cfg: Shared config; uses `inference.batch_size` and `inference.num_workers`.
        df: Metadata rows of the images, e.g. from `select_split`.
        backbone: Feature extractor, e.g. from `split_model` or `load_imagenet_backbone`.
        device: Device the backbone is on.
        desc: Label shown in front of the progress bar, e.g. the split name.

    Returns:
        Features [N, 2048] on the CPU, in the row order of `df`.
    """
    loader = DataLoader(
        IWildCamImages(df, cfg),
        batch_size=cfg["inference"]["batch_size"],
        num_workers=cfg["inference"]["num_workers"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    feats = []
    with tqdm(total=len(df), desc=desc, unit="img") as bar:
        for images, _ in loader:
            feats.append(backbone(normalize(images.to(device, non_blocking=True))).cpu())
            bar.update(len(images))
    return torch.cat(feats)


@torch.no_grad()
def predict(
    cfg: dict,
    df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    return_embeddings: bool = False,
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict class probabilities for a set of images, with a progress bar.

    Args:
        cfg: Shared config; uses `inference.batch_size` and `inference.num_workers`.
        df: Metadata rows of the images, e.g. from `select_split`.
        model: Fine-tuned model from `load_model`.
        device: Device the model is on.
        return_embeddings: Also return the penultimate-layer features.
        desc: Label shown in front of the progress bar, e.g. the split name.

    Returns:
        `(probs, embeddings)`: softmax probabilities [N, num_classes] and
        features [N, 2048] (None unless `return_embeddings`), in the row order of `df`.
    """
    backbone, head = split_model(model)
    feats = extract_features(cfg, df, backbone, device, desc=desc)
    probs = torch.softmax(head(feats.to(device)), dim=1).cpu().numpy()
    return probs, (feats.numpy() if return_embeddings else None)


def prediction_table(cfg: dict, df: pd.DataFrame, split: str, probs: np.ndarray, names: pd.Series) -> pd.DataFrame:
    """Build the per-image predictions table for one split (columns: see module docstring).

    Args:
        cfg: Shared config; uses `dataset.metadata_columns`.
        df: Metadata rows of the images, in the row order of `probs`.
        split: WILDS split name stored in the `split` column.
        probs: Class probabilities [N, num_classes].
        names: Species names by class index, from `load_class_names`.

    Returns:
        One row per image with a fresh 0..n-1 index.
    """
    pred = probs.argmax(axis=1)
    out = pd.DataFrame({
        "id": df["image_id"],
        "split": split,
        "label": df["y"],
        "label_name": df["label_name"],
        "prediction": pred,
        "prediction_name": pd.Series(pred).map(names),
        "confidence": probs.max(axis=1),
    })
    prob_cols = pd.DataFrame(probs, columns=[f"prob_{c}" for c in range(probs.shape[1])])
    return pd.concat([out, df[cfg["dataset"]["metadata_columns"]], prob_cols], axis=1)


def save_predictions(
    cfg: dict, frames: list[pd.DataFrame], embeddings: list[np.ndarray] | None, predictions_key: str, embeddings_key: str
) -> pd.DataFrame:
    """Concatenate per-split tables and write them (and optional embeddings) to disk.

    Args:
        cfg: Shared config; the output paths are `paths.<predictions_key>` and `paths.<embeddings_key>`.
        frames: Per-split tables from `prediction_table`.
        embeddings: Per-split features [N_split, 2048] in the same order, or None to skip.
        predictions_key: Config key of the parquet file, e.g. "predictions".
        embeddings_key: Config key of the .npy file, e.g. "embeddings".

    Returns:
        The concatenated predictions table that was written.
    """
    result = pd.concat(frames, ignore_index=True)
    out_path = resolve(cfg["paths"][predictions_key])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")
    if embeddings is not None:
        emb_path = resolve(cfg["paths"][embeddings_key])
        np.save(emb_path, np.concatenate(embeddings))
        print(f"Wrote embeddings to {emb_path}")
    return result


def run_inference(cfg: dict) -> pd.DataFrame:
    """Predict on all configured splits and write the shared predictions file.

    Writes `paths.predictions` (parquet) and, if `inference.save_embeddings`
    is set, `paths.embeddings` (.npy, row-aligned with the parquet file).

    Args:
        cfg: Shared config.

    Returns:
        The predictions table that was written (columns: see module docstring).
    """
    set_seed(cfg["seed"])
    device = get_device(cfg)
    model = load_model(cfg, device)
    meta = load_metadata(cfg)
    names = load_class_names(cfg)
    save_emb = cfg["inference"]["save_embeddings"]

    frames, all_embs = [], []
    for split in cfg["dataset"]["splits"]:
        df = select_split(meta, split, cfg)
        probs, embs = predict(cfg, df, model, device, return_embeddings=save_emb, desc=split)
        frames.append(prediction_table(cfg, df, split, probs, names))
        all_embs.append(embs)
    return save_predictions(cfg, frames, all_embs if save_emb else None, "predictions", "embeddings")
