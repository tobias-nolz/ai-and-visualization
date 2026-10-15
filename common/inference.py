"""Run the model on the configured splits and save one row per image.

Columns: id, split, label, label_name, prediction, prediction_name, confidence,
prob_0 ... prob_181, plus `dataset.metadata_columns` from the config.
Optional embeddings are saved as a separate .npy, row-aligned with the parquet file.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import resolve
from common.data import IWildCamImages, load_class_names, load_metadata, select_split, set_seed
from common.model import get_device, load_model, split_model


@torch.no_grad()
def predict(cfg: dict, df: pd.DataFrame, model, device, return_embeddings: bool = False):
    """Return (probabilities [N, C], embeddings [N, 2048] or None) for the rows of `df`."""
    backbone, head = split_model(model)
    loader = DataLoader(
        IWildCamImages(df, cfg),
        batch_size=cfg["inference"]["batch_size"],
        num_workers=cfg["inference"]["num_workers"],
        shuffle=False,
    )
    probs, embs = [], []
    for images, _ in loader:
        feats = backbone(images.to(device))
        probs.append(torch.softmax(head(feats), dim=1).cpu())
        if return_embeddings:
            embs.append(feats.cpu())
    probs = torch.cat(probs).numpy()
    return probs, (torch.cat(embs).numpy() if return_embeddings else None)


def run_inference(cfg: dict) -> pd.DataFrame:
    """Predict on all configured splits and write the shared predictions file."""
    set_seed(cfg["seed"])
    device = get_device(cfg)
    model = load_model(cfg, device)
    meta = load_metadata(cfg)
    names = load_class_names(cfg)
    save_emb = cfg["inference"]["save_embeddings"]

    frames, all_embs = [], []
    for split in cfg["dataset"]["splits"]:
        df = select_split(meta, split, cfg)
        print(f"{split}: {len(df)} images")
        probs, embs = predict(cfg, df, model, device, return_embeddings=save_emb)
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
        frames.append(pd.concat([out, df[cfg["dataset"]["metadata_columns"]], prob_cols], axis=1))
        if save_emb:
            all_embs.append(embs)

    result = pd.concat(frames, ignore_index=True)
    out_path = resolve(cfg["paths"]["predictions"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")
    if save_emb:
        emb_path = resolve(cfg["paths"]["embeddings"])
        np.save(emb_path, np.concatenate(all_embs))
        print(f"Wrote embeddings to {emb_path}")
    return result
