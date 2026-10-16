"""Baseline metrics for both pairs (official WILDS metrics: accuracy, macro recall, macro F1).

Models:
  - WILDS fine-tuned ResNet-50: scored from the predictions file (run `make inference` first)
  - ImageNet linear probe: frozen ImageNet ResNet-50 features + logistic regression
    (trained on the GPU) on the `train` split, evaluated on the same splits
  - Majority class: always predicts the most frequent training class ("empty")

Writes `paths.baselines` (CSV) and prints the table. For the linear probe it also
writes per-image predictions (`paths.predictions_probe`, same columns as the
fine-tuned predictions file), the probe weights (`paths.probe_checkpoint`) and,
if `inference.save_embeddings` is set, the ImageNet features (`paths.embeddings_imagenet`).

Usage: python scripts/evaluate_baselines.py [--config path/to/config.yaml]
"""

import argparse

import numpy as np
import pandas as pd
import torch

from common import CONFIG_PATH, load_config, resolve
from common.data import load_class_names, load_metadata, select_split, set_seed
from common.evaluate import fit_linear_probe, wilds_metrics
from common.inference import extract_features, prediction_table, save_predictions
from common.model import get_device, load_imagenet_backbone


def main(cfg: dict) -> pd.DataFrame:
    """Evaluate all baselines on the configured splits and write the outputs listed above.

    Args:
        cfg: Shared config.

    Returns:
        One row per model and split with columns model, split, n, acc,
        recall_macro and f1_macro.

    Raises:
        SystemExit: If the fine-tuned predictions file does not exist yet.
    """
    set_seed(cfg["seed"])
    pred_path = resolve(cfg["paths"]["predictions"])
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found: run `make inference` first.")

    meta = load_metadata(cfg)
    names = load_class_names(cfg)
    train = select_split(meta, "train", cfg)
    majority = int(train["y"].mode()[0])

    rows = []
    preds = pd.read_parquet(pred_path)
    for split, df in preds.groupby("split", sort=False):
        rows.append({"model": "WILDS fine-tuned", "split": split, **wilds_metrics(df["label"], df["prediction"])})
    for split, df in preds.groupby("split", sort=False):
        always = np.full(len(df), majority)
        rows.append({"model": f"Majority class ({names[majority]})", "split": split, **wilds_metrics(df["label"], always)})

    device = get_device(cfg)
    backbone = load_imagenet_backbone(cfg, device)
    train_feats = extract_features(cfg, train, backbone, device, desc="probe: train features")
    probe = fit_linear_probe(cfg, train_feats, torch.tensor(train["y"].to_numpy()), device)
    probe_path = resolve(cfg["paths"]["probe_checkpoint"])
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_path)
    print(f"Wrote linear probe to {probe_path}")

    frames, embeddings = [], []
    for split in cfg["dataset"]["splits"]:
        df = select_split(meta, split, cfg)
        if split == "train":  # same images and order as used for fitting: reuse the features
            feats = train_feats
        else:
            feats = extract_features(cfg, df, backbone, device, desc=f"probe: {split}")
        frames.append(prediction_table(cfg, df, split, probe.predict_proba(feats), names))
        embeddings.append(feats.numpy())
        rows.append({"model": "ImageNet linear probe", "split": split,
                     **wilds_metrics(frames[-1]["label"], frames[-1]["prediction"])})
    save_emb = cfg["inference"]["save_embeddings"]
    save_predictions(cfg, frames, embeddings if save_emb else None, "predictions_probe", "embeddings_imagenet")

    result = pd.DataFrame(rows)
    out_path = resolve(cfg["paths"]["baselines"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(result.round(3).to_string(index=False))
    print(f"Wrote {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_PATH, help="YAML config (default: common/config.yaml)")
    main(load_config(parser.parse_args().config))
