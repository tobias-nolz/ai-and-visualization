"""Measure visible woody cover with SegFormer-B2; run before build_camera_map.py."""
from pathlib import Path
from functools import lru_cache
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

APP = Path(__file__).resolve().parents[1]
DATA = APP.parents[1] / 'data/iwildcam_v2.0'
MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
REVISION = 'de01bae28967510f9ddd496c60a969357195400c'
WOODY = ['tree', 'plant', 'palm']
OPEN = [
    'sky', 'grass', 'earth', 'mountain', 'water', 'sea', 'field', 'rock',
    'sand', 'path', 'river', 'hill', 'dirt track', 'land', 'lake', 'road',
]


def read_image(image):
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        with Image.open(image) as source:
            image = source.convert('RGB')
    width, height = image.size
    return image.crop((0, round(.03 * height), width, round(.95 * height)))


def sample_frames(meta):
    meta = meta[pd.to_datetime(meta.datetime).dt.hour.between(8, 17)].copy()
    names = pd.read_csv(DATA / 'categories.csv').drop_duplicates('y').set_index('y')['name']
    meta['empty'] = meta.y.map(names).isin(['empty', 'motorcycle'])
    rows = []
    for index, (location, group) in enumerate(meta.groupby('location'), 1):
        pool = group.sample(frac=1, random_state=42).sort_values('empty', ascending=False, kind='stable')
        pool = pd.concat([pool.drop_duplicates('seq_id'), pool]).drop_duplicates('image_id').head(48)
        candidates = []
        for _, row in pool.iterrows():
            image = read_image(DATA / 'train' / row.filename)
            rgb = np.asarray(image.resize((128, 96))).astype(float) / 255
            if rgb.mean() < .15 or rgb.mean(2).std() < .04:
                continue
            gray = (abs(rgb[:, :, 0] - rgb[:, :, 1]).mean() + abs(rgb[:, :, 1] - rgb[:, :, 2]).mean()) * 255 < 6
            item = row.to_dict()
            item.update(brightness=rgb.mean(), greyscale=gray)
            candidates.append(item)
        candidates.sort(key=lambda row: (not row['empty'], row['greyscale']))
        if candidates:
            choices = pd.DataFrame(candidates)
            choices = pd.concat([choices.drop_duplicates('seq_id'), choices]).drop_duplicates('image_id')
            rows.extend(choices.head(12).to_dict('records'))
        if index % 25 == 0:
            print(f'Sampled {index} cameras', flush=True)
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_model():
    folder = APP / '.cache/segformer-b2-ade'
    if not (folder / 'pytorch_model.bin').exists():
        snapshot_download(MODEL, revision=REVISION, local_dir=folder,
                          allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'])
    # Official ADE20K preprocessing: 512px resize, ImageNet rescaling/normalization.
    processor = SegformerImageProcessor(size={'height': 512, 'width': 512})
    model = SegformerForSemanticSegmentation.from_pretrained(
        folder, local_files_only=True, use_safetensors=False, weights_only=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


@torch.inference_mode()
def _score_images(images):
    processor, model, device = load_model()
    woody = torch.tensor([model.config.label2id[name] for name in WOODY], device=device)
    outdoor = torch.tensor([model.config.label2id[name] for name in WOODY + OPEN], device=device)
    inputs = processor(images=images, return_tensors='pt').to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        masks = model(**inputs).logits.argmax(1)
    scores = []
    for mask in masks:
        background = torch.isin(mask, outdoor)
        vegetation = torch.isin(mask, woody)
        fraction = float(background.float().mean())
        score = float((background & ~vegetation).sum() / background.sum()) if fraction >= .5 else np.nan
        scores.append((score, fraction))
    return scores


def image_openness(image):
    """Score an image path or PIL image after cropping the timestamp bands.

    Returns a float from 0 to 1, or NaN for dark/flat images or insufficient outdoor background.
    """
    image = read_image(image)
    rgb = np.asarray(image.resize((128, 96))).astype(float) / 255
    if rgb.mean() < .15 or rgb.mean(2).std() < .04:
        return float('nan')
    return _score_images([image])[0][0]


def infer(frames):
    rows = []
    start = time.monotonic()
    for first in range(0, len(frames), 4):
        batch = frames.iloc[first:first + 4]
        images = [read_image(DATA / 'train' / row.filename) for _, row in batch.iterrows()]
        scores = _score_images(images)
        for (_, row), (score, fraction) in zip(batch.iterrows(), scores):
            item = row.to_dict()
            item.update(openness=score, outdoor_fraction=fraction)
            rows.append(item)
        if first % 200 == 0 or first + 4 >= len(frames):
            print(f'Scored {min(first + 4, len(frames))}/{len(frames)} frames ({time.monotonic() - start:.0f}s)', flush=True)
    return pd.DataFrame(rows)


def aggregate(frames, locations):
    rows = []
    for location in locations:
        valid = frames[frames.location.eq(location)].dropna(subset=['openness'])
        per_sequence = valid.groupby('seq_id', sort=False).openness.median()
        rows.append({
            'location': location,
            'openness': per_sequence.median() if len(per_sequence) >= 3 else np.nan,
            'frames': len(valid), 'sequences': len(per_sequence),
            'colour_frames': int((~valid.greyscale).sum()), 'greyscale_frames': int(valid.greyscale.sum()),
            'q25': per_sequence.quantile(.25), 'q75': per_sequence.quantile(.75),
            'image_ids': '|'.join(valid.image_id),
        })
    return pd.DataFrame(rows)


def main():
    meta = pd.read_csv(DATA / 'metadata.csv')
    locations = sorted(meta.loc[meta.split.isin(['train', 'id_test', 'test']), 'location'].unique())
    frames = sample_frames(meta[meta.location.isin(locations)])
    scores = aggregate(infer(frames), locations)
    scores.to_csv(APP / 'results/camera_openness.csv', index=False)
    print(f'{scores.openness.notna().sum()}/{len(scores)} physical cameras have at least three usable sequences.')


if __name__ == '__main__':
    main()
