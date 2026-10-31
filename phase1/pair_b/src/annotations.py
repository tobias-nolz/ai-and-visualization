"""Read the saved decisions used by the notebook and atlas."""
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def read_manifest():
    with (RESULTS / 'viewer_images.csv').open(encoding='utf-8-sig', newline='') as file:
        return {row['image_id']: row for row in csv.DictReader(file)}


def load_reviews():
    saved = json.loads((RESULTS / 'human_review.json').read_text(encoding='utf-8'))
    images = read_manifest()
    annotations = saved['annotations']
    if annotations.keys() != images.keys():
        raise ValueError('The review file must contain one decision for every candidate image.')
    reviews = []
    for image_id, annotation in annotations.items():
        if annotation['status'] not in ['phantom_label', 'not_flagged']:
            raise ValueError(f'Invalid binary review status for {image_id}')
        reviews.append({'image_id': image_id, 'status': annotation['status']})
    return reviews
