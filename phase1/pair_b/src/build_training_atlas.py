"""Update the atlas data from the saved annotation decisions."""
import json
from collections import Counter
from pathlib import Path

from annotations import load_reviews, read_manifest
from analyze_sequences import sequence_pairs

APP = Path(__file__).resolve().parents[1]


def update_training(payload, reviews):
    training = payload['training']
    annotations = {row['image_id']: row for row in reviews}
    image_ids = {
        frame['id']
        for sequence in training['sequences']
        for frame in sequence['frames']
    }
    if not set(annotations) <= image_ids:
        raise ValueError('Some labeled images are missing from the atlas data.')
    camera_counts = Counter()
    camera_flags = Counter()
    suspect_sequences = set()
    suspect_cameras = set()
    for sequence in training['sequences']:
        camera = sequence['camera']
        for frame in sequence['frames']:
            row = annotations.get(frame['id'])
            frame['has_decision'] = row is not None
            frame['training_phantom'] = frame['has_decision'] and row['status'] == 'phantom_label'
            if not frame['has_decision']:
                continue
            camera_counts[camera] += 1
            if frame['training_phantom']:
                camera_flags[camera] += 1
                suspect_sequences.add(sequence['id'])
                suspect_cameras.add(camera)
    for camera in training['cameras']:
        key = camera['camera']
        camera['training_phantoms'] = camera_flags[key]
        camera['decision_count'] = camera_counts[key]
    summary = {
        'source': 'results/human_review.json',
        'saved_decisions': sum(camera_counts.values()),
        'candidate_images': len(read_manifest()),
        'suspect_frames': sum(camera_flags.values()),
        'suspect_sequences': len(suspect_sequences),
        'suspect_cameras': len(suspect_cameras),
    }
    training['annotation_summary'] = summary
    return summary


def main():
    data_path = APP / 'results/atlas_data.json'
    payload = json.loads(data_path.read_text(encoding='utf-8'))
    summary = update_training(payload, load_reviews())
    pairs = sequence_pairs()
    for camera in payload['training']['cameras']:
        group = [row for row in pairs if row['camera'] == camera['camera']]
        camera['sequence_ends'] = {
            'pairs': len(group),
            'first': sum(row['first_flag'] for row in group),
            'last': sum(row['last_flag'] for row in group),
        }
    data_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Updated atlas: {summary['saved_decisions']} saved decisions, "
          f"{summary['suspect_frames']} flags in {summary['suspect_sequences']} sequences.")


if __name__ == '__main__':
    main()
