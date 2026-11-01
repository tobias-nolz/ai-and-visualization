"""Update the atlas data from the saved annotation decisions."""
import json
from collections import Counter
from pathlib import Path

from annotations import load_reviews, read_manifest
from analyze_sequences import sequence_pairs

APP = Path(__file__).resolve().parents[1]


def update_training(payload, reviews):
    training = payload['training']
    images = read_manifest()
    sequences = {sequence['id']: sequence for sequence in training['sequences']}
    frame_ids = {frame['id'] for sequence in sequences.values() for frame in sequence['frames']}
    for row in images.values():
        if row['image_id'] in frame_ids:
            continue
        sequence = sequences.setdefault(row['seq_id'], {
            'id': row['seq_id'], 'camera': int(row['camera']), 'split': 'train', 'frames': []
        })
        sequence['frames'].append({'id': row['image_id'], 'label': row['label'], 't': row['datetime']})
        frame_ids.add(row['image_id'])
    training['sequences'] = list(sequences.values())
    for sequence in training['sequences']:
        sequence['frames'].sort(key=lambda frame: (frame['t'], frame['id']))
    annotations = {row['image_id']: row for row in reviews}
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
        'candidate_images': len(images),
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
          f"{summary['suspect_frames']} hallucinated training labels in {summary['suspect_sequences']} sequences.")


if __name__ == '__main__':
    main()
