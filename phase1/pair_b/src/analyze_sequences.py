"""Compare first and last animal-labeled frames within each sequence/species group."""
from collections import Counter, defaultdict

from annotations import load_reviews, read_manifest


def sequence_pairs():
    reviews = {row['image_id']: row['status'] for row in load_reviews()}
    groups = defaultdict(list)
    for row in read_manifest().values():
        if row['label'] not in ['empty', 'motorcycle']:
            groups[row['camera'], row['seq_id'], row['label']].append(row)

    pairs = []
    for (camera, sequence, label), frames in groups.items():
        frames.sort(key=lambda row: (row['datetime'], row['image_id']))
        if len(frames) < 2:
            continue
        first, last = frames[0], frames[-1]
        # Image IDs do not establish capture order within a tied timestamp.
        if frames[1]['datetime'] == first['datetime'] or frames[-2]['datetime'] == last['datetime']:
            continue
        pairs.append({
            'seq_id': sequence,
            'camera': int(camera),
            'label': label,
            'animal_labeled_frames': len(frames),
            'first_id': first['image_id'],
            'last_id': last['image_id'],
            'first_flag': reviews[first['image_id']] == 'phantom_label',
            'last_flag': reviews[last['image_id']] == 'phantom_label',
        })
    pairs.sort(key=lambda row: (row['camera'], row['seq_id'], row['label']))
    if not pairs:
        raise ValueError('No sequence/species groups have two unambiguous timestamp endpoints.')

    return pairs


def analyze_sequence_ends():
    pairs = sequence_pairs()
    first_count = sum(row['first_flag'] for row in pairs)
    last_count = sum(row['last_flag'] for row in pairs)
    first_rate = first_count / len(pairs)
    last_rate = last_count / len(pairs)
    transitions = Counter((row['first_flag'], row['last_flag']) for row in pairs)
    print(f'{len(pairs):,} paired sequence/species groups')
    print(f'First frame with hallucinated label: {first_count:,} ({first_rate:.1%})')
    print(f'Last frame with hallucinated label: {last_count:,} ({last_rate:.1%})')
    print(f'Last minus first: {(last_rate - first_rate) * 100:.1f} percentage points')
    print(f'Only last hallucinated: {transitions[False, True]:,}, only first hallucinated: {transitions[True, False]:,}')

    return pairs
