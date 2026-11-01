"""Place cameras by segmented clear-background share and species diversity."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
DATA = APP.parents[1] / 'data/iwildcam_v2.0'


def main():
    path = APP / 'results/atlas_data.json'
    atlas = json.loads(path.read_text(encoding='utf-8'))
    cameras = atlas['cameras'] + atlas['training']['cameras']
    keys = {(camera['split'], camera['camera']) for camera in cameras}
    with (APP / 'results/camera_openness.csv').open(encoding='utf-8', newline='') as file:
        scores = {int(row['location']): row for row in csv.DictReader(file)}
    with (DATA / 'categories.csv').open(encoding='utf-8', newline='') as file:
        names = {int(row['y']): row['name'] for row in csv.DictReader(file)}

    sequences = defaultdict(lambda: defaultdict(set))
    animal_images = Counter()
    locations = {}
    with (DATA / 'metadata.csv').open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            key = row['split'], int(row['location_remapped'])
            location = int(row['location'])
            if key in keys:
                locations[key] = location
            labels = sequences[location][row['seq_id']]
            species = names[int(row['y'])]
            if species not in ['empty', 'motorcycle']:
                labels.add(species)
                animal_images[key] += 1

    positioned = 0
    for camera in cameras:
        key = camera['split'], camera['camera']
        location = locations[key]
        measured = scores[location]
        score = measured['openness']
        n = len(sequences[location])
        animals = sum(bool(labels) for labels in sequences[location].values())
        counts = Counter(species for labels in sequences[location].values() for species in labels)
        total = sum(counts.values())
        probabilities = [count / total for count in counts.values()]
        diversity = math.exp(-sum(p * math.log(p) for p in probabilities)) if total else None
        camera['location'] = location
        camera['animal_labeled_images'] = animal_images[key]
        camera['openness'] = float(score) if score else None
        colour = int(measured['colour_frames'])
        grey = int(measured['greyscale_frames'])
        camera['openness_basis'] = 'mixed' if colour and grey else 'colour' if colour else 'greyscale' if grey else None
        camera['openness_frames'] = int(measured['frames'])
        camera['openness_sample'] = measured.get('sample_policy', 'daytime')
        camera['openness_sequences'] = int(measured['sequences'])
        camera['openness_q25'] = float(measured['q25']) if measured['q25'] else None
        camera['openness_q75'] = float(measured['q75']) if measured['q75'] else None
        camera['animal_sequences'] = animals
        camera['total_sequences'] = n
        camera['species_count'] = len(counts)
        camera['effective_species'] = diversity
        positioned += score != '' and diversity is not None
    atlas['summary']['camera_map'] = {
        'openness_source': 'camera_openness.csv; SegFormer-B2 ADE20K, revision de01bae28967510f9ddd496c60a969357195400c; median across sequence medians of non-tree/plant/palm outdoor background pixels; prefer at least three daytime sequences, otherwise available frames with at least one usable sequence',
        'effective_species': 'exp(Shannon entropy) of species counts, each species counted once per sequence across all splits at the physical camera; excludes empty/motorcycle labels',
        'positioned_cameras': positioned,
        'unpositioned_cameras': len(cameras) - positioned,
    }
    path.write_text(json.dumps(atlas, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Updated map: {positioned} camera entries with both axes.', flush=True)


if __name__ == '__main__':
    main()
