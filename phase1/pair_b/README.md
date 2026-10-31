# Phase I, Pair B

## Hallucination atlas and label review

We use the atlas to inspect cases where the model predicts an animal in an image labeled empty. The notebook looks for possible label errors in related training sequences. This can suggest an explanation for the prediction errors, but does not show what caused them.

### Run and label

Use the project Python environment and the dataset at `data/iwildcam_v2.0`. Run the commands below from the repository root:

```sh
python phase1/pair_b/src/server.py
```

Open the [atlas](http://127.0.0.1:8874/) to inspect the results. Use the [labeling tool](http://127.0.0.1:8874/viewer/) if you want to add or change labels:

> Click an image when the animal is absent and click again to undo the mark. The right arrow confirms the page. Changes are saved to `phase1/pair_b/results/human_review.json`, and the tool remembers where you stopped. Check that saving has finished before closing. If the file is missing, the tool starts a new label set.

After changing labels, run the following command and reload the atlas:

```sh
python phase1/pair_b/src/build_training_atlas.py
```

### Files

Paths in this table are relative to `phase1/pair_b`.

| Files | Purpose |
| --- | --- |
| `notebooks/hallucination_audit.ipynb` | Run predictions, match cameras, select images for review and summarize labels. |
| `src/server.py`, `src/web/` | Serve the atlas and labeling tool. |
| `src/annotations.py`, `src/analyze_sequences.py`, `src/build_training_atlas.py` | Read saved labels, compare first and last frames, and update the atlas. |
| `results/viewer_images.csv`, `results/human_review.json` | Images available for labeling and their saved labels. |
| `results/atlas_data.json` | Camera summaries and image sequences shown in the atlas. |
| `src/measure_background.py`, `src/build_camera_map.py`, `results/camera_openness.csv` | Measure background cover and build the camera map. The CSV records scores and sampled image IDs. |

### Rerun the analysis

Run the audit notebook with `phase1/pair_b` as its working directory. It uses the shared `common` package, dataset metadata and classifier weights configured in `common/config.yaml`. Camera matching also needs `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` in `~/.cache/torch/hub/checkpoints/`. Generated analysis tables are ignored by Git.

Keep `results/atlas_data.json` when using the saved results. The update script changes the training labels and counts in this file, but does not recreate the test-image galleries.

To rebuild the camera map, run:

```sh
python phase1/pair_b/src/measure_background.py
python phase1/pair_b/src/build_camera_map.py
```

X shows the estimated share of the outdoor background without trees or shrubs, measured with SegFormer-B2. Each camera score uses up to 12 daytime images from at least three sequences, taking medians within and across sequences. The scorer needs PyTorch >=2.6 and `transformers==4.57.6`. It downloads a fixed version of the model to `.cache/` on the first run. The server and atlas-update script only need Python's standard library.

Y shows species diversity as `exp(Shannon entropy)`. We count each species once per sequence and combine all splits from the same physical camera, excluding empty and motorcycle labels. Point area is based on the fraction of images flagged at that camera and split. Training flags come from the selected review candidates. Gray means no recorded flags. We skip first/last comparisons where timestamps do not identify a unique first or last frame.

The label file combines API labels, automatic `not_flagged` defaults and edits made in the viewer. It does not record the source of each label, so the total does not tell us how many images were checked manually. The viewer writes `phantom_label` when an animal-labeled image is marked absent. A `not_flagged` value can also be a default, so it does not prove that the animal is present or that the dataset label is correct.
