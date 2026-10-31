# Phase I, Pair B

## Confusion Analysis (background swap)

We paste animal cut-outs onto empty camera-trap frames and look at what the model predicts. If it really
recognised the animal, the background shouldn't matter. It does matter quite a lot, see
`notebooks/confusion_territories.ipynb` (prediction-territory explorer and observations).



- `data/species_groups.csv` (repo root): maps the 182 classes to ~30 groups with normal names (all elephants -> "elephant",
  the brocket deer -> "deer", ...). The groups are only used for the colours: a prediction counts as correct
  only if it is the exact species (like the mock-up). "Right group, other species" (e.g. waterbuck -> duiker)
  is its own category, so confusions inside a group are still visible.
- `src/make_cutouts.py`: finds candidate images per group, cuts the animals out, writes `cutouts/`.
- `src/composite.py`: pastes a cut-out onto one cell of a 3x3 grid; also the camera habitats (open / mixed / forest
  from the SegFormer openness in `results/camera_openness.csv`) and the shared paths.
- `src/run_composites.py`: picks the empty backgrounds (evenly over the habitats, day and night), runs the model
  on every combination plus two controls (grey frame, the cut-out's own camera), writes `results/`.
- `cutouts/`: the 45 cut-outs (3 per group, 15 groups), `cutouts.csv` (where each one comes from),
  `sizes.csv` (how big each group is pasted), `candidates.csv` and `cutout_choices.csv` (which candidates we picked).
- `results/camera_openness.csv`: SegFormer openness of every camera (see "Habitats").
  `results/backgrounds.parquet`: the empty background frames (id_test = seen cameras, test = new cameras) with
  openness, habitat and day / night. `results/composites.parquet`: one row per composite. `results/controls.parquet`:
  the control composites.
- `figures/`: `territory_snapshots/` (selected explorer views, saved by the notebook) and `cutout_candidates/`
  (the sheets we picked the cut-outs from).

### How the cut-outs are made

The WILDS images are downsized (796x448) and have no masks. The iWildCam 2022 release on LILA has the same
images at full resolution, plus MegaDetector boxes and DeepMAC masks.

For each group the script takes up to 40 images with exactly one animal (MegaDetector conf >= 0.95), not too
small, not cut off at the frame edge, daytime/colour only, preferably from the evaluation splits. Then SAM
(ViT-H) makes the mask from the MegaDetector box, and we keep the SAM mask that overlaps most with the DeepMAC
one. The contact sheets show the 12 best per group (by mask agreement, size and sharpness) and we picked 3 by
hand: whole animal, different locations, different species of the group where possible.

When pasting:
- every group gets its typical size (median MegaDetector box of that group in the whole dataset)
- big animals are moved inwards so they stay inside the frame
- on infrared backgrounds the cut-out is pasted in greyscale

### Habitats

A camera's habitat comes from its openness, measured with SegFormer-B2 (ADE20K) by `src/measure_background.py`:
for up to 12 daytime frames per camera every pixel gets a class, and openness = share of the outdoor background
(sky, grass, earth, rock, water, ...) that is not tree / plant / palm; median over sequence medians, at least 3
sequences. Open above 2/3, forest below 1/3, mixed in between. It measures visible woody cover, so bare forest floor
counts as open ground. Day/night is per frame (infrared or not), so there are six habitats. Backgrounds: at most 80
per habitat and 20 per camera, cameras in turns: 444 frames, 80 per habitat except open land
(47 frames from 7 cameras by day, 77 from 18 at night), there just aren't many open cameras with empty frames in
id_test/test.

### Reproduce

Run from the repo root, in the ai-and-vis env. Needs `outputs/predictions.parquet` (from `make inference` or a
teammate) and the dataset.

1. SAM weights (2.5 GB) into `checkpoints/`:
   `curl -L -o checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
   (`segment-anything` is in the analysis extras now, so `pip install -e ".[analysis]"` / `conda env update` gets it.)
2. `python phase1/pair_b/src/make_cutouts.py candidates`: candidates, SAM masks, contact sheets
3. Pick cut-outs: look at `figures/cutout_candidates/`, write the indices into `cutouts/cutout_choices.csv`.
4. `python phase1/pair_b/src/make_cutouts.py`: writes the cut-outs, `cutouts.csv` and `sizes.csv`.
5. `python phase1/pair_b/src/measure_background.py` (only needed to redo `camera_openness.csv`, which is in git;
   needs `transformers==4.57.6`, downloads the SegFormer weights to `phase1/pair_b/.cache/` on the first run)
6. `python phase1/pair_b/src/run_composites.py`
7. Run `notebooks/confusion_territories.ipynb` with `phase1/pair_b` on the Python path.

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
