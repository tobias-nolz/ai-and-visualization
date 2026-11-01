# AI and Visualization Lab Project: Visualizing a ResNet-50 on WILDS iWildCam

We analyze how a camera-trap species classifier behaves and why, using visualization.

- **Task:** species classification of camera-trap images (182 classes, incl. `empty`)
- **Model:** ResNet-50 (torchvision), the [WILDS](https://wilds.stanford.edu) ERM baseline checkpoint (`iwildcam_erm_seed0`), used as released (no fine-tuning)
- **Dataset:** WILDS iWildCam v2.0 (~12 GB; splits `train`, `val`/`test` = unseen camera locations, `id_val`/`id_test` = seen locations); [CodaLab dataset bundle](https://worksheets.codalab.org/bundles/0x6313da2b204647e79a14b468131fcd64)
- **Compute:** local GPU
- **Stack:** Python 3.14, PyTorch (CUDA 13.2 build)

The group of four works in two pairs on the **same model and data**:

| Phase | Focus                                                                                                         | Pair A                 | Pair B                 |
|-------|---------------------------------------------------------------------------------------------------------------|------------------------|------------------------|
| I     | How does the model behave?                                                                      | **with** generative AI | without  generative AI              |
| II    | What did the model learn? | without generative AI               | **with** generative AI |

The AI-aided pair logs every important interaction in `ai-interactions/` (e.g. `ai-interactions/ai_log_pair_a.md`).

## Team

| Pair | Member |
|------|--------|
| A    | Lara   |
| A    | Lisa   |
| B    | Martin |
| B    | Tobias |

## Deadlines

| Date           | Deliverable                                                                                                               |
|----------------|---------------------------------------------------------------------------------------------------------------------------|
| **11.11.2026** | Midpoint submission (Phase I): code of both pairs, `ai-interactions/ai_log_pair_a.md`, `ai_reflection_midpoint.md` |
| **09.12.2026** | Final submission (Phase II): code of both pairs, `ai-interactions/ai_log_pair_b.md` |

## Task, Dataset, and Model

- **Task:** Species classification of motion-triggered camera-trap photos across diverse wildlife classes (including empty frames and vehicles). We use visualization to analyze model behavior.
- **Dataset:** [WILDS iWildCam v2.0](https://wilds.stanford.edu), consisting of camera-trap photos collected across global locations. We evaluate on both in-distribution splits (seen camera locations) and out-of-distribution splits (unseen camera locations).
- **Model:** ResNet-50 fine-tuned on iWildCam (the official WILDS checkpoint `iwildcam_erm_seed0`).

## Folder map

```
common/                   shared, installable package (pip install -e .)
  config.yaml             all settings: model, dataset version, splits, seed, paths
  data.py                 metadata, splits, preprocessing, seeds
  model.py                loads the WILDS ResNet-50 checkpoint (+ ImageNet backbone for the baseline)
  inference.py            predictions + confidences (+ optional embeddings)
  evaluate.py             official WILDS metrics + linear-probe baseline
scripts/
  download_data.sh        dataset + checkpoint download
  run_inference.py        writes outputs/predictions.parquet
  evaluate_baselines.py   writes outputs/baselines.csv + linear-probe predictions
notebooks/
  00_quickstart.ipynb     shared demo: data, models and outputs (start here; no analysis; outputs committed)
ai-interactions/          documented GenAI interactions (e.g. ai_log_pair_a.md)
ai_reflection_midpoint.md reflection on Phase I GenAI experience (Pair A)
phase1/, phase2/
  pair_a/, pair_b/        notebooks/, src/, figures/
  comparison.md           Pair A vs. Pair B
data/, checkpoints/, outputs/   gitignored, created by the scripts
```

## Setup and run

Requires `make` and `curl`. Everyone should use the same Python version (3.14).

Without a GPU, it's usually easiest to get the outputs from a teammate (see below).

**1. Environment** (pick one)

Option A: conda (recommended; also works with `mamba` / `micromamba`):

```bash
conda env create -f environment.yml   # run from the repo root; creates env "ai-and-vis"
conda activate ai-and-vis
nbstripout --install                  # register the notebook git filter (once per clone)
```

Update after `pyproject.toml` changes: `conda env update -f environment.yml`.

Both options install the exact package versions pinned in `constraints.txt`, so everyone gets the same versions. After adding or upgrading a package, regenerate it (see the comment at the top of the file) and commit it.

Option B: venv (needs a system Python 3.14 with `venv`, e.g. `sudo apt install python3.14-venv`):

```bash
make setup            # .venv + pip install -e ".[analysis,dev]" + nbstripout git filter
```

Use a different interpreter with `make setup PYTHON=/path/to/python3`.

**2. Run** (the Makefile uses the active `ai-and-vis` conda env, otherwise `.venv`)

```bash
make data             # downloads ~12 GB dataset + ~96 MB checkpoint (skips what exists)
make inference        # writes outputs/predictions.parquet (+ embeddings.npy)
make baselines        # writes outputs/baselines.csv (see "Baseline performance")
make strip-notebooks  # strip outputs from all notebooks (also done automatically on commit)
```

**3. Explore:** open `notebooks/00_quickstart.ipynb` for a tour of the data, multiple models and all outputs.

**Getting the outputs from a teammate instead of re-running:** `make inference` and `make baselines` reproduce identical files on the same machine (other GPUs may differ in the last decimals), so one run can be shared and placed at the same paths: `outputs/predictions.parquet`, `outputs/predictions_probe.parquet` (~230 MB each), `outputs/baselines.csv`, `checkpoints/linear_probe.pt` and, for embedding analyses, `outputs/embeddings.npy` and `outputs/embeddings_imagenet.npy` (~1.7 GB each). Check them from the repo root with `sha256sum -c outputs/SHA256SUMS` (the sender creates it with `sha256sum outputs/*.parquet outputs/*.npy outputs/baselines.csv checkpoints/linear_probe.pt > outputs/SHA256SUMS`). Working with the predictions files alone needs neither the dataset nor a GPU; showing images or running the models does (`make data`).


### Pair A: Species Confusion Network

Interactive Dash dashboard analyzing species misclassifications, error rates, and confusion clusters.

#### Quick Start (Default Paths)

1. Ensure your environment is active and predictions exist (`make inference` or copy from a teammate):
   ```bash
   conda activate ai-and-vis
   python phase1/pair_a/src/species_confusion_network.py
   ```
2. Open **http://127.0.0.1:8050/** in your browser.

> **Note on Photos:** The dashboard runs fully without images (displaying placeholders), but downloading photos via `make data` is strongly recommended for side-by-side species comparisons and mistake slideshows.

#### Custom Paths & CLI Options

To run with predictions or photos stored in custom locations (e.g., external drives or WSL):

```bash
python phase1/pair_a/src/species_confusion_network.py --predictions "/path/to/predictions.parquet" --images-dir "/path/to/iwildcam_v2.0/train"
```

- **`--predictions <PATH>`**: Path to predictions file (default: `outputs/predictions.parquet`).
- **`--images-dir <PATH>`**: Folder containing JPG photos (default: `data/iwildcam_v2.0/train/`). Alternatively, use `--data-dir` for the dataset root.
- **`--check-data`**: Checks if prediction filenames exist on disk and exits without starting the server.
- **`--port <PORT>`**: Web server port (default: `8050`).
- **Windows / WSL:** Supports Windows paths (e.g., `D:\datasets\iwildcam_v2.0\train`) and WSL mounts (e.g., `/mnt/d/...`). Quote paths containing spaces.

#### English Species Names
- English common names are bundled offline for all 182 classes (derived from [WiSE-FT](https://github.com/mlfoundations/wise-ft/blob/4639a3919b51b20beae5814bc6722b962d93ae47/src/datasets/iwildcam_metadata/labels.csv), MIT licensed).
- Scientific Latin names are always visible on hover and in detail panels.

### Predictions files

`outputs/predictions.parquet` (fine-tuned model, `make inference`) and `outputs/predictions_probe.parquet` (linear probe, `make baselines`) have the same columns and rows, one row per image of every split in `dataset.splits`: the evaluation splits first, then `train` (filter with the `split` column):

| Column                                                                                                | Meaning                                |
|-------------------------------------------------------------------------------------------------------|----------------------------------------|
| `id`                                                                                                  | WILDS `image_id`                       |
| `split`                                                                                               | WILDS split                            |
| `label`, `label_name`                                                                                 | true class index and species name      |
| `prediction`, `prediction_name`                                                                       | predicted class index and species name |
| `confidence`                                                                                          | max softmax probability                |
| `prob_0` … `prob_181`                                                                                 | full softmax distribution              |
| `location_remapped`, `location`, `sequence_remapped`, `seq_id`, `category_id`, `datetime`, `filename` | metadata (set in `config.yaml`)        |

`outputs/embeddings.npy` (fine-tuned) and `outputs/embeddings_imagenet.npy` (frozen ImageNet backbone) hold the 2048-d penultimate features of the same images, row-aligned with the matching parquet file (~1.7 GB each; switch off with `inference.save_embeddings: false`). Row `i` of the embeddings belongs to row `i` of the predictions file, including the `train` rows, e.g. for nearest-neighbour searches. Load them with `np.load(path, mmap_mode="r")` to avoid reading everything into memory.

Models for opening up in Phase II: `common.model.load_model(cfg)` (fine-tuned) and `common.model.load_probe_model(cfg)` (ImageNet backbone + linear probe from `checkpoints/linear_probe.pt`, written by `make baselines`). Both map normalized images (`common.data.normalize`) to logits over the 182 classes.

### Baseline performance

Shared reference for both pairs, using the official WILDS metrics (accuracy and macro F1; macro recall is in the CSV):

- **WILDS fine-tuned:** the first of the two models we compare. ImageNet ResNet-50 fine-tuned end to end on `train`.
- **ImageNet linear probe:** the second model: the same ImageNet ResNet-50, frozen; only a logistic-regression classifier is trained on its `train` features. Shows what fine-tuning the whole network adds.
- **Majority class:** always predicts "empty", the most frequent training class. Shows how much of the accuracy comes for free.
- **WILDS-reported:** the official training run's numbers for the released checkpoint (epoch 8); our fine-tuned row should match them.

| Split     | Shift          | Always "empty" acc / F1 | WILDS-reported acc / F1 | Fine-tuned (reproduced) acc / F1 | Linear probe acc / F1 |
|-----------|----------------|-------------------------|-------------------------|----------------------------------|-----------------------|
| `id_val`  | seen cameras   | 0.404 / 0.008           | 0.823 / 0.482           | 0.823 / 0.479                    | 0.788 / 0.432         |
| `val`     | unseen cameras | 0.160 / 0.004           | 0.611 / 0.364           | 0.612 / 0.366                    | 0.524 / 0.293         |
| `id_test` | seen cameras   | 0.246 / 0.005           | 0.760 / 0.471           | 0.760 / 0.474                    | 0.736 / 0.457         |
| `test`    | unseen cameras | 0.330 / 0.005           | 0.703 / 0.324           | 0.702 / 0.323                    | 0.657 / 0.239         |
| `train`   | training data  | 0.370 / 0.003           | 0.989 / 0.794¹          | 0.993 / 0.814                    | 0.996 / 0.991         |

On the evaluation splits, our fine-tuned numbers match the WILDS-reported ones within 0.003 (newer library versions). On `train`, both models reach over 99% accuracy (the probe also near-perfect macro F1): they were trained on these images, so this row measures fit to the training data, not generalization. Regenerate all rows with `make inference && make baselines`; the raw numbers incl. macro recall are in `outputs/baselines.csv`.

¹ WILDS logs `train` metrics during training epoch 8, while the weights are still being updated; our row evaluates the final model afterwards, so the values differ slightly.

## Working rules

- **Pipelines stay separate.** Each pair works only in its own `phaseX/pair_Y/` folder and on its own branch (`phase1-pair-a`, `phase1-pair-b`). Pairs share **only** `common/` and the shared outputs (predictions files, embeddings, checkpoints). Changes to `common/` need agreement from both pairs.
- **No data, checkpoints, or secrets in git.** Large artifacts come from `scripts/download_data.sh`; secrets go in `.env` (gitignored).
- **Notebook outputs are not committed** (except the shared `notebooks/00_quickstart.ipynb`, which sets `keep_output: true`). The nbstripout filter strips them on `git add`; run `make setup` or `nbstripout --install` once per clone to activate it.
- **Log AI use** (AI-aided pair only): one entry per interaction in `ai-interactions/` (e.g. `ai-interactions/ai_log_pair_a.md`), including rejected suggestions and time spent verifying.
