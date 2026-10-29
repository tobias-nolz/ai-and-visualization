# AI and Visualization Lab Project: Visualizing a ResNet-50 on WILDS iWildCam

## Task

We investigate wildlife species classification using the WILDS iWildCam v2.0 dataset, which contains camera-trap images from different locations across 182 classes, including empty scenes.

Our goal is to explore and understand model behavior through visual analysis of its inputs and outputs, identifying interesting patterns and investigating them in depth. Building on these findings, we will examine the model's internal representations to better understand what it has learned and how this relates to its behavior.

Dataset: https://wilds.stanford.edu/datasets/

Paper: https://arxiv.org/abs/2012.07421


## Model

We use a ResNet-50 (PyTorch), pretrained on ImageNet and fine-tuned on WILDS iWildCam using Empirical Risk Minimization (ERM).

Depending on our findings, we may also compare it with an ImageNet-pretrained ResNet-50 without full fine-tuning, using a linear classifier for the iWildCam classes, particularly in Phase II to investigate differences in learned representations.

Repository: https://github.com/p-lambda/wilds

Model checkpoints: https://worksheets.codalab.org/worksheets/0x52cea64d1d3f4fa89de326b4e31aa50a

---

## Team

Group: Lost in latent space

| Pair | Member |
|------|--------|
| A    | Lara   |
| A    | Lisa   |
| B    | Martin |
| B    | Tobias |

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
phase1/, phase2/
  pair_a/, pair_b/        notebooks/, src/, figures/ (+ ai_log.md for the AI-aided pair)
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

**3. Explore:** open `notebooks/00_quickstart.ipynb` for a tour of the data, both models and all outputs.

**Getting the outputs from a teammate instead of re-running:** `make inference` and `make baselines` reproduce identical files on the same machine (other GPUs may differ in the last decimals), so one run can be shared and placed at the same paths: `outputs/predictions.parquet`, `outputs/predictions_probe.parquet` (~230 MB each), `outputs/baselines.csv`, `checkpoints/linear_probe.pt` and, for embedding analyses, `outputs/embeddings.npy` and `outputs/embeddings_imagenet.npy` (~1.7 GB each). Check them from the repo root with `sha256sum -c outputs/SHA256SUMS` (the sender creates it with `sha256sum outputs/*.parquet outputs/*.npy outputs/baselines.csv checkpoints/linear_probe.pt > outputs/SHA256SUMS`). Working with the predictions files alone needs neither the dataset nor a GPU; showing images or running the models does (`make data`).


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