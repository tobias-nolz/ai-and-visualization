# Vis4AI Lab Project: Visualizing a ResNet-50 on WILDS iWildCam

We analyze how a camera-trap species classifier behaves and why, using visualization.

- **Task:** species classification of camera-trap images (182 classes, incl. `empty`)
- **Model:** ResNet-50 (torchvision), the [WILDS](https://wilds.stanford.edu) ERM baseline checkpoint (`iwildcam_erm_seed0`), used as released (no fine-tuning)
- **Dataset:** WILDS iWildCam v2.0 (~12 GB; splits `train`, `val`/`test` = unseen camera locations, `id_val`/`id_test` = seen locations)
- **Compute:** local GPU
- **Stack:** Python 3.11, PyTorch

The group of four works in two pairs on the **same model and data**:

| Phase | Focus | Pair A | Pair B |
|---|---|---|---|
| I | Black-box behavior (inputs/outputs only) | **with** generative AI | without |
| II | Opening the model (representations, saliency/attention, SHAP/LIME, kNN, checkpoints; ≥ 2 techniques per pair) | without | **with** generative AI |

The AI-aided pair logs every important interaction in its `ai_log.md`.

## Team

| Pair | Member | Email |
|---|---|---|
| A | TODO | TODO |
| A | TODO | TODO |
| B | TODO | TODO |
| B | TODO | TODO |

## Deadlines

| Date | Deliverable |
|---|---|
| **11.11.2026** | Midpoint submission (Phase I): code of both pairs, `phase1/pair_a/ai_log.md`, `phase1/comparison.md` |
| **09.12.2026** | Final submission (Phase II): code of both pairs, `phase2/pair_b/ai_log.md`, `phase2/comparison.md`, `reflection/notes.md` |

## Folder map

```
common/                   shared, installable package (pip install -e .)
  config.yaml             all settings: model, dataset version, splits, seed, paths
  data.py                 metadata, splits, preprocessing, seeds
  model.py                loads the WILDS ResNet-50 checkpoint
  inference.py            predictions + confidences (+ optional embeddings)
scripts/
  download_data.sh        dataset + checkpoint download
  run_inference.py        writes outputs/predictions.parquet
agreements/               joint direction for each phase (filled in together)
phase1/, phase2/
  pair_a/, pair_b/        notebooks/, src/, figures/ (+ ai_log.md for the AI-aided pair)
  comparison.md           Pair A vs. Pair B
reflection/notes.md       AI4Vis reflection: failure patterns, best practices
data/, checkpoints/, outputs/   gitignored, created by the scripts
```

## Setup and run

Requires Python 3.11, `make`, `curl`, and a CUDA GPU (falls back to CPU, slowly).

```bash
make setup            # .venv + pip install -e ".[dev]" + nbstripout git filter
make data             # downloads ~12 GB dataset + ~96 MB checkpoint (skips what exists)
make inference        # writes outputs/predictions.parquet
make strip-notebooks  # strip outputs from all notebooks (also done automatically on commit)
```

Use a different interpreter with `make setup PYTHON=/path/to/python3.11`.

### Predictions file

`outputs/predictions.parquet` has one row per image:

| Column | Meaning |
|---|---|
| `id` | WILDS `image_id` |
| `split` | WILDS split |
| `label`, `label_name` | true class index and species name |
| `prediction`, `prediction_name` | predicted class index and species name |
| `confidence` | max softmax probability |
| `prob_0` … `prob_181` | full softmax distribution |
| `location_remapped`, `location`, `sequence_remapped`, `seq_id`, `category_id`, `datetime`, `filename` | metadata (set in `config.yaml`) |

With `inference.save_embeddings: true`, `outputs/embeddings.npy` holds the 2048-d penultimate features, row-aligned with the parquet file.

## Working rules

- **Pipelines stay separate.** Each pair works only in its own `phaseX/pair_Y/` folder and on its own branch (`phase1-pair-a`, `phase1-pair-b`). Pairs share **only** `common/` and the predictions file. Changes to `common/` need agreement from both pairs.
- **No data, checkpoints, or secrets in git.** Large artifacts come from `scripts/download_data.sh`; secrets go in `.env` (gitignored).
- **Notebook outputs are not committed.** The nbstripout filter strips them on `git add`; run `make setup` once per clone to activate it.
- **Log AI use** (AI-aided pair only): one entry per interaction in `ai_log.md`, including rejected suggestions and time spent verifying.
