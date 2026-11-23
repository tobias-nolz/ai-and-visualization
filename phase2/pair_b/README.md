# Phase II, Pair B

**Group:** *Lost in latent space* · **Pair B (AI-aided):** Martin Dallinger (`k12205889`), Tobias Nolz (`k12311040`)

Two parts: **Confusions** (Tobias) and **Hallucinations** (Martin).

## Confusions: why does the background win?

Phase I showed that pasting the same animal cut-out onto different empty frames changes the prediction: an elephant
on savanna is an elephant, the same elephant in a forest becomes a great argus. Phase II opens the ResNet-50 to see
where this comes from. The result is a scrollable story page, "The elephant that turned into a bird".

### Findings
- **Where:** a class score is the mean of a 14 × 14 class activation map, so it splits exactly into the animal's
  cells and the background. In wrong composites the animal still slightly favours the true species (−0.8 logits);
  the background gives the winner +7.5.
- **What (network dissection):** the paper's IoU rule labels only 2 of 2,048 layer4 units, because concepts as big
  as trees (47 % of the pixels) cap the IoU. Labelled by where each unit's *total* activation goes, there are 540
  animal units and only 59 scenery units (27 tree, 20 ground, 12 sky), plus 1,449 mixed.
- **Who votes:** the winner's lead comes from the whole background, roughly in proportion to its area. In forest,
  trees cover 63 % of the frame and give 65 % of the lead.
- **Ablation:** switching off the 256 most tree-leaning units halves "empty" in forest (47 % → 23 %), but 73 % of the
  animals that reappear get the wrong name. Real test accuracy only drops from 70.2 % to 69.3 %.
- **Takeaway:** no forest detector overrules the elephant. A small animal is simply outvoted by a frame full of scenery.

### View the story

```sh
python -m http.server 8000 --directory phase2/pair_b/story   # then open http://127.0.0.1:8000
```

Use → / space to step through the main line. The small controls next to the charts and the "more" boxes are there
for exploring.

### Rerun (repo root, `ai-and-vis` env)

Needs the dataset, `outputs/predictions.parquet`, `outputs/embeddings.npy` and the Phase I results (see the root README).

| Step | Command |
|---|---|
| Animal vs. background split of all 179,820 composites | `python phase2/pair_b/src/evidence_split.py` |
| Network dissection (paper IoU rule) | `python phase2/pair_b/src/dissect.py` |
| Unit profiles, vote split by concept, ablation | `python phase2/pair_b/src/concepts.py` |
| Small extracts for the page (only after the steps above) | `python phase2/pair_b/src/export_story_data.py extract` |
| Page data and images | `python phase2/pair_b/src/export_story_data.py` |

The page can be rebuilt with the last command alone: it only reads files in git plus the dataset frames. Large
intermediates go to `outputs/` (not in git).

### Files
| Path | Content |
|---|---|
| `src/` | the four scripts above |
| `results/` | `evidence_split.parquet`, `dissection_*`, `concepts_*` and the small `story_*` extracts |
| `figures/dissection/` | contact sheets: best units per concept, per model and layer |
| `story/` | the story page (static HTML + D3, works offline) |

### Caveats
- Composites are out of distribution: even on its own camera a pasted cut-out is right only 63 % of the time.
- "Inside the animal" means the animal's grid cells, which also see some of the surroundings.
- Phase I's "night" frames are really greyscale (infrared or black-and-white cameras), so the page says "greyscale".

## Hallucinations: inside Camera 156 (phantom rabbits)

Phase I showed that unseen African test **Camera 156** (`location = 282`) hallucinates **Brazilian rabbit** (`63.0%`) and **lowland paca** (`24.5%`) across `698` empty frames (`275` Night IR, `423` Day RGB). Phase II opens the fine-tuned ResNet-50 and compares it against the **Frozen ImageNet control** (`25.6%` empty, `0.0%` rabbit/paca) in one linked view and five presentation slides.

### Findings
- **The trigger:** Camera 156 has a `24 px` white Bushnell metadata bar along the bottom (`rows 424–447`), shared with five South American training cameras (`Camera 42, 64, 100, 246, 278`; `142` frames, `0%` empty).
- **Where it forms:** Contrastive Grad-CAM shows fine-tuning locks onto the bottom bar from Layer 22 to Layers 40–49 (`50.8% → 77.4%`), while the Frozen ImageNet control stays flat through Layer 40 and never predicts rabbit or paca.
- **Why standard Grad-CAM misses it:** Because the bar directly suppresses the `empty` logit (`z₀`), standard Layer 49 ReLU Grad-CAM clips `99.5%–100%` to zero. Contrastive Layer 49 Grad-CAM and an equal-height `24 px` strip-occlusion sweep (`Δz₀ = +9.0` on the bottom strip) both isolate the bar.
- **Local Layer 49 fix & Row 12 spillover:** Splitting `z₀` across the 14 rows of Layer 49 shows **Row 12** just above the bar (`−4.56`) suppresses `empty` even more than **Row 13** (`−4.02`). Zeroing only Row 13 leaves `33.2%` white-bar neighbors; zeroing **Rows 12–13** (photo untouched, re-normalized mean) restores **`99.6%` empty** and **`0.0%`** white-bar South America neighbors, while changing accuracy on `500` real animals across `30` other test cameras by only `−1.6 pp` (`56.8% → 55.2%`).
- **Nearest training frames:** Across all `129,809` training embeddings, `72.4%` of unfiltered (`89.3%` of predicted-class ExMatchina) top-5 Layer 49 neighbors come from those South American white-bar cameras, snapping back to normal empty forest cameras once the bar is covered or Rows 12–13 are zeroed.

### Run locally

From the repository root (WSL or Windows):

```bash
# Optional: re-extract the JSON bundle (requires GPU + make inference && make baselines outputs)
python phase2/pair_b/src/extract_data.py

# Serve the interactive explorer on http://127.0.0.1:8875/
python phase2/pair_b/src/server.py --port 8875
```

All charts and metrics load from the committed JSON bundle; showing photo thumbnails requires `data/iwildcam_v2.0/train` (`make data`) or `--images-dir <path>`.

### Files
| Path | Content |
|---|---|
| [src/extract_data.py](src/extract_data.py) | Extracts layer trajectories, Grad-CAM, 18-strip occlusion, Layer 49 row zeroing, and full `129,809`-train kNN |
| [results/camera156_whitebox.json](results/camera156_whitebox.json) | Precomputed white-box bundle (`~238 KB`) driving the web app |
| [src/server.py](src/server.py) | Local HTTP server for `/` and cached `/images/` thumbnails (Windows & WSL) |
| [src/web/index.html](src/web/index.html), [src/web/app.js](src/web/app.js) | Single-screen linked explorer (`1440×780` & `16:9`) with strip hover, bar cover, row zeroing, ExMatchina toggle, and verified animal boxes |
| [figures/](figures/) | Five `1920×1080` slides (`slide_0_title.png` .. `slide_4_local_fix.png`) matching our Phase 1 style |
| [AI log](../../ai-interactions/ai_log_pair_b.md) | Combined Phase 2 Pair B AI interaction log (Confusions & Hallucinations) |
