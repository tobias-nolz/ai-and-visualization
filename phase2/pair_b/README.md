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

Single-screen interactive white-box visualization and slide deck explaining why the fine-tuned WILDS iWildCam ResNet-50 hallucinates **Brazilian rabbit** (*Sylvilagus brasiliensis*, `63.0%`) and **lowland paca** (*Cuniculus paca*, `24.5%`) on 698 empty frames of unseen African test **Camera 156** (`location = 282`), compared against the **Frozen ImageNet control** linear-probe baseline (`25.6%` empty, `0.0%` rabbit/paca).

### Key finding

Camera 156 carries a `24 px` white Bushnell metadata bar along the bottom (`rows 424–447` in `448×448` model space). In the training split, five South American cameras (`Camera 42`, `Camera 64`, `Camera 100`, `Camera 246`, `Camera 278`; `142` frames total, `0%` empty) share that exact bar. During fine-tuning, ResNet-50 turns the bar into a shortcut that directly suppresses the `empty` logit (`z₀`), peaking across Layers 40–49 (`50.8% → 77.4%` Contrastive Grad-CAM share on the bottom bar) and pulling `72.4%` of unfiltered (`89.3%` of predicted-class) Layer 49 nearest training neighbors from those South American white-bar cameras. Because Layer 49's receptive field straddles the top edge of the bar, **Row 12** (`z₀ = −4.56`) suppresses `empty` even more strongly than **Row 13** (`z₀ = −4.02`): zeroing only Row 13 still leaves `33.2%` white-bar South America neighbors, whereas spatially local zeroing of **Rows 12–13** inside Layer 49 (with the input photo untouched and re-normalized mean pooling) restores `99.6%` empty predictions and `0.0%` white-bar South America neighbors while changing top-1 accuracy on 500 real animals across 30 other unseen test cameras by only `−1.6 pp` (`56.8% → 55.2%`).

### Structure

- [src/extract_data.py](src/extract_data.py) — Computes the complete white-box bundle across all 698 empty Camera 156 frames (`275` Night IR, `423` Day RGB), 24 Camera 156 real-animal frames, 500 other-camera test animals (`30` unseen cameras), 200 normal empty test frames (`12` cameras), and exact Layer 49 cosine kNN (both unfiltered and ExMatchina-style predicted-class) across all `129,809` training embeddings (`outputs/embeddings.npy` and `outputs/embeddings_imagenet.npy`).
- [results/camera156_whitebox.json](results/camera156_whitebox.json) — Single self-contained precomputed JSON bundle (`~238 KB`).
- [src/server.py](src/server.py) — Local HTTP server serving `/` (with `448×448` model-space rendering, optional `24 px` bottom-bar cover, cached `/images/` thumbnails, and Windows/WSL image path support).
- [src/web/index.html](src/web/index.html) & [src/web/app.js](src/web/app.js) — Zero-scroll single-screen explorer (`1440×780` & `16:9`) following our Phase 1 white/gold/navy design grammar:
  1. **Where the shortcut forms:** ResNet-50 depth curve across `Layer 1, 10, 22, 40, 49` (`Layer 50 = classifier (fc)`) plotting Contrastive Grad-CAM share on the bottom bar (`rows 12–13` at Layer 49) with a clickable `All (698) / Night IR (275) / Day RGB (423)` cohort split vs. the `Frozen ImageNet control` and 12 normal empty cameras, plus a color-swatched baseline prediction bar (`63.0%` rabbit, `24.5%` paca, `9.6%` empty, `2.9%` other).
  2. **Spatial check: Grad-CAM vs. 24 px strips:** `Layer 49 Contrastive` Grad-CAM overlay with an explicit top color legend (amber-orange pushes rabbit/paca, blue pushes empty) and dashed `rows 12–13` guide, pixel-aligned beside an equal-height 18-strip `24 px` horizontal occlusion sweep (`Δz₀` for black and per-image mean fills), interactive strip hover, live `Cover bar (24 px)` toggle, and click-to-zoom lightbox.
  3. **Spatially local Layer 49 fix:** Exact additive per-row `z_empty` decomposition across the `14×14` map (highlighting `Row 12` spillover `−4.56` above `Row 13` `−4.02`) with re-normalized row-zeroing controls (`Baseline`, `Zero row 13`, `Zero 12–13`, `Ctrl 6–7`) and two consistent higher-is-better impact bars (`Camera 156 empty (698)` and `Other 30 test cameras (500 animals)`, with `avgpool` parity and the 24 Camera 156 real-animal frames in hover tooltips).
  4. **Nearest train frames:** Live full-`129,809`-train Layer 49 cosine kNN vote breakdown + 5 clickable training thumbnails, defaulting to ExMatchina-style `Predicted class` neighbors (with an `All classes` toggle and quiet `ImageNet control` baseline toggle) and visually verified animal bounding boxes on non-empty neighbor tiles and in the zoom modal.
- [figures/](figures/) — Five `1920×1080` presentation slides matching our Phase 1 visual style for the narrated walkthrough ending on the live explorer:
  - [figures/slide_0_title.png](figures/slide_0_title.png) — Section opener (`PHASE 2 / PAIR B · Inside Camera 156`)
  - [figures/slide_1_neighbors.png](figures/slide_1_neighbors.png) — `1 / 5` Night IR (`2.5×` rabbit inset) & Day RGB (`2×` paca inset) white-bar training neighbors
  - [figures/slide_2_depth.png](figures/slide_2_depth.png) — `2 / 5` ResNet-50 depth takeover (`Layer 1 → 49`) vs. `Frozen ImageNet control`
  - [figures/slide_3_spatial.png](figures/slide_3_spatial.png) — `3 / 5` `Standard Layer 49 Grad-CAM` (`100% clipped`) vs. `Layer 49 Contrastive` and row-aligned `24 px` strip occlusion
  - [figures/slide_4_local_fix.png](figures/slide_4_local_fix.png) — `4 / 5` Layer 49 per-row suppression (`Row 12` spillover) & local `Zero 12–13` intervention
- [AI log](../../ai-interactions/ai_log_pair_b.md) — Chronological log of AI interactions (both parts of Pair B), accepted/rejected pivots, verification catches, failure patterns, and best practices.

### Run locally

From the repository root (WSL or Windows):

```bash
# Re-extract the JSON bundle (requires GPU + make inference && make baselines outputs)
python phase2/pair_b/src/extract_data.py

# Serve the interactive explorer on http://127.0.0.1:8875/
python phase2/pair_b/src/server.py --port 8875
```
