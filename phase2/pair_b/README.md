# Phase 2 · Pair B — Inside Camera 156 (ResNet-50 White-Box Diagnosis)

**Group:** *Lost in latent space* · **Pair B (AI-aided):** Martin Dallinger (`k12205889`), Tobias Nolz (`k12311040`)

Single-screen interactive white-box visualization and slide deck explaining why the fine-tuned WILDS iWildCam ResNet-50 hallucinates **Brazilian rabbit** (*Sylvilagus brasiliensis*, `63.0%`) and **lowland paca** (*Cuniculus paca*, `24.5%`) on 698 empty frames of unseen African test **Camera 156** (`location = 282`), compared against the **Frozen ImageNet control** linear-probe baseline (`25.6%` empty, `0.0%` rabbit/paca).

## Key Finding

Camera 156 carries a `24 px` white Bushnell metadata bar along the bottom (`rows 424–447` in `448×448` model space). In the training split, five South American cameras (`Camera 42`, `Camera 64`, `Camera 100`, `Camera 246`, `Camera 278`; `142` frames total, `0%` empty) share that exact bar. During fine-tuning, ResNet-50 turns the bar into a shortcut that directly suppresses the `empty` logit (`z₀`), peaking across Layers 40–49 (`50.8% → 77.4%` Contrastive Grad-CAM share on the bottom bar) and pulling `72.4%` of unfiltered (`89.3%` of predicted-class) Layer 49 nearest training neighbors from those South American white-bar cameras. Because Layer 49's receptive field straddles the top edge of the bar, **Row 12** (`z₀ = −4.56`) suppresses `empty` even more strongly than **Row 13** (`z₀ = −4.02`): zeroing only Row 13 still leaves `33.2%` white-bar South America neighbors, whereas spatially local zeroing of **Rows 12–13** inside Layer 49 (with the input photo untouched and re-normalized mean pooling) restores `99.6%` empty predictions and `0.0%` white-bar South America neighbors while changing top-1 accuracy on 500 real animals across 30 other unseen test cameras by only `−1.6 pp` (`56.8% → 55.2%`).

## Structure

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
- [ai_log.md](ai_log.md) — Chronological log of AI interactions, accepted/rejected pivots, verification catches, failure patterns, and best practices.

## Run Locally

From the repository root (WSL or Windows):

```bash
# Re-extract the JSON bundle (requires GPU + make inference && make baselines outputs)
python phase2/pair_b/src/extract_data.py

# Serve the interactive explorer on http://127.0.0.1:8875/
python phase2/pair_b/src/server.py --port 8875
```
