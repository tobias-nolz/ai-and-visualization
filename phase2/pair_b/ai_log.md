# AI Log: Phase II, Pair B (AI-aided)

**Group:** *Lost in latent space* · **Pair B:** Martin Dallinger (`k12205889`), Tobias Nolz (`k12311040`)

One entry per interaction (newest at the bottom). Verbatim prompts are preserved as originally typed (including early shorthand such as `C156` or `7 SA` before we standardized all user-facing labels to `Camera 156` and the exact 5 `0%`-empty white-bar South American training cameras).

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns Observed
- **Early speculation & shallow-layer bias:** On the first pass, the AI speculated about pareidolia (rabbit-shaped rocks) or phantom training labels and pitched methods centered almost exclusively on `layer4 + fc` (including circular `dw*f`-weighted kNN). Pushing it to test `empty` logit suppression and earlier stages (`conv1..layer4` + BatchNorm) first uncovered the actual `24 px` white Bushnell bar shortcut.
- **Overbuilding for a short talk:** When asked for 2–3 interactive designs, the AI produced three sprawling multi-card dashboards (`/atlas`, `/xai`, `/circuits`) packed with text, terse C-codes (`C156`, `C278`), noisy pixel GradientSHAP, and heavy-handed 6/32-channel clamping — all unsuitable for a short narrated demo until explicitly pruned back to one zero-scroll `1440×780` view.
- **Silent technical & data bugs:** Even code that ran cleanly had subtle flaws: `sanitize_json` serialized booleans as integers, the strip-occlusion x-axis silently clamped 10 of 18 negative black-fill bars at `−3.0`, an in-place JSON patch broke reproducibility against `extract_data.py` (batch-size 1 vs. 32), and spawning 4 concurrent PyTorch CUDA processes in WSL2 hit host RAM limits and crashed WSL (`Wsl/Service/E_UNEXPECTED`).
- **What blind AI reviewers still missed:** Unhinted blind sub-agents reliably caught pixel overlaps, axis clipping, JSON-vs-script drift, and image framing issues, but repeatedly missed **human presentation ergonomics** — cryptic abbreviations (`Frozen IN`, `GT:`, `L22`, `r12–13`), noisy `Layer 22` leaf scatter, confusing unfiltered `Brocket deer` neighbors under a `Brazilian rabbit` prediction, an inverted middle metric bar in Panel 3, missing color-legend labels for blue/orange heatmap tints, and dark-gold vs. dark-orange text under the Panel 1 stacked bar without matching swatches.

### 2. Best Practices Learned
- **Probe numbers before writing UI:** Running focused WSL experiments first (698-frame `24 px` strip occlusion; all-14-row `layer4` zeroing with `renorm` vs. `avgpool` parity and full `129,809`-train kNN) revealed the decisive **Row 12 receptive-field spillover** story (`−4.56` vs. `−4.02`) and prevented building UI around weak methods like pixel GradientSHAP or arbitrary channel clamping.
- **Pair blind sub-agent audits with hands-on human click-throughs:** Blind browser (`1440×780` & `1280×720`), blind JSON/code, and blind slide-vs-Phase-1 reviews caught mechanical bugs fast before manual review, while manual click-throughs caught narrative friction and visual encoding ambiguity.
- **Keep GPU work strictly serialized in WSL2** (one CUDA process at a time) and write non-trivial Python scripts to files instead of inline PowerShell/WSL one-liners.
- **Never let AI guess visual coordinates:** Because `iwildcam_v2.0` has no MegaDetector bounding boxes on disk, requiring the AI to visually inspect every unique non-empty neighbor photo on a coordinate grid produced 29 accurate animal bounding boxes and honestly left 14 empty sequence tails unmarked.

---

## Log of AI Interactions

### 2026-11-11: Initial White-Box Brainstorming on Camera 156 Hallucinations

- **Who:** Martin
- **Goal:** Inspect the shared repo (`README.md`, `common/model.py`) and pitch 3 white-box ways to compare the fine-tuned WILDS ResNet-50 against the frozen ImageNet linear probe on unseen African Camera 156 (`location=282`).
- **Prompt (important ones):**
  ```
  Hey, I'm doing Phase 2 for Pair B in phase2/pair_b/ (on WSL /home/martin/ai-and-visualization) and looking into why unseen test camera C156 (location=282, an African camera) predicts 440 sylvilagus brasiliensis rabbits and 171 cuniculus paca on empty frames. Before writing code, check README.md and common/model.py and pitch me 3 ways we can look inside the fine-tuned ResNet-50 vs. the frozen ImageNet linear probe to find out what it actually learned here.
  ```
- **AI output summary:** Reviewed `README.md`, `common/model.py`, `common/evaluate.py`, and Phase 1 notes; pitched 3 white-box approaches largely centered on `layer4 + fc` (including `dw*f`-weighted training kNN) while hypothesizing either sequence-level phantom labels on a Neotropical twin camera or a localized rabbit-shaped rock/stump trigger.
- **Decision:** rejected / redirected
- **Why:** In Phase 1 we had already verified training rabbits were real, and Camera 156 hallucinates rabbit/paca even on frames containing real African animals — suggesting `empty` suppression rather than pareidolia. Staying only at `layer4 + fc` with `dw*f`-weighted kNN also felt too shallow and circular.
- **Problems caused:** wrong initial hypothesis (pareidolia / phantom labels) and overly late-layer pitch before looking at actual logits and earlier blocks.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-11: Deeper Mechanistic Probes (`conv1..layer4`, BN Stats & Logit Audit)

- **Who:** Martin
- **Goal:** Run `make inference && make baselines` and parallel WSL probes testing `empty` logit suppression, `conv1..layer4` + BatchNorm drift/reverts, and input occlusion/SHAP before designing any UI.
- **Prompt (important ones):**
  ```
  hold on, I'm sceptical about a few things here:
  - don't guess pareidolia or phantom labels yet. in phase 1 our checked train rabbits were real, and C156 predicts rabbit/paca even on the 24 frames with real monkeys/birds - so check if `empty` itself is getting suppressed, not just some rabbit-shaped rock.
  - all 3 pitches mostly stay at layer4 + fc (huge receptive fields) and weighting kNN by dw*f feels circular. since both models start from the same ImageNet weights, go deeper into conv1..layer4 + BN stats (e.g. layerwise drift or swapping/patching blocks) to see which stage actually breaks C156, and cross-check unit/spatial maps against real input occlusion or SHAP.

  kick off `make inference && make baselines` in WSL (`/home/martin/ai-and-visualization`) and spawn parallel sub-agents to probe these hypotheses on C156 + train cameras first. don't build any UI yet, just bring back the numbers so we can decide where to go.
  ```
- **AI output summary:** Confirmed there is no rabbit-shaped rock and no phantom labels: `empty` (`y=0`) is suppressed by ~10 logits across Camera 156 because of a `24 px` white Bushnell bottom bar shared with `0%`-empty South American training cameras; also found standard `layer4` Linear CAM misleads (`ρ = −0.354`) while the shortcut enters in `stem..layer2` and consolidates in `layer3–layer4`.
- **Decision:** accepted (findings)
- **Why:** Grounded the entire Phase 2 investigation in verified internal mechanism numbers rather than guesses.
- **Problems caused:** bug / environment crash — launching 4 concurrent WSL2 `libcuda` processes hit WSL's ~15 GiB host-RAM ceiling (`Wsl/Service/E_UNEXPECTED`), forcing a WSL restart and strict serialization of GPU jobs afterward.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-17: White-Box Reframing & Multi-Agent Build of 3 Prototypes (`/atlas`, `/xai`, `/circuits`)

- **Who:** Martin
- **Goal:** Keep the Phase 2 story unmistakably white-box (not Phase 1 black-box bar-pasting/jigsaw tests) and explore 3 interactive prototype views built and audited by independent sub-agents.
- **Prompt (important ones):**
  ```
  nice find on the bar, but careful: bar-pasting and jigsaw tests look like phase 1 black-box changes, whereas phase 2 needs to be clearly white-box according to the task description(weights, layerwise representations, gradients, internal channels, and comparing >=2 XAI methods).
  let's center the story inside the model: (1) how C156, the 7 white-bar SA train cameras, and normal empty frames move/cluster across layer1->layer4 and between Frozen ImageNet vs Fine-Tuned representation space, and (2) comparing why layer4 Grad-CAM fails vs layer2 Grad-CAM + SHAP + latent kNN, plus dissecting the specific internal channels and fc weights that detect the bar and split IR rabbit vs Day paca (keeping bar-masking just as a causal check).
  Pitch me 2-3 clean interactive visualization designs in `phase2/pair_b/` that let someone explore and understand this white-box mechanism clearly - design and code them up end-to-end and I'll select the most promising

  Use multiple sub-agents to do so: orchestrate independent builder / reviewer sub-agents and check if it visually makes sense and if it fits the task description before you show it to me!!! think insanely hard!
  ```
  *(Followed by `build all 3 and show me the results - wsl should be back up by now, no?` and `stop but keep the webserver running - i will now review myself!`.)*
- **AI output summary:** Built `/atlas` (Latent Trajectory & kNN Atlas), `/xai` (Receptive-Field & Multi-XAI Debugger), and `/circuits` (Channel & `fc`-Weight Circuit Dissector) backed by extracted JSON bundles on `http://127.0.0.1:8875/`. Independent browser and rubric sub-agents audited all routes before handoff.
- **Decision:** rejected (prototypes later replaced, though key analytical pieces survived)
- **Why:** Building all 3 surfaced what actually worked vs. failed visually, but clicking through them ourselves showed 3 dense dashboards were far too heavy for a short presentation.
- **Problems caused:**
  - *Caught by rubric/browser sub-agents:* horizontal overflow on `/xai`, wrapped toggle header on `/circuits`, GradientSHAP baseline noise, and a data-vs-prose contradiction (only **5** South American `train` cameras `{42, 64, 100, 246, 278}` are `0%`-empty white-bar cameras, not 7).
  - *Caught by Martin on manual review:* overall cognitive overload, text/card clutter, cryptic C-codes, and noisy pixel GradientSHAP across the forest.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-20: Consolidating to One Single-Screen View & Testing `24 px` Strip Occlusion

- **Who:** Martin
- **Goal:** Replace pixel GradientSHAP with an equal-height `24 px` horizontal strip-occlusion sweep across all 698 empty Camera 156 frames (black vs. image-mean fill) and pitch a consolidated zero-scroll `1440×780` Phase-1-style design before coding UI.
- **Prompt (important ones):**
  ```
  ok i clicked through `/atlas`, `/xai`, and `/circuits`, and honestly this is way too much right now - 3 huge dashboards with lots of text and card clutter like `0.0% empty` would totally lose the audience in a short talk (in phase 1 we kept the atlas super minimal, and for the talk i'd rather spell out Camera 156 / Camera 278 than make people decode C-codes).

  also pixel GradientSHAP clearly shatters into forest noise here. wouldn't a clean horizontal strip-occlusion sweep across all 698 empty Camera 156 frames - using equal-height 24 px strips anchored to the bottom bar so every strip covers the same area, comparing black vs image-mean fill - give a way clearer spatial test next to Grad-CAM?

  before writing UI code: run that 698-frame strip-occlusion check in WSL, and critique the 3 current views to pitch 1-2 ways to merge the strongest white-box parts into one single-screen linked view (fits my browser at ~1440x780 with no scroll, scales cleanly to 16:9, phase 1 white/gold/navy style - also say whether a light Frozen ImageNet baseline is worth keeping).
  ```
- **AI output summary:** Ran the 698-frame 18-strip (`24 px`, anchored `y = 424..448`) occlusion sweep in WSL: only the bottom bar strip produced a huge positive empty-logit jump (`Δz₀ = +9.00` black / `+9.04` mean) while strips `0..14` stayed negative (`−6.11` to `−1.40`), demonstrating why continuous `Δz₀` beats binary `% flipped` (Strip 15 edge case) and why `layer4` Pre-ReLU suppresses `empty` directly. Pitched two single-screen options; Option B paired 3 top columns with a full-width neighbor strip and a quiet Frozen ImageNet baseline.
- **Decision:** modified (selected Option B layout, but challenged Column 3's whole-channel clamping next)
- **Why:** The 18-strip occlusion test was dramatically cleaner than pixel GradientSHAP and Option B matched our minimal Phase 1 visual language.
- **Problems caused:** none (running experiments before UI code avoided wasted frontend work).
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Spatially Local Layer 49 Row Zeroing & Full `129,809`-Train kNN Check

- **Who:** Martin
- **Goal:** Replace whole-channel clamping with a spatially local `layer4` (`14×14`) row-zeroing intervention (photo untouched), testing every row `0..13`, `avgpool` vs. `renorm` mean, 500 other-camera test animals, Camera 156's 24 real animals, full `129,809`-train kNN, and comparing candidate layerwise bar-focus metrics.
- **Prompt (important ones):**
  ```
  nice, that strip-occlusion table is way clearer than pixel GradientSHAP - good catch on Strip 15 showing why continuous `Δz_0` (black + mean) beats binary `% flipped`, and on `L4` Pre-ReLU showing the white bar suppresses `empty` directly.

  i like Option B's left-to-right layout (3 columns on top + full-width neighbor strip on the bottom, keeping Frozen ImageNet strictly as a faint baseline / quiet toggle).

  before we build it, one thing i'm still sceptical about in Column 3 (and `/circuits`) is clamping/zeroing whole channels across the entire image - in a short talk it's hard to defend why 6 or 32 channels instead of 3, and turning off whole channels feels heavy-handed.

  what if we make Column 3's white-box intervention spatially local right inside `layer4` (the 14x14 latent map before `avgpool`): keep the input photo untouched and set ONLY the bottom bar area (bottom 1-2 rows) to 0 across all 2,048 channels?
  - test all 698 empty Camera 156 frames + every individual row `0..13`, plus top-2 and mid-2 row controls
  - check exact `avgpool` vs re-normalized mean over remaining rows (so it's not just scaling)
  - test ~500 test frames with real animals from other cameras (main accuracy check) + Camera 156's 24 real-animal frames (check rabbit/paca hallucinations drop)
  - check where Camera 156's `layer4` nearest train neighbors move after zeroing, querying against ALL 129,809 train embeddings (`outputs/embeddings.npy`, not just the small train kNN sample)

  am i missing any confound here? also check which layerwise bar-focus metric across `stem..layer4` (activation L1/energy share like Bar Act % vs standard Grad-CAM vs contrastive rabbit-minus-empty Grad-CAM vs that `L4` Pre-ReLU empty map) tells the cleanest, most honest story alongside `layer2` Grad-CAM. run in WSL first before any UI code.
  ```
- **AI output summary:** Discovered the key **Row 12 spillover** confound: exact additive per-row `z_empty` showed **Row 12 (`−4.56`)** suppresses `empty` even more strongly than **Row 13 (`−4.02`)** because Layer 49's receptive field straddles `y = 424`. Zeroing Row 13 alone lifts empty to `92.1%` but still leaves `33.2%` white-bar South America neighbors; zeroing **Rows 12–13** (`renorm` mean, identical argmax to exact `avgpool`) restores **`99.6%` empty** and **`0.0%` white-bar SA neighbors** (`89.0%` normal empty) with only `−1.6 pp` (`56.8% → 55.2%`) on 500 animals across 30 unseen test cameras. Also identified Contrastive Grad-CAM as the cleanest monotonic `Layer 1 → 49` metric (`8.8% → 77.4%` vs. `34.9%` Frozen ImageNet control).
- **Decision:** accepted
- **Why:** Spatially local row zeroing inside Layer 49 eliminated arbitrary channel-count choices and exposed a genuine white-box receptive-field insight (`Row 12` spillover) backed by exact full-`129,809`-train retrieval.
- **Problems caused:** none.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Building Option B (`/`) + Independent Blind UI & JSON Audits

- **Who:** Martin
- **Goal:** Build the single-screen Option B explorer on `/` (`Layer 1, 10, 22, 40, 49` + `fc = 50`, spelled-out camera names, clickable Night/Day split, `Cover bar (24 px)` + `renorm` row-zeroing linked to the full `129,809`-train neighbor strip), delete the 3 old prototypes, and run fresh unhinted blind browser + JSON/code audits before handoff.
- **Prompt (important ones):**
  ```
  nice, that row 12 spillover catch (still leaving 33.2% white-bar neighbors if you only zero row 13) is super convincing - way cleaner than 32-channel clamping.

  let's build Option B on `/` in `phase2/pair_b/` (phase 1 white/gold/navy style, zero scroll at 1440x780, scales to 16:9). label the stages by ResNet-50 depth (`Layer 1, 10, 22, 40, 49`, tiny note that `fc` is 50), write out `Camera 156` etc. everywhere instead of C-codes, and keep cards clean (secondary stats in tooltips).

  two quick tweaks to your Option B plan:
  - Col 1: add a clickable Night vs Day split, and show `rows 12-13` cleanly at Layer 49 (check Frozen Probe too) so it doesn't look like a weird dip after Layer 40.
  - Col 2: add strip hover + a `Cover bar` toggle that live-updates predictions and the bottom 129,809-train neighbor strip just like Col 3's `renorm` row-zeroing toggles do, with click-to-zoom on photos.

  save one clean JSON bundle, serve `/` on WSL/Windows, and remove the 3 old prototype views + old JSONs once `/` works. before showing me, run a fresh blind review (browser check at 1440x780 + JSON number audit) without hinting at our story - fix real bugs and tell me what it caught.
  ```
- **AI output summary:** Built `extract_data.py`, `camera156_whitebox.json`, `server.py`, `index.html`, and `app.js`, removed the 3 legacy views/JSONs, and ran two parallel blind audits (`1440×780` & `1280×720` browser reviewer + read-only JSON/code auditor), fixing all flagged defects and re-verifying before reporting back.
- **Decision:** accepted (with follow-up human wording & UX refinements)
- **Why:** Zero-scroll layout and linked interventions worked cleanly right away, and the blind reviews caught several real bugs before manual testing.
- **Problems caused:**
  - *Caught by blind JSON/code audit:* (1) `frames[0]` reproducibility mismatch from an in-place batch-1 patch vs. batch-32 `extract_data.py`; (2) `stripChart` x-axis (`−3.0..+10.5`) silently clamping 10 of 18 negative black-fill bars; (3) `sanitize_json` turning Python `bool`s into ints (`0`/`1`); (4) raw scientific species labels leaking in one tooltip.
  - *Caught by blind browser audit:* `Layer 49` / `L50 (fc)` SVG x-axis overlap, slight photo aspect distortion + strip misalignment (fixed via `ResizeObserver` `syncSpatialSquare()`), active-button hover contrast washout, `1280×720` footer collision, and German-locale dot thousands separators (`129.809`).
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Wording Fixes Missed by Blind Review & Phase 2 Slide Plan

- **Who:** Martin
- **Goal:** Fix cryptic UI abbreviations that blind reviewers let pass, inspect actual Layer 49 training neighbor photos for animal visibility, and outline a minimal 5-step slide flow matching our Phase 1 `Martin-midpoint.mp4` style (no voiceover).
- **Prompt (important ones):**
  ```
  clicked through `http://127.0.0.1:8875/` at 1440x780 - looks super clean, zero scroll. 2 small wording fixes the blind review missed:
  - never abbreviate ImageNet as `IN` - nobody reads `Frozen IN` as ImageNet; label it clearly as a control (`Frozen ImageNet control` / `ImageNet control` in tight pills), and also spell out `Label: Empty` (not `GT: Empty`) and `Layer 22` / `rows 12-13` in Panel 2's footer (no `L22` / `r12-13`)
  - tweak Panel 2's header so it fits all tabs + strips (e.g. `2. Spatial check: Grad-CAM vs. 24 px strips`)

  after that, let's plan our Phase 2 presentation (NO voiceover - i'll record my own ~40s narration to stay under 10 min). match our Phase 1 video style (`C:\Users\marti\Downloads\Martin-midpoint.mp4` / `/mnt/c/Users/marti/Downloads/Martin-midpoint.mp4` - white title slide + JKU logo, minimal `1 / 5` .. `5 / 5` build-up ending on live explorer clicks).

  before rendering slides: inspect the actual Layer 49 Night & Day nearest training photos (like that Camera 278 rabbit tile) to check animal visibility / whether a zoom crop helps, and pitch a super minimal 5-step plan where only what i'll say out loud is printed.
  ```
- **AI output summary:** Applied all wording fixes (`Frozen ImageNet control` / `ImageNet control`, `Label: Empty`, `2. Spatial check: Grad-CAM vs. 24 px strips`), inspected full-resolution Night IR & Day RGB neighbor JPEGs (confirming the Camera 278 rabbit was tiny and washed out, needing a `2.5×` contrast-lifted zoom inset, and the Camera 64 paca needed a `2×` inset), and proposed the 5-step order starting with the white-bar neighbors and ending live on `Day RGB → Zero 12–13`.
- **Decision:** accepted
- **Why:** Eliminated opaque shorthand for talk viewers and verified animal visibility on real neighbor photos before rendering slides.
- **Problems caused:** *Human catch over blind review:* neither blind sub-agent flagged `Frozen IN`, `GT: Empty`, or `L22` / `r12–13` because technically oriented reviewers mentally decoded them.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Rendering `1920×1080` Presentation Slides & Blind Visual Review vs. Phase 1

- **Who:** Martin
- **Goal:** Render the 5 static `1920×1080` slides (`slide_0_title.png` through `slide_4_local_fix.png`) into `phase2/pair_b/figures/` with the official JKU logo and zoom insets, then run a fresh blind visual review against Phase 1 reference frames in `/tmp/phase1_ref/`.
- **Prompt (important ones):**
  ```
  yeah that 5-step order makes sense (starting with the rabbit/paca white-bar neighbors is much clearer, and ending live on `Day RGB -> Zero 12-13` works well).

  render the static 1920x1080 slides (`title` + `1/5..4/5`) into `phase2/pair_b/figures/` first - use the real JKU logo from `Midpoint pres.pptx` or `Martin-midpoint.mp4`, put the `2.5x` rabbit and `2x` paca zoom insets on `1/5`, and keep text minimal (only numbers i'll actually say).

  before showing me, let a fresh blind visual reviewer check every PNG against 1-2 frames from `Martin-midpoint.mp4` (put ref frames in `/tmp`, not `figures/`) without telling it our story - ask what it gets in ~5s plus any overlap, clutter, C-codes, or style mismatch. fix real issues and tell me what it found.
  ```
- **AI output summary:** Extracted the RGBA JKU logo from `Midpoint pres.pptx`, rendered all 5 slides, extracted Phase 1 reference frames via `cv2.VideoCapture` into `/tmp/phase1_ref/`, ran an unhinted blind visual review, fixed every caught defect, and passed a fresh second blind review.
- **Decision:** accepted (with Slide 3 later simplified alongside the UI cleanup)
- **Why:** Matched the Phase 1 white/gold/navy slide rhythm cleanly and caught real visual flaws before review.
- **Problems caused:**
  - *Tooling hiccup:* headless Chrome video capture returned blank white frames; switched to `cv2.VideoCapture` in WSL.
  - *Caught by blind slide review:* right-edge text clipping on Slide 2, colliding x-axis labels, thick pillarboxes and overhanging zoom insets on Slide 1, a double horizontal rule on Slide 4, and undefined shorthand (`Δz₀`, `RF`, `(fc)`).
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Dropping Noisy `Layer 22 Grad-CAM` & Switching to ExMatchina (`Predicted class`) Neighbors

- **Who:** Martin
- **Goal:** Remove noisy `Layer 22 Grad-CAM` from Panel 2 and Slide 3 (defaulting Panel 2 to `Layer 49 Contrastive`), and default the bottom neighbor strip to ExMatchina-style nearest training frames of the predicted class while keeping an `All classes` toggle for transparency.
- **Prompt (important ones):**
  ```
  wait, two things feel off when clicking through `http://127.0.0.1:8875/`:

  1. `Layer 22 Grad-CAM` looks way too noisy on the leaves compared to `Layer 49 Contrastive`, which highlights the bottom bar cleanly. let's default Panel 2 to `Layer 49 Contrastive` and drop `Layer 22 Grad-CAM` (just keep `Layer 49 Contrastive` vs `Layer 49 ReLU` so you can see why standard Grad-CAM clips to zero) - and clean up `slide_3_spatial.png` the same way so the noisy leaf dots are gone.

  2. in the bottom neighbor strip, showing 3 `Camera 42 · Brocket deer` photos right under a `Brazilian rabbit` prediction is super confusing in a talk. let's default the neighbor tiles to ExMatchina-style nearest training frames of the predicted class (so Night rabbit pulls `Camera 278 / 42` rabbits, Day paca pulls `Camera 64` pacas, and `Cover bar` / `Zero 12-13` flips to normal empty cameras) - keep a small `Predicted class | All classes` toggle so the unfiltered `72.4%` white-bar South America stat stays transparent.

  update the JSON/UI + `slide_3_spatial.png` and re-check at 1440x780 before we do the video.
  ```
- **AI output summary:** Extended `extract_data.py` and `camera156_whitebox.json` (`~238 KB`) with exact full-`129,809`-train predicted-class kNN (`knn_pred`, `neighbors_pred`), defaulted the UI to `Layer 49 Contrastive` + `Predicted class` neighbors (Night baseline pulls 5/5 `Camera 278 / 42` rabbits; Day baseline pulls 5/5 `Camera 64` pacas; `Zero 12–13` flips to 5/5 normal empty cameras), and re-rendered `slide_3_spatial.png` as a clean 3-panel layout.
- **Decision:** accepted
- **Why:** Both changes removed major talk-time distractions without hiding the unfiltered `72.4%` white-bar South America statistic.
- **Problems caused:** *Human catch over blind review:* blind reviewers verified math and overlap, not whether showing 3 brocket deer photos under a rabbit prediction or noisy green-leaf Grad-CAM dots would confuse a live audience.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Removing Inverted Panel 3 Bar & Visually Verified Animal Bounding Boxes

- **Who:** Martin
- **Goal:** Simplify Panel 3 to two consistent higher-is-better impact bars (dropping the inverted `Camera 156 real animals (24)` bar) and draw accurate, non-guessed animal bounding boxes on non-empty neighbor tiles and in the zoom modal.
- **Prompt (important ones):**
  ```
  much better now - `Layer 49 Contrastive` and the ExMatchina rabbit/paca neighbors look way clearer!

  two quick UI cleanups before the video:
  1. drop that middle bar in Panel 3 (`Camera 156 real animals (24): rabbit/paca 17/24`) - it's inverted vs the other two bars and confusing without a long explanation. just keep the two clear bars
  2. on the non-empty neighbor tiles (and zoom modal), mark the actual animal with a clean thin bounding box so people can spot those tiny rabbits/pacas right away (check if annotations have bboxes, otherwise inspect each unique non-empty neighbor photo visually so every box sits accurately on the animal - don't guess coordinates).
  ```
- **AI output summary:** Removed the middle bar from Panel 3 (keeping the 24-animal check in the second bar's tooltip). Confirmed `data/iwildcam_v2.0` lacks MegaDetector bbox JSONs, visually inspected all 43 unique non-empty neighbor photos on a coordinate grid, verified 29 exact animal bounding boxes (leaving 14 empty sequence tails/obscured frames `bbox: null`), and mapped them via `object-fit: contain` geometry (`placeBbox`) on both tiles and the zoom modal.
- **Decision:** accepted
- **Why:** Made tiny IR rabbits and pacas immediately visible at presentation scale without fabricating boxes on empty sequence tails, and gave Panel 3 uniform bar polarity.
- **Problems caused:** none (strict visual coordinate verification prevented fabricated boxes).
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---

### 2026-11-21: Explicit Color Legends (Panel 2 & Panel 1 Stacked Bar) and Final Button Cleanup

- **Who:** Martin
- **Goal:** Label what blue and orange mean directly above Panel 2's heatmap (while dropping the `Layer 49 ReLU` button), and fix the unclear dark-orange `paca` segment in Panel 1's bottom stacked bar.
- **Prompt (important ones):**
  ```
  please actually denote what blue and orange indicates here on top with a small legend; also remove the layer 49 ReLU button
  ```
  *(With screenshot of Panel 2, followed by screenshot of Panel 1's bottom bar:)*
  ```
  also, it is currently not clear what the dark orange bar means... - please fix!
  ```
- **AI output summary:** Replaced Panel 2's mode toggle with a single-line inline legend (`Layer 49 Contrastive:` · amber-orange swatch `pushes rabbit` / dynamically `pushes paca` on Day RGB · blue swatch `pushes empty`, dimmed when `Cover bar` is active) and removed the `Layer 49 ReLU` toggle + dead code. Added matching `8×8 px` color swatches (`#DDAA33` rabbit, `#9c6a00` paca, `#004488` empty, `#cccccc` other) plus hover tooltips under Panel 1's stacked bar and spelled out `rabbit/paca` in the control readout, then verified across `All / Night IR / Day RGB` at `1440×780`.
- **Decision:** accepted
- **Why:** Resolved the last two visual-encoding ambiguities before recording and branch push.
- **Problems caused:** *Human catch via screenshots:* Panel 1 had used dark gold text (`#806000`) for `rabbit` and dark orange text (`#9c6a00`) for `paca` with no color swatches, making the two shades look nearly identical in text while the bar showed bright gold vs. dark orange; Panel 2's diverging blue/orange overlay had relied only on footer text rather than an explicit top legend.
- **Time spent verifying:** <!-- TODO: fill in minutes -->

---
