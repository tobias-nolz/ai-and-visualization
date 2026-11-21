# AI Log: Phase II, Pair B (AI-aided)

This log tracks our AI interactions during Phase II (Model Visualization) for Group *Lost in latent space*, Pair B (Martin Dallinger `k12205889`, Tobias Nolz `k12311040`). Verbatim prompts are kept as originally typed.

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns We Observed
- **Jumping to guesses instead of checking data:** Early on, the AI guessed rabbit-shaped rocks or mislabeled training photos and pitched ideas focused only on the final layer. Pushing it to check `empty` suppression and earlier layers first uncovered the actual white-bar shortcut.
- **Overbuilding cluttered dashboards:** When asked for design ideas, the AI built three huge, text-heavy dashboards with cryptic camera codes and noisy pixel maps that would never work in a short talk.
- **Silent bugs behind clean-looking charts:** Several plots looked fine at first glance while quietly cutting off negative bars, misaligning image overlays, or drifting from the data script. Running several GPU scripts at once also crashed WSL.
- **Blind AI reviews still miss what confuses real people:** Separate AI reviewers caught text overlaps, clipped axes, and mismatched numbers, but missed abbreviations like `Frozen IN`, noisy leaf heatmaps, confusing deer neighbors under a rabbit prediction, an inverted metric bar, and missing color legends.

### 2. Best Practices We Learned
- **Run small experiments before building UI:** Checking strip occlusion and local Layer 49 row zeroing first gave us a much stronger story (including the Row 12 spillover) and saved us from keeping weak methods like pixel SHAP or whole-channel clamping.
- **Combine blind AI checks with your own click-through:** Blind browser and data audits cleared out mechanical bugs fast, while clicking through the app myself caught everything that would confuse an audience.
- **One GPU job at a time:** Keeping CUDA scripts strictly sequential stopped WSL from running out of memory and crashing.
- **Check photos directly instead of guessing:** Because the dataset has no animal bounding boxes, having the AI inspect each neighbor photo before placing boxes kept them accurate and left frames without visible animals unmarked.

---

## Log of AI Interactions

### 2026-11-11: Initial White-Box Brainstorming

- **Who:** Martin
- **Goal:** Check the shared repo and brainstorm 3 white-box ways to compare the fine-tuned ResNet-50 against the frozen ImageNet baseline on Camera 156.
- **Prompt (important ones):**
  ```
  Hey, I'm doing Phase 2 for Pair B in phase2/pair_b/ (on WSL /home/martin/ai-and-visualization) and looking into why unseen test camera C156 (location=282, an African camera) predicts 440 sylvilagus brasiliensis rabbits and 171 cuniculus paca on empty frames. Before writing code, check README.md and common/model.py and pitch me 3 ways we can look inside the fine-tuned ResNet-50 vs. the frozen ImageNet linear probe to find out what it actually learned here.
  ```
- **AI output summary:** Pitched 3 ideas mostly focused on the last layer and classifier weights, and guessed the errors came from rabbit-shaped rocks or bad training labels.
- **Decision:** rejected
- **Why:** In Phase 1 we had already seen that training rabbits were real, and Camera 156 predicts rabbits even when real African animals are in the frame. Focusing only on the last layer also felt too narrow.
- **Problems caused:** wrong initial guess (blamed rocks and bad labels before checking the model logits and earlier layers).
- **Time spent verifying:** 15 minutes

---

### 2026-11-11: Testing Logit Suppression and Earlier Layers

- **Who:** Martin
- **Goal:** Run the base model outputs and test whether `empty` is being suppressed, which earlier ResNet layers break Camera 156, and what input occlusion finds before building any UI.
- **Prompt (important ones):**
  ```
  hold on, I'm sceptical about a few things here:
  - don't guess pareidolia or phantom labels yet. in phase 1 our checked train rabbits were real, and C156 predicts rabbit/paca even on the 24 frames with real monkeys/birds - so check if `empty` itself is getting suppressed, not just some rabbit-shaped rock.
  - all 3 pitches mostly stay at layer4 + fc (huge receptive fields) and weighting kNN by dw*f feels circular. since both models start from the same ImageNet weights, go deeper into conv1..layer4 + BN stats (e.g. layerwise drift or swapping/patching blocks) to see which stage actually breaks C156, and cross-check unit/spatial maps against real input occlusion or SHAP.

  kick off `make inference && make baselines` in WSL (`/home/martin/ai-and-visualization`) and spawn parallel sub-agents to probe these hypotheses on C156 + train cameras first. don't build any UI yet, just bring back the numbers so we can decide where to go.
  ```
- **AI output summary:** Showed that the model strongly suppresses the `empty` score because of the `24 px` white Bushnell bar at the bottom of the frame, which matches South American training cameras that have no empty frames. Also showed that standard final-layer Grad-CAM misses the bar.
- **Decision:** accepted
- **Why:** Gave us a solid, verified explanation of what happens inside the network before we started designing charts.
- **Problems caused:** bug — running four GPU jobs in parallel ran WSL out of memory and crashed it, so we had to restart WSL and run GPU scripts one at a time afterward.
- **Time spent verifying:** 25 minutes

---

### 2026-11-17: Building 3 White-Box Dashboard Prototypes

- **Who:** Martin
- **Goal:** Keep the analysis clearly white-box (not Phase 1 image edits) and build 3 interactive prototypes to see which visual angles worked best.
- **Prompt (important ones):**
  ```
  nice find on the bar, but careful: bar-pasting and jigsaw tests look like phase 1 black-box changes, whereas phase 2 needs to be clearly white-box according to the task description(weights, layerwise representations, gradients, internal channels, and comparing >=2 XAI methods).
  let's center the story inside the model: (1) how C156, the 7 white-bar SA train cameras, and normal empty frames move/cluster across layer1->layer4 and between Frozen ImageNet vs Fine-Tuned representation space, and (2) comparing why layer4 Grad-CAM fails vs layer2 Grad-CAM + SHAP + latent kNN, plus dissecting the specific internal channels and fc weights that detect the bar and split IR rabbit vs Day paca (keeping bar-masking just as a causal check).
  Pitch me 2-3 clean interactive visualization designs in `phase2/pair_b/` that let someone explore and understand this white-box mechanism clearly - design and code them up end-to-end and I'll select the most promising

  Use multiple sub-agents to do so: orchestrate independent builder / reviewer sub-agents and check if it visually makes sense and if it fits the task description before you show it to me!!! think insanely hard!
  ```
  *(Followed by `build all 3 and show me the results - wsl should be back up by now, no?` and `stop but keep the webserver running - i will now review myself!`.)*
- **AI output summary:** Built three full dashboards (`/atlas`, `/xai`, `/circuits`) on a local web server and ran automated layout and rubric checks across them.
- **Decision:** rejected
- **Why:** Seeing all three side by side made it obvious that three big dashboards were way too busy for a short talk, even though some of the underlying views were useful.
- **Problems caused:** The AI reviewers caught layout overflows and corrected the camera count (5 white-bar South American training cameras with zero empty frames, not 7). On manual review, I also found way too much text clutter, cryptic camera codes, and noisy pixel SHAP maps.
- **Time spent verifying:** 40 minutes

---

### 2026-11-20: Replacing 3 Dashboards and Pixel SHAP with Strip Occlusion

- **Who:** Martin
- **Goal:** Test a cleaner equal-height `24 px` horizontal strip-occlusion sweep across all 698 empty frames, and pitch a way to merge the best ideas into one simple screen.
- **Prompt (important ones):**
  ```
  ok i clicked through `/atlas`, `/xai`, and `/circuits`, and honestly this is way too much right now - 3 huge dashboards with lots of text and card clutter like `0.0% empty` would totally lose the audience in a short talk (in phase 1 we kept the atlas super minimal, and for the talk i'd rather spell out Camera 156 / Camera 278 than make people decode C-codes).

  also pixel GradientSHAP clearly shatters into forest noise here. wouldn't a clean horizontal strip-occlusion sweep across all 698 empty Camera 156 frames - using equal-height 24 px strips anchored to the bottom bar so every strip covers the same area, comparing black vs image-mean fill - give a way clearer spatial test next to Grad-CAM?

  before writing UI code: run that 698-frame strip-occlusion check in WSL, and critique the 3 current views to pitch 1-2 ways to merge the strongest white-box parts into one single-screen linked view (fits my browser at ~1440x780 with no scroll, scales cleanly to 16:9, phase 1 white/gold/navy style - also say whether a light Frozen ImageNet baseline is worth keeping).
  ```
- **AI output summary:** Ran the 18-strip occlusion sweep and confirmed that only covering the bottom white bar jumps the `empty` score up sharply, under both black and mean fill. Pitched two single-screen layouts; Option B put three columns on top and a neighbor strip along the bottom.
- **Decision:** modified
- **Why:** Strip occlusion was much clearer than pixel SHAP and Option B fit our minimal Phase 1 style, though I still wanted to replace whole-channel clamping in the third column before coding.
- **Problems caused:** none.
- **Time spent verifying:** 20 minutes

---

### 2026-11-21: Replacing Channel Clamping with Local Layer 49 Row Zeroing

- **Who:** Martin
- **Goal:** Test whether zeroing just the bottom rows inside Layer 49 (leaving the input photo untouched) works as a cleaner, spatially local white-box fix without breaking other test cameras.
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
- **AI output summary:** Found that Row 12 (just above the bar) actually suppresses `empty` more strongly than Row 13 because of receptive-field spillover. Zeroing Rows 12–13 inside Layer 49 restored almost all empty predictions and removed all South American white-bar neighbors across the full training set, with barely any drop on other test cameras.
- **Decision:** accepted
- **Why:** Much easier to explain and defend than turning off arbitrary channels, and the Row 12 spillover gave us a strong internal-model finding.
- **Problems caused:** none.
- **Time spent verifying:** 25 minutes

---

### 2026-11-21: Building the Single-Screen App + Blind Reviews

- **Who:** Martin
- **Goal:** Build the consolidated single-screen app on `/`, remove the 3 old prototypes, and run fresh blind browser and data audits before I reviewed it.
- **Prompt (important ones):**
  ```
  nice, that row 12 spillover catch (still leaving 33.2% white-bar neighbors if you only zero row 13) is super convincing - way cleaner than 32-channel clamping.

  let's build Option B on `/` in `phase2/pair_b/` (phase 1 white/gold/navy style, zero scroll at 1440x780, scales to 16:9). label the stages by ResNet-50 depth (`Layer 1, 10, 22, 40, 49`, tiny note that `fc` is 50), write out `Camera 156` etc. everywhere instead of C-codes, and keep cards clean (secondary stats in tooltips).

  two quick tweaks to your Option B plan:
  - Col 1: add a clickable Night vs Day split, and show `rows 12-13` cleanly at Layer 49 (check Frozen Probe too) so it doesn't look like a weird dip after Layer 40.
  - Col 2: add strip hover + a `Cover bar` toggle that live-updates predictions and the bottom 129,809-train neighbor strip just like Col 3's `renorm` row-zeroing toggles do, with click-to-zoom on photos.

  save one clean JSON bundle, serve `/` on WSL/Windows, and remove the 3 old prototype views + old JSONs once `/` works. before showing me, run a fresh blind review (browser check at 1440x780 + JSON number audit) without hinting at our story - fix real bugs and tell me what it caught.
  ```
- **AI output summary:** Built the single-screen explorer, deleted the old dashboards, ran independent browser and data checks, and fixed the issues they flagged before handing it over.
- **Decision:** accepted
- **Why:** The layout fit cleanly without scrolling, all controls stayed linked, and running blind reviews first cleared out real bugs before I touched it.
- **Problems caused:** bug — the blind data audit caught a cut-off strip-chart axis (clipping 10 of 18 bars), a script-vs-JSON mismatch on the first frame, and booleans saved as numbers; the blind browser audit caught overlapping axis labels, slight image stretching, bad button hover contrast, and dot thousands separators. All were fixed.
- **Time spent verifying:** 25 minutes

---

### 2026-11-21: Fixing Abbreviations and Planning the Slides

- **Who:** Martin
- **Goal:** Fix confusing shorthand that the blind reviews missed, check whether animals are actually visible in the neighbor photos, and plan a short 5-step slide flow matching our Phase 1 video.
- **Prompt (important ones):**
  ```
  clicked through `http://127.0.0.1:8875/` at 1440x780 - looks super clean, zero scroll. 2 small wording fixes the blind review missed:
  - never abbreviate ImageNet as `IN` - nobody reads `Frozen IN` as ImageNet; label it clearly as a control (`Frozen ImageNet control` / `ImageNet control` in tight pills), and also spell out `Label: Empty` (not `GT: Empty`) and `Layer 22` / `rows 12-13` in Panel 2's footer (no `L22` / `r12-13`)
  - tweak Panel 2's header so it fits all tabs + strips (e.g. `2. Spatial check: Grad-CAM vs. 24 px strips`)

  after that, let's plan our Phase 2 presentation (NO voiceover - i'll record my own ~40s narration to stay under 10 min). match our Phase 1 video style (`C:\Users\marti\Downloads\Martin-midpoint.mp4` / `/mnt/c/Users/marti/Downloads/Martin-midpoint.mp4` - white title slide + JKU logo, minimal `1 / 5` .. `5 / 5` build-up ending on live explorer clicks).

  before rendering slides: inspect the actual Layer 49 Night & Day nearest training photos (like that Camera 278 rabbit tile) to check animal visibility / whether a zoom crop helps, and pitch a super minimal 5-step plan where only what i'll say out loud is printed.
  ```
- **AI output summary:** Spelled out all shortened labels in the app, checked the full-size neighbor photos (finding the night rabbit was too tiny without a zoom inset), and proposed a 5-step story starting with the white-bar neighbors and ending on the live explorer.
- **Decision:** accepted
- **Why:** Made the dashboard readable for anyone seeing it cold, and checking the actual photos showed we needed zoom insets on Slide 1.
- **Problems caused:** Blind AI reviewers completely missed abbreviations like `Frozen IN` and `GT: Empty` because they understood the jargon themselves.
- **Time spent verifying:** 15 minutes

---

### 2026-11-21: Rendering Slides and Checking Against Phase 1

- **Who:** Martin
- **Goal:** Render the 5 static slides in our Phase 1 style (with the real JKU logo and animal zoom insets) and have a blind reviewer check them against frames from our midpoint video.
- **Prompt (important ones):**
  ```
  yeah that 5-step order makes sense (starting with the rabbit/paca white-bar neighbors is much clearer, and ending live on `Day RGB -> Zero 12-13` works well).

  render the static 1920x1080 slides (`title` + `1/5..4/5`) into `phase2/pair_b/figures/` first - use the real JKU logo from `Midpoint pres.pptx` or `Martin-midpoint.mp4`, put the `2.5x` rabbit and `2x` paca zoom insets on `1/5`, and keep text minimal (only numbers i'll actually say).

  before showing me, let a fresh blind visual reviewer check every PNG against 1-2 frames from `Martin-midpoint.mp4` (put ref frames in `/tmp`, not `figures/`) without telling it our story - ask what it gets in ~5s plus any overlap, clutter, C-codes, or style mismatch. fix real issues and tell me what it found.
  ```
- **AI output summary:** Rendered all five slides with the real JKU logo and zoom insets, checked them with a blind reviewer against Phase 1 frames, fixed the layout flaws it flagged, and passed a second check.
- **Decision:** accepted
- **Why:** Matched our Phase 1 look cleanly without leaving clipped labels or sloppy spacing.
- **Problems caused:** The first slide render had text clipped off the right edge of Slide 2, colliding axis labels, uneven photo borders on Slide 1, a doubled divider line on Slide 4, and leftover shorthand — all caught by the blind review and fixed.
- **Time spent verifying:** 20 minutes

---

### 2026-11-21: Dropping Layer 22 Grad-CAM and Switching to Predicted-Class Neighbors

- **Who:** Martin
- **Goal:** Remove the noisy Layer 22 heatmap from Panel 2 and Slide 3, and default the neighbor strip to nearest training photos of the predicted class so the images match the prediction in a talk.
- **Prompt (important ones):**
  ```
  wait, two things feel off when clicking through `http://127.0.0.1:8875/`:

  1. `Layer 22 Grad-CAM` looks way too noisy on the leaves compared to `Layer 49 Contrastive`, which highlights the bottom bar cleanly. let's default Panel 2 to `Layer 49 Contrastive` and drop `Layer 22 Grad-CAM` (just keep `Layer 49 Contrastive` vs `Layer 49 ReLU` so you can see why standard Grad-CAM clips to zero) - and clean up `slide_3_spatial.png` the same way so the noisy leaf dots are gone.

  2. in the bottom neighbor strip, showing 3 `Camera 42 · Brocket deer` photos right under a `Brazilian rabbit` prediction is super confusing in a talk. let's default the neighbor tiles to ExMatchina-style nearest training frames of the predicted class (so Night rabbit pulls `Camera 278 / 42` rabbits, Day paca pulls `Camera 64` pacas, and `Cover bar` / `Zero 12-13` flips to normal empty cameras) - keep a small `Predicted class | All classes` toggle so the unfiltered `72.4%` white-bar South America stat stays transparent.

  update the JSON/UI + `slide_3_spatial.png` and re-check at 1440x780 before we do the video.
  ```
- **AI output summary:** Added predicted-class neighbor lookups to the data bundle, switched the app to default to `Layer 49 Contrastive` and predicted-class neighbors (keeping `All classes` as a toggle), and cleaned up Slide 3 to three panels.
- **Decision:** accepted
- **Why:** Showing actual rabbits and pacas under those predictions and removing the messy leaf overlay made both the app and Slide 3 much easier to follow.
- **Problems caused:** Another thing the blind AI reviews missed — showing brocket deer photos under a rabbit prediction was technically accurate for unfiltered neighbors, but immediately confusing when clicking through yourself.
- **Time spent verifying:** 20 minutes

---

### 2026-11-21: Simplifying Panel 3 and Adding Animal Bounding Boxes

- **Who:** Martin
- **Goal:** Remove the confusing inverted middle bar in Panel 3 and add accurate bounding boxes around animals in the neighbor photos so viewers can spot them right away.
- **Prompt (important ones):**
  ```
  much better now - `Layer 49 Contrastive` and the ExMatchina rabbit/paca neighbors look way clearer!

  two quick UI cleanups before the video:
  1. drop that middle bar in Panel 3 (`Camera 156 real animals (24): rabbit/paca 17/24`) - it's inverted vs the other two bars and confusing without a long explanation. just keep the two clear bars
  2. on the non-empty neighbor tiles (and zoom modal), mark the actual animal with a clean thin bounding box so people can spot those tiny rabbits/pacas right away (check if annotations have bboxes, otherwise inspect each unique non-empty neighbor photo visually so every box sits accurately on the animal - don't guess coordinates).
  ```
- **AI output summary:** Removed the middle bar (keeping that number in a tooltip), checked that the dataset has no bounding-box annotations, visually inspected every unique non-empty neighbor photo to mark 29 visible animals (leaving 14 frames without visible animals unmarked), and added the boxes to the tiles and zoom popup.
- **Decision:** accepted
- **Why:** Made the tiny night-time rabbits easy to see immediately and kept both Panel 3 bars pointing in the same direction (higher = better).
- **Problems caused:** none.
- **Time spent verifying:** 20 minutes

---

### 2026-11-21: Adding Color Legends and Removing the Extra Button

- **Who:** Martin
- **Goal:** Explain what blue and orange mean directly above Panel 2's heatmap (while removing the unused `Layer 49 ReLU` button), and make it obvious what the dark-orange section means in Panel 1's bottom bar.
- **Prompt (important ones):**
  ```
  please actually denote what blue and orange indicates here on top with a small legend; also remove the layer 49 ReLU button
  ```
  *(With screenshot of Panel 2, followed by screenshot of Panel 1's bottom bar:)*
  ```
  also, it is currently not clear what the dark orange bar means... - please fix!
  ```
- **AI output summary:** Replaced the Panel 2 toggle with a one-line legend above the photo showing orange (`pushes rabbit` / `pushes paca`) vs. blue (`pushes empty`), and added matching color squares next to every label under Panel 1's stacked bar.
- **Decision:** accepted
- **Why:** Fixed the last two visual clarity issues I spotted when taking a final look before recording.
- **Problems caused:** Without color squares under the stacked bar, the gold and dark-orange text shades looked almost identical; sending quick screenshots fixed both issues right away.
- **Time spent verifying:** 15 minutes

---
