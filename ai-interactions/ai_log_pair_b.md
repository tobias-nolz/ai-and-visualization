# AI Interaction Log - Phase 2 (Pair B)

This log tracks our AI interactions during Phase 2 (What did the model learn?) for Group *Lost in latent space*, Pair B (Martin Dallinger `k12205889`, Tobias Nolz `k12311040`). Pair B worked on two parts: **Confusions** (why the background wins, Tobias) and **Hallucinations** (phantom rabbits on Camera 156, Martin). Verbatim prompts are kept as originally typed. The log documents:
- The important prompts we used
- How AI suggestions influenced our decisions
- Which suggestions were accepted, modified, or rejected, and why
- Failure patterns we observed
- Best practices we learned for using AI effectively

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns We Observed

#### Confusions

- **Precise numbers that were wrong:**
  In the concepts run, the AI claimed that two thirds of the "night" frames were taken in daylight, with a median brightness of 0.42. We could not reproduce either number: 42 % are daytime and the median brightness is 0.076. We had already repeated the claim before checking it, and only the next agent caught it.
- **Copying the paper's method would have misled us:**
  With the paper's IoU rule, network dissection labelled 2 of 2,048 units, which looks like "the model has no scenery units". In fact, big concepts like trees (47 % of the pixels) can't reach the IoU cut at all. The AI pointed this out itself this time, but if we had used the method without checking its assumptions on our data, we would have drawn the wrong conclusion.
- **Results that only work on one machine:**
  Two of the chapter figures first read cached arrays from `outputs/`, which isn't in git, so a clean checkout couldn't rebuild the page. Locally everything worked, which is why it was easy to miss.
- **Automated screenshots can mislead:**
  In screenshots that jumped straight to a step, the hero image seemed to switch at the wrong time. When we scrolled through normally, it switched correctly.

#### Hallucinations

- **Guessing before looking at the data:**
  Right at the start, the AI blamed rabbit-shaped rocks or bad training labels and only wanted to inspect the last layer. Once we pushed it to check earlier layers and whether `empty` itself was being suppressed, it found the actual 24 px white bottom bar.
- **Way too much UI by default:**
  When asked for design ideas, it built three huge dashboards full of text, `C156`-style codes, and noisy pixel SHAP maps. None of that works in a short talk until you make it cut everything down to one clean screen.
- **Bugs hidden in charts that look fine:**
  A few plots looked fine at first glance even though negative bars were cut off on the left, photos were slightly stretched, or the saved JSON didn't match the script. Running several GPU scripts at the same time also crashed WSL out of memory.
- **Automated AI reviews still miss what confuses a real viewer:**
  Separate AI reviewers caught text overlaps, cut-off axes, and mismatched numbers pretty well, but they never complained about labels like `Frozen IN` or `GT:`, noisy Layer 22 leaf dots, showing deer photos right under a rabbit prediction, a bar going in the opposite direction from the others, or missing color legends.

---

### 2. Best Practices We Learned

#### Confusions

- **Recompute every headline number before using it:**
  We reran the key numbers of each step ourselves (evidence split, vote shares, ablation, the 73 %). That is how we found the wrong greyscale claim.
- **Ask for built-in checks:**
  We asked the agents to check that the score parts add up to the logit and that the rebuilt composites give the same predictions as in Phase 1, and to compare the ablation with random units. These checks made the results much easier to trust.
- **Load every number on the page from the results:**
  Every number in the story comes from files written by the export script, so the text can't drift away from the results.
- **Let the AI check our own claims:**
  We gave it our claim that "the animal mostly comes back as the wrong one" together with the data. It changed it to "73 % of the animals that reappear get the wrong name", which is what the data supports.
- **Short prompts, one question each:**
  Our first prompts were long lists of specs. Short prompts with one goal and a couple of constraints were easier to review, and the AI looked up the missing details in the code and results on its own.

#### Hallucinations

- **Test the numbers before building any UI:**
  Running the strip-occlusion sweep and Layer 49 row-zeroing checks first gave us a much clearer finding (including that Row 12 above the bar carries even more of the shortcut than Row 13) and kept us from wasting time on pixel SHAP or clamping 32 channels.
- **Use blind AI checks for layout and data bugs, then click through yourself:**
  Fresh sub-agents caught typos, overlaps, and clipped bars fast, while actually clicking through the app myself caught everything that would confuse someone watching the talk.
- **Run GPU jobs one by one in WSL:**
  Parallel PyTorch runs ran WSL out of RAM; running them one after another avoided crashes.
- **Make the AI look at the actual photos instead of guessing:**
  Since our iWildCam files don't include animal bounding boxes, making the AI actually view each neighbor photo before drawing boxes kept them in the right spots and left frames without a visible animal blank.

---

## Log of AI Interactions: Confusions (Tobias)

### 2026-11-11: Network Dissection Pipeline

- **Who:** Tobias
- **Goal:** Find out which units of the ResNet-50 detect scenery (trees, grass, ground, sky, ...) and which detect
  the animal, for both the fine-tuned model and the ImageNet backbone.
- **Prompt (important ones):**
  ```
  Can you implement network dissection (Bau et al. 2017) for our WILDS ResNet-50 in `phase2/pair_b/src/dissect.py`? We want to know which units detect scenery (trees, grass, ground, sky...) and which detect the animal.

  Use ~4000 daytime frames spread over cameras and habitats. Concepts: SegFormer ADE20K masks merged into tree, plant, grass, ground, rock, sky, water, plus "animal" from the MegaDetector boxes and "info strip" for the timestamp bars. Look at layer3 and layer4 of the fine-tuned model and the ImageNet backbone. Same method as the paper (top 0.5 % threshold, IoU > 0.04).

  Save the unit labels and top images to `phase2/pair_b/results/` (small files) and make contact sheets so we can check the labels by eye. GPU is shared, don't commit.
  ```
- **AI output summary:** Wrote `dissect.py` and ran it the way the paper does on 3980 daytime frames. Almost no scenery units got a label: a unit's mask covers about 0.4 % of the pixels, so it can't reach IoU 0.04 against a concept as large as tree (47 %). It listed alternatives and left the choice to us.
- **Decision:** modified
- **Why:** The paper's IoU rule labelled almost no scenery units. We kept that as a negative result and switched to total-activation profiles in the next step.
- **Problems caused:** none found. We checked the IoU bound and the in-box shares against the saved files, and the contact sheets by eye.
- **Time spent verifying:** 30 minutes

---

### 2026-11-11: Splitting Each Prediction into Animal and Background Evidence

- **Who:** Tobias
- **Goal:** For every Phase 1 composite, measure how much of the true-class and predicted-class score comes from
  the pasted animal and how much from the background.
- **Prompt (important ones):**
  ```
  We want to know how much of each background-swap prediction comes from the pasted animal and how much from the background. Can you write `phase2/pair_b/src/evidence_split.py`?

  Rebuild the Phase 1 composites (check the predictions match), then split the true- and predicted-class logit of the fine-tuned model into the part inside the animal mask and the part outside, using the class activation maps. Check that the parts add up. One row per composite in `results/evidence_split.parquet`; keep the features and maps in `outputs/` for later. Do the linear probe too if it's cheap.

  Give me a few summary numbers, e.g. inside vs. outside for correct vs. wrong composites per habitat. GPU is shared, don't commit.
  ```
- **AI output summary:** Wrote `evidence_split.py` and ran it on all 179,820 composites. The predictions match Phase 1 and the parts add up to the logit. In wrong composites the animal region slightly favours the true class (-0.79 logits), while the background strongly favours the winner (+7.49).
- **Decision:** accepted
- **Why:** The split is exact, our checks passed, and it became chapter 2 of the story.
- **Problems caused:** none found. We recomputed the summary table and six random composites from scratch on the CPU; all matched within 0.005.
- **Time spent verifying:** 25 minutes

---

### 2026-11-11: Skeleton of the Story Page

- **Who:** Tobias
- **Goal:** Set up the scrollable story page that will hold all Phase 2 results: hero, Phase 1 recap, and
  placeholders for the chapters that wait on the dissection and evidence-split results.
- **Prompt (important ones):**
  ```
  Please set up the skeleton of our Phase 2 story page in `phase2/pair_b/story/`: a scrollable Distill-style article on why our camera-trap model relies on the background. Working title "The elephant that turned into a bird".

  Static page, no build step, D3 + scrollama stored locally, sticky graphic with text steps. Light editorial look, Inter font, our Phase 1 colours. Data comes from `phase2/pair_b/src/export_story_data.py` (small JSON + WebP), all numbers from the existing results.

  Chapters: a hero where one elephant cut-out cross-fades over a savanna, mixed and forest frame with the model's real answer each time (elephant → great argus), a Phase 1 recap (territory map, grey-frame control, top-row effect), placeholders for chapters 2–5, and "Methods & caveats".

  Check it with playwright screenshots at desktop and laptop width. Don't touch the dissection/evidence-split scripts, don't commit.
  ```
- **AI output summary:** Built the page with D3 and scrollama: hero, Phase 1 recap, placeholders for chapters 2 to 5 and a methods section. All numbers are loaded from JSON. It noted that the "top row = empty" effect shows up on a plain grey frame too.
- **Decision:** accepted
- **Why:** It gave us a working base for the page. Later we dropped the mixed frame and moved the grey-frame and top-row details into "more" boxes.
- **Problems caused:** none found. We recomputed the key numbers from the parquet files. The hero only looked broken in screenshots that jumped straight to a step.
- **Time spent verifying:** 20 minutes

---

### 2026-11-11: Which Scenery the Votes Come From (Concept Profiles, Vote Split, Ablation)

- **Who:** Tobias
- **Goal:** The paper's IoU rule could not label scenery units on our images. Instead, measure where each unit's
  total activation lies, split each class's vote by scene concept, and test causally by switching the
  scenery units off.
- **Prompt (important ones):**
  ```
  Follow-up on the dissection: with the paper's IoU rule scenery units can't get labelled on our data (tree covers ~47 % of the pixels). We don't have to follow the paper exactly, we just want to know which scenery the votes come from. Can you write `phase2/pair_b/src/concepts.py`?

  1. Add val/id_val frames to the probe set so open habitat is better covered.
  2. For each layer4 unit, measure which share of its total activation falls on each concept compared to the concept's area, and label it by its strongest concept (or "mixed").
  3. Split the class logits by concept region, for the composites and real images, and report which concepts carry the winning class in wrong composites per habitat.
  4. Switch off the tree-leaning (and separately ground-leaning) units and compare with random units: how do composite outcomes and real test accuracy change?

  Small result files, GPU is free but stay under ~10 GB. Don't commit.
  ```
- **AI output summary:** Wrote `concepts.py`. Labelled by total activation, the fine-tuned layer4 has 540 animal units and only 59 scenery units, and the winner's lead comes from the whole background, roughly in proportion to its area. Switching off the 256 most tree-leaning units halves "empty" in forest, while test accuracy barely moves (70.2 % → 69.3 %). It also claimed that two thirds of the "night" frames were taken in daylight, which turned out to be wrong (see below).
- **Decision:** modified
- **Why:** The unit labels, vote split and ablation held up when we recomputed them. We dropped the greyscale daylight claim because it was wrong.
- **Problems caused:** A usage limit interrupted the run once. Wrong insight: we couldn't reproduce the "two thirds daylight" claim (median brightness 0.42); the real values are 42 % daytime and a median brightness of 0.076. We repeated it before checking, and the next agent caught it. All other numbers matched our own recomputation and ablation rerun.
- **Time spent verifying:** 45 minutes

---

### 2026-11-11: Cutting the Story Down to a 2-Minute Talk

- **Who:** Tobias
- **Goal:** My part of the final presentation (both phases) has less than 2 minutes, so the story page needs a short
  main line for presenting, with the details in collapsible boxes for readers.
- **Prompt (important ones):**
  ```
  The story page also has to work as my presentation, and I get less than 2 minutes for both phases. Can you restructure it?

  Main line: hero (only open and forest frame) → one territory-map step → chapters 2–5 → one-line takeaway, about 10 steps with ~2 sentences each. Everything else goes into collapsible "more" boxes, closed by default. Add arrow key / space navigation between steps, and keep the hero image box the same size for both frames.

  Numbers still from the JSON. Check with playwright at desktop and laptop width. Don't touch the analysis scripts, don't commit.
  ```
- **AI output summary:** Rebuilt the page as a 9-step main line with the details in closed "more" boxes, and added arrow-key / space navigation.
- **Decision:** accepted
- **Why:** Without a short main line the page couldn't double as my presentation.
- **Problems caused:** none found. We stepped through the page with the arrow keys and checked the screenshots.
- **Time spent verifying:** 10 minutes

---

### 2026-11-19: Filling the Story Chapters with the Results

- **Who:** Tobias
- **Goal:** Replace the placeholders in chapters 2 to 5 with real figures from the evidence split, the dissection,
  the concept vote split and the ablation, and keep the main line short enough for the talk.
- **Prompt (important ones):**
  ```
  Please fill chapters 2–5 of the story page with real figures from our results (extend `export_story_data.py`, no hand-typed numbers):

  - Ch. 2: in wrong composites the animal slightly favours the true class, the background strongly favours the winner. Show this per habitat plus the forest hero composite with its class maps as overlay.
  - Ch. 3: the IoU rule labels almost no scenery units; by total activation there are 540 animal units and only a few dozen scenery units. Maybe a unit mosaic plus 2–3 example units.
  - Ch. 4: plot each concept's vote share against its area share.
  - Ch. 5: switching off the tree units halves "empty" in forest but the animal mostly comes back as the wrong one. Show it vs. random units and the cost on test accuracy.
  - Rewrite the takeaway, something like "No forest detector overrules the elephant. A small animal is simply outvoted by a frame full of scenery."

  Also rename the "night" group to "greyscale", and keep the main line around 250 words. Same look and navigation. Check with playwright, don't commit.
  ```
- **AI output summary:** Filled chapters 2 to 5 with charts from the results and rewrote the takeaway. It also corrected our claim that the animal "mostly" comes back as the wrong one to "73 % of the animals that reappear get the wrong name".
- **Decision:** accepted
- **Why:** Its correction of our "mostly the wrong animal" claim was right, so we kept it.
- **Problems caused:** none found. We recomputed the 73 % (10,925 of 14,962), the forest tree shares (63 % / 65 %) and the greyscale facts.
- **Time spent verifying:** 25 minutes

---

### 2026-11-19: Interactive Controls for the Story Chapters

- **Who:** Tobias
- **Goal:** Drop the unused "Try it yourself" section, make the page rebuildable from files in git, and let
  readers explore each chapter with a few controls while the presented default views stay unchanged.
- **Prompt (important ones):**
  ```
  Three changes to the story page:

  1. Remove the "Try it yourself" section.
  2. Ch. 2 and 4 read arrays from `outputs/` which aren't in git. Save small extracts to `phase2/pair_b/results/` so a clean checkout can rebuild the page.
  3. Add a few controls so readers can explore, e.g. pick the animal on the map, pick another composite in ch. 2, switch the labelling rule in ch. 3, pick habitat in ch. 4, and a slider for how many units get switched off in ch. 5. Fewer good controls beat many.

  The default view of every chart has to stay exactly as it is now since that's what I present. Controls next to their chart and they must not break the keyboard navigation. Check with playwright including the controls, don't commit.
  ```
- **AI output summary:** Removed "Try it yourself", saved small extracts (~170 kB) so the page rebuilds from git, and added one or two controls per chapter with unchanged default views.
- **Decision:** accepted
- **Why:** The default views didn't change, so neither did the version I present.
- **Problems caused:** none found. The k = 256 row matches our earlier ablation rerun, and the export no longer reads `outputs/` by default.
- **Time spent verifying:** 15 minutes

---

## Log of AI Interactions: Hallucinations (Martin)

### 2026-11-11: Initial Brainstorming on Camera 156

- **Who:** Martin
- **Goal:** Look at the repo and brainstorm 3 ways to open up the fine-tuned ResNet-50 vs. the frozen ImageNet baseline on Camera 156.
- **Prompt (important ones):**
  ```
  Hey, I'm doing Phase 2 for Pair B in phase2/pair_b/ (on WSL /home/martin/ai-and-visualization) and looking into why unseen test camera C156 (location=282, an African camera) predicts 440 sylvilagus brasiliensis rabbits and 171 cuniculus paca on empty frames. Before writing code, check README.md and common/model.py and pitch me 3 ways we can look inside the fine-tuned ResNet-50 vs. the frozen ImageNet linear probe to find out what it actually learned here.
  ```
- **AI output summary:** Suggested 3 ideas mostly focused on the last layer and classifier weights, and guessed the hallucinations came from rabbit-shaped rocks or bad training labels.
- **Decision:** rejected
- **Why:** From Phase 1 we already knew the training rabbits were real, and Camera 156 still predicts rabbits even when real African animals are in the frame. Looking only at the last layer also felt too narrow.
- **Problems caused:** Wrong first guess (blamed rocks and bad labels before checking logits and earlier layers).
- **Time spent verifying:** 30 minutes

---

### 2026-11-11: Checking `empty` Suppression and Earlier Layers

- **Who:** Martin
- **Goal:** Run the base model outputs and check if `empty` is getting suppressed, which ResNet layers break Camera 156, and what occlusion finds before building any UI.
- **Prompt (important ones):**
  ```
  hold on, I'm sceptical about a few things here:
  - don't guess pareidolia or phantom labels yet. in phase 1 our checked train rabbits were real, and C156 predicts rabbit/paca even on the 24 frames with real monkeys/birds - so check if `empty` itself is getting suppressed, not just some rabbit-shaped rock.
  - all 3 pitches mostly stay at layer4 + fc (huge receptive fields) and weighting kNN by dw*f feels circular. since both models start from the same ImageNet weights, go deeper into conv1..layer4 + BN stats (e.g. layerwise drift or swapping/patching blocks) to see which stage actually breaks C156, and cross-check unit/spatial maps against real input occlusion or SHAP.

  kick off `make inference && make baselines` in WSL (`/home/martin/ai-and-visualization`) and spawn parallel sub-agents to probe these hypotheses on C156 + train cameras first. don't build any UI yet, just bring back the numbers so we can decide where to go.
  ```
- **AI output summary:** Found that the model strongly suppresses `empty` because of the 24 px white Bushnell bar at the bottom of the photo, which matches South American training cameras that have no empty frames. Also showed that standard final-layer Grad-CAM misses the bar.
- **Decision:** accepted
- **Why:** Gave us a real, checked explanation of what happens inside the network before designing any charts.
- **Problems caused:** Running four GPU jobs at once ran WSL out of memory and crashed it, so we had to restart WSL and run GPU scripts one at a time after that.
- **Time spent verifying:** 60 minutes

---

### 2026-11-17: Building 3 Prototype Dashboards

- **Who:** Martin
- **Goal:** Keep Phase 2 clearly white-box (not Phase 1 image edits) and try out 3 interactive dashboards to see which views actually work.
- **Prompt (important ones):**
  ```
  nice find on the bar, but careful: bar-pasting and jigsaw tests look like phase 1 black-box changes, whereas phase 2 needs to be clearly white-box according to the task description(weights, layerwise representations, gradients, internal channels, and comparing >=2 XAI methods).
  let's center the story inside the model: (1) how C156, the 7 white-bar SA train cameras, and normal empty frames move/cluster across layer1->layer4 and between Frozen ImageNet vs Fine-Tuned representation space, and (2) comparing why layer4 Grad-CAM fails vs layer2 Grad-CAM + SHAP + latent kNN, plus dissecting the specific internal channels and fc weights that detect the bar and split IR rabbit vs Day paca (keeping bar-masking just as a causal check).
  Pitch me 2-3 clean interactive visualization designs in `phase2/pair_b/` that let someone explore and understand this white-box mechanism clearly - design and code them up end-to-end and I'll select the most promising

  Use multiple sub-agents to do so: orchestrate independent builder / reviewer sub-agents and check if it visually makes sense and if it fits the task description before you show it to me!!! think insanely hard!
  ```
  *(Followed by `build all 3 and show me the results - wsl should be back up by now, no?` and `stop but keep the webserver running - i will now review myself!`.)*
- **AI output summary:** Built three separate dashboards (`/atlas`, `/xai`, `/circuits`) on a local server and ran automated layout and checklist reviews on them.
- **Decision:** rejected
- **Why:** Once I clicked through all three, it was obvious they were way too big and busy for a short talk, even though parts of them were useful.
- **Problems caused:** The AI reviewers caught layout overflows and fixed the camera count (5 white-bar South American training cameras with zero empty frames, not 7). Clicking through myself, there was still way too much text, `C156`-style codes everywhere, and noisy pixel SHAP maps.
- **Time spent verifying:** 90 minutes

---

### 2026-11-20: Replacing Pixel SHAP with 24 px Strip Occlusion & Cutting Down to One Screen

- **Who:** Martin
- **Goal:** Run an equal-height 24 px horizontal strip-occlusion sweep across all 698 empty frames, and figure out how to merge the useful parts into one simple screen.
- **Prompt (important ones):**
  ```
  ok i clicked through `/atlas`, `/xai`, and `/circuits`, and honestly this is way too much right now - 3 huge dashboards with lots of text and card clutter like `0.0% empty` would totally lose the audience in a short talk (in phase 1 we kept the atlas super minimal, and for the talk i'd rather spell out Camera 156 / Camera 278 than make people decode C-codes).

  also pixel GradientSHAP clearly shatters into forest noise here. wouldn't a clean horizontal strip-occlusion sweep across all 698 empty Camera 156 frames - using equal-height 24 px strips anchored to the bottom bar so every strip covers the same area, comparing black vs image-mean fill - give a way clearer spatial test next to Grad-CAM?

  before writing UI code: run that 698-frame strip-occlusion check in WSL, and critique the 3 current views to pitch 1-2 ways to merge the strongest white-box parts into one single-screen linked view (fits my browser at ~1440x780 with no scroll, scales cleanly to 16:9, phase 1 white/gold/navy style - also say whether a light Frozen ImageNet baseline is worth keeping).
  ```
- **AI output summary:** Ran the 18-strip sweep and showed that only covering the bottom white bar pushes the `empty` logit up sharply (under both black and mean fill). Pitched two single-screen options; Option B had three columns on top and a neighbor strip along the bottom.
- **Decision:** modified
- **Why:** Strip occlusion was way clearer than pixel SHAP and Option B matched our minimal Phase 1 look, though I still wanted to replace whole-channel clamping in Column 3 before building it.
- **Problems caused:** none.
- **Time spent verifying:** 60 minutes

---

### 2026-11-21: Testing Local Layer 49 Row Zeroing Instead of Channel Clamping

- **Who:** Martin
- **Goal:** Check if zeroing just the bottom rows inside Layer 49 (without touching the input photo) fixes Camera 156 cleanly without hurting other test cameras.
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
- **AI output summary:** Found that Row 12 (just above the white bar) actually suppresses `empty` even more than Row 13 because Layer 49's receptive field bleeds upward. Zeroing Rows 12–13 fixed almost all empty predictions and dropped all South American white-bar neighbors across the full training set, with barely any accuracy drop on other cameras.
- **Decision:** accepted
- **Why:** Way easier to explain and defend than turning off arbitrary channels, and the Row 12 spillover was a really interesting finding.
- **Problems caused:** none.
- **Time spent verifying:** 60 minutes

---

### 2026-11-21: Building the Single-Screen App + Blind Reviews

- **Who:** Martin
- **Goal:** Build the single-screen explorer on `/`, remove the 3 old prototypes, and let fresh blind reviewers check the browser UI and JSON numbers before I looked at it.
- **Prompt (important ones):**
  ```
  nice, that row 12 spillover catch (still leaving 33.2% white-bar neighbors if you only zero row 13) is super convincing - way cleaner than 32-channel clamping.

  let's build Option B on `/` in `phase2/pair_b/` (phase 1 white/gold/navy style, zero scroll at 1440x780, scales to 16:9). label the stages by ResNet-50 depth (`Layer 1, 10, 22, 40, 49`, tiny note that `fc` is 50), write out `Camera 156` etc. everywhere instead of C-codes, and keep cards clean (secondary stats in tooltips).

  two quick tweaks to your Option B plan:
  - Col 1: add a clickable Night vs Day split, and show `rows 12-13` cleanly at Layer 49 (check Frozen Probe too) so it doesn't look like a weird dip after Layer 40.
  - Col 2: add strip hover + a `Cover bar` toggle that live-updates predictions and the bottom 129,809-train neighbor strip just like Col 3's `renorm` row-zeroing toggles do, with click-to-zoom on photos.

  save one clean JSON bundle, serve `/` on WSL/Windows, and remove the 3 old prototype views + old JSONs once `/` works. before showing me, run a fresh blind review (browser check at 1440x780 + JSON number audit) without hinting at our story - fix real bugs and tell me what it caught.
  ```
- **AI output summary:** Built the single-screen app, removed the 3 old dashboards, ran independent browser and JSON checks, and fixed the issues they found before handing it over.
- **Decision:** accepted
- **Why:** Everything fit on one screen without scrolling, all controls stayed linked, and the blind reviews caught several real bugs before I even opened it.
- **Problems caused:** The blind JSON check caught a cut-off strip-chart axis (clipping 10 of 18 bars), a script-vs-JSON mismatch on the first frame, and booleans saved as numbers. The blind browser check caught overlapping axis labels, slight photo stretching, bad button hover contrast, and dot thousands separators. All were fixed.
- **Time spent verifying:** 60 minutes

---

### 2026-11-21: Fixing Abbreviations and Planning the Slides

- **Who:** Martin
- **Goal:** Fix confusing shorthand the blind reviews missed, check if animals are actually visible in the neighbor photos, and plan a short 5-step slide sequence matching our Phase 1 video.
- **Prompt (important ones):**
  ```
  clicked through `http://127.0.0.1:8875/` at 1440x780 - looks super clean, zero scroll. 2 small wording fixes the blind review missed:
  - never abbreviate ImageNet as `IN` - nobody reads `Frozen IN` as ImageNet; label it clearly as a control (`Frozen ImageNet control` / `ImageNet control` in tight pills), and also spell out `Label: Empty` (not `GT: Empty`) and `Layer 22` / `rows 12-13` in Panel 2's footer (no `L22` / `r12-13`)
  - tweak Panel 2's header so it fits all tabs + strips (e.g. `2. Spatial check: Grad-CAM vs. 24 px strips`)

  after that, let's plan our Phase 2 presentation (NO voiceover - i'll record my own ~40s narration to stay under 10 min). match our Phase 1 video style (`C:\Users\marti\Downloads\Martin-midpoint.mp4` / `/mnt/c/Users/marti/Downloads/Martin-midpoint.mp4` - white title slide + JKU logo, minimal `1 / 5` .. `5 / 5` build-up ending on live explorer clicks).

  before rendering slides: inspect the actual Layer 49 Night & Day nearest training photos (like that Camera 278 rabbit tile) to check animal visibility / whether a zoom crop helps, and pitch a super minimal 5-step plan where only what i'll say out loud is printed.
  ```
- **AI output summary:** Spelled out all shortened labels in the app, checked the full-size neighbor photos (seeing that the night rabbit was way too small without a zoom inset), and proposed a 5-step sequence starting with the white-bar neighbors and ending on the live app.
- **Decision:** accepted
- **Why:** Made the labels readable for anyone seeing the app for the first time, and checking the actual photos showed we needed zoom insets on Slide 1.
- **Problems caused:** The blind AI reviewers completely missed shorthand like `Frozen IN` and `GT: Empty` because they already knew what it meant.
- **Time spent verifying:** 45 minutes

---

### 2026-11-21: Rendering Slides and Checking Against Phase 1

- **Who:** Martin
- **Goal:** Render the 5 static slides in our Phase 1 style (with the real JKU logo and zoom insets) and have a blind reviewer compare them against frames from our midpoint video.
- **Prompt (important ones):**
  ```
  yeah that 5-step order makes sense (starting with the rabbit/paca white-bar neighbors is much clearer, and ending live on `Day RGB -> Zero 12-13` works well).

  render the static 1920x1080 slides (`title` + `1/5..4/5`) into `phase2/pair_b/figures/` first - use the real JKU logo from `Midpoint pres.pptx` or `Martin-midpoint.mp4`, put the `2.5x` rabbit and `2x` paca zoom insets on `1/5`, and keep text minimal (only numbers i'll actually say).

  before showing me, let a fresh blind visual reviewer check every PNG against 1-2 frames from `Martin-midpoint.mp4` (put ref frames in `/tmp`, not `figures/`) without telling it our story - ask what it gets in ~5s plus any overlap, clutter, C-codes, or style mismatch. fix real issues and tell me what it found.
  ```
- **AI output summary:** Rendered all five slides with the real JKU logo and zoom insets, ran a blind visual check against Phase 1 frames, fixed the layout issues it caught, and re-checked them.
- **Decision:** accepted
- **Why:** Matched our Phase 1 slides cleanly without cut-off text or awkward spacing.
- **Problems caused:** The first render had text cut off on the right of Slide 2, colliding axis labels, uneven photo borders on Slide 1, a doubled divider line on Slide 4, and leftover shorthand. The blind review caught all of them.
- **Time spent verifying:** 60 minutes

---

### 2026-11-21: Dropping Layer 22 Grad-CAM and Switching to Predicted-Class Neighbors

- **Who:** Martin
- **Goal:** Drop the noisy Layer 22 heatmap from Panel 2 and Slide 3, and default the neighbor strip to the closest training photos of the predicted class so the photos actually match the prediction.
- **Prompt (important ones):**
  ```
  wait, two things feel off when clicking through `http://127.0.0.1:8875/`:

  1. `Layer 22 Grad-CAM` looks way too noisy on the leaves compared to `Layer 49 Contrastive`, which highlights the bottom bar cleanly. let's default Panel 2 to `Layer 49 Contrastive` and drop `Layer 22 Grad-CAM` (just keep `Layer 49 Contrastive` vs `Layer 49 ReLU` so you can see why standard Grad-CAM clips to zero) - and clean up `slide_3_spatial.png` the same way so the noisy leaf dots are gone.

  2. in the bottom neighbor strip, showing 3 `Camera 42 · Brocket deer` photos right under a `Brazilian rabbit` prediction is super confusing in a talk. let's default the neighbor tiles to ExMatchina-style nearest training frames of the predicted class (so Night rabbit pulls `Camera 278 / 42` rabbits, Day paca pulls `Camera 64` pacas, and `Cover bar` / `Zero 12-13` flips to normal empty cameras) - keep a small `Predicted class | All classes` toggle so the unfiltered `72.4%` white-bar South America stat stays transparent.

  update the JSON/UI + `slide_3_spatial.png` and re-check at 1440x780 before we do the video.
  ```
- **AI output summary:** Added predicted-class neighbors to the JSON, switched the app to default to `Layer 49 Contrastive` and predicted-class neighbors (keeping `All classes` as a toggle), and cleaned up Slide 3 to three panels.
- **Decision:** accepted
- **Why:** Showing actual rabbits and pacas under those predictions and dropping the messy leaf overlay made both the app and Slide 3 way easier to follow.
- **Problems caused:** Another thing the blind AI reviews missed: showing brocket deer photos under a rabbit prediction was technically right for unfiltered kNN, but confusing the second you actually look at it.
- **Time spent verifying:** 50 minutes

---

### 2026-11-21: Cleaning Up Panel 3 and Adding Animal Bounding Boxes

- **Who:** Martin
- **Goal:** Drop the confusing middle bar in Panel 3 and draw thin bounding boxes around animals in the neighbor photos so people can actually spot them.
- **Prompt (important ones):**
  ```
  much better now - `Layer 49 Contrastive` and the ExMatchina rabbit/paca neighbors look way clearer!

  two quick UI cleanups before the video:
  1. drop that middle bar in Panel 3 (`Camera 156 real animals (24): rabbit/paca 17/24`) - it's inverted vs the other two bars and confusing without a long explanation. just keep the two clear bars
  2. on the non-empty neighbor tiles (and zoom modal), mark the actual animal with a clean thin bounding box so people can spot those tiny rabbits/pacas right away (check if annotations have bboxes, otherwise inspect each unique non-empty neighbor photo visually so every box sits accurately on the animal - don't guess coordinates).
  ```
- **AI output summary:** Removed the middle bar (kept that check in a tooltip), confirmed the dataset has no bounding-box files, inspected each unique non-empty neighbor photo to mark 29 visible animals (leaving 14 frames without a visible animal blank), and added the boxes to the tiles and zoom view.
- **Decision:** accepted
- **Why:** Made the tiny night-time rabbits easy to spot right away, and left both bars in Panel 3 going the same way (higher = better).
- **Problems caused:** none.
- **Time spent verifying:** 50 minutes

---

### 2026-11-21: Adding Color Legends and Dropping the Extra Button

- **Who:** Martin
- **Goal:** Label what blue and orange mean directly above Panel 2's heatmap (and drop the unused `Layer 49 ReLU` button), and make the dark-orange bar under Panel 1 easy to read.
- **Prompt (important ones):**
  ```
  please actually denote what blue and orange indicates here on top with a small legend; also remove the layer 49 ReLU button
  ```
  *(With screenshot of Panel 2, followed by screenshot of Panel 1's bottom bar:)*
  ```
  also, it is currently not clear what the dark orange bar means... - please fix!
  ```
- **AI output summary:** Replaced Panel 2's toggle with a small one-line legend above the photo showing orange (`pushes rabbit` / `pushes paca`) vs. blue (`pushes empty`), and added matching color squares next to every label under Panel 1's stacked bar.
- **Decision:** accepted
- **Why:** Fixed the last two small clarity issues I noticed on a final look.
- **Problems caused:** Without color squares under the stacked bar, the gold and dark-orange text shades looked almost the same; sending quick screenshots got both fixed right away.
- **Time spent verifying:** 35 minutes

---
