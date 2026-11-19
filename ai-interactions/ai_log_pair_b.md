# AI Interaction Log - Phase 2 (Pair B)

This log tracks our AI interactions during Phase 2 (What did the model learn?). It documents:
- The important prompts we used
- How AI suggestions influenced our decisions
- Which suggestions were accepted, adjusted, or rejected, and why
- Failure patterns we observed
- Best practices we learned for using AI effectively

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns We Observed

- **Precise numbers that were wrong:**
  In the concepts run, the AI claimed that two thirds of the "night" frames were taken in daylight, with a median brightness of 0.42. We could not reproduce either number: 42 % are daytime and the median brightness is 0.076. We had already repeated the claim before checking it, and only the next agent caught it.
- **Copying the paper's method would have misled us:**
  With the paper's IoU rule, network dissection labelled 2 of 2,048 units, which looks like "the model has no scenery units". In fact, big concepts like trees (47 % of the pixels) can't reach the IoU cut at all. The AI pointed this out itself this time, but if we had used the method without checking its assumptions on our data, we would have drawn the wrong conclusion.
- **Results that only work on one machine:**
  Two of the chapter figures first read cached arrays from `outputs/`, which isn't in git, so a clean checkout couldn't rebuild the page. Locally everything worked, which is why it was easy to miss.
- **Automated screenshots can mislead:**
  In screenshots that jumped straight to a step, the hero image seemed to switch at the wrong time. When we scrolled through normally, it switched correctly.

---

### 2. Best Practices We Learned

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

---

## Log of AI Interactions

### 2026-11-11: Network dissection pipeline

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
- **Decision:** adjusted
- **Why:** The paper's IoU rule labelled almost no scenery units. We kept that as a negative result and switched to total-activation profiles in the next step.
- **Problems caused:** none found. We checked the IoU bound and the in-box shares against the saved files, and the contact sheets by eye.
- **Time spent verifying:** 30 minutes
---

### 2026-11-11: Splitting each prediction into animal and background evidence

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

### 2026-11-11: Skeleton of the story page

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

### 2026-11-11: Which scenery the votes come from (concept profiles, vote split, ablation)

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
- **Decision:** partially accepted
- **Why:** The unit labels, vote split and ablation held up when we recomputed them. We dropped the greyscale daylight claim because it was wrong.
- **Problems caused:** A usage limit interrupted the run once. Wrong insight: we couldn't reproduce the "two thirds daylight" claim (median brightness 0.42); the real values are 42 % daytime and a median brightness of 0.076. We repeated it before checking, and the next agent caught it. All other numbers matched our own recomputation and ablation rerun.
- **Time spent verifying:** 45 minutes
---

### 2026-11-11: Cutting the story down to a 2-minute talk

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

### 2026-11-19: Filling the story chapters with the results

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

### 2026-11-19: Interactive controls for the story chapters

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
