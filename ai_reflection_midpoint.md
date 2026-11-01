# Phase I Midpoint Reflection: Our Experience with GenAI

**Pair A:**
- **Lisa Krimbacher** — Student ID: `k12209298`
- **Lara Pötzl** — Student ID: `k12216038`

---

## 1. What We Built & How We Collaborated with GenAI

For Phase I, our pair was assigned to use Generative AI to investigate the black-box error behavior of a fine-tuned ResNet-50 classifier on the iWildCam camera-trap dataset. Our goal was to answer: *How does the model behave, which mistakes does it make, and under what conditions?*

We built two interactive visualization tools:
1. **Species Confusion Network & Matrix Explorer:** An interactive Dash dashboard analyzing fine-grained species confusions, identifying dominant "error sinks", and visualizing how misclassifications shift between in-distribution (seen cameras) and out-of-distribution (unseen cameras) splits.
2. **False Detection Gallery:** An interactive tool to inspect "hallucinations"—cases where the model predicts an animal in images labeled as empty.

### GenAI as Tool vs. GenAI as Collaborator
Throughout Phase I, our interaction with GenAI moved between two modes:
- **As a Tool (Human asks, GenAI implements):** For concrete programming tasks—such as generating Dash boilerplate, deriving Bézier curve control points, writing CSS flexbox layouts, and setting up command-line arguments.
- **As a Collaborator (Iterative co-design):** For exploring analytical directions, brainstorming visualization ideas, critiquing our interface designs, and sanity-checking whether our findings made sense.

---

## 2. Pushing for Deeper Interactivity & Broader Analysis

Because GenAI took over a lot of the initial coding boilerplate, we used that saved time to explore the data more broadly and build much more interactive views than a standard static analysis:

### 2.1 How We Broadened the Analysis
Instead of settling for a single standard error chart or a static confusion matrix, we investigated **two completely orthogonal dimensions of model failure**:
1. **Species Confusion Dynamics under Domain Shift:** Analyzing which species the model confuses, which cameras act as local error hotspots, and how error patterns change when moving from seen camera locations (`id_val`/`id_test`) to completely unseen camera locations (`val`/`test`).
2. **Hallucination Auditing on Empty Frames:** Since empty background frames make up ~40% of the dataset, we investigated false positives in isolation, inspecting how illumination, vegetation density, and camera hardware trigger spurious animal detections.

### 2.2 How We Implemented Richer Interaction
Having AI handle boilerplate allowed us to implement significantly deeper interactivity than a typical static report:
- **Coordinated Multi-View Brushing & Linking:** Clicking a species node in the network graph instantly synchronizes the entire dashboard: focusing the Confusion Matrix, filtering incoming/outgoing mistake bar charts, highlighting camera location distributions, and loading representative image pairs.
- **Dynamic Client-Side Collision Avoidance:** We integrated a custom client-side JavaScript engine (`confusion_labels.js`) that measures rendered DOM elements and dynamically moves text labels into open space around nodes on every pan and zoom.
- **Live Multi-Metric Filtering & Re-Sorting:** Interactive sliders for ground-truth support, raw error count, and error rate percentage, paired with a toggle to re-sort camera charts between absolute count and percentage rate on the fly.
- **Side-by-Side Photo Comparisons & Slideshows:** Embedded image carousels allowing users to compare true species photos directly against misclassified prediction images for each specific camera trap.

---

## 3. How AI Suggestions Affected Subsequent Analytical Decisions

Working with GenAI fundamentally steered several of our key analytical milestones:

1. **Initial Direction Selection:** When we kicked off the project, we asked the AI to brainstorm 10 potential error-analysis angles for iWildCam. It proposed directed confusion networks and false-detection audits. This directly steered our decision to split our work into the Network Explorer and the False Detection Gallery.
2. **Early Course Correction on Seasonal Analysis:** When exploring temporal patterns, the AI initially suggested splitting photos by calendar seasons. However, questioning the AI about dataset details revealed that iWildCam comprises 414 cameras worldwide across varied equatorial and hemispheric biomes. This critical interaction steered us to drop calendar seasons completely and focus on camera-level environmental shifts instead.
3. **Pivoting from Flat Circles to Clustered Tiers:** The AI's initial network layout placed all species along one massive circular perimeter. When we observed that this created severe overcrowding and collisions, we steered the AI to compute connected components, placing primary clusters in an inner orbit and satellite pairs on an outer ring.
4. **Rejecting Speculative "Illusion Buckets":** During our hallucination analysis, the AI suggested categorizing false alarms into four heuristic "illusion types" (sun glare, foliage, reflections, pareidolia). After implementing a prototype, our inspection revealed that the classification was arbitrary guesswork. This steered us to reject the heuristic and maintain strict empirical rigor based only on verifiable metadata.

---

## 4. Origin of Ideas: Us vs. GenAI (Accepted, Modified, Rejected)

- **Ideas suggested by GenAI:**
  - Initial brainstorming of the directed confusion network and false-positive audit concepts.
  - Using quadratic Bézier curves to route directed arrows around intervening nodes.
  - Using LRU caching and normalization functions for fast offline species taxonomy lookup.
- **Ideas that came from us (Human):**
  - All core analytical questions (e.g. investigating camera-specific hotspots, comparing ID vs. OOD splits).
  - The focus-and-context side panel architecture and coordinated view updates.
  - Consolidating ID and OOD camera splits into unified dataset views for a cleaner overview.
  - Insisting on bilingual species names (English common names + scientific Latin in subtitles).
  - Moving label collision avoidance from server-side approximations into client-side DOM measurements.
- **Suggestions refined or rejected:**
  - *Refined:* Network layout algorithms (iteratively improved from a single flat ring to orbital tiers).
  - *Rejected:* Calendar-season splitting (ecologically unsound) and heuristic illusion buckets (unfalsifiable guesswork).

---

## 5. GenAI to Transform Data & Derived Data Reliability

We used GenAI to enrich the dataset’s raw labels, but critically audited its reliability:

- **Bilingual Species Names:** We had the AI map numeric IDs and Latin binomials to standardized English common names (using WiSE-FT).
- **Evaluating Derived Data:** We audited small samples and caught instances where the AI generated over-specific common names for generic genus labels (e.g. mapping an unspecified *Didelphis sp.* to a specific Virginia opossum). We manually corrected these to keep generic classifications honest.
- **Did the value justify the verification effort?**
  - **Yes** for the taxonomy normalization: having clear English names made the network graph and dropdowns immediately intuitive to explore, and verifying 182 names took only ~20 minutes.
  - **No** for heuristic attribute classification: verifying the AI's "illusion types" took extensive time and proved the labels had no scientific ground truth, leading to their removal.

---

## 6. GenAI as a Critic & Its Impact on Our Confidence

- **Constructive Critiques:** The AI helped identify visual clutter in camera bar charts (leading to the Count vs. Rate toggle) and pointed out contrast inconsistencies on directed arrowheads.
- **Misleading Critiques & Confidence Impact:** When asked to interpret *why* specific animals were confused, the AI often hallucinated plausible-sounding morphological reasons (like antler shapes) that were invisible in blurry camera trap images. This made us **more skeptical and cautious overall**: we learned never to rely on an AI explanation without verifying the raw parquet data and photo evidence ourselves.

---

## 7. Comparison with the Human-Only Pair (Pair B)

- **Breadth & Implementation Velocity:** Because GenAI handled the boilerplate code, we were able to build two full interactive Dash applications with rich multi-view linking. The human-only pair, having to write all visualization and callback code by hand, naturally focused on fewer views.
- **Cognitive Auditing Trade-off:** While Pair B maintained full mental ownership of every line of code, our main cognitive burden was constant auditing. The AI introduced subtle, silent bugs—such as plotting raw error counts instead of rates in the confusion matrix heatmap, or leaking split names across hot-reloads. We spent significant time verifying numbers to ensure our conclusions were sound.

---

## 8. Failure Patterns Observed During AI-Assisted Analysis

1. **Total Visual Blindness:** The model generates code from text tokens, completely unaware of visual density, font rendering, or text collisions. Static pixel offsets repeatedly failed across screen sizes, requiring our client-side DOM solution (`confusion_labels.js`).
2. **Poor UI Ergonomics:** The AI regularly placed controls in awkward, disconnected locations (like placing the "Reset Focus" button in the global header). Spatial proximity must be explicitly dictated in prompts.
3. **Plausible Hallucinations:** The AI readily invents convincing explanations and heuristic groupings that sound smart on paper but fail under real data inspection.

---

## 9. Emerging Best Practices for Using AI Effectively

1. **Enforce UI layout and DOM hierarchy explicitly in prompts.**
2. **Iterate in small, single-responsibility steps** to prevent regressions in complex callbacks.
3. **Decouple data computation (Python backend) from visual collision avoidance (client-side DOM measurement).**
4. **Never accept AI-generated interpretations of model behavior without validating them against the raw data.**
