# AI Interaction Log - Phase 1

This log tracks all AI interactions and reflections during Phase 1: Error Analysis.

---

## Reflection: Failure Patterns & Emerging Best Practices

### 1. Failure Patterns Observed During AI-Assisted Analysis

- **Lack of Inherent Visual Aesthetics and Ergonomic Layout:**
  The LLM generated code that (almost alwalys) worked from a technical standpoint, but the resulting interface was not always visually pleasing or intuitive out of the box. The visualizations looked fine, but not perfect.
- **Poor Spatial Proximity for Controls:**
  The model frequently placed interactive controls far away from the actual components they manipulate. For instance, it initially placed the "Reset Focus" button up in the top global filter bar rather than directly next to the network graph and confusion matrix header. Similarly, threshold readouts were placed in separate rows rather than directly next to the slider tracks.
- **Blindness to Visual Density and Overlap:**
  The AI has no visual feedback loop, so it cannot "see" when rendered elements collide. In early confusion network iterations, species text labels severely overlapped, opaque white text boxes chopped into adjacent nodes and edges, and dense confusion arrows criss-crossed into an unreadable knot. The AI could not resolve these issues until explicitly instructed and given a screenshot.
- **Silent Logical Mismatches in Views:**
  The model initially plotted raw integer misclassification counts inside the Confusion Matrix even when the user had selected "Confusion Rate" ranking, creating inverted and misleading visual heatmaps. Such inconcistencies had to be manually found.

---

### 2. Emerging Best Practices for Using AI in Visualization & Model Analysis

- **Enforce UI Proximity Explicitly in Prompts:**
  Do not let the AI decide where to place buttons and filters. Always specify spatial grouping explicitly (e.g., *"place the Reset Focus button in the top-right corner of the visualization card next to the title"*).
- **Maintain Tight, Incremental Feedback Loops:**
  Large prompts that request multiple layout and styling changes simultaneously can result in broken CSS, event listener conflicts, or unintended regressions (such as the stuck hover tooltip timer). Break visual refinement into small, isolated iterations (e.g., fixing aspect ratio, eliminating scrollbars, tuning font weights).
- **Disambiguate Mathematical Notation in UI Text:**
  When asking the AI to document formulas, ensure it separates functional argument notation from explanatory aliases. Leaving the AI to format formulas on its own often results in ambiguous expressions (like placing definitions directly inside a formula, thereby creating an ambiguous meaning for parentheses).

---

## Prompts & Interactions

### 2026-10-19: Brainstorming Error Analysis Concepts

- **Who:** Lisa (+ Lara)
- **Goal:** Explore visualization strategies for diagnosing fine-tuned ResNet-50 failure modes on iWildCam.
- **Prompt (important ones):**
  ```
  I'm part of the AI-aided pair for Phase I. We chose to use the iWildCam dataset (see README). Please brainstorm in detail what kind of patterns we could investigate. Note that the goal is to generate good visualizations, as this is the focus of this course. We'd love to do something interactive or very involved, as we want the best possible grade.

  [README.md]
  [2026 AI & VIS Lab 02 Assignment Introduction.pdf]
  [Moodle Description of Phase I]
  ```
- **AI output summary:** Suggested 10 ideas for interactive visualizations, highlighting class-level accuracy breakdowns, species confusion networks, confidence calibration, camera trap location error patterns, and temporal confusion shifts.
- **Decision:** partially accepted
- **Why:** Chose the directed species confusion network as our primary focus because it directly visualizes fine-grained misclassifications under domain shift, while keeping camera location breakdowns as a secondary dimension.
- **Problems caused:** none
- **Time spent verifying:** 105 minutes
---

### 2026-10-19: Selecting Dashboard Tech Stack

- **Who:** Lisa
- **Goal:** Choose an interactive Python dashboard framework suitable for graph manipulation and linked views.
- **Prompt (important ones):**
  ```
  Which Python dashboard framework is best suited for building this? I want an interactive web app where clicking nodes in a network graph updates side panels and filters other charts.
  ```
- **AI output summary:** Evaluated Dash, Streamlit, and Panel. Recommended Dash + Plotly because Dash's callback graph natively supports bidirectional component interactions and clicking scatter nodes to update auxiliary panels.
- **Decision:** accepted
- **Why:** Dash provides flexible reactive state management required for coordinating linked views between the main network and side panels without reloading the entire page.
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-19: First Draft of Species Confusion Network

- **Who:** Lisa
- **Goal:** Implement the initial working prototype of the species confusion network dashboard.
- **Prompt (important ones):**
  ```
  I want to create a Species Confusion Network in Dash. Nodes should be species and directed edges should indicate that species A is predicted as species B.

  The dashboard should include:
  - Dataset split selector (val, id_val, test, id_test)
  - Threshold sliders for class support, edge error count, and confusion rate
  - Visual encodings: node color = error rate, node size = support, edge width = error count
  - Click-to-filter on nodes with a side panel showing top confused pairs
  - Raw data table at the bottom
  ```
- **AI output summary:** Generated initial `species_confusion_network.py` using Dash and Plotly with circular NetworkX layout, threshold sliders, click callbacks, and a tabular confusion data view.
- **Decision:** accepted
- **Why:** Produced a functional interactive prototype that ran out of the box.
- **Problems caused:** Plotly scatter traces did not natively render directional edge arrows; raw data table was clunky and took up too much vertical space.
- **Time spent verifying:** 30 minutes
---

### 2026-10-19: Layout Restructuring and Visual Hierarchy

- **Who:** Lisa
- **Goal:** Establish clear visual hierarchy, remove tabular clutter, and add high-level KPI cards.
- **Prompt (important ones):**
  ```
  The dashboard looks cluttered and the raw data table at the bottom takes up too much room without giving quick insights. Can we remove the table and reorganize the layout into top KPI summary cards, filter controls, a community-clustered network, and a side-by-side error breakdown panel?
  ```
- **AI output summary:** Restructured layout into top KPI summary cards, a clean horizontal filter bar, community-clustered network view, and a dedicated sidebar for inspecting confused pairs.
- **Decision:** accepted
- **Why:** Eliminates unnecessary raw data clutter and immediately directs attention to key error metrics and high-volume confusion clusters.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-19: Representative Species Photos and Review Notebook

- **Who:** Lisa
- **Goal:** Add representative camera-trap photos to hover cards and build an interactive widget to verify image quality across classes.
- **Prompt (important ones):**
  ```
  Can you add representative pictures for each animal species to hover tooltips? Also, can you build a Jupyter notebook where I can easily review sample images for all classes and approve or select better images if any are blurry or occluded?
  ```
- **AI output summary:** Added image hover tooltips to the dashboard and built `review_representative_images.ipynb` with an interactive IPython widget that cycles through candidates and writes approved paths to `representative_images.json`.
- **Decision:** accepted
- **Why:** Having authentic camera trap photos makes biological misclassifications immediately interpretable.
- **Problems caused:** Tooltip initially jumped to the top-left of the browser window; resolved by setting relative positioning on the parent container.
- **Time spent verifying:** 45 minutes
---

### 2026-10-19: Scientific Terminology and Metric Formula Clarification

- **Who:** Lisa
- **Goal:** Enforce scientific species taxonomy, fix layout to community clustering, and provide explicit mathematical definitions.
- **Prompt (important ones):**
  ```
  Please standardize the dashboard on scientific species names only, and remove informal notes or disclaimers. Also, could you display the mathematical formulas for error rate and confusion rate across multiple lines so they are clear and easy to read?
  ```
- **AI output summary:** Standardized taxonomy across data loaders and UI components, locked community clustering as the layout algorithm, and added a multi-line formula legend.
- **Decision:** accepted
- **Why:** Ensures academic rigor, consistency with biological literature, and clear documentation of evaluation metrics.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-19: Sidebar Previews, Tooltip Sizing, and Dynamic Slider Badges

- **Who:** Lisa
- **Goal:** Add hover previews to sidebar ranking charts, prevent tooltip overflow, and improve slider readability.
- **Prompt (important ones):**
  ```
  In the sidebar, can you add image hovers to the bar charts so we can see what these confused animals look like? Also, the hover text sometimes overflows the rectangle, and on the threshold sliders the 1% mark is hard to read—can you add badges showing the current slider values?
  ```
- **AI output summary:** Added photo hover tooltips to sidebar charts, added dynamic blue value badges above sliders, and applied CSS box-sizing and wrapping to prevent tooltip text overflow.
- **Decision:** accepted
- **Why:** Allows users to identify species directly from ranking lists without having to find them on the network first, and makes slider thresholds easy to read.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-19: Harmonizing Colorbar and Palette Styling

- **Who:** Lisa
- **Goal:** Align colorbar styling and palettes across the network graph and confusion matrix views.
- **Prompt (important ones):**
  ```
  The error bars for the directed network and the confusion matrix look completely different. Can you style both in the modern, borderless look of the confusion matrix using the Reds palette?
  ```
- **AI output summary:** Unified colorscale to `Reds` across both views, positioned colorbar titles horizontally at the top, and removed bounding borders.
- **Decision:** accepted
- **Why:** Delivers a consistent visual identity across visualization modes.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-19: Professional Polish and Academic Styling

- **Who:** Lisa
- **Goal:** Replace misleading statistical terminology, remove informal emojis, and refine visual styling.
- **Prompt (important ones):**
  ```
  "Distribution" is a bit misleading here since only the top cameras are listed. Can we change it to "Top Camera Locations by Error Count"? Also, please remove all emojis and make the UI look clean and professional for a scientific report.
  ```
- **AI output summary:** Replaced "distribution" with accurate wording, removed all emojis across UI elements, refined button and card styling, and added subtle gridlines to charts.
- **Decision:** accepted
- **Why:** Produces a publication-ready academic appearance suitable for lab submission.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Macro F1 Benchmark Metric Integration

- **Who:** Lisa
- **Goal:** Add the official WILDS benchmark Macro F1 score alongside split accuracy in the top KPI banner.
- **Prompt (important ones):**
  ```
  Can you please add Macro F1 to the accuracy KPI card? In the WILDS benchmark for iWildCam, Macro F1 is the official evaluation metric because class imbalance is so extreme.
  ```
- **AI output summary:** Integrated scikit-learn's `f1_score` (macro-averaged over present split classes) into KPI precomputations and displayed both accuracy and Macro F1 side-by-side in the primary KPI card.
- **Decision:** accepted
- **Why:** Directly aligns dashboard reporting with official WILDS evaluation benchmark standards.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Dataset Split Navigation and Defaulting to OOD Test Set

- **Who:** Lisa
- **Goal:** Reposition the dataset split dropdown to the top-left, order test sets first, and default to the OOD test set.
- **Prompt (important ones):**
  ```
  Can we reorganize the dataset split selector?
  - Move the split dropdown up into the top row on the far left.
  - Reorder the options so test sets appear above validation sets, and default to the OOD test set on startup.
  - Clean up the split labels to be descriptive (like "OOD test (unseen cameras)") without technical abbreviations or nested parentheses.
  ```
- **AI output summary:** Moved split selector to top left, reordered options with OOD test as initial default, and cleaned up label wording.
- **Decision:** accepted
- **Why:** Places the primary experimental control in the natural top-left reading position and focuses analysis immediately on the unseen test distribution.
- **Problems caused:** none
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Total Split Images KPI Card

- **Who:** Lisa
- **Goal:** Add a dedicated summary card showing total images, camera count, and present species count.
- **Prompt (important ones):**
  ```
  Could you add a summary card showing the total number of images in the selected split? It would also be great to show how many cameras and species classes are present in that split, and remove the redundant error count subtitle from the accuracy card.
  ```
- **AI output summary:** Added a Total Split Images KPI card displaying formatted total images, camera count, and present species count, and removed redundant ratio subtitles from the accuracy card.
- **Decision:** accepted
- **Why:** Provides immediate high-level context on split size and class representation while decluttering the accuracy card.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Visual Hover Comparison for Confusion Matrix and Directed Edges

- **Who:** Lisa
- **Goal:** Enable side-by-side true vs predicted photo comparisons when hovering over network edges and confusion matrix cells.
- **Prompt (important ones):**
  ```
  When hovering over a directed edge in the network or a cell in the confusion matrix, can you display a popup showing side-by-side photos of the true species and the mistaken species, along with the misclassification count and error rate?
  ```
- **AI output summary:** Wired 2D `customdata` into matrix heatmap cells and edge scatter traces, connecting them to hover callbacks that render side-by-side photos of the true and predicted animals.
- **Decision:** accepted
- **Why:** Comparing the physical appearance of the two confused animals side-by-side immediately reveals morphological similarities that caused the model error.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Dynamic Threshold Sliders and Confusion Matrix Metric Guard

- **Who:** Lisa
- **Goal:** Toggle sliders based on ranking mode and enforce count-based sorting in the confusion matrix.
- **Prompt (important ones):**
  ```
  I want to clean up the slider controls and confusion matrix view:
  - Only show the "Min Confusion Count" slider when sorting by count, and "Min Confusion Rate" when sorting by rate.
  - When switching to the Confusion Matrix view, automatically lock the metric to count and disable rate, because rate-based sorting produces misleading matrix views.
  ```
- **AI output summary:** Added dynamic visibility callbacks for threshold sliders and enforced count-only mode when Confusion Matrix view is selected.
- **Decision:** accepted
- **Why:** Prevents confusing slider states and ensures matrix heatmaps are always sorted by valid integer counts.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Editable Direct-Input Slider Badges and Focus Reset Repositioning

- **Who:** Lisa
- **Goal:** Replace static slider badges with editable input fields and reposition reset controls.
- **Prompt (important ones):**
  ```
  A few usability improvements for controls:
  - Can the blue slider badges be editable text fields so I can type in an exact threshold instead of dragging?
  - Remove the default white slider input boxes.
  - Move the "Reset Focus" button next to the main visualization title, and add a global "Reset Filters" button in the control panel.
  ```
- **AI output summary:** Replaced slider badges with debounced numeric input fields, removed default slider boxes, moved Reset Focus next to the plot title, and added a global Reset Filters button.
- **Decision:** accepted
- **Why:** Allows precise threshold entry and places reset controls right where the user is looking.
- **Problems caused:** Direct typing initially updated on each keystroke; resolved by adding debouncing to numeric inputs.
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Filtering Focus Species Dropdown to Active Classes

- **Who:** Lisa
- **Goal:** Filter the focus dropdown to species present in the split and permanently include the empty class.
- **Prompt (important ones):**
  ```
  In the focus species dropdown, can you remove all species that have no images or confusions in the current split? Also, please remove the "Empty Species" toggle checkbox and always include the empty class in the graph and matrix.
  ```
- **AI output summary:** Filtered dropdown options to active species in the split, removed the empty toggle, and permanently included the empty class in calculations.
- **Decision:** accepted
- **Why:** Removes inactive ghost options from the dropdown, and ensures empty-frame false alarms (a major failure mode in camera traps) are always visible.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Fixing Startup ValueError on OOD Test Split

- **Who:** Lisa
- **Goal:** Fix startup `ValueError` when initializing or switching the active split to OOD test.
- **Prompt (important ones):**
  ```
  When switching the default split to test (OOD test set), the app crashes on startup with:
  ValueError: Length of values (98) does not match length of index (102) at stats["top_err_cam_supp"] in summarize_split.
  Can you fix this?
  ```
- **AI output summary:** Identified that only 98 classes incur errors in the test split out of 102 species. Aligned camera aggregations to `stats.index` and added zero/NA defaults for classes with zero mistakes.
- **Decision:** accepted
- **Why:** Critical bug fix ensuring crash-free initialization across all dataset splits.
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-22: Collision-Free Graph Layout and Outward Label Placement

- **Who:** Lisa
- **Goal:** Resolve overlapping node text labels and improve label styling in the network graph.
- **Prompt (important ones):**
  ```
  In the community network view, several node labels are overlapping. Can we remove the opaque white background boxes behind the text, make the font slightly larger and semi-bold, and position labels outward from cluster centers so they don't collide?
  ```
- **AI output summary:** Removed white label backgrounds, increased font size to 11px semi-bold, and implemented outward radial label positioning to prevent collisions.
- **Decision:** accepted
- **Why:** Greatly improves graph readability and restores clean transparent text rendering.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Starburst Hub Layout for Focused Species Inspection

- **Who:** Lisa
- **Goal:** Implement a starburst hub layout for focused species mode to eliminate arrow crossings.
- **Prompt (important ones):**
  ```
  When a species is selected, the arrows get tangled and criss-cross each other. Can we use a starburst layout where the selected species is placed in the center, outgoing errors orbit on the right, and incoming false predictions orbit on the left?
  ```
- **AI output summary:** Built a dual-orbit starburst layout for focused species mode, separating outgoing mistakes and incoming false predictions onto opposite sides.
- **Decision:** accepted
- **Why:** Completely eliminates arrow crossings and makes directional error flow immediately clear at a glance.
- **Problems caused:** none
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Dynamic Arrowhead Scaling and Legibility Balancing

- **Who:** Lisa
- **Goal:** Scale network arrowheads proportionally with line thickness and improve color contrast.
- **Prompt (important ones):**
  ```
  A couple visual tweaks for the graph:
  - Can arrowheads scale with line thickness so thick lines have clearly visible direction arrows? Please make sure they don't turn into giant triangles.
  - The darkest red in the error rate colorscale is almost black, making dark text impossible to read. Can you lighten the top end of the colorscale?
  - Rename the visualization mode radio button from "Directed Network" to "Community Clusters".
  ```
- **AI output summary:** Implemented proportional arrowhead scaling, lightened the high-end red colorscale to crimson, and renamed the mode to Community Clusters.
- **Decision:** accepted
- **Why:** Maintains clear directional arrows across edge widths and ensures node labels remain readable over high-error nodes.
- **Problems caused:** Initial arrowhead scaling formula caused excessive triangular expansion; rebalanced with moderate linear scaling.
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Visual Encodings Legend and HTML Size Legend

- **Who:** Lisa
- **Goal:** Add an accurate node size legend and move visual encoding cards directly beneath the graph.
- **Prompt (important ones):**
  ```
  The visual encodings need clear documentation:
  - Add a visual circle size legend for Image Count (Support) below the error rate colorbar. Plotly's default legend clamps all markers to 19px, so please use a custom SVG/HTML legend that actually shows the different circle sizes.
  - Move the visual encoding cards (Node Size, Node Color, Arrow Width) directly inside the Confusion Network box under the plot.
  - In the formula cards, clarify that (A -> B) is functional notation for the pair, and don't put "(1 - Recall)" inside the formula where it looks like multiplication.
  ```
- **AI output summary:** Created a custom SVG node size legend, integrated visual encoding cards beneath the network, and cleaned up mathematical formula notations.
- **Decision:** accepted
- **Why:** Accurately communicates graphical mappings without relying on Plotly's clamped legend symbols.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Sidebar Ranking HTML Bar Charts and Photo Header

- **Who:** Lisa
- **Goal:** Replace sidebar Plotly bar charts with fast HTML bars and add a focused species photo header.
- **Prompt (important ones):**
  ```
  Can we refine the right-hand inspection sidebar?
  - Replace the Plotly horizontal bar charts with lightweight HTML/CSS bars so that hover tooltips trigger directly on the species text names rather than the bars.
  - Remove redundant x-axis tick numbers at the bottom of the bars since each bar already has its exact count at the end.
  - In the focus panel header, show a clean representative photo (240x180, 4:3 aspect ratio) alongside the species name and stats.
  - Fix section titles: use lowercase "mistaken for ..." and add symmetric ellipsis to "... mistaken as <species>".
  ```
- **AI output summary:** Created `make_html_bar_chart` with text-attached hover tooltips, removed tick numbers, added a 4:3 representative photo in the header, and standardized section titles.
- **Decision:** accepted
- **Why:** Significantly improves sidebar rendering responsiveness, prevents tooltip clipping, and gives instant visual confirmation of the focused animal.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Contextualizing Camera Confounding with Total Occurrences

- **Who:** Lisa
- **Goal:** Show species sightings and total camera captures alongside camera error counts.
- **Prompt (important ones):**
  ```
  When showing camera trap confusions, showing just the raw error count can be misleading because 10 errors out of 10 images is very different from 10 errors out of 1,000 images.
  Can you contextualize camera confusions by showing the total occurrences? For each camera mention, include:
  1. How many true sightings of that species were captured at that camera trap.
  2. How many total photos the camera took in the current split.
  ```
- **AI output summary:** Precomputed species-camera occurrence mappings and displayed true species sightings alongside total camera captures across all camera tooltips.
- **Decision:** accepted
- **Why:** Critical for distinguishing whether a camera trap has severe domain shift or simply high animal activity.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Interactive Multi-Image Slideshow (Diashow) Component

- **Who:** Lisa
- **Goal:** Build a client-side slideshow component allowing users to browse diverse sample photos.
- **Prompt (important ones):**
  ```
  For all image hovers, can we add a diashow carousel? Users should be able to click arrow buttons (‹ and ›) to browse through different candidate photos for a species or camera, with a counter badge like 1/5.
  Please keep the curated representative photo as the first image you see, and make sure navigation doesn't dismiss the popup.
  ```
- **AI output summary:** Created `make_diashow` with client-side event listeners and keyboard navigation, preserving curated representative photos at index 0.
- **Decision:** accepted
- **Why:** Allows users to inspect variations in animal posture, lighting, and camera angles directly within the interface.
- **Problems caused:** Button clicks initially caused popup dismissal due to event bubbling; resolved by stopping propagation on navigation buttons.
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Camera Misclassification Diashow and Dynamic Prediction Captions

- **Who:** Lisa
- **Goal:** Display actual misclassified photos in the sidebar and show what the model predicted.
- **Prompt (important ones):**
  ```
  Two improvements for the camera trap view:
  - Move the camera misclassification diashow into the persistent right sidebar and show the actual photos of that species misclassified at that camera trap.
  - Add dynamic caption text under each photo showing what the model predicted for that specific shot (e.g. "Model predicted: coyote"), updating synchronously as the user clicks through images.
  ```
- **AI output summary:** Moved camera diashows into the sidebar, pre-indexed misclassified images by camera, and added dynamic prediction caption text.
- **Decision:** accepted
- **Why:** Provides deep qualitative error inspection by showing the exact images that confused the model alongside the model's prediction.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Standardizing Images to Native 4:3 Camera Trap Aspect Ratio

- **Who:** Lisa
- **Goal:** Standardize all animal and camera images to native 4:3 aspect ratio without letterboxing.
- **Prompt (important ones):**
  ```
  Camera trap images have a native 4:3 aspect ratio, but some of the camera views are being cropped into narrow strips. Can we standardize all photo containers to the natural 4:3 ratio so nothing gets cropped or letterboxed?
  ```
- **AI output summary:** Standardized image containers and diashow viewports to 4:3 aspect ratio with responsive auto-height.
- **Decision:** accepted
- **Why:** Ensures authentic camera trap image presentation without awkward cropping or distortion.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Restoring Native Hover Tooltips and Eliminating Timer Conflicts

- **Who:** Lisa
- **Goal:** Fix stuck hover tooltips and restore smooth tooltip appearance and dismissal.
- **Prompt (important ones):**
  ```
  The hovering isn't working properly anymore—tooltips either get stuck on the screen or don't appear when hovering over nodes. Can you fix this?
  ```
- **AI output summary:** Identified obsolete mouseover timer scripts overriding Dash tooltip display state. Removed timer scripts and set pointer-events to none on the tooltip container.
- **Decision:** accepted
- **Why:** Restores instant, reliable hover tooltips that cleanly appear on hover and disappear on unhover.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Eliminating Popup Horizontal Scrollbar and Aspect Ratio Sizing

- **Who:** Lisa
- **Goal:** Remove horizontal scrollbar on hover popups and enforce strict overflow containment.
- **Prompt (important ones):**
  ```
  I don't like the shifting horizontal scrollbar at the bottom of the hover popup. If needed, fix the width of the popup and let the height be automatic to preserve the correct aspect ratio.
  ```
- **AI output summary:** Enforced `overflow-x: hidden !important` on popup containers, set responsive fixed width, and enabled automatic height to eliminate horizontal scrollbars.
- **Decision:** accepted
- **Why:** Eliminates annoying scrollbar jumping while preserving clean 4:3 image presentation.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Simplifying Top KPI and Node Hover Popups

- **Who:** Lisa
- **Goal:** Remove camera breakdown text from top KPI popups and node hover tooltips to avoid redundancy.
- **Prompt (important ones):**
  ```
  For the Top Confusion Pair and Top Over-Predicted Sink KPI popups, please remove the sentence about the cameras. Also remove the camera section from the network node hover tooltip, keeping the photo, species metadata, and error rate.
  ```
- **AI output summary:** Removed camera text lines from KPI popups and node tooltips, keeping top-level previews concise and leaving in-depth camera analysis to the dedicated sidebar.
- **Decision:** accepted
- **Why:** Prevents redundant camera metrics across multiple cards and keeps high-level hover cards focused on morphological comparisons.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Final Popup and Typographic Polish

- **Who:** Lisa
- **Goal:** Remove slideshow buttons from the top pair card, remove confidence percentages, and correct minor typographic details.
- **Prompt (important ones):**
  ```
  A few final polish items:
  - In the Top Confusion Pair KPI popup, remove the diashow slider controls and just show single static photos.
  - Remove the confidence percentage from the prediction captions in the camera diashow.
  - Remove the bullet point before the camera sighting count in the sink tooltip.
  - Add a whitespace after "for" in "mistaken for ...".
  ```
- **AI output summary:** Set max_images=1 for top pair popup, removed confidence percentages from diashow captions, removed bullet point prefixes, and corrected whitespace spacing.
- **Decision:** accepted
- **Why:** Eliminates button clutter from static KPI previews, streamlines captions, and ensures clean typographic alignment.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Dashboard Header Typography and Model Specification

- **Who:** Lisa
- **Goal:** Enlarge the dashboard title heading, improve typographic hierarchy, and specify model details.
- **Prompt (important ones):**
  ```
  The header looks a bit small and understated for a full-screen dashboard:
  - Can you enlarge the main heading to 35px, center it, and make it bold?
  - Update the subtitle to clearly specify that we are analyzing fine-tuned ResNet-50 on iWildCam.
  - Adjust margins and padding so the header commands the top of the screen nicely.
  ```
- **AI output summary:** Scaled dashboard title to 35px bold, centered header elements, updated subtitle description, and refined container padding.
- **Decision:** accepted
- **Why:** Establishes strong typographic hierarchy and immediately clarifies the model and task context for viewers.
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-24: Restoring Network Edge Arrowheads & Hub Spoke Clearance

- **Who:** Lisa
- **Goal:** Ensure all confusion connections show directed arrowheads without collapsing on short edges or colliding with central hub nodes.
- **Prompt (important ones):**
  ```
  A few issues with directed arrowheads in the confusion network:
  - Can you check why there are no arrowheads for some connections here?
  - On the central hub cluster, the arrows are overlapping and occluding the middle node.
  - Make sure all connections consistently display directed arrows with adequate clearance.
  ```
- **AI output summary:** Diagnosed that Plotly suppressed arrowheads on short connections because the vector length between index points fell below ~8–10 px. Enforced a minimum tangent vector length of 0.35 data units (~16 px) so every connection renders an arrowhead. Resolved hub node overlap by offsetting spoke midpoint boundaries (inward spokes stop at 50%, outward spokes start at 50%) and expanding radial clearance for dense hub components ($m \ge 6$).
- **Decision:** accepted
- **Why:** Restores complete directional readability across all confusion connections while keeping central hub nodes unobstructed.
- **Problems caused:** Subsequent opacity scaling created an unintended contrast discrepancy on heavy edges.
- **Time spent verifying:** 15 minutes
---

### 2026-10-24: Color Harmonization and Uniform Edge Opacity

- **Who:** Lisa
- **Goal:** Eliminate color discrepancies between edge strokes and arrowheads, and standardize edge opacity across mistake volumes.
- **Prompt (important ones):**
  ```
  The arrow styling and opacity need refinement:
  - The thick connection at the bottom has a different, much darker color than the rest of the arrows.
  - Can we remove dynamic opacity so all connections share the same consistent color?
  - Line thickness should communicate error volume, but the color and opacity should remain uniform across all edges.
  ```
- **AI output summary:** Identified that dynamic opacity scaling caused high-error edges to render at 0.90 opacity (an isolated dark slate), while low-error edges rendered at 0.40 opacity (faint silver-gray), and earlier stroke vs. arrowhead RGB channels were mismatched. Removed dynamic opacity in overview mode and unified all overview edges, ensuring error magnitude is expressed strictly through stroke width without color or darkness discrepancies.
- **Decision:** accepted
- **Why:** Delivers clean visual hierarchy across the entire network, preventing heavy confusion edges from appearing as mismatched dark lines.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---
