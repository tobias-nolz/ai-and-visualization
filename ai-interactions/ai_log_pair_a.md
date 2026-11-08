# AI Interaction Log - Phase 1 (Pair A)

This log tracks our AI interactions and reflections during Phase 1 (Error Analysis). It documents:
- The important prompts we used
- How AI suggestions influenced our decisions
- Which suggestions were accepted, adjusted, or rejected, and why
- Failure patterns we observed
- Best practices we learned for using AI effectively

*(For our full reflection, see [`ai_reflection_midpoint.md`](../ai_reflection_midpoint.md)).*

---

## Summary: Failure Patterns & Best Practices

### 1. Failure Patterns We Observed

- **No visual awareness (blind to overlap):**
  The AI writes code based on text, so it cannot "see" what the screen actually looks like. In early versions of the network, animal names overlapped badly, white text boxes cut into lines and circles, and arrows crossed over each other into a knot. The AI couldn't fix this on its own until we pointed it out with screenshots and clear layout instructions.
- **Awkward UI placement:**
  The AI often put buttons and sliders far away from the charts they belong to. For example, it put the "Reset Focus" button all the way up in the top page header instead of next to the confusion network. It also placed threshold readouts in separate rows rather than directly next to the sliders.
- **Silent logic bugs:**
  The AI sometimes wrote code that ran without throwing an error, but was logically wrong. For example, it plotted raw mistake counts in the confusion matrix heatmap even when we selected "Confusion Rate", which created misleading charts. We had to manually check the numbers to catch these mistakes.

---

### 2. Best Practices We Learned

- **Tell the AI exactly where to put UI elements:**
  Don't leave button placement to the AI. Always specify where elements belong (e.g. "place the Reset button in the top-right corner next to the graph title").
- **Work in small, step-by-step prompts:**
  Asking for too many styling or callback changes at once often resulted in broken CSS or subtle bugs (like hover popups getting stuck). Asking for one specific change at a time worked much better.
- **Double-check formulas and numbers:**
  When asking the AI to document math or display formulas, check the output carefully. It often produced ambiguous notation (like putting explanatory words inside parentheses where they looked like multiplication).

---

## Log of AI Interactions

### 2026-10-19: Brainstorming Error Analysis Concepts

- **Who:** Lisa (+ Lara)
- **Goal:** Brainstorm visualization formats for our two chosen focus areas: species confusions and empty-frame hallucinations.
- **Prompt (important ones):**
  ```
  I'm part of the AI-aided pair for Phase I. We chose to use the iWildCam dataset (see README). Please brainstorm in detail what kind of patterns we could investigate. Note that the goal is to generate good visualizations, as this is the focus of this course. We'd love to do something interactive or very involved, as we want the best possible grade. Our topics are confusions and hallucinations.

  [README.md]
  [2026 AI & VIS Lab 02 Assignment Introduction.pdf]
  [Moodle Description of Phase I]
  ```
- **AI output summary:** Gave 10 ideas for visualizations, including class accuracy breakdowns, a directed confusion network, confidence calibration, camera location errors, and seasonal shifts.
- **Decision:** partially accepted
- **Why:** We already knew we wanted to investigate species confusions and empty-frame hallucinations. From the AI's suggestions, we picked the directed confusion network concept for the species confusion task because showing mistakes as graph arrows is much clearer than a giant, crowded 182×182 matrix. Also, we focused on the empty-frame hallucinations in our second tool.
- **Problems caused:** none
- **Time spent verifying:** 105 minutes
---

### 2026-10-19: Selecting Dashboard Tech Stack

- **Who:** Lisa
- **Goal:** Pick a Python dashboard framework that can handle interactive network graphs and linked side panels.
- **Prompt (important ones):**
  ```
  Which Python dashboard framework is best suited for building this? I want an interactive web app where clicking nodes in a network graph updates side panels and filters other charts.
  ```
- **AI output summary:** Compared Dash, Streamlit, and Panel. Recommended Dash + Plotly because Dash makes it easy to click on graph nodes and update other panels without reloading the whole page.
- **Decision:** accepted
- **Why:** Dash's callback system fits well with what we wanted to build (clicking on species to update side charts).
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-19: First Draft of Species Confusion Network

- **Who:** Lisa
- **Goal:** Build a first working prototype of the species confusion dashboard in Dash.
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
- **AI output summary:** Generated a first version of `species_confusion_network.py` using Dash and Plotly with a circular layout, threshold sliders, click events, and a big data table at the bottom.
- **Decision:** accepted
- **Why:** Gave us a working prototype right away that we could run and test.
- **Problems caused:** Plotly scatter plots didn't draw arrowheads on lines by default, and the raw table at the bottom was clunky and took up way too much space.
- **Time spent verifying:** 30 minutes
---

### 2026-10-19: Layout Restructuring and Visual Hierarchy

- **Who:** Lisa
- **Goal:** Clean up the layout, delete the giant data table, and add summary stat cards at the top.
- **Prompt (important ones):**
  ```
  The dashboard looks cluttered and the raw data table at the bottom takes up too much room without giving quick insights. Can we remove the table and reorganize the layout into top summary cards, filter controls, a community-clustered network, and a side-by-side error breakdown panel?
  ```
- **AI output summary:** Reorganized the page with summary cards at the top, a horizontal filter bar, clustered network nodes, and an inspection sidebar on the right.
- **Decision:** accepted
- **Why:** Got rid of table clutter so you can immediately see key numbers and main confusion clusters.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-19: Representative Species Photos and Review Notebook

- **Who:** Lisa
- **Goal:** Add example animal photos to hover popups, and create a notebook to pick good, clear photos for each species.
- **Prompt (important ones):**
  ```
  Can you please add representative pictures for each animal species to hover tooltips? Also, can you build a Jupyter notebook where I can easily review sample images for all classes and approve or select better images if any are blurry or occluded?
  ```
- **AI output summary:** Added image tooltips to the dashboard and wrote `review_representative_images.ipynb` with a simple button widget to cycle through pictures and save the best ones into `representative_images.json`.
- **Decision:** accepted
- **Why:** Seeing the actual animal photos makes it much easier to understand why the model got confused.
- **Problems caused:** The popup initially jumped to the top-left corner of the browser window. Fixed it by setting relative positioning on the container.
- **Time spent verifying:** 45 minutes
---

### 2026-10-19: Scientific Terminology and Metric Formula Clarification

- **Who:** Lisa
- **Goal:** Stick to official scientific species names and write out metric formulas clearly over multiple lines.
- **Prompt (important ones):**
  ```
  Please standardize the dashboard on scientific species names only, and remove informal notes or disclaimers. Also, could you display the mathematical formulas for error rate and confusion rate across multiple lines so they are clear and easy to read?
  ```
- **AI output summary:** Used scientific names everywhere across the app, kept community clustering as the layout, and added a multi-line formula explanation.
- **Decision:** accepted
- **Why:** Keeps the names consistent with the official dataset and makes formulas easier to read.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-19: Sidebar Previews, Tooltip Sizing, and Dynamic Slider Badges

- **Who:** Lisa
- **Goal:** Add photo previews when hovering over sidebar bars, stop text from overflowing popups, and show current slider values clearly.
- **Prompt (important ones):**
  ```
  In the sidebar, can you add image hovers to the bar charts so we can see what these confused animals look like? Also, the hover text sometimes overflows the rectangle, and on the threshold sliders the 1% mark is hard to read—can you add badges showing the current slider values?
  ```
- **AI output summary:** Added hover photo popups to the sidebar charts, put blue value badges above the sliders, and fixed CSS text wrapping so popup text doesn't spill over.
- **Decision:** accepted
- **Why:** You can see photos directly from the ranking list without searching the network, and slider numbers are easy to read.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-19: Harmonizing Colorbar and Palette Styling

- **Who:** Lisa
- **Goal:** Make the colorbars and color schemes look the same across the network and confusion matrix views.
- **Prompt (important ones):**
  ```
  The error bars for the directed network and the confusion matrix look completely different. Can you style both in the modern, borderless look of the confusion matrix using the Reds palette?
  ```
- **AI output summary:** Switched both views to the `Reds` color palette, centered the titles at the top, and removed outer borders.
- **Decision:** accepted
- **Why:** Makes the two views look consistent and clean.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-19: Professional Polish and Academic Styling

- **Who:** Lisa
- **Goal:** Fix imprecise wording, remove emojis, and give the UI a clean, simple look.
- **Prompt (important ones):**
  ```
  "Distribution" is a bit misleading here since only the top cameras are listed. Can we change it to "Top Camera Locations by Error Count"? Also, please remove all emojis and make the UI look clean and professional for a scientific report.
  ```
- **AI output summary:** Replaced "distribution" with more accurate wording, removed all emojis, cleaned up button and card styles, and added subtle grid lines.
- **Decision:** accepted
- **Why:** Looks cleaner and more accurate for our lab submission.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Macro F1 Benchmark Metric Integration

- **Who:** Lisa
- **Goal:** Add Macro F1 to the accuracy card because iWildCam has huge class imbalance.
- **Prompt (important ones):**
  ```
  Can you please add Macro F1 to the accuracy card? 
  ```
- **AI output summary:** Calculated scikit-learn's Macro F1 score across present classes and displayed it right next to accuracy in the top card.
- **Decision:** accepted
- **Why:** On iWildCam, simple accuracy is misleading because empty images make up so much of the dataset. Macro F1 is the official benchmark metric.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Dataset Split Navigation and Defaulting to OOD Test Set

- **Who:** Lisa
- **Goal:** Move the split selector to the top-left, put test sets first, and default to the test set on startup.
- **Prompt (important ones):**
  ```
  Can we reorganize the dataset split selector?
  - Move the split dropdown up into the top row on the far left.
  - Reorder the options so test sets appear above validation sets, and default to the OOD test set on startup.
  - Clean up the split labels to be descriptive (like "OOD test (unseen cameras)") without technical abbreviations or nested parentheses.
  ```
- **AI output summary:** Moved the dropdown to the top left, made the test set default, and simplified the split labels.
- **Decision:** accepted
- **Why:** Puts the main control right where you look first, and shows the unseen camera test set right away.
- **Problems caused:** none
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Total Split Images KPI Card

- **Who:** Lisa
- **Goal:** Add a card showing how many total pictures, cameras, and animal classes are in the selected split.
- **Prompt (important ones):**
  ```
  Could you please add a summary card showing the total number of images in the selected split? It would also be great to show how many cameras and species classes are present in that split, and remove the redundant error count subtitle from the accuracy card.
  ```
- **AI output summary:** Added a card showing total image count, camera count, and present classes, and removed redundant text from the accuracy card.
- **Decision:** accepted
- **Why:** Gives quick context on how big the split is without cluttering other cards.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Visual Hover Comparison for Confusion Matrix and Directed Edges

- **Who:** Lisa
- **Goal:** Show photos of the true animal and the predicted animal side by side when hovering over lines or matrix cells.
- **Prompt (important ones):**
  ```
  When hovering over a directed edge in the network or a cell in the confusion matrix, can you display a popup showing side-by-side photos of the true species and the mistaken species, along with the misclassification count and error rate?
  ```
- **AI output summary:** Passed image paths into customdata and wired up hover tooltips to show both photos side by side with the mistake count.
- **Decision:** accepted
- **Why:** Seeing both animals side by side makes it easy to see why the model got confused (e.g. similar body shape or fur).
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Dynamic Threshold Sliders and Confusion Matrix Metric Guard

- **Who:** Lisa
- **Goal:** Only show relevant sliders (count vs. rate) and make sure the matrix view only sorts by count.
- **Prompt (important ones):**
  ```
  I want to clean up the slider controls and confusion matrix view:
  - Only show the "Min Confusion Count" slider when sorting by count, and "Min Confusion Rate" when sorting by rate.
  - When switching to the Confusion Matrix view, automatically lock the metric to count and disable rate, because rate-based sorting produces misleading matrix views.
  ```
- **AI output summary:** Hid the rate slider when sorting by count and vice versa, and locked sorting to count when looking at the confusion matrix.
- **Decision:** accepted
- **Why:** Avoids confusing slider states and keeps the matrix heatmap legible.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Editable Direct-Input Slider Badges and Focus Reset Repositioning

- **Who:** Lisa
- **Goal:** Let users type exact numbers into slider badges instead of just dragging, and move the Reset buttons to better spots.
- **Prompt (important ones):**
  ```
  A few usability improvements for controls:
  - Can the blue slider badges be editable text fields so I can type in an exact threshold instead of dragging?
  - Remove the default white slider input boxes.
  - Move the "Reset Focus" button next to the main visualization title, and add a global "Reset Filters" button in the control panel.
  ```
- **AI output summary:** Turned the blue slider badges into number inputs, moved "Reset Focus" next to the plot title, and added a "Reset Filters" button in the control panel.
- **Decision:** accepted
- **Why:** Typing exact threshold numbers is much easier than dragging a small slider, and the reset buttons are placed right where you need them.
- **Problems caused:** Typing each digit triggered an instant update. Fixed it by adding debouncing so it waits until you finish typing.
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Filtering Focus Species Dropdown to Active Classes

- **Who:** Lisa
- **Goal:** Remove species from the focus dropdown that aren't in the current split, and make sure empty frames are always included.
- **Prompt (important ones):**
  ```
  In the focus species dropdown, can you remove all species that have no images or confusions in the current split? Also, please remove the "Empty Species" toggle checkbox and always include the empty class in the graph and matrix.
  ```
- **AI output summary:** Filtered the dropdown to only active species in that split, removed the empty toggle, and kept the empty class always enabled.
- **Decision:** accepted
- **Why:** Removes useless empty options from the dropdown, and ensures empty frames (a huge source of errors) are always visible.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Fixing Startup ValueError on OOD Test Split

- **Who:** Lisa
- **Goal:** Fix the crash when starting up on the test split (`ValueError: Length of values (98) does not match length of index (102)`).
- **Prompt (important ones):**
  ```
  When switching the default split to test (OOD test set), the app crashes on startup with:
  ValueError: Length of values (98) does not match length of index (102) at stats["top_err_cam_supp"] in summarize_split.
  Can you please fix this?
  ```
- **AI output summary:** Found that 4 species had zero mistakes in the test set. Filled missing species with zeros so the data lengths matched.
- **Decision:** accepted
- **Why:** The app was crashing on launch, this fixed it.
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-22: Collision-Free Graph Layout and Outward Label Placement

- **Who:** Lisa
- **Goal:** Stop species names from overlapping in the network view and make labels easier to read.
- **Prompt (important ones):**
  ```
  In the community network view, several node labels are overlapping. Can we remove the opaque white background boxes behind the text, make the font slightly larger and semi-bold, and position labels outward from cluster centers so they don't collide?
  ```
- **AI output summary:** Removed white boxes behind text, bumped font size slightly, and pushed labels outward from cluster centers.
- **Decision:** accepted
- **Why:** Text looks much cleaner and names don't overlap as badly.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Starburst Hub Layout for Focused Species Inspection

- **Who:** Lisa
- **Goal:** Fix crossed-over arrow lines when focusing on a species.
- **Prompt (important ones):**
  ```
  When a species is selected, the arrows get tangled and criss-cross each other. Can we use a starburst layout where the selected species is placed in the center, outgoing errors orbit on the right, and incoming false predictions orbit on the left?
  ```
- **AI output summary:** Built a starburst hub layout that puts the selected animal in the center and connected species along the ring, with radial spokes that never cross.
- **Decision:** accepted
- **Why:** Completely untangles the arrows so you can see mistake directions right away.
- **Problems caused:** none
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Dynamic Arrowhead Scaling

- **Who:** Lisa
- **Goal:** Make arrowheads scale with line thickness and brighten dark colors so text stays readable.
- **Prompt (important ones):**
  ```
  A couple visual tweaks for the graph:
  - Can arrowheads scale with line thickness so thick lines have clearly visible direction arrows? Please make sure they don't turn into giant triangles.
  - The darkest red in the error rate colorscale is almost black, making dark text impossible to read. Can you lighten the top end of the colorscale?
  - Rename the visualization mode radio button from "Directed Network" to "Community Clusters".
  ```
- **AI output summary:** Scaled arrowheads proportionally, lightened the darkest red color, and renamed the mode to "Community Clusters".
- **Decision:** accepted
- **Why:** Direction arrows are clearly visible even on thick lines, and text stays readable on dark red nodes.
- **Problems caused:** Arrowheads initially got way too big, fixed by toning down the scaling formula.
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Visual Encodings Legend and HTML Size Legend

- **Who:** Lisa
- **Goal:** Add an accurate circle size legend for image counts, and move encoding cards directly under the graph.
- **Prompt (important ones):**
  ```
  The visual encodings need clear documentation:
  - Add a visual circle size legend for Image Count (Support) below the error rate colorbar.
  - Move the visual encoding cards (Node Size, Node Color, Arrow Width) directly inside the Confusion Network box under the plot.
  - In the formula cards, clarify that (A -> B) is functional notation for the pair, and don't put "(1 - Recall)" inside the formula where it looks like multiplication.
  ```
- **AI output summary:** Created a custom HTML/SVG circle size legend, placed the encoding cards below the network, and cleaned up the formula text.
- **Decision:** accepted
- **Why:** Plotly's default legend shrank all circles to the same size, the custom legend actually shows real relative sizes.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Sidebar Ranking HTML Bar Charts and Photo Header

- **Who:** Lisa
- **Goal:** Swap Plotly bar charts in the sidebar with simple HTML bars, and add a representative photo to the header.
- **Prompt (important ones):**
  ```
  Can we please refine the right-hand inspection sidebar?
  - Replace the Plotly horizontal bar charts with lightweight HTML/CSS bars so that hovers trigger directly on the species text names rather than the bars.
  - Remove redundant x-axis tick numbers at the bottom of the bars since each bar already has its exact count at the end.
  - In the focus panel header, show a clean representative photo alongside the species name and stats.
  - Fix section titles: use lowercase "mistaken for ..." and add symmetric ellipsis to "... mistaken as <species>".
  ```
- **AI output summary:** Wrote `make_html_bar_chart` with hover tooltips on text names, removed useless tick numbers, added a 4:3 photo in the header, and cleaned up section titles.
- **Decision:** accepted
- **Why:** The sidebar renders much faster, tooltips don't get cut off, and you can see the animal photo right at the top.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Contextualizing Camera Confounding with Total Occurrences

- **Who:** Lisa
- **Goal:** Show total animal sightings and total camera shots alongside error counts so high-activity cameras don't look worse than they are.
- **Prompt (important ones):**
  ```
  When showing camera trap confusions, showing just the raw error count can be misleading because 10 errors out of 10 images is very different from 10 errors out of 1,000 images.
  Can you contextualize camera confusions by showing the total occurrences? For each camera mention, include:
  1. How many true sightings of that species were captured at that camera trap.
  2. How many total photos the camera took in the current split.
  ```
- **AI output summary:** Precalculated animal sightings per camera and total camera pictures, showing both numbers in camera tooltips.
- **Decision:** accepted
- **Why:** 10 mistakes out of 10 pictures is a broken camera, but 10 mistakes out of 1,000 pictures is actually great. This shows the difference clearly.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-22: Interactive Multi-Image Slideshow (Diashow) Component

- **Who:** Lisa
- **Goal:** Let users click through sample pictures for an animal or camera using next/previous arrow buttons.
- **Prompt (important ones):**
  ```
  For all image hovers, can we add a diashow carousel? Users should be able to click arrow buttons to browse through different candidate photos for a species or camera, with a counter badge like 1/5.
  Please keep the curated representative photo as the first image you see.
  ```
- **AI output summary:** Built a slideshow carousel (`make_diashow`) with arrow buttons and a counter (e.g. 1/5), keeping the best photo first.
- **Decision:** accepted
- **Why:** Lets you flip through different photos to see different angles, lighting, and postures.
- **Problems caused:** Clicking arrows initially closed the popup, fixed by stopping the click event from bubbling up.
- **Time spent verifying:** 25 minutes
---

### 2026-10-22: Camera Misclassification Diashow and Dynamic Prediction Captions

- **Who:** Lisa
- **Goal:** Show the actual photos of misclassified animals at each camera trap, with captions showing what the model guessed.
- **Prompt (important ones):**
  ```
  Two improvements for the camera trap view:
  - Move the camera misclassification diashow into the right sidebar and show the actual photos of that species misclassified at that camera trap.
  - Add dynamic caption text under each photo showing what the model predicted for that specific shot (e.g. "Model predicted: coyote"), updating synchronously as the user clicks through images.
  ```
- **AI output summary:** Moved camera slideshows into the right sidebar and added dynamic captions showing what the model predicted for that specific photo.
- **Decision:** accepted
- **Why:** You can see the exact photo that fooled the model and what it thought it saw.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-10-22: Standardizing Images to Native 4:3 Camera Trap Aspect Ratio

- **Who:** Lisa
- **Goal:** Make all photos use the camera's natural 4:3 shape so nothing gets stretched or cropped into narrow strips.
- **Prompt (important ones):**
  ```
  Some of the camera views are being cropped into narrow strips. Can we standardize all photo containers ratios so nothing gets cropped or letterboxed?
  ```
- **AI output summary:** Set image containers and slideshows to 4:3 aspect ratio with auto height.
- **Decision:** accepted
- **Why:** Keeps camera trap images looking natural without weird stretching or cropping.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Restoring Native Hover Tooltips and Eliminating Timer Conflicts

- **Who:** Lisa
- **Goal:** Fix tooltips getting stuck on screen or not showing up when hovering.
- **Prompt (important ones):**
  ```
  The hovering isn't working properly anymore—tooltips either get stuck on the screen or don't appear when hovering over nodes. Can you fix this?
  ```
- **AI output summary:** Found old mouseover timer scripts that interfered with Dash. Removed the timers and set `pointer-events: none` on the tooltip box.
- **Decision:** accepted
- **Why:** Tooltips now appear and disappear smoothly when hovering over nodes.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Eliminating Popup Horizontal Scrollbar and Aspect Ratio Sizing

- **Who:** Lisa
- **Goal:** Get rid of the annoying horizontal scrollbar at the bottom of hover popups.
- **Prompt (important ones):**
  ```
  I don't like the shifting horizontal scrollbar at the bottom of the hover popup. If needed, fix the width of the popup and let the height be automatic to preserve the correct aspect ratio.
  ```
- **AI output summary:** Added `overflow-x: hidden`, set a fixed popup width, and let height adjust automatically.
- **Decision:** accepted
- **Why:** Stops scrollbars from jumping around and keeps the photos at a clean 4:3 aspect ratio.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Simplifying Top KPI and Node Hover Popups

- **Who:** Lisa
- **Goal:** Remove camera text from the top KPI cards and node hover popups so they stay short and focused.
- **Prompt (important ones):**
  ```
  For the Top Confusion Pair and Top Over-Predicted Sink popups, please remove the sentence about the cameras. Also remove the camera section from the network node hover tooltip, keeping the photo, species metadata, and error rate.
  ```
- **AI output summary:** Removed camera lines from KPI popups and node tooltips, keeping just the photo, species info, and error rate.
- **Decision:** accepted
- **Why:** Keeps hover cards simple and avoids repeating camera details that are already in the sidebar.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Final Popup and Typographic Polish

- **Who:** Lisa
- **Goal:** Remove slideshow buttons from the top pair popup, drop confidence percentages from captions, and fix spacing typos.
- **Prompt (important ones):**
  ```
  A few final polish items:
  - In the Top Confusion Pair KPI popup, remove the diashow slider controls and just show single static photos.
  - Remove the confidence percentage from the prediction captions in the camera diashow.
  - Remove the bullet point before the camera sighting count in the sink tooltip.
  - Add a whitespace after "for" in "mistaken for ...".
  ```
- **AI output summary:** Set top pair popup to single photos without slider buttons, removed confidence numbers from captions, and fixed minor whitespace and typo issues.
- **Decision:** accepted
- **Why:** Cleans up the preview cards and fixes small visual details.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes
---

### 2026-10-22: Dashboard Header Typography and Model Specification

- **Who:** Lisa
- **Goal:** Make the main dashboard title larger and bold, and state clearly that we are analyzing fine-tuned ResNet-50 on iWildCam.
- **Prompt (important ones):**
  ```
  The header looks a bit small and understated for a full-screen dashboard:
  - Can you enlarge the main heading, center it, and make it bold?
  - Update the subtitle to clearly specify that we are analyzing fine-tuned ResNet-50 on iWildCam.
  - Adjust margins and padding so the header commands the top of the screen nicely.
  ```
- **AI output summary:** Increased title to 35px bold, centered header items, and updated the subtitle text.
- **Decision:** accepted
- **Why:** The header was way too small before, now it clearly introduces the project at the top of the page.
- **Problems caused:** none
- **Time spent verifying:** 5 minutes
---

### 2026-10-24: Restoring Network Edge Arrowheads & Hub Spoke Clearance

- **Who:** Lisa
- **Goal:** Make sure all confusion lines have arrowheads, and stop arrows from drawing directly over the middle node in hub clusters.
- **Prompt (important ones):**
  ```
  A few issues with directed arrowheads in the confusion network:
  - Can you check why there are no arrowheads for some connections here?
  - On the central hub cluster, the arrows are overlapping and occluding the middle node.
  - Make sure all connections consistently display directed arrows?.
  ```
- **AI output summary:** Found that Plotly hid arrowheads on short lines under 10px, set a minimum line length so every line gets an arrow. Shifted spoke lines so they stop before overlapping the central node.
- **Decision:** accepted
- **Why:** You can clearly see mistake directions on every connection, and the center node isn't covered by lines.
- **Problems caused:** Thick lines ended up darker than thin lines, fixed in the next step.
- **Time spent verifying:** 15 minutes
---

### 2026-10-24: Color Harmonization and Uniform Edge Opacity

- **Who:** Lisa
- **Goal:** Make all arrows use the exact same color and opacity so heavy lines don't look completely different from light ones.
- **Prompt (important ones):**
  ```
  The arrow styling and opacity need refinement:
  - The thick connection at the bottom has a different, much darker color than the rest of the arrows.
  - Can we remove dynamic opacity so all connections share the same consistent color?
  - Line thickness should communicate error volume, but the color and opacity should remain the same across all edges.
  ```
- **AI output summary:** Removed dynamic opacity so all lines share the same gray tone, using only line thickness to show error count.
- **Decision:** accepted
- **Why:** Makes the graph look clean and consistent, thick lines show bigger error counts without turning into mismatched black bars.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-25: Increasing Inter-Cluster Spacing and In-Cluster Node Clearance

- **Who:** Lisa
- **Goal:** Spread out clusters and nodes in the network so species names don't overlap as much.
- **Prompt (important ones):**
  ```
  sometimes the species names are overlapping. how could this be fixed?
  - can we please first try it by just increasing cluster distnaces?
  - is it possible to have the in-cluster distances larger so that there is less overlap of the text with other nodes?
  ```
- **AI output summary:** Pushed clusters farther apart, made inner circles wider, and placed labels just outside node circle boundaries.
- **Decision:** accepted
- **Why:** Makes species names much easier to read without text drawing right over the circles.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-10-19: Checking if seasonal analysis actually makes sense

- **Who:** Lara
- **Goal:** Check if splitting camera trap photos by four calendar seasons is ecologically sound or just a bad assumption.
- **Prompt (important ones):**
  ```
  where was this dataset recorded, does seasons really make any sense for this use case for the dataset iWildCam?
  ```
- **AI output summary:** this dataset has 414 camera-trap locations distributed globally, therefore dividing the dataset into four calendar seasons is potentially misleading.
- **Decision:** accepted
- **Why:** moving forward with a visualization question according to seasonal changes does not really make sense when this is not generalizable good enough
- **Problems caused:** none
- **Time spent verifying:** 5

---

### 2026-10-19: Setting up the False Detection Gallery pipeline

- **Who:** Lara
- **Goal:** Drop the broken audit script we had before and start fresh with Idea 4 (auditing false detections on empty frames).
- **Prompt (important ones):**
  ```
  Idea 4: Does the model hallucinate animals in empty images?
  Visualization concept: False Detection Gallery
  Build a gallery of images labeled empty that the model classifies as an animal.
  Allow users to organize the gallery by predicted species, confidence and camera location. Add grouping by visual background characteristics such as darkness, vegetation density or color distribution.
  ```
- **AI output summary:** Outlined a Python script to filter empty frames from `predictions.parquet`, compute visual background metrics with PIL and NumPy, and structure the data for an interactive gallery.
- **Decision:** modified
- **Why:** The AI initially tried to load huge external computer vision libraries and wanted to re-run model inference on the whole 12 GB dataset from scratch. I stopped it and told it to just use our existing `predictions.parquet` and lightweight PIL/NumPy operations.
- **Problems caused:** bug 
- **Time spent verifying:** 15

---

### 2026-10-19: building the gallery pipeline and notebook

- **Who:** Lara
- **Goal:** Turn our analysis into an interactive website and an accompanying notebook to walk through the data.
- **Prompt (important ones):**
  ```
  ```
- **AI output summary:** code
- **Decision:** modified
- **Why:**  produced a functional prototype
- **Problems caused:** none
- **Time spent verifying:** 15

---

### 2026-10-19: converting to light theme and adding interactive charts

- **Who:** Lara
- **Goal:** replace the dark terminal-style layout with a clean, light-themed dashboard that includes interactive charts linked to example images.
- **Prompt (important ones):**
  ```
  Refactor the gallery interface to a clean, light-themed research layout. Make sure any percentage plotted clearly states what it is calculating.
  ```
- **AI output summary:** modified code
- **Decision:** accepted/modified
- **Why:** light theme seemed a better choice, personally. also plots were not good enough readable this the chances in the percentage plotting
- **Problems caused:** none
- **Time spent verifying:** 3

---

### 2026-10-19: cleaning up camera bar chart percentages

- **Who:** Lara
- **Goal:** Fix the confusing numbers.
- **Prompt (important ones):**
  ```
  The floating percentages on the camera bar chart conflict with the y-axis. Add a switch between 'Count of False Alarms' and 'Error Rate'.
  ```
- **AI output summary:** Added a button between error count and false alarm rate, updated the bar labels.
- **Decision:** accepted
- **Why:** better visualizaion
- **Problems caused:** none
- **Time spent verifying:** 1

---

### 2026-10-19: sorting camera bars descending when switching to error rate

- **Who:** Lara
- **Goal:** make sure the camera bar chart stays sorted descending when switching from count view to error rate view.
- **Prompt (important ones):**
  ```
  Update the camera hotspot chart logic so that switching the metric between Count and Error Rate re-sorts the bars in descending order.
  ```
- **AI output summary:** modified coded to update the ordering
- **Decision:** accepted
- **Why:** Simple fix, keeps the worst problem cameras on the left no matter which view you're looking at.
- **Problems caused:** none
- **Time spent verifying:** 3

---

### 2026-10-19: Fixing empty gallery bug when clicking confusion matrix cells

- **Who:** Lara
- **Goal:** Fix the bug where clicking on the confusion matrix showed zero photos, and add an inspector card below it.
- **Prompt (important ones):**
  ```
  Fix an bug in the confusion matrix: clicking a cell currently displays zero gallery 
  ```
- **AI output summary:** found the bug and updated code 
- **Decision:** accepted
- **Why:** Clicking the cell was broken 
- **Problems caused:** none
- **Time spent verifying:** 12

---

### 2026-10-26: making the dashboard more intuitive with thematic colors

- **Who:** Lara
- **Goal:** make the dashboard friendlier for non-experts  and color-code the different sections.
- **Prompt (important ones):**
  ```
  Improve the dashboard information. Apply distinct color accents to the analytical sections
  ```
- **AI output summary:** color-coded the tabs, and added a short explanation card.
- **Decision:** accepted/modified
- **Why:** changed explaination a little
- **Problems caused:** none
- **Time spent verifying:** 12

---

### 2026-10-26: connecting explainer cards to tabs

- **Who:** Lara
- **Goal:** Let users click on the top overview cards to jump straight down to that specific analysis tab.
- **Prompt (important ones):**
  ```
  i like the overview cards, however can we turn it into interactive navigational controls: clicking a card should activate and scroll down to its detailed analysis tab.
  ```
- **AI output summary:** Added click handlers to the top summary cards that switch tabs and scroll to them
- **Decision:** accepted
- **Why:** makes the usage more natural and easier
- **Problems caused:** none
- **Time spent verifying:** 8

---

### 2026-10-28: adding phantom species ranking tab

- **Who:** Lara
- **Goal:** Add a table showing which animals get hallucinated most often by the model.
- **Prompt (important ones):**
  ```
  Add a dedicated 'Phantom Species' tab featuring a ranked table of all species predicted on empty frames. Include error count, percentage of total false alarms, and mean model confidence, with clickable rows that filter the gallery.
  ```
- **AI output summary:** code, built species table
- **Decision:** accepted
- **Why:** good overview on what species are hallucinated into empty frames 
- **Problems caused:** none
- **Time spent verifying:** 10

---

### 2026-10-28: Coloring card borders by species and camera

- **Who:** Lara
- **Goal:** Make it obvious at a glance if two errors are the same species or from the same camera by color-coding card borders.
- **Prompt (important ones):**
  ```
  can we colour in **only the baclground each** - colour the background behind the image - with an colour representing the species, and below the part - where "camera" and stuff in a color representing the camera which took the foto
  ```
- **AI output summary:** colour coded according to the prompt
- **Decision:** accepted
- **Why:** can see it at first glance
- **Problems caused:** none
- **Time spent verifying:** 7

---

### 2026-10-28: allow choosing how many images are displayed at a time

- **Who:** Lara
- **Goal:** allow choosing how many images are displayed at a time
- **Prompt (important ones):**
  ```
  also make this changeable - so that more or less images are presented, depending on the user: Displaying 36 false detection frames:
  ```
- **AI output summary:** Added page size pills and buttons to browse through results.
- **Decision:** accepted
- **Why:** browsing through images is better
- **Problems caused:** none
- **Time spent verifying:** 4

---

### 2026-10-28: Deleting the duplicate tab buttons

- **Who:** Lara
- **Goal:** Get rid of the second row of tab buttons
- **Prompt (important ones):**
  ```
  please remove this:
  [Screenshot showing the horizontal tab button bar]
  ```
- **AI output summary:** removed it
- **Decision:** accepted
- **Why:** I had two identical sets of tab buttons stacked on top of each other.
- **Problems caused:** none
- **Time spent verifying:** 4

---

### 2026-10-30: Using real scientific binomial names

- **Who:** Lara
- **Goal:** Use proper scientific binomial names (*Panthera onca*, *Equus ferus*) matching the official dataset instead of informal names.
- **Prompt (important ones):**
  ```
  can you please keep the original name of the species? keep the emoji but also the real scientific name
  [Screenshot showing species header]
  ```
- **AI output summary:** Updated names with scientific names alongside animal emoji icons, kept the common name in paranthesis
- **Decision:** accepted
- **Why:** somewhere inbetween the LLM added names for the species that are more in the day to day usage for most. however, wanted to stick with the true scientifical naming - however, wanted to keep the emojies next to it, so that the scientific names are more intuitive
- **Problems caused:** did not remove it all
- **Time spent verifying:** 6

---

### 2026-10-30: Removing English names everywhere

- **Who:** Lara
- **Goal:** Completely scrub English common names for the species
- **Prompt (important ones):**
  ```
  no this is still wrong -  Argusianus argus (Great Argus)
  🐆 Panthera onca (Jaguar)
  🐎 Equus ferus (Wild Horse)
  🦃 Crax rubra (Great Curassow)
  cut out the other naming entirly alsooo not in paranthesis
  ```
- **AI output summary:** cleaned the naming 
- **Decision:** accepted
- **Why:** keep everything consistent with the official dataset labels.
- **Problems caused:** none
- **Time spent verifying:** 5

---

### 2026-10-30: grouping errors into 4 physical illusion buckets

- **Who:** Lara
- **Goal:** See if grouping errors into 4 physical illusion buckets (flash glints, branches, camouflage leaves, sun glare) actually works.
- **Prompt (important ones):**
  ```
  Implement a prototype "grouping errors into 4 physical illusion buckets": classify empty-frame false detections into physical visual mechanisms (specular flash reflections, branch pareidolia, foliage camouflage, and sun glare) using heuristic rules on image features.
  ```
- **AI output summary:** Added heuristic rules assigning every false alarm to one of the four illusion archetypes, with badges on cards and a dedicated dossier tab.
- **Decision:** rejected
- **Why:** Honestly, after looking at the results, it felt like total guesswork. can't actually prove whether the AI was fooled by a specific branch or a leafy patch.
- **Problems caused:** just speculations, bug
- **Time spent verifying:** 20

---

### 2026-10-31: Removing duplicate background buttons

- **Who:** Lara
- **Goal:** Get rid of the extra row of background buttons since we already have a dropdown for it
- **Prompt (important ones):**
  ```
  Remove the duplicate visual background pill button row above the model tabs. Consolidate background filtering into the primary toolbar dropdown.
  ```
- **AI output summary:** Deleted code
- **Decision:** accepted
- **Why:** We had the exact same filter in two spots
- **Problems caused:** none
- **Time spent verifying:** 4

---

### 2026-10-31: removing a section 

- **Who:** Lara
- **Goal:** drop a section implemented 
- **Prompt (important ones):**
  ```
  Remove the confidence comparison chapter comparing false negatives (missed animals) with false positives (hallucinations).
  ```
- **AI output summary:** Removed code
- **Decision:** accepted
- **Why:** was not intuitive enough
- **Problems caused:** none
- **Time spent verifying:** 6

---

### 2026-10-31: Combining ID and OOD into Unified Dataset Splits

- **Who:** Lisa
- **Goal:** Merge in-distribution and out-of-distribution data into single test, validation, and training sets.
- **Prompt (important ones):**
  ```
  please summarise ID and OOD into one dataset each. so, there should be one test set, one validation set and one training set
  ```
- **AI output summary:** Remapped `id_test` into `test` and `id_val` into `val` so the dropdown just shows three simple splits: Test Set, Validation Set, and Training Set.
- **Decision:** accepted
- **Why:** Much simpler to look at three main splits instead of constantly switching between ID and OOD camera locations.
- **Problems caused:** During hot-reload, the Dash app kept the old split names in memory, so `id_val` and `id_test` still showed up at the bottom of the dropdown. Fixed by doing the remapping inside `create_app()` too and filtering the dropdown choices.
- **Time spent verifying:** 10 minutes
---

### 2026-11-01: Integrating English Species Names and Bilingual Headings

- **Who:** Lisa
- **Goal:** Show common English names for animals alongside their scientific names so they are easier to recognize.
- **Prompt (important ones):**
  ```
  Can we add English common names for the species so they are easier to recognize in the network, tooltips, and side panel? Keep the scientific names accessible too.
  Also add compact readable abbreviations for node labels so they fit nicely around clusters without losing the full name on hover.
  ```
- **AI output summary:**
  1. Added a CSV table with English animal names and wrote a helper script (`species_names.py`) to look them up.
  2. Put English names on hover cards, side panels, and dropdowns, with the scientific name in italics underneath.
  3. Shortened long names on network circles (like "W. peccary") so labels stay neat, while keeping the full name in the popup.
- **Decision:** accepted
- **Why:** Latin names are hard to recognize unless you're a biologist, having English names makes exploring the dashboard way more intuitive.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes
---

### 2026-11-01: Client-Side Dynamic Collision Avoidance for Network Labels

- **Who:** Lisa
- **Goal:** Stop species names from overlapping with nodes, arrows, or each other, and add horizontal scrolling on small screens.
- **Prompt (important ones):**
  ```
  The species names sometimes still collide with nodes, arrows, or each other in dense clusters.
  Can we add dynamic label collision avoidance that measures the rendered text and moves labels into clear slots?
  Also make sure the canvas scrolls horizontally on narrow screens instead of squishing the network.
  ```
- **AI output summary:** Wrote a small JavaScript script (`confusion_labels.js`) that measures text bounding boxes directly in the browser and moves overlapping labels into open space around nodes on zoom and pan. Added a minimum canvas width so the graph doesn't get squished.
- **Decision:** accepted
- **Why:** Python can't easily see how text actually renders in the browser, letting JavaScript measure the rendered text boxes and shift them in real time fixed the overlapping labels completely.
- **Problems caused:** none
- **Time spent verifying:** 20 minutes
---

### 2026-11-01: Configurable Dataset Paths and Filename Diagnostics CLI

- **Who:** Lisa
- **Goal:** Let the app take custom folder paths from the command line so anyone can run it on different machines (or external hard drives) without editing the code.
- **Prompt (important ones):**
  ```
  Make image and prediction paths configurable via command line arguments so the viewer can run when images are stored in a different folder or drive.
  Add a flag to quickly verify whether prediction filenames match images on disk without having to start the server.
  Document path usage and Windows/WSL examples in the README.
  ```
- **AI output summary:**
  1. Added command-line arguments (like `--images-dir` and `--predictions`) to point to custom folders.
  2. Added a `--check-data` flag to quickly check if prediction image files actually exist on disk before launching the dashboard.
  3. Added examples for Windows and WSL in the README.
- **Decision:** accepted
- **Why:** Everyone on the team has slightly different folder structures and drives, so this lets anyone run the app without breaking file paths.
- **Problems caused:** none
- **Time spent verifying:** 10 minutes

---

### 2026-11-01: Multi-Sequence Trajectory Chart

- **Who:** Lara
- **Goal:** Expand burst analysis beyond a static sequence into an interactive multi-sequence trajectory chart plotting confidence across consecutive frames.
- **Prompt (important ones):**
  ```
  Expand the photo burst analysis: instead of a single static sequence, build an interactive multi-sequence trajectory chart plotting model confidence across consecutive frames (Frame 1 to 10) for all affected burst sequences in the validation split. Enable interactive hover popovers displaying the actual photo.
  ```
- **AI output summary:** Updated `hallucinations.py` to extract all 33 multi-frame burst sequences with persistence filtering and quick spotlights, added instant hover photos, and connected sequence clicks to an active filmstrip.
- **Decision:** rejected
- **Why:** the information presented was not easy to understand - visualizaiton was not good for the content 
- **Problems caused:** none
- **Time spent verifying:** 8 minutes

---

### 2026-11-02: Burst Sequence Heatmap Matrix

- **Who:** Lara
- **Goal:** Replace the messy, counter-intuitive line plot with an intuitive Heatmap Matrix 
- **Prompt (important ones):**
  ```
  i think this is really really bad and counter intuitive - you cant really see anything on this plot need sth that is more intuitive and where you can see at one sight what this is about and if there is a clear trend or not
  ```
- **AI output summary:** Replaced the 33-line plot in `hallucinations.py` with a 2D Heatmap Matrix tracking multi-frame sequences, color-coded by prediction status, with hover image popups.
- **Decision:** accepted
- **Why:** the heatmap matrix makes it instantly obvious that in ovre 58% of burst sequences, the model is permanently locked in an error state across all 10 frames.
- **Problems caused:** none
- **Time spent verifying:** 15 minutes

---

### 2026-11-02: Focusing Error Audit on Test Set Domain Shift

- **Who:** Lara
- **Goal:** Shift the false detection audit from validation data to the out-of-distribution `test`
- **Prompt (important ones):**
  ```
  We need to look at the test set. Does the model hallucinate differently when deployed at completely new, unseen camera locations?
  ```
- **AI output summary:** Recompiled the audit pipeline on the `test` split
- **Decision:** accepted
- **Why:** In-distribution error rates hide the real-world deployment problem. Auditing unseen camera traps exposes extreme site-specific failure shortcuts.
- **Problems caused:** none
- **Time spent verifying:** 18 minutes

---

