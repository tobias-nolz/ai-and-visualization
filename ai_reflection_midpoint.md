# Phase I Midpoint Reflection: Our Experience with GenAI

**Pair A:**
- **Lisa Krimbacher** — Student ID: `k12209298`
- **Lara Pötzl** — Student ID: `k12216038`

----
Important prompts are documented in [`ai-interactions/ai_log_pair_a.md`](ai-interactions/ai_log_pair_a.md).
---

## 1. How We Collaborated with GenAI

In Phase I, our pair used Generative AI to look into where and why the fine-tuned ResNet-50 model makes mistakes on the iWildCam camera-trap dataset.

### GenAI as a Tool vs. as a Collaborator
Throughout the project, we worked with the AI in two different ways:
- **As a tool (we ask, AI writes the code):** Writing boilerplate Dash code, calculating node positions and spacing, writing CSS styles, and setting up command-line arguments.
- **As a collaborator (thinking together):** Brainstorming ways to visualize errors, getting ideas for the UI, and double-checking whether our findings actually made sense.

---

## 2. How We Used GenAI for Richer Interactivity and Broader Analysis

Because the AI handled most of the repetitive Dash code and callbacks, we didn't have to spend all our time just getting basic charts to show up. This gave us the time to explore the data in more detail and make our visualizations much more interactive.

### 2.1 The Two Tasks We Looked At
Together with Pair B, we decided to investigate the model's behavior in two subtasks:
1. **Confusions between animal species** 
2. **Hallucinations (confusions with the class `empty`)** 

For Phase I, Pair A used GenAI. We built two interactive tools:
1. **Species Confusion Network:** An interactive Dash dashboard that shows which animal species get confused with each other, which camera traps produce the most of these errors, and how confusion patterns look across different dataset splits.
2. **Hallucinations Gallery:** An interactive notebook to inspect empty images where the model still predicted an animal.

### 2.2 What We Made Interactive
Writing complex linked callbacks in Dash by hand is usually tedious and time-consuming, especially when you are not that familiar with the library yet. Having the AI help with callback structure let us add several interactive features:
- **Connected views:** Clicking a species in the network graph immediately updates the rest of the dashboard: it zooms into that animal in the confusion matrix, filters the error bar charts, highlights the camera locations where it happened, and loads example pictures.
- **Side-by-side photo comparison:** Looking at error numbers only tells you so much. We added an image viewer so you can compare the actual camera trap photo against what the model predicted, making it easier to spot why it failed.
- **Quick filtering and sorting:** Sliders let you filter out rare species or small error counts, a dropdown lets you rank network arrows by confusion rate versus total error count, and camera tooltips show local error rates alongside total photo counts.
- **Label collision script (`confusion_labels.js`):** In dense parts of the network, labels constantly overlapped. We used the AI to help write a small JavaScript helper that checks label boxes in the browser and moves overlapping text out of the way when you zoom or pan.
- **Interactive filtering:** Users can adjust confidence threshold sliders, filter by camera trap or phantom species, and re-sort hundreds of cards on the fly (by confidence, camera, darkness, or vegetation density).
- **Cross chart interaction:** Clicking any bar in the camera chart or phantom species list, instantly filters the gallery below to that specific camera or animal.
- **Burst sequence tracking:** A heatmap showing photo sequences over time with photo previews on hover, helping us see whether mistakes are just short one-frame glitches or if the model stays stuck on the error across the entire burst.
- **Detailed prediction inspection:** Clicking an image card opens a pop-up showing the model's top predictions and confidence scores, making it clear how sure the model was and what other animals it considered.

---

## 3. How AI Suggestions Influenced Our Decisions

Working with the AI led to several practical changes during the project:

1. **Using a network graph for species confusion:** We knew we wanted to study species confusions, but a full 182×182 confusion matrix is huge and hard to read. The AI suggested showing confusions as a directed network graph. We liked this because it makes it immediately obvious which animals get confused with which, and in which direction the mistakes go.
2. **Dropping calendar seasons:** When looking into time patterns, the AI first suggested grouping images by calendar seasons (spring, summer, fall, winter). But when we looked closer at the dataset, we realized the 414 cameras are spread across the globe—including equatorial regions and different hemispheres where four seasons don't apply. Discussing this with the AI helped us drop calendar seasons early and look at camera locations instead.
3. **Fixing crowded network circles:** The AI's first layout placed every species on one giant circle. This quickly became an unreadable mess with overlapping lines and labels. We told the AI to find connected groups of confused species instead, and we split them into an inner circle for bigger clusters and an outer ring for small pairs.
4. **Dropping fake 'illusion' categories:** In our empty image analysis, the AI suggested grouping false detections into four categories (like sun glare, foliage, reflections, and pareidolia). We tried this out, but when we checked the photos, we realized these labels were just wild guesses with no real ground truth. So we threw the idea out and stuck strictly to real metadata like lighting and location.
5. **Data usage:** The AI initially defaulted to evaluating empty-frame errors on the in-distribution validation split (`id_val`). We directed the analysis specifically to the out-of-distribution `test` split.
6. **Enforcing tool separation over one combined tool:** The AI repeatedly suggested merging the Species Confusion Network and the Hallucination Gallery into a single monolithic script/dashboard. We actively rejected this. Keeping the tools separate ensured clean codebases and clear mental models.

---

## 4. Which Suggestions We Accepted, Changed, or Rejected

Throughout the design and implementation process, we kept track of what worked and what didn't (for the detailed chronological entries with prompts and decisions, see [`ai-interactions/ai_log_pair_a.md`](ai-interactions/ai_log_pair_a.md)):

### 4.1 Suggestions We Accepted or Adjusted
- **Accepted:** Using a network graph for species confusion (instead of only relying on a matrix table), and math formulas to bend arrow lines so they don't draw directly over other nodes.
- **Adjusted:** 
  - The AI's single-circle layout was way too crowded and we turned it into clustered rings based on connected groups.
  - The AI's UI layout didn't always make sense. For example, it often put buttons far away from what they were actually controlling (like placing the "Reset Focus" button up in the global page header instead of right next to the network graph), so we had to tell it to reorganize controls so related buttons sit next to the charts they affect. Similarly, it added a toggle checkbox to include or exclude the `empty` class, suggesting that this would drastically change the visualization. In reality that wasn't the case, and empty frames are an important part of error analysis that hiding them made no sense anyway, so we told the AI to remove the toggle and always keep `empty` included.
    - In the Hallucination Gallery, the AI generated speculative diagnostic text blurbs claiming "why" an error happened. We stripped out this speculative text and kept only empirical facts: ground-truth label, top-5 softmax probabilities, sequence frame index, and metadata.
- **Our own ideas:** Deciding on the two core tasks together with Pair B, splitting them into two separate tools, designing the side panel layout, combining ID and OOD into unified dataset splits (Test Set, Validation Set, and Training Set) to keep the analysis clean, adding English animal names, and measuring rendered elements in JavaScript so labels don't overlap.

### 4.2 Suggestions We Rejected
- **Calendar seasons:** Didn't make ecological sense since cameras are placed in totally different climate zones worldwide.
- **Heuristic illusion categories:** Guessing why the model saw an animal on an empty image (glare vs. foliage) was unfalsifiable guesswork, so we dropped it.
- **Merging both tools into one dashboard:** We insisted on keeping `species_confusion_network.py` and `hallucinations.py` separate.
- **33-line burst sequence plot:** An illegible spaghetti chart that we replaced with the sequence heatmap grid.

---

## 5. Failure Patterns We Observed

1. **Visual blindness:** The AI doesn't see what the screen looks like. It wrote code that technically ran, but labels overlapped, lines crossed over each other, and things looked cluttered until we gave it screenshots and detailed layout instructions.
2. **Weird UI choices:** The AI often put buttons and sliders in random places (like putting a 'Reset' button far away at the top of the page instead of next to the chart it controls). We had to tell it exactly where to place every UI element.
3. **Convincing hallucinations:** The AI can sound very smart and confident when explaining model mistakes or suggesting categories, but upon closer inspection, it sometimes just made things up. We therefore had to verify against the raw data and photos.
4. **Merging everything:** When asked to extend an analysis, the AI defaulted to dumping everything into one file or proposing to merge separate tools, which quickly degraded readability and maintainability without human boundaries.


---

## 6. Best Practices for Working with AI

1. **Be very specific about UI layout:** Don't let the AI guess where things go, tell it the exact spot and hierarchy.
2. **Work in small steps:** Ask for one change or one callback at a time. Big prompts with lots of requests often broke existing code or introduced subtle bugs.
3. **Don't let the AI guess reasons without data:** Always verify AI claims against the raw dataset and actual photos before trusting them.
4. **Use screenshots in prompting:** Sharing screenshots of the rendered browser UI alongside code allowed the AI to grasp visual issues better, that text descriptions alone.


