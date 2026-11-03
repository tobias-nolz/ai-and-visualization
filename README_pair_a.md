# Pair A: Species Confusion Network

**Pair A (AI-aided pair in Phase I):**
- **Lisa Krimbacher** (`k12209298`)
- **Lara Pötzl** (`k12216038`)

In Phase I, our pair was assigned to use Generative AI to analyze where and why the fine-tuned ResNet-50 model makes mistakes on the WILDS iWildCam camera-trap dataset. 

This README covers Pair A's interactive visualization tools: the **Species Confusion Network** (`phase1/pair_a/src/species_confusion_network.py`). It is an interactive Dash dashboard built to study which animal species get confused with each other, which camera traps produce the most errors, and how these misclassifications look across dataset splits.

TODO: Add Hallucination Part!

---

## What the Dashboard Shows

- **Directed Confusion Network:** Shows confusion flows between species as a network graph. Directed arrows point from the true animal to what the model predicted. Line thickness shows how many times that mistake happened, and circle size shows how common that species is in the dataset.
- **Confusion Matrix View:** A matrix heatmap focused on the most confused species pairs.
- **Linked Inspector Sidebar:** Clicking any animal in the network or matrix immediately updates the sidebar to focus on that species:
  - Bar charts of what it gets mistaken for (and what gets mistaken for it).
  - Which camera locations had the most errors for that species.
  - Side-by-side photo comparisons showing the true animal photo next to what the model predicted.
- **Camera Error Breakdown & Context:** Shows which camera locations had the most errors for that species. Hovering over any camera displays both the absolute mistake count and the true sighting error rate, plus total camera shots, so you can immediately tell a broken camera apart from one that just took thousands of photos. A dedicated callout card highlights the worst camera trap with photos of actual misclassified animals taken at that location.
- **English & Scientific Names:** Displays plain English common names on nodes and dropdowns, while keeping full scientific Latin names visible on hover and in detail panels.

---

## Quick Start

### 1. Requirements & Setup
Make sure the environment is activated and the predictions file exists:

```bash
conda activate ai-and-vis
```

If you don't have `outputs/predictions.parquet` yet, you can generate it with:
```bash
make inference
```
*(or copy `outputs/predictions.parquet` from a teammate).*

### 2. Launch the Dashboard
Run the Python script from the repository root:

```bash
python phase1/pair_a/src/species_confusion_network.py
```

Then open your browser at **http://127.0.0.1:8050/**.

---

## Important Note on Photos and Local Paths

A fresh git clone contains the code, but predictions and dataset images are gitignored. 

The dashboard's standard default paths are:
- Predictions: `outputs/predictions.parquet`
- Photos: `data/iwildcam_v2.0/train/`

The viewer can run with or without the image folder:
- **Without images:** All network graphs, matrix heatmaps, error metrics, and camera charts work right away without downloading the 12 GB dataset. Photo slots will simply show clean placeholders.
- **With images (recommended):** Seeing the actual photos side by side is what makes the errors understandable (for example, spotting how bad night lighting, camera angles, similar-looking species, or occluding branches fooled the network).

**Good to know:**
- **No GPU required:** The viewer itself never runs inference and does not need a GPU. You only need to run model inference if you don't already have a predictions file from a teammate.
- **Direct photo loading:** The viewer loads photos directly from the filenames listed in the prediction file, so it does not even need a separate `metadata.csv` to find images.
- **Check data without starting Dash:** Append `--check-data` to quickly check whether all image files exist on disk without launching the web server. It scans every unique filename in the predictions and exits with code 0 if all are found, or code 1 if files are missing.

---

## CLI Options & Custom Paths

If your prediction file or photos are saved in a different location or on an external drive, pass the paths using command-line arguments instead of changing the code:

```bash
python phase1/pair_a/src/species_confusion_network.py --predictions "/path/to/predictions.parquet" --images-dir "/path/to/iwildcam_v2.0/train"
```

### Available Arguments:
- `--predictions <PATH>`: Path to the predictions parquet file (default: `outputs/predictions.parquet`).
- `--images-dir <PATH>`: Folder containing the JPG photos directly (default: `data/iwildcam_v2.0/train/`). Overrides `--data-dir`.
- `--data-dir <PATH>`: Root dataset folder containing `train/` and `metadata.csv` (alternative to `--images-dir`).
- `--metadata <PATH>`: Optional path to `metadata.csv` if you want to override the metadata fallback.
- `--representatives <PATH>`: Optional path to override the curated `representative_images.json` map.
- `--check-data`: Scans all image filenames referenced in the predictions file to verify they exist on disk, prints a diagnostic summary, and exits without starting the server.
- `--port <PORT>`: Port to run the Dash web server on (default: `8050`).

### Path Resolution and Operating Systems:
- **Defaults vs Relative Paths:** Default paths resolve from the repository root. If you pass an explicit relative path, it resolves from your current terminal working directory.
- **Windows and WSL:** Both native Windows paths and WSL mount paths are supported. Always wrap paths in quotes if they contain spaces:
  - **Windows Python:** `--images-dir "D:\datasets\iwildcam_v2.0\train"`
  - **WSL Python:** `--images-dir "/mnt/d/datasets/iwildcam_v2.0/train"`
- **Restarting:** Remember to restart the viewer script after changing paths or moving image files.

---

## English Species Names

- **182 Class Mappings:** English common names are bundled locally for all 182 classes so the dashboard is easy to explore without having to look up Latin taxonomy.
- **Data Source & License:** Most common names come from the [WiSE-FT iWildCam table](https://github.com/mlfoundations/wise-ft/blob/4639a3919b51b20beae5814bc6722b962d93ae47/src/datasets/iwildcam_metadata/labels.csv), and its [MIT license](phase1/pair_a/data/iwildcam_english_names.LICENSE) is included in our repository.
- **Specific Taxonomic Adjustments:**
  - Genus-only categories (such as *Mazama sp.* or *Tayassu sp.*) remain generic rather than guessing a specific species.
  - Two additional species names not covered in WiSE-FT are documented from authoritative biological databases: [Animal Diversity Web](https://animaldiversity.org/accounts/Cercopithecus_ascanius/) for the red-tailed monkey (*Cercopithecus ascanius*) and the [US Geological Survey](https://www.usgs.gov/publications/potamochoerus-porcus-artiodactyla-suidae) for the red river hog (*Potamochoerus porcus*).
- **Label Formatting:** On the network graph, compact abbreviations are used for dense clusters (e.g. *"W. peccary"*), while full English common names and scientific Latin names (in italics) are always shown in tooltips and sidebar panels.

---

## Tips for Exploring the Dashboard

1. **Switch Dataset Splits:** Use the dropdown at the top left to switch between the **Test Set**, **Validation Set**, and **Training Set**.
2. **Click to Focus:** Click any node in the graph (or a row in the sidebar) to lock the dashboard onto that animal and see all its specific errors and photos. Click **"Reset Focus"** to zoom back out.
3. **Compare Photos:** In the sidebar or hover tooltips, click the ‹ and › buttons on photo slideshows to flip through different camera trap shots of that species.

