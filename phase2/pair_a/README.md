# Pair A: Phase II Visualizations

**Pair A:**
- **Lisa Krimbacher** (`k12209298`)
- **Lara Pötzl** (`k12216038`)

We are the non-AI-aided pair for Phase II.

## Visualization 1: Nearest-Neighbor Viewer

[`notebooks/neighbor_viewer.ipynb`](notebooks/neighbor_viewer.ipynb) compares the same animal cut-out across different backgrounds. Each row shows a composite and its five closest training images, retrieved using cosine similarity between 2,048-dimensional features before the final classification layer. Retrieval uses distinct training sequences and excludes cut-out source sequences. 

### Quick Start

Activate the `ai-and-vis` environment, open the notebook and run all cells. A GPU is recommended for initial feature extraction. Browsing then uses results held in memory.

Composite predictions come from Phase I. The notebook reconstructs the images and extracts their features for retrieval.

## Visualization 2: 

