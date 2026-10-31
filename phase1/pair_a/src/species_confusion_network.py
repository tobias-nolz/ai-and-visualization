"""Interactive black-box species confusion & domain-shift explorer for WILDS iWildCam.

Place this file at phase1/pair_a/src/species_confusion_network.py, then run
from the repository root:

    python phase1/pair_a/src/species_confusion_network.py

It reads outputs/predictions.parquet. To use a different file:

    python phase1/pair_a/src/species_confusion_network.py --predictions path/to/file.parquet

Requirements: dash, plotly, pandas, pyarrow, networkx, numpy.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from sklearn.metrics import f1_score


COLUMNS = [
    "id", "split", "label", "label_name", "prediction",
    "prediction_name", "confidence", "location", "filename",
]
SPLIT_DESCRIPTIONS = {
    "test": "Test Set",
    "val": "Validation Set",
    "train": "Training Set",
}

# Lighter, accessible red scale avoiding near-black burgundy so overlapping text remains legible
ERROR_RATE_COLORSCALE = [
    [0.0, "#fff5f5"],
    [0.2, "#fed7d7"],
    [0.4, "#feb2b2"],
    [0.6, "#fc8181"],
    [0.8, "#f56565"],
    [1.0, "#e53e3e"],
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = (REPO_ROOT / "data/iwildcam_v2.0/train").resolve() if (REPO_ROOT / "data/iwildcam_v2.0/train").is_dir() else Path("data/iwildcam_v2.0/train").resolve()
META_CSV = (REPO_ROOT / "data/iwildcam_v2.0/metadata.csv").resolve() if (REPO_ROOT / "data/iwildcam_v2.0/metadata.csv").is_file() else Path("data/iwildcam_v2.0/metadata.csv").resolve()
REPR_JSON = (REPO_ROOT / "phase1/pair_a/data/representative_images.json").resolve() if (REPO_ROOT / "phase1/pair_a/data/representative_images.json").is_file() else Path("phase1/pair_a/data/representative_images.json").resolve()


@lru_cache(maxsize=1)
def _load_fallback_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not META_CSV.is_file() or not DATA_DIR.is_dir():
        return mapping
    try:
        import csv
        with open(META_CSV, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                y = int(row["y"])
                if y not in mapping:
                    fn = row["filename"]
                    if (DATA_DIR / fn).is_file():
                        mapping[y] = fn
                if len(mapping) >= 182:
                    break
    except Exception:
        pass
    return mapping


@lru_cache(maxsize=1)
def load_image_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    if REPR_JSON.is_file():
        try:
            import json
            with open(REPR_JSON, mode="r") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if (DATA_DIR / v).is_file():
                        mapping[int(k)] = v
        except Exception:
            pass
    if len(mapping) < 182:
        fb = _load_fallback_map()
        for k, v in fb.items():
            if k not in mapping:
                mapping[k] = v
    return mapping


@lru_cache(maxsize=800)
def _read_image_base64(filename: str) -> str | None:
    p = DATA_DIR / filename
    if not p.is_file():
        return None
    try:
        import base64
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def get_class_image(class_id: int) -> str | None:
    # 1. Check representative images json
    mapping = load_image_map()
    fn = mapping.get(int(class_id)) if hasattr(class_id, "__int__") else mapping.get(class_id)
    if fn:
        b64 = _read_image_base64(fn)
        if b64:
            return b64
    # 2. Check CLASS_MULTI_IMAGE_MAP (from dataset)
    multi = CLASS_MULTI_IMAGE_MAP.get(int(class_id), [])
    for m_fn in multi:
        b64 = _read_image_base64(m_fn)
        if b64:
            return b64
    return None


CAMERA_IMAGE_MAP: dict[str, str] = {}


def get_camera_image(location_id: int | str) -> str | None:
    fn = CAMERA_IMAGE_MAP.get(str(location_id))
    if not fn:
        return None
    return _read_image_base64(fn)


CLASS_MULTI_IMAGE_MAP: dict[int, list[str]] = {}
CAMERA_MULTI_IMAGE_MAP: dict[str, list[str]] = {}


def get_class_images(class_id: int, max_images: int = 5) -> list[str]:
    fns = CLASS_MULTI_IMAGE_MAP.get(int(class_id), [])
    if not fns:
        single = get_class_image(class_id)
        return [single] if single else []
    results = []
    for fn in fns[:max_images]:
        b64 = _read_image_base64(fn)
        if b64:
            results.append(b64)
    if not results:
        single = get_class_image(class_id)
        if single:
            results.append(single)
    return results


def get_camera_images(location_id: int | str, max_images: int = 5) -> list[str]:
    fns = CAMERA_MULTI_IMAGE_MAP.get(str(location_id), [])
    if not fns:
        single = get_camera_image(location_id)
        return [single] if single else []
    results = []
    for fn in fns[:max_images]:
        b64 = _read_image_base64(fn)
        if b64:
            results.append(b64)
    if not results:
        single = get_camera_image(location_id)
        if single:
            results.append(single)
    return results


MISTAKE_SPEC_CAM_MAP: dict[tuple[int, str], list[dict]] = {}
MISTAKE_PAIR_CAM_MAP: dict[tuple[int, int, str], list[dict]] = {}
MISTAKE_SINK_CAM_MAP: dict[tuple[int, str], list[dict]] = {}


def get_mistake_images_for_species_cam(
    class_id: int, location_id: int | str, max_images: int = 5
) -> tuple[list[str], list[str]]:
    items = MISTAKE_SPEC_CAM_MAP.get((int(class_id), str(location_id)), [])
    results: list[str] = []
    captions: list[str] = []
    for it in items[:max_images]:
        fn = it["filename"]
        b64 = _read_image_base64(fn)
        if b64:
            results.append(b64)
            p_name = it.get("pred_name", "Unknown")
            caption = f"Model predicted: <strong style='color: #0284c7;'>{p_name}</strong>"
            captions.append(caption)
    if not results:
        fallback = get_camera_images(location_id, max_images=max_images)
        return fallback, ["" for _ in fallback]
    return results, captions


def get_mistake_images_for_pair(src_id: int, dst_id: int, location_id: int | str, max_images: int = 5) -> list[str]:
    fns = MISTAKE_PAIR_CAM_MAP.get((int(src_id), int(dst_id), str(location_id)), [])
    results = []
    for it in fns[:max_images]:
        fn = it["filename"] if isinstance(it, dict) else it
        b64 = _read_image_base64(fn)
        if b64:
            results.append(b64)
    if not results:
        fallback_imgs, _ = get_mistake_images_for_species_cam(src_id, location_id, max_images=max_images)
        return fallback_imgs
    return results


def get_mistake_images_for_sink(sink_id: int, location_id: int | str, max_images: int = 5) -> list[str]:
    fns = MISTAKE_SINK_CAM_MAP.get((int(sink_id), str(location_id)), [])
    results = []
    for it in fns[:max_images]:
        fn = it["filename"] if isinstance(it, dict) else it
        b64 = _read_image_base64(fn)
        if b64:
            results.append(b64)
    if not results:
        return get_camera_images(location_id, max_images=max_images)
    return results


def make_diashow(
    images: list[str],
    captions: list[str] | None = None,
    aspect_ratio: str | None = None,
    width: str = "100%",
    border_color: str = "#cbd5e1",
    height: str | None = None,
) -> html.Div:
    if not images:
        empty_style = {
            "width": width, "maxWidth": "100%", "background": "#f8fafc",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "fontSize": "10px", "color": "#94a3b8", "borderRadius": "4px",
            "border": f"1px solid {border_color}", "boxSizing": "border-box",
        }
        if height:
            empty_style["height"] = height
        else:
            empty_style["height"] = "135px"
        return html.Div("No photo", style=empty_style)

    first_img = images[0]
    img_style = {
        "width": "100%", "maxWidth": "100%", "display": "block", "boxSizing": "border-box",
        "borderRadius": "4px",
    }
    if height:
        img_style["height"] = height
        img_style["objectFit"] = "cover"
    elif aspect_ratio:
        img_style["aspectRatio"] = aspect_ratio
        img_style["height"] = "auto"
        img_style["objectFit"] = "cover"
    else:
        img_style["height"] = "auto"

    first_caption = captions[0] if (captions and len(captions) > 0 and captions[0]) else None
    caption_block = None
    if first_caption:
        caption_block = html.Div(
            dcc.Markdown(first_caption, dangerously_allow_html=True),
            className="diashow-caption",
            style={
                "fontSize": "11px", "color": "#334155", "marginTop": "6px",
                "textAlign": "center", "background": "#ffffff", "padding": "4px 8px",
                "borderRadius": "6px", "border": f"1px solid {border_color}",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
                "minHeight": "20px", "boxSizing": "border-box",
            }
        )

    if len(images) == 1:
        return html.Div([
            html.Img(
                src=first_img,
                style={**img_style, "border": f"1px solid {border_color}"},
            ),
            caption_block if caption_block else html.Div(),
        ], style={"position": "relative", "borderRadius": "4px", "width": width, "maxWidth": "100%", "boxSizing": "border-box", "flexShrink": "0" if width != "100%" else None})

    images_json = json.dumps(images)
    captions_json = json.dumps(captions) if captions else "[]"
    return html.Div([
        html.Div([
            html.Img(
                src=first_img,
                className="diashow-img",
                style=img_style,
            ),
            html.Div(
                "‹",
                className="diashow-btn diashow-prev",
                **{"data-step": "-1"},
                title="Previous photo (Left arrow or click)",
            ),
            html.Div(
                "›",
                className="diashow-btn diashow-next",
                **{"data-step": "1"},
                title="Next photo (Right arrow or click)",
            ),
            html.Div(
                f"1 / {len(images)}",
                className="diashow-counter",
            ),
        ], style={
            "position": "relative", "borderRadius": "4px",
            "overflow": "hidden", "border": f"1px solid {border_color}",
            "boxSizing": "border-box", "width": "100%", "maxWidth": "100%",
        }),
        caption_block if caption_block else html.Div(),
    ], className="diashow-container", style={"width": width, "maxWidth": "100%", "boxSizing": "border-box", "flexShrink": "0" if width != "100%" else None}, **{"data-idx": "0", "data-images": images_json, "data-captions": captions_json})


def read_predictions(path: Path) -> pd.DataFrame:
    actual_path = path
    if not actual_path.is_file() and (REPO_ROOT / path).is_file():
        actual_path = REPO_ROOT / path
    if not actual_path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Run `make inference` from your repository root first, "
            "or provide --predictions PATH."
        )
    data = pd.read_parquet(actual_path, columns=COLUMNS)
    if data.empty:
        raise ValueError("Prediction file contains no rows.")
    data = data.dropna(subset=["split", "label", "prediction"])
    data["split"] = data["split"].astype(str)
    # Combine ID and OOD splits into unified test and validation datasets
    data["split"] = data["split"].replace({
        "id_test": "test",
        "id_val": "val",
    })
    data["label"] = data["label"].astype(int)
    data["prediction"] = data["prediction"].astype(int)
    data["location"] = data["location"].astype(str)
    return data


def class_names(data: pd.DataFrame) -> dict[int, str]:
    """Map class IDs to names, including classes seen only as predictions."""
    result: dict[int, str] = {}
    for id_col, name_col in [("label", "label_name"), ("prediction", "prediction_name")]:
        for class_id, name in data[[id_col, name_col]].drop_duplicates().itertuples(index=False, name=None):
            if pd.notna(name) and str(name).strip():
                result[int(class_id)] = str(name)
    return result


def summarize_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-class statistics, directed confusion edges, and camera location confounding."""
    support = data.groupby("label").size().rename("support")
    mistakes = data.loc[data["label"] != data["prediction"]]
    error_counts = mistakes.groupby("label").size().rename("errors")
    stats = support.to_frame().join(error_counts).fillna({"errors": 0})
    stats[["support", "errors"]] = stats[["support", "errors"]].astype(int)
    stats["error_rate"] = stats["errors"] / stats["support"]

    false_positives = mistakes.groupby("prediction").size().rename("falsely_predicted_count")
    stats = stats.join(false_positives).fillna({"falsely_predicted_count": 0})
    stats["falsely_predicted_count"] = stats["falsely_predicted_count"].astype(int)

    # Species camera presence and top error cameras
    species_cams = data.groupby("label")["location"].nunique().rename("cams_present")
    stats = stats.join(species_cams).fillna({"cams_present": 1})
    stats["cams_present"] = stats["cams_present"].astype(int)

    # Precompute camera and species occurrence maps
    cam_total_s = data.groupby("location").size()
    cam_total_map = {str(k): int(v) for k, v in cam_total_s.items()}
    cam_errs_s = mistakes.groupby("location").size()
    cam_errs_map = {str(k): int(v) for k, v in cam_errs_s.items()}
    spec_cam_s = data.groupby(["label", "location"]).size()
    spec_cam_map = {(int(l), str(c)): int(v) for (l, c), v in spec_cam_s.items()}

    if not mistakes.empty:
        spec_loc_grp = mistakes.groupby(["label", "location"]).size().reset_index(name="cnt")
        top_spec_loc = spec_loc_grp.sort_values(["label", "cnt"], ascending=[True, False]).drop_duplicates("label").set_index("label")

        top_err_cam_list = []
        top_err_cnt_list = []
        top_err_supp_list = []
        top_err_tot_list = []
        for lbl in stats.index:
            if lbl in top_spec_loc.index:
                loc = str(top_spec_loc.loc[lbl, "location"])
                cnt = int(top_spec_loc.loc[lbl, "cnt"])
                supp = spec_cam_map.get((int(lbl), loc), cnt)
                tot = cam_total_map.get(loc, 0)
            else:
                loc = "N/A"
                cnt = 0
                supp = 0
                tot = 0
            top_err_cam_list.append(loc)
            top_err_cnt_list.append(cnt)
            top_err_supp_list.append(supp)
            top_err_tot_list.append(tot)

        stats["top_err_cam"] = top_err_cam_list
        stats["top_err_cam_cnt"] = top_err_cnt_list
        stats["top_err_cam_supp"] = top_err_supp_list
        stats["top_err_cam_total"] = top_err_tot_list

        sink_loc_grp = mistakes.groupby(["prediction", "location"]).size().reset_index(name="cnt")
        top_sink_loc = sink_loc_grp.sort_values(["prediction", "cnt"], ascending=[True, False]).drop_duplicates("prediction").set_index("prediction")

        top_sink_cam_list = []
        top_sink_cnt_list = []
        top_sink_supp_list = []
        top_sink_tot_list = []
        top_sink_errs_list = []
        for lbl in stats.index:
            if lbl in top_sink_loc.index:
                loc = str(top_sink_loc.loc[lbl, "location"])
                cnt = int(top_sink_loc.loc[lbl, "cnt"])
                supp = spec_cam_map.get((int(lbl), loc), 0)
                tot = cam_total_map.get(loc, 0)
                errs = cam_errs_map.get(loc, 0)
            else:
                loc = "N/A"
                cnt = 0
                supp = 0
                tot = 0
                errs = 0
            top_sink_cam_list.append(loc)
            top_sink_cnt_list.append(cnt)
            top_sink_supp_list.append(supp)
            top_sink_tot_list.append(tot)
            top_sink_errs_list.append(errs)

        stats["top_sink_cam"] = top_sink_cam_list
        stats["top_sink_cam_cnt"] = top_sink_cnt_list
        stats["top_sink_cam_supp"] = top_sink_supp_list
        stats["top_sink_cam_total"] = top_sink_tot_list
        stats["top_sink_cam_errs"] = top_sink_errs_list
    else:
        stats["top_err_cam"] = "N/A"
        stats["top_err_cam_cnt"] = 0
        stats["top_err_cam_supp"] = 0
        stats["top_err_cam_total"] = 0
        stats["top_sink_cam"] = "N/A"
        stats["top_sink_cam_cnt"] = 0
        stats["top_sink_cam_supp"] = 0
        stats["top_sink_cam_total"] = 0
        stats["top_sink_cam_errs"] = 0

    edges = (
        mistakes.groupby(["label", "prediction"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"label": "source", "prediction": "target"})
    )
    edges = edges.merge(
        support.rename("source_support"),
        left_on="source", right_index=True, how="left",
    )
    edges["rate"] = edges["count"] / edges["source_support"]

    loc_grp = (
        mistakes.groupby(["label", "prediction", "location"])
        .size()
        .reset_index(name="loc_cnt")
    )
    top_locs = loc_grp.sort_values(
        ["label", "prediction", "loc_cnt"], ascending=[True, True, False]
    ).drop_duplicates(subset=["label", "prediction"])
    num_cams = loc_grp.groupby(["label", "prediction"]).size().rename("num_cams")

    edges = edges.merge(
        top_locs.rename(columns={
            "label": "source", "prediction": "target",
            "location": "top_loc", "loc_cnt": "top_loc_cnt"
        }),
        on=["source", "target"], how="left"
    )
    edges = edges.merge(
        num_cams.rename_axis(["source", "target"]).reset_index(),
        on=["source", "target"], how="left"
    )
    edges["num_cams"] = edges["num_cams"].fillna(1).astype(int)
    edges["top_loc_cnt"] = edges["top_loc_cnt"].fillna(0).astype(int)
    edges["top_loc_pct"] = edges["top_loc_cnt"] / edges["count"]
    edges["top_loc_src_supp"] = [spec_cam_map.get((int(s), str(l)), int(c)) for s, l, c in zip(edges["source"], edges["top_loc"], edges["top_loc_cnt"])]
    edges["top_loc_cam_total"] = [cam_total_map.get(str(l), 0) for l in edges["top_loc"]]

    # Compute overall camera trap statistics
    cam_stats = {}
    for loc, tot in cam_total_map.items():
        err = cam_errs_map.get(loc, 0)
        cam_stats[loc] = {
            "total": tot,
            "errors": err,
            "error_rate": err / tot if tot > 0 else 0.0,
        }
    mistakes.attrs["cam_stats"] = cam_stats
    mistakes.attrs["spec_cam_map"] = spec_cam_map
    mistakes.attrs["cam_total_map"] = cam_total_map

    return stats, edges, mistakes


def visible_edges(
    edges: pd.DataFrame,
    names: dict[int, str],
    min_support: int,
    min_count: int,
    min_rate_pct: float,
    max_edges: int,
    sort_by: str,
    focused_class: int | None,
) -> pd.DataFrame:
    mask = (edges["source_support"] >= min_support)
    if sort_by == "rate":
        mask = mask & (edges["rate"] * 100 >= min_rate_pct)
    else:
        mask = mask & (edges["count"] >= min_count)
    result = edges.loc[mask].copy()
    if focused_class is not None:
        result = result.loc[
            (result["source"] == focused_class) | (result["target"] == focused_class)
        ]
    order = ["rate", "count", "source", "target"] if sort_by == "rate" else ["count", "rate", "source", "target"]
    return result.sort_values(order, ascending=[False, False, True, True]).head(max_edges)


# -----------------------------------------------------------------------------
# GRAPH LAYOUT ALGORITHMS (COMMUNITY CLUSTERS)
# -----------------------------------------------------------------------------

def compute_layout(
    edges: pd.DataFrame,
    stats: pd.DataFrame,
) -> tuple[dict[int, tuple[float, float]], list[int], dict[int, tuple[float, float]]]:
    """Compute community clusters layout: groups connected species subgraphs into visual clusters."""
    all_nodes = sorted(list(set(edges["source"]).union(set(edges["target"]))))
    if not all_nodes:
        return {}, [], {}

    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(int(row.source), int(row.target), weight=float(row.count))
    for n in all_nodes:
        if not G.has_node(n):
            G.add_node(n)

    components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    positions: dict[int, tuple[float, float]] = {}
    node_centers: dict[int, tuple[float, float]] = {}
    ordered_nodes: list[int] = []
    num_comp = len(components)

    # Compute radius for each component proportional to its node count
    comp_radii = []
    for comp in components:
        m = len(comp)
        if m == 1:
            comp_radii.append(0.0)
        elif m == 2:
            comp_radii.append(1.20)
        else:
            base_r = 1.30 + 0.22 * m if m >= 6 else 0.85 + 0.16 * m
            comp_radii.append(max(1.50, base_r))

    comp_centers: dict[int, tuple[float, float]] = {}
    if num_comp == 1:
        comp_centers[0] = (0.0, 0.0)
    elif num_comp > 1 and len(components[0]) >= 6:
        # Dominant cluster sits at center (0, 0); smaller satellite clusters orbit around it with guaranteed clearance
        comp_centers[0] = (0.0, 0.0)
        r_center = comp_radii[0]
        num_sat = num_comp - 1
        max_sat_r = max(comp_radii[1:]) if num_sat > 0 else 0.6
        orbit_r = r_center + max_sat_r + 5.4
        min_orbit_for_sat = (num_sat * (max_sat_r * 2 + 3.6)) / (2 * np.pi)
        orbit_r = max(orbit_r, min_orbit_for_sat)
        # Offset starting angle by 45 degrees so satellites never sit on the 3 o'clock axis
        phi = np.linspace(np.pi / 4, 2 * np.pi + np.pi / 4, num_sat, endpoint=False)
        for i in range(num_sat):
            comp_centers[i + 1] = (orbit_r * np.cos(phi[i]), orbit_r * np.sin(phi[i]))
    else:
        # Multiple smaller clusters distributed on a ring with generous spacing
        max_r = max(comp_radii) if comp_radii else 0.8
        ring_r = max(3.8, (num_comp * (max_r * 2 + 3.4)) / (2 * np.pi))
        phi = np.linspace(np.pi / 4, 2 * np.pi + np.pi / 4, num_comp, endpoint=False)
        for idx in range(num_comp):
            comp_centers[idx] = (ring_r * np.cos(phi[idx]), ring_r * np.sin(phi[idx]))

    for idx, comp in enumerate(components):
        comp_nodes = sorted(list(comp), key=lambda n: stats.loc[n, "support"] if n in stats.index else 0, reverse=True)
        ordered_nodes.extend(comp_nodes)
        cx, cy = comp_centers[idx]
        m = len(comp_nodes)
        rad = comp_radii[idx]

        # Detect dominant hub node in large components (e.g. 'empty' or central sink)
        hub_node = None
        if m >= 6:
            comp_set = set(comp_nodes)
            deg_counts = {n: 0 for n in comp_nodes}
            for row in edges.itertuples(index=False):
                s, t = int(row.source), int(row.target)
                if s in comp_set and t in comp_set:
                    deg_counts[s] += 1
                    deg_counts[t] += 1
            if deg_counts:
                best_node = max(deg_counts, key=deg_counts.get)
                if deg_counts[best_node] >= 4 and deg_counts[best_node] >= 0.35 * (m - 1):
                    hub_node = best_node

        if m == 1:
            positions[comp_nodes[0]] = (cx, cy)
            node_centers[comp_nodes[0]] = (cx, cy)
        elif m == 2:
            # Place two nodes perpendicular to radial vector from origin so labels project outward away from other clusters
            if abs(cx) > 1e-4 or abs(cy) > 1e-4:
                cluster_angle = np.arctan2(cy, cx)
                tangent = cluster_angle + np.pi / 2
                positions[comp_nodes[0]] = (cx + rad * np.cos(tangent), cy + rad * np.sin(tangent))
                positions[comp_nodes[1]] = (cx - rad * np.cos(tangent), cy - rad * np.sin(tangent))
            else:
                positions[comp_nodes[0]] = (cx - rad, cy)
                positions[comp_nodes[1]] = (cx + rad, cy)
            node_centers[comp_nodes[0]] = (cx, cy)
            node_centers[comp_nodes[1]] = (cx, cy)
        elif hub_node is not None:
            # Place dominant hub at center of cluster; peripheral nodes evenly on ring (radial spokes never cross)
            positions[hub_node] = (cx, cy)
            node_centers[hub_node] = (cx, cy)
            ring_nodes = [n for n in comp_nodes if n != hub_node]
            local_angles = np.linspace(0, 2 * np.pi, len(ring_nodes), endpoint=False)
            for node, la in zip(ring_nodes, local_angles):
                positions[node] = (cx + rad * np.cos(la), cy + rad * np.sin(la))
                node_centers[node] = (cx, cy)
        else:
            local_angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
            for node, la in zip(comp_nodes, local_angles):
                positions[node] = (cx + rad * np.cos(la), cy + rad * np.sin(la))
                node_centers[node] = (cx, cy)

    return positions, ordered_nodes, node_centers


def bezier_curve(
    p0: np.ndarray,
    p1: np.ndarray,
    curvature: float = 0.16,
    n_pts: int = 100,
) -> np.ndarray:
    """Sample a quadratic curve; the displayed spline removes visible corners."""
    t = np.linspace(0, 1, n_pts)[:, None]
    direction = p1 - p0
    normal = np.array([-direction[1], direction[0]])
    control = (p0 + p1) / 2 + curvature * normal
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1


def route_edge(src, dst, positions, clearance, mutual=False):
    """Prefer a gentle curve; bend farther only to clear another species node.

    Reciprocal edges keep positive curvature in their own direction, putting
    them on opposite sides. Clearance guides routing at the initial view;
    Plotly handles the endpoint's pixel spacing separately at every zoom.
    """
    p0, p1 = np.asarray(positions[src]), np.asarray(positions[dst])
    bends = [0.24, 0.34, 0.46, 0.60] if mutual else [0.12, -0.12, 0.24, -0.24, 0.40, -0.40, 0.60, -0.60]
    obstacles = [node for node in positions if node not in (src, dst)]
    candidates = []
    for bend in bends:
        points = bezier_curve(p0, p1, bend)
        penalty = 0.0
        for node in obstacles:
            distance = np.linalg.norm(points - positions[node], axis=1).min()
            penetration = max(0.0, clearance[node] - distance)
            penalty += (penetration / clearance[node]) ** 2
        candidates.append((penalty, abs(bend), points))
    return min(candidates, key=lambda candidate: candidate[:2])[2]


# -----------------------------------------------------------------------------
# PLOTLY FIGURES
# -----------------------------------------------------------------------------

def network_figure(
    edges: pd.DataFrame,
    stats: pd.DataFrame,
    names: dict[int, str],
    focused_class: int | None,
) -> go.Figure:
    fig = go.Figure()

    if edges.empty:
        fig.add_annotation(
            text="<b>No confusion pairs match these thresholds.</b><br>Try lowering the sliders or selecting another focus class.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 15, "color": "#64748b"},
        )
        fig.update_layout(
            height=640, xaxis={"visible": False}, yaxis={"visible": False},
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        )
        return fig

    positions, ordered_nodes, node_centers = compute_layout(edges, stats)
    max_count = max(float(edges["count"].max()), 1.0)
    max_support = max(float(stats.loc[stats.index.isin(ordered_nodes), "support"].max() or 1.0), 1.0)
    edge_pairs = set(zip(edges["source"].astype(int), edges["target"].astype(int)))
    # Marker diameters are pixels. Reuse the exact same sizes for endpoint gaps.
    diameters = {
        node: 16.0 + 20.0 * np.sqrt(float(stats.loc[node, "support"]) / max_support)
        if node in stats.index else 16.0
        for node in ordered_nodes
    }
    coordinates = np.asarray(list(positions.values()))
    units_per_pixel = 1.2 * max(np.ptp(coordinates[:, 0]) / 520, np.ptp(coordinates[:, 1]) / 590, 0.002)
    clearance = {node: (diameters[node] / 2 + 9) * units_per_pixel for node in ordered_nodes}


    # 1. Edges
    for row in edges.itertuples(index=False):
        src, dst = int(row.source), int(row.target)
        if src not in positions or dst not in positions:
            continue
        pts = route_edge(src, dst, positions, clearance, mutual=(dst, src) in edge_pairs)

        is_outgoing = (focused_class is not None and src == focused_class)
        is_incoming = (focused_class is not None and dst == focused_class)

        rel_weight = np.sqrt(float(row.count) / max_count)

        if focused_class is None:
            # Uniform opacity so high-count connections scale purely by width and never appear as an isolated dark color
            edge_color = "rgba(100, 116, 139, 0.48)"
            line_width = 1.4 + 5.5 * rel_weight
        elif is_outgoing:
            edge_color = "rgba(234, 88, 12, 0.92)"
            line_width = 2.5 + 6.0 * rel_weight
        elif is_incoming:
            edge_color = "rgba(2, 132, 199, 0.92)"
            line_width = 2.5 + 6.0 * rel_weight
        else:
            edge_color = "rgba(148, 163, 184, 0.22)"
            line_width = 1.2

        # Draw the head from the rendered spline, not the untrimmed endpoint.
        # That keeps its base centered on the shaft at every zoom level.
        target_gap = diameters[dst] / 2 + (3.5 if dst == focused_class else 1.5) / 2

        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines",
            meta={"edge_geometry": {
                "start_gap": diameters[src] / 2 + (3.5 if src == focused_class else 1.5) / 2,
                "end_gap": target_gap,
                "head_length": max(10.0, line_width * 1.6),
                "head_half_width": max(3.5, line_width * 0.8),
            }},
            line={
                "width": line_width,
                "color": edge_color,
                "shape": "spline",
                "smoothing": 1.0,
                "simplify": False,
            },
            hoverinfo="skip", showlegend=False,
        ))

        mid_pt = pts[int(len(pts) * 0.5)]
        src_name = names.get(src, str(src))
        dst_name = names.get(dst, str(dst))
        top_loc = str(getattr(row, "top_loc", "Unknown"))
        top_loc_pct = float(getattr(row, "top_loc_pct", 0.0))
        top_loc_cnt = int(getattr(row, "top_loc_cnt", 0))
        num_cams = int(getattr(row, "num_cams", 1))
        loc_src_supp = int(getattr(row, "top_loc_src_supp", 0))
        loc_cam_total = int(getattr(row, "top_loc_cam_total", 0))

        fig.add_trace(go.Scatter(
            x=[mid_pt[0]], y=[mid_pt[1]], mode="markers",
            marker={"size": 16, "color": "rgba(0,0,0,0.001)"},
            hoverinfo="none",
            customdata=[["edge", src, dst, int(row.count), float(row.rate), str(top_loc), float(top_loc_pct), top_loc_cnt, num_cams, loc_src_supp, loc_cam_total]],
            showlegend=False,
        ))

    # 2. Nodes
    node_x = []
    node_y = []
    node_sizes = []
    node_colors = []
    node_borders = []
    node_border_colors = []

    for class_id in ordered_nodes:
        x, y = positions[class_id]
        node_x.append(x)
        node_y.append(y)
        if class_id in stats.index:
            supp = int(stats.loc[class_id, "support"])
            errs = int(stats.loc[class_id, "errors"])
            rate = float(stats.loc[class_id, "error_rate"])
            fp = int(stats.loc[class_id, "falsely_predicted_count"])
        else:
            supp, errs, rate, fp = 0, 0, 0.0, 0

        size = diameters[class_id]
        node_sizes.append(size)
        node_colors.append(rate)

        name = names.get(class_id, str(class_id))
        is_focused = (class_id == focused_class)
        node_borders.append(3.5 if is_focused else 1.5)
        node_border_colors.append("#1d4ed8" if is_focused else "#ffffff")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker={
            "size": node_sizes,
            "color": node_colors,
            "colorscale": ERROR_RATE_COLORSCALE,
            "cmin": 0.0, "cmax": 1.0,
            "showscale": True,
            "colorbar": {
                "title": {
                    "text": "<b>Species Error Rate</b><br>&nbsp;",
                    "side": "top",
                    "font": {
                        "size": 11,
                        "family": "Inter, Arial, sans-serif",
                        "color": "#0f172a",
                    },
                },
                "tickformat": ".0%",
                "ticks": "outside",
                "ticklen": 4,
                "tickcolor": "#cbd5e1",
                "tickfont": {
                    "size": 10,
                    "family": "Inter, Arial, sans-serif",
                    "color": "#64748b",
                },
                "len": 0.38,
                "thickness": 16,
                "outlinewidth": 0,
                "y": 0.76,
                "x": 1.0,
            },
            "line": {"width": node_borders, "color": node_border_colors},
            "opacity": 1.0,
        },
        hoverinfo="none",
        customdata=[["node", int(nid)] for nid in ordered_nodes],
        showlegend=False,
    ))

    # 3. Community Cluster Labels (projected outward from cluster center)
    for idx, class_id in enumerate(ordered_nodes):
        x, y = positions[class_id]
        cx, cy = node_centers.get(class_id, (0.0, 0.0))
        dist_from_c = np.hypot(x - cx, y - cy)
        raw_name = names.get(class_id, str(class_id))
        display_name = raw_name if len(raw_name) <= 16 else raw_name[:14] + "…"

        node_r_data = (diameters.get(class_id, 16.0) / 2.0) * (units_per_pixel / 1.2)

        if dist_from_c < 1e-3:
            # Hub node placed at center of cluster: position label directly below node with clearance
            lx = x
            ly = y - (node_r_data + 0.22)
            x_anchor = "center"
            y_anchor = "top"
        else:
            angle = np.arctan2(y - cy, x - cx)
            # Alternate label offset distance to prevent adjacent text collisions in dense clusters
            label_dist = node_r_data + (0.16 if idx % 2 == 0 else 0.28)
            lx = x + label_dist * np.cos(angle)
            ly = y + label_dist * np.sin(angle)
            x_anchor = "left" if np.cos(angle) > 0.08 else ("right" if np.cos(angle) < -0.08 else "center")
            y_anchor = "bottom" if np.sin(angle) > 0.08 else ("top" if np.sin(angle) < -0.08 else "middle")
        is_focused = (class_id == focused_class)

        fig.add_annotation(
            x=lx, y=ly, text=display_name,
            showarrow=False,
            xanchor=x_anchor, yanchor=y_anchor,
            font={
                "size": 12.5 if is_focused else 11,
                "color": "#1d4ed8" if is_focused else "#1e293b",
                "weight": 700 if is_focused else 600,
                "family": "Inter, Arial, sans-serif",
            },
        )

    fig.update_layout(
        height=640,
        margin={"l": 35, "r": 130, "t": 25, "b": 25},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest", clickmode="event+select",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "size": 11},
        showlegend=False,
    )
    return fig


def confusion_matrix_figure(
    edges: pd.DataFrame,
    names: dict[int, str],
    focused_class: int | None,
) -> go.Figure:
    fig = go.Figure()
    if edges.empty:
        fig.add_annotation(text="No data for heatmap", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=640)
        return fig

    unique_species = sorted(
        list(set(edges["source"]).union(set(edges["target"]))),
        key=lambda s: edges.loc[(edges["source"] == s) | (edges["target"] == s), "count"].sum(),
        reverse=True,
    )[:25]

    spec_map = {sid: i for i, sid in enumerate(unique_species)}
    matrix = np.zeros((len(unique_species), len(unique_species)), dtype=int)

    edge_lookup = {}
    for row in edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        cnt = int(row.count)
        rate = float(getattr(row, "rate", 0.0))
        top_loc = str(getattr(row, "top_loc", "Unknown"))
        top_pct = float(getattr(row, "top_loc_pct", 0.0))
        top_cnt = int(getattr(row, "top_loc_cnt", 0))
        num_cams = int(getattr(row, "num_cams", 1))
        edge_lookup[(s, t)] = (cnt, rate, top_loc, top_pct, top_cnt, num_cams)
        if s in spec_map and t in spec_map:
            matrix[spec_map[s], spec_map[t]] = cnt

    labels = [names.get(sid, str(sid)) for sid in unique_species]
    labels_short = [l if len(l) <= 16 else l[:14] + "…" for l in labels]

    customdata = []
    for r in range(len(unique_species)):
        row_custom = []
        for c in range(len(unique_species)):
            s = unique_species[r]
            t = unique_species[c]
            if (s, t) in edge_lookup:
                cnt, rate, top_loc, top_pct, top_cnt, num_cams = edge_lookup[(s, t)]
            else:
                cnt = int(matrix[r, c])
                rate, top_loc, top_pct, top_cnt, num_cams = 0.0, "", 0.0, 0, 1
            row_custom.append(["edge", s, t, cnt, rate, top_loc, top_pct, top_cnt, num_cams])
        customdata.append(row_custom)

    fig.add_trace(go.Heatmap(
        z=matrix,
        x=labels_short,
        y=labels_short,
        colorscale=ERROR_RATE_COLORSCALE,
        hoverongaps=False,
        text=matrix,
        texttemplate="%{text}",
        customdata=customdata,
        hoverinfo="none",
        colorbar={
            "title": {
                "text": "<b>Errors</b><br>&nbsp;",
                "side": "top",
                "font": {
                    "size": 11,
                    "family": "Inter, Arial, sans-serif",
                    "color": "#0f172a",
                },
            },
            "ticks": "outside",
            "ticklen": 4,
            "tickcolor": "#cbd5e1",
            "tickfont": {
                "size": 10,
                "family": "Inter, Arial, sans-serif",
                "color": "#64748b",
            },
            "len": 0.65,
            "thickness": 18,
            "outlinewidth": 0,
            "y": 0.48,
        },
    ))

    fig.update_layout(
        height=640,
        margin={"l": 140, "r": 60, "t": 30, "b": 100},
        xaxis={"title": "Predicted Species (Model Output)", "tickangle": -45, "automargin": True},
        yaxis={"title": "True Species (Ground Truth)", "autorange": "reversed", "automargin": True},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "size": 11},
    )
    return fig


def horizontal_bar(
    labels: list[str],
    values: list[int | float],
    color: str,
    title_x: str,
    customdata: list | None = None,
    hovertemplate: str | None = None,
    hoverinfo: str | None = None,
    height: int = 240,
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color,
        customdata=customdata,
        hoverinfo=hoverinfo,
        hovertemplate=hovertemplate,
    ))
    fig.update_layout(
        height=height,
        margin={"l": 140, "r": 20, "t": 15, "b": 35},
        xaxis={"title": title_x, "gridcolor": "#f1f5f9", "zerolinecolor": "#e2e8f0"},
        yaxis={"automargin": True, "autorange": "reversed", "showgrid": False},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "size": 11},
    )
    return fig


def make_pair_tooltip_content(
    src_id: int, dst_id: int, count: int, rate: float, names: dict[int, str],
    top_loc: str = "", top_loc_pct: float = 0.0, top_loc_cnt: int = 0, num_cams: int = 1,
    show_close: bool = False, loc_src_supp: int = 0, loc_cam_total: int = 0,
    include_cam: bool = False,
):
    src_name = names.get(src_id, str(src_id))
    dst_name = names.get(dst_id, str(dst_id))

    cam_info = None
    if include_cam and top_loc and top_loc != "Unknown" and top_loc != "None":
        bias_badge = None
        if top_loc_pct >= 0.75:
            bias_badge = html.Div(
                f"High camera bias: {top_loc_pct:.0%} of this error is concentrated at Camera #{top_loc}",
                style={"fontSize": "10px", "fontWeight": "600", "color": "#c2410c", "marginTop": "3px"}
            )
        elif num_cams >= 3:
            bias_badge = html.Div(
                f"Widespread confusion: spread across {num_cams} camera traps",
                style={"fontSize": "10px", "fontWeight": "600", "color": "#0369a1", "marginTop": "3px"}
            )

        effective_top_cnt = top_loc_cnt if top_loc_cnt > 0 else int(round(count * top_loc_pct))
        loc_supp_str = f" out of {loc_src_supp:,} true {src_name} sightings" if loc_src_supp > 0 else ""
        err_rate_str = f" ({effective_top_cnt / loc_src_supp:.1%} error rate)" if loc_src_supp > 0 else ""
        cam_tot_str = f" · {loc_cam_total:,} total camera images" if loc_cam_total > 0 else ""

        cam_info = html.Div([
            html.Div([
                html.Span(f"Top Location: Camera #{top_loc}", style={"fontWeight": "700", "color": "#0f172a"}),
                html.Span(f" ({top_loc_pct:.0%} of this pair)", style={"color": "#64748b", "fontWeight": "500", "marginLeft": "4px"}),
            ], style={"fontSize": "10px"}),
            html.Div(
                f"{effective_top_cnt:,} errors{loc_supp_str}{err_rate_str}{cam_tot_str}",
                style={"fontSize": "10px", "fontWeight": "600", "color": "#dc2626" if (loc_src_supp > 0 and effective_top_cnt / loc_src_supp > 0.4) else "#334155", "marginTop": "2px"}
            ),
            html.Div(f"Camera Spread: Occurs across {num_cams} camera trap{'s' if num_cams != 1 else ''} in split", style={"fontSize": "10px", "color": "#64748b", "marginTop": "1px"}),
            bias_badge if bias_badge else html.Div(),
        ], style={"marginTop": "6px", "paddingTop": "6px", "borderTop": "1px solid #f1f5f9"})

    return html.Div([
        html.Div([
            html.Div([
                html.Div(f"True: {src_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                make_diashow(get_class_images(src_id, max_images=1), border_color="#fed7aa"),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
            html.Div([
                html.Div(f"Pred: {dst_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                make_diashow(get_class_images(dst_id, max_images=1), border_color="#bfdbfe"),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
        html.Div(f"Misclassifications: {count:,} ({rate:.1%} of true {src_name})", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "whiteSpace": "normal", "wordBreak": "break-word"}),
        cam_info if cam_info else html.Div(),
    ], style={"position": "relative", "padding": "10px", "width": "290px", "maxWidth": "100%", "boxSizing": "border-box", "whiteSpace": "normal", "overflowX": "hidden", "background": "white", "borderRadius": "8px"})


def make_sink_tooltip_content(
    class_id: int, fp_count: int, support: int, names: dict[int, str],
    top_sink_cam: str = "", top_sink_cam_cnt: int = 0,
    cam_total: int = 0, cam_true_supp: int = 0, cam_errs: int = 0
):
    name = names.get(class_id, str(class_id))

    cam_info = None
    if top_sink_cam and top_sink_cam != "N/A" and top_sink_cam_cnt > 0:
        cam_tot_str = f" out of {cam_total:,} total camera images" if cam_total > 0 else ""
        supp_str = (
            f"{cam_true_supp:,} true {name} sightings at this camera"
            if cam_true_supp > 0
            else f"0 true {name} sightings at this camera (pure false alarm)"
        )
        cam_info = html.Div([
            html.Div(f"Most false alarms at Camera #{top_sink_cam}:", style={"fontSize": "10px", "color": "#0f172a", "fontWeight": "700"}),
            html.Div(
                f"{top_sink_cam_cnt:,} false alarms{cam_tot_str}",
                style={"fontSize": "10.5px", "color": "#0284c7", "fontWeight": "600", "marginTop": "1px", "whiteSpace": "normal", "wordBreak": "break-word"}
            ),
            html.Div(supp_str, style={"fontSize": "9.5px", "color": "#64748b", "marginTop": "2px", "whiteSpace": "normal", "wordBreak": "break-word"}),
        ], style={"marginTop": "6px", "paddingTop": "6px", "borderTop": "1px solid #f1f5f9"})

    return html.Div([
        html.Div(name, style={"fontWeight": "bold", "fontSize": "12px", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "normal", "wordBreak": "break-word"}),
        make_diashow(get_class_images(class_id, max_images=1), border_color="#bfdbfe"),
        html.Div(f"Falsely predicted for other classes: {fp_count:,} times", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "marginTop": "6px", "marginBottom": "2px", "whiteSpace": "normal", "wordBreak": "break-word"}),
        html.Div(f"Ground-truth support: {support:,} true {name} images in split", style={"fontSize": "10px", "color": "#64748b", "whiteSpace": "normal", "wordBreak": "break-word"}),
        cam_info if cam_info else html.Div(),
    ], style={"padding": "10px", "width": "250px", "maxWidth": "100%", "boxSizing": "border-box", "whiteSpace": "normal", "overflowX": "hidden", "background": "white", "borderRadius": "8px"})


def make_cam_tooltip_content(
    cam_id: str, err_cnt: int, err_pct: float, cam_total: int, cam_errs: int, cam_err_rate: float,
    top_pred_id: int, top_pred_cnt: int, focus_id: int, names: dict[int, str],
    focus_cam_supp: int = 0
):
    focus_name = names.get(focus_id, str(focus_id))
    pred_name = names.get(top_pred_id, str(top_pred_id))

    species_share = (err_cnt / cam_errs) if cam_errs > 0 else 0.0

    supp_line = None
    if focus_cam_supp > 0:
        loc_err_rate = err_cnt / focus_cam_supp
        supp_line = html.Div(
            f"{err_cnt:,} errors out of {focus_cam_supp:,} true {focus_name} sightings at this camera ({loc_err_rate:.1%} error rate)",
            style={"fontSize": "10px", "fontWeight": "600", "color": "#dc2626" if loc_err_rate > 0.4 else "#334155", "marginTop": "2px"}
        )
    else:
        supp_line = html.Div(
            f"0 true {focus_name} sightings at this camera (all {err_cnt:,} are other species mistaken for {focus_name})",
            style={"fontSize": "10px", "color": "#64748b", "marginTop": "2px"}
        )

    preview = None
    if top_pred_id in names and top_pred_cnt > 0:
        preview = html.Div([
            html.Div(f"Top Confusion at Camera #{cam_id}:", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "marginBottom": "4px", "marginTop": "6px"}),
            html.Div([
                html.Div([
                    html.Div(f"True: {focus_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                    make_diashow(get_class_images(focus_id, max_images=1), border_color="#fed7aa"),
                ], style={"flex": "1", "minWidth": "0"}),
                html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                html.Div([
                    html.Div(f"Pred: {pred_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                    make_diashow(get_class_images(top_pred_id, max_images=1), border_color="#bfdbfe"),
                ], style={"flex": "1", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "4px"}),
            html.Div(f"Model confused this pair {top_pred_cnt:,} times at this location.", style={"fontSize": "10px", "color": "#64748b"}),
        ], style={"paddingTop": "6px", "borderTop": "1px solid #f1f5f9"})

    return html.Div([
        html.Div([
            html.Span(f"Camera #{cam_id}", style={"fontWeight": "bold", "fontSize": "13px", "color": "#7c3aed"}),
            html.Span("Trap Location", style={"fontSize": "10px", "background": "#f3e8ff", "color": "#6b21a8", "padding": "1px 6px", "borderRadius": "4px", "marginLeft": "6px", "fontWeight": "600"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.Div(f"Camera Total: {cam_total:,} images in split · {cam_err_rate:.1%} error rate ({cam_errs:,} errors)", style={"fontSize": "10px", "color": "#475569", "marginBottom": "3px"}),
        html.Div(f"{focus_name}: {err_cnt:,} confusions ({err_pct:.1%} of all {focus_name} mistakes)", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "marginBottom": "2px"}),
        supp_line if supp_line else html.Div(),
        html.Div(f"Accounts for {species_share:.1%} of all mistakes at this camera", style={"fontSize": "10px", "color": "#64748b"}),
        preview if preview else html.Div(),
    ], style={"padding": "10px", "width": "290px", "maxWidth": "100%", "boxSizing": "border-box", "whiteSpace": "normal", "overflowX": "hidden", "background": "white", "borderRadius": "8px"})


# -----------------------------------------------------------------------------
# DASH APPLICATION
# -----------------------------------------------------------------------------

SUPPORT_LEGEND_STYLE = {
    "position": "absolute",
    "top": "315px",
    "right": "12px",
    "zIndex": 5,
    "pointerEvents": "none",
    "background": "rgba(255, 255, 255, 0.90)",
    "padding": "8px 10px",
    "borderRadius": "8px",
    "border": "1px solid #e2e8f0",
    "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
}


def build_support_legend(stats: pd.DataFrame, visible_nodes: list[int]) -> html.Div:
    if not visible_nodes or stats.empty:
        return html.Div()

    visible_supports = stats.loc[stats.index.isin(visible_nodes), "support"] if not stats.empty else pd.Series([1.0])
    max_support = float(visible_supports.max() if not visible_supports.empty else 1.0)
    min_support = max(float(visible_supports.min() if not visible_supports.empty else 1.0), 1.0)

    if max_support <= 0:
        max_support = 1.0

    if max_support >= min_support * 2.5:
        s_high = max_support
        mid_sqrt = 0.5 * (1.0 + np.sqrt(min_support / max_support))
        s_mid = max((mid_sqrt ** 2) * max_support, min_support * 1.2)
        s_low = min_support
        support_points = [
            (s_high, 16.0 + 20.0),
            (s_mid, 16.0 + 20.0 * np.sqrt(s_mid / max_support)),
            (s_low, 16.0 + 20.0 * np.sqrt(s_low / max_support)),
        ]
    elif max_support >= min_support * 1.3:
        support_points = [
            (max_support, 16.0 + 20.0),
            (min_support, 16.0 + 20.0 * np.sqrt(min_support / max_support)),
        ]
    else:
        support_points = [
            (max_support, 16.0 + 20.0),
        ]

    rows = []
    for supp_val, sz_val in support_points:
        if supp_val >= 1000:
            rounded_supp = int(round(supp_val / 50.0)) * 50
        elif supp_val >= 100:
            rounded_supp = int(round(supp_val / 10.0)) * 10
        else:
            rounded_supp = max(1, int(round(supp_val)))

        d = int(round(sz_val))
        rows.append(html.Div([
            html.Div([
                html.Div(style={
                    "width": f"{d}px",
                    "height": f"{d}px",
                    "borderRadius": "50%",
                    "background": "#f8fafc",
                    "border": "1.5px solid #64748b",
                    "boxSizing": "border-box",
                }),
            ], style={"width": "40px", "height": "38px", "display": "flex", "alignItems": "center", "justifyContent": "center", "flexShrink": "0"}),
            html.Span(
                f"{rounded_supp:,} images",
                style={"fontSize": "10.5px", "color": "#475569", "fontWeight": "500", "marginLeft": "6px", "whiteSpace": "nowrap"}
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"}))

    return html.Div([
        html.Div("Image Count (Support)", style={"fontWeight": "700", "fontSize": "11px", "color": "#0f172a", "marginBottom": "6px"}),
        html.Div(rows),
    ])


def make_html_bar_chart(
    items: list[tuple[str, int | float, html.Div]],
    color: str,
    title_x: str,
    max_val: int | float | None = None,
) -> html.Div:
    if not items:
        return html.Div(
            "No data available for current filters.",
            style={"fontSize": "11px", "color": "#94a3b8", "fontStyle": "italic", "padding": "8px 0"}
        )

    if max_val is None:
        max_val = max(val for _, val, _ in items) if items else 1
    if max_val <= 0:
        max_val = 1

    rows = []
    for label, val, popup in items:
        pct = (val / max_val) * 100 if max_val > 0 else 0
        rows.append(html.Div([
            # Text label container (hover trigger)
            html.Div([
                html.Span(
                    label,
                    title=label,
                    className="sidebar-text-label",
                    style={
                        "display": "block",
                        "width": "150px",
                        "fontSize": "11px",
                        "color": "#334155",
                        "fontWeight": "600",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "textAlign": "right",
                        "paddingRight": "8px",
                        "cursor": "pointer",
                        "lineHeight": "20px",
                    }
                ),
                html.Div(popup, className="sidebar-text-popup", style={"display": "none"}),
            ], className="sidebar-text-hover-item", style={"flexShrink": "0", "width": "150px", "position": "relative"}),

            # Bar track and colored bar (non-interactive, no tooltip occlusion)
            html.Div([
                html.Div(style={
                    "width": f"{pct:.1f}%",
                    "background": color,
                    "height": "14px",
                    "borderRadius": "3px",
                    "minWidth": "3px" if val > 0 else "0px",
                    "transition": "width 0.2s ease",
                }),
            ], style={
                "flex": "1",
                "display": "flex",
                "alignItems": "center",
                "minWidth": "0",
                "height": "18px",
                "background": "#f8fafc",
                "borderRadius": "3px",
                "padding": "0 2px",
            }),

            # Value count
            html.Span(
                f"{int(val):,}",
                style={
                    "fontSize": "11px",
                    "color": "#475569",
                    "fontWeight": "600",
                    "marginLeft": "8px",
                    "minWidth": "35px",
                    "textAlign": "right",
                    "flexShrink": "0",
                }
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"}))

    # Axis scale footer
    axis_footer = html.Div([
        html.Div(style={"width": "150px", "flexShrink": "0"}),
        html.Div([
            html.Div(
                title_x,
                style={
                    "fontSize": "10px",
                    "color": "#64748b",
                    "textAlign": "center",
                    "borderTop": "1px solid #e2e8f0",
                    "paddingTop": "4px",
                    "marginTop": "2px",
                    "fontWeight": "500",
                },
            ),
        ], style={"flex": "1"}),
        html.Div(style={"width": "35px", "marginLeft": "8px", "flexShrink": "0"}),
    ], style={"display": "flex", "alignItems": "flex-start", "marginTop": "2px"})

    return html.Div(rows + [axis_footer], style={"padding": "4px 0 10px 0"})


def build_side_panel(
    subset: pd.DataFrame,
    stats: pd.DataFrame,
    edges: pd.DataFrame,
    mistakes: pd.DataFrame,
    names: dict[int, str],
    focus: int | None,
    split: str = "test",
) -> html.Div:
    if focus is None:
        split_desc = SPLIT_DESCRIPTIONS.get(split, split)
        top_pairs = edges.nlargest(6, "count")
        top_pair_items = []
        for r in top_pairs.itertuples():
            src_name = names.get(int(r.source), str(r.source))
            dst_name = names.get(int(r.target), str(r.target))
            label = f"{src_name} → {dst_name}"
            cnt = int(r.count)
            rate = float(r.rate)
            top_loc = str(getattr(r, "top_loc", "Unknown"))
            top_loc_pct = float(getattr(r, "top_loc_pct", 0.0))
            top_loc_cnt = int(getattr(r, "top_loc_cnt", 0))
            num_cams = int(getattr(r, "num_cams", 1))
            loc_src_supp = int(getattr(r, "top_loc_src_supp", 0))
            loc_cam_total = int(getattr(r, "top_loc_cam_total", 0))
            popup = make_pair_tooltip_content(int(r.source), int(r.target), cnt, rate, names, top_loc, top_loc_pct, top_loc_cnt, num_cams, loc_src_supp=loc_src_supp, loc_cam_total=loc_cam_total, include_cam=True)
            top_pair_items.append((label, cnt, popup))
        chart_top_pairs = make_html_bar_chart(top_pair_items, "#ea580c", "Misclassifications")

        sinks = stats.nlargest(6, "falsely_predicted_count")
        sink_items = []
        for idx in sinks.index:
            label = names.get(int(idx), str(idx))
            cnt = int(sinks.loc[idx, "falsely_predicted_count"])
            supp = int(sinks.loc[idx, "support"] if "support" in sinks.columns else 0)
            top_cam = str(sinks.loc[idx, "top_sink_cam"]) if "top_sink_cam" in sinks.columns else ""
            top_cam_cnt = int(sinks.loc[idx, "top_sink_cam_cnt"]) if "top_sink_cam_cnt" in sinks.columns else 0
            cam_total = int(sinks.loc[idx, "top_sink_cam_total"]) if "top_sink_cam_total" in sinks.columns else 0
            cam_true_supp = int(sinks.loc[idx, "top_sink_cam_supp"]) if "top_sink_cam_supp" in sinks.columns else 0
            cam_errs = int(sinks.loc[idx, "top_sink_cam_errs"]) if "top_sink_cam_errs" in sinks.columns else 0
            popup = make_sink_tooltip_content(int(idx), cnt, supp, names, top_cam, top_cam_cnt, cam_total, cam_true_supp, cam_errs)
            sink_items.append((label, cnt, popup))
        chart_sinks = make_html_bar_chart(sink_items, "#0284c7", "Falsely Predicted Count")

        return html.Div([
            html.Div([
                html.H3("Split-Wide Error Patterns", style={"margin": "0 0 4px 0", "fontSize": "16px"}),
                html.P(f"Global insights across {len(subset):,} images in {split_desc}. Select any species to inspect individual confusions.",
                       style={"fontSize": "12px", "color": "#64748b", "margin": "0 0 12px 0"}),
            ]),
            html.H4("Top Confusion Pairs", style={"margin": "0", "fontSize": "13px", "color": "#334155"}),
            chart_top_pairs,
            html.H4("Top 'Sink' Classes (Most Falsely Over-Predicted)", style={"margin": "10px 0 0 0", "fontSize": "13px", "color": "#334155"}),
            chart_sinks,
        ])

    else:
        focus_name = names.get(focus, str(focus))
        focus_supp = int(stats.loc[focus, "support"]) if focus in stats.index else 0
        focus_errs = int(stats.loc[focus, "errors"]) if focus in stats.index else 0
        focus_rate = float(stats.loc[focus, "error_rate"]) if focus in stats.index else 0.0

        out_edges = edges.loc[edges["source"] == focus].nlargest(6, "count")
        out_items = []
        for r in out_edges.itertuples():
            dst_name = names.get(int(r.target), str(r.target))
            cnt = int(r.count)
            rate = float(r.rate)
            top_loc = str(getattr(r, "top_loc", "Unknown"))
            top_loc_pct = float(getattr(r, "top_loc_pct", 0.0))
            top_loc_cnt = int(getattr(r, "top_loc_cnt", 0))
            num_cams = int(getattr(r, "num_cams", 1))
            loc_src_supp = int(getattr(r, "top_loc_src_supp", 0))
            loc_cam_total = int(getattr(r, "top_loc_cam_total", 0))
            popup = make_pair_tooltip_content(int(focus), int(r.target), cnt, rate, names, top_loc, top_loc_pct, top_loc_cnt, num_cams, loc_src_supp=loc_src_supp, loc_cam_total=loc_cam_total, include_cam=True)
            out_items.append((dst_name, cnt, popup))
        chart_out = make_html_bar_chart(out_items, "#ea580c", "Outgoing Errors")

        in_edges = edges.loc[edges["target"] == focus].nlargest(6, "count")
        in_items = []
        for r in in_edges.itertuples():
            src_name = names.get(int(r.source), str(r.source))
            cnt = int(r.count)
            rate = float(getattr(r, "rate", 0.0))
            top_loc = str(getattr(r, "top_loc", "Unknown"))
            top_loc_pct = float(getattr(r, "top_loc_pct", 0.0))
            top_loc_cnt = int(getattr(r, "top_loc_cnt", 0))
            num_cams = int(getattr(r, "num_cams", 1))
            loc_src_supp = int(getattr(r, "top_loc_src_supp", 0))
            loc_cam_total = int(getattr(r, "top_loc_cam_total", 0))
            popup = make_pair_tooltip_content(int(r.source), int(focus), cnt, rate, names, top_loc, top_loc_pct, top_loc_cnt, num_cams, loc_src_supp=loc_src_supp, loc_cam_total=loc_cam_total, include_cam=True)
            in_items.append((src_name, cnt, popup))
        chart_in = make_html_bar_chart(in_items, "#0284c7", "Incoming False Predictions")

        focus_mistakes = mistakes.loc[mistakes["label"] == focus]
        cam_counts = focus_mistakes["location"].value_counts().head(5)
        cam_stats = mistakes.attrs.get("cam_stats", {})
        spec_cam_map = mistakes.attrs.get("spec_cam_map", {})
        cam_items = []
        for c in cam_counts.index:
            c_cnt = int(cam_counts[c])
            c_pct = c_cnt / focus_errs if focus_errs > 0 else 0.0
            cs = cam_stats.get(str(c), {"total": 0, "errors": 0, "error_rate": 0.0})
            cam_total = int(cs.get("total", 0))
            cam_errs = int(cs.get("errors", 0))
            cam_err_rate = float(cs.get("error_rate", 0.0))
            focus_cam_supp = int(spec_cam_map.get((int(focus), str(c)), 0))
            cam_sub = focus_mistakes.loc[focus_mistakes["location"] == c]
            if not cam_sub.empty:
                top_pred = int(cam_sub["prediction"].value_counts().index[0])
                top_pred_cnt = int(cam_sub["prediction"].value_counts().iloc[0])
            else:
                top_pred = 0
                top_pred_cnt = 0
            popup = make_cam_tooltip_content(
                str(c), c_cnt, c_pct, cam_total, cam_errs, cam_err_rate,
                top_pred, top_pred_cnt, int(focus), names, focus_cam_supp=focus_cam_supp
            )
            cam_items.append((f"Camera #{c}", c_cnt, popup))
        chart_cam = make_html_bar_chart(cam_items, "#8b5cf6", "Errors at Camera")

        focus_imgs = get_class_images(focus)
        species_diashow = (
            make_diashow(focus_imgs, width="240px", border_color="#cbd5e1")
            if focus_imgs
            else None
        )

        header_block = html.Div([
            html.H3(focus_name, style={"margin": "0 0 4px 0", "fontSize": "18px", "color": "#0f172a", "wordBreak": "break-word"}),
            html.Div(
                f"{focus_supp:,} true images | {focus_errs:,} errors | {focus_rate:.1%} error rate",
                style={
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "color": "#ea580c" if focus_rate > 0.5 else "#334155",
                    "lineHeight": "1.35",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                [
                    html.Div(f"Representative photos of {focus_name}:", style={"fontSize": "10.5px", "color": "#64748b", "fontStyle": "italic", "marginBottom": "6px", "textAlign": "center"}),
                    html.Div(species_diashow, style={"display": "flex", "justifyContent": "center"}),
                ],
                style={"marginTop": "6px", "marginBottom": "4px"}
            ) if species_diashow else html.Div(),
        ], style={
            "marginBottom": "14px",
            "paddingBottom": "10px",
            "borderBottom": "1px solid #f1f5f9",
        })

        cam_section = None
        if len(cam_counts) > 0:
            top_cam = str(cam_counts.index[0])
            top_cam_cnt = int(cam_counts.iloc[0])
            top_cam_supp = int(spec_cam_map.get((int(focus), top_cam), 0))
            top_cam_tot = int(cam_stats.get(top_cam, {}).get("total", 0))
            cam_mistake_imgs, cam_mistake_captions = get_mistake_images_for_species_cam(focus, top_cam)

            supp_text = f" out of {top_cam_supp:,} true sightings ({top_cam_cnt / top_cam_supp:.1%} error rate)" if top_cam_supp > 0 else ""
            tot_text = f" · {top_cam_tot:,} total camera images" if top_cam_tot > 0 else ""

            cam_section = html.Div([
                html.H4(
                    f"Worst Camera Trap (Camera #{top_cam})",
                    style={"margin": "0 0 2px 0", "fontSize": "13px", "fontWeight": "700", "color": "#7c3aed"}
                ),
                html.Div(f"{top_cam_cnt:,} errors{supp_text}{tot_text}", style={"fontSize": "11px", "fontWeight": "600", "color": "#dc2626", "marginBottom": "4px"}),
                html.Div(f"Actual misclassified photos of {focus_name} captured at Camera #{top_cam}:", style={"fontSize": "10.5px", "color": "#475569", "fontStyle": "italic", "marginBottom": "6px"}),
                html.Div(
                    make_diashow(cam_mistake_imgs, captions=cam_mistake_captions, width="240px", border_color="#ddd6fe")
                    if cam_mistake_imgs
                    else html.Div("No photos available for this camera.", style={"fontSize": "11px", "color": "#94a3b8"}),
                    style={"display": "flex", "justifyContent": "center"}
                ),
            ], style={"background": "#faf5ff", "padding": "12px", "borderRadius": "8px", "border": "1px solid #e9d5ff", "marginTop": "10px"})

        return html.Div([
            header_block,
            html.H4([
                html.Span("→ ", style={"color": "#ea580c", "fontWeight": "bold", "fontSize": "15px"}),
                f"Top Outgoing Errors ({focus_name} mistaken for ...)",
            ], style={"margin": "0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            chart_out,
            html.H4([
                html.Span("← ", style={"color": "#0284c7", "fontWeight": "bold", "fontSize": "15px"}),
                f"Top Incoming False Predictions (... mistaken as {focus_name})",
            ], style={"margin": "12px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            chart_in,
            html.H4("Top Camera Locations by Error Count", style={"margin": "12px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            chart_cam,
            cam_section if cam_section else html.Div(),
        ])


KPI_CARD_STYLE = {
    "background": "#ffffff", "padding": "12px 18px", "borderRadius": "10px",
    "border": "1px solid #e2e8f0", "flex": "1 1 180px", "minWidth": "150px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}
KPI_TITLE_STYLE = {"fontSize": "11px", "color": "#64748b", "fontWeight": "600", "textTransform": "uppercase"}
KPI_VAL_STYLE = {"fontSize": "18px", "fontWeight": "700", "color": "#0f172a", "marginTop": "2px"}


def build_kpis(subset: pd.DataFrame, stats: pd.DataFrame, edges: pd.DataFrame, filtered: pd.DataFrame, names: dict[int, str]) -> list:
    total_images = len(subset)
    total_errors = int((subset["label"] != subset["prediction"]).sum())
    correct_count = total_images - total_errors
    acc = correct_count / total_images if total_images else 0.0
    num_cams = subset["location"].nunique() if "location" in subset.columns else 0
    num_classes = subset["label"].nunique() if "label" in subset.columns else 0

    if not subset.empty:
        y_true = subset["label"].values
        y_pred = subset["prediction"].values
        labels = np.unique(y_true)
        f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    else:
        f1_macro = 0.0

    top_pair_text = "N/A"
    top_pair_popup = None
    if not edges.empty:
        top_p = edges.nlargest(1, "count").iloc[0]
        src_id = int(top_p["source"])
        dst_id = int(top_p["target"])
        cnt_val = int(top_p["count"])
        src_name = names.get(src_id, str(src_id))
        dst_name = names.get(dst_id, str(dst_id))
        src_supp = int(stats.loc[src_id, "support"]) if (not stats.empty and src_id in stats.index and "support" in stats.columns) else int(top_p.get("source_support", 0))
        dst_supp = int(stats.loc[dst_id, "support"]) if (not stats.empty and dst_id in stats.index and "support" in stats.columns) else 0
        rate_val = float(top_p["rate"]) if "rate" in top_p else (cnt_val / src_supp if src_supp else 0.0)
        top_pair_text = f"{src_name} → {dst_name} ({cnt_val:,} errors)"

        top_pair_popup = html.Div([
            html.Div("Visual Comparison (Split's #1 Misclassification):", style={"fontWeight": "700", "fontSize": "11px", "color": "#0f172a", "marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Div(f"True: {src_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                    html.Div(f"{src_supp:,} true images in split", style={"fontSize": "9.5px", "color": "#64748b", "marginBottom": "3px"}),
                    make_diashow(get_class_images(src_id, max_images=1), border_color="#fed7aa"),
                ], style={"flex": "1", "minWidth": "0"}),
                html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                html.Div([
                    html.Div(f"Mistaken for: {dst_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                    html.Div(f"{dst_supp:,} true images in split" if dst_supp else "Predicted class", style={"fontSize": "9.5px", "color": "#64748b", "marginBottom": "3px"}),
                    make_diashow(get_class_images(dst_id, max_images=1), border_color="#bfdbfe"),
                ], style={"flex": "1", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
            html.Div(
                f"Model misclassified {src_name} as {dst_name} {cnt_val:,} times in this split ({rate_val:.1%} of all {src_supp:,} true {src_name} images).",
                style={"fontSize": "10px", "color": "#64748b", "lineHeight": "1.35"},
            ),
        ], className="kpi-popup", style={"width": "320px", "maxWidth": "100%", "boxSizing": "border-box"})

    top_sink_text = "N/A"
    top_sink_popup = None
    if not stats.empty and "falsely_predicted_count" in stats.columns:
        sink_id = int(stats["falsely_predicted_count"].idxmax())
        sink_cnt = int(stats.loc[sink_id, "falsely_predicted_count"])
        sink_supp = int(stats.loc[sink_id, "support"]) if "support" in stats.columns else 0
        sink_name = names.get(sink_id, str(sink_id))
        top_sink_text = f"{sink_name} ({sink_cnt:,} times)"

        top_sink_popup = html.Div([
            html.Div("Most Falsely Over-Predicted Species:", style={"fontWeight": "700", "fontSize": "11px", "color": "#0f172a", "marginBottom": "4px"}),
            html.Div(sink_name, style={"fontSize": "11px", "fontWeight": "700", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
            make_diashow(get_class_images(sink_id, max_images=1), border_color="#bfdbfe"),
            html.Div(f"Falsely predicted {sink_cnt:,} times for other classes ({sink_supp:,} true {sink_name} images in split).", style={"fontSize": "10px", "color": "#64748b", "lineHeight": "1.35", "marginTop": "6px"}),
        ], className="kpi-popup", style={"width": "240px", "maxWidth": "100%", "boxSizing": "border-box"})

    return [
        html.Div([
            html.Div("Total Split Images", style=KPI_TITLE_STYLE),
            html.Div(f"{total_images:,}", style=KPI_VAL_STYLE),
            html.Div(
                f"{num_cams} cameras · {num_classes} species" if num_cams else f"{num_classes} species",
                style={"fontSize": "11px", "color": "#64748b", "marginTop": "2px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
            ),
        ], style=KPI_CARD_STYLE),
        html.Div([
            html.Div("Split Accuracy & Macro F1", style=KPI_TITLE_STYLE),
            html.Div([
                html.Span(f"{acc:.1%}", style={"color": "#0f172a"}),
                html.Span(" Acc", style={"fontSize": "11px", "color": "#64748b", "fontWeight": "600", "marginRight": "8px"}),
                html.Span("·", style={"color": "#94a3b8", "fontWeight": "bold", "marginRight": "8px"}),
                html.Span(f"{f1_macro:.1%}", style={"color": "#0f172a"}),
                html.Span(" F1", style={"fontSize": "11px", "color": "#64748b", "fontWeight": "600"}),
            ], style=KPI_VAL_STYLE),
            html.Div(
                f"{correct_count:,} / {total_images:,} correct",
                style={"fontSize": "11px", "color": "#64748b", "marginTop": "2px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
            ),
        ], style=KPI_CARD_STYLE),
        html.Div([
            html.Div("Top Confusion Pair", style=KPI_TITLE_STYLE),
            html.Div(top_pair_text, style={**KPI_VAL_STYLE, "fontSize": "14px", "color": "#ea580c"}),
            top_pair_popup if top_pair_popup else None,
        ], className="kpi-card-hover" if top_pair_popup else "", style=KPI_CARD_STYLE),
        html.Div([
            html.Div("Top Over-Predicted Sink", style=KPI_TITLE_STYLE),
            html.Div(top_sink_text, style={**KPI_VAL_STYLE, "fontSize": "14px", "color": "#0284c7"}),
            top_sink_popup if top_sink_popup else None,
        ], className="kpi-card-hover" if top_sink_popup else "", style=KPI_CARD_STYLE),
    ]


def create_app(data: pd.DataFrame):
    data = data.copy()
    if "split" in data.columns:
        data["split"] = data["split"].astype(str).replace({
            "id_test": "test",
            "id_val": "val",
        })

    global CAMERA_IMAGE_MAP, CLASS_MULTI_IMAGE_MAP, CAMERA_MULTI_IMAGE_MAP
    global MISTAKE_SPEC_CAM_MAP, MISTAKE_PAIR_CAM_MAP, MISTAKE_SINK_CAM_MAP
    if "location" in data.columns and "filename" in data.columns:
        CAMERA_IMAGE_MAP = data.drop_duplicates("location").set_index("location")["filename"].to_dict()

        # Build multi-image map for cameras (up to 5 per camera)
        cam_map: dict[str, list[str]] = {}
        cam_grp = data.drop_duplicates(subset=["location", "filename"])
        for loc, grp in cam_grp.groupby("location"):
            loc_str = str(loc)
            primary = CAMERA_IMAGE_MAP.get(loc_str)
            fns = [primary] if (primary and (DATA_DIR / primary).is_file()) else []
            for fn in grp["filename"]:
                if fn not in fns and (DATA_DIR / fn).is_file():
                    fns.append(fn)
                if len(fns) >= 5:
                    break
            cam_map[loc_str] = fns
        CAMERA_MULTI_IMAGE_MAP = cam_map

    rep_map = load_image_map()
    class_map: dict[int, list[str]] = {}
    if "label" in data.columns and "filename" in data.columns:
        # Build multi-image map for species (up to 5 per class across diverse locations)
        loc_subset = ["label", "location", "filename"] if "location" in data.columns else ["label", "filename"]
        class_grp = data.drop_duplicates(subset=loc_subset)
        for cid, grp in class_grp.groupby("label"):
            cid_int = int(cid)
            primary = rep_map.get(cid_int)
            fns = [primary] if (primary and (DATA_DIR / primary).is_file()) else []
            for fn in grp["filename"]:
                if fn not in fns and (DATA_DIR / fn).is_file():
                    fns.append(fn)
                if len(fns) >= 5:
                    break
            class_map[cid_int] = fns
    CLASS_MULTI_IMAGE_MAP = class_map

    names = class_names(data)
    unique_splits = set(data["split"].unique())
    splits = [s for s in SPLIT_DESCRIPTIONS if s in unique_splits]
    if not splits:
        splits = sorted(unique_splits)
    default_split = "test" if "test" in splits else splits[0]

    # Pre-index actual misclassified images with model predictions for each confounding camera context
    all_mistakes = data.loc[data["label"] != data["prediction"]]
    spec_cam_dict: dict[tuple[int, str], list[dict]] = {}
    if "label" in all_mistakes.columns and "location" in all_mistakes.columns and "filename" in all_mistakes.columns:
        for (l, loc), grp in all_mistakes.groupby(["label", "location"]):
            items = []
            for r in grp.itertuples():
                fn = r.filename
                if (DATA_DIR / fn).is_file():
                    pred_id = int(r.prediction)
                    pred_name = names.get(pred_id, getattr(r, "prediction_name", str(pred_id)))
                    conf = float(getattr(r, "confidence", 0.0))
                    items.append({
                        "filename": fn,
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "confidence": conf,
                    })
                if len(items) >= 8:
                    break
            spec_cam_dict[(int(l), str(loc))] = items
    MISTAKE_SPEC_CAM_MAP = spec_cam_dict

    pair_cam_dict: dict[tuple[int, int, str], list[dict]] = {}
    if "label" in all_mistakes.columns and "prediction" in all_mistakes.columns and "location" in all_mistakes.columns and "filename" in all_mistakes.columns:
        for (s, t, loc), grp in all_mistakes.groupby(["label", "prediction", "location"]):
            items = []
            for r in grp.itertuples():
                fn = r.filename
                if (DATA_DIR / fn).is_file():
                    pred_id = int(r.prediction)
                    pred_name = names.get(pred_id, getattr(r, "prediction_name", str(pred_id)))
                    conf = float(getattr(r, "confidence", 0.0))
                    items.append({
                        "filename": fn,
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "confidence": conf,
                    })
                if len(items) >= 8:
                    break
            pair_cam_dict[(int(s), int(t), str(loc))] = items
    MISTAKE_PAIR_CAM_MAP = pair_cam_dict

    sink_cam_dict: dict[tuple[int, str], list[dict]] = {}
    if "prediction" in all_mistakes.columns and "location" in all_mistakes.columns and "filename" in all_mistakes.columns:
        for (p, loc), grp in all_mistakes.groupby(["prediction", "location"]):
            items = []
            for r in grp.itertuples():
                fn = r.filename
                if (DATA_DIR / fn).is_file():
                    pred_id = int(r.prediction)
                    pred_name = names.get(pred_id, getattr(r, "prediction_name", str(pred_id)))
                    conf = float(getattr(r, "confidence", 0.0))
                    items.append({
                        "filename": fn,
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "confidence": conf,
                    })
                if len(items) >= 8:
                    break
            sink_cam_dict[(int(p), str(loc))] = items
    MISTAKE_SINK_CAM_MAP = sink_cam_dict

    @lru_cache(maxsize=10)
    def get_split(split: str):
        subset = data.loc[data["split"] == split]
        stats, edges, mistakes = summarize_split(subset)
        return subset, stats, edges, mistakes

    # Pre-render initial figures and side content so page is never empty
    init_subset, init_stats, init_edges, init_mistakes = get_split(default_split)
    init_filtered = visible_edges(init_edges, names, 30, 5, 1.0, 25, "rate", None)
    init_fig = network_figure(init_filtered, init_stats, names, None)
    init_nodes = list(set(init_filtered["source"]).union(set(init_filtered["target"]))) if not init_filtered.empty else []
    init_legend = build_support_legend(init_stats, init_nodes)
    init_legend_style = SUPPORT_LEGEND_STYLE
    init_kpis = build_kpis(init_subset, init_stats, init_edges, init_filtered, names)
    init_side = build_side_panel(init_subset, init_stats, init_edges, init_mistakes, names, None, default_split)

    init_active_species = sorted(
        list(set(init_edges["source"]).union(set(init_edges["target"]))),
        key=lambda sid: names.get(sid, str(sid)).lower(),
    )
    init_focus_options = [
        {"label": f"{names.get(sid, str(sid))} [{sid}]", "value": sid}
        for sid in init_active_species
    ]

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "iWildCam | Species Confusion Explorer"
    app.index_string = app.index_string.replace(
        "</head>",
        """<style>
.kpi-card-hover {
    position: relative;
    cursor: pointer;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card-hover:hover {
    border-color: #93c5fd !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08) !important;
}
.kpi-popup {
    display: none !important;
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    padding: 12px;
    z-index: 99999;
    pointer-events: auto !important;
    white-space: normal;
    text-align: left;
    max-height: 85vh;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}
.kpi-popup::before {
    content: "";
    position: absolute;
    top: -6px;
    left: 28px;
    width: 10px;
    height: 10px;
    background: #ffffff;
    border-left: 1px solid #cbd5e1;
    border-top: 1px solid #cbd5e1;
    transform: rotate(45deg);
}
.kpi-popup::after {
    content: "";
    position: absolute;
    top: -8px;
    left: 0;
    right: 0;
    height: 8px;
}
.kpi-card-hover:hover .kpi-popup {
    display: block !important;
}
.sidebar-text-hover-item {
    position: relative;
    cursor: pointer;
}
.sidebar-text-hover-item:hover .sidebar-text-label {
    color: #1d4ed8 !important;
    text-decoration: underline;
}
.sidebar-text-popup {
    display: none !important;
    position: absolute;
    top: 50%;
    right: calc(100% + 8px);
    transform: translateY(-50%);
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    z-index: 999999;
    pointer-events: auto !important;
    white-space: normal;
    text-align: left;
    max-height: 85vh;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}
.sidebar-text-popup::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -6px;
    margin-top: -5px;
    width: 10px;
    height: 10px;
    background: #ffffff;
    border-right: 1px solid #cbd5e1;
    border-top: 1px solid #cbd5e1;
    transform: rotate(45deg);
}
.sidebar-text-popup::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    right: -10px;
    width: 10px;
}
.sidebar-text-hover-item:hover .sidebar-text-popup {
    display: block !important;
}
.blue-slider-input {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    padding: 1px 4px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    width: 52px !important;
    text-align: center !important;
    outline: none !important;
    box-sizing: border-box !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.blue-slider-input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}
.rc-slider ~ input,
.rc-slider-input,
.dash-slider input[type="number"]:not(.blue-slider-input) {
    display: none !important;
}
.reset-btn {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #334155;
    padding: 0 14px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: all 0.15s ease;
}
.reset-btn:hover {
    background: #f8fafc;
    border-color: #94a3b8;
    color: #0f172a;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.diashow-container {
    position: relative;
    user-select: none;
}
.diashow-btn {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 22px;
    height: 22px;
    background: rgba(15, 23, 42, 0.65);
    color: #ffffff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: bold;
    cursor: pointer;
    z-index: 10;
    transition: background 0.15s ease, transform 0.15s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
    line-height: 1;
}
.diashow-btn:hover {
    background: rgba(15, 23, 42, 0.95);
    transform: translateY(-50%) scale(1.12);
}
.diashow-prev {
    left: 4px;
}
.diashow-next {
    right: 4px;
}
.diashow-counter {
    position: absolute;
    bottom: 4px;
    right: 4px;
    background: rgba(15, 23, 42, 0.72);
    color: #ffffff;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 6px;
    backdrop-filter: blur(2px);
    z-index: 10;
    letter-spacing: 0.3px;
    pointer-events: none;
}
</style>
<script>
document.addEventListener('click', function(e) {
    const btn = e.target.closest('.diashow-btn');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();

    const container = btn.closest('.diashow-container');
    if (!container) return;

    let images = [];
    try {
        images = JSON.parse(container.getAttribute('data-images') || '[]');
    } catch(err) {
        images = [];
    }
    if (!images || images.length <= 1) return;

    const step = parseInt(btn.getAttribute('data-step') || '1', 10);
    let idx = parseInt(container.getAttribute('data-idx') || '0', 10);
    idx = (idx + step + images.length) % images.length;
    container.setAttribute('data-idx', idx);

    const imgEl = container.querySelector('.diashow-img');
    const counterEl = container.querySelector('.diashow-counter');
    const captionEl = container.querySelector('.diashow-caption');
    if (imgEl && images[idx]) {
        imgEl.src = images[idx];
    }
    if (counterEl) {
        counterEl.textContent = (idx + 1) + ' / ' + images.length;
    }
    if (captionEl) {
        let captions = [];
        try {
            captions = JSON.parse(container.getAttribute('data-captions') || '[]');
        } catch(err) {
            captions = [];
        }
        if (captions && captions[idx] !== undefined) {
            captionEl.innerHTML = captions[idx];
        }
    }
}, true);

document.addEventListener('keydown', function(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const hoveredContainer = document.querySelector('.diashow-container:hover');
    if (!hoveredContainer) return;
    e.preventDefault();
    const step = (e.key === 'ArrowRight') ? 1 : -1;
    let images = [];
    try {
        images = JSON.parse(hoveredContainer.getAttribute('data-images') || '[]');
    } catch(err) {
        images = [];
    }
    if (!images || images.length <= 1) return;
    let idx = parseInt(hoveredContainer.getAttribute('data-idx') || '0', 10);
    idx = (idx + step + images.length) % images.length;
    hoveredContainer.setAttribute('data-idx', idx);
    const imgEl = hoveredContainer.querySelector('.diashow-img');
    const counterEl = hoveredContainer.querySelector('.diashow-counter');
    const captionEl = hoveredContainer.querySelector('.diashow-caption');
    if (imgEl && images[idx]) {
        imgEl.src = images[idx];
    }
    if (counterEl) {
        counterEl.textContent = (idx + 1) + ' / ' + images.length;
    }
    if (captionEl) {
        let captions = [];
        try {
            captions = JSON.parse(hoveredContainer.getAttribute('data-captions') || '[]');
        } catch(err) {
            captions = [];
        }
        if (captions && captions[idx] !== undefined) {
            captionEl.innerHTML = captions[idx];
        }
    }
});

// Use the same rendered curve for the shaft endpoint and arrowhead base.
// SVG lengths are pixels, so gaps and head sizes stay fixed after zoom/resize.
function trimConfusionEdges(graph) {
    for (const trace of graph._fullData || []) {
        const gap = trace.meta && trace.meta.edge_geometry;
        if (!gap) continue;
        const group = graph.querySelector('.trace' + trace.uid);
        const path = group && group.querySelector('path.js-line');
        if (!path) continue;
        const length = path.getTotalLength();
        // Find the actual circle intersections, rather than subtracting a
        // radius along a bent path (which leaves a visible gap).
        const source = path.getPointAtLength(0);
        const target = path.getPointAtLength(length);
        function boundary(center, radius, fromSource) {
            let low = 0, high = length;
            for (let i = 0; i < 32; i++) {
                const mid = (low + high) / 2;
                const point = path.getPointAtLength(mid);
                const inside = Math.hypot(point.x - center.x, point.y - center.y) < radius;
                if (inside === fromSource) low = mid;
                else high = mid;
            }
            return (low + high) / 2;
        }
        const startLength = boundary(source, gap.start_gap, true);
        const tipLength = boundary(target, gap.end_gap, false);
        const available = tipLength - startLength;
        group.style.visibility = available > 0 ? 'visible' : 'hidden';
        if (available <= 0) continue;

        // Short edges still get a head: reduce its size instead of hiding it.
        const headLength = Math.min(gap.head_length, available * 0.55);
        const headWidth = gap.head_half_width * headLength / gap.head_length;
        const baseLength = tipLength - headLength;
        const visible = baseLength - startLength;
        const tip = path.getPointAtLength(tipLength);
        const base = path.getPointAtLength(baseLength);
        const dx = tip.x - base.x, dy = tip.y - base.y;
        const distance = Math.hypot(dx, dy) || 1;
        const nx = -dy / distance * headWidth;
        const ny = dx / distance * headWidth;

        let head = group.querySelector('.confusion-arrowhead');
        if (!head) {
            head = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            head.classList.add('confusion-arrowhead');
            head.setAttribute('pointer-events', 'none');
            group.appendChild(head);
        }
        head.setAttribute('d', `M ${tip.x} ${tip.y} L ${base.x + nx} ${base.y + ny} L ${base.x - nx} ${base.y - ny} Z`);
        head.setAttribute('fill', trace.line.color);
        head.setAttribute('stroke', 'none');

        // Stop at the head's base; no translucent shaft underneath the triangle.
        path.style.strokeDasharray = `0 ${startLength} ${visible} ${length}`;
        path.style.strokeDashoffset = '0';
        path.style.strokeLinecap = 'butt';
    }
}

function connectConfusionEdges() {
    const graph = document.querySelector('#network .js-plotly-plot');
    if (!graph || typeof graph.on !== 'function' || graph.dataset.edgeTrimming) return;
    graph.dataset.edgeTrimming = 'ready';
    graph.on('plotly_afterplot', () => trimConfusionEdges(graph));
    trimConfusionEdges(graph);
}

// Dash mounts the graph after this page script has loaded.
new MutationObserver(connectConfusionEdges).observe(document.documentElement, {
    childList: true,
    subtree: true
});

</script>
</head>""",
    )

    card_style = {
        "background": "#ffffff",
        "padding": "18px",
        "borderRadius": "12px",
        "border": "1px solid #e2e8f0",
        "boxShadow": "0 1px 3px 0 rgba(0, 0, 0, 0.05)",
    }
    label_style = {"fontWeight": "600", "fontSize": "13px", "color": "#334155", "marginBottom": "6px"}

    def control(title, component):
        return html.Div(
            [html.Div(title, style=label_style), component],
            style={"flex": "1 1 200px", "minWidth": "170px"},
        )

    def slider_control(title, component, val_id, initial_val, min_val, max_val, step_val, container_id=None, hidden=False):
        style = {"flex": "1 1 200px", "minWidth": "170px"}
        if hidden:
            style["display"] = "none"
        div_kwargs = {"style": style}
        if container_id:
            div_kwargs["id"] = container_id
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(title, style=label_style),
                        dcc.Input(
                            id=val_id,
                            type="number",
                            min=min_val,
                            max=max_val,
                            step=step_val,
                            value=initial_val,
                            debounce=True,
                            className="blue-slider-input",
                            style={
                                "background": "#eff6ff",
                                "color": "#1d4ed8",
                                "border": "1px solid #bfdbfe",
                                "padding": "1px 4px",
                                "borderRadius": "6px",
                                "fontWeight": "700",
                                "fontSize": "12px",
                                "width": "52px",
                                "textAlign": "center",
                                "outline": "none",
                                "boxSizing": "border-box",
                            },
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "4px"},
                ),
                component,
            ],
            **div_kwargs,
        )

    app.layout = html.Div([
        # Top Header
        html.Div([
            html.H1(
                "iWildCam: Species Confusion Explorer",
                style={
                    "margin": "0 0 8px 0",
                    "color": "#0f172a",
                    "fontSize": "35px",
                    "fontWeight": "800",
                    "letterSpacing": "-0.03em",
                    "lineHeight": "1.2",
                },
            ),
            html.P(
                "Analyze species confusion patterns in fine-tuned ResNet-50.",
                style={
                    "margin": "0",
                    "color": "#64748b",
                    "fontSize": "16px",
                    "fontWeight": "400",
                },
            ),
        ], style={"textAlign": "center", "marginBottom": "26px", "paddingTop": "10px"}),

        # KPI & Dataset Split Row
        html.Div([
            # Dataset Split selector card (on the left)
            html.Div([
                html.Div("Dataset Split", style=KPI_TITLE_STYLE),
                dcc.Dropdown(
                    id="split", clearable=False, value=default_split,
                    options=[{"label": SPLIT_DESCRIPTIONS.get(s, s), "value": s} for s in splits],
                    style={"fontSize": "13px", "marginTop": "4px"},
                ),
            ], style={
                **KPI_CARD_STYLE,
                "flex": "1.1 1 240px",
                "minWidth": "220px",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
            }),
            # KPI Metrics Banner (the 3 cards)
            html.Div(
                id="kpi-banner",
                children=init_kpis,
                style={"display": "flex", "gap": "14px", "flex": "3 1 600px", "flexWrap": "wrap"},
            ),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px", "alignItems": "stretch"}),

        # Control Panel Card
        html.Div([
            html.Div([
                control("View Visualization", dcc.RadioItems(
                    id="view-mode",
                    options=[
                        {"label": " Community Clusters", "value": "network"},
                        {"label": " Confusion Matrix", "value": "matrix"},
                    ],
                    value="network",
                    inline=True,
                    style={"fontSize": "13px", "fontWeight": "500", "paddingTop": "6px"},
                )),
                html.Div([
                    html.Div("Focus Species (Click node to select)", style=label_style),
                    dcc.Dropdown(
                        id="focus", clearable=True, placeholder="All species (global overview)",
                        options=init_focus_options,
                    ),
                ], style={"flex": "2 1 360px", "minWidth": "260px"}),
                html.Button(
                    "Reset Filters", id="reset-filters-btn", n_clicks=0,
                    title="Reset all filters, view mode, and focus to defaults (keeps current dataset split)",
                    className="reset-btn",
                    style={"alignSelf": "flex-end"},
                ),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "14px", "alignItems": "flex-end"}),

            html.Div([
                html.Div(
                    [
                        html.Div("Rank Confusion Arrows By", id="rank-label", style=label_style),
                        dcc.Dropdown(
                            id="rank", clearable=False, value="rate", options=[
                                {"label": "Confusion Rate (% of true class)", "value": "rate"},
                                {"label": "Confusion Count (Total errors)", "value": "count"},
                            ],
                        ),
                    ],
                    style={"flex": "1 1 200px", "minWidth": "170px"},
                ),
                slider_control(
                    "Min Ground-Truth Images",
                    dcc.Slider(
                        id="min-support", min=1, max=200, step=1, value=30,
                        marks={1: "1", 50: "50", 100: "100", 150: "150", 200: "200"},
                        allow_direct_input=False,
                    ),
                    "min-support-val", 30, 1, 200, 1,
                ),
                slider_control(
                    "Min Confusion Count",
                    dcc.Slider(
                        id="min-count", min=1, max=50, step=1, value=5,
                        marks={1: "1", 10: "10", 20: "20", 35: "35", 50: "50"},
                        allow_direct_input=False,
                    ),
                    "min-count-val", 5, 1, 50, 1,
                    container_id="min-count-container",
                    hidden=True,
                ),
                slider_control(
                    "Min Confusion Rate (%)",
                    dcc.Slider(
                        id="min-rate", min=0, max=25, step=0.5, value=1.0,
                        marks={0: "0%", 5: "5%", 10: "10%", 15: "15%", 20: "20%", 25: "25%"},
                        allow_direct_input=False,
                    ),
                    "min-rate-val", 1.0, 0, 25, 0.5,
                    container_id="min-rate-container",
                    hidden=False,
                ),
                slider_control(
                    "Max Arrows Displayed",
                    dcc.Slider(
                        id="top-edges", min=5, max=50, step=5, value=25,
                        marks={5: "5", 15: "15", 25: "25", 35: "35", 50: "50"},
                        allow_direct_input=False,
                    ),
                    "top-edges-val", 25, 5, 50, 5,
                ),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
        ], style={**card_style, "marginBottom": "16px"}),

        # Main Row
        html.Div([
            # Left Card: Network or Heatmap
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(id="main-graph-title", children=f"Species Confusion Network ({default_split}) - Community Clusters", style={"margin": "0 0 4px 0", "fontSize": "17px", "fontWeight": "700", "color": "#0f172a"}),
                        html.Div(id="main-graph-subtitle", children=[
                            html.Div("Directed arrows point True Species → Misclassified Species."),
                            html.Div("Click any node to inspect details.", style={"color": "#64748b", "fontSize": "11.5px", "marginTop": "2px"}),
                        ], style={"margin": "0", "fontSize": "12px", "color": "#475569", "lineHeight": "1.45"}),
                    ], style={"flex": "1", "minWidth": "0", "marginRight": "12px"}),
                    html.Button(
                        "Reset Focus", id="reset-focus-btn", n_clicks=0,
                        title="Reset focus to all species",
                        className="reset-btn",
                        style={"height": "32px"},
                    ),
                ], style={"display": "flex", "alignItems": "flex-start", "justifyContent": "space-between", "marginBottom": "8px"}),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="network", figure=init_fig, clear_on_unhover=True, config={"displaylogo": False, "responsive": True}),
                        type="dot",
                    ),
                    html.Div(id="support-size-legend", children=init_legend, style=init_legend_style),
                    dcc.Tooltip(
                        id="graph-tooltip",
                        direction="right",
                        background_color="white",
                        border_color="#cbd5e1",
                        zindex=9999,
                        style={"padding": "0px", "borderRadius": "10px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal", "pointerEvents": "none", "maxHeight": "85vh", "overflowY": "auto", "overflowX": "hidden", "boxSizing": "border-box"},
                    ),
                ], style={"position": "relative"}),

                # Visual Encodings Guide (integrated directly beneath the network graph)
                html.Div([
                    html.Div([
                        html.Span("Network Visual Encodings", style={"fontWeight": "700", "fontSize": "12px", "color": "#0f172a"}),
                        html.Span("— Reference guide for node and arrow properties used in the graph", style={"fontSize": "11px", "color": "#64748b", "marginLeft": "6px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        # 1. Species Support (Node Size)
                        html.Div([
                            html.Div([
                                html.Div("Species Support", style={"fontWeight": "700", "fontSize": "11.5px", "color": "#0f172a"}),
                                html.Span("Node Size", style={"fontSize": "9.5px", "fontWeight": "600", "background": "#f1f5f9", "color": "#475569", "padding": "1px 5px", "borderRadius": "4px"}),
                            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                            html.Div("Ground-truth images in split. Sized via square root to balance rare and common classes.", style={"fontSize": "10.5px", "color": "#475569", "lineHeight": "1.35", "margin": "4px 0 6px 0"}),
                            html.Div("Count of True Images", style={"fontFamily": "ui-monospace, monospace", "fontSize": "9.5px", "background": "#ffffff", "border": "1px solid #e2e8f0", "padding": "2px 4px", "borderRadius": "4px", "fontWeight": "600", "color": "#1e293b", "marginTop": "auto", "textAlign": "center"}),
                        ], style={
                            "flex": "1 1 180px", "minWidth": "160px", "background": "#f8fafc",
                            "border": "1px solid #e2e8f0", "borderRadius": "8px", "padding": "10px 12px",
                            "display": "flex", "flexDirection": "column",
                        }),

                        # 2. Species Error Rate (Node Color)
                        html.Div([
                            html.Div([
                                html.Div("Species Error Rate", style={"fontWeight": "700", "fontSize": "11.5px", "color": "#0f172a"}),
                                html.Span("Node Color", style={"fontSize": "9.5px", "fontWeight": "600", "background": "#fef2f2", "color": "#dc2626", "padding": "1px 5px", "borderRadius": "4px"}),
                            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                            html.Div("Fraction of ground-truth images misclassified (= 1 − Recall). Shaded from white to dark red.", style={"fontSize": "10.5px", "color": "#475569", "lineHeight": "1.35", "margin": "4px 0 6px 0"}),
                            html.Div("Errors(A) ÷ Support(A)", style={"fontFamily": "ui-monospace, monospace", "fontSize": "9.5px", "background": "#ffffff", "border": "1px solid #e2e8f0", "padding": "2px 4px", "borderRadius": "4px", "fontWeight": "600", "color": "#1e293b", "marginTop": "auto", "textAlign": "center"}),
                        ], style={
                            "flex": "1 1 180px", "minWidth": "160px", "background": "#f8fafc",
                            "border": "1px solid #e2e8f0", "borderRadius": "8px", "padding": "10px 12px",
                            "display": "flex", "flexDirection": "column",
                        }),

                        # 3. Confusion Rate (Arrow Width)
                        html.Div([
                            html.Div([
                                html.Div("Confusion Rate", style={"fontWeight": "700", "fontSize": "11.5px", "color": "#0f172a"}),
                                html.Span("Arrow Width", style={"fontSize": "9.5px", "fontWeight": "600", "background": "#fff7ed", "color": "#ea580c", "padding": "1px 5px", "borderRadius": "4px"}),
                            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                            html.Div("Pairwise error share: fraction of species A mistaken specifically as species B.", style={"fontSize": "10.5px", "color": "#475569", "lineHeight": "1.35", "margin": "4px 0 6px 0"}),
                            html.Div("Mistakes(A → B) ÷ Support(A)", style={"fontFamily": "ui-monospace, monospace", "fontSize": "9.5px", "background": "#ffffff", "border": "1px solid #e2e8f0", "padding": "2px 4px", "borderRadius": "4px", "fontWeight": "600", "color": "#1e293b", "marginTop": "auto", "textAlign": "center"}),
                        ], style={
                            "flex": "1 1 180px", "minWidth": "160px", "background": "#f8fafc",
                            "border": "1px solid #e2e8f0", "borderRadius": "8px", "padding": "10px 12px",
                            "display": "flex", "flexDirection": "column",
                        }),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={
                    "marginTop": "14px",
                    "paddingTop": "12px",
                    "borderTop": "1px solid #f1f5f9",
                }),
            ], style={**card_style, "flex": "2 1 640px", "minWidth": "480px"}),

            # Right Card: Insights
            html.Div(id="side-panel-content", children=init_side, style={**card_style, "flex": "1 1 380px", "minWidth": "350px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    ], style={"maxWidth": "1600px", "margin": "20px auto", "padding": "0 24px", "background": "#f8fafc", "fontFamily": "Inter, Arial, sans-serif"})

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    @app.callback(
        Output("focus", "options"),
        Output("focus", "value"),
        Input("network", "clickData"),
        Input("reset-focus-btn", "n_clicks"),
        Input("reset-filters-btn", "n_clicks"),
        Input("split", "value"),
        State("focus", "value"),
        prevent_initial_call=True,
    )
    def update_focus(click_data, reset_clicks, reset_filter_clicks, split, current_focus):
        trigger = ctx.triggered_id
        _, _, edges, _ = get_split(split)
        active_species = sorted(
            list(set(edges["source"]).union(set(edges["target"]))),
            key=lambda sid: names.get(sid, str(sid)).lower(),
        )
        options = [
            {"label": f"{names.get(sid, str(sid))} [{sid}]", "value": sid}
            for sid in active_species
        ]

        if trigger in ("reset-focus-btn", "reset-filters-btn"):
            return options, None
        elif trigger == "split":
            new_focus = current_focus if (current_focus is not None and current_focus in set(active_species)) else None
            return options, new_focus
        elif trigger == "network" and click_data and click_data.get("points"):
            pt = click_data["points"][0]
            tag = pt.get("customdata")
            if isinstance(tag, (list, tuple)) and len(tag) >= 2:
                if tag[0] == "node":
                    clicked = int(tag[1])
                    return options, None if clicked == current_focus else clicked
                elif tag[0] == "edge":
                    src = int(tag[1])
                    return options, src
        return options, no_update

    # Callback: Reset view mode to Directed Network
    @app.callback(
        Output("view-mode", "value"),
        Input("reset-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_view_mode(reset_clicks):
        return "network"

    # Callbacks: Bidirectional synchronization between sliders and blue input fields & Reset Filters
    @app.callback(
        Output("min-support", "value"),
        Output("min-support-val", "value"),
        Input("min-support", "value"),
        Input("min-support-val", "value"),
        Input("reset-filters-btn", "n_clicks"),
    )
    def sync_min_support(slider_val, input_val, reset_clicks):
        trigger = ctx.triggered_id
        if trigger == "reset-filters-btn":
            return 30, 30
        if trigger == "min-support-val":
            if input_val is None:
                return no_update, no_update
            val = max(1, min(200, int(input_val)))
            return val, val
        if slider_val is None:
            return no_update, no_update
        return slider_val, slider_val

    @app.callback(
        Output("min-count", "value"),
        Output("min-count-val", "value"),
        Input("min-count", "value"),
        Input("min-count-val", "value"),
        Input("reset-filters-btn", "n_clicks"),
    )
    def sync_min_count(slider_val, input_val, reset_clicks):
        trigger = ctx.triggered_id
        if trigger == "reset-filters-btn":
            return 5, 5
        if trigger == "min-count-val":
            if input_val is None:
                return no_update, no_update
            val = max(1, min(50, int(input_val)))
            return val, val
        if slider_val is None:
            return no_update, no_update
        return slider_val, slider_val

    @app.callback(
        Output("min-rate", "value"),
        Output("min-rate-val", "value"),
        Input("min-rate", "value"),
        Input("min-rate-val", "value"),
        Input("reset-filters-btn", "n_clicks"),
    )
    def sync_min_rate(slider_val, input_val, reset_clicks):
        trigger = ctx.triggered_id
        if trigger == "reset-filters-btn":
            return 1.0, 1.0
        if trigger == "min-rate-val":
            if input_val is None:
                return no_update, no_update
            val = round(max(0.0, min(25.0, float(input_val))), 1)
            return val, val
        if slider_val is None:
            return no_update, no_update
        return slider_val, slider_val

    @app.callback(
        Output("top-edges", "value"),
        Output("top-edges-val", "value"),
        Input("top-edges", "value"),
        Input("top-edges-val", "value"),
        Input("reset-filters-btn", "n_clicks"),
    )
    def sync_top_edges(slider_val, input_val, reset_clicks):
        trigger = ctx.triggered_id
        if trigger == "reset-filters-btn":
            return 25, 25
        if trigger == "top-edges-val":
            if input_val is None:
                return no_update, no_update
            val = max(5, min(50, int(input_val)))
            return val, val
        if slider_val is None:
            return no_update, no_update
        return slider_val, slider_val

    # Callback: Synchronize Rank Metric with View Mode & Reset Filters
    @app.callback(
        Output("rank", "value"),
        Output("rank", "disabled"),
        Output("rank", "options"),
        Output("rank-label", "children"),
        Input("view-mode", "value"),
        Input("reset-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_rank_with_view_mode(view_mode, reset_clicks):
        trigger = ctx.triggered_id
        if trigger == "reset-filters-btn":
            options = [
                {"label": "Confusion Rate (% of true class)", "value": "rate"},
                {"label": "Confusion Count (Total errors)", "value": "count"},
            ]
            return "rate", False, options, "Rank Confusion Arrows By"
        if view_mode == "matrix":
            options = [
                {"label": "Confusion Count (Total errors)", "value": "count"},
                {"label": "Confusion Rate (Community Clusters only)", "value": "rate", "disabled": True},
            ]
            return "count", True, options, "Rank Metric (Count only in Matrix)"
        else:
            options = [
                {"label": "Confusion Rate (% of true class)", "value": "rate"},
                {"label": "Confusion Count (Total errors)", "value": "count"},
            ]
            return "rate", False, options, "Rank Confusion Arrows By"

    # Callback: Toggle visibility of Min Confusion Count vs Min Confusion Rate sliders
    @app.callback(
        Output("min-count-container", "style"),
        Output("min-rate-container", "style"),
        Input("rank", "value"),
    )
    def toggle_slider_visibility(rank):
        shown_style = {"flex": "1 1 200px", "minWidth": "170px", "display": "block"}
        hidden_style = {"flex": "1 1 200px", "minWidth": "170px", "display": "none"}
        if rank == "rate":
            return hidden_style, shown_style
        return shown_style, hidden_style

    # Callback 1: Main Graph & Title
    @app.callback(
        Output("network", "figure"),
        Output("main-graph-title", "children"),
        Output("main-graph-subtitle", "children"),
        Output("support-size-legend", "children"),
        Output("support-size-legend", "style"),
        Input("split", "value"),
        Input("focus", "value"),
        Input("view-mode", "value"),
        Input("rank", "value"),
        Input("min-support", "value"),
        Input("min-count", "value"),
        Input("min-rate", "value"),
        Input("top-edges", "value"),
    )
    def update_main_graph(split, focus, view_mode, rank, support, count, rate, limit):
        subset, stats, edges, _ = get_split(split)
        # Enforce that the confusion matrix is strictly rendered using confusion count
        effective_rank = "count" if view_mode == "matrix" else rank
        filtered = visible_edges(
            edges, names, int(support), int(count), float(rate), int(limit),
            effective_rank, focus,
        )
        if view_mode == "matrix":
            fig = confusion_matrix_figure(filtered, names, focus)
            title = f"Confusion Matrix Heatmap ({split})"
            sub = html.Div([
                html.Div("Rows: True Ground-Truth Species | Columns: Model Predictions."),
                html.Div("Off-diagonal numbers show error volumes.", style={"color": "#64748b", "fontSize": "11.5px", "marginTop": "2px"}),
            ], style={"fontSize": "12px", "color": "#475569", "marginTop": "4px", "lineHeight": "1.45"})
            leg_children = html.Div()
            leg_style = {"display": "none"}
        else:
            fig = network_figure(filtered, stats, names, focus)
            vis_nodes = list(set(filtered["source"]).union(set(filtered["target"]))) if not filtered.empty else []
            leg_children = build_support_legend(stats, vis_nodes)
            leg_style = SUPPORT_LEGEND_STYLE
            if focus is not None:
                focus_name = names.get(focus, str(focus))
                title = f"Species Confusion Network ({split}) — Focused on {focus_name}"
                sub = html.Div([
                    html.Div([
                        html.Span("→ Orange arrows", style={"color": "#ea580c", "fontWeight": "700"}),
                        html.Span(f" = Outgoing errors ({focus_name} mistaken for ...)"),
                    ]),
                    html.Div([
                        html.Span("← Blue arrows", style={"color": "#0284c7", "fontWeight": "700"}),
                        html.Span(f" = Incoming false predictions (... mistaken as {focus_name})"),
                    ]),
                    html.Div(
                        "Click any node to inspect details.",
                        style={"color": "#64748b", "fontSize": "11.5px", "marginTop": "2px"}
                    ),
                ], style={"fontSize": "12px", "color": "#475569", "marginTop": "4px", "lineHeight": "1.45"})
            else:
                title = f"Species Confusion Network ({split}) - Community Clusters"
                sub = html.Div([
                    html.Div("Directed arrows point True Species → Misclassified Species."),
                    html.Div("Click any node to inspect details.", style={"color": "#64748b", "fontSize": "11.5px", "marginTop": "2px"}),
                ], style={"fontSize": "12px", "color": "#475569", "marginTop": "4px", "lineHeight": "1.45"})
        return fig, title, sub, leg_children, leg_style

    # Callback 2: KPIs & Side Insights Panel
    @app.callback(
        Output("kpi-banner", "children"),
        Output("side-panel-content", "children"),
        Input("split", "value"),
        Input("focus", "value"),
        Input("rank", "value"),
        Input("min-support", "value"),
        Input("min-count", "value"),
        Input("min-rate", "value"),
        Input("top-edges", "value"),
    )
    def update_insights(split, focus, rank, support, count, rate, limit):
        subset, stats, edges, mistakes = get_split(split)
        filtered = visible_edges(
            edges, names, int(support), int(count), float(rate), int(limit),
            rank, focus,
        )
        kpis = build_kpis(subset, stats, edges, filtered, names)
        side = build_side_panel(subset, stats, edges, mistakes, names, focus, split)
        return kpis, side

    # Callback 3: Interactive Hover Tooltip with Animal Photos
    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Output("graph-tooltip", "direction"),
        Input("network", "hoverData"),
        State("split", "value"),
    )
    def update_tooltip(hover_data, split):
        if not hover_data or not hover_data.get("points"):
            return False, no_update, no_update, no_update

        pt = hover_data["points"][0]
        bbox = pt.get("bbox")
        tag = pt.get("customdata")
        if not tag or not isinstance(tag, (list, tuple)) or len(tag) < 2 or not bbox:
            return False, no_update, no_update, no_update

        x_mid = (bbox.get("x0", 0) + bbox.get("x1", 0)) / 2
        y_mid = (bbox.get("y0", 0) + bbox.get("y1", 0)) / 2
        if y_mid < 140:
            direction = "bottom"
        elif y_mid > 500:
            direction = "top"
        elif x_mid > 440:
            direction = "left"
        else:
            direction = "right"

        tag_type = tag[0]

        if tag_type == "node":
            class_id = int(tag[1])
            name = names.get(class_id, str(class_id))
            _, stats, _, _ = get_split(split)

            if class_id in stats.index:
                supp = int(stats.loc[class_id, "support"])
                errs = int(stats.loc[class_id, "errors"])
                rate = float(stats.loc[class_id, "error_rate"])
                fp = int(stats.loc[class_id, "falsely_predicted_count"])
            else:
                supp, errs, rate, fp = 0, 0, 0.0, 0

            species_imgs = get_class_images(class_id, max_images=1)
            img_block = make_diashow(species_imgs, border_color="#e2e8f0")

            content = html.Div([
                img_block,
                html.Div([
                    html.Div(name, style={"fontWeight": "bold", "fontSize": "13px", "color": "#0f172a", "margin": "6px 0 4px 0", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div(f"Class ID: {class_id} | Support: {supp:,} true images", style={"fontSize": "11px", "color": "#64748b", "marginBottom": "3px", "whiteSpace": "normal"}),
                    html.Div(f"Error Rate: {rate:.1%} ({errs:,} errors / {supp:,} true images)", style={"fontSize": "11px", "fontWeight": "600", "color": "#dc2626" if rate > 0.4 else "#16a34a", "marginBottom": "3px", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div(f"Falsely predicted for other classes: {fp:,} times", style={"fontSize": "10px", "color": "#64748b", "marginBottom": "4px", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div("Click node to inspect photos & camera breakdown in sidebar", style={"fontSize": "10px", "color": "#2563eb", "fontStyle": "italic", "marginTop": "4px"}),
                ]),
            ], style={
                "position": "relative",
                "padding": "10px",
                "width": "260px",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "whiteSpace": "normal",
                "overflowX": "hidden",
                "background": "white",
                "borderRadius": "10px",
            })

            return True, bbox, content, direction

        elif tag_type == "edge":
            src = int(tag[1])
            dst = int(tag[2])
            count = int(tag[3]) if len(tag) > 3 else 0
            rate = float(tag[4]) if len(tag) > 4 else 0.0
            top_loc = str(tag[5]) if len(tag) > 5 else ""
            top_loc_pct = float(tag[6]) if len(tag) > 6 else 0.0
            top_loc_cnt = int(tag[7]) if len(tag) > 7 else int(round(count * top_loc_pct))
            num_cams = int(tag[8]) if len(tag) > 8 else 1
            loc_src_supp = int(tag[9]) if len(tag) > 9 else 0
            loc_cam_total = int(tag[10]) if len(tag) > 10 else 0

            content = make_pair_tooltip_content(
                src, dst, count, rate, names, top_loc, top_loc_pct, top_loc_cnt, num_cams,
                show_close=True, loc_src_supp=loc_src_supp, loc_cam_total=loc_cam_total
            )
            return True, bbox, content, direction

        return False, no_update, no_update, no_update

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/predictions.parquet"),
        help="Path to the model predictions parquet (relative to current directory)",
    )
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    data = read_predictions(args.predictions)
    print(f"Loaded {len(data):,} iWildCam predictions from {args.predictions}")
    create_app(data).run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
