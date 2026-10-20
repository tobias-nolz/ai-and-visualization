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
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update


COLUMNS = [
    "id", "split", "label", "label_name", "prediction",
    "prediction_name", "confidence", "location", "filename",
]
SPLIT_DESCRIPTIONS = {
    "val": "OOD validation (unseen cameras)",
    "id_val": "ID validation (seen cameras)",
    "test": "OOD test (unseen cameras)",
    "id_test": "ID test (seen cameras)",
    "train": "Training set (seen cameras)",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = Path("data/iwildcam_v2.0/train")
if not DATA_DIR.is_dir() and (REPO_ROOT / "data/iwildcam_v2.0/train").is_dir():
    DATA_DIR = REPO_ROOT / "data/iwildcam_v2.0/train"

META_CSV = Path("data/iwildcam_v2.0/metadata.csv")
if not META_CSV.is_file() and (REPO_ROOT / "data/iwildcam_v2.0/metadata.csv").is_file():
    META_CSV = REPO_ROOT / "data/iwildcam_v2.0/metadata.csv"

REPR_JSON = Path("phase1/pair_a/data/representative_images.json")
if not REPR_JSON.is_file() and (REPO_ROOT / "phase1/pair_a/data/representative_images.json").is_file():
    REPR_JSON = REPO_ROOT / "phase1/pair_a/data/representative_images.json"


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


def load_image_map() -> dict[int, str]:
    mapping = dict(_load_fallback_map())
    target_json = REPR_JSON if REPR_JSON.is_file() else (REPO_ROOT / "phase1/pair_a/data/representative_images.json")
    if target_json.is_file():
        try:
            import json
            with open(target_json, mode="r") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if (DATA_DIR / v).is_file():
                        mapping[int(k)] = v
        except Exception:
            pass
    return mapping


@lru_cache(maxsize=400)
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
    mapping = load_image_map()
    fn = mapping.get(class_id)
    if not fn:
        return None
    return _read_image_base64(fn)


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

    edges = edges.merge(
        top_locs.rename(columns={
            "label": "source", "prediction": "target",
            "location": "top_loc", "loc_cnt": "top_loc_cnt"
        }),
        on=["source", "target"], how="left"
    )
    edges["top_loc_pct"] = edges["top_loc_cnt"] / edges["count"]

    return stats, edges, mistakes


def visible_edges(
    edges: pd.DataFrame,
    names: dict[int, str],
    min_support: int,
    min_count: int,
    min_rate_pct: float,
    max_edges: int,
    sort_by: str,
    show_empty: bool,
    focused_class: int | None,
) -> pd.DataFrame:
    result = edges.loc[
        (edges["source_support"] >= min_support)
        & (edges["count"] >= min_count)
        & (edges["rate"] * 100 >= min_rate_pct)
    ].copy()
    if not show_empty:
        empty_ids = {k for k, v in names.items() if v.lower().strip() == "empty"}
        result = result.loc[
            ~result["source"].isin(empty_ids)
            & ~result["target"].isin(empty_ids)
        ]
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

    comp_centers: dict[int, tuple[float, float]] = {}
    if num_comp == 1:
        comp_centers[0] = (0.0, 0.0)
    else:
        phi = np.linspace(0, 2 * np.pi, num_comp, endpoint=False)
        center_dist = max(1.5, 0.9 + 0.35 * num_comp)
        for idx in range(num_comp):
            comp_centers[idx] = (center_dist * np.cos(phi[idx]), center_dist * np.sin(phi[idx]))

    for idx, comp in enumerate(components):
        comp_nodes = sorted(list(comp), key=lambda n: stats.loc[n, "support"] if n in stats.index else 0, reverse=True)
        ordered_nodes.extend(comp_nodes)
        cx, cy = comp_centers[idx]
        m = len(comp_nodes)
        if m == 1:
            positions[comp_nodes[0]] = (cx, cy)
            node_centers[comp_nodes[0]] = (cx, cy)
        else:
            rad = 0.55 + 0.10 * np.sqrt(m)
            local_angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
            for node, la in zip(comp_nodes, local_angles):
                positions[node] = (cx + rad * np.cos(la), cy + rad * np.sin(la))
                node_centers[node] = (cx, cy)

    return positions, ordered_nodes, node_centers


def bezier_curve(p0: np.ndarray, p1: np.ndarray, n_pts: int = 14) -> np.ndarray:
    t = np.linspace(0, 1, n_pts)[:, None]
    mid = (p0 + p1) / 2.0
    diff = p1 - p0
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        normal = np.array([0.0, 0.0])
    else:
        normal = np.array([-diff[1], diff[0]]) / dist
    c = mid + 0.16 * dist * normal
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


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
            text="<b>No confusion pairs match these thresholds.</b><br>Try lowering the sliders or enabling 'empty' species.",
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

    # 1. Edges
    for row in edges.itertuples(index=False):
        src, dst = int(row.source), int(row.target)
        if src not in positions or dst not in positions:
            continue
        p0 = np.array(positions[src])
        p1 = np.array(positions[dst])
        pts = bezier_curve(p0, p1)

        is_outgoing = (focused_class is not None and src == focused_class)
        is_incoming = (focused_class is not None and dst == focused_class)

        if focused_class is None:
            edge_color = "rgba(100, 116, 139, 0.42)"
            line_width = 1.2 + 5.5 * np.sqrt(float(row.count) / max_count)
            arrow_color = "#64748b"
        elif is_outgoing:
            edge_color = "rgba(234, 88, 12, 0.90)"
            line_width = 2.5 + 6.0 * np.sqrt(float(row.count) / max_count)
            arrow_color = "#ea580c"
        elif is_incoming:
            edge_color = "rgba(2, 132, 199, 0.90)"
            line_width = 2.5 + 6.0 * np.sqrt(float(row.count) / max_count)
            arrow_color = "#0284c7"
        else:
            edge_color = "rgba(203, 213, 225, 0.12)"
            line_width = 1.0
            arrow_color = "rgba(203, 213, 225, 0.12)"

        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines",
            line={"width": line_width, "color": edge_color},
            hoverinfo="skip", showlegend=False,
        ))

        arrow_idx_tail = int(len(pts) * 0.58)
        arrow_idx_head = int(len(pts) * 0.68)
        if focused_class is None or is_outgoing or is_incoming:
            fig.add_annotation(
                x=float(pts[arrow_idx_head, 0]),
                y=float(pts[arrow_idx_head, 1]),
                ax=float(pts[arrow_idx_tail, 0]),
                ay=float(pts[arrow_idx_tail, 1]),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2,
                arrowsize=1.2 if (is_outgoing or is_incoming) else 1.0,
                arrowwidth=2.0 if (is_outgoing or is_incoming) else 1.4,
                arrowcolor=arrow_color,
            )

        mid_pt = pts[int(len(pts) * 0.5)]
        src_name = names.get(src, str(src))
        dst_name = names.get(dst, str(dst))
        top_loc = str(getattr(row, "top_loc", "Unknown"))
        top_loc_pct = float(getattr(row, "top_loc_pct", 0.0))

        fig.add_trace(go.Scatter(
            x=[mid_pt[0]], y=[mid_pt[1]], mode="markers",
            marker={"size": 16, "color": "rgba(0,0,0,0.001)"},
            hoverinfo="none",
            customdata=[["edge", src, dst, int(row.count), float(row.rate), str(top_loc), float(top_loc_pct)]],
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

        size = 16.0 + 20.0 * np.sqrt(supp / max_support)
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
            "colorscale": "Reds",
            "cmin": 0.0, "cmax": 1.0,
            "showscale": True,
            "colorbar": {
                "title": {"text": "Species Error Rate"},
                "tickformat": ".0%",
                "len": 0.7,
                "outlinewidth": 0,
            },
            "line": {"width": node_borders, "color": node_border_colors},
            "opacity": 0.98,
        },
        hoverinfo="none",
        customdata=[["node", int(nid)] for nid in ordered_nodes],
        showlegend=False,
    ))

    # 3. Community Cluster Labels (projected outward from cluster center)
    for class_id in ordered_nodes:
        x, y = positions[class_id]
        cx, cy = node_centers.get(class_id, (0.0, 0.0))
        angle = np.arctan2(y - cy, x - cx)
        raw_name = names.get(class_id, str(class_id))
        display_name = raw_name if len(raw_name) <= 18 else raw_name[:16] + "…"

        lx = x + 0.12 * np.cos(angle)
        ly = y + 0.12 * np.sin(angle)
        x_anchor = "left" if np.cos(angle) > 0.08 else ("right" if np.cos(angle) < -0.08 else "center")
        y_anchor = "bottom" if np.sin(angle) > 0.08 else ("top" if np.sin(angle) < -0.08 else "middle")
        is_focused = (class_id == focused_class)

        fig.add_annotation(
            x=lx, y=ly, text=display_name,
            showarrow=False,
            xanchor=x_anchor, yanchor=y_anchor,
            font={
                "size": 11 if is_focused else 9.5,
                "color": "#1d4ed8" if is_focused else "#1e293b",
                "weight": 700 if is_focused else 500,
                "family": "Inter, Arial, sans-serif",
            },
        )

    fig.update_layout(
        height=640,
        margin={"l": 40, "r": 70, "t": 30, "b": 30},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest", clickmode="event+select",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "size": 11},
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

    for row in edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        if s in spec_map and t in spec_map:
            matrix[spec_map[s], spec_map[t]] = int(row.count)

    labels = [names.get(sid, str(sid)) for sid in unique_species]
    labels_short = [l if len(l) <= 16 else l[:14] + "…" for l in labels]

    fig.add_trace(go.Heatmap(
        z=matrix,
        x=labels_short,
        y=labels_short,
        colorscale="Reds",
        hoverongaps=False,
        text=matrix,
        texttemplate="%{text}",
        customdata=[[unique_species[r], unique_species[c]] for r in range(len(unique_species)) for c in range(len(unique_species))],
        hovertemplate="<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br>Misclassifications: <b>%{z:,}</b><extra></extra>",
        colorbar={"title": {"text": "Errors"}, "len": 0.7, "outlinewidth": 0},
    ))

    fig.update_layout(
        height=640,
        margin={"l": 140, "r": 40, "t": 30, "b": 100},
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


def make_pair_tooltip_content(src_id: int, dst_id: int, count: int, rate: float, names: dict[int, str]):
    src_name = names.get(src_id, str(src_id))
    dst_name = names.get(dst_id, str(dst_id))
    src_img = get_class_image(src_id)
    dst_img = get_class_image(dst_id)

    return html.Div([
        html.Div([
            html.Div([
                html.Div(f"True: {src_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                html.Img(src=src_img, style={"width": "100%", "height": "80px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #fed7aa"}) if src_img else html.Div("No photo", style={"height": "80px", "background": "#f8fafc", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "10px", "color": "#94a3b8"}),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
            html.Div([
                html.Div(f"Pred: {dst_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                html.Img(src=dst_img, style={"width": "100%", "height": "80px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #bfdbfe"}) if dst_img else html.Div("No photo", style={"height": "80px", "background": "#f8fafc", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "10px", "color": "#94a3b8"}),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
        html.Div(f"Misclassifications: {count:,} ({rate:.1%} of true {src_name})", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "whiteSpace": "normal", "wordBreak": "break-word"}),
    ], style={"padding": "10px", "width": "300px", "boxSizing": "border-box", "whiteSpace": "normal", "background": "white", "borderRadius": "8px"})


def make_sink_tooltip_content(class_id: int, fp_count: int, support: int, names: dict[int, str]):
    name = names.get(class_id, str(class_id))
    img_uri = get_class_image(class_id)

    return html.Div([
        html.Div(name, style={"fontWeight": "bold", "fontSize": "12px", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "normal", "wordBreak": "break-word"}),
        html.Img(src=img_uri, style={"width": "100%", "height": "115px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #bfdbfe", "marginBottom": "6px"}) if img_uri else None,
        html.Div(f"Falsely predicted for other classes: {fp_count:,} times", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "marginBottom": "2px"}),
        html.Div(f"Ground-truth support: {support:,} true images in split", style={"fontSize": "10px", "color": "#64748b"}),
    ], style={"padding": "10px", "width": "240px", "boxSizing": "border-box", "whiteSpace": "normal", "background": "white", "borderRadius": "8px"})


# -----------------------------------------------------------------------------
# DASH APPLICATION
# -----------------------------------------------------------------------------

def build_side_panel(
    subset: pd.DataFrame,
    stats: pd.DataFrame,
    edges: pd.DataFrame,
    mistakes: pd.DataFrame,
    names: dict[int, str],
    focus: int | None,
) -> html.Div:
    if focus is None:
        top_pair_preview = None
        if not edges.empty:
            top_p = edges.nlargest(1, "count").iloc[0]
            src_id = int(top_p["source"])
            dst_id = int(top_p["target"])
            src_name = names.get(src_id, str(src_id))
            dst_name = names.get(dst_id, str(dst_id))
            src_img = get_class_image(src_id)
            dst_img = get_class_image(dst_id)
            p_cnt = int(top_p["count"])
            p_loc = getattr(top_p, "top_loc", "Unknown")
            p_pct = float(getattr(top_p, "top_loc_pct", 0.0))

            top_pair_preview = html.Div([
                html.Div("Most Frequent Confusion in Split (Visual Comparison):", style={"fontWeight": "600", "fontSize": "13px", "color": "#0f172a", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div(f"True: {src_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#ea580c", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                        html.Img(src=src_img, style={"width": "100%", "height": "110px", "objectFit": "cover", "borderRadius": "6px", "border": "1px solid #fed7aa"}) if src_img else html.Div("No photo", style={"height": "110px", "background": "#f1f5f9", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "11px", "color": "#94a3b8"}),
                    ], style={"flex": "1", "minWidth": "0"}),
                    html.Div("→", style={"alignSelf": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                    html.Div([
                        html.Div(f"Predicted: {dst_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                        html.Img(src=dst_img, style={"width": "100%", "height": "110px", "objectFit": "cover", "borderRadius": "6px", "border": "1px solid #bfdbfe"}) if dst_img else html.Div("No photo", style={"height": "110px", "background": "#f1f5f9", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "11px", "color": "#94a3b8"}),
                    ], style={"flex": "1", "minWidth": "0"}),
                ], style={"display": "flex", "gap": "6px", "background": "#f8fafc", "padding": "10px", "borderRadius": "8px", "border": "1px solid #e2e8f0", "marginBottom": "8px"}),
                html.Div(f"Model misclassifies this pair {p_cnt:,} times ({p_pct:.0%} at Camera #{p_loc}). Compare appearance and habitat above.",
                         style={"fontSize": "11px", "color": "#64748b", "marginBottom": "12px"}),
            ])

        top_pairs = edges.nlargest(6, "count").iloc[::-1]
        pair_labels = [f"{names.get(int(r.source), str(r.source))} → {names.get(int(r.target), str(r.target))}" for r in top_pairs.itertuples()]
        pair_custom = [
            ["pair", int(r.source), int(r.target), int(r.count), float(r.rate)]
            for r in top_pairs.itertuples()
        ]
        fig_top_pairs = horizontal_bar(
            pair_labels, list(top_pairs["count"]), "#ea580c", "Misclassifications",
            customdata=pair_custom,
            hoverinfo="none",
            height=210,
        )

        sinks = stats.nlargest(6, "falsely_predicted_count").iloc[::-1]
        sink_labels = [names.get(int(idx), str(idx)) for idx in sinks.index]
        sink_custom = [
            ["sink", int(idx), int(sinks.loc[idx, "falsely_predicted_count"]), int(sinks.loc[idx, "support"] if "support" in sinks.columns else 0)]
            for idx in sinks.index
        ]
        fig_sinks = horizontal_bar(
            sink_labels, list(sinks["falsely_predicted_count"]), "#0284c7", "Falsely Predicted Count",
            customdata=sink_custom,
            hoverinfo="none",
            height=190,
        )

        return html.Div([
            html.Div([
                html.H3("Split-Wide Error Patterns", style={"margin": "0 0 4px 0", "fontSize": "16px"}),
                html.P("Global insights for unseen camera traps (OOD). Select any species to inspect individual confusions.",
                       style={"fontSize": "12px", "color": "#64748b", "margin": "0 0 12px 0"}),
            ]),
            top_pair_preview if top_pair_preview else html.Div(),
            html.H4("Top Confusion Pairs", style={"margin": "0", "fontSize": "13px", "color": "#334155"}),
            html.Div([
                dcc.Graph(id="fig-top-pairs", figure=fig_top_pairs, clear_on_unhover=True, config={"displayModeBar": False}),
                dcc.Tooltip(
                    id="tooltip-top-pairs",
                    direction="top",
                    background_color="white",
                    border_color="#cbd5e1",
                    zindex=99999,
                    style={"padding": "0px", "borderRadius": "8px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal"},
                ),
            ], style={"position": "relative"}),
            html.H4("Top 'Sink' Classes (Most Falsely Over-Predicted)", style={"margin": "10px 0 0 0", "fontSize": "13px", "color": "#334155"}),
            html.Div([
                dcc.Graph(id="fig-sinks", figure=fig_sinks, clear_on_unhover=True, config={"displayModeBar": False}),
                dcc.Tooltip(
                    id="tooltip-sinks",
                    direction="top",
                    background_color="white",
                    border_color="#cbd5e1",
                    zindex=99999,
                    style={"padding": "0px", "borderRadius": "8px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal"},
                ),
            ], style={"position": "relative"}),
        ])

    else:
        focus_name = names.get(focus, str(focus))
        focus_supp = int(stats.loc[focus, "support"]) if focus in stats.index else 0
        focus_errs = int(stats.loc[focus, "errors"]) if focus in stats.index else 0
        focus_rate = float(stats.loc[focus, "error_rate"]) if focus in stats.index else 0.0
        focus_img = get_class_image(focus)

        out_edges = edges.loc[edges["source"] == focus].nlargest(6, "count").iloc[::-1]
        out_labels = [names.get(int(r.target), str(r.target)) for r in out_edges.itertuples()]
        out_custom = [
            ["pair", int(focus), int(r.target), int(r.count), float(r.rate)]
            for r in out_edges.itertuples()
        ]
        fig_out = horizontal_bar(
            out_labels, list(out_edges["count"]) if not out_edges.empty else [],
            "#ea580c", "Outgoing Errors",
            customdata=out_custom,
            hoverinfo="none",
            height=190,
        )

        in_edges = edges.loc[edges["target"] == focus].nlargest(6, "count").iloc[::-1]
        in_labels = [names.get(int(r.source), str(r.source)) for r in in_edges.itertuples()]
        in_custom = [
            ["pair", int(r.source), int(focus), int(r.count), float(getattr(r, "rate", 0.0))]
            for r in in_edges.itertuples()
        ]
        fig_in = horizontal_bar(
            in_labels, list(in_edges["count"]) if not in_edges.empty else [],
            "#0284c7", "Incoming False Predictions",
            customdata=in_custom,
            hoverinfo="none",
            height=190,
        )

        focus_mistakes = mistakes.loc[mistakes["label"] == focus]
        cam_counts = focus_mistakes["location"].value_counts().head(5).iloc[::-1]
        fig_cam = horizontal_bar(
            [f"Camera #{c}" for c in cam_counts.index],
            list(cam_counts.values) if not cam_counts.empty else [],
            "#8b5cf6", "Errors at Camera",
            hovertemplate="<b>%{y}</b>: %{x:,} errors<extra></extra>",
            height=170,
        )

        photo_section = None
        if not out_edges.empty:
            top_dst = int(out_edges.iloc[-1]["target"])
            top_dst_name = names.get(top_dst, str(top_dst))
            top_dst_img = get_class_image(top_dst)
            top_err_cnt = int(out_edges.iloc[-1]["count"])

            photo_section = html.Div([
                html.Div("Visual Comparison: Selected Animal vs. Top Mistake", style={"fontWeight": "600", "fontSize": "13px", "color": "#0f172a", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div(f"True: {focus_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#ea580c", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                        html.Img(src=focus_img, style={"width": "100%", "height": "115px", "objectFit": "cover", "borderRadius": "6px", "border": "1px solid #fed7aa"}) if focus_img else html.Div("No photo", style={"height": "115px", "background": "#f1f5f9", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "11px", "color": "#94a3b8"}),
                    ], style={"flex": "1", "minWidth": "0"}),
                    html.Div("→", style={"alignSelf": "center", "fontSize": "20px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                    html.Div([
                        html.Div(f"Mistaken for: {top_dst_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                        html.Img(src=top_dst_img, style={"width": "100%", "height": "115px", "objectFit": "cover", "borderRadius": "6px", "border": "1px solid #bfdbfe"}) if top_dst_img else html.Div("No photo", style={"height": "115px", "background": "#f1f5f9", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "11px", "color": "#94a3b8"}),
                    ], style={"flex": "1", "minWidth": "0"}),
                ], style={"display": "flex", "gap": "6px", "background": "#f8fafc", "padding": "10px", "borderRadius": "8px", "border": "1px solid #e2e8f0", "marginBottom": "6px"}),
                html.Div(f"Top mistake: misclassified as {top_dst_name} {top_err_cnt:,} times.",
                         style={"fontSize": "11px", "color": "#64748b", "marginBottom": "12px"}),
            ])
        elif focus_img:
            photo_section = html.Div([
                html.Div("Representative Camera-Trap Photo:", style={"fontWeight": "600", "fontSize": "13px", "color": "#0f172a", "marginBottom": "6px"}),
                html.Img(src=focus_img, style={"width": "100%", "height": "140px", "objectFit": "cover", "borderRadius": "8px", "border": "1px solid #cbd5e1", "marginBottom": "12px"}),
            ])

        return html.Div([
            html.Div([
                html.H3(focus_name, style={"margin": "0 0 4px 0", "fontSize": "18px", "color": "#0f172a"}),
                html.Div(f"{focus_supp:,} true images | {focus_errs:,} errors | {focus_rate:.1%} error rate",
                         style={"fontSize": "13px", "fontWeight": "600", "color": "#ea580c" if focus_rate > 0.5 else "#334155", "marginBottom": "10px"}),
            ]),
            photo_section if photo_section else html.Div(),
            html.H4(f"Top Outgoing Errors ({focus_name} Mistaken For...)", style={"margin": "0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            html.Div([
                dcc.Graph(id="fig-out", figure=fig_out, clear_on_unhover=True, config={"displayModeBar": False}),
                dcc.Tooltip(
                    id="tooltip-out",
                    direction="top",
                    background_color="white",
                    border_color="#cbd5e1",
                    zindex=99999,
                    style={"padding": "0px", "borderRadius": "8px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal"},
                ),
            ], style={"position": "relative"}),
            html.H4(f"Top Incoming False Predictions (Mistaken as {focus_name})", style={"margin": "12px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            html.Div([
                dcc.Graph(id="fig-in", figure=fig_in, clear_on_unhover=True, config={"displayModeBar": False}),
                dcc.Tooltip(
                    id="tooltip-in",
                    direction="top",
                    background_color="white",
                    border_color="#cbd5e1",
                    zindex=99999,
                    style={"padding": "0px", "borderRadius": "8px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal"},
                ),
            ], style={"position": "relative"}),
            html.H4("Top Camera Locations by Error Count", style={"margin": "12px 0 4px 0", "fontSize": "13px", "fontWeight": "600", "color": "#334155"}),
            dcc.Graph(figure=fig_cam, config={"displayModeBar": False}),
        ])


def build_kpis(subset: pd.DataFrame, stats: pd.DataFrame, edges: pd.DataFrame, filtered: pd.DataFrame, names: dict[int, str]) -> list:
    total_images = len(subset)
    total_errors = int((subset["label"] != subset["prediction"]).sum())
    acc = (total_images - total_errors) / total_images if total_images else 0.0

    top_pair_text = "N/A"
    top_pair_popup = None
    if not edges.empty:
        top_p = edges.nlargest(1, "count").iloc[0]
        src_id = int(top_p["source"])
        dst_id = int(top_p["target"])
        cnt_val = int(top_p["count"])
        src_name = names.get(src_id, str(src_id))
        dst_name = names.get(dst_id, str(dst_id))
        src_img = get_class_image(src_id)
        dst_img = get_class_image(dst_id)
        top_pair_text = f"{src_name} → {dst_name} ({cnt_val:,} errors)"

        top_pair_popup = html.Div([
            html.Div("Visual Comparison (Split's #1 Misclassification):", style={"fontWeight": "700", "fontSize": "11px", "color": "#0f172a", "marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Div(f"True: {src_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "marginBottom": "3px"}),
                    html.Img(src=src_img, style={"width": "100%", "height": "90px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #fed7aa"}) if src_img else html.Div("No photo", style={"height": "90px", "background": "#f8fafc", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "10px", "color": "#94a3b8"}),
                ], style={"flex": "1", "minWidth": "0"}),
                html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                html.Div([
                    html.Div(f"Mistaken for: {dst_name}", style={"fontSize": "10px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "marginBottom": "3px"}),
                    html.Img(src=dst_img, style={"width": "100%", "height": "90px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #bfdbfe"}) if dst_img else html.Div("No photo", style={"height": "90px", "background": "#f8fafc", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "10px", "color": "#94a3b8"}),
                ], style={"flex": "1", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
            html.Div(f"Model misclassifies this pair {cnt_val:,} times in this split.", style={"fontSize": "10px", "color": "#64748b"}),
        ], className="kpi-popup", style={"width": "300px"})

    top_sink_text = "N/A"
    top_sink_popup = None
    if not stats.empty and "falsely_predicted_count" in stats.columns:
        sink_id = int(stats["falsely_predicted_count"].idxmax())
        sink_cnt = int(stats.loc[sink_id, "falsely_predicted_count"])
        sink_supp = int(stats.loc[sink_id, "support"]) if "support" in stats.columns else 0
        sink_name = names.get(sink_id, str(sink_id))
        sink_img = get_class_image(sink_id)
        top_sink_text = f"{sink_name} ({sink_cnt:,} times)"

        top_sink_popup = html.Div([
            html.Div("Most Falsely Over-Predicted Species:", style={"fontWeight": "700", "fontSize": "11px", "color": "#0f172a", "marginBottom": "4px"}),
            html.Div(sink_name, style={"fontSize": "11px", "fontWeight": "700", "color": "#0284c7", "marginBottom": "4px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Img(src=sink_img, style={"width": "100%", "height": "110px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #bfdbfe", "marginBottom": "6px"}) if sink_img else html.Div("No photo", style={"height": "110px", "background": "#f8fafc", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontSize": "10px", "color": "#94a3b8"}),
            html.Div(f"Falsely predicted {sink_cnt:,} times for other classes ({sink_supp:,} true images in split).", style={"fontSize": "10px", "color": "#64748b", "lineHeight": "1.35"}),
        ], className="kpi-popup", style={"width": "220px"})

    kpi_card_style = {
        "background": "#ffffff", "padding": "12px 18px", "borderRadius": "10px",
        "border": "1px solid #e2e8f0", "flex": "1 1 200px", "minWidth": "160px",
        "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
    }
    kpi_title_style = {"fontSize": "11px", "color": "#64748b", "fontWeight": "600", "textTransform": "uppercase"}
    kpi_val_style = {"fontSize": "18px", "fontWeight": "700", "color": "#0f172a", "marginTop": "2px"}

    return [
        html.Div([
            html.Div("Split Accuracy", style=kpi_title_style),
            html.Div(f"{acc:.1%} ({total_errors:,} errors / {total_images:,})", style=kpi_val_style),
        ], style=kpi_card_style),
        html.Div([
            html.Div([
                html.Span("Top Confusion Pair", style=kpi_title_style),
                html.Span("Preview", style={"fontSize": "10px", "color": "#64748b", "background": "#f8fafc", "border": "1px solid #e2e8f0", "padding": "1px 6px", "borderRadius": "4px", "marginLeft": "6px", "textTransform": "none", "fontWeight": "500"}) if top_pair_popup else None,
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(top_pair_text, style={**kpi_val_style, "fontSize": "14px", "color": "#ea580c"}),
            top_pair_popup if top_pair_popup else None,
        ], className="kpi-card-hover" if top_pair_popup else "", style=kpi_card_style),
        html.Div([
            html.Div([
                html.Span("Top Over-Predicted Sink", style=kpi_title_style),
                html.Span("Preview", style={"fontSize": "10px", "color": "#64748b", "background": "#f8fafc", "border": "1px solid #e2e8f0", "padding": "1px 6px", "borderRadius": "4px", "marginLeft": "6px", "textTransform": "none", "fontWeight": "500"}) if top_sink_popup else None,
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(top_sink_text, style={**kpi_val_style, "fontSize": "14px", "color": "#0284c7"}),
            top_sink_popup if top_sink_popup else None,
        ], className="kpi-card-hover" if top_sink_popup else "", style=kpi_card_style),
        html.Div([
            html.Div("Active Filtered Graph", style=kpi_title_style),
            html.Div(f"{len(filtered)} directed arrows | {len(set(filtered['source']) | set(filtered['target']))} species", style={**kpi_val_style, "fontSize": "14px"}),
        ], style=kpi_card_style),
    ]


def create_app(data: pd.DataFrame):
    names = class_names(data)
    splits = sorted(
        data["split"].unique(),
        key=lambda v: list(SPLIT_DESCRIPTIONS).index(v) if v in SPLIT_DESCRIPTIONS else 999,
    )
    default_split = "val" if "val" in splits else splits[0]

    @lru_cache(maxsize=10)
    def get_split(split: str):
        subset = data.loc[data["split"] == split]
        stats, edges, mistakes = summarize_split(subset)
        return subset, stats, edges, mistakes

    # Pre-render initial figures and side content so page is never empty
    init_subset, init_stats, init_edges, init_mistakes = get_split(default_split)
    init_filtered = visible_edges(init_edges, names, 30, 5, 1.0, 25, "count", False, None)
    init_fig = network_figure(init_filtered, init_stats, names, None)
    init_kpis = build_kpis(init_subset, init_stats, init_edges, init_filtered, names)
    init_side = build_side_panel(init_subset, init_stats, init_edges, init_mistakes, names, None)

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "iWildCam | Species Confusion & Domain Shift"
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
    display: none;
    position: absolute;
    top: calc(100% + 8px);
    left: 0;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    padding: 12px;
    z-index: 99999;
    pointer-events: none;
    white-space: normal;
    text-align: left;
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
.kpi-card-hover:hover .kpi-popup {
    display: block;
}
</style></head>""",
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

    def slider_control(title, component, val_id, initial_val):
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(title, style=label_style),
                        html.Span(
                            initial_val,
                            id=val_id,
                            style={
                                "background": "#eff6ff",
                                "color": "#1d4ed8",
                                "border": "1px solid #bfdbfe",
                                "padding": "1px 8px",
                                "borderRadius": "6px",
                                "fontWeight": "700",
                                "fontSize": "12px",
                                "minWidth": "42px",
                                "textAlign": "center",
                                "display": "inline-block",
                            },
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "4px"},
                ),
                component,
            ],
            style={"flex": "1 1 200px", "minWidth": "170px"},
        )

    app.layout = html.Div([
        # Top Header
        html.Div([
            html.H2("iWildCam: Species Confusion Explorer & Domain Shift", style={"margin": "0 0 6px 0", "color": "#0f172a"}),
            html.P(
                "Analyze directional misclassifications and species confusion clusters in ResNet-50.",
                style={"margin": "0", "color": "#64748b", "fontSize": "14px"},
            ),
        ], style={"marginBottom": "16px"}),

        # KPI Metrics Row
        html.Div(id="kpi-banner", children=init_kpis, style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px"}),

        # Control Panel Card
        html.Div([
            html.Div([
                control("Dataset split", dcc.Dropdown(
                    id="split", clearable=False, value=default_split,
                    options=[{"label": f"{s} ({SPLIT_DESCRIPTIONS.get(s, s)})", "value": s} for s in splits],
                )),
                control("View Visualization", dcc.RadioItems(
                    id="view-mode",
                    options=[
                        {"label": " Directed Network", "value": "network"},
                        {"label": " Confusion Matrix", "value": "matrix"},
                    ],
                    value="network",
                    inline=True,
                    style={"fontSize": "13px", "fontWeight": "500", "paddingTop": "6px"},
                )),
                html.Div([
                    html.Div("Focus Species (Click node to select)", style=label_style),
                    html.Div([
                        html.Div(
                            dcc.Dropdown(
                                id="focus", clearable=True, placeholder="All species (global overview)",
                                options=[{
                                    "label": f"{name} [{idx}]",
                                    "value": idx
                                } for idx, name in sorted(names.items(), key=lambda kv: kv[1].lower())],
                            ),
                            style={"flex": "1", "minWidth": "0"},
                        ),
                        html.Button(
                            "Reset Focus", id="reset-focus-btn", n_clicks=0,
                            title="Reset focus to all species",
                            style={
                                "background": "#ffffff", "border": "1px solid #cbd5e1",
                                "color": "#334155", "padding": "0 14px", "borderRadius": "8px",
                                "fontWeight": "600", "cursor": "pointer", "fontSize": "13px",
                                "whiteSpace": "nowrap", "height": "36px", "display": "flex",
                                "alignItems": "center", "flexShrink": "0",
                                "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
                            },
                        ),
                    ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
                ], style={"flex": "1.5 1 280px", "minWidth": "240px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "14px"}),

            html.Div([
                control("Rank Confusion Arrows By", dcc.Dropdown(
                    id="rank", clearable=False, value="count", options=[
                        {"label": "Confusion Count (Total errors)", "value": "count"},
                        {"label": "Confusion Rate (% of true class)", "value": "rate"},
                    ],
                )),
                slider_control(
                    "Min Ground-Truth Images",
                    dcc.Slider(
                        id="min-support", min=1, max=200, step=1, value=30,
                        marks={1: "1", 50: "50", 100: "100", 150: "150", 200: "200"},
                    ),
                    "min-support-val", "30",
                ),
                slider_control(
                    "Min Confusion Count",
                    dcc.Slider(
                        id="min-count", min=1, max=50, step=1, value=5,
                        marks={1: "1", 10: "10", 20: "20", 35: "35", 50: "50"},
                    ),
                    "min-count-val", "5",
                ),
                slider_control(
                    "Min Confusion Rate (%)",
                    dcc.Slider(
                        id="min-rate", min=0, max=25, step=0.5, value=1.0,
                        marks={0: "0%", 5: "5%", 10: "10%", 15: "15%", 20: "20%", 25: "25%"},
                    ),
                    "min-rate-val", "1.0%",
                ),
                slider_control(
                    "Max Arrows Displayed",
                    dcc.Slider(
                        id="top-edges", min=5, max=50, step=5, value=25,
                        marks={5: "5", 15: "15", 25: "25", 35: "35", 50: "50"},
                    ),
                    "top-edges-val", "25",
                ),
                control("Empty Species", dcc.Checklist(
                    id="show-empty", options=[{"label": " Show 'empty' confusions", "value": "yes"}], value=[],
                    style={"fontSize": "13px", "paddingTop": "8px"},
                )),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
        ], style={**card_style, "marginBottom": "16px"}),

        # Main Row
        html.Div([
            # Left Card: Network or Heatmap
            html.Div([
                html.Div([
                    html.H3(id="main-graph-title", children="Species Confusion Network (val) - Community Clusters", style={"margin": "0 0 4px 0", "fontSize": "16px"}),
                    html.P(id="main-graph-subtitle", children="Directed arrows point True Species → Misclassified Species. Node color = Error rate. Click any node to inspect.", style={"margin": "0", "fontSize": "12px", "color": "#64748b"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id="network", figure=init_fig, clear_on_unhover=True, config={"displaylogo": False, "responsive": True}),
                        type="dot",
                    ),
                    dcc.Tooltip(
                        id="graph-tooltip",
                        direction="right",
                        background_color="white",
                        border_color="#cbd5e1",
                        zindex=9999,
                        style={"padding": "0px", "borderRadius": "10px", "boxShadow": "0 8px 24px rgba(0,0,0,0.15)", "whiteSpace": "normal"},
                    ),
                ], style={"position": "relative"}),
            ], style={**card_style, "flex": "2 1 640px", "minWidth": "480px"}),

            # Right Card: Insights
            html.Div(id="side-panel-content", children=init_side, style={**card_style, "flex": "1 1 380px", "minWidth": "350px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Footer / Metric Definitions & Formulas
        html.Div([
            html.Div("Metric Definitions & Mathematical Formulations", style={"fontWeight": "700", "fontSize": "13px", "color": "#0f172a", "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    html.Span("• Species Error Rate (node color): ", style={"fontWeight": "600", "color": "#0f172a"}),
                    html.Span("Fraction of ground-truth images misclassified by the model: "),
                    html.Span("Errors ÷ Ground-Truth Support (1 − Recall)", style={"fontFamily": "monospace", "background": "#f1f5f9", "padding": "2px 6px", "borderRadius": "4px", "fontWeight": "600", "color": "#0f172a"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Span("• Confusion Rate (arrow thickness): ", style={"fontWeight": "600", "color": "#0f172a"}),
                    html.Span("Fraction of species A specifically mistaken for species B: "),
                    html.Span("Mistakes (A → B) ÷ Ground-Truth Support of A", style={"fontFamily": "monospace", "background": "#f1f5f9", "padding": "2px 6px", "borderRadius": "4px", "fontWeight": "600", "color": "#0f172a"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Span("• Falsely Predicted (sink count): ", style={"fontWeight": "600", "color": "#0f172a"}),
                    html.Span("Total false positive predictions — how often the model predicted this species when another animal was actually present."),
                ]),
            ], style={"fontSize": "12px", "color": "#334155", "lineHeight": "1.6"}),
        ], style={
            "padding": "16px 20px", "background": "#ffffff", "borderRadius": "10px",
            "border": "1px solid #e2e8f0", "marginTop": "16px", "boxShadow": "0 1px 2px rgba(0,0,0,0.03)",
        }),

    ], style={"maxWidth": "1600px", "margin": "20px auto", "padding": "0 24px", "background": "#f8fafc", "fontFamily": "Inter, Arial, sans-serif"})

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------

    @app.callback(
        Output("focus", "value"),
        Input("network", "clickData"),
        Input("reset-focus-btn", "n_clicks"),
        State("focus", "value"),
        prevent_initial_call=True,
    )
    def update_focus(click_data, reset_clicks, current_focus):
        trigger = ctx.triggered_id
        if trigger == "reset-focus-btn":
            return None
        if click_data and click_data.get("points"):
            pt = click_data["points"][0]
            tag = pt.get("customdata")
            if isinstance(tag, (list, tuple)) and len(tag) >= 2:
                if tag[0] == "node":
                    clicked = int(tag[1])
                    return None if clicked == current_focus else clicked
                elif tag[0] == "edge":
                    src = int(tag[1])
                    return src
        return no_update

    # Callback: Slider Text Readouts
    @app.callback(
        Output("min-support-val", "children"),
        Output("min-count-val", "children"),
        Output("min-rate-val", "children"),
        Output("top-edges-val", "children"),
        Input("min-support", "value"),
        Input("min-count", "value"),
        Input("min-rate", "value"),
        Input("top-edges", "value"),
    )
    def update_slider_readouts(support, count, rate, limit):
        return (
            f"{int(support):,}",
            f"{int(count):,}",
            f"{float(rate):.1f}%",
            f"{int(limit):,}",
        )

    # Callback 1: Main Graph & Title
    @app.callback(
        Output("network", "figure"),
        Output("main-graph-title", "children"),
        Output("main-graph-subtitle", "children"),
        Input("split", "value"),
        Input("focus", "value"),
        Input("view-mode", "value"),
        Input("rank", "value"),
        Input("show-empty", "value"),
        Input("min-support", "value"),
        Input("min-count", "value"),
        Input("min-rate", "value"),
        Input("top-edges", "value"),
    )
    def update_main_graph(split, focus, view_mode, rank, include_empty, support, count, rate, limit):
        subset, stats, edges, _ = get_split(split)
        filtered = visible_edges(
            edges, names, int(support), int(count), float(rate), int(limit),
            rank, bool(include_empty), focus,
        )
        if view_mode == "matrix":
            fig = confusion_matrix_figure(filtered, names, focus)
            title = f"Confusion Matrix Heatmap ({split})"
            sub = "Rows: True Ground-Truth Species | Columns: Model Predictions. Off-diagonal numbers show error volumes."
        else:
            fig = network_figure(filtered, stats, names, focus)
            title = f"Species Confusion Network ({split}) - Community Clusters"
            sub = "Directed arrows point True Species → Misclassified Species. Node color = Error rate. Click any node to inspect."
        return fig, title, sub

    # Callback 2: KPIs & Side Insights Panel
    @app.callback(
        Output("kpi-banner", "children"),
        Output("side-panel-content", "children"),
        Input("split", "value"),
        Input("focus", "value"),
        Input("rank", "value"),
        Input("show-empty", "value"),
        Input("min-support", "value"),
        Input("min-count", "value"),
        Input("min-rate", "value"),
        Input("top-edges", "value"),
    )
    def update_insights(split, focus, rank, include_empty, support, count, rate, limit):
        subset, stats, edges, mistakes = get_split(split)
        filtered = visible_edges(
            edges, names, int(support), int(count), float(rate), int(limit),
            rank, bool(include_empty), focus,
        )
        kpis = build_kpis(subset, stats, edges, filtered, names)
        side = build_side_panel(subset, stats, edges, mistakes, names, focus)
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
            img_uri = get_class_image(class_id)
            _, stats, _, _ = get_split(split)

            if class_id in stats.index:
                supp = int(stats.loc[class_id, "support"])
                errs = int(stats.loc[class_id, "errors"])
                rate = float(stats.loc[class_id, "error_rate"])
                fp = int(stats.loc[class_id, "falsely_predicted_count"])
            else:
                supp, errs, rate, fp = 0, 0, 0.0, 0

            content = html.Div([
                html.Img(
                    src=img_uri,
                    style={
                        "width": "100%", "height": "135px", "objectFit": "cover",
                        "borderRadius": "6px", "marginBottom": "8px",
                        "border": "1px solid #e2e8f0", "display": "block",
                    }
                ) if img_uri else None,
                html.Div([
                    html.Div(name, style={"fontWeight": "bold", "fontSize": "13px", "color": "#0f172a", "marginBottom": "4px", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div(f"Class ID: {class_id} | Support: {supp:,} true images", style={"fontSize": "11px", "color": "#64748b", "marginBottom": "3px", "whiteSpace": "normal"}),
                    html.Div(f"Error Rate: {rate:.1%} ({errs:,} errors / {supp:,} true images)", style={"fontSize": "11px", "fontWeight": "600", "color": "#dc2626" if rate > 0.4 else "#16a34a", "marginBottom": "3px", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div(f"Falsely predicted for other classes: {fp:,} times", style={"fontSize": "10px", "color": "#64748b", "marginBottom": "4px", "whiteSpace": "normal", "wordBreak": "break-word"}),
                    html.Div("Click node to inspect details in sidebar", style={"fontSize": "10px", "color": "#2563eb", "fontStyle": "italic"}),
                ]),
            ], style={
                "padding": "12px",
                "width": "280px",
                "boxSizing": "border-box",
                "whiteSpace": "normal",
                "background": "white",
                "borderRadius": "10px",
            })

            return True, bbox, content, direction

        elif tag_type == "edge":
            src = int(tag[1])
            dst = int(tag[2])
            src_name = names.get(src, str(src))
            dst_name = names.get(dst, str(dst))
            src_img = get_class_image(src)
            dst_img = get_class_image(dst)

            count = tag[3] if len(tag) > 3 else "N/A"
            rate = float(tag[4]) if len(tag) > 4 else 0.0
            top_loc = tag[5] if len(tag) > 5 else "Unknown"
            top_loc_pct = float(tag[6]) if len(tag) > 6 else 0.0

            content = html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"True: {src_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#ea580c", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                        html.Img(src=src_img, style={"width": "100%", "height": "85px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #fed7aa"}) if src_img else None,
                    ], style={"flex": "1", "minWidth": "0"}),
                    html.Div("→", style={"alignSelf": "center", "fontSize": "16px", "fontWeight": "bold", "color": "#64748b", "padding": "0 6px"}),
                    html.Div([
                        html.Div(f"Pred: {dst_name}", style={"fontSize": "11px", "fontWeight": "700", "color": "#0284c7", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.25", "marginBottom": "3px"}),
                        html.Img(src=dst_img, style={"width": "100%", "height": "85px", "objectFit": "cover", "borderRadius": "4px", "border": "1px solid #bfdbfe"}) if dst_img else None,
                    ], style={"flex": "1", "minWidth": "0"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "8px"}),
                html.Div(f"Misclassifications: {count:,} ({rate:.1%} of true {src_name})", style={"fontSize": "11px", "fontWeight": "600", "color": "#0f172a", "marginBottom": "3px", "whiteSpace": "normal", "wordBreak": "break-word", "lineHeight": "1.35"}),
                html.Div(f"Camera #{top_loc}: {top_loc_pct:.0%} of this error", style={"fontSize": "10px", "color": "#64748b"}),
            ], style={
                "padding": "12px",
                "width": "320px",
                "boxSizing": "border-box",
                "whiteSpace": "normal",
                "background": "white",
                "borderRadius": "10px",
            })

            return True, bbox, content, direction

        return False, no_update, no_update, no_update

    # Callback 4: Sidebar Bar Chart Hover Tooltips
    def _handle_bar_tooltip(hover_data, kind):
        if not hover_data or not hover_data.get("points"):
            return False, no_update, no_update, no_update
        pt = hover_data["points"][0]
        bbox = pt.get("bbox")
        tag = pt.get("customdata")
        if not tag or not isinstance(tag, (list, tuple)) or not bbox:
            return False, no_update, no_update, no_update
        y0 = bbox.get("y0", 0)
        direction = "bottom" if y0 < 65 else "top"
        if kind == "pair" and len(tag) >= 5:
            content = make_pair_tooltip_content(int(tag[1]), int(tag[2]), int(tag[3]), float(tag[4]), names)
            return True, bbox, content, direction
        elif kind == "sink" and len(tag) >= 4:
            content = make_sink_tooltip_content(int(tag[1]), int(tag[2]), int(tag[3]), names)
            return True, bbox, content, direction
        return False, no_update, no_update, no_update

    @app.callback(
        Output("tooltip-top-pairs", "show"),
        Output("tooltip-top-pairs", "bbox"),
        Output("tooltip-top-pairs", "children"),
        Output("tooltip-top-pairs", "direction"),
        Input("fig-top-pairs", "hoverData"),
    )
    def update_tooltip_top_pairs(h):
        return _handle_bar_tooltip(h, "pair")

    @app.callback(
        Output("tooltip-sinks", "show"),
        Output("tooltip-sinks", "bbox"),
        Output("tooltip-sinks", "children"),
        Output("tooltip-sinks", "direction"),
        Input("fig-sinks", "hoverData"),
    )
    def update_tooltip_sinks(h):
        return _handle_bar_tooltip(h, "sink")

    @app.callback(
        Output("tooltip-out", "show"),
        Output("tooltip-out", "bbox"),
        Output("tooltip-out", "children"),
        Output("tooltip-out", "direction"),
        Input("fig-out", "hoverData"),
    )
    def update_tooltip_out(h):
        return _handle_bar_tooltip(h, "pair")

    @app.callback(
        Output("tooltip-in", "show"),
        Output("tooltip-in", "bbox"),
        Output("tooltip-in", "children"),
        Output("tooltip-in", "direction"),
        Input("fig-in", "hoverData"),
    )
    def update_tooltip_in(h):
        return _handle_bar_tooltip(h, "pair")

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
