"""Interactive black-box species confusion explorer for WILDS iWildCam.

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


COLUMNS = [
    "id", "split", "label", "label_name", "prediction",
    "prediction_name", "confidence", "location", "filename",
]
SPLIT_DESCRIPTIONS = {
    "id_val": "ID validation (seen cameras)",
    "val": "OOD validation (unseen cameras)",
    "id_test": "ID test (seen cameras)",
    "test": "OOD test (unseen cameras)",
    "train": "Training images",
}


def read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Run `make inference` from your repository root first, "
            "or provide --predictions PATH."
        )
    # Don't load prob_0 ... prob_181. This app only needs labels and metadata.
    data = pd.read_parquet(path, columns=COLUMNS)
    if data.empty:
        raise ValueError("Prediction file contains no rows.")
    data = data.dropna(subset=["split", "label", "prediction"])
    data["split"] = data["split"].astype(str)
    data["label"] = data["label"].astype(int)
    data["prediction"] = data["prediction"].astype(int)
    return data


def class_names(data: pd.DataFrame) -> dict[int, str]:
    """Map class IDs to names, including classes seen only as predictions."""
    result: dict[int, str] = {}
    for id_col, name_col in [
        ("label", "label_name"), ("prediction", "prediction_name")
    ]:
        for class_id, name in data[[id_col, name_col]].drop_duplicates().itertuples(index=False, name=None):
            if pd.notna(name) and str(name).strip():
                result[int(class_id)] = str(name)
    return result


def summarize_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-class stats and directed wrong-label pairs.

    A -> B means ground truth A and prediction B. The denominator is ALWAYS
    the number of real A images in the entire split, even when `empty` is hidden.
    """
    support = data.groupby("label").size().rename("support")
    mistakes = data.loc[data["label"] != data["prediction"]]
    error_counts = mistakes.groupby("label").size().rename("errors")
    stats = support.to_frame().join(error_counts).fillna({"errors": 0})
    stats[["support", "errors"]] = stats[["support", "errors"]].astype(int)
    stats["error_rate"] = stats["errors"] / stats["support"]

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
    return stats, edges


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
    # Stable tie-breaking makes screenshots and comparisons reproducible.
    order = ["rate", "count", "source", "target"] if sort_by == "rate" else ["count", "rate", "source", "target"]
    return result.sort_values(order, ascending=[False, False, True, True]).head(max_edges)


def network_figure(
    edges: pd.DataFrame, stats: pd.DataFrame, names: dict[int, str],
    focused_class: int | None,
) -> go.Figure:
    fig = go.Figure()
    if edges.empty:
        fig.add_annotation(
            text="No confusion pairs match these filters.<br>Try reducing the thresholds or including 'empty'.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 15},
        )
        fig.update_layout(
            height=630, xaxis={"visible": False}, yaxis={"visible": False},
            margin={"l": 15, "r": 15, "t": 35, "b": 20},
        )
        return fig

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), weight=float(np.log1p(row.count)))

    # The fixed seed ensures reproducible layout for a given graph.
    positions = nx.spring_layout(graph, seed=42, weight="weight", iterations=120)
    max_count = float(edges["count"].max())

    for row in edges.itertuples(index=False):
        src, dst = int(row.source), int(row.target)
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        if focused_class is None:
            color = "rgba(98, 116, 135, 0.48)"
        else:
            color = "#e18d4a" if src == focused_class else "#5287b9"
        width = 1.1 + 6 * np.sqrt(float(row.count) / max_count)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line={"width": width, "color": color},
            hoverinfo="skip", showlegend=False,
        ))
        # Arrowhead placed part-way along the edge: true class -> predicted class.
        fig.add_annotation(
            x=x0 * 0.36 + x1 * 0.64,
            y=y0 * 0.36 + y1 * 0.64,
            ax=x0 * 0.52 + x1 * 0.48,
            ay=y0 * 0.52 + y1 * 0.48,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.25,
            arrowwidth=2.2, arrowcolor=color,
        )
        # A transparent midpoint marker makes each edge easy to hover.
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode="markers", marker={"size": 17, "color": "rgba(0,0,0,0.01)"},
            customdata=[["edge", src, dst]],
            hovertemplate=(
                f"<b>{names.get(src, str(src))} → {names.get(dst, str(dst))}</b>"
                f"<br>Wrong predictions: {row.count:,}"
                f"<br>Confusion rate: {row.rate:.1%}"
                f"<br>Total images of true species: {row.source_support:,}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    node_ids = list(graph.nodes)
    node_sizes = []
    node_colors = []
    hover_labels = []
    node_labels = []
    for class_id in node_ids:
        if class_id in stats.index:
            support = int(stats.loc[class_id, "support"])
            error_count = int(stats.loc[class_id, "errors"])
            error_rate = float(stats.loc[class_id, "error_rate"])
        else:
            support, error_count, error_rate = 0, 0, float("nan")
        node_sizes.append(float(np.clip(17 + 4 * np.log1p(support), 18, 45)))
        node_colors.append(error_rate)
        name = names.get(class_id, str(class_id))
        node_labels.append(name if len(name) <= 19 else name[:16] + "…")
        hover_labels.append(
            f"<b>{name}</b><br>Ground-truth images: {support:,}"
            f"<br>Wrong predictions: {error_count:,}"
            f"<br>Error rate: {error_rate:.1%}" if support else
            f"<b>{name}</b><br>No ground-truth examples in this split"
        )
    fig.add_trace(go.Scatter(
        x=[positions[k][0] for k in node_ids],
        y=[positions[k][1] for k in node_ids],
        mode="markers+text",
        text=node_labels, textposition="top center",
        textfont={"size": 10},
        hovertext=hover_labels,
        customdata=[["node", int(k)] for k in node_ids],
        hovertemplate="%{hovertext}<extra></extra>",
        marker={
            "size": node_sizes,
            "color": node_colors,
            "colorscale": "YlOrRd", "cmin": 0, "cmax": 1,
            "showscale": True,
            "colorbar": {"title": {"text": "Error rate"}, "tickformat": ".0%", "len": 0.68},
            "line": {"width": [3 if k == focused_class else 1 for k in node_ids],
                     "color": ["#265e9c" if k == focused_class else "#ffffff" for k in node_ids]},
            "opacity": 0.97,
        },
        showlegend=False,
    ))
    fig.update_layout(
        height=630, margin={"l": 15, "r": 45, "t": 30, "b": 15},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False,
               "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest", clickmode="event+select",
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif"},
    )
    return fig


def bar_figure(
    edges: pd.DataFrame, names: dict[int, str], focus: int,
    direction: str, show_empty: bool,
) -> go.Figure:
    """For outgoing: true focus -> predicted class. For incoming: true class -> predicted focus."""
    view = edges.loc[edges["source"] == focus] if direction == "out" else edges.loc[edges["target"] == focus]
    if not show_empty:
        empty = {k for k, v in names.items() if v.lower().strip() == "empty"}
        view = view.loc[~view["source"].isin(empty) & ~view["target"].isin(empty)]
    view = view.nlargest(8, "count").iloc[::-1]
    partner_col = "target" if direction == "out" else "source"
    labels = [names.get(int(v), str(v)) for v in view[partner_col]]
    fig = go.Figure(go.Bar(
        x=view["count"], y=labels, orientation="h",
        marker_color="#e18d4a" if direction == "out" else "#5287b9",
        customdata=np.column_stack([view["rate"], view["source_support"]]) if not view.empty else [],
        hovertemplate=(
            "%{y}<br>Confusions: %{x:,}"
            "<br>Rate within true class: %{customdata[0]:.1%}"
            "<br>True-class support: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=270, margin={"l": 130, "r": 10, "t": 20, "b": 35},
        xaxis_title="Incorrect predictions", yaxis={"automargin": True},
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def create_app(data: pd.DataFrame):
    from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

    names = class_names(data)
    splits = sorted(data["split"].unique(), key=lambda v: list(SPLIT_DESCRIPTIONS).index(v) if v in SPLIT_DESCRIPTIONS else 999)
    default_split = "val" if "val" in splits else splits[0]

    @lru_cache(maxsize=8)
    def get_split(split: str):
        subset = data.loc[data["split"] == split]
        stats, edges = summarize_split(subset)
        return subset, stats, edges

    app = Dash(__name__)
    app.title = "iWildCam | Species Confusion Network"
    label_style = {"fontWeight": "600", "fontSize": "13px", "marginBottom": "5px"}
    card_style = {"background": "white", "padding": "15px", "borderRadius": "10px", "border": "1px solid #e0e5ec"}

    def control(title, component):
        return html.Div([html.Div(title, style=label_style), component], style={"flex": "1 1 195px", "minWidth": "165px"})

    app.layout = html.Div([
        html.H2("iWildCam: Species Confusion Network", style={"marginBottom": "4px"}),
        html.P("Each arrow points from the true species to the incorrectly predicted species. "
               "Click a node to investigate its confusions.", style={"color": "#526174"}),
        html.Div([
            control("Dataset split", dcc.Dropdown(
                id="split", clearable=False, value=default_split,
                options=[{"label": SPLIT_DESCRIPTIONS.get(s, s), "value": s} for s in splits],
            )),
            control("Focus species", dcc.Dropdown(
                id="focus", clearable=True, placeholder="All species (click a node)",
                options=[{"label": f"{name} ({idx})", "value": idx}
                         for idx, name in sorted(names.items(), key=lambda kv: kv[1].lower())],
            )),
            control("Rank confusion arrows by", dcc.Dropdown(
                id="rank", clearable=False, value="count", options=[
                    {"label": "Number of wrong predictions", "value": "count"},
                    {"label": "Confusion rate (% of true class)", "value": "rate"},
                ],
            )),
            control("Include 'empty' species", dcc.Checklist(
                id="show-empty", options=[{"label": "Show empty-class connections", "value": "yes"}], value=[],
            )),
        ], style={**card_style, "display": "flex", "flexWrap": "wrap", "gap": "15px", "marginBottom": "12px"}),
        html.Div([
            control("Minimum true-species images", dcc.Slider(
                id="min-support", min=1, max=200, step=1, value=30,
                marks={1: "1", 50: "50", 100: "100", 200: "200"},
            )),
            control("Minimum confusion count", dcc.Slider(
                id="min-count", min=1, max=50, step=1, value=5,
                marks={1: "1", 10: "10", 25: "25", 50: "50"},
            )),
            control("Minimum confusion rate (%)", dcc.Slider(
                id="min-rate", min=0, max=25, step=0.5, value=1,
                marks={0: "0", 5: "5", 10: "10", 25: "25"},
            )),
            control("Maximum arrows shown", dcc.Slider(
                id="top-edges", min=5, max=60, step=5, value=25,
                marks={5: "5", 25: "25", 40: "40", 60: "60"},
            )),
        ], style={**card_style, "display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "12px"}),
        html.Div(id="network-stats", style={"margin": "10px 0", "fontWeight": "600"}),
        html.Div([
            html.Div([
                dcc.Graph(id="network", config={"displaylogo": False}),
                html.P("Node color = overall misclassification rate of its true species. "
                       "Arrow thickness = number of confusions. "
                       "Hover over arrows to see class-normalized rates. "
                       "Hidden 'empty' arrows still count toward each species' overall error rate.",
                       style={"fontSize": "12px", "color": "#526174"}),
            ], style={**card_style, "flex": "2 1 590px", "minWidth": 0}),
            html.Div([
                html.Div(id="focus-summary", style={"marginBottom": "8px"}),
                html.H4("True species → wrong prediction", style={"marginBottom": "2px"}),
                dcc.Graph(id="outgoing", config={"displayModeBar": False}),
                html.H4("Other true species → selected prediction", style={"marginBottom": "2px"}),
                dcc.Graph(id="incoming", config={"displayModeBar": False}),
            ], style={**card_style, "flex": "1 1 360px", "minWidth": 0}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.Div([
            html.H3("High-confidence wrong predictions (selected true species)"),
            html.P("Rows are examples for further inspection. File paths refer to the local iWildCam dataset.",
                   style={"fontSize": "13px", "color": "#526174"}),
            dash_table.DataTable(
                id="example-table",
                columns=[{"name": v, "id": k} for k, v in [
                    ("id", "Image ID"), ("true", "True species"),
                    ("predicted", "Predicted species"), ("confidence", "Confidence"),
                    ("location", "Camera"), ("filename", "Filename"),
                ]],
                page_size=10, sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Arial", "fontSize": 12,
                            "textAlign": "left", "padding": "7px", "maxWidth": "230px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f1f4f8"},
            ),
        ], style={**card_style, "marginTop": "12px"}),
    ], style={"maxWidth": "1550px", "margin": "24px auto", "padding": "0 20px",
              "background": "#f7f9fb", "fontFamily": "Arial, sans-serif"})

    @app.callback(
        Output("focus", "value"),
        Input("network", "clickData"),
        State("focus", "value"),
        prevent_initial_call=True,
    )
    def select_node(click, currently_selected):
        if not click or not click.get("points"):
            return no_update
        tag = click["points"][0].get("customdata")
        if not isinstance(tag, (list, tuple)) or len(tag) != 2 or tag[0] != "node":
            return no_update
        chosen = int(tag[1])
        return no_update if chosen == currently_selected else chosen

    @app.callback(
        Output("network", "figure"), Output("network-stats", "children"),
        Input("split", "value"), Input("focus", "value"), Input("rank", "value"),
        Input("show-empty", "value"), Input("min-support", "value"),
        Input("min-count", "value"), Input("min-rate", "value"), Input("top-edges", "value"),
    )
    def update_network(split, focus, rank, include, support, count, rate, limit):
        subset, stats, edges = get_split(split)
        filtered = visible_edges(
            edges, names, int(support), int(count), float(rate), int(limit),
            rank, bool(include), focus,
        )
        fig = network_figure(filtered, stats, names, focus)
        total_errors = int((subset["label"] != subset["prediction"]).sum())
        summary = (
            f"{len(subset):,} images in '{split}' | {total_errors:,} incorrect predictions | "
            f"{len(filtered):,} visible directed confusion pairs | "
            f"{len(set(filtered['source']) | set(filtered['target']))} visible species"
        )
        return fig, summary

    @app.callback(
        Output("focus-summary", "children"), Output("outgoing", "figure"),
        Output("incoming", "figure"), Output("example-table", "data"),
        Input("split", "value"), Input("focus", "value"), Input("show-empty", "value"),
    )
    def update_details(split, focus, include):
        subset, stats, edges = get_split(split)
        if focus is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(height=270, margin={"l": 15, "r": 10, "t": 10, "b": 10},
                                    xaxis={"visible": False}, yaxis={"visible": False})
            return "Click a node or choose a species to inspect its errors.", empty_fig, empty_fig, []

        name = names.get(focus, str(focus))
        if focus in stats.index:
            r = stats.loc[focus]
            heading = html.Div([
                html.H3(name, style={"marginBottom": "5px"}),
                html.Div(f"{int(r['support']):,} true images | {int(r['errors']):,} incorrect | "
                         f"{float(r['error_rate']):.1%} error rate"),
            ])
        else:
            heading = html.Div([html.H3(name), "No true images in this split."])

        out = bar_figure(edges, names, focus, "out", bool(include))
        incoming = bar_figure(edges, names, focus, "in", bool(include))
        cases = subset.loc[
            (subset["label"] == focus) & (subset["prediction"] != focus)
        ].sort_values("confidence", ascending=False).head(60)
        examples = [{
            "id": str(row.id), "true": names.get(int(row.label), str(row.label)),
            "predicted": names.get(int(row.prediction), str(row.prediction)),
            "confidence": round(float(row.confidence), 4),
            "location": str(row.location), "filename": str(row.filename),
        } for row in cases.itertuples(index=False)]
        return heading, out, incoming, examples

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
