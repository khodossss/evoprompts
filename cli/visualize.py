"""Render evolution graph from graph.json with Plotly."""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go


def _short(prompt: str, max_len: int = 60) -> str:
    text = prompt.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def build_evolution_graph(output_dir: str) -> go.Figure:
    graph_path = os.path.join(output_dir, "graph.json")
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    max_gen = max((n["generation"] for n in nodes.values()), default=0)

    gen_groups: dict[int, list[str]] = {}
    for nid, ndata in nodes.items():
        gen_groups.setdefault(ndata["generation"], []).append(nid)

    parents_of: dict[str, list[str]] = {}
    for e in edges:
        parents_of.setdefault(e["target"], []).append(e["source"])

    pos: dict[str, tuple[float, float]] = {}

    gen_groups.setdefault(0, []).sort()
    for i, nid in enumerate(gen_groups[0]):
        x = (i - (len(gen_groups[0]) - 1) / 2) * 1.5
        pos[nid] = (x, 0)

    # Order each subsequent generation by the average x of its parents
    # to reduce edge crossings.
    for g in range(1, max_gen + 1):
        if g not in gen_groups:
            continue
        nids = gen_groups[g]

        def _barycenter(nid: str, _placed: dict[str, tuple[float, float]] = pos) -> float:
            pars = parents_of.get(nid, [])
            xs = [_placed[p][0] for p in pars if p in _placed]
            return sum(xs) / len(xs) if xs else 0.0

        nids.sort(key=_barycenter)
        gen_groups[g] = nids

        for i, nid in enumerate(nids):
            x = (i - (len(nids) - 1) / 2) * 1.5
            pos[nid] = (x, -g)

    all_fitness = [n["fitness"] for n in nodes.values()]
    min_f = min(all_fitness) if all_fitness else 0
    max_f = max(all_fitness) if all_fitness else 1

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_labels: list[str] = []

    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        edge_labels.append(e["label"])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )

    edge_label_trace = go.Scatter(
        x=edge_mid_x, y=edge_mid_y,
        mode="text",
        text=edge_labels,
        textfont=dict(size=8, color="#666"),
        hoverinfo="none",
    )

    node_ids = [nid for nid in nodes if nid in pos]
    node_x = [pos[nid][0] for nid in node_ids]
    node_y = [pos[nid][1] for nid in node_ids]
    node_color = [nodes[nid]["fitness"] for nid in node_ids]
    node_text = [
        f"<b>Gen {nodes[nid]['generation']}</b> | id: {nid} | fitness: {nodes[nid]['fitness']:.2%}<br>"
        f"mutation: {nodes[nid]['mutation']}<br>"
        f"{_short(nodes[nid]['prompt'], 80)}"
        for nid in node_ids
    ]
    node_labels = [f"{nodes[nid]['fitness']:.0%}" for nid in node_ids]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=20,
            color=node_color,
            colorscale="RdYlGn",
            cmin=min_f,
            cmax=max(max_f, 0.01),
            colorbar=dict(title="Fitness", thickness=15),
            line=dict(width=1, color="#333"),
        ),
    )

    max_per_gen = max(len(v) for v in gen_groups.values()) if gen_groups else 1
    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Evolution Graph", font=dict(size=18)),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False, zeroline=False,
                tickmode="array",
                tickvals=[-g for g in range(max_gen + 1)],
                ticktext=[f"Gen {g}" for g in range(max_gen + 1)],
            ),
            plot_bgcolor="white",
            height=300 + max_gen * 150,
            width=max(800, max_per_gen * 180),
        ),
    )

    return fig


def visualize(output_dir: str):
    fig = build_evolution_graph(output_dir)
    html_path = os.path.join(output_dir, "evolution_graph.html")
    fig.write_html(html_path)
    print(f"Saved evolution graph -> {html_path}")
    fig.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize(sys.argv[1])
    else:
        print("Usage: python -m cli.visualize <output_dir>")
