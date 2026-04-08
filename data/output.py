"""Output manager. Saves each step to ./output/<timestamp>/genN_step.json
and accumulates an evolution graph as nodes/edges are produced."""

from __future__ import annotations

import json
import os
from datetime import datetime

from rich.console import Console

console = Console()

_output_dir: str = ""
_graph: dict = {"nodes": [], "edges": []}


def get_output_dir() -> str:
    return _output_dir


def get_graph() -> dict:
    return _graph


def init_output() -> str:
    global _output_dir, _graph
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _output_dir = os.path.join("output", ts)
    os.makedirs(_output_dir, exist_ok=True)
    _graph = {"nodes": [], "edges": []}
    console.print(f"[bold green]Output dir: {_output_dir}[/bold green]")
    return _output_dir


def save_step(step_name: str, generation: int, population: list[dict]):
    filename = f"gen{generation}_{step_name}.json"
    path = os.path.join(_output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(population, f, indent=2, ensure_ascii=False)
    console.print(f"  [dim]Saved -> {path}[/dim]")


def add_node(node_id: str, generation: int, prompt: str, fitness: float, mutation: str | None):
    _graph["nodes"].append({
        "id": node_id,
        "generation": generation,
        "prompt": prompt,
        "fitness": fitness,
        "mutation": mutation or "seed",
    })


def add_edge(parent_id: str, child_id: str, label: str):
    if parent_id and child_id and parent_id != child_id:
        _graph["edges"].append({
            "source": parent_id,
            "target": child_id,
            "label": label,
        })


def save_graph():
    path = os.path.join(_output_dir, "graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_graph, f, indent=2, ensure_ascii=False)
    console.print(f"[bold]Graph saved -> {path}[/bold]")
