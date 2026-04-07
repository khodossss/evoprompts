"""Render the evolution map — a visual summary of the evolutionary process."""

from __future__ import annotations
import json

from rich.console import Console
from rich.table import Table

from core.state import EvolutionState

console = Console()


def print_evolution_map(state: EvolutionState):
    """Print a rich summary of the entire evolution."""
    console.print()
    console.rule("[bold green]EVOLUTION MAP")

    table = Table(title="Fitness by Generation")
    table.add_column("Gen", justify="center", style="cyan")
    table.add_column("Best", justify="center", style="green")
    table.add_column("Avg", justify="center", style="yellow")
    table.add_column("Best Prompt (truncated)", style="white", max_width=60)

    for rec in state["history"]:
        table.add_row(
            str(rec["generation"]),
            f"{rec['best_fitness']:.2%}",
            f"{rec['avg_fitness']:.2%}",
            rec["best_prompt"][:60] + "...",
        )

    console.print(table)

    best = state["best_ever"]
    console.print()
    console.print("[bold green]BEST PROMPT FOUND:[/bold green]")
    console.print(f"  Fitness: {best['fitness']:.2%}")
    console.print(f"  Generation: {best['generation']}")
    console.print(f"  Mutation: {best['mutation']}")
    console.print(f"  Prompt:\n{best['prompt']}")

    console.print()
    console.rule("Final Population")
    for i, ind in enumerate(state["population"]):
        console.print(
            f"  [{i+1}] fitness={ind['fitness']:.2%} "
            f"mutation={ind['mutation']} "
            f"gen={ind['generation']}"
        )
        console.print(f"      {ind['prompt'][:80]}...")


def save_evolution_map(state: EvolutionState, path: str = "evolution_results.json"):
    """Save full evolution history to JSON."""
    data = {
        "best_ever": state["best_ever"],
        "generations": state["history"],
        "final_population": state["population"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[bold]Results saved to {path}[/bold]")
