"""Entry point for evolutionary prompt optimization."""

from dotenv import load_dotenv
from rich.console import Console

from cli.evolution_map import print_evolution_map, save_evolution_map
from cli.visualize import visualize
from core import EvolutionConfig
from core.graph import build_graph
from data.output import get_output_dir, save_graph
from steps.common import init_steps

load_dotenv()
console = Console()


def main():
    config = EvolutionConfig(
        population_size=8,
        max_generations=10,
        top_k=3,
        max_eval_samples=30,
        initial_prompt="Answer this question. Provide only the exact final answer, nothing else.",
    )

    console.rule("[bold blue]EvoPrompts")
    for field_name, field_val in vars(config).items():
        console.print(f"  {field_name}: {field_val}")
    console.print()

    init_steps(config)

    graph = build_graph()
    final_state = graph.invoke(
        {
            "population": [],
            "generation": 0,
            "history": [],
            "best_ever": {
                "id": "",
                "prompt": "",
                "fitness": 0.0,
                "parent_a": None,
                "parent_b": None,
                "mutation": None,
                "generation": 0,
            },
            "plateau_counter": 0,
            "done": False,
        },
        {"recursion_limit": 4 * config.max_generations + 10},
    )

    print_evolution_map(final_state)
    save_evolution_map(final_state)
    save_graph()
    visualize(get_output_dir())


if __name__ == "__main__":
    main()
