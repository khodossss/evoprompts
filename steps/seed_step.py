"""Seed initial population."""

from __future__ import annotations

from core.state import Individual, new_id
from data.output import add_node, save_step
import steps.common as common

SEED_SYSTEM = (
    "You are a serious, professional prompt engineer optimizing LLM system prompts for maximum accuracy. "
    "You will be given an initial prompt. Generate a STRICTLY task-focused variant that aims to maximize "
    "correct answers. You may use techniques like expert personas, decomposition, strict output formatting, "
    "or self-verification, but the variant MUST instruct the solver to do any reasoning INTERNALLY and "
    "output ONLY the final answer with no working shown. "
    "Do NOT add humor, jokes, creative writing, or anything unrelated to solving the task correctly. "
    "The prompt must be concise, clear, and directly actionable. "
    "Return ONLY the new prompt text. Do NOT prefix it with labels like 'Variant', 'Prompt:', "
    "numbering, or any meta-commentary."
)


@common.timed_step
def seed_population(state: dict) -> dict:
    common.console.rule("[bold blue]Step: Seed Population")

    population: list[Individual] = [Individual(
        id=new_id(),
        prompt=common.config.initial_prompt,
        fitness=0.0,
        parent_a=None,
        parent_b=None,
        mutation=None,
        generation=0,
    )]

    user_msg = (
        "Generate a new system prompt for the same task. Use a different optimization strategy "
        "than the original but stay strictly task-focused.\n\n"
        f"Original prompt:\n{common.config.initial_prompt}"
    )
    calls = [(SEED_SYSTEM, user_msg) for _ in range(1, common.config.population_size)]

    results = common.llm.evolution_batch(calls)
    errors: list[dict] = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            common.console.print(f"  [red]Seed [{i + 2}] error: {result}[/red]")
            errors.append({
                "step": "seed", "index": i + 2, "error": str(result),
                "system_prompt": SEED_SYSTEM, "user_prompt": calls[i][1],
            })
            continue

        prompt = result.strip()
        if not prompt:
            common.console.print(f"  [yellow]Seed [{i + 2}] empty, retrying...[/yellow]")
            prompt = common.llm.evolution_call(system=SEED_SYSTEM, user=calls[i][1]).strip()
        if not prompt:
            common.console.print(f"  [red]Seed [{i + 2}] still empty, skipping[/red]")
            errors.append({
                "step": "seed", "index": i + 2, "error": "empty after retry",
                "system_prompt": SEED_SYSTEM, "user_prompt": calls[i][1],
            })
            continue

        population.append(Individual(
            id=new_id(),
            prompt=prompt,
            fitness=0.0,
            parent_a=None,
            parent_b=None,
            mutation="seed",
            generation=0,
        ))
        common.console.print(f"  Seed [{i + 2}/{common.config.population_size}]: {prompt[:80]}...")

    for ind in population:
        add_node(ind["id"], 0, ind["prompt"], 0.0, ind["mutation"])

    save_step("seed", 0, [{"id": ind["id"], "prompt": ind["prompt"]} for ind in population])
    if errors:
        save_step("seed_errors", 0, errors)

    return {
        "population": population,
        "generation": 0,
        "history": [],
        "plateau_counter": 0,
        "done": False,
    }
