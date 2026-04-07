"""Step: selection + crossover to produce next generation."""

from __future__ import annotations
import random

from core.state import Individual, new_id
from data.output import save_step, add_node, add_edge
import steps.common as common

CROSSOVER_SYSTEM = (
    "You are an AI prompt optimization tool. Your job is to combine two instruction prompts into one better prompt. "
    "Given TWO prompts for the same task, create a child prompt that "
    "combines the strongest elements of both. Return ONLY the new prompt."
)


@common.timed_step
def evolve(state: dict) -> dict:
    """Selection + crossover (batched). Mutation is a separate step."""
    gen = state["generation"] + 1
    common.console.rule(f"[bold magenta]Step: Evolve -> generation {gen}")

    ranked = sorted(state["population"], key=lambda x: x["fitness"], reverse=True)
    survivors = ranked[: common.config.top_k]
    common.console.print(f"  Selected top-{common.config.top_k}, best={survivors[0]['fitness']:.2%}")

    save_step("selection", gen, survivors)

    new_population: list[Individual] = []

    # Elites — new id each generation so graph shows them as separate nodes
    for s in survivors:
        eid = new_id()
        new_population.append(Individual(
            id=eid,
            prompt=s["prompt"],
            fitness=s["fitness"],
            parent_a=s["id"],
            parent_b=None,
            mutation="elite",
            generation=gen,
        ))
        add_node(eid, gen, s["prompt"], s["fitness"], "elite")
        add_edge(s["id"], eid, "elite")

    # Pre-decide crossover vs clone for remaining slots
    offspring_meta: list[dict] = []
    crossover_calls: list[tuple[str, str]] = []
    crossover_indices: list[int] = []

    while len(new_population) + len(offspring_meta) < common.config.population_size:
        idx = len(offspring_meta)
        if random.random() < common.config.crossover_rate and len(survivors) >= 2:
            p_a, p_b = random.sample(survivors, 2)
            crossover_calls.append((
                CROSSOVER_SYSTEM,
                f"Parent A:\n{p_a['prompt']}\n\nParent B:\n{p_b['prompt']}",
            ))
            crossover_indices.append(idx)
            offspring_meta.append({
                "parent_a_id": p_a["id"],
                "parent_b_id": p_b["id"],
                "op": "crossover",
                "prompt": None,
            })
        else:
            parent = random.choice(survivors)
            offspring_meta.append({
                "parent_a_id": parent["id"],
                "parent_b_id": None,
                "op": "clone",
                "prompt": parent["prompt"],
            })

    # Batch all crossover calls
    errors: list[dict] = []
    if crossover_calls:
        common.console.print(f"  Firing {len(crossover_calls)} crossover calls in parallel...")
        results = common.llm.evolution_batch(crossover_calls)
        for ci, result in zip(crossover_indices, results):
            if isinstance(result, Exception) or not result:
                offspring_meta[ci]["prompt"] = state["population"][0]["prompt"]  # fallback
                offspring_meta[ci]["op"] = "clone"
                offspring_meta[ci]["parent_b_id"] = None
                errors.append({"step": "crossover", "index": ci, "error": str(result), "parent_a": offspring_meta[ci].get("parent_a_id"), "parent_b": offspring_meta[ci].get("parent_b_id")})
            else:
                offspring_meta[ci]["prompt"] = result.strip()

    for meta in offspring_meta:
        cid = new_id()
        new_population.append(Individual(
            id=cid,
            prompt=meta["prompt"],
            fitness=0.0,
            parent_a=meta["parent_a_id"],
            parent_b=meta["parent_b_id"],
            mutation=meta["op"],
            generation=gen,
        ))
        add_node(cid, gen, meta["prompt"], 0.0, meta["op"])
        add_edge(meta["parent_a_id"], cid, meta["op"])
        if meta["parent_b_id"]:
            add_edge(meta["parent_b_id"], cid, "crossover")
        common.console.print(f"  New [{len(new_population)}/{common.config.population_size}] via {meta['op']}: {meta['prompt'][:60]}...")

    save_step("crossover", gen, new_population)
    if errors:
        save_step("crossover_errors", gen, errors)
    return {"population": new_population, "generation": gen}
