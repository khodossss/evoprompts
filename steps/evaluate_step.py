"""Evaluate population fitness."""

from __future__ import annotations

from core.state import GenerationRecord
from data.output import get_graph, save_step
import steps.common as common


@common.timed_step
def evaluate_population(state: dict) -> dict:
    gen = state["generation"]
    common.console.rule(f"[bold yellow]Step: Evaluate (gen {gen})")
    population = state["population"]
    dataset = common.dataset

    # Only evaluate freshly produced individuals; elites carry their fitness over.
    to_eval: list[tuple[int, int]] = []
    for p_idx, ind in enumerate(population):
        if ind["mutation"] == "elite":
            continue
        for s_idx in range(len(dataset)):
            to_eval.append((p_idx, s_idx))

    eval_details: list[dict] = []
    errors: list[dict] = []

    if to_eval:
        calls = [
            (population[p_idx]["prompt"], dataset[s_idx]["question"])
            for p_idx, s_idx in to_eval
        ]
        common.console.print(f"  Firing {len(calls)} inference calls in parallel...")
        results = common.llm.inference_batch_multi(calls)

        scores: dict[int, int] = {}
        counts: dict[int, int] = {}
        for (p_idx, s_idx), result in zip(to_eval, results):
            scores.setdefault(p_idx, 0)
            counts.setdefault(p_idx, 0)
            counts[p_idx] += 1

            if isinstance(result, Exception):
                common.console.print(f"  [red]Error p={p_idx} s={s_idx}: {result}[/red]")
                predicted = f"ERROR: {result}"
                correct = False
                errors.append({
                    "step": "evaluate",
                    "prompt_idx": p_idx,
                    "sample_idx": s_idx,
                    "error": str(result),
                    "prompt": population[p_idx]["prompt"][:200],
                })
            else:
                predicted = common.extract_answer(result)
                correct = common.answers_match(predicted, dataset[s_idx]["answer"])
                if correct:
                    scores[p_idx] += 1

            eval_details.append({
                "prompt_idx": p_idx,
                "prompt": population[p_idx]["prompt"],
                "question": dataset[s_idx]["question"],
                "expected": dataset[s_idx]["answer"],
                "predicted": predicted,
                "correct": correct,
                "raw_output": result if not isinstance(result, Exception) else str(result),
            })

        graph_nodes = {n["id"]: n for n in get_graph()["nodes"]}
        for p_idx, score in scores.items():
            fitness = score / counts[p_idx]
            population[p_idx]["fitness"] = fitness
            node = graph_nodes.get(population[p_idx]["id"])
            if node:
                node["fitness"] = fitness
            common.console.print(
                f"  [{p_idx + 1}/{len(population)}] fitness={fitness:.2%}  {population[p_idx]['prompt'][:60]}..."
            )

    best = max(population, key=lambda x: x["fitness"])
    avg = sum(x["fitness"] for x in population) / len(population)

    record = GenerationRecord(
        generation=gen,
        best_fitness=best["fitness"],
        avg_fitness=avg,
        best_prompt=best["prompt"],
        population=list(population),
    )
    history = list(state["history"]) + [record]

    best_ever = state.get("best_ever") or best
    if best["fitness"] >= best_ever["fitness"]:
        best_ever = best

    plateau_counter = state["plateau_counter"]
    if len(history) >= 2 and history[-1]["best_fitness"] <= history[-2]["best_fitness"]:
        plateau_counter += 1
    else:
        plateau_counter = 0

    common.console.print(
        f"  [bold]Gen {gen}: best={best['fitness']:.2%}, avg={avg:.2%}, plateau={plateau_counter}[/bold]"
    )

    save_step("evaluate", gen, population)
    save_step("evaluate_details", gen, eval_details)
    if errors:
        save_step("eval_errors", gen, errors)

    return {
        "population": population,
        "history": history,
        "best_ever": best_ever,
        "plateau_counter": plateau_counter,
    }
