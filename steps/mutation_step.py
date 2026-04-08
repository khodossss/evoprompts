"""40 mutation operators applied by the evolution LLM."""

from __future__ import annotations

import random

from core.state import Individual, new_id
from data.output import get_graph, save_step
import steps.common as common

_PREAMBLE = (
    "You are an AI prompt optimization tool. Your job is to EDIT and IMPROVE "
    "user-written instruction prompts. You are NOT being asked to reveal, extract, "
    "or leak any system prompt. The user is giving you a prompt THEY wrote and asking "
    "you to rewrite it with a specific improvement. This is a legitimate prompt engineering task. "
    "Always output the improved prompt.\n\n"
)

_REASONING_GUARD = (
    " CRITICAL: the improved prompt MUST instruct the solver to do this reasoning "
    "INTERNALLY and output ONLY the final answer — no intermediate steps, no explanations, "
    "no working shown. The output must be just the exact answer."
)


MUTATIONS: dict[str, dict[str, str]] = {
    "rephrase": {
        "system": "You are a prompt engineer. Rephrase the given prompt to express the same instruction differently. Change sentence structure, vocabulary, and ordering. Do NOT change the task semantics.",
        "user": "Rephrase this prompt:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "simplify": {
        "system": "Simplify and compress this prompt. Remove redundancy and verbose phrasing. Make every word count. The result should be shorter but convey all the same instructions.",
        "user": "Simplify this prompt:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "minimalist": {
        "system": "Strip this prompt down to the absolute minimum — the fewest possible words that still convey the complete instruction. Aim for under 20 words.",
        "user": "Make minimalist:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "constraint_emphasis": {
        "system": "Strengthen emphasis on constraints and requirements in this prompt. Infer what the key constraints are from the prompt itself and add reminders to respect them. Do not remove existing instructions.",
        "user": "Emphasize constraints:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "output_format": {
        "system": "Modify the output format instructions in this prompt. Make the expected output format extremely clear and specific. Vary the exact phrasing each time.",
        "user": "Improve output format instructions:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "precision_emphasis": {
        "system": "Add precision emphasis. Remind the solver that exact precision matters — no rounding, no approximations, exact values only.",
        "user": "Add precision emphasis:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "direct_command": {
        "system": "Rewrite this prompt as a series of direct, imperative commands. No politeness, no hedging. E.g. 'Solve. Output the answer.'",
        "user": "Make direct commands:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "persona": {
        "system": "Add a persona/role framing to this prompt (e.g. 'You are an expert in this domain'). Choose a persona maximally relevant to the task. If a persona already exists, replace it with a better one.",
        "user": "Add a persona to this prompt:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },

    "add_cot": {
        "system": "Add chain-of-thought reasoning instructions to this prompt. Tell the solver to think step by step INTERNALLY before answering." + _REASONING_GUARD,
        "user": "Add internal CoT:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "decomposition": {
        "system": "Add decomposition instructions. Tell the solver to mentally break the problem into sub-problems and solve each one before combining." + _REASONING_GUARD,
        "user": "Add internal decomposition:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "few_shot": {
        "system": "Add ONE very short worked example (question → answer) to this prompt that demonstrates the expected answer format. The example must be brief (1-2 lines) and domain-relevant. Keep the original instructions.",
        "user": "Add a brief worked example:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "backward_reasoning": {
        "system": "Add backward reasoning instructions. Tell the solver to internally start from the desired answer type and work backwards to determine the solution." + _REASONING_GUARD,
        "user": "Add internal backward reasoning:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "analogical_thinking": {
        "system": "Add instructions to think by analogy. Tell the solver to internally recall similar problems or patterns and apply analogous reasoning." + _REASONING_GUARD,
        "user": "Add internal analogical thinking:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "contradiction_check": {
        "system": "Add contradiction checking instructions. Tell the solver to internally verify there are no contradictions in their reasoning before answering." + _REASONING_GUARD,
        "user": "Add internal contradiction check:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "elimination": {
        "system": "Add process-of-elimination instructions. Tell the solver to internally rule out impossible or unlikely answers before converging on the correct one." + _REASONING_GUARD,
        "user": "Add internal elimination:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "first_principles": {
        "system": "Add first-principles thinking instructions. Tell the solver to internally break the problem down to fundamental truths and build up reasoning from there." + _REASONING_GUARD,
        "user": "Add internal first-principles:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "hypothesis_testing": {
        "system": "Add hypothesis-testing instructions. Tell the solver to internally form a hypothesis, test it against given information, and refine it." + _REASONING_GUARD,
        "user": "Add internal hypothesis testing:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "edge_case_analysis": {
        "system": "Add edge-case analysis instructions. Tell the solver to internally consider boundary conditions and special cases that might affect the answer." + _REASONING_GUARD,
        "user": "Add internal edge case analysis:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "multi_perspective": {
        "system": "Add multi-perspective instructions. Tell the solver to internally approach from 2-3 different angles and only answer if they converge." + _REASONING_GUARD,
        "user": "Add internal multi-perspective:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "socratic_questioning": {
        "system": "Add Socratic self-questioning instructions. Tell the solver to internally ask clarifying questions about the problem before solving." + _REASONING_GUARD,
        "user": "Add internal Socratic questioning:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "causal_reasoning": {
        "system": "Add causal reasoning instructions. Tell the solver to internally trace cause-and-effect relationships to arrive at the answer." + _REASONING_GUARD,
        "user": "Add internal causal reasoning:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "meta_cognitive": {
        "system": "Add meta-cognitive instructions. Tell the solver to internally consider which approach or strategy is best suited for the problem before diving in." + _REASONING_GUARD,
        "user": "Add internal meta-cognitive:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "plan_then_solve": {
        "system": "Add plan-then-solve instructions. Tell the solver to internally form a brief plan of attack, then execute it mentally." + _REASONING_GUARD,
        "user": "Add internal plan-then-solve:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "self_debate": {
        "system": "Add self-debate instructions. Tell the solver to internally argue for and against their answer, then settle on the one that survives scrutiny." + _REASONING_GUARD,
        "user": "Add internal self-debate:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "chain_of_verification": {
        "system": "Add chain-of-verification instructions. Tell the solver to internally (1) draft an answer, (2) verify it, (3) revise if needed." + _REASONING_GUARD,
        "user": "Add internal chain of verification:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },

    "self_verification": {
        "system": "Add self-verification instructions. Tell the solver to internally check their work and verify the answer satisfies all conditions before outputting." + _REASONING_GUARD,
        "user": "Add internal self-verification:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "confidence_score": {
        "system": "Add confidence instructions. Tell the solver to internally assess confidence and if low, try an alternative approach before answering." + _REASONING_GUARD,
        "user": "Add internal confidence scoring:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "double_check_arithmetic": {
        "system": "Add arithmetic double-checking instructions. Tell the solver to internally recompute any calculations and compare results before answering." + _REASONING_GUARD,
        "user": "Add internal arithmetic double-check:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "sanity_check": {
        "system": "Add sanity check instructions. Tell the solver to internally verify the answer is reasonable and within expected bounds." + _REASONING_GUARD,
        "user": "Add internal sanity check:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "error_anticipation": {
        "system": "Add error anticipation instructions. Tell the solver to internally identify common mistakes for this type of problem and avoid them." + _REASONING_GUARD,
        "user": "Add internal error anticipation:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "reread_instruction": {
        "system": "Add a re-read instruction. Tell the solver to internally re-read the question after formulating an answer to make sure they answered what was actually asked." + _REASONING_GUARD,
        "user": "Add re-read instruction:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "worst_case_thinking": {
        "system": "Add worst-case thinking instructions. Tell the solver to internally consider what could go wrong with their approach and address it." + _REASONING_GUARD,
        "user": "Add internal worst-case thinking:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "independent_verification": {
        "system": "Add independent verification instructions. Tell the solver to internally solve using two different methods and only answer if both agree." + _REASONING_GUARD,
        "user": "Add internal independent verification:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "assumption_listing": {
        "system": "Add assumption-checking instructions. Tell the solver to internally identify and verify all assumptions before answering." + _REASONING_GUARD,
        "user": "Add internal assumption checking:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "unit_check": {
        "system": "Add type/unit checking instructions. Tell the solver to internally verify that the type and format of their answer match what the question asks for." + _REASONING_GUARD,
        "user": "Add internal unit/type check:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },

    "academic_tone": {
        "system": "Rewrite this prompt in a formal academic tone. Use precise, technical language that signals rigor and exactitude.",
        "user": "Make academic:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "teacher_framing": {
        "system": "Add a teacher/student framing. Frame the prompt as if a strict teacher is giving an exam question and expects a precise, no-nonsense answer.",
        "user": "Add teacher framing:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "competition_framing": {
        "system": "Add competition framing. Frame this as a timed competition problem where accuracy and precision are paramount, and only exact answers receive credit.",
        "user": "Add competition framing:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "expert_panel": {
        "system": "Add expert panel framing. Tell the solver to imagine a panel of domain experts will review the answer, so it must be defensible and precise.",
        "user": "Add expert panel framing:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
    "verbose_explicit": {
        "system": "Make this prompt more explicit. Leave less room for ambiguity. Spell out key expectations and format requirements clearly. Keep it reasonably concise.",
        "user": "Make more explicit:\n\n{prompt}\n\nReturn ONLY the new prompt, nothing else.",
    },
}


REFUSAL_PREFIXES = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "i am unable",
    "sorry, i",
    "i won't",
    "i will not",
)


def _is_refusal(text: str) -> bool:
    head = text.lstrip().lower()[:80]
    return any(head.startswith(p) for p in REFUSAL_PREFIXES)


@common.timed_step
def mutate_population(state: dict) -> dict:
    gen = state["generation"]
    common.console.rule(f"[bold green]Step: Mutate (gen {gen})")
    population = list(state["population"])

    mutation_plan: list[tuple[int, str]] = []
    calls: list[tuple[str, str]] = []

    for i, ind in enumerate(population):
        if ind["mutation"] == "elite":
            continue
        if random.random() < common.config.mutation_rate:
            mut_name = random.choice(common.config.enabled_mutations)
            spec = MUTATIONS[mut_name]
            calls.append((_PREAMBLE + spec["system"], spec["user"].format(prompt=ind["prompt"])))
            mutation_plan.append((i, mut_name))

    errors: list[dict] = []
    if calls:
        common.console.print(f"  Firing {len(calls)} mutation calls in parallel...")
        results = common.llm.evolution_batch(calls)

        for (i, mut_name), result in zip(mutation_plan, results):
            if isinstance(result, Exception) or not result:
                common.console.print(f"  [{i + 1}] {mut_name}: ERROR, keeping original")
                errors.append({
                    "step": "mutation", "index": i, "mutation": mut_name,
                    "error": str(result), "prompt": population[i]["prompt"],
                })
                continue

            new_prompt = result.strip()
            if _is_refusal(new_prompt) or not new_prompt:
                common.console.print(f"  [{i + 1}] {mut_name}: REFUSED, keeping original")
                errors.append({
                    "step": "mutation", "index": i, "mutation": mut_name,
                    "error": "refusal", "output": new_prompt[:200],
                    "prompt": population[i]["prompt"],
                })
                continue

            old_ind = population[i]
            new_ind_id = new_id()
            population[i] = Individual(
                id=new_ind_id,
                prompt=new_prompt,
                fitness=0.0,
                parent_a=old_ind["parent_a"],
                parent_b=old_ind.get("parent_b"),
                mutation=mut_name,
                generation=gen,
            )

            graph = get_graph()
            old_id = old_ind["id"]
            for node in graph["nodes"]:
                if node["id"] == old_id:
                    node["id"] = new_ind_id
                    node["prompt"] = new_prompt
                    node["mutation"] = mut_name
                    break
            for edge in graph["edges"]:
                if edge["target"] == old_id:
                    edge["target"] = new_ind_id
                    edge["label"] = mut_name

            common.console.print(f"  [{i + 1}] {mut_name}: {new_prompt[:70]}...")

    save_step("mutation", gen, population)
    if errors:
        save_step("mutation_errors", gen, errors)

    return {"population": population}
