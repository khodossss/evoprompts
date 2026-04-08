"""Stop-condition check."""

from __future__ import annotations

import steps.common as common


@common.timed_step
def check_stop(state: dict) -> dict:
    common.console.rule(f"[bold cyan]Step: Check Stop (gen {state['generation']})")
    done = (
        state["generation"] >= common.config.max_generations - 1
        or state["plateau_counter"] >= common.config.plateau_patience
        or state["best_ever"]["fitness"] >= 1.0
    )
    if done:
        if state["generation"] >= common.config.max_generations - 1:
            reason = "max_gen"
        elif state["plateau_counter"] >= common.config.plateau_patience:
            reason = "plateau"
        else:
            reason = "perfect"
        common.console.print(f"[bold red]Stopping: {reason}[/bold red]")
    else:
        common.console.print("  Continuing evolution...")
    return {"done": done}
