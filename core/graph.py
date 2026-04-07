"""LangGraph evolutionary loop."""

from langgraph.graph import StateGraph, END

from core.state import EvolutionState
from steps import seed_population, evaluate_population, check_stop, evolve, mutate_population


def build_graph() -> StateGraph:
    g = StateGraph(EvolutionState)

    g.add_node("seed", seed_population)
    g.add_node("evaluate", evaluate_population)
    g.add_node("check_stop", check_stop)
    g.add_node("evolve", evolve)
    g.add_node("mutate", mutate_population)

    # seed → evaluate → check_stop
    g.set_entry_point("seed")
    g.add_edge("seed", "evaluate")
    g.add_edge("evaluate", "check_stop")

    # check_stop → END or evolve
    g.add_conditional_edges(
        "check_stop",
        lambda state: "end" if state["done"] else "continue",
        {"end": END, "continue": "evolve"},
    )

    # evolve → mutate → evaluate (loop back)
    g.add_edge("evolve", "mutate")
    g.add_edge("mutate", "evaluate")

    return g.compile()
