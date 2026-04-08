"""LangGraph state definition."""

from __future__ import annotations

import uuid
from typing import TypedDict


def new_id() -> str:
    return uuid.uuid4().hex[:8]


class Individual(TypedDict):
    id: str
    prompt: str
    fitness: float
    parent_a: str | None
    parent_b: str | None
    mutation: str | None
    generation: int


class GenerationRecord(TypedDict):
    generation: int
    best_fitness: float
    avg_fitness: float
    best_prompt: str
    population: list[Individual]


class EvolutionState(TypedDict):
    population: list[Individual]
    generation: int
    history: list[GenerationRecord]
    best_ever: Individual
    plateau_counter: int
    done: bool
