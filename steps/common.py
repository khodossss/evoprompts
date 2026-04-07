"""Shared state across all steps — initialized once via init_steps()."""

from __future__ import annotations
import re
import time
from functools import wraps

from rich.console import Console

from core.config import EvolutionConfig
from core.llm import LLMClient
from data.dataset import load_eval_dataset
from data.output import init_output

console = Console()

llm: LLMClient = None  # type: ignore[assignment]
config: EvolutionConfig = None  # type: ignore[assignment]
dataset: list[dict] = []


def init_steps(cfg: EvolutionConfig):
    global llm, config, dataset
    config = cfg
    llm = LLMClient(cfg)
    dataset = load_eval_dataset(cfg)
    init_output()
    console.print(f"[bold green]Loaded {len(dataset)} eval samples[/bold green]")


_total_start: float = 0.0


def timed_step(fn):
    """Decorator that prints elapsed time for each graph step."""
    @wraps(fn)
    def wrapper(state):
        global _total_start
        if not _total_start:
            _total_start = time.time()
        start = time.time()
        result = fn(state)
        elapsed = time.time() - start
        total = time.time() - _total_start
        console.print(f"  [dim]⏱ {fn.__name__}: {elapsed:.1f}s (total: {total:.1f}s)[/dim]")
        return result
    return wrapper


def extract_answer(text: str) -> str:
    """Extract the final answer from LLM output — last fraction or last number."""
    text = text.strip()

    # Look for fractions (a/b) — take the last one
    fractions = re.findall(r'-?\d+/\d+', text)
    if fractions:
        return fractions[-1]

    # Otherwise last number (int or decimal)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    # Fallback: last non-empty line, stripped
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text


def _parse_latex_frac(s: str) -> tuple[int, int] | None:
    """Parse \\frac{a}{b} or \\dfrac{a}{b} and return (a, b)."""
    m = re.match(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _to_float(s: str) -> float | None:
    """Try to convert a string to float, handling fractions and latex."""
    s = s.strip().replace(" ", "")
    # LaTeX frac
    frac = _parse_latex_frac(s)
    if frac:
        return frac[0] / frac[1] if frac[1] != 0 else None
    # Plain fraction: a/b
    m = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if m:
        denom = int(m.group(2))
        return int(m.group(1)) / denom if denom != 0 else None
    # Plain number
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str, expected: str) -> bool:
    """Compare answers with normalization: exact string, numeric, fraction."""
    p = predicted.strip()
    e = expected.strip()

    # 1. Exact string match
    if p == e:
        return True

    # 2. Normalize both and compare strings (lowercase, strip $, spaces)
    def _norm(s):
        return s.lower().replace("$", "").replace(" ", "").replace("\\,", "")
    if _norm(p) == _norm(e):
        return True

    # 3. Numeric comparison with tolerance
    pf = _to_float(p)
    ef = _to_float(e)
    if pf is not None and ef is not None:
        if ef == 0:
            return abs(pf) < 1e-9
        return abs(pf - ef) / max(abs(ef), 1e-12) < 0.01  # 1% relative tolerance

    return False


def evaluate_prompt(prompt: str) -> float:
    """Run a single prompt on all dataset samples concurrently, return accuracy (0-1)."""
    if not dataset:
        return 0.0

    questions = [s["question"] for s in dataset]
    results = llm.inference_batch(prompt, questions)

    correct = 0
    for sample, result in zip(dataset, results):
        if isinstance(result, Exception):
            console.print(f"[red]Inference error: {result}[/red]")
            continue
        predicted = extract_answer(result)
        if answers_match(predicted, sample["answer"]):
            correct += 1

    return correct / len(dataset)
