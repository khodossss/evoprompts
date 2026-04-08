"""Shared step state and helpers. Initialised once via init_steps()."""

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

_total_start: float = 0.0


def init_steps(cfg: EvolutionConfig):
    global llm, config, dataset, _total_start
    config = cfg
    llm = LLMClient(cfg)
    dataset = load_eval_dataset(cfg)
    init_output()
    _total_start = 0.0
    console.print(f"[bold green]Loaded {len(dataset)} eval samples[/bold green]")


def timed_step(fn):
    """Print elapsed time after each graph step."""
    @wraps(fn)
    def wrapper(state):
        global _total_start
        if not _total_start:
            _total_start = time.time()
        start = time.time()
        result = fn(state)
        elapsed = time.time() - start
        total = time.time() - _total_start
        console.print(f"  [dim]{fn.__name__}: {elapsed:.1f}s (total: {total:.1f}s)[/dim]")
        return result
    return wrapper


_LATEX_FRAC_RE = re.compile(r"\\d?frac\s*\{(-?\d+)\}\s*\{(-?\d+)\}")
_PLAIN_FRAC_RE = re.compile(r"-?\d+/-?\d+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Pick the most likely final answer from the model output.

    Looks at the last non-empty line and tries, in order:
    LaTeX ``\\frac{a}{b}`` -> ``a/b``; otherwise if the line contains
    any LaTeX command (backslash) it is returned verbatim (sans ``$``);
    otherwise plain ``a/b``, last number, or the line itself.
    """
    text = text.strip()
    if not text:
        return text

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    candidate = lines[-1] if lines else text

    latex = _LATEX_FRAC_RE.findall(candidate)
    if latex:
        a, b = latex[-1]
        return f"{a}/{b}"

    if "\\" in candidate:
        return candidate.replace("$", "").strip()

    fractions = _PLAIN_FRAC_RE.findall(candidate)
    if fractions:
        return fractions[-1]

    numbers = _NUMBER_RE.findall(candidate)
    if numbers:
        return numbers[-1]

    return candidate


def _parse_latex_frac(s: str) -> tuple[int, int] | None:
    m = _LATEX_FRAC_RE.match(s.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _to_float(s: str) -> float | None:
    s = s.strip().replace(" ", "")

    frac = _parse_latex_frac(s)
    if frac:
        return frac[0] / frac[1] if frac[1] != 0 else None

    m = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if m:
        denom = int(m.group(2))
        return int(m.group(1)) / denom if denom != 0 else None

    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str, expected: str) -> bool:
    """Exact, normalized, or numeric (1% relative tolerance) match."""
    p = predicted.strip()
    e = expected.strip()

    if p == e:
        return True

    def _norm(s: str) -> str:
        return s.lower().replace("$", "").replace(" ", "").replace("\\,", "")

    if _norm(p) == _norm(e):
        return True

    pf = _to_float(p)
    ef = _to_float(e)
    if pf is not None and ef is not None:
        if ef == 0:
            return abs(pf) < 1e-9
        return abs(pf - ef) / max(abs(ef), 1e-12) < 0.01

    return False
