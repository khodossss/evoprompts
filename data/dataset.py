"""Load evaluation dataset from HuggingFace."""

from datasets import load_dataset

from core.config import EvolutionConfig


def _normalize_answer(raw: str) -> str:
    """GSM8K stores the final answer after a `####` marker."""
    raw = str(raw).strip()
    if "####" in raw:
        raw = raw.split("####")[-1]
    return raw.replace(",", "").strip()


def load_eval_dataset(config: EvolutionConfig) -> list[dict]:
    ds = load_dataset(config.dataset_name, config.dataset_config, split="test")
    samples = [
        {"question": row["question"], "answer": _normalize_answer(row["answer"])}
        for row in ds
    ]
    if config.max_eval_samples is not None:
        samples = samples[: config.max_eval_samples]
    return samples
