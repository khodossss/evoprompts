"""Load evaluation dataset from HuggingFace."""

from datasets import load_dataset
from core.config import EvolutionConfig


def load_eval_dataset(config: EvolutionConfig) -> list[dict]:
    ds = load_dataset(config.dataset_name, config.dataset_config, split="train")
    samples = [{"question": row["problem"], "answer": str(row["answer"]).strip()} for row in ds]
    if config.max_eval_samples is not None:
        samples = samples[: config.max_eval_samples]
    return samples
