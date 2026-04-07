from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    # LLM models
    evolution_model: str = "gpt-5-nano-2025-08-07"
    inference_provider: str = "google"  # "openai" or "google"
    inference_model: str = "gemini-3.1-flash-lite-preview"

    # Evolution parameters
    population_size: int = 8
    max_generations: int = 10
    top_k: int = 4
    mutation_rate: float = 0.7
    crossover_rate: float = 0.5
    plateau_patience: int = 3

    # Evaluation
    mode: str = "fast"  # "fast" or "precise" — fast multiplies fitness by 0.5
    max_eval_samples: int | None = None
    inference_max_tokens: int = 8192
    evolution_max_tokens: int = 16384

    # Dataset
    dataset_name: str = "MathArena/hmmt_feb_2025"
    dataset_config: str = "default"

    # Seed
    initial_prompt: str = "Solve this math problem. Provide only the final numerical answer."

    # Mutations
    enabled_mutations: list[str] = field(default_factory=lambda: [
        # Original 10
        "rephrase", "add_cot", "persona", "decomposition", "few_shot",
        "constraint_emphasis", "output_format", "simplify", "self_verification", "meta_cognitive",
        # Reasoning & Logic
        "backward_reasoning", "analogical_thinking", "contradiction_check", "elimination",
        "first_principles", "hypothesis_testing", "edge_case_analysis", "multi_perspective",
        "socratic_questioning", "causal_reasoning",
        # Confidence & Accuracy
        "confidence_score", "double_check_arithmetic", "sanity_check", "error_anticipation",
        "precision_emphasis", "reread_instruction", "worst_case_thinking", "independent_verification",
        "assumption_listing", "unit_check",
        # Domain & Style
        "academic_tone", "direct_command", "teacher_framing", "competition_framing",
        "expert_panel", "minimalist", "verbose_explicit", "chain_of_verification",
        "plan_then_solve", "self_debate",
    ])
