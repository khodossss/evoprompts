"""LLM clients."""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core.config import EvolutionConfig

MAX_CONCURRENT = 10
MAX_RETRIES = 3


def _build_inference_llm(config: EvolutionConfig):
    if config.inference_provider == "google":
        return ChatGoogleGenerativeAI(
            model=config.inference_model,
            temperature=0,
            max_output_tokens=config.inference_max_tokens,
        )
    return ChatOpenAI(
        model=config.inference_model,
        reasoning_effort="minimal",
        max_completion_tokens=config.inference_max_tokens,
    )


class LLMClient:
    def __init__(self, config: EvolutionConfig):
        self.evolution_llm = ChatOpenAI(
            model=config.evolution_model,
            reasoning_effort="medium",
            max_completion_tokens=config.evolution_max_tokens,
        )
        self.inference_llm = _build_inference_llm(config)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    @staticmethod
    def _extract_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content) if content else ""

    def evolution_call(self, system: str, user: str) -> str:
        response = self.evolution_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return self._extract_content(response.content)

    async def _acall_with_retry(self, llm, system: str, user: str) -> str:
        for _ in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    response = await llm.ainvoke([
                        SystemMessage(content=system),
                        HumanMessage(content=user),
                    ])
                    content = self._extract_content(response.content)
                    if content and content.strip():
                        return content
            except Exception as e:
                err = str(e).lower()
                if "invalid_prompt" in err or "usage policy" in err or "400" in err:
                    return ""
            await asyncio.sleep(0.5)
        return ""

    def _run_batch(self, tasks):
        async def _run():
            self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            return await asyncio.gather(*tasks, return_exceptions=True)
        return asyncio.run(_run())

    def evolution_batch(self, calls: list[tuple[str, str]]) -> list[str]:
        tasks = [self._acall_with_retry(self.evolution_llm, s, u) for s, u in calls]
        return self._run_batch(tasks)

    def inference_batch_multi(self, calls: list[tuple[str, str]]) -> list[str]:
        tasks = [self._acall_with_retry(self.inference_llm, s, u) for s, u in calls]
        return self._run_batch(tasks)
