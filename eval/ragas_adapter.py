"""Bridge any LangChain chat model into RAGAS 0.4's modern metrics.

RAGAS 0.4 metrics (`ragas.metrics.collections`) call `llm.agenerate(prompt, ResponseModel)`.
Its built-in `llm_factory` is tied to specific SDK clients, so this adapter lets the same
provider-swappable LangChain model used by the app act as the judge.
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from ragas.llms.base import InstructorBaseRagasLLM

from clausewatch.llm import acall_waiting_for_quota, structured_runnable


class LangChainRagasLLM(InstructorBaseRagasLLM):
    def __init__(self, llm: BaseChatModel, max_concurrency: int = 2):
        self.llm = llm
        self._max_concurrency = max_concurrency
        self._sem: asyncio.Semaphore | None = None

    def _runnable(self, response_model):
        return structured_runnable(self.llm, response_model)

    def generate(self, prompt, response_model):
        return self._runnable(response_model).invoke([HumanMessage(prompt)])

    async def agenerate(self, prompt, response_model):
        if self._sem is None:  # bind to the running event loop (each question uses asyncio.run)
            self._sem = asyncio.Semaphore(self._max_concurrency)
        async with self._sem:
            runnable = self._runnable(response_model)
            return await acall_waiting_for_quota(
                lambda: runnable.ainvoke([HumanMessage(prompt)]), label=f"judge:{response_model.__name__}")
