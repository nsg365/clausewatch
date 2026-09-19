"""LLM factory. The rest of the codebase only sees LangChain `BaseChatModel`s, so the
provider can be swapped with CLAUSEWATCH_LLM_PROVIDER / CLAUSEWATCH_LLM_MODEL.

Default: Groq free tier (`openai/gpt-oss-120b` for the agent, a *different* model
family as the evaluation judge so the system is not grading itself)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import lru_cache
from typing import Awaitable, Callable, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

from clausewatch.config import get_settings

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
log = logging.getLogger("clausewatch.llm")

_RETRY_IN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")


def daily_limit_delay(exc: BaseException) -> float | None:
    """Seconds to wait if `exc` (or its cause) is a provider tokens/requests-per-day 429, else None."""
    err: BaseException | None = exc
    while err is not None:
        msg = str(err)
        if "429" in msg and "per day" in msg:
            m = _RETRY_IN.search(msg)
            if not m:
                return 60.0
            h, mi, se = (float(x) if x else 0.0 for x in m.groups())
            return min(max(h * 3600 + mi * 60 + se + 15, 30.0), 1800.0)
        err = err.__cause__ or err.__context__
    return None


def call_waiting_for_quota(fn: Callable[[], R], label: str = "llm") -> R:
    """Batch jobs only (eval): block through free-tier daily limits instead of failing."""
    while True:
        try:
            return fn()
        except Exception as e:
            delay = daily_limit_delay(e)
            if delay is None:
                raise
            print(f"[{label}] daily token limit reached, waiting {delay / 60:.1f} min", flush=True)
            time.sleep(delay)


async def acall_waiting_for_quota(fn: Callable[[], Awaitable[R]], label: str = "llm") -> R:
    while True:
        try:
            return await fn()
        except Exception as e:
            delay = daily_limit_delay(e)
            if delay is None:
                raise
            print(f"[{label}] daily token limit reached, waiting {delay / 60:.1f} min", flush=True)
            await asyncio.sleep(delay)


def _reasoning_kwargs(model: str, effort: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": effort}
    if model.startswith("qwen/qwen3"):
        return {"reasoning_effort": "none"}  # grading prompts are simple; save the token budget
    return {}


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter | None:
    rpm = get_settings().requests_per_minute
    if not rpm:
        return None
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=3)


@lru_cache
def get_chat_model(effort: str | None = None, model: str | None = None, max_tokens: int | None = None,
                   rate_limit: bool = True) -> BaseChatModel:
    s = get_settings()
    model = model or s.llm_model
    effort = effort or s.reasoning_effort
    provider = s.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=0,
            max_tokens=max_tokens or s.max_tokens,
            max_retries=8,  # the SDK honours retry-after on 429 (tokens-per-minute limit)
            rate_limiter=_rate_limiter() if rate_limit else None,
            **_reasoning_kwargs(model, effort),
        )
    if provider == "anthropic":  # optional; requires ANTHROPIC_API_KEY
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=s.max_tokens, reasoning_effort=effort,
                             thinking={"type": "adaptive"}, max_retries=4, rate_limiter=_rate_limiter())
    if provider == "openai":  # optional; requires langchain-openai + OPENAI_API_KEY
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, reasoning_effort=effort, rate_limiter=_rate_limiter())
    if provider == "ollama":  # optional, fully local; requires langchain-ollama + `ollama serve`
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0, num_ctx=16384)
    raise ValueError(f"Unknown llm_provider: {provider}")


def get_judge_model() -> BaseChatModel:
    s = get_settings()
    # Groq's free tier also caps output tokens per minute for this model (1k), so keep judge outputs short.
    # No client-side limiter: RAGAS drives this from an event loop, and a blocking limiter there can
    # stall the loop; the SDK's own 429 handling plus `acall_waiting_for_quota` cover the limits.
    return get_chat_model(model=s.judge_model, max_tokens=s.judge_max_tokens, rate_limit=False)


def _structured_method(model: str) -> str:
    provider = get_settings().llm_provider
    if provider in ("anthropic", "ollama"):
        return "json_schema"
    if provider == "groq":
        # Groq supports JSON-schema decoding on gpt-oss and qwen3; other models use tool calling.
        return "json_schema" if model.startswith(("openai/gpt-oss", "qwen/qwen3")) else "function_calling"
    return "function_calling"


def structured_runnable(llm: BaseChatModel, schema: type[T]):
    """`llm.with_structured_output` with the best decoding mode for the provider/model,
    retried on malformed output."""
    model = getattr(llm, "model_name", None) or getattr(llm, "model", "")
    method = _structured_method(model)
    kwargs = {}
    retry_on: tuple[type[BaseException], ...] = (OutputParserException, ValueError)
    if get_settings().llm_provider == "groq":
        import groq

        # Groq validates constrained output server-side and returns 400 `json_validate_failed`
        # when a generation is cut off; that is transient, so retry it like a parse error.
        retry_on += (groq.BadRequestError,)
        if model.startswith("openai/gpt-oss"):
            kwargs["strict"] = True  # constrained decoding: output always matches the schema
    return llm.with_structured_output(schema, method=method, **kwargs).with_retry(
        retry_if_exception_type=retry_on, stop_after_attempt=3
    )


def structured_call(
    schema: type[T],
    system: str,
    user: str,
    *,
    effort: str | None = None,
    llm: BaseChatModel | None = None,
) -> T:
    """One LLM call constrained to a Pydantic schema."""
    llm = llm or get_chat_model(effort)
    runnable = structured_runnable(llm, schema)
    messages = [SystemMessage(system), HumanMessage(user)]
    if get_settings().wait_on_daily_limit:
        return call_waiting_for_quota(lambda: runnable.invoke(messages), label=schema.__name__)
    return runnable.invoke(messages)
