"""Central configuration. Every value can be overridden with a CLAUSEWATCH_* env var or .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # provider keys (GROQ_API_KEY, ...) for the SDK clients


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUSEWATCH_", env_file=ROOT / ".env", extra="ignore"
    )

    # --- LLM (swappable: see clausewatch/llm.py) ---
    llm_provider: str = "groq"  # groq | anthropic | openai | ollama
    llm_model: str = "openai/gpt-oss-120b"
    reasoning_effort: str = "medium"
    # Routing is a small classification task: a smaller model (with its own rate-limit bucket).
    router_model: str | None = "openai/gpt-oss-20b"
    # Per-pattern risk checks run the main model at lower reasoning effort.
    router_effort: str = "low"
    # In-graph verifier (fresh context, no access to the draft's reasoning). None = llm_model.
    verifier_model: str | None = None
    # RAGAS judge: a different model family from the generator/verifier, used only for evaluation.
    judge_model: str = "qwen/qwen3.8-27b"
    judge_max_tokens: int = 800
    max_tokens: int = 3000
    requests_per_minute: int = 20  # client-side throttle for free-tier rate limits (0 = off)
    wait_on_daily_limit: bool = False  # batch jobs (eval) block through daily quotas

    # --- Embeddings / reranker (any sentence-transformers compatible model) ---
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str | None = None  # auto: mps > cuda > cpu

    # --- Vector store ---
    qdrant_url: str | None = None  # e.g. http://localhost:6333 ; unset => embedded local mode
    qdrant_path: Path = ROOT / "data" / "qdrant"
    collection: str = "cuad_contracts"

    # --- Data ---
    data_dir: Path = ROOT / "data"
    catalog_path: Path = ROOT / "data" / "catalog.json"
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # --- Retrieval ---
    dense_k: int = 30
    sparse_k: int = 30
    fused_k: int = 30
    rerank_top_n: int = 6
    rrf_k: int = 60

    # --- Agent ---
    max_verify_retries: int = 2
    min_support_ratio: float = 0.8
    risk_top_n: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
