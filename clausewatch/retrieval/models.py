"""Swappable embedding model and cross-encoder reranker."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from clausewatch.config import get_settings


def _device() -> str:
    s = get_settings()
    if s.device:
        return s.device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# Models that expect an instruction prefix on queries (asymmetric retrieval).
_QUERY_PROMPTS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        query_kwargs = {"normalize_embeddings": True}
        if prompt := _QUERY_PROMPTS.get(s.embedding_model):
            query_kwargs["prompt"] = prompt
        return HuggingFaceEmbeddings(
            model_name=s.embedding_model,
            model_kwargs={"device": _device()},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            query_encode_kwargs=query_kwargs,
        )
    if s.embedding_provider == "openai":  # optional; requires langchain-openai + OPENAI_API_KEY
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model)
    raise ValueError(f"Unknown embedding_provider: {s.embedding_provider}")


class Reranker:
    """Cross-encoder reranker: scores (query, passage) pairs jointly."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or get_settings().reranker_model, device=_device())

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=16)
        return [float(x) for x in scores]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
