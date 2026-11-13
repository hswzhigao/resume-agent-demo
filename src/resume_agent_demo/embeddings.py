"""向量化：本地确定性 hash 向量（离线/测试）与 OpenAI 兼容接口客户端。"""

import hashlib
import math
from typing import Protocol

from .clients import OpenAIChatClient, OpenAIEmbeddingClient, RerankClient, build_http_client
from .tokenizer import tokenize


class EmbeddingClient(Protocol):
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashEmbedding:
    """确定性特征哈希向量：无依赖、可离线，用于开发与测试（质量有限）。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1 if digest[4] % 2 == 0 else -1
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedding_client(settings) -> EmbeddingClient:
    """按配置选择：OpenAI 兼容接口，或本地 hash 降级。"""

    if settings.has_embedding:
        return OpenAIEmbeddingClient(
            build_http_client(settings.embedding_base_url, settings.embedding_api_key),
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return LocalHashEmbedding(dim=settings.local_embedding_dim)


def get_llm_client(settings) -> OpenAIChatClient | None:
    if settings.has_llm:
        return OpenAIChatClient(
            build_http_client(settings.llm_base_url, settings.llm_api_key),
            model=settings.llm_model,
        )
    return None


def get_rerank_client(settings) -> RerankClient | None:
    if settings.has_rerank:
        return RerankClient(
            build_http_client(settings.rerank_base_url, settings.rerank_api_key),
            model=settings.rerank_model,
            path=settings.rerank_path,
        )
    return None
