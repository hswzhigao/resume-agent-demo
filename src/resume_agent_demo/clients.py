"""OpenAI 兼容外部客户端：embeddings / chat / rerank。

统一用可注入的 httpx.Client，便于测试（MockTransport）与超时控制。
"""

import json
from typing import Any

import httpx


class OpenAIEmbeddingClient:
    def __init__(self, client: httpx.Client, model: str, dim: int) -> None:
        self._client = client
        self.model = model
        self.dim = dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post("/embeddings", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        ordered = sorted(data, key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]


class OpenAIChatClient:
    def __init__(self, client: httpx.Client, model: str, temperature: float = 0.0) -> None:
        self._client = client
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class RerankClient:
    """Jina / Cohere 风格 rerank 接口；失败时由调用方决定降级。"""

    def __init__(self, client: httpx.Client, model: str, path: str = "/rerank") -> None:
        self._client = client
        self.model = model
        self.path = path

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        resp = self._client.post(self.path, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"rerank endpoint unavailable: {resp.status_code}") from e
        results = resp.json()["results"]
        return sorted(results, key=lambda r: r["relevance_score"], reverse=True)


def build_http_client(base_url: str, api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )


def json_or_none(text: str) -> dict[str, Any] | list[Any] | None:
    """LLM 输出宽容解析：剥掉 markdown 代码块后尝试 json.loads。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
