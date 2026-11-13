import httpx
import pytest

from resume_agent_demo.clients import OpenAIChatClient, OpenAIEmbeddingClient, RerankClient


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.test/v1",
        headers={"Authorization": "Bearer sk-test"},
    )


def test_embedding_client_posts_and_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        assert request.headers["authorization"] == "Bearer sk-test"
        import json

        body = json.loads(request.content)
        assert body["model"] == "text-embedding-v3"
        assert body["input"] == ["Java 后端", "AI 应用"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [3.0, 4.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            },
        )

    c = OpenAIEmbeddingClient(_client(handler), model="text-embedding-v3", dim=2)
    vecs = c.embed_batch(["Java 后端", "AI 应用"])
    assert vecs == [[1.0, 2.0], [3.0, 4.0]]
    assert c.dim == 2


def test_chat_client_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "你好"}}]},
        )

    c = OpenAIChatClient(_client(handler), model="test-model")
    out = c.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert out == "你好"


def test_rerank_client_parses_sorted():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body["query"] == "Java"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    c = RerankClient(_client(handler), model="rerank-v2")
    results = c.rerank("Java", ["doc0", "doc1"], top_n=2)
    assert results[0]["index"] == 1
    assert results[0]["relevance_score"] > results[1]["relevance_score"]


def test_rerank_client_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no rerank"})

    c = RerankClient(_client(handler), model="rerank-v2")
    with pytest.raises(RuntimeError):
        c.rerank("Java", ["doc0"], top_n=1)
