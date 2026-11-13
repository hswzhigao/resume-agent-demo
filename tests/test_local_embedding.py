
from resume_agent_demo.embeddings import LocalHashEmbedding


def test_deterministic_and_normalized():
    e = LocalHashEmbedding(dim=256)
    v1 = e.embed("Java 后端 高并发")
    v2 = e.embed("Java 后端 高并发")
    assert v1 == v2
    assert len(v1) == 256
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6


def test_different_texts_point_different_directions():
    e = LocalHashEmbedding(dim=256)
    a = e.embed("Java 后端 高并发 微服务")
    b = e.embed("护士 会计 审计 报表")
    cos = sum(x * y for x, y in zip(a, b))
    assert cos < 0.5


def test_similar_texts_share_direction():
    e = LocalHashEmbedding(dim=256)
    a = e.embed("Java 高并发")
    b = e.embed("Java 高并发 Redis")
    cos = sum(x * y for x, y in zip(a, b))
    assert cos > 0.5


def test_batch_matches_single():
    e = LocalHashEmbedding(dim=128)
    texts = ["Java 后端", "AI 应用"]
    batch = e.embed_batch(texts)
    assert len(batch) == 2
    assert batch[0] == e.embed("Java 后端")
    assert batch[1] == e.embed("AI 应用")
