"""检索评测：hit@k 与 MRR（Mean Reciprocal Rank）。

retriever 可注入，评测逻辑本身不依赖数据库，便于离线测试与切换实现对比。
"""

from typing import Any, Callable

from .retriever import SearchHit as Hit  # noqa: F401  (评测集/测试复用同一结构)

DefaultRetriever = Callable[..., list[Hit]]


def _default_retriever(engine, settings, query: str, top_k: int = 5) -> list[Hit]:
    from .retriever import search

    return search(engine, settings, query, top_k=top_k)


def evaluate(
    engine,
    settings,
    dataset: list[dict[str, Any]],
    k: int = 5,
    retriever: DefaultRetriever | None = None,
) -> dict[str, Any]:
    """dataset: [{"query": str, "expected": [标题子串, ...]}]，返回聚合指标与逐条明细。"""
    retriever = retriever or _default_retriever
    hits = 0
    rr_sum = 0.0
    details: list[dict[str, Any]] = []

    for item in dataset:
        query = item["query"]
        expected: list[str] = item["expected"]
        results = retriever(engine, settings, query, top_k=k)
        titles = [h.doc_title for h in results]

        rank: int | None = None
        for i, title in enumerate(titles, 1):
            if any(exp in title for exp in expected):
                rank = i
                break

        hit = rank is not None
        if hit:
            hits += 1
            rr_sum += 1.0 / rank
        details.append({"query": query, "hit": hit, "rank": rank, "top_titles": titles[:k]})

    total = len(dataset) or 1
    return {
        "queries": len(dataset),
        "k": k,
        "hit_at_k": hits / total,
        "mrr": rr_sum / total,
        "details": details,
    }
