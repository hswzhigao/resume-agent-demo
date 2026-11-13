"""混合召回：向量路（pgvector 余弦）+ 关键词路（trigram/ILIKE + 覆盖率）→ RRF 融合 → 可选重排。"""

from dataclasses import dataclass

from sqlalchemy.engine import Engine

from .clients import RerankClient
from .config import Settings
from .embeddings import get_embedding_client, get_rerank_client
from .pii import mask_pii
from .store import keyword_candidates, vector_candidates
from .tokenizer import tokenize


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """把多路排名融合为一个分数字典，按分数降序（Reciprocal Rank Fusion）。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


@dataclass
class SearchHit:
    doc_title: str
    source: str
    section: str
    content: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


def search(engine: Engine, settings: Settings, query: str, top_k: int = 5) -> list[SearchHit]:
    embedder = get_embedding_client(settings)
    query_vec = embedder.embed(query)

    vec_rows = vector_candidates(engine, query_vec, limit=30)
    tokens = tokenize(query)
    kw_rows = keyword_candidates(engine, query, tokens, limit=30)

    vec_ranked = [r["id"] for r in vec_rows]
    kw_ranked = [r["id"] for r in _rank_by_coverage(kw_rows, tokens)]
    fused = rrf_fuse([vec_ranked, kw_ranked])

    rows_by_id = {r["id"]: r for r in vec_rows}
    for r in kw_rows:
        rows_by_id.setdefault(r["id"], r)

    vec_pos = {rid: i for i, rid in enumerate(vec_ranked)}
    kw_pos = {rid: i for i, rid in enumerate(kw_ranked)}

    ordered_ids = list(fused.keys())[: max(top_k, 1)]
    hits: list[SearchHit] = []
    for rid in ordered_ids:
        row = rows_by_id[rid]
        hits.append(
            SearchHit(
                doc_title=row["title"],
                source=row["source"],
                section=row.get("section") or "",
                content=row["content"],
                score=fused[rid],
                vector_rank=vec_pos.get(rid),
                keyword_rank=kw_pos.get(rid),
            )
        )

    hits = _maybe_rerank(settings, query, hits, top_k)
    if settings.mask_pii:
        for h in hits:
            h.content = mask_pii(h.content)
    return hits[:top_k]


def _rank_by_coverage(rows: list[dict], tokens: list[str]) -> list[dict]:
    """关键词路排序：查询 token 在内容中的覆盖率。"""
    if not tokens:
        return rows
    wanted = set(tokens)

    def coverage(row: dict) -> float:
        content_tokens = set(tokenize(row["content"]))
        return len(content_tokens & wanted) / len(wanted)

    scored = [(coverage(r), r) for r in rows]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for score, r in scored if score > 0]


def _maybe_rerank(
    settings: Settings, query: str, hits: list[SearchHit], top_k: int
) -> list[SearchHit]:
    client: RerankClient | None = get_rerank_client(settings)
    if client is None or len(hits) < 2:
        return hits
    try:
        results = client.rerank(query, [h.content for h in hits], top_n=top_k)
    except RuntimeError:
        return hits  # rerank 不可用时静默降级
    reranked = []
    for r in results:
        hit = hits[r["index"]]
        hit.score = r["relevance_score"]
        reranked.append(hit)
    return reranked
