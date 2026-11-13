"""PostgreSQL + pgvector 存储：建库、写入、两路召回的 SQL。

向量维度在建表时确定并记录在 rag_meta，切换 embedding 模型需重建（ingest --reset）。
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DimMismatch(RuntimeError):
    """已有库的向量维度与当前配置不一致。"""


def init_db(engine: Engine, dim: int) -> None:
    """创建扩展与表；已存在时校验维度一致。"""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        has_trgm = _try_create_pg_trgm(conn)
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        )
        row = conn.execute(
            text("SELECT value FROM rag_meta WHERE key = 'embedding_dim'")
        ).scalar()
        if row is not None and row != str(dim):
            raise DimMismatch(
                f"数据库向量维度是 {row}，当前配置是 {dim}。请执行 ingest --reset 重建。"
            )
        conn.execute(
            text("INSERT INTO rag_meta (key, value) VALUES ('embedding_dim', :d) "
                 "ON CONFLICT (key) DO NOTHING"),
            {"d": str(dim)},
        )
        conn.execute(text("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, "
                          "source TEXT NOT NULL, title TEXT NOT NULL, "
                          "created_at TIMESTAMPTZ NOT NULL)"))
        conn.execute(
            text(f"CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "
                 f"content TEXT NOT NULL, section TEXT NOT NULL DEFAULT '', idx INT NOT NULL, "
                 f"embedding vector({dim}))")
        )
        if has_trgm:
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON chunks USING GIN "
                     "(content gin_trgm_ops)")
            )


def _try_create_pg_trgm(conn) -> bool:
    """尝试启用 pg_trgm；发行版缺 contrib 时降级为纯 ILIKE 关键词路。"""
    try:
        with conn.begin_nested():
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        return True
    except Exception:  # noqa: BLE001
        return False


def reset_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks"))
        conn.execute(text("DROP TABLE IF EXISTS documents"))
        conn.execute(text("DROP TABLE IF EXISTS rag_meta"))


def insert_document(
    engine: Engine,
    source: str,
    title: str,
    rows: list[tuple[str, str, int, list[float]]],
) -> str:
    """rows: (content, section, idx, embedding)。"""
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO documents (id, source, title, created_at) "
                 "VALUES (:id, :source, :title, :created_at)"),
            {"id": doc_id, "source": source, "title": title, "created_at": now},
        )
        for content, section, idx, vec in rows:
            conn.execute(
                text("INSERT INTO chunks (id, doc_id, content, section, idx, embedding) "
                     "VALUES (:id, :doc_id, :content, :section, :idx, :embedding)"),
                {
                    "id": uuid.uuid4().hex,
                    "doc_id": doc_id,
                    "content": content,
                    "section": section,
                    "idx": idx,
                    "embedding": vec,
                },
            )
    return doc_id


def vector_candidates(engine: Engine, query_vec: list[float], limit: int = 30) -> list[dict]:
    """向量余弦距离最近的前 limit 条。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT c.id, c.content, c.section, d.title, d.source, "
                 "c.embedding <=> (:vec)::vector AS distance "
                 "FROM chunks c JOIN documents d ON d.id = c.doc_id "
                 "ORDER BY c.embedding <=> (:vec)::vector LIMIT :limit"),
            {"vec": query_vec, "limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def keyword_candidates(
    engine: Engine, query: str, tokens: list[str], limit: int = 30
) -> list[dict]:
    """关键词路候选：有 pg_trgm 用相似度，否则纯 ILIKE；覆盖率打分在调用方完成。"""
    if not tokens:
        return []
    with engine.connect() as conn:
        has_trgm = conn.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'pg_trgm'")
        ).scalar() > 0
    if has_trgm:
        clauses = ["similarity(c.content, :q) > 0.02"]
    else:
        clauses = []
    params: dict = {"q": query, "limit": limit}
    kept = tokens[:12]
    for i, tok in enumerate(kept):
        clauses.append(f"c.content ILIKE :tok{i}")
        params[f"tok{i}"] = f"%{tok}%"
    sql = (
        "SELECT c.id, c.content, c.section, d.title, d.source "
        "FROM chunks c JOIN documents d ON d.id = c.doc_id "
        f"WHERE {' OR '.join(clauses)} LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def doc_exists(engine: Engine, source: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM documents WHERE source = :s LIMIT 1"), {"s": source}
        ).scalar()
    return row is not None


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)
