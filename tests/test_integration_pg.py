"""集成测试：真实 PostgreSQL + pgvector（模块级自建数据，顺序无关）。"""

import pytest
from sqlalchemy import text

from resume_agent_demo.ingest import ingest_folder
from resume_agent_demo.retriever import search
from resume_agent_demo.store import DimMismatch, init_db


@pytest.fixture(scope="module", autouse=True)
def fresh_db(engine, settings, tmp_path_factory):
    """每个模块先重置并入库最小样例，保证顺序无关。"""
    sample = tmp_path_factory.mktemp("sample")
    (sample / "zhang.md").write_text(
        "# 张三\n\n男，8 年 Java 后端，Spring Boot、MySQL、Redis。\n\n"
        "## 项目\n\n亿级 C 端房产平台，峰值 1000 QPS，Redis 缓存热点房源。\n\n"
        "## 联系\n\n电话 13800001111，邮箱 zhangsan@example.com",
        encoding="utf-8",
    )
    (sample / "li.md").write_text(
        "# 李四\n\n3 年 Python，FastAPI、数据处理，做过 RAG 知识库与爬虫。\n\n"
        "## 项目\n\n企业知识库检索，pgvector 相似召回，LLM 摘要。",
        encoding="utf-8",
    )
    return ingest_folder(engine, settings, sample, reset=True)


def test_ingest_returns_counts(fresh_db):
    assert fresh_db["docs"] == 2
    assert fresh_db["chunks"] >= 4


def test_init_db_records_dim(engine, settings):
    from resume_agent_demo.embeddings import get_embedding_client

    dim = get_embedding_client(settings).dim
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT value FROM rag_meta WHERE key = 'embedding_dim'")
        ).scalar()
    assert value == str(dim)


def test_init_db_dim_mismatch_raises(engine, settings):
    from resume_agent_demo.embeddings import get_embedding_client

    current_dim = get_embedding_client(settings).dim
    with pytest.raises(DimMismatch):
        init_db(engine, dim=current_dim + 8)


def test_search_finds_relevant_resume(engine, settings):
    hits = search(engine, settings, "Java 高并发 后端", top_k=2)
    assert len(hits) == 2
    assert "Java" in hits[0].content


def test_search_keyword_side_ranks_chinese(engine, settings):
    hits = search(engine, settings, "RAG 知识库 pgvector", top_k=2)
    assert any("李四" in h.doc_title or "RAG" in h.content for h in hits)


def test_search_mask_pii_output(engine, mask_settings):
    hits = search(engine, mask_settings, "联系方式 电话", top_k=5)
    joined = "\n".join(h.content for h in hits)
    assert "13800001111" not in joined
    assert "138****11" in joined
