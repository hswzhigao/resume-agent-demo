"""LangGraph 启发式模式单测（无数据库、无 LLM，retriever 注入假实现）。"""

from resume_agent_demo.config import Settings
from resume_agent_demo.graph import run_match
from resume_agent_demo.retriever import SearchHit


def fake_retriever(engine, settings, query, top_k=5):
    return [
        SearchHit(
            doc_title="张三", source="z.md", section="项目",
            content="Java 高并发 房产平台 Redis 缓存", score=0.1,
            vector_rank=0, keyword_rank=0,
        ),
        SearchHit(
            doc_title="李四", source="l.md", section="项目",
            content="Python RAG 知识库 FastAPI", score=0.05,
            vector_rank=1, keyword_rank=None,
        ),
        SearchHit(
            doc_title="张三", source="z.md", section="技能",
            content="Spring Boot MySQL Kafka", score=0.03,
            vector_rank=2, keyword_rank=1,
        ),
    ]


def test_heuristic_mode_end_to_end():
    settings = Settings(mask_pii=False)
    result = run_match(
        None, settings, "招聘 Java 后端 高并发 Redis", top_k=2, retriever=fake_retriever
    )
    assert result["mode"] == "heuristic"
    assert "java" in result["query"].lower()
    titles = [c["title"] for c in result["candidates"]]
    assert len(titles) == 2
    # 张三命中 Java/高并发/Redis 关键词更多，应排第一
    assert titles[0] == "张三"
    assert "92" in result["explanation"] or "分" in result["explanation"]
    assert "[1]" in result["explanation"]  # 引用编号


def test_heuristic_query_drops_markdown_noise():
    settings = Settings(mask_pii=False)
    jd = "# 职位\n\n## 任职要求\n\n1. 扎实的 Java 基础\n\n2. 熟悉 Spring Boot"
    result = run_match(None, settings, jd, top_k=2, retriever=fake_retriever)
    q = result["query"]
    assert "##" not in q
    assert "#" not in q
    assert "java" in q.lower()


def test_heuristic_mode_scores_are_bounded():
    settings = Settings(mask_pii=False)
    result = run_match(None, settings, "JD 文本", top_k=5, retriever=fake_retriever)
    for cand in result["candidates"]:
        assert 0 <= cand["score"] <= 100


def test_no_candidates_explains_ingest_hint():
    settings = Settings(mask_pii=False)

    def empty_retriever(engine, settings, query, top_k=5):
        return []

    result = run_match(None, settings, "任何 JD", top_k=3, retriever=empty_retriever)
    assert "入库" in result["explanation"]


def test_match_emits_trace_event():
    from resume_agent_demo.tracing import RecordingSink

    settings = Settings(mask_pii=False)
    sink = RecordingSink()
    result = run_match(
        None, settings, "招聘 Java 后端", top_k=2,
        retriever=fake_retriever, trace_sink=sink,
    )
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["name"] == "match"
    assert event["metadata"]["mode"] == "heuristic"
    assert event["duration_ms"] >= 0
    assert result["mode"] in event["metadata"]["mode"]
