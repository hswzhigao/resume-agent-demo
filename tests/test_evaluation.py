"""评测集与指标测试：hit@k / MRR（retriever 注入假实现，无数据库）。"""

from resume_agent_demo.evaluation import Hit, evaluate

DOCS = {"张三": ["Java 高并发 Redis"], "李四": ["Python RAG FastAPI"], "王五": ["React Node 全栈"]}


def fake_retriever_factory(order_by_query):
    def fake(engine, settings, query, top_k=5):
        hits = [
            Hit(doc_title=t, source=f"{t}.md", section="项目", content="内容", score=1.0 / (i + 1))
            for i, t in enumerate(order_by_query(query))
        ]
        return hits[:top_k]

    return fake


def test_hit_at_k_counts_expected_in_topk():
    orders = {"Java": ["张三", "李四", "王五"], "RAG": ["李四", "张三", "王五"]}
    retriever = fake_retriever_factory(lambda q: orders[q])
    dataset = [
        {"query": "Java", "expected": ["张三"]},
        {"query": "RAG", "expected": ["李四"]},
    ]
    report = evaluate(None, None, dataset, k=2, retriever=retriever)
    assert report["hit_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["queries"] == 2


def test_mrr_reflects_rank():
    retriever = fake_retriever_factory(lambda q: ["李四", "张三", "王五"])
    dataset = [{"query": "Java", "expected": ["张三"]}]  # 张三排第 2 → MRR = 0.5
    report = evaluate(None, None, dataset, k=2, retriever=retriever)
    assert report["mrr"] == 0.5
    assert report["hit_at_k"] == 1.0


def test_miss_beyond_k():
    retriever = fake_retriever_factory(lambda q: ["王五", "李四", "张三"])
    dataset = [{"query": "Java", "expected": ["张三"]}]  # k=1 时张三不在 top1
    report = evaluate(None, None, dataset, k=1, retriever=retriever)
    assert report["hit_at_k"] == 0.0
    assert report["mrr"] == 0.0


def test_expected_match_by_substring():
    """expected 支持子串匹配，便于标题带修饰的场景。"""
    retriever = fake_retriever_factory(lambda q: ["张三 —— Java 后端工程师"])
    dataset = [{"query": "Java", "expected": ["张三"]}]
    report = evaluate(None, None, dataset, k=1, retriever=retriever)
    assert report["hit_at_k"] == 1.0


def test_per_query_details():
    retriever = fake_retriever_factory(lambda q: ["王五", "李四"])
    dataset = [{"query": "Java", "expected": ["张三"]}]
    report = evaluate(None, None, dataset, k=2, retriever=retriever)
    detail = report["details"][0]
    assert detail["hit"] is False
    assert detail["rank"] is None
    assert detail["top_titles"] == ["王五", "李四"]
