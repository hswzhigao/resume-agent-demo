from resume_agent_demo.retriever import rrf_fuse


def test_rrf_boosts_items_in_multiple_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]])
    assert fused["b"] > fused["a"]
    assert fused["a"] > fused["c"]


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([["x", "y", "z"]])
    keys = list(fused.keys())
    assert keys == ["x", "y", "z"]


def test_rrf_k_changes_relative_weight():
    top = rrf_fuse([["a", "b"], ["b"]], k=1)
    # k 越小，排名靠前权重越大：b 在两个列表都是第一名，应仍领先
    assert top["b"] > top["a"]


def test_rrf_empty_input():
    assert rrf_fuse([]) == {}
