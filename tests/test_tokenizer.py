from resume_agent_demo.tokenizer import tokenize


def test_ascii_words_lowercased():
    toks = tokenize("Java 高并发 Redis")
    assert "java" in toks
    assert "redis" in toks


def test_cjk_unigram_and_bigram():
    toks = tokenize("高并发")
    assert "高并" in toks
    assert "并发" in toks


def test_punctuation_dropped():
    toks = tokenize("Java、Spring！Redis?")
    assert "java" in toks
    assert "spring" in toks
    assert all("、" not in t and "！" not in t for t in toks)
