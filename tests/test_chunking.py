from resume_agent_demo.chunking import chunk_markdown


def test_splits_by_headers():
    md = "# 简历\n\n张三\n\n## 技能\n\nJava、Spring\n\n## 项目\n\n招聘系统"
    chunks = chunk_markdown(md)
    sections = [c.section for c in chunks]
    assert any("简历" in s for s in sections)
    assert any("技能" in s for s in sections)
    assert any("项目" in s for s in sections)
    joined = "\n".join(c.content for c in chunks)
    assert "张三" in joined
    assert "招聘系统" in joined


def test_content_keeps_section_title_prefix():
    md = "## 技能\n\nJava、Spring"
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert "技能" in chunks[0].content
    assert "Java" in chunks[0].content


def test_long_single_paragraph_hard_split_with_overlap():
    body = "这是一段很长的内容。" * 60  # 600 chars
    md = f"## 长文本\n\n{body}"
    chunks = chunk_markdown(md, max_chars=200, overlap=40)
    assert len(chunks) >= 3
    for c in chunks:
        body = c.content.split("\n\n", 1)[1]
        assert len(body) <= 200 + 40
    # 相邻块首尾重叠（content 带标题前缀，按块体比较）
    first_body = chunks[0].content.split("\n\n", 1)[1]
    second_body = chunks[1].content.split("\n\n", 1)[1]
    assert second_body.startswith(first_body[-40:])


def test_empty_text_returns_empty_list():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_chunks_have_sequential_index():
    md = "## A\n\n内容A\n\n## B\n\n内容B\n\n## C\n\n内容C"
    chunks = chunk_markdown(md)
    assert [c.index for c in chunks] == list(range(len(chunks)))
