import pytest

from resume_agent_demo.parsing import iter_resume_files, parse_file


def test_parse_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 张三\n\nJava 后端", encoding="utf-8")
    assert "Java" in parse_file(f)


def test_parse_txt(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("Python 数据处理", encoding="utf-8")
    assert "Python" in parse_file(f)


def test_unsupported_suffix_raises(tmp_path):
    f = tmp_path / "c.docx"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="unsupported"):
        parse_file(f)


def test_iter_resume_files_sorted_and_filtered(tmp_path):
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "ignore.docx").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.md").write_text("x", encoding="utf-8")
    files = iter_resume_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["a.md", "b.md", "c.pdf", "d.md"]
