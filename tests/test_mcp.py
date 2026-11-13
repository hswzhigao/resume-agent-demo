"""MCP server：工具注册与输出格式（不启动 stdio 循环）。"""

import asyncio

from resume_agent_demo.mcp_server import build_mcp, format_hits
from resume_agent_demo.retriever import SearchHit


def test_format_hits_masks_pii():
    hits = [
        SearchHit(
            doc_title="张三", source="z.md", section="联系",
            content="电话 13800001111 邮箱 zhangsan@example.com", score=0.1,
        )
    ]
    out = format_hits(hits, mask=True)
    assert "13800001111" not in out
    assert "138****11" in out
    assert "[1] 张三 · 联系" in out


def test_format_hits_empty_hint():
    assert "无命中" in format_hits([], mask=True)


def test_mcp_server_exposes_two_tools():
    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"search_resumes", "explain_match"} <= names
