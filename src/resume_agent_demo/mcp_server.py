"""MCP stdio server：把检索与匹配暴露为 Agent 可调用的工具。

Claude Code 配置示例（mcpServers 节点）：
  "resume-agent-demo": {
      "command": "uv",
      "args": ["--directory", "/绝对路径/resume-agent-demo",
               "run", "python", "-m", "resume_agent_demo.mcp_server"],
      "env": { "DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5433/resumes" }
  }
"""

from .config import Settings, get_settings
from .graph import run_match
from .pii import mask_pii
from .retriever import SearchHit, search
from .store import make_engine


def format_hits(hits: list[SearchHit], mask: bool) -> str:
    if not hits:
        return "（简历库无命中，请先调用 ingest 或确认关键词）"
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        content = mask_pii(h.content) if mask else h.content
        lines.append(f"[{i}] {h.doc_title} · {h.section or '正文'}\n{content}")
    return "\n\n".join(lines)


def get_services() -> tuple[Settings, object]:
    """进程内惰性单例：设置与数据库引擎。"""
    global _settings, _engine
    try:
        return _settings, _engine  # type: ignore[name-defined]
    except NameError:
        _settings = get_settings()
        _engine = make_engine(_settings.database_url)
        return _settings, _engine


def build_mcp():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("resume-agent-demo")

    @mcp.tool()
    def search_resumes(query: str, top_k: int = 5) -> str:
        """按关键词或 JD 片段检索简历库，返回带引用编号的简历片段。"""
        settings, engine = get_services()
        hits = search(engine, settings, query, top_k=top_k)
        return format_hits(hits, mask=settings.mask_pii)

    @mcp.tool()
    def explain_match(jd: str, top_k: int = 3) -> str:
        """输入完整职位描述，返回最匹配候选人的排序与引用式解释。"""
        settings, engine = get_services()
        result = run_match(engine, settings, jd, top_k=top_k)
        return result["explanation"]

    return mcp


def main() -> None:
    build_mcp().run()


if __name__ == "__main__":
    main()
