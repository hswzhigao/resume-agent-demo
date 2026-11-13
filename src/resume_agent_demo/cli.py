"""命令行：ingest / search / match。"""

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .graph import run_match
from .ingest import ingest_folder
from .retriever import search
from .store import make_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resume-agent-demo", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="入库一个目录的简历（默认 data/sample）")
    p_ingest.add_argument("folder", nargs="?", default="data/sample")
    p_ingest.add_argument("--reset", action="store_true", help="重建库（切换向量模型后必须）")

    p_search = sub.add_parser("search", help="关键词/语义检索")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)

    p_match = sub.add_parser("match", help="输入 JD，输出匹配排序与解释")
    p_match.add_argument("jd_source", help="JD 文本，或 .md/.txt 文件路径")
    p_match.add_argument("-k", "--top-k", type=int, default=3)

    args = parser.parse_args(argv)
    settings = get_settings()
    engine = make_engine(settings.database_url)

    if args.cmd == "ingest":
        result = ingest_folder(engine, settings, Path(args.folder), reset=args.reset)
        print(f"入库完成：{result['docs']} 份简历，{result['chunks']} 个片段")
        return 0

    if args.cmd == "search":
        hits = search(engine, settings, args.query, top_k=args.top_k)
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h.doc_title} · {h.section or '正文'} (score={h.score:.4f})")
            print(f"    {h.content[:120]}...")
        if not hits:
            print("（无命中）")
        return 0

    if args.cmd == "match":
        source = Path(args.jd_source)
        jd = source.read_text(encoding="utf-8") if source.exists() else args.jd_source
        result = run_match(engine, settings, jd, top_k=args.top_k)
        print(f"[mode={result['mode']} query={result['query']!r}]")
        print(result["explanation"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
