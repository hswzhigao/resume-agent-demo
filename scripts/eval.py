"""检索质量评测：对真实库跑 data/eval/queries.jsonl，输出 hit@k / MRR。

用法：uv run python scripts/eval.py [-k 3]
数据库优先级同测试：DATABASE_URL → pgserver 嵌入式。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume_agent_demo.config import get_settings  # noqa: E402
from resume_agent_demo.evaluation import evaluate  # noqa: E402
from resume_agent_demo.store import make_engine  # noqa: E402


def load_dataset() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "queries.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_engine():
    settings = get_settings()
    if os.environ.get("DATABASE_URL"):
        return make_engine(settings.database_url)
    import pgserver

    data_dir = os.path.expanduser("~/.cache/resume-agent-demo/pgdata")
    server = pgserver.get_server(data_dir)
    uri = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    return make_engine(uri)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=3, help="hit@k 的 k（默认 3）")
    parser.add_argument("--dataset", default=None, help="自定义评测集 jsonl 路径")
    args = parser.parse_args()

    dataset = load_dataset() if not args.dataset else [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    engine = build_engine()
    report = evaluate(engine, get_settings(), dataset, k=args.k)

    print(f"queries={report['queries']}  hit@{report['k']}={report['hit_at_k']:.2%}  "
          f"MRR={report['mrr']:.3f}")
    for d in report["details"]:
        mark = "✓" if d["hit"] else "✗"
        rank = d["rank"] if d["rank"] is not None else "-"
        print(f"  {mark} rank={rank:<3} {d['query']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
