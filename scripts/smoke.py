"""端到端冒烟：pgserver 存活期间依次执行 CLI 的 ingest / search / match。"""

import os
import sys

import pgserver

DATA_DIR = os.path.expanduser("~/.cache/resume-agent-demo/pgdata")
server = pgserver.get_server(DATA_DIR)
uri = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
os.environ["DATABASE_URL"] = uri
print(f"[smoke] db ready: {uri[:70]}...")

from resume_agent_demo.cli import main  # noqa: E402

print("\n=== 1. ingest ===")
assert main(["ingest", "data/sample", "--reset"]) == 0

print("\n=== 2. search ===")
assert main(["search", "Java 高并发", "-k", "2"]) == 0

print("\n=== 3. match ===")
assert main(["match", "data/jd/sample_jd.md", "-k", "3"]) == 0

print("\n[smoke] ALL OK")
sys.exit(0)
